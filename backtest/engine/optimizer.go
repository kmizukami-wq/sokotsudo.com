package engine

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/sokotsudo/backtest/strategy"
)

// ParamSet holds one combination of strategy parameters.
type ParamSet struct {
	MomentumPeriod int
	EntryThreshold float64
	ExitThreshold  float64
	ATRPeriod      int
	ATRMultiplier  float64
}

// OptResult pairs a parameter set with its backtest metrics.
type OptResult struct {
	Params  ParamSet
	Metrics strategy.PerformanceMetrics
	CAGR    float64
	Score   float64
}

// GridSearchConfig defines the search space.
type GridSearchConfig struct {
	MomentumPeriods []int
	EntryThresholds []float64
	ExitThresholds  []float64
	ATRPeriods      []int
	ATRMultipliers  []float64
	InitialCapital  float64
	MinTrades       int // minimum trades required (0 = no filter)
	TopN            int
}

// DefaultGridSearchConfig returns a broad parameter grid optimized for trade frequency.
func DefaultGridSearchConfig() GridSearchConfig {
	return GridSearchConfig{
		MomentumPeriods: []int{3, 5, 7, 10, 14, 20},
		EntryThresholds: []float64{0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05},
		ExitThresholds:  []float64{-0.003, -0.005, -0.008, -0.01, -0.015, -0.02, -0.03},
		ATRPeriods:      []int{3, 5, 7, 10, 14},
		ATRMultipliers:  []float64{0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0},
		InitialCapital:  100000.0,
		MinTrades:       50,
		TopN:            20,
	}
}

// RunGridSearch tests all parameter combinations and returns the top results.
func RunGridSearch(data []strategy.OHLCV, cfg GridSearchConfig) []OptResult {
	var combos []ParamSet
	for _, mp := range cfg.MomentumPeriods {
		for _, et := range cfg.EntryThresholds {
			for _, ex := range cfg.ExitThresholds {
				for _, ap := range cfg.ATRPeriods {
					for _, am := range cfg.ATRMultipliers {
						combos = append(combos, ParamSet{
							MomentumPeriod: mp,
							EntryThreshold: et,
							ExitThreshold:  ex,
							ATRPeriod:      ap,
							ATRMultiplier:  am,
						})
					}
				}
			}
		}
	}

	numBars := len(data)
	results := make([]OptResult, len(combos))

	ch := make(chan int, len(combos))
	for i := range combos {
		ch <- i
	}
	close(ch)

	var wg sync.WaitGroup
	workers := runtime.NumCPU()
	if workers < 1 {
		workers = 1
	}

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range ch {
				p := combos[idx]
				strat := strategy.NewMomentumStrategy()
				strat.MomentumPeriod = p.MomentumPeriod
				strat.EntryThreshold = p.EntryThreshold
				strat.ExitThreshold = p.ExitThreshold
				strat.ATRPeriod = p.ATRPeriod
				strat.ATRMultiplier = p.ATRMultiplier

				eng := NewBacktestEngine(data, strat, cfg.InitialCapital)
				_, metrics := eng.Run()

				cagr := computeCAGR(cfg.InitialCapital, metrics.FinalEquity, numBars)
				score := objectiveScore(cagr, metrics.MaxDrawdown, metrics.TotalTrades, cfg.MinTrades)

				results[idx] = OptResult{
					Params:  p,
					Metrics: metrics,
					CAGR:    cagr,
					Score:   score,
				}
			}
		}()
	}
	wg.Wait()

	// Filter by minimum trades
	if cfg.MinTrades > 0 {
		var filtered []OptResult
		for _, r := range results {
			if r.Metrics.TotalTrades >= cfg.MinTrades {
				filtered = append(filtered, r)
			}
		}
		results = filtered
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if cfg.TopN > 0 && cfg.TopN < len(results) {
		return results[:cfg.TopN]
	}
	return results
}

func computeCAGR(initialCap, finalEquity float64, bars int) float64 {
	if initialCap <= 0 || finalEquity <= 0 || bars <= 0 {
		return 0
	}
	years := float64(bars) / 252.0
	if years <= 0 {
		return 0
	}
	return math.Pow(finalEquity/initialCap, 1.0/years) - 1.0
}

// objectiveScore maximizes compound return (CAGR) with a drawdown penalty.
// Prioritizes raw CAGR but penalizes extreme drawdown.
func objectiveScore(cagr, maxDD float64, trades, minTrades int) float64 {
	if trades < minTrades {
		return -999 // filtered out
	}
	if cagr <= 0 {
		return cagr
	}
	// Primary: CAGR. Penalty if MaxDD > 30%.
	ddPenalty := 1.0
	if maxDD > 30 {
		ddPenalty = 30.0 / maxDD
	}
	return cagr * ddPenalty
}

// BuildFineGrid creates a fine-grained grid around the best results from a broad search.
func BuildFineGrid(topResults []OptResult) GridSearchConfig {
	// Collect unique values and expand around them
	momSet := make(map[int]bool)
	etSet := make(map[float64]bool)
	exSet := make(map[float64]bool)
	apSet := make(map[int]bool)
	amSet := make(map[float64]bool)

	for _, r := range topResults {
		p := r.Params
		// Momentum period: +/- 2 steps
		for d := -2; d <= 2; d++ {
			v := p.MomentumPeriod + d
			if v >= 2 {
				momSet[v] = true
			}
		}
		// Entry threshold: fine steps around best
		for _, mult := range []float64{0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5} {
			v := p.EntryThreshold * mult
			if v > 0.001 {
				etSet[math.Round(v*10000)/10000] = true
			}
		}
		// Exit threshold
		for _, mult := range []float64{0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5} {
			v := p.ExitThreshold * mult
			if v < -0.001 {
				exSet[math.Round(v*10000)/10000] = true
			}
		}
		// ATR period: +/- 2
		for d := -2; d <= 2; d++ {
			v := p.ATRPeriod + d
			if v >= 2 {
				apSet[v] = true
			}
		}
		// ATR multiplier: fine steps
		for _, mult := range []float64{0.7, 0.85, 1.0, 1.15, 1.3} {
			v := math.Round(p.ATRMultiplier*mult*10) / 10
			if v >= 0.2 {
				amSet[v] = true
			}
		}
	}

	return GridSearchConfig{
		MomentumPeriods: sortedInts(momSet),
		EntryThresholds: sortedFloats(etSet),
		ExitThresholds:  sortedFloats(exSet),
		ATRPeriods:      sortedInts(apSet),
		ATRMultipliers:  sortedFloats(amSet),
	}
}

func sortedInts(m map[int]bool) []int {
	s := make([]int, 0, len(m))
	for k := range m {
		s = append(s, k)
	}
	sort.Ints(s)
	return s
}

func sortedFloats(m map[float64]bool) []float64 {
	s := make([]float64, 0, len(m))
	for k := range m {
		s = append(s, k)
	}
	sort.Float64s(s)
	return s
}

// PrintOptResults displays the optimization results table.
func PrintOptResults(results []OptResult) {
	fmt.Printf("%-5s %4s %7s %7s %4s %5s | %7s %6s %10s %8s %7s %8s\n",
		"Rank", "Mom", "Entry%", "Exit%", "ATR", "ATRm",
		"Trades", "Win%", "Return%", "CAGR%", "MaxDD%", "Score")
	fmt.Println(strings.Repeat("-", 102))
	for i, r := range results {
		fmt.Printf("%-5d %4d %7.3f %7.3f %4d %5.1f | %7d %5.1f%% %+10.2f%% %+7.2f%% %6.2f%% %8.4f\n",
			i+1,
			r.Params.MomentumPeriod,
			r.Params.EntryThreshold*100,
			r.Params.ExitThreshold*100,
			r.Params.ATRPeriod,
			r.Params.ATRMultiplier,
			r.Metrics.TotalTrades,
			r.Metrics.WinRate,
			r.Metrics.TotalReturn,
			r.CAGR*100,
			r.Metrics.MaxDrawdown,
			r.Score,
		)
	}
}
