package engine

import (
	"math"
	"testing"
	"time"

	"github.com/sokotsudo/backtest/strategy"
)

func TestComputeCAGR(t *testing.T) {
	// $100K -> $200K in 252 bars (1 year) = 100% CAGR
	cagr := computeCAGR(100000, 200000, 252)
	if math.Abs(cagr-1.0) > 0.01 {
		t.Errorf("CAGR = %f, want ~1.0", cagr)
	}

	// $100K -> $100K = 0% CAGR
	cagr = computeCAGR(100000, 100000, 252)
	if math.Abs(cagr) > 0.001 {
		t.Errorf("CAGR = %f, want ~0.0", cagr)
	}

	// Edge cases
	if computeCAGR(0, 100000, 252) != 0 {
		t.Error("expected 0 for zero initial cap")
	}
	if computeCAGR(100000, 0, 252) != 0 {
		t.Error("expected 0 for zero final equity")
	}
}

func TestObjectiveScore(t *testing.T) {
	// Positive CAGR with moderate drawdown, above min trades
	score := objectiveScore(0.20, 10.0, 100, 50)
	if score <= 0 {
		t.Errorf("expected positive score, got %f", score)
	}

	// Higher CAGR same DD -> higher score
	score2 := objectiveScore(0.40, 10.0, 100, 50)
	if score2 <= score {
		t.Errorf("higher CAGR should give higher score: %f vs %f", score2, score)
	}

	// Below min trades -> very negative
	score3 := objectiveScore(0.50, 5.0, 10, 50)
	if score3 >= 0 {
		t.Errorf("below min trades should be very negative, got %f", score3)
	}
}

func makeTestData() []strategy.OHLCV {
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	var data []strategy.OHLCV
	price := 100.0
	for i := 0; i < 75; i++ {
		var change float64
		if i < 30 {
			change = 0.1
		} else if i < 55 {
			change = price * 0.005
		} else {
			change = price * -0.01
		}
		price += change
		data = append(data, strategy.OHLCV{
			Time: base.AddDate(0, 0, i), Open: price - 0.3,
			High: price + 0.5, Low: price - 0.5, Close: price,
		})
	}
	return data
}

func TestRunGridSearch_SmallGrid(t *testing.T) {
	data := makeTestData()
	cfg := GridSearchConfig{
		MomentumPeriods: []int{3, 5},
		EntryThresholds: []float64{0.01, 0.02},
		ExitThresholds:  []float64{-0.01},
		ATRPeriods:      []int{5},
		ATRMultipliers:  []float64{1.0},
		InitialCapital:  100000.0,
		MinTrades:       0, // no filter for small test
		TopN:            3,
	}

	results := RunGridSearch(data, cfg)

	// 2*2*1*1*1 = 4 combos, TopN=3
	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
	}

	// Should be sorted by score descending
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: [%d].Score=%f > [%d].Score=%f", i, results[i].Score, i-1, results[i-1].Score)
		}
	}

	// Each result should have valid equity
	for i, r := range results {
		if r.Metrics.FinalEquity <= 0 {
			t.Errorf("result[%d] has non-positive final equity: %f", i, r.Metrics.FinalEquity)
		}
	}
}
