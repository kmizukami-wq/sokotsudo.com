package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/sokotsudo/backtest/engine"
	"github.com/sokotsudo/backtest/strategy"
)

func main() {
	csvPath := flag.String("data", "testdata/sample.csv", "path to CSV data file")
	optimize := flag.Bool("optimize", false, "run parameter grid search optimization")
	topN := flag.Int("top", 20, "number of top parameter sets to display")
	flag.Parse()
	if flag.NArg() > 0 {
		*csvPath = flag.Arg(0)
	}

	data, err := engine.LoadCSV(*csvPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d bars from %s\n", len(data), *csvPath)
	fmt.Printf("Period: %s ~ %s\n\n", data[0].Time.Format("2006-01-02"), data[len(data)-1].Time.Format("2006-01-02"))

	initialCap := 100000.0

	if *optimize {
		runOptimization(data, initialCap, *topN)
		return
	}

	runDefaultBacktest(data, initialCap)
}

func runOptimization(data []strategy.OHLCV, initialCap float64, topN int) {
	// === Phase 1: Broad Grid Search ===
	fmt.Println("=== Phase 1: Broad Grid Search ===")
	fmt.Println()

	cfg := engine.DefaultGridSearchConfig()
	cfg.TopN = topN
	cfg.InitialCapital = initialCap

	results := engine.RunGridSearch(data, cfg)
	engine.PrintOptResults(results)
	fmt.Println()

	if len(results) == 0 {
		fmt.Println("No valid results found.")
		return
	}

	// === Phase 2: Fine Grid Search around top 3 ===
	fmt.Println("=== Phase 2: Fine-Tuning Around Best Parameters ===")
	fmt.Println()

	fineCfg := engine.BuildFineGrid(results[:min(3, len(results))])
	fineCfg.InitialCapital = initialCap
	fineCfg.MinTrades = cfg.MinTrades
	fineCfg.TopN = topN

	fineResults := engine.RunGridSearch(data, fineCfg)
	engine.PrintOptResults(fineResults)
	fmt.Println()

	// Use the best from fine-tuning if it's better
	if len(fineResults) > 0 && fineResults[0].Score > results[0].Score {
		results = fineResults
	}

	// Run best parameters with full detail
	best := results[0].Params
	fmt.Println("=== Best Parameters - Full Backtest ===")
	fmt.Printf("Momentum Period:  %d\n", best.MomentumPeriod)
	fmt.Printf("Entry Threshold:  %.3f%%\n", best.EntryThreshold*100)
	fmt.Printf("Exit Threshold:   %.3f%%\n", best.ExitThreshold*100)
	fmt.Printf("ATR Period:       %d\n", best.ATRPeriod)
	fmt.Printf("ATR Multiplier:   %.1f\n", best.ATRMultiplier)
	fmt.Println()

	bestStrat := strategy.NewMomentumStrategy()
	bestStrat.MomentumPeriod = best.MomentumPeriod
	bestStrat.EntryThreshold = best.EntryThreshold
	bestStrat.ExitThreshold = best.ExitThreshold
	bestStrat.ATRPeriod = best.ATRPeriod
	bestStrat.ATRMultiplier = best.ATRMultiplier

	bestEng := engine.NewBacktestEngine(data, bestStrat, initialCap)
	bestTrades, bestMetrics := bestEng.Run()
	printResults(bestMetrics, initialCap)
	fmt.Println()
	printTradeLog(bestTrades)
	fmt.Println()

	// Year-by-year with best parameters
	startYear := data[0].Time.Year()
	endYear := data[len(data)-1].Time.Year()

	fmt.Println("=== Year-by-Year Summary (Best Parameters) ===")
	fmt.Printf("%-6s %7s %8s %9s %9s %9s %12s\n", "Year", "Trades", "WinRate", "Return", "MaxDD", "Sharpe", "FinalEquity")
	fmt.Println("--------------------------------------------------------------------------")

	for year := startYear; year <= endYear; year++ {
		yearData := engine.FilterDataByYear(data, year)
		if len(yearData) < 30 {
			continue
		}
		yearStrat := strategy.NewMomentumStrategy()
		yearStrat.MomentumPeriod = best.MomentumPeriod
		yearStrat.EntryThreshold = best.EntryThreshold
		yearStrat.ExitThreshold = best.ExitThreshold
		yearStrat.ATRPeriod = best.ATRPeriod
		yearStrat.ATRMultiplier = best.ATRMultiplier

		yearEng := engine.NewBacktestEngine(yearData, yearStrat, initialCap)
		_, yearMetrics := yearEng.Run()

		fmt.Printf("%-6d %7d %7.2f%% %+8.2f%% %8.2f%% %8.4f  $%10.2f\n",
			year,
			yearMetrics.TotalTrades,
			yearMetrics.WinRate,
			yearMetrics.TotalReturn,
			yearMetrics.MaxDrawdown,
			yearMetrics.SharpeRatio,
			yearMetrics.FinalEquity,
		)
	}

	fmt.Println("--------------------------------------------------------------------------")
	fmt.Printf("%-6s %7d %7.2f%% %+8.2f%% %8.2f%% %8.4f  $%10.2f\n",
		"ALL",
		bestMetrics.TotalTrades,
		bestMetrics.WinRate,
		bestMetrics.TotalReturn,
		bestMetrics.MaxDrawdown,
		bestMetrics.SharpeRatio,
		bestMetrics.FinalEquity,
	)
}

func runDefaultBacktest(data []strategy.OHLCV, initialCap float64) {
	ref := strategy.NewMomentumStrategy()
	fmt.Println("=== Strategy Parameters ===")
	fmt.Printf("Momentum Period:  %d\n", ref.MomentumPeriod)
	fmt.Printf("Entry Threshold:  %.2f%%\n", ref.EntryThreshold*100)
	fmt.Printf("Exit Threshold:   %.2f%%\n", ref.ExitThreshold*100)
	fmt.Printf("ATR Period:       %d\n", ref.ATRPeriod)
	fmt.Printf("ATR Multiplier:   %.1f\n", ref.ATRMultiplier)
	fmt.Println()

	// Full period
	fmt.Println("=== Full Period Backtest ===")
	strat := strategy.NewMomentumStrategy()
	eng := engine.NewBacktestEngine(data, strat, initialCap)
	trades, metrics := eng.Run()
	printResults(metrics, initialCap)
	printTradeLog(trades)

	// Year-by-year
	startYear := data[0].Time.Year()
	endYear := data[len(data)-1].Time.Year()

	fmt.Println()
	fmt.Println("=== Year-by-Year Summary ===")
	fmt.Printf("%-6s %7s %8s %9s %9s %9s %12s\n", "Year", "Trades", "WinRate", "Return", "MaxDD", "Sharpe", "FinalEquity")
	fmt.Println("--------------------------------------------------------------------------")

	for year := startYear; year <= endYear; year++ {
		yearData := engine.FilterDataByYear(data, year)
		if len(yearData) < 30 {
			continue
		}
		yearStrat := strategy.NewMomentumStrategy()
		yearEng := engine.NewBacktestEngine(yearData, yearStrat, initialCap)
		_, yearMetrics := yearEng.Run()

		fmt.Printf("%-6d %7d %7.2f%% %+8.2f%% %8.2f%% %8.4f  $%10.2f\n",
			year,
			yearMetrics.TotalTrades,
			yearMetrics.WinRate,
			yearMetrics.TotalReturn,
			yearMetrics.MaxDrawdown,
			yearMetrics.SharpeRatio,
			yearMetrics.FinalEquity,
		)
	}
	fmt.Println("--------------------------------------------------------------------------")
	fmt.Printf("%-6s %7d %7.2f%% %+8.2f%% %8.2f%% %8.4f  $%10.2f\n",
		"ALL",
		metrics.TotalTrades,
		metrics.WinRate,
		metrics.TotalReturn,
		metrics.MaxDrawdown,
		metrics.SharpeRatio,
		metrics.FinalEquity,
	)
}

func printResults(m strategy.PerformanceMetrics, initialCap float64) {
	fmt.Printf("Initial Capital:  $%.2f\n", initialCap)
	fmt.Printf("Final Equity:     $%.2f\n", m.FinalEquity)
	fmt.Printf("Total Trades:     %d\n", m.TotalTrades)
	fmt.Printf("Winning Trades:   %d\n", m.WinningTrades)
	fmt.Printf("Losing Trades:    %d\n", m.LosingTrades)
	fmt.Printf("Win Rate:         %.2f%%\n", m.WinRate)
	fmt.Printf("Total Return:     %.2f%%\n", m.TotalReturn)
	fmt.Printf("Max Drawdown:     %.2f%%\n", m.MaxDrawdown)
	fmt.Printf("Sharpe Ratio:     %.4f\n", m.SharpeRatio)
}

func printTradeLog(trades []strategy.Trade) {
	if len(trades) == 0 {
		fmt.Println("No trades generated.")
		return
	}
	fmt.Printf("%-12s %-12s %10s %10s %12s %8s\n", "Entry Date", "Exit Date", "Entry", "Exit", "PnL", "PnL%")
	fmt.Println("------------------------------------------------------------------------")
	for _, t := range trades {
		fmt.Printf("%-12s %-12s %10.2f %10.2f %12.2f %+7.2f%%\n",
			t.EntryTime.Format("2006-01-02"),
			t.ExitTime.Format("2006-01-02"),
			t.EntryPrice,
			t.ExitPrice,
			t.PnL,
			t.PnLPercent,
		)
	}
}
