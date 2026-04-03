package main

import (
	"fmt"
	"os"

	"github.com/sokotsudo/backtest/engine"
	"github.com/sokotsudo/backtest/strategy"
)

func main() {
	csvPath := "testdata/sample.csv"
	if len(os.Args) > 1 {
		csvPath = os.Args[1]
	}

	// Load data
	data, err := engine.LoadCSV(csvPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d bars from %s\n", len(data), csvPath)
	fmt.Printf("Period: %s ~ %s\n\n", data[0].Time.Format("2006-01-02"), data[len(data)-1].Time.Format("2006-01-02"))

	initialCap := 100000.0

	// Print strategy parameters
	ref := strategy.NewMomentumStrategy()
	fmt.Println("=== Strategy Parameters ===")
	fmt.Printf("Momentum Period:  %d\n", ref.MomentumPeriod)
	fmt.Printf("Entry Threshold:  %.2f%%\n", ref.EntryThreshold*100)
	fmt.Printf("Exit Threshold:   %.2f%%\n", ref.ExitThreshold*100)
	fmt.Printf("ATR Period:       %d\n", ref.ATRPeriod)
	fmt.Printf("ATR Multiplier:   %.1f\n", ref.ATRMultiplier)
	fmt.Println()

	// === Full period backtest ===
	fmt.Println("=== Full Period Backtest ===")
	strat := strategy.NewMomentumStrategy()
	eng := engine.NewBacktestEngine(data, strat, initialCap)
	trades, metrics := eng.Run()
	printResults(metrics)
	printTradeLog(trades)

	// === Year-by-year backtest ===
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

	// Print full period summary row
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
	fmt.Println()

	// Print year-by-year trade details
	for year := startYear; year <= endYear; year++ {
		yearData := engine.FilterDataByYear(data, year)
		if len(yearData) < 30 {
			continue
		}
		yearStrat := strategy.NewMomentumStrategy()
		yearEng := engine.NewBacktestEngine(yearData, yearStrat, initialCap)
		yearTrades, _ := yearEng.Run()

		if len(yearTrades) > 0 {
			fmt.Printf("=== %d Trade Log ===\n", year)
			printTradeLog(yearTrades)
			fmt.Println()
		}
	}
}

func printResults(m strategy.PerformanceMetrics) {
	fmt.Printf("Initial Capital:  $%.2f\n", 100000.0)
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
