#!/usr/bin/env python3
"""
BB 3σ逆張り戦略バックテスト（4時間足 BTC/JPY）

ルール:
  エントリー: 終値が BB(20, 3σ) の上限/下限にタッチ → 逆張り
  利確: 終値が BB中央（20期間SMA）にタッチ
  損切り: なし（中央到達まで保持）
"""

import sys
import math
import numpy as np
from backtest import fetch_hourly_data, BacktestResult, Trade
from indicators import bollinger_bands


def aggregate_to_4h(data):
    """1時間足 → 4時間足に集約."""
    closes_1h = data["close"]
    highs_1h = data["high"]
    lows_1h = data["low"]
    volumes_1h = data["volume"]
    timestamps_1h = data["timestamps"]

    n = len(closes_1h)
    n4 = n // 4

    closes = np.zeros(n4)
    highs = np.zeros(n4)
    lows = np.zeros(n4)
    volumes = np.zeros(n4)
    timestamps = np.zeros(n4, dtype=np.int64)

    for i in range(n4):
        s = i * 4
        e = s + 4
        closes[i] = closes_1h[e - 1]           # 最後の終値
        highs[i] = np.max(highs_1h[s:e])       # 最高値
        lows[i] = np.min(lows_1h[s:e])          # 最安値
        volumes[i] = np.sum(volumes_1h[s:e])    # 出来高合計
        timestamps[i] = timestamps_1h[e - 1]    # 最後のタイムスタンプ

    return {
        "close": closes, "high": highs, "low": lows,
        "volume": volumes, "timestamps": timestamps,
    }


def backtest_bb_reversal(data_4h, initial_capital=1_000_000,
                         bb_period=20, bb_sigma=3.0,
                         capital_ratio=0.90, label=None):
    """BB 3σ逆張り → SMA20利確."""

    name = label or f"BB{bb_sigma}σ Reversal (4h, period={bb_period})"
    result = BacktestResult(name=name, initial_capital=initial_capital)
    closes = data_4h["close"]
    n = len(closes)

    # BB計算
    bb_upper, bb_middle, bb_lower = bollinger_bands(closes, period=bb_period, num_std=bb_sigma)

    equity = initial_capital
    position = None  # {side, size, entry, bar}
    hold_bars_list = []

    for i in range(bb_period, n):
        result.equity_curve.append(equity)
        price = closes[i]

        if position is None:
            # --- エントリー ---
            if not np.isnan(bb_lower[i]) and price <= bb_lower[i]:
                # ロング: 下限バンドタッチ
                size = (equity * capital_ratio) / price
                position = {"side": "LONG", "size": size, "entry": price, "bar": i}

            elif not np.isnan(bb_upper[i]) and price >= bb_upper[i]:
                # ショート: 上限バンドタッチ
                size = (equity * capital_ratio) / price
                position = {"side": "SHORT", "size": size, "entry": price, "bar": i}

        else:
            # --- 利確: BB中央（SMA20）タッチ ---
            if np.isnan(bb_middle[i]):
                continue

            should_close = False
            if position["side"] == "LONG" and price >= bb_middle[i]:
                should_close = True
            elif position["side"] == "SHORT" and price <= bb_middle[i]:
                should_close = True

            if should_close:
                if position["side"] == "LONG":
                    pnl_pct = (price - position["entry"]) / position["entry"]
                else:
                    pnl_pct = (position["entry"] - price) / position["entry"]

                pnl_jpy = pnl_pct * position["entry"] * position["size"]
                equity += pnl_jpy
                hold_bars = i - position["bar"]
                hold_bars_list.append(hold_bars)

                result.trades.append(Trade(
                    position["side"], position["entry"], price,
                    position["bar"], i, position["size"],
                    pnl_pct, pnl_jpy, "BB_MID"))
                position = None

    # 残ポジション決済
    if position:
        price = closes[-1]
        if position["side"] == "LONG":
            pnl_pct = (price - position["entry"]) / position["entry"]
        else:
            pnl_pct = (position["entry"] - price) / position["entry"]
        pnl_jpy = pnl_pct * position["entry"] * position["size"]
        equity += pnl_jpy
        hold_bars_list.append(n - 1 - position["bar"])
        result.trades.append(Trade(
            position["side"], position["entry"], price,
            position["bar"], n - 1, position["size"],
            pnl_pct, pnl_jpy, "END"))
    result.equity_curve.append(equity)

    return result, hold_bars_list


def main():
    print("=" * 80)
    print("  BB 3σ逆張り戦略バックテスト（4時間足 BTC/JPY）")
    print("=" * 80)

    data = fetch_hourly_data(days=365 * 3)
    n_1h = len(data["close"])
    print(f"\n1時間足: {n_1h} candles ({n_1h/24:.0f} days)")

    data_4h = aggregate_to_4h(data)
    n_4h = len(data_4h["close"])
    print(f"4時間足: {n_4h} candles ({n_4h/6:.0f} days)")
    print(f"BTC/JPY range: {data_4h['close'].min():,.0f} — {data_4h['close'].max():,.0f}")

    # === メインバックテスト ===
    print("\nRunning BB 3.0σ reversal (period=20)...")
    r1, holds1 = backtest_bb_reversal(data_4h, bb_period=20, bb_sigma=3.0,
                                       label="BB 3.0σ Reversal (4h P20)")

    # === σ比較 ===
    print("Running BB 2.5σ reversal...")
    r2, holds2 = backtest_bb_reversal(data_4h, bb_period=20, bb_sigma=2.5,
                                       label="BB 2.5σ Reversal (4h P20)")

    print("Running BB 2.0σ reversal...")
    r3, holds3 = backtest_bb_reversal(data_4h, bb_period=20, bb_sigma=2.0,
                                       label="BB 2.0σ Reversal (4h P20)")

    all_results = [(r1, holds1), (r2, holds2), (r3, holds3)]

    # === 結果表示 ===
    print(f"\n{'='*100}")
    print(f"  結果比較")
    print(f"{'='*100}")
    hdr = (f"{'Strategy':<32} {'Annual%':>8} {'Return%':>9} {'Trades':>7} "
           f"{'WinR%':>7} {'AvgWin%':>8} {'AvgLos%':>8} {'MaxDD%':>8} {'Sharpe':>7}")
    print(hdr)
    print("-" * 100)

    for r, holds in all_results:
        years = len(r.equity_curve) / (6 * 365) if r.equity_curve else 1  # 6 bars/day for 4h
        ann = ((r.equity_curve[-1] / r.initial_capital) ** (1 / years) - 1) * 100 if years > 0 and r.equity_curve[-1] > 0 else -100
        print(f"{r.name:<32} {ann:>7.1f}% {r.total_return:>8.1f}% {r.num_trades:>7d} "
              f"{r.win_rate:>6.1f}% {r.avg_win:>7.2f}% {r.avg_loss:>7.2f}% "
              f"{r.max_drawdown:>7.1f}% {r.sharpe_ratio:>7.2f}")

    print(f"{'='*100}")

    # === 3σ詳細分析 ===
    r = r1
    holds = holds1
    if r.trades:
        print(f"\n{'='*80}")
        print(f"  BB 3.0σ 詳細分析")
        print(f"{'='*80}")

        longs = [t for t in r.trades if t.side == "LONG"]
        shorts = [t for t in r.trades if t.side == "SHORT"]

        print(f"\n  LONG: {len(longs)}回  "
              f"勝率={sum(1 for t in longs if t.pnl_pct > 0)/len(longs)*100:.1f}%" if longs else "")
        print(f"  SHORT: {len(shorts)}回  "
              f"勝率={sum(1 for t in shorts if t.pnl_pct > 0)/len(shorts)*100:.1f}%" if shorts else "")

        if holds:
            holds_hours = [h * 4 for h in holds]  # 4h bars → hours
            print(f"\n  保有時間:")
            print(f"    平均: {np.mean(holds_hours):.0f}h ({np.mean(holds_hours)/24:.1f}日)")
            print(f"    中央値: {np.median(holds_hours):.0f}h ({np.median(holds_hours)/24:.1f}日)")
            print(f"    最短: {np.min(holds_hours):.0f}h")
            print(f"    最長: {np.max(holds_hours):.0f}h ({np.max(holds_hours)/24:.0f}日)")

        wins = [t for t in r.trades if t.pnl_pct > 0]
        losses = [t for t in r.trades if t.pnl_pct <= 0]
        if wins:
            print(f"\n  勝ちトレード: 平均 +{np.mean([t.pnl_pct for t in wins])*100:.2f}%  "
                  f"最大 +{max(t.pnl_pct for t in wins)*100:.2f}%")
        if losses:
            print(f"  負けトレード: 平均 {np.mean([t.pnl_pct for t in losses])*100:.2f}%  "
                  f"最大 {min(t.pnl_pct for t in losses)*100:.2f}%")

        # Equity progression
        print(f"\n  資金推移 (初期100万円):")
        eq = np.array(r.equity_curve)
        total_bars = len(eq)
        for pct in [25, 50, 75, 100]:
            idx = min(int(total_bars * pct / 100), total_bars - 1)
            months = idx / (6 * 30)
            print(f"    {months:>5.0f}ヶ月後: {eq[idx]:>12,.0f} JPY")

        print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
