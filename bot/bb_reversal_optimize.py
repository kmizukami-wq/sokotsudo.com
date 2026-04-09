#!/usr/bin/env python3
"""
BB 3σ逆張り戦略 — 損切り最適化スイープ

トレンド継続の定義:
  1. 固定%損切り: エントリーから X% 逆行
  2. ATR損切り: ATR × 倍率
  3. タイムストップ: X本(4h)以内にBB中央未達なら決済
  4. ADXフィルター: ADX > 閾値なら「トレンド中」として損切り/エントリー拒否
  5. 連続バンド外: N本連続でバンド外にいたら損切り
"""

import sys
import math
import itertools
import numpy as np
from backtest import fetch_hourly_data, BacktestResult, Trade
from indicators import bollinger_bands, atr, adx, rsi


def aggregate_to_4h(data):
    closes_1h, highs_1h, lows_1h = data["close"], data["high"], data["low"]
    volumes_1h, timestamps_1h = data["volume"], data["timestamps"]
    n = len(closes_1h)
    n4 = n // 4
    closes = np.zeros(n4)
    highs = np.zeros(n4)
    lows = np.zeros(n4)
    volumes = np.zeros(n4)
    timestamps = np.zeros(n4, dtype=np.int64)
    for i in range(n4):
        s = i * 4
        closes[i] = closes_1h[s + 3]
        highs[i] = np.max(highs_1h[s:s+4])
        lows[i] = np.min(lows_1h[s:s+4])
        volumes[i] = np.sum(volumes_1h[s:s+4])
        timestamps[i] = timestamps_1h[s + 3]
    return {"close": closes, "high": highs, "low": lows,
            "volume": volumes, "timestamps": timestamps}


def backtest_bb_opt(closes, highs_arr, lows_arr, n,
                    bb_upper, bb_middle, bb_lower,
                    atr_arr, adx_arr, rsi_arr,
                    initial_capital=1_000_000,
                    capital_ratio=0.90,
                    # Stop-loss params
                    sl_fixed_pct=None,       # e.g. 0.03 = 3%
                    sl_atr_mult=None,        # e.g. 2.0
                    time_stop_bars=None,     # e.g. 20 (= 80h)
                    adx_entry_max=None,      # e.g. 25: don't enter if ADX > 25
                    adx_exit_th=None,        # e.g. 30: exit if ADX > 30
                    consec_outside=None,     # e.g. 3: exit if 3 bars outside band
                    bb_period=20):

    equity = initial_capital
    position = None
    trades = []
    eq_curve = []

    for i in range(bb_period, n):
        eq_curve.append(equity)
        price = closes[i]

        if position is None:
            # --- Entry ---
            if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
                continue

            # ADX entry filter
            if adx_entry_max is not None and not np.isnan(adx_arr[i]):
                if adx_arr[i] > adx_entry_max:
                    continue

            direction = None
            if price <= bb_lower[i]:
                direction = "LONG"
            elif price >= bb_upper[i]:
                direction = "SHORT"

            if direction:
                size = (equity * capital_ratio) / price
                position = {"side": direction, "size": size, "entry": price,
                            "bar": i, "consec": 0}
        else:
            if np.isnan(bb_middle[i]):
                continue

            should_close = False
            reason = ""

            # 1. Take profit: BB middle touch
            if position["side"] == "LONG" and price >= bb_middle[i]:
                should_close = True
                reason = "TP"
            elif position["side"] == "SHORT" and price <= bb_middle[i]:
                should_close = True
                reason = "TP"

            # 2. Fixed % stop-loss
            if not should_close and sl_fixed_pct is not None:
                if position["side"] == "LONG":
                    loss = (position["entry"] - price) / position["entry"]
                else:
                    loss = (price - position["entry"]) / position["entry"]
                if loss >= sl_fixed_pct:
                    should_close = True
                    reason = "SL_FIX"

            # 3. ATR stop-loss
            if not should_close and sl_atr_mult is not None and not np.isnan(atr_arr[i]):
                sl_dist = atr_arr[i] * sl_atr_mult / price
                if position["side"] == "LONG":
                    loss = (position["entry"] - price) / position["entry"]
                else:
                    loss = (price - position["entry"]) / position["entry"]
                if loss >= sl_dist:
                    should_close = True
                    reason = "SL_ATR"

            # 4. Time stop
            if not should_close and time_stop_bars is not None:
                if i - position["bar"] >= time_stop_bars:
                    should_close = True
                    reason = "TIME"

            # 5. ADX exit (trend detected)
            if not should_close and adx_exit_th is not None and not np.isnan(adx_arr[i]):
                if adx_arr[i] > adx_exit_th:
                    should_close = True
                    reason = "ADX_EXIT"

            # 6. Consecutive bars outside band
            if not should_close and consec_outside is not None:
                outside = False
                if position["side"] == "LONG" and price < bb_lower[i]:
                    outside = True
                elif position["side"] == "SHORT" and price > bb_upper[i]:
                    outside = True

                if outside:
                    position["consec"] += 1
                else:
                    position["consec"] = 0

                if position["consec"] >= consec_outside:
                    should_close = True
                    reason = "CONSEC"

            if should_close:
                if position["side"] == "LONG":
                    pnl_pct = (price - position["entry"]) / position["entry"]
                else:
                    pnl_pct = (position["entry"] - price) / position["entry"]
                pnl_jpy = pnl_pct * position["entry"] * position["size"]
                equity += pnl_jpy
                trades.append((position["side"], pnl_pct, i - position["bar"], reason))
                position = None

    # Close remaining
    if position:
        price = closes[-1]
        if position["side"] == "LONG":
            pnl_pct = (price - position["entry"]) / position["entry"]
        else:
            pnl_pct = (position["entry"] - price) / position["entry"]
        pnl_jpy = pnl_pct * position["entry"] * position["size"]
        equity += pnl_jpy
        trades.append((position["side"], pnl_pct, n - 1 - position["bar"], "END"))
    eq_curve.append(equity)

    # Metrics
    years = len(eq_curve) / (6 * 365) if eq_curve else 1
    final = equity
    annual = ((final / initial_capital) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else -100
    total_ret = (final / initial_capital - 1) * 100

    num_trades = len(trades)
    wins = sum(1 for _, p, _, _ in trades if p > 0)
    win_rate = wins / num_trades * 100 if num_trades > 0 else 0

    peak = eq_curve[0]
    max_dd = 0
    for eq in eq_curve:
        if eq > peak: peak = eq
        dd = (eq - peak) / peak
        if dd < max_dd: max_dd = dd
    max_dd *= 100

    if len(eq_curve) > 1:
        ea = np.array(eq_curve)
        rets = np.diff(ea) / ea[:-1]
        sharpe = (np.mean(rets) / np.std(rets)) * math.sqrt(6 * 365) if np.std(rets) > 0 else 0
    else:
        sharpe = 0

    win_pcts = [p * 100 for _, p, _, _ in trades if p > 0]
    loss_pcts = [p * 100 for _, p, _, _ in trades if p <= 0]
    avg_win = np.mean(win_pcts) if win_pcts else 0
    avg_loss = np.mean(loss_pcts) if loss_pcts else 0

    tp_count = sum(1 for _, _, _, r in trades if r == "TP")
    sl_count = sum(1 for _, _, _, r in trades if r.startswith("SL"))
    time_count = sum(1 for _, _, _, r in trades if r == "TIME")
    adx_count = sum(1 for _, _, _, r in trades if r == "ADX_EXIT")
    consec_count = sum(1 for _, _, _, r in trades if r == "CONSEC")

    avg_hold = np.mean([h for _, _, h, _ in trades]) * 4 if trades else 0  # hours

    return {
        "annual": annual, "total_ret": total_ret, "trades": num_trades,
        "win_rate": win_rate, "max_dd": max_dd, "sharpe": sharpe,
        "avg_win": avg_win, "avg_loss": avg_loss, "final": final,
        "tp": tp_count, "sl": sl_count, "time": time_count,
        "adx_exit": adx_count, "consec": consec_count,
        "avg_hold_h": avg_hold,
    }


def main():
    print("=" * 90)
    print("  BB 3σ逆張り — 損切り最適化スイープ")
    print("=" * 90)

    data = fetch_hourly_data(days=365 * 3)
    data_4h = aggregate_to_4h(data)
    closes = data_4h["close"]
    highs_arr = data_4h["high"]
    lows_arr = data_4h["low"]
    n = len(closes)
    print(f"4時間足: {n} candles ({n/6:.0f} days)")

    # Pre-compute indicators
    print("Computing indicators...")
    bb_upper, bb_middle, bb_lower = bollinger_bands(closes, period=20, num_std=3.0)
    atr_arr = atr(highs_arr, lows_arr, closes, period=14)
    adx_arr, _, _ = adx(highs_arr, lows_arr, closes, period=14)
    rsi_arr = rsi(closes, period=14)

    # ===================================================================
    # Phase 1: Individual stop-loss type comparison
    # ===================================================================
    print(f"\n{'='*110}")
    print("  Phase 1: 損切りタイプ別比較")
    print(f"{'='*110}")

    configs = [
        ("損切りなし (baseline)", {}),
        # Fixed %
        ("固定SL 2%", {"sl_fixed_pct": 0.02}),
        ("固定SL 3%", {"sl_fixed_pct": 0.03}),
        ("固定SL 5%", {"sl_fixed_pct": 0.05}),
        ("固定SL 7%", {"sl_fixed_pct": 0.07}),
        ("固定SL 10%", {"sl_fixed_pct": 0.10}),
        # ATR
        ("ATR×1.5", {"sl_atr_mult": 1.5}),
        ("ATR×2.0", {"sl_atr_mult": 2.0}),
        ("ATR×3.0", {"sl_atr_mult": 3.0}),
        # Time stop
        ("タイムストップ 10本(40h)", {"time_stop_bars": 10}),
        ("タイムストップ 20本(80h)", {"time_stop_bars": 20}),
        ("タイムストップ 30本(120h)", {"time_stop_bars": 30}),
        # ADX entry filter
        ("ADXフィルタ 入場<20", {"adx_entry_max": 20}),
        ("ADXフィルタ 入場<25", {"adx_entry_max": 25}),
        ("ADXフィルタ 入場<30", {"adx_entry_max": 30}),
        # ADX exit
        ("ADX決済 >25", {"adx_exit_th": 25}),
        ("ADX決済 >30", {"adx_exit_th": 30}),
        ("ADX決済 >35", {"adx_exit_th": 35}),
        # Consecutive outside
        ("連続バンド外 2本", {"consec_outside": 2}),
        ("連続バンド外 3本", {"consec_outside": 3}),
    ]

    hdr = (f"{'Strategy':<28} {'Annual%':>8} {'Trades':>7} {'WinR%':>7} "
           f"{'AvgW%':>7} {'AvgL%':>7} {'MaxDD%':>8} {'Sharpe':>7} "
           f"{'TP':>4} {'SL':>4} {'TIME':>4} {'ADX':>4} {'Hold':>6}")
    print(hdr)
    print("-" * 110)

    phase1_results = []
    for name, params in configs:
        r = backtest_bb_opt(closes, highs_arr, lows_arr, n,
                            bb_upper, bb_middle, bb_lower,
                            atr_arr, adx_arr, rsi_arr, **params)
        phase1_results.append((name, params, r))
        print(f"{name:<28} {r['annual']:>7.1f}% {r['trades']:>7d} {r['win_rate']:>6.1f}% "
              f"{r['avg_win']:>6.2f}% {r['avg_loss']:>6.2f}% {r['max_dd']:>7.1f}% "
              f"{r['sharpe']:>7.2f} {r['tp']:>4d} {r['sl']:>4d} {r['time']:>4d} "
              f"{r['adx_exit']:>4d} {r['avg_hold_h']:>5.0f}h")

    print(f"{'='*110}")

    # ===================================================================
    # Phase 2: Combination sweep (top performers)
    # ===================================================================
    print(f"\n{'='*110}")
    print("  Phase 2: 組み合わせスイープ")
    print(f"{'='*110}")

    sl_fixed_opts = [None, 0.03, 0.05, 0.07, 0.10]
    sl_atr_opts = [None, 1.5, 2.0, 3.0]
    time_opts = [None, 15, 20, 30]
    adx_entry_opts = [None, 20, 25, 30]
    adx_exit_opts = [None, 25, 30, 35]
    consec_opts = [None, 2, 3]

    # Full combo is too large; smart pruning
    # Use: (sl_type × time × adx_filter)
    combo_results = []
    count = 0

    for sl_fix in sl_fixed_opts:
        for sl_atr in sl_atr_opts:
            if sl_fix is not None and sl_atr is not None:
                continue  # don't combine two SL types
            for ts in time_opts:
                for adx_e in adx_entry_opts:
                    for adx_x in adx_exit_opts:
                        for cc in consec_opts:
                            r = backtest_bb_opt(
                                closes, highs_arr, lows_arr, n,
                                bb_upper, bb_middle, bb_lower,
                                atr_arr, adx_arr, rsi_arr,
                                sl_fixed_pct=sl_fix, sl_atr_mult=sl_atr,
                                time_stop_bars=ts, adx_entry_max=adx_e,
                                adx_exit_th=adx_x, consec_outside=cc)

                            tag = []
                            if sl_fix: tag.append(f"SL{sl_fix*100:.0f}%")
                            if sl_atr: tag.append(f"ATR×{sl_atr}")
                            if ts: tag.append(f"T{ts}")
                            if adx_e: tag.append(f"Ae<{adx_e}")
                            if adx_x: tag.append(f"Ax>{adx_x}")
                            if cc: tag.append(f"C{cc}")
                            tag_str = " ".join(tag) if tag else "baseline"

                            combo_results.append((tag_str, r))
                            count += 1

    print(f"Tested: {count} combinations")

    # Sort by annual return
    combo_results.sort(key=lambda x: x[1]["annual"], reverse=True)

    # Top 30
    print(f"\n  TOP 30 (by Annual Return)")
    print(hdr)
    print("-" * 110)
    for rank, (tag, r) in enumerate(combo_results[:30], 1):
        print(f"{tag:<28} {r['annual']:>7.1f}% {r['trades']:>7d} {r['win_rate']:>6.1f}% "
              f"{r['avg_win']:>6.2f}% {r['avg_loss']:>6.2f}% {r['max_dd']:>7.1f}% "
              f"{r['sharpe']:>7.2f} {r['tp']:>4d} {r['sl']:>4d} {r['time']:>4d} "
              f"{r['adx_exit']:>4d} {r['avg_hold_h']:>5.0f}h")

    # Best risk-adjusted (Sharpe)
    good = [(t, r) for t, r in combo_results if r["sharpe"] > 0 and r["trades"] > 20]
    good.sort(key=lambda x: x[1]["sharpe"], reverse=True)

    if good:
        print(f"\n  BEST RISK-ADJUSTED (Sharpe > 0, Trades > 20)")
        print(hdr)
        print("-" * 110)
        for rank, (tag, r) in enumerate(good[:15], 1):
            print(f"{tag:<28} {r['annual']:>7.1f}% {r['trades']:>7d} {r['win_rate']:>6.1f}% "
                  f"{r['avg_win']:>6.2f}% {r['avg_loss']:>6.2f}% {r['max_dd']:>7.1f}% "
                  f"{r['sharpe']:>7.2f} {r['tp']:>4d} {r['sl']:>4d} {r['time']:>4d} "
                  f"{r['adx_exit']:>4d} {r['avg_hold_h']:>5.0f}h")

    # Profitable strategies
    profitable = [(t, r) for t, r in combo_results if r["annual"] > 0]
    print(f"\n  黒字戦略: {len(profitable)} / {count}")

    # Best overall
    if combo_results:
        best_tag, best_r = combo_results[0]
        print(f"\n{'='*110}")
        print(f"  BEST STRATEGY: {best_tag}")
        print(f"{'='*110}")
        print(f"  年利:     {best_r['annual']:>+.1f}%")
        print(f"  3年累計:  {best_r['total_ret']:>+.1f}%")
        print(f"  トレード: {best_r['trades']}回")
        print(f"  勝率:     {best_r['win_rate']:.1f}%")
        print(f"  平均勝ち: +{best_r['avg_win']:.2f}%")
        print(f"  平均負け: {best_r['avg_loss']:.2f}%")
        print(f"  MaxDD:    {best_r['max_dd']:.1f}%")
        print(f"  Sharpe:   {best_r['sharpe']:.2f}")
        print(f"  平均保有: {best_r['avg_hold_h']:.0f}h")
        print(f"  決済内訳: TP={best_r['tp']} SL={best_r['sl']} "
              f"TIME={best_r['time']} ADX={best_r['adx_exit']}")
        if best_r['annual'] > 0:
            print(f"\n  100万円 → 3年後 {best_r['final']:,.0f} JPY")
        print(f"{'='*110}")


if __name__ == "__main__":
    main()
