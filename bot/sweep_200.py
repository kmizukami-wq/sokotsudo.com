#!/usr/bin/env python3
"""
Fast exhaustive parameter sweep to find strategies with >200% annual return.

Pre-trains ML models once, then sweeps all parameter combinations using
cached predictions. This is ~100x faster than per-run training.
"""

import sys
import math
import logging
import itertools
import numpy as np
from datetime import datetime

# Suppress all logging from risk_manager (PositionSizer logs every bar)
logging.getLogger("risk_manager").setLevel(logging.CRITICAL)

from backtest import (
    fetch_hourly_data, BacktestResult, Trade,
    atr, rsi, adx, realized_volatility,
)
from risk_manager import (
    VolatilityRegime, PositionSizer, DynamicStopLoss,
    EntryFilter, AdaptiveThreshold,
)
from ml_model import FeatureBuilder, SignalPredictor, features_to_matrix


def pretrain_walk_forward(X, valid_mask, labels, full_mask, n,
                          train_hours=24*180, embargo_hours=24*7,
                          retrain_interval=24*30):
    """Pre-train walk-forward ML and cache per-bar predictions.

    Returns:
        prob_long: array[n] of P(long correct) for each bar
        prob_short: array[n] of P(short correct) for each bar
    """
    prob_long = np.full(n, 0.5)
    prob_short = np.full(n, 0.5)

    predictor = None
    last_train = 0

    for i in range(200, n):
        if (predictor is None or i - last_train >= retrain_interval) and i >= train_hours:
            ts = max(0, i - train_hours)
            te = i - embargo_hours
            if te > ts + 100:
                idx = np.arange(ts, te)
                m = full_mask[idx]
                Xt, yt = X[idx[m]], labels[idx[m]]
                if len(Xt) >= 100:
                    predictor = SignalPredictor()
                    predictor.train(Xt, yt)
                    last_train = i

        if predictor is not None and predictor.is_ready and valid_mask[i]:
            x_i = X[i:i+1]
            pl = predictor.predict_proba(x_i, direction="LONG")
            ps = predictor.predict_proba(x_i, direction="SHORT")
            prob_long[i] = float(pl[0]) if hasattr(pl, '__len__') else float(pl)
            prob_short[i] = float(ps[0]) if hasattr(ps, '__len__') else float(ps)

    return prob_long, prob_short


def fast_backtest(closes, n,
                  atr_arr, rsi_arr, adx_arr, rvol_annual,
                  prob_long, prob_short,
                  initial_capital=1_000_000,
                  capital_ratio=0.90,
                  leverage=1.0,
                  adx_th=25.0,
                  sl_mult=2.5,
                  lb_long=72,
                  lb_short=24,
                  prob_skip=0.40,
                  prob_boost=0.55,
                  boost_mult=1.2):
    """Ultra-fast backtest — all inline, no object creation, no logging."""

    effective_cap = min(capital_ratio * leverage, 5.0)

    # Inline vol regime thresholds
    VOL_LOW_TH = 0.30
    VOL_HIGH_TH = 0.50
    # Inline adaptive thresholds
    TH_LOW = 0.0075
    TH_NORMAL = 0.01
    TH_HIGH = 0.015
    # Inline vol multipliers for position sizing
    VM_LOW = 1.2
    VM_NORMAL = 1.0
    VM_HIGH = 0.6
    # Inline SL regime multipliers
    SL_LOW = 1.5
    SL_NORMAL = 2.0
    SL_HIGH = 2.5
    SL_MIN = 0.015
    SL_MAX = 0.06

    equity = initial_capital
    position = None  # (side_is_long, size, entry, extreme, bar)
    num_trades = 0
    wins = 0
    win_pnls = []
    loss_pnls = []
    eq_samples = []  # sample every 24h for faster sharpe/dd

    start_bar = max(lb_long, 200)

    for i in range(start_bar, n):
        if i % 24 == 0:
            eq_samples.append(equity)

        price = closes[i]

        # Volatility regime (inline)
        rv = rvol_annual[i]
        if rv != rv:  # NaN check
            regime = 1  # NORMAL
        elif rv < VOL_LOW_TH:
            regime = 0  # LOW
        elif rv > VOL_HIGH_TH:
            regime = 2  # HIGH
        else:
            regime = 1  # NORMAL

        if position is None:
            # Momentum
            prev_l = closes[i - lb_long]
            if prev_l <= 0:
                continue
            mom_l = (price - prev_l) / prev_l

            # Adaptive threshold
            if regime == 0:
                threshold = TH_LOW
            elif regime == 2:
                threshold = TH_HIGH
            else:
                threshold = TH_NORMAL

            direction = 0  # 0=none, 1=long, -1=short
            if mom_l > threshold:
                # MTF confirmation
                if i >= lb_short:
                    mom_s = (price - closes[i - lb_short]) / closes[i - lb_short]
                    if mom_s <= 0:
                        continue
                # ADX filter
                cur_adx = adx_arr[i]
                if cur_adx == cur_adx and cur_adx < adx_th:  # NaN-safe
                    continue
                # RSI filter (don't long if overbought)
                cur_rsi = rsi_arr[i]
                if cur_rsi == cur_rsi and cur_rsi > 70:
                    continue
                direction = 1
            elif mom_l < -threshold:
                if i >= lb_short:
                    mom_s = (price - closes[i - lb_short]) / closes[i - lb_short]
                    if mom_s >= 0:
                        continue
                cur_adx = adx_arr[i]
                if cur_adx == cur_adx and cur_adx < adx_th:
                    continue
                cur_rsi = rsi_arr[i]
                if cur_rsi == cur_rsi and cur_rsi < 30:
                    continue
                direction = -1

            if direction != 0:
                # ML filter
                prob = prob_long[i] if direction == 1 else prob_short[i]
                if prob <= prob_skip:
                    continue

                sm = boost_mult if prob > prob_boost else 1.0

                # Position sizing (inline)
                if regime == 0:
                    vm = VM_LOW
                elif regime == 2:
                    vm = VM_HIGH
                else:
                    vm = VM_NORMAL

                size = (equity * effective_cap * vm * sm) / price
                if size < 0.001:
                    continue

                # side_is_long, size, entry, extreme, bar
                position = (direction == 1, size, price, price, i)

        else:
            side_is_long, size, entry, extreme, bar = position

            # Dynamic stop (inline)
            cur_atr = atr_arr[i]
            if cur_atr != cur_atr or price <= 0:  # NaN
                stop_pct = 0.025
            else:
                if regime == 0:
                    sm = SL_LOW
                elif regime == 2:
                    sm = SL_HIGH
                else:
                    sm = SL_NORMAL
                stop_pct = (cur_atr * sm) / price
                if stop_pct < SL_MIN:
                    stop_pct = SL_MIN
                elif stop_pct > SL_MAX:
                    stop_pct = SL_MAX

            should_close = False
            if side_is_long:
                if price > extreme:
                    extreme = price
                if (price - extreme) / extreme <= -stop_pct:
                    should_close = True
            else:
                if price < extreme:
                    extreme = price
                if extreme > 0 and (price - extreme) / extreme >= stop_pct:
                    should_close = True

            if not should_close:
                position = (side_is_long, size, entry, extreme, bar)
            else:
                if side_is_long:
                    pnl_pct = (price - entry) / entry
                else:
                    pnl_pct = (entry - price) / entry
                equity += pnl_pct * entry * size
                num_trades += 1
                if pnl_pct > 0:
                    wins += 1
                    win_pnls.append(pnl_pct * 100)
                else:
                    loss_pnls.append(pnl_pct * 100)
                position = None

    # Close remaining
    if position:
        side_is_long, size, entry, extreme, bar = position
        price = closes[-1]
        if side_is_long:
            pnl_pct = (price - entry) / entry
        else:
            pnl_pct = (entry - price) / entry
        equity += pnl_pct * entry * size
        num_trades += 1
        if pnl_pct > 0:
            wins += 1
            win_pnls.append(pnl_pct * 100)
        else:
            loss_pnls.append(pnl_pct * 100)
    eq_samples.append(equity)

    # Compute metrics
    days_total = (n - start_bar) / 24
    years = days_total / 365 if days_total > 0 else 1
    final = equity
    total_ret = (final / initial_capital - 1) * 100
    annual = ((final / initial_capital) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else -100

    win_rate = wins / num_trades * 100 if num_trades > 0 else 0

    # Max drawdown from daily samples
    peak = eq_samples[0] if eq_samples else initial_capital
    max_dd = 0
    for eq in eq_samples:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak
        if dd < max_dd:
            max_dd = dd
    max_dd *= 100

    # Sharpe from daily samples
    if len(eq_samples) > 1:
        eq_arr = np.array(eq_samples)
        rets = np.diff(eq_arr) / eq_arr[:-1]
        std = np.std(rets)
        sharpe = (np.mean(rets) / std) * math.sqrt(365) if std > 0 else 0
    else:
        sharpe = 0

    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0

    return {
        "annual": annual, "total_ret": total_ret,
        "trades": num_trades, "win_rate": win_rate,
        "max_dd": max_dd, "sharpe": sharpe,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "final_eq": final,
    }


def main():
    print("=" * 80)
    print("  EXHAUSTIVE SWEEP: Finding >200% Annual Return Strategies")
    print("=" * 80)

    data = fetch_hourly_data(days=365 * 3)
    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    volumes = data["volume"]
    timestamps = data["timestamps"]
    n = len(closes)

    if n < 1000:
        print("ERROR: Not enough data")
        sys.exit(1)

    # Pre-compute indicators
    print("\nPre-computing indicators...")
    atr_arr = atr(highs, lows, closes, period=14)
    rsi_arr = rsi(closes, period=14)
    adx_arr, _, _ = adx(highs, lows, closes, period=14)
    rvol_arr = realized_volatility(closes, period=30)
    rvol_annual = rvol_arr * math.sqrt(24 * 365)

    # Pre-compute ML features
    print("Pre-computing ML features...")
    fb = FeatureBuilder()
    features = fb.build(closes, highs, lows, volumes, timestamps)

    # Pre-train walk-forward models for each horizon
    ml_cache = {}
    for hz in [5, 12]:
        print(f"\nPre-training walk-forward ML (horizon={hz}h)...")
        labels = fb.build_labels(closes, horizon=hz, threshold=0.01)
        X, valid = features_to_matrix(features, fb.feature_names)
        lv = np.isfinite(labels)
        fm = valid & lv

        for train_d in [180]:
            for retrain_d in [30]:
                key = (hz, train_d, retrain_d)
                print(f"  Training hz={hz} train={train_d}d retrain={retrain_d}d...")
                pl, ps = pretrain_walk_forward(
                    X, valid, labels, fm, n,
                    train_hours=24*train_d,
                    embargo_hours=24*7,
                    retrain_interval=24*retrain_d,
                )
                ml_cache[key] = (pl, ps)

    print(f"\nML cache: {len(ml_cache)} pre-trained models")

    # ===================================================================
    # SWEEP CONFIGS
    # ===================================================================
    sweep_params = {
        "leverage":      [1.0, 1.5, 2.0, 2.5, 3.0],
        "adx_th":        [20.0, 25.0, 30.0],
        "sl_mult":       [1.5, 2.0, 2.5, 3.0],
        "lb_long":       [48, 72, 96],
        "lb_short":      [12, 24],
        "prob_skip":     [0.30, 0.35, 0.40, 0.45],
        "prob_boost":    [0.50, 0.55, 0.60],
        "boost_mult":    [1.0, 1.5, 2.0, 2.5],
        "label_horizon": [5, 12],
        "train_days":    [180],
        "retrain_days":  [30],
    }

    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    all_combos = list(itertools.product(*values))
    total = len(all_combos)

    print(f"\nTotal combinations: {total}")
    print(f"Starting sweep at {datetime.now().strftime('%H:%M:%S')}...\n")

    results = []
    best_annual = 0
    count_200 = 0

    for idx, combo in enumerate(all_combos):
        config = dict(zip(keys, combo))

        # Skip invalid combos: prob_skip >= prob_boost
        if config["prob_skip"] >= config["prob_boost"]:
            continue

        ml_key = (config["label_horizon"], config["train_days"], config["retrain_days"])
        prob_long, prob_short = ml_cache[ml_key]

        r = fast_backtest(
            closes, n,
            atr_arr, rsi_arr, adx_arr, rvol_annual,
            prob_long, prob_short,
            capital_ratio=0.90,
            leverage=config["leverage"],
            adx_th=config["adx_th"],
            sl_mult=config["sl_mult"],
            lb_long=config["lb_long"],
            lb_short=config["lb_short"],
            prob_skip=config["prob_skip"],
            prob_boost=config["prob_boost"],
            boost_mult=config["boost_mult"],
        )

        results.append((config, r))

        if r["annual"] > 200:
            count_200 += 1

        if r["annual"] > best_annual:
            best_annual = r["annual"]
            tag = (f"L{config['leverage']:.1f} ADX{int(config['adx_th'])} "
                   f"SL{config['sl_mult']:.1f} LB{config['lb_long']}/{config['lb_short']} "
                   f"P{config['prob_skip']:.2f}/{config['prob_boost']:.2f} "
                   f"B{config['boost_mult']:.1f} H{config['label_horizon']} "
                   f"T{config['train_days']}d R{config['retrain_days']}d")
            print(f"  [{idx+1}/{total}] NEW BEST: {r['annual']:.1f}% annual | "
                  f"Sharpe={r['sharpe']:.2f} MaxDD={r['max_dd']:.1f}% "
                  f"Trades={r['trades']} WR={r['win_rate']:.1f}% | {tag}")

        if (idx + 1) % 10000 == 0:
            print(f"  [{idx+1}/{total}] Progress... best={best_annual:.1f}%, >200%: {count_200}")

    # ===================================================================
    # RESULTS
    # ===================================================================
    print(f"\nSweep completed at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Tested: {len(results)} valid combinations")

    # Sort by annual return
    results.sort(key=lambda x: x[1]["annual"], reverse=True)

    # Top 30
    print(f"\n{'='*140}")
    print(f"  TOP 30 STRATEGIES (by Annual Return)")
    print(f"{'='*140}")
    hdr = (f"{'#':>3} {'Annual%':>8} {'Return%':>9} {'Trades':>7} "
           f"{'WinR%':>7} {'AvgWin%':>8} {'AvgLos%':>8} {'MaxDD%':>8} {'Sharpe':>7} {'Config'}")
    print(hdr)
    print("-" * 140)

    for rank, (cfg, r) in enumerate(results[:30], 1):
        tag = (f"L{cfg['leverage']:.1f} ADX{int(cfg['adx_th'])} "
               f"SL{cfg['sl_mult']:.1f} LB{cfg['lb_long']}/{cfg['lb_short']} "
               f"P{cfg['prob_skip']:.2f}/{cfg['prob_boost']:.2f} "
               f"B{cfg['boost_mult']:.1f} H{cfg['label_horizon']} "
               f"T{cfg['train_days']}d R{cfg['retrain_days']}d")
        print(f"{rank:>3} {r['annual']:>7.1f}% {r['total_ret']:>8.1f}% {r['trades']:>7d} "
              f"{r['win_rate']:>6.1f}% {r['avg_win']:>7.2f}% {r['avg_loss']:>7.2f}% "
              f"{r['max_dd']:>7.1f}% {r['sharpe']:>7.2f}  {tag}")

    print(f"{'='*140}")

    # Count >200%
    over_200 = [(c, r) for c, r in results if r["annual"] > 200]
    print(f"\n  Strategies with >200% annual: {len(over_200)} / {len(results)}")

    # Best risk-adjusted
    good_risk = [(c, r) for c, r in results
                 if r["annual"] > 100 and r["sharpe"] > 1.0 and r["max_dd"] > -40]
    good_risk.sort(key=lambda x: x[1]["sharpe"], reverse=True)

    if good_risk:
        print(f"\n{'='*140}")
        print(f"  BEST RISK-ADJUSTED (Annual>100%, Sharpe>1.0, MaxDD>-40%)")
        print(f"{'='*140}")
        print(hdr)
        print("-" * 140)
        for rank, (cfg, r) in enumerate(good_risk[:20], 1):
            tag = (f"L{cfg['leverage']:.1f} ADX{int(cfg['adx_th'])} "
                   f"SL{cfg['sl_mult']:.1f} LB{cfg['lb_long']}/{cfg['lb_short']} "
                   f"P{cfg['prob_skip']:.2f}/{cfg['prob_boost']:.2f} "
                   f"B{cfg['boost_mult']:.1f} H{cfg['label_horizon']} "
                   f"T{cfg['train_days']}d R{cfg['retrain_days']}d")
            print(f"{rank:>3} {r['annual']:>7.1f}% {r['total_ret']:>8.1f}% {r['trades']:>7d} "
                  f"{r['win_rate']:>6.1f}% {r['avg_win']:>7.2f}% {r['avg_loss']:>7.2f}% "
                  f"{r['max_dd']:>7.1f}% {r['sharpe']:>7.2f}  {tag}")
        print(f"{'='*140}")

    # Sharpe > 1.5 strategies
    high_sharpe = [(c, r) for c, r in results if r["sharpe"] > 1.5 and r["trades"] > 50]
    high_sharpe.sort(key=lambda x: x[1]["annual"], reverse=True)

    if high_sharpe:
        print(f"\n{'='*140}")
        print(f"  HIGH SHARPE (>1.5) STRATEGIES")
        print(f"{'='*140}")
        print(hdr)
        print("-" * 140)
        for rank, (cfg, r) in enumerate(high_sharpe[:20], 1):
            tag = (f"L{cfg['leverage']:.1f} ADX{int(cfg['adx_th'])} "
                   f"SL{cfg['sl_mult']:.1f} LB{cfg['lb_long']}/{cfg['lb_short']} "
                   f"P{cfg['prob_skip']:.2f}/{cfg['prob_boost']:.2f} "
                   f"B{cfg['boost_mult']:.1f} H{cfg['label_horizon']} "
                   f"T{cfg['train_days']}d R{cfg['retrain_days']}d")
            print(f"{rank:>3} {r['annual']:>7.1f}% {r['total_ret']:>8.1f}% {r['trades']:>7d} "
                  f"{r['win_rate']:>6.1f}% {r['avg_win']:>7.2f}% {r['avg_loss']:>7.2f}% "
                  f"{r['max_dd']:>7.1f}% {r['sharpe']:>7.2f}  {tag}")
        print(f"{'='*140}")

    # Parameter importance
    print(f"\n{'='*80}")
    print(f"  PARAMETER IMPORTANCE (avg annual return by parameter value)")
    print(f"{'='*80}")
    for key in keys:
        print(f"\n  {key}:")
        for val in sweep_params[key]:
            subset = [r["annual"] for c, r in results if c[key] == val]
            if subset:
                avg = np.mean(subset)
                med = np.median(subset)
                top5 = np.mean(sorted(subset, reverse=True)[:max(1, len(subset)//20)])
                print(f"    {str(val):>8} → avg={avg:>7.1f}%  med={med:>7.1f}%  top5%={top5:>7.1f}%")


if __name__ == "__main__":
    main()
