#!/usr/bin/env python3
"""
BB_Reversal_Martin: 実データバックテスト（Yahoo Finance M15）
"""

import datetime
import numpy as np
import pandas as pd
import yfinance as yf

RISK_PCT       = 0.8
RR_RATIO       = 2.0
BE_TRIGGER_RR  = 1.0
PARTIAL_RR     = 1.5
PARTIAL_PCT    = 0.5
TRAIL_ATR_MULT = 0.5
MAX_HOLD_BARS  = 20
MARTIN = [1.0, 1.5, 2.0]
SL_MULTS = {1: 2.0, 2: 1.8, 3: 1.5}
ATR_FILTER_MULT = 2.5
INITIAL_EQUITY = 500000

PAIRS = {
    "USDJPY": {"ticker": "USDJPY=X", "pip": 0.01, "pip_mult": 100, "spread": 0.3, "tick_value": 100},
    "EURJPY": {"ticker": "EURJPY=X", "pip": 0.01, "pip_mult": 100, "spread": 0.5, "tick_value": 100},
    "GBPJPY": {"ticker": "GBPJPY=X", "pip": 0.01, "pip_mult": 100, "spread": 0.7, "tick_value": 100},
    "AUDJPY": {"ticker": "AUDJPY=X", "pip": 0.01, "pip_mult": 100, "spread": 0.5, "tick_value": 100},
    "EURUSD": {"ticker": "EURUSD=X", "pip": 0.0001, "pip_mult": 10000, "spread": 0.3, "tick_value": 150},
    "AUDUSD": {"ticker": "AUDUSD=X", "pip": 0.0001, "pip_mult": 10000, "spread": 0.5, "tick_value": 150},
}

def fetch_data(ticker):
    df = yf.download(ticker, period="60d", interval="15m", progress=False)
    if df.empty:
        return None
    df = df.droplevel("Ticker", axis=1) if isinstance(df.columns, pd.MultiIndex) else df
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    df = df[["open", "high", "low", "close"]].dropna()
    df["time"] = df.index
    df = df.reset_index(drop=True)
    return df

def calc_indicators(df, spread_pips=1.0):
    c = df["close"]
    df["sma200"] = c.rolling(200).mean()
    df["sma50"]  = c.rolling(50).mean()
    df["sma20"]  = c.rolling(20).mean()
    df["sma200_up"] = df["sma200"] > df["sma200"].shift(5)
    df["sma50_up"]  = df["sma50"] > df["sma50"].shift(5)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=10, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=10, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    tr = pd.concat([df["high"] - df["low"],
                     (df["high"] - c.shift(1)).abs(),
                     (df["low"] - c.shift(1)).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr14_ma100"] = df["atr14"].rolling(100).mean()
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2.5 * bb_std
    df["bb_lower"] = bb_mid - 2.5 * bb_std
    fbb_mid = c.rolling(10).mean()
    fbb_std = c.rolling(10).std()
    df["fbb_upper"] = fbb_mid + 2.0 * fbb_std
    df["fbb_lower"] = fbb_mid - 2.0 * fbb_std
    df["spread_pips"] = spread_pips
    return df.dropna().reset_index(drop=True)

def check_signals(row, prev, enabled=(1, 2, 3)):
    if row["atr14"] >= row["atr14_ma100"] * ATR_FILTER_MULT:
        return 0
    c1, c2, rsi = row["close"], prev["close"], row["rsi"]
    up200, up50 = row["sma200_up"], row["sma50_up"]
    if 1 in enabled:
        if up200 and c2 <= prev["bb_lower"] and c1 > row["bb_lower"] and rsi < 42:
            return 1
        if not up200 and c2 >= prev["bb_upper"] and c1 < row["bb_upper"] and rsi > 58:
            return -1
    if 2 in enabled:
        if up200 and up50 and c1 <= row["fbb_lower"] and rsi < 48:
            return 2
        if not up200 and not up50 and c1 >= row["fbb_upper"] and rsi > 52:
            return -2
    if 3 in enabled:
        gap = abs(row["sma20"] - row["sma50"])
        if gap >= row["atr14"] * 2:
            lo, hi = min(row["sma20"], row["sma50"]), max(row["sma20"], row["sma50"])
            if up200 and lo <= c1 <= hi and 30 <= rsi <= 50:
                return 3
            if not up200 and lo <= c1 <= hi and 50 <= rsi <= 70:
                return -3
    return 0

def run_backtest(df, cfg, enabled=(1, 2, 3)):
    pip_mult = cfg["pip_mult"]
    tick_val = cfg["tick_value"]
    pip_unit = cfg["pip"]
    min_lot = 0.01
    lot_step = 0.01
    equity = float(INITIAL_EQUITY)
    equity_peak = equity
    max_dd_pct = 0.0
    month_start_eq = equity
    martin_stage = 0
    consec_losses = 0
    trades = []
    monthly_pnl = []
    current_month = None

    i = 1
    while i < len(df) - MAX_HOLD_BARS - 1:
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        row_month = pd.to_datetime(row["time"]).month
        if current_month is None:
            current_month = row_month
            month_start_eq = equity
        elif row_month != current_month:
            monthly_pnl.append({"month": current_month,
                                "pnl_pct": (equity - month_start_eq) / month_start_eq * 100})
            current_month = row_month
            month_start_eq = equity

        sig = check_signals(row, prev, enabled=enabled)
        if sig == 0:
            i += 1
            continue

        direction = 1 if sig > 0 else -1
        sig_type = abs(sig)
        atr = row["atr14"]
        sl_mult = SL_MULTS.get(sig_type, 1.5)
        sl_dist = atr * sl_mult
        tp_dist = sl_dist * RR_RATIO
        spread_dist = cfg["spread"] * pip_unit

        entry_price = row["close"]
        risk_amount = equity * RISK_PCT / 100.0
        martin_mult = MARTIN[min(martin_stage, 2)]
        sl_pips = sl_dist * pip_mult
        if sl_pips <= 0:
            i += 1
            continue
        raw_lots = (risk_amount * martin_mult) / (sl_pips * tick_val)
        lots = max(min_lot, int(raw_lots / lot_step) * lot_step)
        lots = round(lots, 2)

        sl_price = entry_price - direction * sl_dist
        tp_price = entry_price + direction * tp_dist
        be_activated = False
        partial_closed = False
        remaining_lots = lots
        realized_pnl = 0.0
        result = "timeout"
        exit_bar = i
        pnl = 0.0
        is_timeout = False

        for j in range(1, MAX_HOLD_BARS + 1):
            if i + j >= len(df):
                break
            bar = df.iloc[i + j]
            bar_atr = bar["atr14"] if not np.isnan(bar["atr14"]) else atr
            bar_high = bar["high"]
            bar_low = bar["low"]

            if direction == 1:
                # BE/部分利確をhighで判定
                if not be_activated and (bar_high - entry_price) >= sl_dist * BE_TRIGGER_RR:
                    sl_price = entry_price + spread_dist
                    be_activated = True
                if not partial_closed and (bar_high - entry_price) >= sl_dist * PARTIAL_RR:
                    close_lots = round(remaining_lots * PARTIAL_PCT, 2)
                    if close_lots >= min_lot and (remaining_lots - close_lots) >= min_lot:
                        partial_price = entry_price + sl_dist * PARTIAL_RR
                        realized_pnl += (partial_price - entry_price) * pip_mult * close_lots * tick_val
                        remaining_lots = round(remaining_lots - close_lots, 2)
                        partial_closed = True
                        trail_sl = partial_price - bar_atr * TRAIL_ATR_MULT
                        if trail_sl > sl_price:
                            sl_price = trail_sl
                if partial_closed:
                    trail_sl = bar_high - bar_atr * TRAIL_ATR_MULT
                    if trail_sl > sl_price:
                        sl_price = trail_sl
                # SL判定
                if bar_low <= sl_price:
                    pnl = (sl_price - entry_price) * pip_mult * remaining_lots * tick_val + realized_pnl
                    result = "BE" if be_activated else "SL"
                    exit_bar = i + j
                    break
                # TP判定
                if bar_high >= tp_price:
                    pnl = tp_dist * pip_mult * remaining_lots * tick_val + realized_pnl
                    result = "TP"
                    exit_bar = i + j
                    break
            else:
                if not be_activated and (entry_price - bar_low) >= sl_dist * BE_TRIGGER_RR:
                    sl_price = entry_price - spread_dist
                    be_activated = True
                if not partial_closed and (entry_price - bar_low) >= sl_dist * PARTIAL_RR:
                    close_lots = round(remaining_lots * PARTIAL_PCT, 2)
                    if close_lots >= min_lot and (remaining_lots - close_lots) >= min_lot:
                        partial_price = entry_price - sl_dist * PARTIAL_RR
                        realized_pnl += (entry_price - partial_price) * pip_mult * close_lots * tick_val
                        remaining_lots = round(remaining_lots - close_lots, 2)
                        partial_closed = True
                        trail_sl = partial_price + bar_atr * TRAIL_ATR_MULT
                        if trail_sl < sl_price:
                            sl_price = trail_sl
                if partial_closed:
                    trail_sl = bar_low + bar_atr * TRAIL_ATR_MULT
                    if trail_sl < sl_price:
                        sl_price = trail_sl
                if bar_high >= sl_price:
                    pnl = (entry_price - sl_price) * pip_mult * remaining_lots * tick_val + realized_pnl
                    result = "BE" if be_activated else "SL"
                    exit_bar = i + j
                    break
                if bar_low <= tp_price:
                    pnl = tp_dist * pip_mult * remaining_lots * tick_val + realized_pnl
                    result = "TP"
                    exit_bar = i + j
                    break
        else:
            exit_price = df.iloc[min(i + MAX_HOLD_BARS, len(df) - 1)]["close"]
            pnl = direction * (exit_price - entry_price) * pip_mult * remaining_lots * tick_val + realized_pnl
            is_timeout = True

        spread_cost = cfg["spread"] * tick_val * lots
        pnl -= spread_cost

        if is_timeout or result == "BE":
            consec_losses = 0
            martin_stage = 0
        elif pnl < 0:
            consec_losses += 1
            if consec_losses >= 3:
                martin_stage = 0
                consec_losses = 0
            else:
                martin_stage = min(consec_losses, 2)
        else:
            consec_losses = 0
            martin_stage = 0

        equity += pnl
        equity_peak = max(equity_peak, equity)
        dd_pct = (equity_peak - equity) / equity_peak * 100 if equity_peak > 0 else 0
        max_dd_pct = max(max_dd_pct, dd_pct)

        trades.append({"result": result, "pnl": pnl, "sig_type": sig_type,
                        "lots": lots, "be": be_activated, "partial": partial_closed,
                        "equity": equity})
        i = exit_bar + 1

    if month_start_eq > 0 and current_month is not None:
        monthly_pnl.append({"month": current_month,
                            "pnl_pct": (equity - month_start_eq) / month_start_eq * 100})
    return trades, max_dd_pct, monthly_pnl

def summarize(trades, max_dd_pct, monthly_pnl):
    if not trades:
        return None
    df_t = pd.DataFrame(trades)
    n = len(df_t)
    wins = df_t[df_t["pnl"] > 0]
    losses = df_t[df_t["pnl"] <= 0]
    wr = len(wins) / n * 100
    total_pnl = df_t["pnl"].sum()
    gross_w = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_l = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    median_monthly = pd.DataFrame(monthly_pnl)["pnl_pct"].median() if monthly_pnl else 0
    results = df_t["result"].value_counts().to_dict()
    return {"n": n, "wr": wr, "pnl": total_pnl, "pf": pf, "dd": max_dd_pct,
            "median_monthly": median_monthly, "results": results,
            "be_count": int(df_t["be"].sum()), "partial_count": int(df_t["partial"].sum()),
            "monthly": monthly_pnl}

def main():
    print("=" * 90)
    print("BB_Reversal_Martin 実データバックテスト (Yahoo Finance M15, 約3ヶ月)")
    print(f"初期資金: ¥{INITIAL_EQUITY:,} | リスク: {RISK_PCT}% | RR: {RR_RATIO}")
    print("=" * 90)

    strategies = {"1.BB逆張り": (1,), "2.高速BB": (2,), "3.押し目": (3,), "全統合": (1, 2, 3)}
    all_results = {}
    pair_data = {}

    # データ取得
    print("\n[データ取得中]")
    for pair_name, cfg in PAIRS.items():
        print(f"  {pair_name} ...", end=" ")
        df = fetch_data(cfg["ticker"])
        if df is None or len(df) < 300:
            print("FAILED")
            continue
        df = calc_indicators(df, cfg["spread"])
        pair_data[pair_name] = df
        dates = f"{df['time'].iloc[0].date()} ~ {df['time'].iloc[-1].date()}"
        print(f"{len(df)} bars ({dates})")

    # バックテスト
    print("\n[バックテスト実行]")
    for pair_name in pair_data:
        cfg = PAIRS[pair_name]
        df = pair_data[pair_name]
        for sn, en in strategies.items():
            trades, dd, monthly = run_backtest(df, cfg, enabled=en)
            all_results[(pair_name, sn)] = summarize(trades, dd, monthly)

    # ロジック別詳細
    for sn in strategies:
        print(f"\n{'='*90}")
        print(f"  【{sn}】")
        print(f"{'='*90}")
        print(f"  {'ペア':>8s}  {'SP':>4s}  {'PF':>5s}  {'勝率':>6s}  {'最大DD':>7s}  {'純損益':>12s}  {'月利中央':>7s}  {'N':>4s}  {'BE':>4s}  {'部分':>4s}  {'決済内訳'}")
        print(f"  {'-'*90}")
        for pn in pair_data:
            cfg = PAIRS[pn]
            s = all_results.get((pn, sn))
            if s and s["n"] > 0:
                pf_s = f"{s['pf']:.2f}" if s["pf"] < 100 else "∞"
                pnl_s = f"+¥{s['pnl']:,.0f}" if s["pnl"] >= 0 else f"-¥{abs(s['pnl']):,.0f}"
                res = " ".join(f"{k}:{v}" for k, v in sorted(s["results"].items()))
                print(f"  {pn:>8s}  {cfg['spread']:>4.1f}  {pf_s:>5s}  {s['wr']:>5.1f}%  {s['dd']:>6.1f}%  {pnl_s:>12s}  {s['median_monthly']:>+6.1f}%  {s['n']:>4d}  {s['be_count']:>4d}  {s['partial_count']:>4d}  {res}")
            else:
                print(f"  {pn:>8s}  {cfg['spread']:>4.1f}    -      -       -            -        -     0     -     -  シグナルなし")

    # PF/損益クロス集計
    sn_list = list(strategies.keys())
    print(f"\n{'='*90}")
    print(f"  【PF クロス集計】")
    print(f"{'='*90}")
    hdr = f"  {'ペア':>8s}" + "".join(f"  {s:>12s}" for s in sn_list)
    print(hdr)
    print(f"  {'-'*(8 + 14 * len(sn_list))}")
    for pn in pair_data:
        row = f"  {pn:>8s}"
        for sn in sn_list:
            s = all_results.get((pn, sn))
            if s and s["n"] > 0:
                row += f"  {s['pf']:>12.2f}" if s["pf"] < 100 else f"  {'∞':>12s}"
            else:
                row += f"  {'---':>12s}"
        print(row)

    print(f"\n{'='*90}")
    print(f"  【純損益 クロス集計 (¥)】")
    print(f"{'='*90}")
    print(hdr)
    print(f"  {'-'*(8 + 14 * len(sn_list))}")
    totals = {sn: 0 for sn in sn_list}
    for pn in pair_data:
        row = f"  {pn:>8s}"
        for sn in sn_list:
            s = all_results.get((pn, sn))
            v = s["pnl"] if s and s["n"] > 0 else 0
            row += f"  {v:>+11,.0f} "
            totals[sn] += v
        print(row)
    print(f"  {'-'*(8 + 14 * len(sn_list))}")
    print(f"  {'合計':>8s}" + "".join(f"  {totals[sn]:>+11,.0f} " for sn in sn_list))

    # 月利テーブル (全統合)
    month_names = {1:"1月",2:"2月",3:"3月",4:"4月",5:"5月",6:"6月",
                   7:"7月",8:"8月",9:"9月",10:"10月",11:"11月",12:"12月"}
    pair_names = list(pair_data.keys())

    print(f"\n{'='*90}")
    print(f"  【月利(%) - 全統合】")
    print(f"{'='*90}")
    mhdr = f"  {'月':>4s}" + "".join(f"  {pn:>10s}" for pn in pair_names)
    print(mhdr)
    print(f"  {'-'*(4 + 12 * len(pair_names))}")

    pair_monthly = {}
    for pn in pair_names:
        s = all_results.get((pn, "全統合"))
        if s and s.get("monthly"):
            pair_monthly[pn] = {m["month"]: m["pnl_pct"] for m in s["monthly"]}
        else:
            pair_monthly[pn] = {}

    all_months = sorted(set(m for pm in pair_monthly.values() for m in pm))
    for mo in all_months:
        row = f"  {month_names.get(mo, str(mo)):>4s}"
        for pn in pair_names:
            val = pair_monthly[pn].get(mo)
            if val is not None:
                row += f"  {val:>+9.1f}%"
            else:
                row += f"  {'---':>10s}"
        print(row)

    # 判定
    print(f"\n{'='*90}")
    print(f"  【全統合 判定】")
    print(f"{'='*90}")
    print(f"  {'ペア':>8s}  {'SP':>4s}  {'PF':>5s}  {'勝率':>6s}  {'最大DD':>7s}  {'純損益':>12s}  {'月利中央':>7s}  {'判定'}")
    print(f"  {'-'*70}")
    for pn in pair_data:
        cfg = PAIRS[pn]
        s = all_results.get((pn, "全統合"))
        if s and s["n"] > 0:
            pf_s = f"{s['pf']:.2f}" if s["pf"] < 100 else "∞"
            pnl_s = f"+¥{s['pnl']:,.0f}" if s["pnl"] >= 0 else f"-¥{abs(s['pnl']):,.0f}"
            if s["pf"] >= 1.3 and s["dd"] < 10:
                v = "稼働推奨"
            elif s["pf"] >= 1.1:
                v = "条件付き"
            else:
                v = "要改善"
            print(f"  {pn:>8s}  {cfg['spread']:>4.1f}  {pf_s:>5s}  {s['wr']:>5.1f}%  {s['dd']:>6.1f}%  {pnl_s:>12s}  {s['median_monthly']:>+6.1f}%  {v}")

    print(f"\n※Yahoo Finance M15実データ（約3ヶ月）。BE/部分利確/トレーリング/マーチン全実装。")

if __name__ == "__main__":
    main()
