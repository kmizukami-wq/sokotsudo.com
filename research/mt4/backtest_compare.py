#!/usr/bin/env python3
"""
BB_Reversal_Martin v1.00 vs v1.02 バックテスト比較
変更点:
  1. ATRフィルター: MA(ATR14,100)ループ → ATR(100) 単一呼び出し
  2. スプレッドフィルター: なし → MaxSpreadPips=3.0
  3. BB逆張りローソク足確認: なし → 陽線/陰線チェック
"""

import json
import urllib.request
import datetime
import numpy as np
import pandas as pd
from io import StringIO

# ============================================================
# データ取得 (Yahoo Finance CSV API fallback)
# ============================================================
def fetch_forex_data(pair="USDJPY=X", days=180):
    """Yahoo Finance からCSVでFXデータ取得（1h足）"""
    end = int(datetime.datetime.now().timestamp())
    start = end - days * 86400
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{pair}"
        f"?period1={start}&period2={end}&interval=1h&events=history"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            csv_text = resp.read().decode("utf-8")
        df = pd.read_csv(StringIO(csv_text), parse_dates=["Date"])
        df = df.rename(columns={"Date": "time", "Open": "open", "High": "high",
                                 "Low": "low", "Close": "close", "Volume": "volume"})
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        print(f"  Yahoo Finance: {len(df)} bars ({pair}, 1h)")
        return df
    except Exception as e:
        print(f"  Yahoo Finance failed: {e}")
        return None


def generate_synthetic_data(bars=5000):
    """Yahoo取得失敗時の合成データ（GBM + mean-reversion）"""
    np.random.seed(42)
    dt_minutes = 15
    price = 150.0
    prices = []
    t = datetime.datetime(2025, 1, 1, 0, 0)

    for i in range(bars):
        # skip weekends
        if t.weekday() >= 5:
            t += datetime.timedelta(minutes=dt_minutes)
            continue

        vol = 0.0003
        # 東京・ロンドン・NY で微妙にボラ変更
        hour_jst = (t.hour + 9) % 24
        if 6 <= hour_jst < 9:
            vol = 0.0001  # 早朝低ボラ
        elif 16 <= hour_jst < 22:
            vol = 0.0005  # ロンドン・NY 高ボラ

        ret = np.random.normal(0, vol) - 0.00001 * (price - 150.0)  # mean-revert
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(np.random.normal(0, vol * 0.5)))
        l = min(o, c) * (1 - abs(np.random.normal(0, vol * 0.5)))
        prices.append({"time": t, "open": o, "high": h, "low": l, "close": c, "volume": 100})
        price = c
        t += datetime.timedelta(minutes=dt_minutes)

    df = pd.DataFrame(prices)
    print(f"  Synthetic: {len(df)} bars (M15, GBM + mean-reversion)")
    return df


# ============================================================
# インジケータ計算
# ============================================================
def calc_indicators(df):
    """全インジケータを一括計算"""
    c = df["close"]

    # SMA
    df["sma200"] = c.rolling(200).mean()
    df["sma50"] = c.rolling(50).mean()
    df["sma20"] = c.rolling(20).mean()

    # SMA方向（5バー前比較）
    df["sma200_up"] = df["sma200"] > df["sma200"].shift(5)
    df["sma50_up"] = df["sma50"] > df["sma50"].shift(5)

    # RSI(10)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(10).mean()
    loss = (-delta.clip(upper=0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR(14)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift(1)).abs(),
        (df["low"] - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # ATR MA(100) — v1.00方式: ATR(14)の100期間移動平均
    df["atr14_ma100"] = df["atr14"].rolling(100).mean()

    # ATR(100) — v1.02方式: 100期間の平均真幅
    df["atr100"] = tr.rolling(100).mean()

    # BB(20, 2.5σ)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2.5 * bb_std
    df["bb_lower"] = bb_mid - 2.5 * bb_std

    # BB(10, 2.0σ) - Fast BB
    fbb_mid = c.rolling(10).mean()
    fbb_std = c.rolling(10).std()
    df["fbb_upper"] = fbb_mid + 2.0 * fbb_std
    df["fbb_lower"] = fbb_mid - 2.0 * fbb_std

    # 擬似スプレッド（早朝時間帯で拡大）
    if "time" in df.columns:
        hours = pd.to_datetime(df["time"]).dt.hour
        # JST想定: 6-9時にスプレッド拡大
        df["spread_pips"] = np.where(
            ((hours + 9) % 24 >= 6) & ((hours + 9) % 24 < 9),
            np.random.uniform(3.0, 8.0, len(df)),  # 早朝: 3-8 pips
            np.random.uniform(0.3, 1.5, len(df))   # 通常: 0.3-1.5 pips
        )
    else:
        df["spread_pips"] = 1.0

    return df.dropna().reset_index(drop=True)


# ============================================================
# シグナル判定
# ============================================================
def check_signals_v100(row, prev_row, atr_filter_mult=2.5):
    """v1.00: 元のロジック（UTC時間フィルター、スプレッドなし、ローソク足確認なし）"""
    # ATRフィルター: ATR(14) vs MA(ATR14, 100)
    if row["atr14"] >= row["atr14_ma100"] * atr_filter_mult:
        return 0, "atr_filter"

    close1 = row["close"]
    close2 = prev_row["close"]
    rsi = row["rsi"]
    sma200_up = row["sma200_up"]
    sma50_up = row["sma50_up"]

    # シグナル1: BB逆張り（ローソク足確認なし）
    if sma200_up and close2 <= prev_row["bb_lower"] and close1 > row["bb_lower"] and rsi < 42:
        return 1, "BUY_BB"
    if not sma200_up and close2 >= prev_row["bb_upper"] and close1 < row["bb_upper"] and rsi > 58:
        return -1, "SELL_BB"

    # シグナル2: 高速BB
    if sma200_up and sma50_up and close1 <= row["fbb_lower"] and rsi < 48:
        return 2, "BUY_FBB"
    if not sma200_up and not sma50_up and close1 >= row["fbb_upper"] and rsi > 52:
        return -2, "SELL_FBB"

    # シグナル3: 押し目
    sma_gap = abs(row["sma20"] - row["sma50"])
    if sma_gap >= row["atr14"] * 2:
        lower = min(row["sma20"], row["sma50"])
        upper = max(row["sma20"], row["sma50"])
        if sma200_up and lower <= close1 <= upper and 30 <= rsi <= 50:
            return 3, "BUY_PB"
        if not sma200_up and lower <= close1 <= upper and 50 <= rsi <= 70:
            return -3, "SELL_PB"

    return 0, "none"


def check_signals_v102(row, prev_row, atr_filter_mult=2.5, max_spread=3.0):
    """v1.02: 最適化版（スプレッドフィルター、ATR(100)、ローソク足確認）"""
    # スプレッドフィルター
    if row["spread_pips"] > max_spread:
        return 0, "spread_filter"

    # ATRフィルター: ATR(14) vs ATR(100)
    if row["atr14"] >= row["atr100"] * atr_filter_mult:
        return 0, "atr_filter"

    close1 = row["close"]
    close2 = prev_row["close"]
    open1 = row["open"]
    rsi = row["rsi"]
    sma200_up = row["sma200_up"]
    sma50_up = row["sma50_up"]

    # シグナル1: BB逆張り + ローソク足方向確認
    if sma200_up and close2 <= prev_row["bb_lower"] and close1 > row["bb_lower"] and close1 > open1 and rsi < 42:
        return 1, "BUY_BB"
    if not sma200_up and close2 >= prev_row["bb_upper"] and close1 < row["bb_upper"] and close1 < open1 and rsi > 58:
        return -1, "SELL_BB"

    # シグナル2: 高速BB
    if sma200_up and sma50_up and close1 <= row["fbb_lower"] and rsi < 48:
        return 2, "BUY_FBB"
    if not sma200_up and not sma50_up and close1 >= row["fbb_upper"] and rsi > 52:
        return -2, "SELL_FBB"

    # シグナル3: 押し目
    sma_gap = abs(row["sma20"] - row["sma50"])
    if sma_gap >= row["atr14"] * 2:
        lower = min(row["sma20"], row["sma50"])
        upper = max(row["sma20"], row["sma50"])
        if sma200_up and lower <= close1 <= upper and 30 <= rsi <= 50:
            return 3, "BUY_PB"
        if not sma200_up and lower <= close1 <= upper and 50 <= rsi <= 70:
            return -3, "SELL_PB"

    return 0, "none"


# ============================================================
# 簡易バックテスト（RR固定のTP/SL判定）
# ============================================================
def run_backtest(df, signal_func, label, deduct_spread=False, **kwargs):
    """シグナル発生後、ATR×SL倍率のSL / RR倍のTPで損益判定"""
    sl_mults = {1: 2.0, 2: 1.8, 3: 1.5}  # シグナルタイプ別SL倍率
    rr_ratio = 2.0
    max_hold = 20  # バー

    trades = []
    i = 1
    in_trade = False
    entry_bar = 0

    while i < len(df) - max_hold:
        if in_trade:
            i += 1
            continue

        row = df.iloc[i]
        prev = df.iloc[i - 1]
        sig, sig_name = signal_func(row, prev, **kwargs)

        if sig == 0:
            i += 1
            continue

        # エントリー
        direction = 1 if sig > 0 else -1
        entry_price = row["close"]
        atr = row["atr14"]
        sl_mult = sl_mults.get(abs(sig), 1.5)
        sl_dist = atr * sl_mult
        tp_dist = sl_dist * rr_ratio
        spread_cost = row["spread_pips"] if deduct_spread else 0

        sl_price = entry_price - direction * sl_dist
        tp_price = entry_price + direction * tp_dist

        # 決済判定（最大max_holdバー）
        result = "timeout"
        pnl_pips = 0
        exit_bar = i

        for j in range(1, max_hold + 1):
            if i + j >= len(df):
                break
            bar = df.iloc[i + j]

            if direction == 1:  # BUY
                if bar["low"] <= sl_price:
                    result = "SL"
                    pnl_pips = -sl_dist * 100 - spread_cost
                    exit_bar = i + j
                    break
                if bar["high"] >= tp_price:
                    result = "TP"
                    pnl_pips = tp_dist * 100 - spread_cost
                    exit_bar = i + j
                    break
            else:  # SELL
                if bar["high"] >= sl_price:
                    result = "SL"
                    pnl_pips = -sl_dist * 100 - spread_cost
                    exit_bar = i + j
                    break
                if bar["low"] <= tp_price:
                    result = "TP"
                    pnl_pips = tp_dist * 100 - spread_cost
                    exit_bar = i + j
                    break

        if result == "timeout":
            exit_price = df.iloc[min(i + max_hold, len(df) - 1)]["close"]
            pnl_pips = direction * (exit_price - entry_price) * 100 - spread_cost

        trades.append({
            "entry_bar": i,
            "signal": sig_name,
            "direction": "BUY" if direction == 1 else "SELL",
            "entry": entry_price,
            "sl_dist": sl_dist,
            "result": result,
            "pnl_pips": pnl_pips,
            "spread": row["spread_pips"],
        })

        i = exit_bar + 1
        in_trade = False

    return trades


# ============================================================
# レポート出力
# ============================================================
def print_report(trades, label):
    if not trades:
        print(f"\n{'='*50}")
        print(f"  {label}: シグナルなし")
        return {}

    df_t = pd.DataFrame(trades)
    total = len(df_t)
    wins = len(df_t[df_t["pnl_pips"] > 0])
    losses = len(df_t[df_t["pnl_pips"] < 0])
    win_rate = wins / total * 100 if total > 0 else 0
    total_pnl = df_t["pnl_pips"].sum()
    avg_win = df_t[df_t["pnl_pips"] > 0]["pnl_pips"].mean() if wins > 0 else 0
    avg_loss = df_t[df_t["pnl_pips"] <= 0]["pnl_pips"].mean() if losses > 0 else 0
    profit_factor = abs(df_t[df_t["pnl_pips"] > 0]["pnl_pips"].sum() / df_t[df_t["pnl_pips"] <= 0]["pnl_pips"].sum()) if losses > 0 and df_t[df_t["pnl_pips"] <= 0]["pnl_pips"].sum() != 0 else float("inf")

    # 最大ドローダウン
    cumsum = df_t["pnl_pips"].cumsum()
    peak = cumsum.cummax()
    dd = (cumsum - peak).min()

    # シグナル内訳
    sig_counts = df_t["signal"].value_counts().to_dict()

    # 結果内訳
    result_counts = df_t["result"].value_counts().to_dict()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  トレード数:    {total}")
    print(f"  勝率:          {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  合計損益:      {total_pnl:+.1f} pips")
    print(f"  平均利益:      {avg_win:+.1f} pips")
    print(f"  平均損失:      {avg_loss:+.1f} pips")
    print(f"  PF:            {profit_factor:.2f}")
    print(f"  最大DD:        {dd:.1f} pips")
    print(f"  決済内訳:      {result_counts}")
    print(f"  シグナル内訳:  {sig_counts}")

    # スプレッドフィルターの効果（v1.02のみ）
    if "spread_filter" in [t.get("filtered_by", "") for t in trades]:
        spread_filtered = sum(1 for t in trades if t.get("filtered_by") == "spread_filter")
        print(f"  スプレッド除外: {spread_filtered} 回")

    return {
        "total": total, "win_rate": win_rate, "total_pnl": total_pnl,
        "pf": profit_factor, "max_dd": dd, "signals": sig_counts,
    }


def count_filtered(df, signal_func, **kwargs):
    """フィルター別の除外数をカウント"""
    counts = {}
    for i in range(1, len(df)):
        _, reason = signal_func(df.iloc[i], df.iloc[i-1], **kwargs)
        if reason not in ("none",) and reason.endswith("_filter"):
            counts[reason] = counts.get(reason, 0) + 1
    return counts


# ============================================================
# メイン
# ============================================================
def main():
    print("=" * 60)
    print("BB_Reversal_Martin バックテスト: v1.00 vs v1.02")
    print("=" * 60)

    # データ取得
    print("\n[1] データ取得中...")
    df = fetch_forex_data("USDJPY=X", days=180)
    if df is None or len(df) < 300:
        print("  → Yahoo失敗、合成データを使用")
        df = generate_synthetic_data(8000)

    # インジケータ計算
    print("\n[2] インジケータ計算中...")
    df = calc_indicators(df)
    print(f"  有効バー数: {len(df)}")

    # ATRフィルター差異の確認
    print("\n[3] ATRフィルター比較: MA(ATR14,100) vs ATR(100)")
    atr_corr = df["atr14_ma100"].corr(df["atr100"])
    atr_diff = ((df["atr14_ma100"] - df["atr100"]) / df["atr14_ma100"] * 100).describe()
    print(f"  相関係数: {atr_corr:.4f}")
    print(f"  差異(%):  mean={atr_diff['mean']:.2f}%, std={atr_diff['std']:.2f}%, max={atr_diff['max']:.2f}%")

    # フィルター統計
    print("\n[4] フィルター除外統計...")
    f_v100 = count_filtered(df, check_signals_v100)
    f_v102 = count_filtered(df, check_signals_v102)
    print(f"  v1.00: {f_v100}")
    print(f"  v1.02: {f_v102}")

    # ============================================================
    # [5] スプレッドコスト込みバックテスト（実態に近い）
    # ============================================================
    print("\n[5] バックテスト実行（スプレッドコスト込み）...")
    trades_v100 = run_backtest(df, check_signals_v100, "v1.00", deduct_spread=True)
    trades_v102 = run_backtest(df, check_signals_v102, "v1.02", deduct_spread=True)

    r1 = print_report(trades_v100, "v1.00 (Original) + スプレッドコスト")
    r2 = print_report(trades_v102, "v1.02 (Optimized) + スプレッドコスト")

    # ============================================================
    # [6] 各変更の個別効果テスト
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  各変更の個別効果テスト（スプレッドコスト込み）")
    print(f"{'='*60}")

    # A) スプレッドフィルターのみ追加（ローソク確認なし、ATR=v1.00式）
    def check_signals_spread_only(row, prev_row, atr_filter_mult=2.5, max_spread=3.0):
        if row["spread_pips"] > max_spread:
            return 0, "spread_filter"
        if row["atr14"] >= row["atr14_ma100"] * atr_filter_mult:
            return 0, "atr_filter"
        close1, close2, rsi = row["close"], prev_row["close"], row["rsi"]
        sma200_up, sma50_up = row["sma200_up"], row["sma50_up"]
        if sma200_up and close2 <= prev_row["bb_lower"] and close1 > row["bb_lower"] and rsi < 42:
            return 1, "BUY_BB"
        if not sma200_up and close2 >= prev_row["bb_upper"] and close1 < row["bb_upper"] and rsi > 58:
            return -1, "SELL_BB"
        if sma200_up and sma50_up and close1 <= row["fbb_lower"] and rsi < 48:
            return 2, "BUY_FBB"
        if not sma200_up and not sma50_up and close1 >= row["fbb_upper"] and rsi > 52:
            return -2, "SELL_FBB"
        sma_gap = abs(row["sma20"] - row["sma50"])
        if sma_gap >= row["atr14"] * 2:
            lower, upper = min(row["sma20"], row["sma50"]), max(row["sma20"], row["sma50"])
            if sma200_up and lower <= close1 <= upper and 30 <= rsi <= 50:
                return 3, "BUY_PB"
            if not sma200_up and lower <= close1 <= upper and 50 <= rsi <= 70:
                return -3, "SELL_PB"
        return 0, "none"

    # B) ローソク足確認のみ追加（スプレッドフィルターなし、ATR=v1.00式）
    def check_signals_candle_only(row, prev_row, atr_filter_mult=2.5):
        if row["atr14"] >= row["atr14_ma100"] * atr_filter_mult:
            return 0, "atr_filter"
        close1, close2, open1, rsi = row["close"], prev_row["close"], row["open"], row["rsi"]
        sma200_up, sma50_up = row["sma200_up"], row["sma50_up"]
        if sma200_up and close2 <= prev_row["bb_lower"] and close1 > row["bb_lower"] and close1 > open1 and rsi < 42:
            return 1, "BUY_BB"
        if not sma200_up and close2 >= prev_row["bb_upper"] and close1 < row["bb_upper"] and close1 < open1 and rsi > 58:
            return -1, "SELL_BB"
        if sma200_up and sma50_up and close1 <= row["fbb_lower"] and rsi < 48:
            return 2, "BUY_FBB"
        if not sma200_up and not sma50_up and close1 >= row["fbb_upper"] and rsi > 52:
            return -2, "SELL_FBB"
        sma_gap = abs(row["sma20"] - row["sma50"])
        if sma_gap >= row["atr14"] * 2:
            lower, upper = min(row["sma20"], row["sma50"]), max(row["sma20"], row["sma50"])
            if sma200_up and lower <= close1 <= upper and 30 <= rsi <= 50:
                return 3, "BUY_PB"
            if not sma200_up and lower <= close1 <= upper and 50 <= rsi <= 70:
                return -3, "SELL_PB"
        return 0, "none"

    trades_a = run_backtest(df, check_signals_spread_only, "A", deduct_spread=True)
    trades_b = run_backtest(df, check_signals_candle_only, "B", deduct_spread=True)

    ra = print_report(trades_a, "A) スプレッドフィルターのみ")
    rb = print_report(trades_b, "B) ローソク足確認のみ")

    # ============================================================
    # [7] 比較サマリー
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  最終比較サマリー（スプレッドコスト込み）")
    print(f"{'='*60}")
    results = {"v1.00": r1, "A)スプレッド": ra, "B)ローソク": rb, "v1.02(全部)": r2}
    header = f"  {'':18s} {'トレード':>7s} {'勝率':>7s} {'損益(pips)':>11s} {'PF':>6s} {'最大DD':>9s}"
    print(header)
    print("  " + "-" * 58)
    for name, r in results.items():
        if r:
            print(f"  {name:18s} {r['total']:>7d} {r['win_rate']:>6.1f}% {r['total_pnl']:>+10.1f} {r['pf']:>6.2f} {r['max_dd']:>+8.1f}")
        else:
            print(f"  {name:18s}   シグナルなし")

    print(f"\n※合成M15データによるシミュレーション。実MT4データでの検証推奨。")
    print(f"  スプレッドは早朝(JST6-9時)3-8pips、通常0.3-1.5pipsで擬似生成。")


if __name__ == "__main__":
    main()
