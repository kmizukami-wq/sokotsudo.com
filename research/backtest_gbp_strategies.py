#!/usr/bin/env python3
"""
GBP通貨ペア 月利10%戦略バックテスト
====================================
対象: GBP/JPY, GBP/USD
データ: yfinance 15分足 60日
戦略:
  1. ロンドンブレイクアウト（アジアレンジ）
  2. BB逆張り + 軽マーチン
  3. トレンドフォロー（EMA + ATRトレイリング）
"""

import numpy as np
import pandas as pd
import yfinance as yf
from collections import defaultdict
from datetime import datetime

# ============================================================
# 共通設定
# ============================================================
INITIAL_CAPITAL = 1_000_000  # 初期資金（円）

PAIRS = {
    'GBPJPY=X': {
        'name': 'GBP/JPY',
        'pip': 0.01,
        'spread_pips': 1.0,
        'quote_to_jpy': 1.0,
    },
    'GBPUSD=X': {
        'name': 'GBP/USD',
        'pip': 0.0001,
        'spread_pips': 0.7,
        'quote_to_jpy': 150.0,  # 後で実レートに更新
    },
}


def fetch_data():
    """yfinanceから15分足60日データ取得"""
    print(">>> データ取得中...")
    data = {}
    # GBPUSDのquote_to_jpyを更新
    try:
        uj = yf.download('USDJPY=X', period='1d', interval='1d', progress=False)
        if isinstance(uj.columns, pd.MultiIndex):
            uj.columns = uj.columns.get_level_values(0)
        PAIRS['GBPUSD=X']['quote_to_jpy'] = float(uj['Close'].iloc[-1])
        print(f"  USD/JPY = {PAIRS['GBPUSD=X']['quote_to_jpy']:.2f}")
    except Exception:
        pass

    for ticker, cfg in PAIRS.items():
        try:
            df = yf.download(ticker, period='60d', interval='15m', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            data[ticker] = df
            days = (df.index[-1] - df.index[0]).days
            print(f"  {cfg['name']}: {len(df)}本 ({days}日)")
        except Exception as e:
            print(f"  {cfg['name']}: Error - {e}")
    return data


def print_strategy_results(name, trades_df, capital, initial=INITIAL_CAPITAL):
    """統一フォーマットで結果出力"""
    if len(trades_df) == 0:
        print(f"\n  {name}: トレードなし")
        return None

    net = capital - initial
    ret = (capital / initial - 1) * 100
    days = (trades_df['date'].max() - trades_df['date'].min()).days
    months = max(days / 30.0, 0.5)
    monthly = ((capital / initial) ** (1 / months) - 1) * 100 if months > 0 else 0

    wins = (trades_df['pnl'] > 0).sum()
    losses = (trades_df['pnl'] <= 0).sum()
    wr = wins / len(trades_df) * 100

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # MaxDD
    eq = trades_df['equity'].values
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = dd.max() * 100

    # 月別リターン
    trades_df = trades_df.copy()
    trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
    monthly_rets = {}
    for m, grp in trades_df.groupby('month'):
        eq_start = grp['equity'].iloc[0] - grp['pnl'].iloc[0]
        eq_end = grp['equity'].iloc[-1]
        monthly_rets[str(m)] = (eq_end / eq_start - 1) * 100

    months_10pct = sum(1 for r in monthly_rets.values() if r >= 10)

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  期間          : {days}日 ({months:.1f}ヶ月)")
    print(f"  取引回数      : {len(trades_df)} ({len(trades_df)/months:.1f}回/月)")
    print(f"  勝率          : {wr:.1f}% ({wins}勝 {losses}敗)")
    print(f"  PF            : {pf:.2f}")
    print(f"  純損益        : ¥{net:>+,.0f} ({ret:>+.1f}%)")
    print(f"  月利          : {monthly:.2f}%")
    print(f"  MaxDD         : {max_dd:.1f}%")
    print(f"  月利10%達成月 : {months_10pct}/{len(monthly_rets)}ヶ月")

    if monthly_rets:
        print(f"\n  --- 月別リターン ---")
        for m, r in sorted(monthly_rets.items()):
            marker = " ★" if r >= 10 else (" ⚠️" if r < 0 else "")
            print(f"    {m}: {r:>+8.2f}%{marker}")

    return {
        'name': name, 'trades': len(trades_df), 'win_rate': wr,
        'pf': pf, 'monthly': monthly, 'max_dd': max_dd, 'net': net,
        'ret': ret, 'months_10pct': months_10pct, 'total_months': len(monthly_rets),
    }


# ============================================================
# 戦略1: ロンドンブレイクアウト（アジアレンジ）
# ============================================================
def london_breakout(df, spread, pip, q2j,
                    tp_mult=1.5, risk_pct=0.02,
                    min_range_pips=20, max_range_pips=120):
    """
    アジアレンジ(00:00-07:00 UTC)のHigh/Lowを計測
    ロンドン開始(07:00-16:00 UTC)でブレイクアウト
    トレードは複数バーにまたがって追跡する
    """
    capital = float(INITIAL_CAPITAL)
    trades = []

    for date_val, day_data in df.groupby(df.index.date):
        # アジアセッション (00:00-07:00 UTC)
        asian = day_data.between_time('00:00', '06:59')
        if len(asian) < 4:
            continue
        asian_high = float(asian['High'].max())
        asian_low = float(asian['Low'].min())
        asian_range = asian_high - asian_low
        range_pips = asian_range / pip

        if range_pips < min_range_pips or range_pips > max_range_pips:
            continue

        # ロンドンセッション (07:00-16:00 UTC)
        london = day_data.between_time('07:00', '15:59')
        if len(london) == 0:
            continue

        tp = asian_range * tp_mult
        sl = asian_range
        in_trade = False
        entry = 0.0
        tp_price = 0.0
        sl_price = 0.0
        direction = 0  # 1=long, -1=short
        pnl_r = None

        for idx, bar in london.iterrows():
            high = float(bar['High'])
            low = float(bar['Low'])
            close = float(bar['Close'])

            if in_trade:
                # 既存ポジションのTP/SL判定
                if direction == 1:
                    if low <= sl_price:
                        pnl_r = -1.0
                    elif high >= tp_price:
                        pnl_r = tp_mult
                else:
                    if high >= sl_price:
                        pnl_r = -1.0
                    elif low <= tp_price:
                        pnl_r = tp_mult

                if pnl_r is not None:
                    risk_amount = capital * risk_pct
                    pnl_jpy = pnl_r * risk_amount
                    capital += pnl_jpy
                    trades.append({
                        'date': idx, 'pnl': pnl_jpy, 'equity': capital,
                        'r_mult': pnl_r, 'range_pips': range_pips,
                    })
                    in_trade = False
                    break
                continue

            # 新規ブレイクアウト検出
            if high > asian_high + spread:
                entry = asian_high + spread
                tp_price = entry + tp
                sl_price = entry - sl
                direction = 1
                in_trade = True

                # 同一バー内でTP/SL判定（SL優先: 安値側を先にチェック）
                if low <= sl_price:
                    pnl_r = -1.0
                elif high >= tp_price:
                    pnl_r = tp_mult

                if pnl_r is not None:
                    risk_amount = capital * risk_pct
                    pnl_jpy = pnl_r * risk_amount
                    capital += pnl_jpy
                    trades.append({
                        'date': idx, 'pnl': pnl_jpy, 'equity': capital,
                        'r_mult': pnl_r, 'range_pips': range_pips,
                    })
                    in_trade = False
                    break

            elif low < asian_low - spread:
                entry = asian_low - spread
                tp_price = entry - tp
                sl_price = entry + sl
                direction = -1
                in_trade = True

                if high >= sl_price:
                    pnl_r = -1.0
                elif low <= tp_price:
                    pnl_r = tp_mult

                if pnl_r is not None:
                    risk_amount = capital * risk_pct
                    pnl_jpy = pnl_r * risk_amount
                    capital += pnl_jpy
                    trades.append({
                        'date': idx, 'pnl': pnl_jpy, 'equity': capital,
                        'r_mult': pnl_r, 'range_pips': range_pips,
                    })
                    in_trade = False
                    break

        # セッション終了時にまだポジションが残っていたら時価決済
        if in_trade and pnl_r is None:
            last_close = float(london.iloc[-1]['Close'])
            if direction == 1:
                pnl_r = (last_close - entry) / sl
            else:
                pnl_r = (entry - last_close) / sl
            risk_amount = capital * risk_pct
            pnl_jpy = pnl_r * risk_amount
            capital += pnl_jpy
            trades.append({
                'date': london.index[-1], 'pnl': pnl_jpy, 'equity': capital,
                'r_mult': pnl_r, 'range_pips': range_pips,
            })

    return pd.DataFrame(trades), capital


# ============================================================
# 戦略2: BB逆張り + 軽マーチン
# ============================================================
def calc_indicators_bb(df):
    """BB + RSI + ATR 指標計算"""
    c = df['Close'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)

    df = df.copy()
    df['SMA200'] = pd.Series(c).rolling(200).mean().values
    df['SMA50'] = pd.Series(c).rolling(50).mean().values
    df['SMA20'] = pd.Series(c).rolling(20).mean().values

    sma20 = pd.Series(c).rolling(20).mean()
    std20 = pd.Series(c).rolling(20).std(ddof=0)
    df['BB_upper'] = (sma20 + 2.5 * std20).values
    df['BB_lower'] = (sma20 - 2.5 * std20).values

    sma10 = pd.Series(c).rolling(10).mean()
    std10 = pd.Series(c).rolling(10).std(ddof=0)
    df['FBB_upper'] = (sma10 + 2.0 * std10).values
    df['FBB_lower'] = (sma10 - 2.0 * std10).values

    # RSI (Wilder EWM)
    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1.0/10, min_periods=10, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/10, min_periods=10, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = (100 - 100 / (1 + rs)).values

    # ATR
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    df['ATR'] = pd.Series(tr).rolling(14).mean().values
    df['ATR_MA100'] = pd.Series(df['ATR']).rolling(100).mean().values
    df['SMA200_up'] = df['SMA200'] > pd.Series(df['SMA200']).shift(5).values
    df['SMA50_up'] = df['SMA50'] > pd.Series(df['SMA50']).shift(5).values

    return df


def bb_reversal_martin(df, spread, pip, q2j,
                       risk_pct=0.008, rr_ratio=2.0):
    """BB逆張り + Fast BB + Pullback, 3段マーチン"""
    SL_ATR_MULT = {'BB': 2.5, 'FBB': 2.2, 'PB': 1.8}
    MARTIN = [1.0, 1.5, 2.0]
    MAX_HOLD = 20
    RSI_BB_BUY = 42; RSI_BB_SELL = 58
    RSI_FBB_BUY = 48; RSI_FBB_SELL = 52
    ATR_FILTER = 2.5

    df = calc_indicators_bb(df)
    capital = float(INITIAL_CAPITAL)
    trades = []
    position = None
    martin_stage = 0
    consec_losses = 0

    rows = df.to_dict('index')
    indices = list(rows.keys())

    for i in range(1, len(indices)):
        idx = indices[i]
        prev_idx = indices[i - 1]
        row = rows[idx]
        prev_row = rows[prev_idx]
        close = float(row['Close'])
        high = float(row['High'])
        low = float(row['Low'])

        # 取引時間フィルター (07:00-21:00 UTC)
        hour = idx.hour
        if hour < 7 or hour >= 21:
            continue

        # ポジション管理
        if position is not None:
            position['bars'] += 1
            d = position['dir']
            sl = position['sl']
            tp = position['tp']
            entry = position['entry']
            lots = position['lots']
            sld = position['sl_dist']
            closed = False
            pnl = 0.0

            if d == 'BUY':
                # BE
                if not position['be'] and (high - entry) >= sld * 1.0:
                    position['be'] = True
                    position['sl'] = entry + spread
                    sl = position['sl']
                # Trailing
                if position['be']:
                    ts = high - float(row['ATR']) * 0.5
                    if ts > sl:
                        position['sl'] = ts
                        sl = ts
                if low <= sl:
                    pnl = (sl - entry - spread) * lots * q2j
                    closed = True
                elif high >= tp:
                    pnl = (tp - entry - spread) * lots * q2j
                    closed = True
            else:
                if not position['be'] and (entry - low) >= sld * 1.0:
                    position['be'] = True
                    position['sl'] = entry - spread
                    sl = position['sl']
                if position['be']:
                    ts = low + float(row['ATR']) * 0.5
                    if ts < sl:
                        position['sl'] = ts
                        sl = ts
                if high >= sl:
                    pnl = (entry - sl - spread) * lots * q2j
                    closed = True
                elif low <= tp:
                    pnl = (entry - tp - spread) * lots * q2j
                    closed = True

            if not closed and position['bars'] >= MAX_HOLD:
                if d == 'BUY':
                    pnl = (close - entry - spread) * lots * q2j
                else:
                    pnl = (entry - close - spread) * lots * q2j
                closed = True

            if closed:
                capital += pnl
                trades.append({
                    'date': idx, 'pnl': pnl, 'equity': capital,
                    'signal': position['sig'],
                })
                if pnl < 0:
                    consec_losses += 1
                    martin_stage = min(consec_losses, 2)
                else:
                    consec_losses = 0
                    martin_stage = 0
                position = None
            if position is not None:
                continue

        # シグナル検出
        if any(pd.isna(row.get(k, np.nan)) for k in ['SMA200', 'BB_upper', 'BB_lower', 'RSI', 'ATR', 'ATR_MA100']):
            continue
        if row['ATR'] >= row['ATR_MA100'] * ATR_FILTER:
            continue

        signal = None
        sma200_up = row['SMA200_up']

        # BB reversal
        prev_close = float(prev_row['Close'])
        if sma200_up and prev_close <= prev_row['BB_lower'] and close > row['BB_lower'] and row['RSI'] < RSI_BB_BUY:
            signal = ('BUY', 'BB')
        elif not sma200_up and prev_close >= prev_row['BB_upper'] and close < row['BB_upper'] and row['RSI'] > RSI_BB_SELL:
            signal = ('SELL', 'BB')

        # Fast BB
        if signal is None:
            sma50_up = row['SMA50_up']
            if sma200_up and sma50_up and close <= row['FBB_lower'] and row['RSI'] < RSI_FBB_BUY:
                signal = ('BUY', 'FBB')
            elif not sma200_up and not sma50_up and close >= row['FBB_upper'] and row['RSI'] > RSI_FBB_SELL:
                signal = ('SELL', 'FBB')

        if signal is None:
            continue

        direction, sig_type = signal
        atr = float(row['ATR'])
        sl_mult = SL_ATR_MULT[sig_type]
        sl_dist = atr * sl_mult
        tp_dist = sl_dist * rr_ratio

        if direction == 'BUY':
            ep = close + spread / 2
            slp = ep - sl_dist
            tpp = ep + tp_dist
        else:
            ep = close - spread / 2
            slp = ep + sl_dist
            tpp = ep - tp_dist

        ra = capital * risk_pct * MARTIN[martin_stage]
        if sl_dist <= 0:
            continue
        lots = ra / (sl_dist * q2j)

        position = {
            'dir': direction, 'entry': ep, 'sl': slp, 'tp': tpp,
            'sl_dist': sl_dist, 'lots': lots, 'sig': sig_type,
            'bars': 0, 'be': False, 'partial': False,
        }

    return pd.DataFrame(trades), capital


# ============================================================
# 戦略3: トレンドフォロー（EMA + ATRトレイリング）
# ============================================================
def trend_follow_ema(df, spread, pip, q2j,
                     fast_ema=20, slow_ema=50,
                     atr_mult=2.0, risk_pct=0.015):
    """EMAクロス + ATRトレイリングストップ (ロンドン-NY時間限定)"""
    df = df.copy()
    c = df['Close'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)

    df['EMA_fast'] = c.ewm(span=fast_ema).mean()
    df['EMA_slow'] = c.ewm(span=slow_ema).mean()

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    df['signal'] = 0
    df.loc[df['EMA_fast'] > df['EMA_slow'], 'signal'] = 1
    df.loc[df['EMA_fast'] < df['EMA_slow'], 'signal'] = -1

    capital = float(INITIAL_CAPITAL)
    trades = []
    pos = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    trailing_stop = 0.0
    sl_dist_entry = 0.0

    for i in range(slow_ema + 14, len(df)):
        row = df.iloc[i]
        sig = int(row['signal'])
        atr = float(row['ATR'])
        close_val = float(row['Close'])
        high_val = float(row['High'])
        low_val = float(row['Low'])
        hour = df.index[i].hour

        # ロンドン-NY時間のみ (07:00-21:00 UTC)
        if hour < 7 or hour >= 21:
            continue

        if pd.isna(atr) or atr <= 0:
            continue

        # 新規エントリー
        if pos == 0 and sig != 0:
            pos = sig
            entry_price = close_val
            sl_dist_entry = atr * atr_mult
            if pos == 1:
                trailing_stop = entry_price - sl_dist_entry
            else:
                trailing_stop = entry_price + sl_dist_entry

        elif pos == 1:
            new_stop = close_val - atr * atr_mult
            trailing_stop = max(trailing_stop, new_stop)

            if low_val <= trailing_stop or sig == -1:
                exit_price = max(trailing_stop, low_val)
                pnl_pips = (exit_price - entry_price - spread) / pip
                r_mult = (exit_price - entry_price - spread) / sl_dist_entry
                pnl_jpy = r_mult * capital * risk_pct
                capital += pnl_jpy
                trades.append({
                    'date': df.index[i], 'pnl': pnl_jpy, 'equity': capital,
                    'r_mult': r_mult, 'pips': pnl_pips,
                })
                pos = 0

        elif pos == -1:
            new_stop = close_val + atr * atr_mult
            trailing_stop = min(trailing_stop, new_stop)

            if high_val >= trailing_stop or sig == 1:
                exit_price = min(trailing_stop, high_val)
                pnl_pips = (entry_price - exit_price - spread) / pip
                r_mult = (entry_price - exit_price - spread) / sl_dist_entry
                pnl_jpy = r_mult * capital * risk_pct
                capital += pnl_jpy
                trades.append({
                    'date': df.index[i], 'pnl': pnl_jpy, 'equity': capital,
                    'r_mult': r_mult, 'pips': pnl_pips,
                })
                pos = 0

    return pd.DataFrame(trades), capital


# ============================================================
# 戦略3b: 高勝率トレンドフォロー（フィルター強化版）
# ============================================================
def calc_adx(high, low, close, period=14):
    """ADX (Average Directional Index) 計算"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def trend_follow_filtered(df, spread, pip, q2j,
                          fast_ema=50, slow_ema=200,
                          atr_mult=3.5, risk_pct=0.02,
                          adx_thresh=20, use_h1_filter=True,
                          candle_filter=True):
    """
    高勝率トレンドフォロー - フィルター強化版
    追加フィルター:
      1. ADX > adx_thresh: トレンドが出ている時のみエントリー
      2. H1 EMA方向確認: 上位足トレンドと一致する方向のみ
      3. ローソク足確認: 強い足（実体>レンジ50%）でのみエントリー
    """
    df = df.copy()
    c = df['Close'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)
    o = df['Open'].astype(float)

    df['EMA_fast'] = c.ewm(span=fast_ema).mean()
    df['EMA_slow'] = c.ewm(span=slow_ema).mean()

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # ADX
    df['ADX'], df['PLUS_DI'], df['MINUS_DI'] = calc_adx(h, l, c, 14)

    # M15シグナル
    df['signal'] = 0
    df.loc[df['EMA_fast'] > df['EMA_slow'], 'signal'] = 1
    df.loc[df['EMA_fast'] < df['EMA_slow'], 'signal'] = -1

    # H1 EMA方向（M15を4本でリサンプル）
    if use_h1_filter:
        h1 = df['Close'].resample('1h').last().dropna()
        h1_ema20 = h1.ewm(span=20).mean()
        h1_ema50 = h1.ewm(span=50).mean()
        h1_trend = pd.Series(0, index=h1.index)
        h1_trend[h1_ema20 > h1_ema50] = 1
        h1_trend[h1_ema20 < h1_ema50] = -1
        # M15に展開（forward fill）
        df['h1_trend'] = h1_trend.reindex(df.index, method='ffill').fillna(0).astype(int)
    else:
        df['h1_trend'] = df['signal']

    # ローソク足の強さ（実体/全体の比率）
    body = (c - o).abs()
    candle_range = h - l
    df['body_ratio'] = (body / candle_range.replace(0, np.nan)).fillna(0)

    capital = float(INITIAL_CAPITAL)
    trades = []
    pos = 0
    entry_price = 0.0
    trailing_stop = 0.0
    sl_dist_entry = 0.0
    skipped = 0

    start_idx = max(slow_ema + 14, 60)  # ADXの安定化にも余裕
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        sig = int(row['signal'])
        atr = float(row['ATR'])
        adx = float(row['ADX'])
        close_val = float(row['Close'])
        open_val = float(row['Open'])
        high_val = float(row['High'])
        low_val = float(row['Low'])
        hour = df.index[i].hour
        h1_dir = int(row['h1_trend'])

        # ロンドン-NY時間のみ
        if hour < 7 or hour >= 21:
            continue
        if pd.isna(atr) or atr <= 0 or pd.isna(adx):
            continue

        # === ポジション管理（フィルターの影響なし: 既存ポジションは通常通り管理） ===
        if pos == 1:
            new_stop = close_val - atr * atr_mult
            trailing_stop = max(trailing_stop, new_stop)
            if low_val <= trailing_stop or sig == -1:
                exit_price = max(trailing_stop, low_val)
                r_mult = (exit_price - entry_price - spread) / sl_dist_entry
                pnl_jpy = r_mult * capital * risk_pct
                capital += pnl_jpy
                trades.append({
                    'date': df.index[i], 'pnl': pnl_jpy, 'equity': capital,
                    'r_mult': r_mult, 'pips': (exit_price - entry_price - spread) / pip,
                })
                pos = 0
            continue

        if pos == -1:
            new_stop = close_val + atr * atr_mult
            trailing_stop = min(trailing_stop, new_stop)
            if high_val >= trailing_stop or sig == 1:
                exit_price = min(trailing_stop, high_val)
                r_mult = (entry_price - exit_price - spread) / sl_dist_entry
                pnl_jpy = r_mult * capital * risk_pct
                capital += pnl_jpy
                trades.append({
                    'date': df.index[i], 'pnl': pnl_jpy, 'equity': capital,
                    'r_mult': r_mult, 'pips': (entry_price - exit_price - spread) / pip,
                })
                pos = 0
            continue

        # === 新規エントリー: フィルター適用 ===
        if sig == 0:
            continue

        # フィルター1: ADX — トレンドが出ていない時はスキップ
        if adx < adx_thresh:
            skipped += 1
            continue

        # フィルター2: H1トレンド方向と一致するか
        if use_h1_filter and h1_dir != sig:
            skipped += 1
            continue

        # フィルター3: ローソク足確認 — 弱い足ではエントリーしない
        if candle_filter:
            br = float(row['body_ratio'])
            if br < 0.5:
                skipped += 1
                continue
            # 方向確認: 買いなら陽線、売りなら陰線
            if sig == 1 and close_val <= open_val:
                skipped += 1
                continue
            if sig == -1 and close_val >= open_val:
                skipped += 1
                continue

        # 全フィルター通過 → エントリー
        pos = sig
        entry_price = close_val
        sl_dist_entry = atr * atr_mult
        if pos == 1:
            trailing_stop = entry_price - sl_dist_entry
        else:
            trailing_stop = entry_price + sl_dist_entry

    return pd.DataFrame(trades), capital, skipped


# ============================================================
# 戦略4: 分割エントリー（スケールイン）トレンドフォロー
# ============================================================
def trend_follow_scalein(df, spread, pip, q2j,
                         fast_ema=50, slow_ema=200,
                         risk_pct=0.02, n_splits=10,
                         grid_atr_mult=0.3, tp_atr_mult=1.0,
                         max_sl_atr_mult=4.0):
    """
    分割エントリー（ナンピン型）トレンドフォロー

    EMAクロスでシグナル発生後:
      - 1/n_splits ずつ分割エントリー
      - 価格が逆行するたびにATR×grid_atr_mult間隔で追加
      - TP = 平均建値 + ATR × tp_atr_mult
      - SL = 最悪エントリー - ATR × (余裕分)

    メリット: 平均建値が改善 → 小さな戻しで利確 → 勝率大幅UP
    リスク: 全ポジションが損切りになると大きな損失
    """
    df = df.copy()
    c = df['Close'].astype(float)
    h = df['High'].astype(float)
    l = df['Low'].astype(float)

    df['EMA_fast'] = c.ewm(span=fast_ema).mean()
    df['EMA_slow'] = c.ewm(span=slow_ema).mean()

    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    df['signal'] = 0
    df.loc[df['EMA_fast'] > df['EMA_slow'], 'signal'] = 1
    df.loc[df['EMA_fast'] < df['EMA_slow'], 'signal'] = -1

    capital = float(INITIAL_CAPITAL)
    trades = []

    # ポジション状態
    pos_dir = 0           # 1=long, -1=short, 0=flat
    entries = []          # [(price, lots), ...]
    next_grid_price = 0   # 次の追加エントリー価格
    grid_step = 0         # グリッド間隔
    total_sl = 0          # 全体SL価格
    signal_atr = 0        # シグナル時のATR

    start_idx = max(slow_ema + 14, 60)
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        sig = int(row['signal'])
        atr = float(row['ATR'])
        close_val = float(row['Close'])
        high_val = float(row['High'])
        low_val = float(row['Low'])
        hour = df.index[i].hour

        if hour < 7 or hour >= 21:
            continue
        if pd.isna(atr) or atr <= 0:
            continue

        # === ポジション管理 ===
        if pos_dir != 0 and len(entries) > 0:
            avg_entry = sum(e[0] * e[1] for e in entries) / sum(e[1] for e in entries)
            total_lots = sum(e[1] for e in entries)
            tp_price = avg_entry + pos_dir * signal_atr * tp_atr_mult

            # TP判定
            hit_tp = False
            hit_sl = False
            if pos_dir == 1:
                if high_val >= tp_price:
                    hit_tp = True
                if low_val <= total_sl:
                    hit_sl = True
            else:
                if low_val <= tp_price:
                    hit_tp = True
                if high_val >= total_sl:
                    hit_sl = True

            # シグナル反転 → 即決済
            if sig == -pos_dir:
                exit_price = close_val
                if pos_dir == 1:
                    pnl_per_lot = (exit_price - avg_entry - spread) * q2j
                else:
                    pnl_per_lot = (avg_entry - exit_price - spread) * q2j
                pnl_jpy = pnl_per_lot * total_lots
                capital += pnl_jpy
                trades.append({
                    'date': df.index[i], 'pnl': pnl_jpy, 'equity': capital,
                    'r_mult': pnl_jpy / (capital * risk_pct) if capital > 0 else 0,
                    'pips': (exit_price - avg_entry) * pos_dir / pip,
                    'n_entries': len(entries), 'exit_type': 'REVERSAL',
                })
                pos_dir = 0
                entries = []
                continue

            if hit_tp:
                if pos_dir == 1:
                    pnl_per_lot = (tp_price - avg_entry - spread) * q2j
                else:
                    pnl_per_lot = (avg_entry - tp_price - spread) * q2j
                pnl_jpy = pnl_per_lot * total_lots
                capital += pnl_jpy
                trades.append({
                    'date': df.index[i], 'pnl': pnl_jpy, 'equity': capital,
                    'r_mult': pnl_jpy / (capital * risk_pct) if capital > 0 else 0,
                    'pips': signal_atr * tp_atr_mult / pip,
                    'n_entries': len(entries), 'exit_type': 'TP',
                })
                pos_dir = 0
                entries = []
                continue

            if hit_sl:
                if pos_dir == 1:
                    pnl_per_lot = (total_sl - avg_entry - spread) * q2j
                else:
                    pnl_per_lot = (avg_entry - total_sl - spread) * q2j
                pnl_jpy = pnl_per_lot * total_lots
                capital += pnl_jpy
                trades.append({
                    'date': df.index[i], 'pnl': pnl_jpy, 'equity': capital,
                    'r_mult': pnl_jpy / (capital * risk_pct) if capital > 0 else 0,
                    'pips': (total_sl - avg_entry) * pos_dir / pip,
                    'n_entries': len(entries), 'exit_type': 'SL',
                })
                pos_dir = 0
                entries = []
                continue

            # 追加エントリー（グリッド）
            if len(entries) < n_splits:
                should_add = False
                if pos_dir == 1 and low_val <= next_grid_price:
                    should_add = True
                elif pos_dir == -1 and high_val >= next_grid_price:
                    should_add = True

                if should_add:
                    add_price = next_grid_price
                    add_lots = (capital * risk_pct / n_splits) / (signal_atr * max_sl_atr_mult * q2j)
                    if add_lots > 0:
                        entries.append((add_price, add_lots))
                        next_grid_price = add_price - pos_dir * grid_step
                        # SL更新: 最悪エントリーからATR×1.0の余裕
                        if pos_dir == 1:
                            total_sl = min(e[0] for e in entries) - signal_atr * 1.0
                        else:
                            total_sl = max(e[0] for e in entries) + signal_atr * 1.0

            continue

        # === 新規シグナル ===
        if sig == 0:
            continue

        # 新規ポジション開始（1/n_splits）
        pos_dir = sig
        signal_atr = atr
        grid_step = atr * grid_atr_mult
        lot_per_entry = (capital * risk_pct / n_splits) / (atr * max_sl_atr_mult * q2j)
        if lot_per_entry <= 0:
            pos_dir = 0
            continue

        entries = [(close_val, lot_per_entry)]
        next_grid_price = close_val - pos_dir * grid_step

        if pos_dir == 1:
            total_sl = close_val - atr * max_sl_atr_mult
        else:
            total_sl = close_val + atr * max_sl_atr_mult

    return pd.DataFrame(trades), capital


def london_bk_filtered(df, spread, pip, q2j,
                       tp_mult=1.5, risk_pct=0.02,
                       min_range_pips=20, max_range_pips=120):
    """
    高勝率ロンドンBK — フィルター強化版
    追加フィルター:
      1. 終値ベースBK: ヒゲではなく終値がレンジ外で確定したらエントリー
      2. 前日トレンド方向フィルター: 前日の終値 vs 始値で方向確認
      3. 曜日フィルター: 火-木のみ（月曜=持越しギャップ、金曜=手仕舞い圧力）
    """
    capital = float(INITIAL_CAPITAL)
    trades = []
    prev_day_dir = 0  # 1=前日陽線(上昇), -1=前日陰線(下降)

    day_groups = list(df.groupby(df.index.date))

    for day_idx, (date_val, day_data) in enumerate(day_groups):
        # 曜日フィルター: 火水木のみ (1=火, 2=水, 3=木)
        weekday = date_val.weekday()  # 0=月, 4=金
        if weekday < 1 or weekday > 3:
            # 前日方向の更新は行う
            if len(day_data) > 0:
                day_open = float(day_data.iloc[0]['Open'])
                day_close = float(day_data.iloc[-1]['Close'])
                prev_day_dir = 1 if day_close > day_open else -1
            continue

        # アジアセッション
        asian = day_data.between_time('00:00', '06:59')
        if len(asian) < 4:
            continue
        asian_high = float(asian['High'].max())
        asian_low = float(asian['Low'].min())
        asian_range = asian_high - asian_low
        range_pips = asian_range / pip

        if range_pips < min_range_pips or range_pips > max_range_pips:
            continue

        # ロンドンセッション
        london = day_data.between_time('07:00', '15:59')
        if len(london) == 0:
            continue

        tp = asian_range * tp_mult
        sl = asian_range
        in_trade = False
        entry = 0.0
        tp_price = 0.0
        sl_price = 0.0
        direction = 0
        pnl_r = None

        for idx, bar in london.iterrows():
            high = float(bar['High'])
            low = float(bar['Low'])
            close = float(bar['Close'])

            if in_trade:
                if direction == 1:
                    if low <= sl_price:
                        pnl_r = -1.0
                    elif high >= tp_price:
                        pnl_r = tp_mult
                else:
                    if high >= sl_price:
                        pnl_r = -1.0
                    elif low <= tp_price:
                        pnl_r = tp_mult

                if pnl_r is not None:
                    risk_amount = capital * risk_pct
                    pnl_jpy = pnl_r * risk_amount
                    capital += pnl_jpy
                    trades.append({
                        'date': idx, 'pnl': pnl_jpy, 'equity': capital,
                        'r_mult': pnl_r, 'range_pips': range_pips,
                    })
                    in_trade = False
                    break
                continue

            # 終値ベースBK（ヒゲではなく終値で判定 = ダマシ軽減）
            # + 前日トレンド方向フィルター
            if close > asian_high + spread:
                # ロングBK: 前日が陰線(下降)なら逆張りBKなのでスキップ
                if prev_day_dir == -1:
                    continue
                entry = close
                tp_price = entry + tp
                sl_price = entry - sl
                direction = 1
                in_trade = True

                if low <= sl_price:
                    pnl_r = -1.0
                elif high >= tp_price:
                    pnl_r = tp_mult

                if pnl_r is not None:
                    risk_amount = capital * risk_pct
                    pnl_jpy = pnl_r * risk_amount
                    capital += pnl_jpy
                    trades.append({
                        'date': idx, 'pnl': pnl_jpy, 'equity': capital,
                        'r_mult': pnl_r, 'range_pips': range_pips,
                    })
                    in_trade = False
                    break

            elif close < asian_low - spread:
                if prev_day_dir == 1:
                    continue
                entry = close
                tp_price = entry - tp
                sl_price = entry + sl
                direction = -1
                in_trade = True

                if high >= sl_price:
                    pnl_r = -1.0
                elif low <= tp_price:
                    pnl_r = tp_mult

                if pnl_r is not None:
                    risk_amount = capital * risk_pct
                    pnl_jpy = pnl_r * risk_amount
                    capital += pnl_jpy
                    trades.append({
                        'date': idx, 'pnl': pnl_jpy, 'equity': capital,
                        'r_mult': pnl_r, 'range_pips': range_pips,
                    })
                    in_trade = False
                    break

        # セッション終了時に時価決済
        if in_trade and pnl_r is None:
            last_close = float(london.iloc[-1]['Close'])
            if direction == 1:
                pnl_r = (last_close - entry) / sl
            else:
                pnl_r = (entry - last_close) / sl
            risk_amount = capital * risk_pct
            pnl_jpy = pnl_r * risk_amount
            capital += pnl_jpy
            trades.append({
                'date': london.index[-1], 'pnl': pnl_jpy, 'equity': capital,
                'r_mult': pnl_r, 'range_pips': range_pips,
            })

        # 前日方向の更新
        if len(day_data) > 0:
            day_open = float(day_data.iloc[0]['Open'])
            day_close = float(day_data.iloc[-1]['Close'])
            prev_day_dir = 1 if day_close > day_open else -1

    return pd.DataFrame(trades), capital


# ============================================================
# 複合ポートフォリオ
# ============================================================
def composite_portfolio(all_results):
    """3戦略 × 2ペアの複合ポートフォリオシミュレーション"""
    # 全トレードを時系列で結合
    all_trades = []
    weights = {'london_bk': 0.40, 'bb_martin': 0.35, 'trend': 0.25}

    for key, (trades_df, _) in all_results.items():
        if len(trades_df) == 0:
            continue
        strat = key.split('_', 1)[1] if '_' in key else key
        # 戦略名を抽出
        for s in weights:
            if s in key:
                w = weights[s]
                break
        else:
            w = 0.33

        for _, row in trades_df.iterrows():
            all_trades.append({
                'date': row['date'],
                'pnl': row['pnl'] * w,
                'strategy': key,
            })

    if not all_trades:
        return

    tdf = pd.DataFrame(all_trades).sort_values('date')
    capital = float(INITIAL_CAPITAL)
    equities = []
    for _, t in tdf.iterrows():
        capital += t['pnl']
        equities.append({'date': t['date'], 'equity': capital})

    edf = pd.DataFrame(equities)
    net = capital - INITIAL_CAPITAL
    ret = (capital / INITIAL_CAPITAL - 1) * 100
    days = (edf['date'].max() - edf['date'].min()).days
    months = max(days / 30.0, 0.5)
    monthly = ((capital / INITIAL_CAPITAL) ** (1 / months) - 1) * 100

    peak = np.maximum.accumulate(edf['equity'].values)
    dd = (peak - edf['equity'].values) / peak
    max_dd = dd.max() * 100

    # 月別
    edf_c = edf.copy()
    edf_c['month'] = pd.to_datetime(edf_c['date']).dt.to_period('M')
    prev_eq = INITIAL_CAPITAL
    monthly_rets = {}
    for m in sorted(edf_c['month'].unique()):
        grp = edf_c[edf_c['month'] == m]
        end_eq = grp['equity'].iloc[-1]
        monthly_rets[str(m)] = (end_eq / prev_eq - 1) * 100
        prev_eq = end_eq

    months_10 = sum(1 for r in monthly_rets.values() if r >= 10)

    print(f"\n{'#' * 70}")
    print(f"  複合ポートフォリオ (ロンドンBK 40% / BB逆張り 35% / トレンド 25%)")
    print(f"{'#' * 70}")
    print(f"  期間          : {days}日 ({months:.1f}ヶ月)")
    print(f"  総取引回数    : {len(tdf)}")
    print(f"  純損益        : ¥{net:>+,.0f} ({ret:>+.1f}%)")
    print(f"  月利          : {monthly:.2f}%")
    print(f"  MaxDD         : {max_dd:.1f}%")
    print(f"  月利10%達成月 : {months_10}/{len(monthly_rets)}ヶ月")

    if monthly_rets:
        print(f"\n  --- 月別リターン ---")
        for m, r in sorted(monthly_rets.items()):
            marker = " ★" if r >= 10 else (" ⚠️" if r < 0 else "")
            print(f"    {m}: {r:>+8.2f}%{marker}")


# ============================================================
# メイン
# ============================================================
def main():
    print("=" * 70)
    print("  GBP通貨ペア 月利10%戦略バックテスト")
    print(f"  実行日: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    data = fetch_data()
    if not data:
        print("データ取得失敗")
        return

    all_results = {}
    all_stats = []

    for ticker, cfg in PAIRS.items():
        if ticker not in data:
            continue
        df = data[ticker]
        spread = cfg['spread_pips'] * cfg['pip']
        pip = cfg['pip']
        q2j = cfg['quote_to_jpy']
        name = cfg['name']

        print(f"\n\n{'█' * 70}")
        print(f"  {name} ({ticker})")
        print(f"{'█' * 70}")

        # 戦略1: ロンドンブレイクアウト (パラメータグリッド)
        min_r = 30 if 'JPY' in ticker else 20
        max_r = 120 if 'JPY' in ticker else 80
        for tp_mult in [1.0, 1.5, 2.0]:
            for risk_pct in [0.02, 0.05, 0.08]:
                trades, cap = london_breakout(
                    df, spread, pip, q2j,
                    tp_mult=tp_mult, risk_pct=risk_pct,
                    min_range_pips=min_r, max_range_pips=max_r,
                )
                r_label = f"{int(risk_pct*100)}%"
                key = f"{name}_london_bk_TP{tp_mult}_R{r_label}"
                stats = print_strategy_results(
                    f"ロンドンBK {name} (TP={tp_mult}x, Risk={r_label})", trades, cap
                )
                if stats:
                    all_stats.append(stats)
                all_results[key] = (trades, cap)

        # 戦略2: BB逆張り + マーチン
        trades, cap = bb_reversal_martin(df, spread, pip, q2j)
        key = f"{name}_bb_martin"
        stats = print_strategy_results(
            f"BB逆張り+マーチン {name} (Risk=0.8%, RR=2.0)", trades, cap
        )
        if stats:
            all_stats.append(stats)
        all_results[key] = (trades, cap)

        # 戦略3: トレンドフォロー (長めEMAで15分足ノイズ軽減)
        for fast, slow in [(50, 200), (20, 100)]:
            for atr_m in [2.5, 3.5]:
                trades, cap = trend_follow_ema(
                    df, spread, pip, q2j,
                    fast_ema=fast, slow_ema=slow,
                    atr_mult=atr_m, risk_pct=0.02,
                )
                key = f"{name}_trend_EMA{fast}_{slow}_ATR{atr_m}"
                stats = print_strategy_results(
                    f"トレンドフォロー {name} (EMA{fast}/{slow}, ATR×{atr_m})", trades, cap
                )
                if stats:
                    all_stats.append(stats)
                all_results[key] = (trades, cap)

        # 戦略4: 分割エントリー（スケールイン）トレンドフォロー
        for n_sp in [5, 10]:
            for grid_m in [0.2, 0.3, 0.5]:
                for tp_m in [0.5, 1.0, 1.5]:
                    trades, cap = trend_follow_scalein(
                        df, spread, pip, q2j,
                        fast_ema=50, slow_ema=200,
                        risk_pct=0.02, n_splits=n_sp,
                        grid_atr_mult=grid_m, tp_atr_mult=tp_m,
                        max_sl_atr_mult=4.0,
                    )
                    key = f"{name}_scalein_{n_sp}x_G{grid_m}_TP{tp_m}"
                    stats = print_strategy_results(
                        f"◆分割エントリー {name} ({n_sp}分割, Grid={grid_m}ATR, TP={tp_m}ATR)",
                        trades, cap
                    )
                    if stats:
                        all_stats.append(stats)
                        # 追加情報: 平均分割数
                        if len(trades) > 0 and 'n_entries' in trades.columns:
                            avg_n = trades['n_entries'].mean()
                            print(f"  平均分割数    : {avg_n:.1f} / {n_sp}")
                    all_results[key] = (trades, cap)

        # 戦略3b: 高勝率トレンドフォロー（フィルター強化版）
        for fast, slow in [(50, 200), (20, 100)]:
            for atr_m in [2.5, 3.5]:
                for adx_t in [20, 25]:
                    trades, cap, skipped = trend_follow_filtered(
                        df, spread, pip, q2j,
                        fast_ema=fast, slow_ema=slow,
                        atr_mult=atr_m, risk_pct=0.02,
                        adx_thresh=adx_t,
                        use_h1_filter=True, candle_filter=True,
                    )
                    key = f"{name}_trend_filtered_EMA{fast}_{slow}_ATR{atr_m}_ADX{adx_t}"
                    stats = print_strategy_results(
                        f"★高勝率TF {name} (EMA{fast}/{slow}, ATR×{atr_m}, ADX>{adx_t})",
                        trades, cap
                    )
                    if stats:
                        all_stats.append(stats)
                        print(f"  フィルター除外 : {skipped}回 (ダマシ回避)")
                    all_results[key] = (trades, cap)

        # 戦略1b: 高勝率ロンドンBK（フィルター強化版）
        min_r = 30 if 'JPY' in ticker else 20
        max_r = 120 if 'JPY' in ticker else 80
        for tp_mult in [1.0, 1.5, 2.0]:
            for risk_pct in [0.02, 0.05]:
                trades, cap = london_bk_filtered(
                    df, spread, pip, q2j,
                    tp_mult=tp_mult, risk_pct=risk_pct,
                    min_range_pips=min_r, max_range_pips=max_r,
                )
                r_label = f"{int(risk_pct*100)}%"
                key = f"{name}_london_bk_filtered_TP{tp_mult}_R{r_label}"
                stats = print_strategy_results(
                    f"★高勝率BK {name} (TP={tp_mult}x, Risk={r_label}, 火水木+トレンド)",
                    trades, cap
                )
                if stats:
                    all_stats.append(stats)
                all_results[key] = (trades, cap)

    # サマリー比較
    if all_stats:
        print(f"\n\n{'=' * 70}")
        print(f"  全戦略サマリー比較")
        print(f"{'=' * 70}")
        print(f"  {'戦略名':<45s} {'取引':>4s} {'勝率':>6s} {'PF':>6s} {'月利':>7s} {'DD':>6s} {'10%月':>5s}")
        print(f"  {'-' * 70}")
        for s in sorted(all_stats, key=lambda x: x['monthly'], reverse=True):
            print(f"  {s['name']:<45s} {s['trades']:4d} {s['win_rate']:5.1f}% {s['pf']:6.2f} {s['monthly']:>6.2f}% {s['max_dd']:5.1f}% {s['months_10pct']:>2d}/{s['total_months']}")

    # 複合ポートフォリオ (各ペアのベスト設定を使用)
    best_per_strat = {}
    for key, (trades, cap) in all_results.items():
        if len(trades) == 0:
            continue
        for strat_key in ['london_bk', 'bb_martin', 'trend']:
            if strat_key in key:
                if strat_key not in best_per_strat:
                    best_per_strat[strat_key] = (key, trades, cap)
                else:
                    _, _, prev_cap = best_per_strat[strat_key]
                    if cap > prev_cap:
                        best_per_strat[strat_key] = (key, trades, cap)

    composite_results = {}
    for strat_key, (key, trades, cap) in best_per_strat.items():
        composite_results[key] = (trades, cap)

    if composite_results:
        composite_portfolio(composite_results)

    # 結論
    print(f"\n\n{'=' * 70}")
    print(f"  結論")
    print(f"{'=' * 70}")
    if all_stats:
        best = max(all_stats, key=lambda x: x['monthly'])
        print(f"\n  最高月利: {best['monthly']:.2f}% ({best['name']})")
        achievable = best['monthly'] >= 10
        print(f"  月利10%達成可能性: {'達成圏内' if achievable else '現パラメータでは未達'}")
        if not achievable:
            print(f"  → リスク率引き上げ or 複合運用で月利10%を目指す必要あり")
            print(f"  → ただしMaxDD増大とのトレードオフに注意")
    print()


if __name__ == '__main__':
    main()
