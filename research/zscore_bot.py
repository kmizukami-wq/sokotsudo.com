#!/usr/bin/env python3
"""
Zスコア逆張り自動売買システム
============================
- zscore_config.json で通貨ペア追加・パラメータ変更可能
- 3つのモードで動作:
  1. signal  : 今日のシグナルを表示（日次チェック用）
  2. backtest: 過去データでバックテスト
  3. monitor : 継続的にシグナルを監視（cron/タスクスケジューラ用）

使い方:
  python zscore_bot.py signal                # 今日のシグナル確認
  python zscore_bot.py backtest              # 全ペアバックテスト
  python zscore_bot.py backtest EUR/GBP      # 特定ペアのみ
  python zscore_bot.py monitor               # 継続監視（毎日実行）
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'zscore_config.json')
DATA_PATH = os.path.join(SCRIPT_DIR, 'data_fx_long.csv')
LOG_PATH = os.path.join(SCRIPT_DIR, 'trades.log')


# ============================================================
# 設定読み込み
# ============================================================
def load_config() -> dict:
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def get_enabled_pairs(config: dict) -> dict:
    return {k: v for k, v in config['pairs'].items() if v.get('enabled', True)}


# ============================================================
# データ取得
# ============================================================
def fetch_latest_prices(pairs: list, days: int = 90) -> pd.DataFrame:
    """Frankfurter API から直近の為替データを取得"""
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    # EUR基準でUSD, GBP, JPYを取得
    symbols = set()
    for pair in pairs:
        base, quote = pair.split('/')
        symbols.add(base)
        symbols.add(quote)
    symbols.discard('EUR')
    symbols_str = ','.join(sorted(symbols))

    url = f"https://api.frankfurter.app/{start}..{end}?from=EUR&to={symbols_str}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            print(f"  [ERROR] API error: {r.status_code}")
            return pd.DataFrame()

        data = r.json()
        rows = []
        for date_str, rates in data.get('rates', {}).items():
            row = {'date': date_str}
            row['EUR/USD'] = rates.get('USD')
            row['EUR/GBP'] = rates.get('GBP')
            row['EUR/JPY'] = rates.get('JPY')
            if row['EUR/USD'] and row['EUR/GBP']:
                row['GBP/USD'] = row['EUR/USD'] / row['EUR/GBP']
            if row['EUR/USD'] and row['EUR/JPY']:
                row['USD/JPY'] = row['EUR/JPY'] / row['EUR/USD']
            rows.append(row)

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        return df

    except Exception as e:
        print(f"  [ERROR] データ取得失敗: {e}")
        return pd.DataFrame()


def load_historical_data() -> pd.DataFrame:
    """ローカルの長期データを読み込み"""
    if not os.path.exists(DATA_PATH):
        print(f"  [WARN] {DATA_PATH} が見つかりません。APIからデータを取得します。")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df.set_index('date', inplace=True)

    # カラム名を統一
    rename = {
        'eurusd': 'EUR/USD', 'eurgbp': 'EUR/GBP', 'eurjpy': 'EUR/JPY',
        'usdjpy': 'USD/JPY', 'gbpusd': 'GBP/USD',
    }
    df = df.rename(columns=rename)
    return df.sort_index()


# ============================================================
# Zスコア計算
# ============================================================
def calculate_zscore(prices: pd.Series, window: int) -> pd.Series:
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return (prices - mean) / std


# ============================================================
# シグナル判定
# ============================================================
def check_signals(config: dict, data: pd.DataFrame) -> list:
    """全ペアの現在のシグナルを判定"""
    signals = []
    pairs = get_enabled_pairs(config)

    for pair_name, params in pairs.items():
        if pair_name not in data.columns:
            continue

        prices = data[pair_name].dropna()
        if len(prices) < params['window'] + 5:
            continue

        z = calculate_zscore(prices, params['window'])
        current_z = z.iloc[-1]
        prev_z = z.iloc[-2] if len(z) > 1 else 0
        current_price = prices.iloc[-1]
        ma = prices.rolling(params['window']).mean().iloc[-1]
        std = prices.rolling(params['window']).std().iloc[-1]

        signal = {
            'pair': pair_name,
            'price': current_price,
            'ma': ma,
            'std': std,
            'z_score': current_z,
            'prev_z': prev_z,
            'action': 'WAIT',
            'params': params,
        }

        entry_z = params['entry_z']
        exit_z = params['exit_z']
        stop_z = params['stop_z']

        # エントリーシグナル
        if current_z > entry_z:
            signal['action'] = 'SELL'
            signal['reason'] = f'Z={current_z:.2f} > +{entry_z}（割高→売り）'
        elif current_z < -entry_z:
            signal['action'] = 'BUY'
            signal['reason'] = f'Z={current_z:.2f} < -{entry_z}（割安→買い）'
        # 決済シグナル
        elif abs(current_z) < exit_z:
            signal['action'] = 'CLOSE'
            signal['reason'] = f'Z={current_z:.2f}（平均回帰→利確決済）'
        elif abs(current_z) > stop_z:
            signal['action'] = 'STOP'
            signal['reason'] = f'Z={current_z:.2f}（乖離拡大→損切り）'
        else:
            signal['reason'] = f'Z={current_z:.2f}（様子見）'

        signals.append(signal)

    return signals


# ============================================================
# バックテスト
# ============================================================
def backtest_pair(prices: pd.Series, params: dict) -> pd.DataFrame:
    """1ペアのバックテスト"""
    window = params['window']
    entry_z = params['entry_z']
    exit_z = params['exit_z']
    stop_z = params['stop_z']
    risk_pct = params['risk_pct'] / 100

    z = calculate_zscore(prices, window)

    results = []
    equity = 1.0
    pos = 0
    ez = 0.0
    entry_date = None

    for i in range(window, len(z)):
        zi = z.iloc[i]
        if pd.isna(zi):
            continue

        if pos == 0:
            if zi > entry_z:
                pos = -1; ez = zi; entry_date = z.index[i]
            elif zi < -entry_z:
                pos = 1; ez = zi; entry_date = z.index[i]

        elif pos == 1:
            if zi > -exit_z or zi < -stop_z:
                pnl = (-ez - (-zi)) / entry_z * risk_pct
                equity *= (1 + pnl)
                results.append({
                    'entry_date': entry_date, 'exit_date': z.index[i],
                    'direction': 'BUY', 'entry_z': ez, 'exit_z': zi,
                    'pnl_pct': pnl * 100, 'equity': equity,
                    'days': (z.index[i] - entry_date).days,
                    'reason': 'TP' if zi > -exit_z else 'SL'
                })
                pos = 0

        elif pos == -1:
            if zi < exit_z or zi > stop_z:
                pnl = (ez - zi) / entry_z * risk_pct
                equity *= (1 + pnl)
                results.append({
                    'entry_date': entry_date, 'exit_date': z.index[i],
                    'direction': 'SELL', 'entry_z': ez, 'exit_z': zi,
                    'pnl_pct': pnl * 100, 'equity': equity,
                    'days': (z.index[i] - entry_date).days,
                    'reason': 'TP' if zi < exit_z else 'SL'
                })
                pos = 0

    return pd.DataFrame(results)


# ============================================================
# コマンド: signal
# ============================================================
def cmd_signal():
    config = load_config()
    print("=" * 70)
    print(f"  Zスコア逆張り シグナル確認 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("=" * 70)

    pairs = list(get_enabled_pairs(config).keys())
    data = fetch_latest_prices(pairs, days=90)

    if data.empty:
        print("  データ取得に失敗しました")
        return

    print(f"  データ期間: {data.index[0].date()} ~ {data.index[-1].date()}\n")

    signals = check_signals(config, data)

    for s in signals:
        icon = {'BUY': '🟢↑買', 'SELL': '🔴↓売', 'CLOSE': '🟡決済',
                'STOP': '⛔損切', 'WAIT': '⚪様子見'}
        print(f"  {s['pair']:<10} {icon.get(s['action'], '?')}")
        print(f"    価格: {s['price']:.4f}  MA({s['params']['window']}): {s['ma']:.4f}  "
              f"σ: {s['std']:.6f}")
        print(f"    Zスコア: {s['z_score']:+.2f}  ({s['reason']})")
        print(f"    Risk: {s['params']['risk_pct']}%\n")


# ============================================================
# コマンド: backtest
# ============================================================
def cmd_backtest(target_pair=None):
    config = load_config()
    data = load_historical_data()

    if data.empty:
        pairs = list(get_enabled_pairs(config).keys())
        data = fetch_latest_prices(pairs, days=365*10)
        if data.empty:
            print("データなし"); return

    years = (data.index[-1] - data.index[0]).days / 365.25

    print("=" * 70)
    print(f"  Zスコア逆張り バックテスト")
    print(f"  データ: {data.index[0].date()} ~ {data.index[-1].date()} ({years:.1f}年)")
    print("=" * 70)

    pairs = get_enabled_pairs(config)
    if target_pair:
        pairs = {k: v for k, v in pairs.items() if k == target_pair}

    summary = []
    for pair_name, params in pairs.items():
        if pair_name not in data.columns:
            print(f"\n  {pair_name}: データなし"); continue

        prices = data[pair_name].dropna()
        pair_years = (prices.index[-1] - prices.index[0]).days / 365.25
        rdf = backtest_pair(prices, params)

        if len(rdf) == 0:
            print(f"\n  {pair_name}: トレードなし"); continue

        final = rdf['equity'].iloc[-1]
        cagr = ((final ** (1/pair_years)) - 1) * 100
        monthly = ((final ** (1/(pair_years*12))) - 1) * 100
        rm = rdf['equity'].cummax()
        max_dd = ((rm - rdf['equity']) / rm).max() * 100
        wins = (rdf['pnl_pct'] > 0).sum()
        losses = (rdf['pnl_pct'] <= 0).sum()
        stops = (rdf['reason'] == 'SL').sum()

        print(f"\n  {'='*60}")
        print(f"  {pair_name} (W={params['window']}, Z=±{params['entry_z']}, R={params['risk_pct']}%)")
        print(f"  {'='*60}")
        print(f"  取引: {len(rdf)}回 ({len(rdf)/(pair_years*12):.1f}回/月) | "
              f"勝率: {wins}/{len(rdf)} ({wins/len(rdf)*100:.1f}%) | 損切: {stops}回")
        print(f"  月利: {monthly:.2f}% | CAGR: {cagr:.1f}% | MaxDD: {max_dd:.1f}%")
        print(f"  平均保有: {rdf['days'].mean():.0f}日 | 平均損益: {rdf['pnl_pct'].mean():.2f}%")

        # 年別
        rdf['year'] = pd.to_datetime(rdf['exit_date']).dt.year
        print(f"  年別: ", end="")
        for yr in sorted(rdf['year'].unique()):
            yd = rdf[rdf['year'] == yr]
            prev_eq = yd['equity'].iloc[0] / (1 + yd['pnl_pct'].iloc[0]/100)
            yr_ret = (yd['equity'].iloc[-1] / prev_eq - 1) * 100
            print(f"{yr}:{yr_ret:+.0f}% ", end="")
        print()

        summary.append({
            'pair': pair_name, 'trades': len(rdf), 'win_rate': wins/len(rdf)*100,
            'monthly': monthly, 'cagr': cagr, 'max_dd': max_dd, 'stops': stops
        })

    if len(summary) > 1:
        print(f"\n\n  {'='*60}")
        print(f"  サマリー")
        print(f"  {'='*60}")
        print(f"  {'Pair':<10} {'Trades':>6} {'WR%':>6} {'月利':>7} {'CAGR':>8} {'MaxDD':>7} {'損切':>4}")
        print(f"  {'-'*55}")
        for s in summary:
            print(f"  {s['pair']:<10} {s['trades']:>6} {s['win_rate']:>5.1f}% "
                  f"{s['monthly']:>6.2f}% {s['cagr']:>7.1f}% {s['max_dd']:>6.1f}% {s['stops']:>4}")


# ============================================================
# コマンド: monitor
# ============================================================
def cmd_monitor():
    """シグナル確認してログ記録（cron向け）"""
    config = load_config()
    pairs = list(get_enabled_pairs(config).keys())
    data = fetch_latest_prices(pairs, days=90)

    if data.empty:
        return

    signals = check_signals(config, data)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    actionable = [s for s in signals if s['action'] in ('BUY', 'SELL', 'CLOSE', 'STOP')]

    if actionable:
        log_lines = []
        for s in actionable:
            line = (f"[{timestamp}] {s['action']} {s['pair']} "
                    f"Z={s['z_score']:.2f} Price={s['price']:.4f} "
                    f"Risk={s['params']['risk_pct']}%")
            log_lines.append(line)
            print(line)

        with open(LOG_PATH, 'a') as f:
            f.write('\n'.join(log_lines) + '\n')
    else:
        print(f"[{timestamp}] シグナルなし")


# ============================================================
# メイン
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == 'signal':
        cmd_signal()
    elif cmd == 'backtest':
        target = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_backtest(target)
    elif cmd == 'monitor':
        cmd_monitor()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python zscore_bot.py [signal|backtest|monitor]")
