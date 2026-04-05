#!/usr/bin/env python3
"""
Zスコア逆張り LINE通知システム
==============================
エントリー・利確・損切りのタイミングをLINEに通知する。
取引は手動で行う。

■ セットアップ手順:
  1. LINE Developers (https://developers.line.biz/) にログイン
  2. 新しいプロバイダー → 新しいチャネル → Messaging API を作成
  3. チャネルアクセストークン（長期）を発行
  4. 作成したBotの公式アカウントをLINEで友だち追加
  5. あなたのユーザーIDを取得（Basic settings → Your user ID）
  6. zscore_notify_config.json に TOKEN と USER_ID を設定
  7. python zscore_notify.py test  で通知テスト

■ 日次運用:
  毎朝7:00 (JST) に以下を実行（cron設定推奨）:
  python zscore_notify.py check

■ cron設定例:
  0 7 * * 1-5 cd /path/to/research && python3 zscore_notify.py check
"""

import json
import sys
import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'zscore_notify_config.json')
STATE_PATH = os.path.join(SCRIPT_DIR, 'zscore_state.json')
LOG_PATH = os.path.join(SCRIPT_DIR, 'zscore_notify.log')


# ============================================================
# 設定
# ============================================================
def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        print(f"[ERROR] {CONFIG_PATH} が見つかりません。")
        print(f"  zscore_notify_config.json を作成してください。")
        sys.exit(1)
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def load_state() -> dict:
    """ポジション状態を読み込み（どのペアを保有中か）"""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, 'r') as f:
            return json.load(f)
    return {"positions": {}}


def save_state(state: dict):
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


# ============================================================
# LINE Messaging API 送信
# ============================================================
def send_line(config: dict, message: str) -> bool:
    """LINE Messaging API でプッシュメッセージ送信"""
    token = config['line']['channel_access_token']
    user_id = config['line']['user_id']

    if not token or token == 'YOUR_CHANNEL_ACCESS_TOKEN':
        print(f"[LINE] トークン未設定。コンソールに出力:")
        print(f"  {message}")
        return False

    url = 'https://api.line.me/v2/bot/message/push'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    payload = {
        'to': user_id,
        'messages': [{'type': 'text', 'text': message}]
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        else:
            print(f"[LINE ERROR] {r.status_code}: {r.text}")
            return False
    except Exception as e:
        print(f"[LINE ERROR] {e}")
        return False


# ============================================================
# 為替データ取得
# ============================================================
def fetch_realtime_prices(api_key: str) -> dict:
    """Twelve Data APIからリアルタイム価格取得（7 APIクレジット→26ペア計算）"""
    if not api_key or api_key == 'YOUR_TWELVE_DATA_API_KEY':
        return {}

    symbols = 'EUR/USD,EUR/GBP,EUR/JPY,EUR/AUD,EUR/NZD,EUR/CAD,EUR/CHF'
    try:
        r = requests.get('https://api.twelvedata.com/price',
            params={'symbol': symbols, 'apikey': api_key}, timeout=10)
        data = r.json()
        if 'code' in data:
            print(f"  [WARN] Twelve Data: {data.get('message', 'error')}")
            return {}

        eu = float(data['EUR/USD']['price'])
        eg = float(data['EUR/GBP']['price'])
        ej = float(data['EUR/JPY']['price'])
        ea = float(data['EUR/AUD']['price'])
        en = float(data['EUR/NZD']['price'])
        ec = float(data['EUR/CAD']['price'])
        ef = float(data['EUR/CHF']['price'])

        return {
            'EUR/USD': eu, 'EUR/GBP': eg, 'EUR/JPY': ej,
            'EUR/AUD': ea, 'EUR/NZD': en, 'EUR/CAD': ec, 'EUR/CHF': ef,
            'GBP/USD': eu/eg, 'USD/JPY': ej/eu, 'AUD/NZD': en/ea,
            'AUD/JPY': ej/ea, 'NZD/JPY': ej/en, 'AUD/USD': eu/ea,
            'NZD/USD': eu/en, 'GBP/JPY': ej/eg, 'USD/CAD': ec/eu,
            'GBP/CAD': ec/eg, 'CAD/JPY': ej/ec, 'AUD/CAD': ec/ea,
            'NZD/CAD': ec/en, 'CAD/CHF': ef/ec, 'USD/CHF': ef/eu,
            'GBP/CHF': ef/eg, 'CHF/JPY': ej/ef, 'AUD/CHF': ef/ea,
            'NZD/CHF': ef/en,
        }
    except Exception as e:
        print(f"  [ERROR] Twelve Data: {e}")
        return {}


def fetch_prices(pairs_config: dict, days: int = 90) -> pd.DataFrame:
    """Frankfurter API (ECB) から過去データ取得"""
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    url = f"https://api.frankfurter.app/{start}..{end}?from=EUR&to=USD,GBP,JPY,AUD,NZD,CAD,CHF"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()

        data = r.json()
        rows = []
        for date_str, rates in data.get('rates', {}).items():
            usd = rates.get('USD')
            gbp = rates.get('GBP')
            jpy = rates.get('JPY')
            aud = rates.get('AUD')
            nzd = rates.get('NZD')
            cad = rates.get('CAD')
            chf = rates.get('CHF')
            row = {'date': date_str}
            if usd: row['EUR/USD'] = usd
            if gbp: row['EUR/GBP'] = gbp
            if jpy: row['EUR/JPY'] = jpy
            if aud: row['EUR/AUD'] = aud
            if nzd: row['EUR/NZD'] = nzd
            if cad: row['EUR/CAD'] = cad
            if chf: row['EUR/CHF'] = chf
            if usd and gbp: row['GBP/USD'] = usd / gbp
            if usd and jpy: row['USD/JPY'] = jpy / usd
            if aud and nzd: row['AUD/NZD'] = nzd / aud
            if aud and jpy: row['AUD/JPY'] = jpy / aud
            if nzd and jpy: row['NZD/JPY'] = jpy / nzd
            if aud and usd: row['AUD/USD'] = usd / aud
            if nzd and usd: row['NZD/USD'] = usd / nzd
            if gbp and jpy: row['GBP/JPY'] = jpy / gbp
            if usd and cad: row['USD/CAD'] = cad / usd
            if gbp and cad: row['GBP/CAD'] = cad / gbp
            if cad and jpy: row['CAD/JPY'] = jpy / cad
            if aud and cad: row['AUD/CAD'] = cad / aud
            if nzd and cad: row['NZD/CAD'] = cad / nzd
            if cad and chf: row['CAD/CHF'] = chf / cad
            if usd and chf: row['USD/CHF'] = chf / usd
            if gbp and chf: row['GBP/CHF'] = chf / gbp
            if chf and jpy: row['CHF/JPY'] = jpy / chf
            if aud and chf: row['AUD/CHF'] = chf / aud
            if nzd and chf: row['NZD/CHF'] = chf / nzd
            rows.append(row)

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date').set_index('date')
    except Exception as e:
        print(f"[ERROR] データ取得失敗: {e}")
        return pd.DataFrame()


# ============================================================
# Zスコア計算 & シグナル判定
# ============================================================
def analyze_pair(prices: pd.Series, params: dict) -> dict:
    """1ペアのZスコア計算とシグナル判定"""
    window = params['window']
    if len(prices) < window + 5:
        return None

    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    z = (prices - ma) / std

    current_z = z.iloc[-1]
    current_price = prices.iloc[-1]
    current_ma = ma.iloc[-1]
    current_std = std.iloc[-1]

    return {
        'price': current_price,
        'ma': current_ma,
        'std': current_std,
        'z': current_z,
        'date': prices.index[-1].strftime('%Y-%m-%d'),
    }


def determine_action(analysis: dict, params: dict, current_position: dict) -> dict:
    """
    現在のポジション状態とZスコアからアクションを判定

    Returns: {'action': 'ENTRY_BUY'|'ENTRY_SELL'|'TAKE_PROFIT'|'STOP_LOSS'|'HOLD'|'WAIT',
              'message': str}
    """
    z = analysis['z']
    entry_z = params['entry_z']
    exit_z = params['exit_z']
    stop_z = params['stop_z']

    has_position = current_position.get('direction') is not None

    if not has_position:
        # ポジションなし → エントリー判定
        if z > entry_z:
            return {
                'action': 'ENTRY_SELL',
                'message': f'売りエントリー（Z={z:+.2f} > +{entry_z}）'
            }
        elif z < -entry_z:
            return {
                'action': 'ENTRY_BUY',
                'message': f'買いエントリー（Z={z:+.2f} < -{entry_z}）'
            }
        else:
            return {
                'action': 'WAIT',
                'message': f'様子見（Z={z:+.2f}）'
            }
    else:
        # ポジションあり → 決済判定
        direction = current_position['direction']

        if direction == 'BUY':
            if z > -exit_z:
                return {
                    'action': 'TAKE_PROFIT',
                    'message': f'利確決済（Z={z:+.2f}、平均回帰）'
                }
            elif z < -stop_z:
                return {
                    'action': 'STOP_LOSS',
                    'message': f'損切り決済（Z={z:+.2f} < -{stop_z}）'
                }
            else:
                return {
                    'action': 'HOLD',
                    'message': f'保有継続（Z={z:+.2f}）'
                }

        elif direction == 'SELL':
            if z < exit_z:
                return {
                    'action': 'TAKE_PROFIT',
                    'message': f'利確決済（Z={z:+.2f}、平均回帰）'
                }
            elif z > stop_z:
                return {
                    'action': 'STOP_LOSS',
                    'message': f'損切り決済（Z={z:+.2f} > +{stop_z}）'
                }
            else:
                return {
                    'action': 'HOLD',
                    'message': f'保有継続（Z={z:+.2f}）'
                }

    return {'action': 'WAIT', 'message': '不明'}


# ============================================================
# ロット計算
# ============================================================
def calculate_lot(pair: str, params: dict, account_balance: float) -> dict:
    """口座残高とSL幅から推奨ロット（1000通貨単位）を計算"""
    risk_pct = params.get('risk_pct', 15) / 100
    stop_z = params.get('stop_z', 2.0)
    entry_z = params.get('entry_z', 1.5)
    window = params.get('window', 30)

    # pip値
    quote = pair.split('/')[1]
    pip_size = 0.01 if quote == 'JPY' else 0.0001
    pip_jpy_map = {'JPY': 1000, 'USD': 1500, 'GBP': 2000,
                   'CAD': 1100, 'CHF': 1700, 'AUD': 1000, 'NZD': 900}
    pip_value = pip_jpy_map.get(quote, 1500)

    # SL幅 = σ × (entry_z + stop_z) をpipsに換算
    # σの概算値（ペア別）
    sigma_map = {
        'EUR/GBP': 0.0071, 'EUR/USD': 0.0137, 'EUR/JPY': 1.72,
        'EUR/CAD': 0.0165, 'EUR/CHF': 0.0077, 'EUR/AUD': 0.0193,
        'EUR/NZD': 0.0235, 'USD/JPY': 1.38, 'USD/CAD': 0.0123,
        'USD/CHF': 0.0138, 'AUD/USD': 0.0108, 'AUD/NZD': 0.0102,
        'AUD/JPY': 1.36, 'AUD/CAD': 0.0096, 'AUD/CHF': 0.0113,
        'NZD/USD': 0.0100, 'NZD/JPY': 1.22, 'NZD/CHF': 0.0100,
        'NZD/CAD': 0.0099, 'CAD/JPY': 1.32, 'CAD/CHF': 0.0114,
        'CHF/JPY': 1.29, 'GBP/USD': 0.0170, 'GBP/JPY': 2.10,
        'GBP/CHF': 0.0160, 'GBP/CAD': 0.0200,
    }
    sigma = sigma_map.get(pair, 0.01)
    sl_price = sigma * (entry_z + stop_z)
    sl_pips = sl_price / pip_size

    # ロット計算
    # pip_value は1ロット（10万通貨）あたりの1pipの円価値
    # 1通貨あたりの1pipの円価値 = pip_value / 100000
    pip_value_per_unit = pip_value / 100000
    max_loss = account_balance * risk_pct
    units = max_loss / (sl_pips * pip_value_per_unit)
    lot_1000 = max(1000, int(units / 1000) * 1000)  # 1000通貨単位に切り捨て
    actual_loss = sl_pips * pip_value_per_unit * lot_1000

    # 損切り価格の計算
    sl_pips_rounded = round(sl_pips)

    return {
        'lot_units': lot_1000,
        'lot': lot_1000 / 100000,
        'sl_pips': sl_pips_rounded,
        'max_loss': actual_loss,
        'max_loss_pct': actual_loss / account_balance * 100,
    }


# ============================================================
# 通知メッセージ作成
# ============================================================
def format_entry_message(pair: str, analysis: dict, action: dict, params: dict,
                         account_balance: float = None) -> str:
    direction = '🔴 売り SELL' if action['action'] == 'ENTRY_SELL' else '🟢 買い BUY'

    # ロット計算
    lot_info = ""
    if account_balance and account_balance > 0:
        lc = calculate_lot(pair, params, account_balance)
        # 損切り価格
        price = analysis['price']
        pip_size = 0.01 if pair.split('/')[1] == 'JPY' else 0.0001
        if action['action'] == 'ENTRY_BUY':
            sl_price = price - lc['sl_pips'] * pip_size
        else:
            sl_price = price + lc['sl_pips'] * pip_size

        lot_info = f"""
💰 推奨ロット: {lc['lot_units']:,}通貨
損切り: {sl_price:.4f}（{lc['sl_pips']}pips）
最大損失: {lc['max_loss']:,.0f}円（口座の{lc['max_loss_pct']:.1f}%）"""

    return f"""
━━━━━━━━━━━━━━━━
📊 {pair} エントリー通知
━━━━━━━━━━━━━━━━
方向: {direction}
日付: {analysis['date']}

価格: {analysis['price']:.4f}
30日MA: {analysis['ma']:.4f}
Zスコア: {analysis['z']:+.2f}
{lot_info}

⚠️ 手動でエントリーしてください
""".strip()


def format_exit_message(pair: str, analysis: dict, action: dict, position: dict) -> str:
    if action['action'] == 'TAKE_PROFIT':
        icon = '✅'
        exit_type = '利確'
    else:
        icon = '⛔'
        exit_type = '損切り'

    entry_z = position.get('entry_z', 0)
    entry_price = position.get('entry_price', 0)
    entry_date = position.get('entry_date', '?')
    direction = position.get('direction', '?')

    return f"""
━━━━━━━━━━━━━━━━
{icon} {pair} {exit_type}通知
━━━━━━━━━━━━━━━━
方向: {direction}
エントリー: {entry_date} @ {entry_price:.4f} (Z={entry_z:+.2f})
現在: {analysis['date']} @ {analysis['price']:.4f} (Z={analysis['z']:+.2f})

{action['message']}

⚠️ 手動で決済してください
""".strip()


def format_daily_summary(results: list) -> str:
    """日次サマリーメッセージ"""
    date = datetime.now().strftime('%Y-%m-%d')
    lines = [f"📋 Zスコア日次レポート ({date})", ""]

    active = [r for r in results if r['action']['action'] in ('HOLD',)]
    waiting = [r for r in results if r['action']['action'] == 'WAIT']

    if active:
        lines.append("【保有中】")
        for r in active:
            pos = r.get('position', {})
            lines.append(f"  {r['pair']}: {pos.get('direction','-')} Z={r['analysis']['z']:+.2f}")
        lines.append("")

    if waiting:
        lines.append("【待機中】")
        for r in waiting:
            lines.append(f"  {r['pair']}: Z={r['analysis']['z']:+.2f}")

    return '\n'.join(lines)


# ============================================================
# コマンド: check（メイン処理）
# ============================================================
def cmd_check():
    config = load_config()
    state = load_state()

    pairs_config = config['pairs']

    # 過去30日のデータ取得（MA/σ計算用）
    data = fetch_prices(pairs_config, days=90)
    if data.empty:
        print("[ERROR] 過去データ取得失敗")
        return

    # リアルタイム価格取得（Twelve Data）
    api_key = config.get('twelve_data', {}).get('api_key', '')
    live_prices = fetch_realtime_prices(api_key)

    if live_prices:
        # リアルタイム価格を最終行として追加
        today = datetime.now().strftime('%Y-%m-%d')
        live_row = {pair: live_prices.get(pair) for pair in data.columns if pair in live_prices}
        if live_row:
            live_df = pd.DataFrame([live_row], index=pd.DatetimeIndex([today]))
            data = pd.concat([data, live_df])
            data = data[~data.index.duplicated(keep='last')]
            data = data.sort_index()
        source = "リアルタイム (Twelve Data)"
    else:
        source = "日次 (ECB)"

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] シグナルチェック ({source})")
    print(f"  データ: {data.index[0].date()} ~ {data.index[-1].date()}")

    results = []
    notifications = []

    for pair_name, params in pairs_config.items():
        if not params.get('enabled', True):
            continue
        if pair_name not in data.columns:
            continue

        prices = data[pair_name].dropna()
        analysis = analyze_pair(prices, params)
        if analysis is None:
            continue

        current_position = state.get('positions', {}).get(pair_name, {})
        action = determine_action(analysis, params, current_position)

        result = {
            'pair': pair_name, 'analysis': analysis,
            'action': action, 'position': current_position,
        }
        results.append(result)

        # アクションに応じた処理
        if action['action'] in ('ENTRY_BUY', 'ENTRY_SELL'):
            account_balance = config.get('account_balance', 0)
            msg = format_entry_message(pair_name, analysis, action, params, account_balance)
            notifications.append(msg)

            direction = 'BUY' if action['action'] == 'ENTRY_BUY' else 'SELL'
            state.setdefault('positions', {})[pair_name] = {
                'direction': direction,
                'entry_price': analysis['price'],
                'entry_z': analysis['z'],
                'entry_date': analysis['date'],
            }
            print(f"  🔔 {pair_name}: {action['message']}")

        elif action['action'] in ('TAKE_PROFIT', 'STOP_LOSS'):
            msg = format_exit_message(pair_name, analysis, action, current_position)
            notifications.append(msg)

            state.get('positions', {}).pop(pair_name, None)
            print(f"  🔔 {pair_name}: {action['message']}")

        elif action['action'] == 'HOLD':
            print(f"  📍 {pair_name}: {action['message']}")

        else:
            print(f"  ⚪ {pair_name}: {action['message']}")

    # LINE送信（シグナルがある場合のみ）
    if notifications:
        for msg in notifications:
            send_line(config, msg)
            log_entry = f"[{datetime.now().isoformat()}] {msg}\n"
            with open(LOG_PATH, 'a') as f:
                f.write(log_entry)
        # シグナルと一緒にサマリーも送信
        summary = format_daily_summary(results)
        send_line(config, summary)
    else:
        print("  シグナルなし → 通知スキップ")

    # 最終チェック時刻を記録
    state['last_checked'] = datetime.now().isoformat()

    # 状態保存
    save_state(state)
    print(f"\n  通知送信: {len(notifications)}件")


# ============================================================
# コマンド: test（通知テスト）
# ============================================================
def cmd_test():
    config = load_config()
    msg = f"🔧 Zスコア逆張り通知テスト\n\n時刻: {datetime.now().strftime('%Y-%m-%d %H:%M')}\nステータス: 正常\n\n対象ペア:\n"
    for pair, params in config['pairs'].items():
        if params.get('enabled'):
            msg += f"  {pair} (W={params['window']}, Z=±{params['entry_z']}, R={params['risk_pct']}%)\n"

    ok = send_line(config, msg)
    if ok:
        print("✅ LINE通知テスト成功！")
    else:
        print("❌ LINE通知テスト失敗。設定を確認してください。")


# ============================================================
# コマンド: status（現在の状態表示）
# ============================================================
def cmd_status():
    state = load_state()
    positions = state.get('positions', {})

    print("=" * 50)
    print("  現在のポジション状態")
    print("=" * 50)

    if not positions:
        print("  ポジションなし")
    else:
        for pair, pos in positions.items():
            print(f"  {pair}: {pos['direction']} @ {pos['entry_price']:.4f} "
                  f"(Z={pos['entry_z']:+.2f}, {pos['entry_date']})")


# ============================================================
# コマンド: reset（ポジション状態リセット）
# ============================================================
def cmd_reset():
    save_state({"positions": {}})
    print("✅ ポジション状態をリセットしました")


# ============================================================
# メイン
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        print("コマンド:")
        print("  check  : シグナル確認 & LINE通知（毎日実行）")
        print("  test   : LINE通知テスト")
        print("  status : 現在のポジション状態表示")
        print("  reset  : ポジション状態リセット")
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == 'check':
        cmd_check()
    elif cmd == 'test':
        cmd_test()
    elif cmd == 'status':
        cmd_status()
    elif cmd == 'reset':
        cmd_reset()
    else:
        print(f"Unknown command: {cmd}")
