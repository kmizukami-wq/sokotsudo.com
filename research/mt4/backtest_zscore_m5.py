#!/usr/bin/env python3
"""
ZscoreScalper: M5データでの精密バックテスト
EA仕様を忠実に再現:
- Entry: M15バー確定時のみ (3本目のM5毎)
- Z Exit: M5バー毎 (Zscoreは直近30本M15の30本基準で計算)
- TP/SL: M5 high/lowで近似 (本来はティック)
- Timeout: 2時間 = 24本のM5
- FXTF 1万通貨(0.1lot)まで スプレッド0
"""
import numpy as np
import pandas as pd
import yfinance as yf

# ============= ZscoreScalper パラメータ =============
WINDOW=30            # Zscore計算用M15バー数
ENTRY_Z=0.51
EXIT_Z=0.5
STOP_Z=6.0
TIMEOUT_M5=24        # 2時間 = M5×24本
TP_PIPS=15
SL_PIPS=15
LOT=0.1
SPREAD_PIPS=0.0      # FXTF 1万通貨(0.1lot)までスプレッド0
TRADE_START_H=0      # UTC
TRADE_END_H=21
INITIAL_EQUITY=500000

FXTF_PAIRS = {
    "EURUSD":"EURUSD=X","USDJPY":"USDJPY=X","EURJPY":"EURJPY=X","GBPUSD":"GBPUSD=X",
    "GBPJPY":"GBPJPY=X","AUDJPY":"AUDJPY=X","NZDJPY":"NZDJPY=X","ZARJPY":"ZARJPY=X",
    "CHFJPY":"CHFJPY=X","USDCHF":"USDCHF=X","AUDUSD":"AUDUSD=X","EURGBP":"EURGBP=X",
    "NZDUSD":"NZDUSD=X","USDCAD":"USDCAD=X","CADJPY":"CADJPY=X","AUDCHF":"AUDCHF=X",
    "EURAUD":"EURAUD=X","AUDNZD":"AUDNZD=X","EURCAD":"EURCAD=X","EURCHF":"EURCHF=X",
    "GBPAUD":"GBPAUD=X","AUDCAD":"AUDCAD=X","EURNZD":"EURNZD=X","GBPCAD":"GBPCAD=X",
    "GBPCHF":"GBPCHF=X","GBPNZD":"GBPNZD=X",
}

def pcfg(name):
    # tick_value: 1.0lot(=100,000通貨)あたりの1pip=JPY
    j=name.endswith("JPY")
    if j: tv=1000
    elif name in("EURUSD","GBPUSD","AUDUSD","NZDUSD"): tv=1500
    else: tv=1200
    return{"pip":0.01 if j else 0.0001,"pip_mult":100 if j else 10000,"tick_value":tv}

def fetch_m5(ticker):
    df=yf.download(ticker,period="60d",interval="5m",progress=False)
    if df.empty:return None
    df=df.droplevel("Ticker",axis=1) if isinstance(df.columns,pd.MultiIndex) else df
    df=df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close"})
    df=df[["open","high","low","close"]].dropna()
    df["time"]=df.index; df=df.reset_index(drop=True)
    return df

def resample_m15(df_m5):
    """M5 → M15バー集約"""
    df=df_m5.copy()
    df["time"]=pd.to_datetime(df["time"]).dt.tz_localize(None)
    df=df.set_index("time")
    m15=df.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    m15=m15.reset_index()
    return m15

def calc_zscore_series(m15_df):
    """M15のZscore計算（各バーで直近WINDOW本）"""
    c=m15_df["close"]
    mean=c.rolling(WINDOW).mean()
    std=c.rolling(WINDOW).std()
    z=(c-mean)/std.replace(0,np.nan)
    m15_df["zscore"]=z
    return m15_df

def run_zscore_m5(df_m5,cfg):
    """M5データで精密バックテスト - EAの動作を忠実に再現"""
    pip_mult=cfg["pip_mult"]; tv=cfg["tick_value"]; pu=cfg["pip"]
    tp_dist=TP_PIPS*pu; sl_dist=SL_PIPS*pu

    # M15バー集約（過去のM15 close一覧を保持）
    m15=resample_m15(df_m5)
    m15["time"]=pd.to_datetime(m15["time"]).dt.tz_localize(None)
    m15_closes=m15["close"].values
    m15_times=m15["time"].values

    df_m5=df_m5.copy()
    df_m5["time"]=pd.to_datetime(df_m5["time"]).dt.tz_localize(None)
    df_m5["time_m15_start"]=df_m5["time"].dt.floor("15min")
    df_m5["m5_minute"]=df_m5["time"].dt.minute
    # M15バー確定直後の最初のM5 bar（分=0,15,30,45）でエントリー判定
    df_m5["is_m15_open"]=df_m5["m5_minute"].isin([0,15,30,45])

    # 各M5 barで、直前の完了M15バー(WINDOW本)と「現在のM15バーの進行中close」を使って
    # Z-scoreを動的計算
    def calc_zscore_at(m5_idx):
        """M5 bar m5_idxの時点でのZscore（CalcZscore()相当）"""
        row=df_m5.iloc[m5_idx]
        t=row["time"]
        # 完了したM15バーのcloseを探す（現在のM15開始前のバー）
        m15_start=row["time_m15_start"]
        # 完了M15の配列インデックス
        prev_m15_idx=np.searchsorted(m15_times,m15_start,side="left")-1
        if prev_m15_idx<WINDOW-2:
            return np.nan
        # 直近 WINDOW-1 本の完了M15 close + 現在のM5 close (bar 0 相当)
        hist=m15_closes[prev_m15_idx-WINDOW+2:prev_m15_idx+1]  # WINDOW-1本
        current=row["close"]
        arr=np.append(hist,current)  # WINDOW本
        mean=arr.mean()
        var=(arr**2).mean()-mean**2
        if var<=0: return 0
        std=np.sqrt(var)
        return (current-mean)/std

    # 事前計算
    print("  Z計算中...",end=" ",flush=True)
    df_m5["zscore"]=[calc_zscore_at(i) for i in range(len(df_m5))]
    print("done",end=" ")

    eq=float(INITIAL_EQUITY); ep=eq; mdd=0.0; mseq=eq
    trades=[]; monthly=[]; cm=None
    in_pos=False; direction=0; entry_price=0; entry_idx=0

    for i in range(WINDOW*3,len(df_m5)-1):  # WINDOW*3 = M15 WINDOW本の最低M5数
        bar=df_m5.iloc[i]
        t=pd.to_datetime(bar["time"])
        utc_hour=t.hour  # UTCと仮定（Yahoo M5はUTC）

        # 月替わり
        rm=t.month
        if cm is None: cm=rm; mseq=eq
        elif rm!=cm:
            monthly.append({"m":cm,"p":(eq-mseq)/mseq*100}); cm=rm; mseq=eq

        # ポジション保有中
        if in_pos:
            # TP/SL判定（M5 high/lowで近似）
            if direction==1:
                if bar["low"]<=entry_price-sl_dist:
                    pnl=-sl_dist*pip_mult*LOT*tv
                    pnl-=SPREAD_PIPS*tv*LOT
                    trades.append({"result":"SL","pnl":pnl})
                    eq+=pnl; ep=max(ep,eq)
                    dd=(ep-eq)/ep*100 if ep>0 else 0; mdd=max(mdd,dd)
                    in_pos=False; continue
                if bar["high"]>=entry_price+tp_dist:
                    pnl=tp_dist*pip_mult*LOT*tv
                    pnl-=SPREAD_PIPS*tv*LOT
                    trades.append({"result":"TP","pnl":pnl})
                    eq+=pnl; ep=max(ep,eq)
                    dd=(ep-eq)/ep*100 if ep>0 else 0; mdd=max(mdd,dd)
                    in_pos=False; continue
            else:
                if bar["high"]>=entry_price+sl_dist:
                    pnl=-sl_dist*pip_mult*LOT*tv
                    pnl-=SPREAD_PIPS*tv*LOT
                    trades.append({"result":"SL","pnl":pnl})
                    eq+=pnl; ep=max(ep,eq)
                    dd=(ep-eq)/ep*100 if ep>0 else 0; mdd=max(mdd,dd)
                    in_pos=False; continue
                if bar["low"]<=entry_price-tp_dist:
                    pnl=tp_dist*pip_mult*LOT*tv
                    pnl-=SPREAD_PIPS*tv*LOT
                    trades.append({"result":"TP","pnl":pnl})
                    eq+=pnl; ep=max(ep,eq)
                    dd=(ep-eq)/ep*100 if ep>0 else 0; mdd=max(mdd,dd)
                    in_pos=False; continue

            # Timeout（2h = 24本のM5）
            if i-entry_idx>=TIMEOUT_M5:
                pnl=direction*(bar["close"]-entry_price)*pip_mult*LOT*tv
                pnl-=SPREAD_PIPS*tv*LOT
                trades.append({"result":"TO","pnl":pnl})
                eq+=pnl; ep=max(ep,eq)
                dd=(ep-eq)/ep*100 if ep>0 else 0; mdd=max(mdd,dd)
                in_pos=False; continue

            # Z Exit (M5バー毎)
            z=bar["zscore"]
            if not np.isnan(z):
                if direction==1 and (z>-EXIT_Z or z<-STOP_Z):
                    pnl=(bar["close"]-entry_price)*pip_mult*LOT*tv
                    pnl-=SPREAD_PIPS*tv*LOT
                    trades.append({"result":"Z","pnl":pnl})
                    eq+=pnl; ep=max(ep,eq)
                    dd=(ep-eq)/ep*100 if ep>0 else 0; mdd=max(mdd,dd)
                    in_pos=False; continue
                if direction==-1 and (z<EXIT_Z or z>STOP_Z):
                    pnl=(entry_price-bar["close"])*pip_mult*LOT*tv
                    pnl-=SPREAD_PIPS*tv*LOT
                    trades.append({"result":"Z","pnl":pnl})
                    eq+=pnl; ep=max(ep,eq)
                    dd=(ep-eq)/ep*100 if ep>0 else 0; mdd=max(mdd,dd)
                    in_pos=False; continue

            # 取引時間外で強制決済
            if not(TRADE_START_H<=utc_hour<TRADE_END_H):
                pnl=direction*(bar["close"]-entry_price)*pip_mult*LOT*tv
                pnl-=SPREAD_PIPS*tv*LOT
                trades.append({"result":"OutTime","pnl":pnl})
                eq+=pnl; ep=max(ep,eq)
                dd=(ep-eq)/ep*100 if ep>0 else 0; mdd=max(mdd,dd)
                in_pos=False; continue

        # エントリー判定（M15バー確定直後のM5 & 取引時間内 & ノーポジ）
        if not in_pos and bar["is_m15_open"] and TRADE_START_H<=utc_hour<TRADE_END_H:
            z=bar["zscore"]
            if not np.isnan(z):
                if z>ENTRY_Z:  # SHORT
                    direction=-1; entry_price=bar["close"]; entry_idx=i; in_pos=True
                elif z<-ENTRY_Z:  # LONG
                    direction=1; entry_price=bar["close"]; entry_idx=i; in_pos=True

    if mseq>0 and cm is not None:
        monthly.append({"m":cm,"p":(eq-mseq)/mseq*100})
    return trades,mdd,monthly,eq

def summarize(trades,mdd,monthly,final_eq):
    if not trades:return None
    dt=pd.DataFrame(trades); n=len(dt)
    w=dt[dt["pnl"]>0]; l=dt[dt["pnl"]<=0]
    wr=len(w)/n*100; tp=dt["pnl"].sum()
    gw=w["pnl"].sum() if len(w)>0 else 0; gl=abs(l["pnl"].sum()) if len(l)>0 else 0
    pf=gw/gl if gl>0 else float("inf")
    mm=pd.DataFrame(monthly)["p"].median() if monthly else 0
    r=dt["result"].value_counts().to_dict()
    return{"n":n,"wr":wr,"pnl":tp,"pf":pf,"dd":mdd,"mm":mm,"r":r,"final":final_eq}

def main():
    print("="*105)
    print("ZscoreScalper M5精密バックテスト（FXTF 1万通貨=0.1lot, スプレッド0）")
    print(f"初期: ¥{INITIAL_EQUITY:,} | Entry:M15確定 / Z_Exit:M5 / TP/SL:M5 high-low")
    print("="*105)

    results=[]
    for pn,ticker in FXTF_PAIRS.items():
        cfg=pcfg(pn)
        print(f"  {pn}...",end=" ",flush=True)
        df=fetch_m5(ticker)
        if df is None or len(df)<500: print("SKIP"); continue
        trades,dd,monthly,eq=run_zscore_m5(df,cfg)
        s=summarize(trades,dd,monthly,eq)
        if s:
            results.append({"pair":pn,**s})
            pf_s=f"{s['pf']:.2f}" if s['pf']<100 else "∞"
            pnl_s=f"+¥{s['pnl']:,.0f}" if s['pnl']>=0 else f"-¥{abs(s['pnl']):,.0f}"
            print(f"N={s['n']:>4d} PF={pf_s:>5s} WR={s['wr']:.1f}% DD={s['dd']:.1f}% PnL={pnl_s}")

    if not results: print("No results"); return
    df_r=pd.DataFrame(results).sort_values("pf",ascending=False)

    print(f"\n{'='*105}")
    print(f"  【ZscoreScalper 成績一覧】M5精密 / PF降順")
    print(f"{'='*105}")
    print(f"  {'ペア':>7s}  {'PF':>5s}  {'勝率':>6s}  {'最大DD':>7s}  {'純損益':>12s}  {'月利':>6s}  {'N':>5s}  {'決済内訳':>30s}  {'判定'}")
    print(f"  {'-'*100}")
    for _,r in df_r.iterrows():
        pf_s=f"{r['pf']:.2f}" if r['pf']<100 else "∞"
        pnl_s=f"+¥{r['pnl']:,.0f}" if r['pnl']>=0 else f"-¥{abs(r['pnl']):,.0f}"
        res=" ".join(f"{k}:{v}" for k,v in sorted(r['r'].items()))
        if r['pf']>=1.3 and r['dd']<15: v="★稼働推奨"
        elif r['pf']>=1.2 and r['dd']<20: v="◎有望"
        elif r['pf']>=1.1: v="○条件付き"
        elif r['pf']>=1.0: v="△様子見"
        else: v="×見送り"
        print(f"  {r['pair']:>7s}  {pf_s:>5s}  {r['wr']:>5.1f}%  {r['dd']:>6.1f}%  {pnl_s:>12s}  {r['mm']:>+5.1f}%  {r['n']:>5d}  {res:>30s}  {v}")

    profitable=df_r[df_r["pnl"]>0]
    print(f"\n{'='*105}")
    print(f"  【サマリー】")
    print(f"{'='*105}")
    print(f"  全ペア: {len(df_r)} | プラス: {len(profitable)} | PF≥1.2&DD<20%: {len(df_r[(df_r['pf']>=1.2)&(df_r['dd']<20)])}")
    print(f"  合計損益: ¥{df_r['pnl'].sum():+,.0f}")

main()
