#!/usr/bin/env python3
"""
BB_Reversal_Martin: FXTF全通貨ペア H1バックテスト（約2.8年）
M15ロジックをH1に適応: インジケータ期間を1/4に調整
"""
import numpy as np
import pandas as pd
import yfinance as yf

RISK_PCT=0.8; RR_RATIO=2.0; BE_TRIGGER_RR=1.0; PARTIAL_RR=1.5; PARTIAL_PCT=0.5
TRAIL_ATR_MULT=0.5; MARTIN=[1.0,1.5,2.0]; ATR_FILTER_MULT=2.5; INITIAL_EQUITY=500000
SL_MULTS={1:2.0,2:1.8,3:1.5}

# H1用: 最大保有バー = M15の20本 ÷ 4 = 5本
MAX_HOLD_BARS=5

# インジケータ期間もM15→H1で1/4に調整
SMA_LONG=50   # M15:200 → H1:50
SMA_MED=12    # M15:50  → H1:12
SMA_SHORT=5   # M15:20  → H1:5
RSI_PERIOD=10 # RSIはそのまま（感度調整不要）
ATR_PERIOD=14 # ATRもそのまま
ATR_MA=25     # M15:100 → H1:25
BB_PERIOD=5   # M15:20  → H1:5
BB_DEV=2.5
FBB_PERIOD=3  # M15:10  → H1:3 (min 2)
FBB_DEV=2.0
SMA_DIR_SHIFT=2 # M15:5 → H1:2 (方向判定シフト)

FXTF_PAIRS = {
    "EURUSD":{"t":"EURUSD=X","sp":0.3},"USDJPY":{"t":"USDJPY=X","sp":0.3},
    "EURJPY":{"t":"EURJPY=X","sp":0.5},"GBPUSD":{"t":"GBPUSD=X","sp":0.7},
    "GBPJPY":{"t":"GBPJPY=X","sp":0.7},"AUDJPY":{"t":"AUDJPY=X","sp":0.5},
    "NZDJPY":{"t":"NZDJPY=X","sp":0.8},"ZARJPY":{"t":"ZARJPY=X","sp":1.5},
    "CHFJPY":{"t":"CHFJPY=X","sp":1.5},"USDCHF":{"t":"USDCHF=X","sp":1.5},
    "AUDUSD":{"t":"AUDUSD=X","sp":0.5},"EURGBP":{"t":"EURGBP=X","sp":1.0},
    "NZDUSD":{"t":"NZDUSD=X","sp":1.0},"USDCAD":{"t":"USDCAD=X","sp":1.5},
    "CADJPY":{"t":"CADJPY=X","sp":1.5},"AUDCHF":{"t":"AUDCHF=X","sp":2.0},
    "EURAUD":{"t":"EURAUD=X","sp":1.5},"AUDNZD":{"t":"AUDNZD=X","sp":2.0},
    "EURCAD":{"t":"EURCAD=X","sp":2.0},"EURCHF":{"t":"EURCHF=X","sp":1.5},
    "GBPAUD":{"t":"GBPAUD=X","sp":2.0},"AUDCAD":{"t":"AUDCAD=X","sp":2.0},
    "EURNZD":{"t":"EURNZD=X","sp":2.5},"GBPCAD":{"t":"GBPCAD=X","sp":2.5},
    "GBPCHF":{"t":"GBPCHF=X","sp":2.0},"GBPNZD":{"t":"GBPNZD=X","sp":3.0},
}

def pcfg(name,sp):
    j=name.endswith("JPY")
    return{"pip":0.01 if j else 0.0001,"pip_mult":100 if j else 10000,"spread":sp,"tick_value":100 if j else 150}

def fetch(ticker):
    df=yf.download(ticker,period="730d",interval="1h",progress=False)
    if df.empty:return None
    df=df.droplevel("Ticker",axis=1) if isinstance(df.columns,pd.MultiIndex) else df
    df=df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close"})
    df=df[["open","high","low","close"]].dropna()
    df["time"]=df.index;df=df.reset_index(drop=True)
    return df

def calc_ind(df):
    c=df["close"]
    df["sma_long"]=c.rolling(SMA_LONG).mean()
    df["sma_med"]=c.rolling(SMA_MED).mean()
    df["sma_short"]=c.rolling(SMA_SHORT).mean()
    df["sma_long_up"]=df["sma_long"]>df["sma_long"].shift(SMA_DIR_SHIFT)
    df["sma_med_up"]=df["sma_med"]>df["sma_med"].shift(SMA_DIR_SHIFT)
    d=c.diff();g=d.clip(lower=0).ewm(span=RSI_PERIOD,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(span=RSI_PERIOD,adjust=False).mean()
    df["rsi"]=100-(100/(1+g/l.replace(0,np.nan)))
    tr=pd.concat([df["high"]-df["low"],(df["high"]-c.shift(1)).abs(),(df["low"]-c.shift(1)).abs()],axis=1).max(axis=1)
    df["atr"]=tr.rolling(ATR_PERIOD).mean()
    df["atr_ma"]=df["atr"].rolling(ATR_MA).mean()
    bm=c.rolling(BB_PERIOD).mean();bs=c.rolling(BB_PERIOD).std()
    df["bb_upper"]=bm+BB_DEV*bs;df["bb_lower"]=bm-BB_DEV*bs
    fp=max(FBB_PERIOD,2);fm=c.rolling(fp).mean();fs=c.rolling(fp).std()
    df["fbb_upper"]=fm+FBB_DEV*fs;df["fbb_lower"]=fm-FBB_DEV*fs
    return df.dropna().reset_index(drop=True)

def check_sig(row,prev,enabled=(1,2,3)):
    if row["atr"]>=row["atr_ma"]*ATR_FILTER_MULT:return 0
    c1,c2,rsi=row["close"],prev["close"],row["rsi"]
    ul,um=row["sma_long_up"],row["sma_med_up"]
    if 1 in enabled:
        if ul and c2<=prev["bb_lower"] and c1>row["bb_lower"] and rsi<42:return 1
        if not ul and c2>=prev["bb_upper"] and c1<row["bb_upper"] and rsi>58:return -1
    if 2 in enabled:
        if ul and um and c1<=row["fbb_lower"] and rsi<48:return 2
        if not ul and not um and c1>=row["fbb_upper"] and rsi>52:return -2
    if 3 in enabled:
        gap=abs(row["sma_short"]-row["sma_med"])
        if gap>=row["atr"]*2:
            lo,hi=min(row["sma_short"],row["sma_med"]),max(row["sma_short"],row["sma_med"])
            if ul and lo<=c1<=hi and 30<=rsi<=50:return 3
            if not ul and lo<=c1<=hi and 50<=rsi<=70:return -3
    return 0

def run_bt(df,cfg,enabled=(1,2,3)):
    pm=cfg["pip_mult"];tv=cfg["tick_value"];pu=cfg["pip"]
    eq=float(INITIAL_EQUITY);ep=eq;mdd=0.0;mseq=eq
    ms=0;cl=0;trades=[];monthly=[];cm=None
    i=1
    while i<len(df)-MAX_HOLD_BARS-1:
        row=df.iloc[i];prev=df.iloc[i-1]
        rm=pd.to_datetime(row["time"]).month
        ry=pd.to_datetime(row["time"]).year
        rm_key=(ry,rm)
        if cm is None:cm=rm_key;mseq=eq
        elif rm_key!=cm:
            monthly.append({"ym":cm,"pnl_pct":(eq-mseq)/mseq*100});cm=rm_key;mseq=eq
        sig=check_sig(row,prev,enabled=enabled)
        if sig==0:i+=1;continue
        d=1 if sig>0 else -1;st=abs(sig);atr=row["atr"]
        slm=SL_MULTS.get(st,1.5);sld=atr*slm;tpd=sld*RR_RATIO;spd=cfg["spread"]*pu
        ep_=row["close"];ra=eq*RISK_PCT/100.0;mm=MARTIN[min(ms,2)]
        sp=sld*pm
        if sp<=0:i+=1;continue
        lots=max(0.01,int((ra*mm)/(sp*tv)/0.01)*0.01)
        slp=ep_-d*sld;tpp=ep_+d*tpd
        be=False;pc=False;rl=lots;rp=0.0;res="timeout";eb=i;pnl=0.0;to=False
        for j in range(1,MAX_HOLD_BARS+1):
            if i+j>=len(df):break
            b=df.iloc[i+j];ba=b["atr"] if not np.isnan(b["atr"]) else atr
            if d==1:
                if not be and(b["high"]-ep_)>=sld*BE_TRIGGER_RR:slp=ep_+spd;be=True
                if not pc and(b["high"]-ep_)>=sld*PARTIAL_RR:
                    cl_=round(rl*PARTIAL_PCT,2)
                    if cl_>=0.01 and(rl-cl_)>=0.01:
                        rp+=(sld*PARTIAL_RR)*pm*cl_*tv;rl=round(rl-cl_,2);pc=True
                        ts=ep_+sld*PARTIAL_RR-ba*TRAIL_ATR_MULT
                        if ts>slp:slp=ts
                if pc:
                    ts=b["high"]-ba*TRAIL_ATR_MULT
                    if ts>slp:slp=ts
                if b["low"]<=slp:pnl=(slp-ep_)*pm*rl*tv+rp;res="BE" if be else "SL";eb=i+j;break
                if b["high"]>=tpp:pnl=tpd*pm*rl*tv+rp;res="TP";eb=i+j;break
            else:
                if not be and(ep_-b["low"])>=sld*BE_TRIGGER_RR:slp=ep_-spd;be=True
                if not pc and(ep_-b["low"])>=sld*PARTIAL_RR:
                    cl_=round(rl*PARTIAL_PCT,2)
                    if cl_>=0.01 and(rl-cl_)>=0.01:
                        rp+=(sld*PARTIAL_RR)*pm*cl_*tv;rl=round(rl-cl_,2);pc=True
                        ts=ep_-sld*PARTIAL_RR+ba*TRAIL_ATR_MULT
                        if ts<slp:slp=ts
                if pc:
                    ts=b["low"]+ba*TRAIL_ATR_MULT
                    if ts<slp:slp=ts
                if b["high"]>=slp:pnl=(ep_-slp)*pm*rl*tv+rp;res="BE" if be else "SL";eb=i+j;break
                if b["low"]<=tpp:pnl=tpd*pm*rl*tv+rp;res="TP";eb=i+j;break
        else:
            ex=df.iloc[min(i+MAX_HOLD_BARS,len(df)-1)]["close"]
            pnl=d*(ex-ep_)*pm*rl*tv+rp;to=True
        pnl-=cfg["spread"]*tv*lots
        if to or res=="BE":cl=0;ms=0
        elif pnl<0:
            cl+=1
            if cl>=3:ms=0;cl=0
            else:ms=min(cl,2)
        else:cl=0;ms=0
        eq+=pnl;ep=max(ep,eq);dd=(ep-eq)/ep*100 if ep>0 else 0;mdd=max(mdd,dd)
        trades.append({"result":res,"pnl":pnl,"be":be,"partial":pc})
        i=eb+1
    if mseq>0 and cm is not None:
        monthly.append({"ym":cm,"pnl_pct":(eq-mseq)/mseq*100})
    return trades,mdd,monthly

def summarize(trades,mdd,monthly):
    if not trades:return None
    dt=pd.DataFrame(trades);n=len(dt)
    w=dt[dt["pnl"]>0];l=dt[dt["pnl"]<=0]
    wr=len(w)/n*100;tp=dt["pnl"].sum()
    gw=w["pnl"].sum() if len(w)>0 else 0;gl=abs(l["pnl"].sum()) if len(l)>0 else 0
    pf=gw/gl if gl>0 else float("inf")
    mm=pd.DataFrame(monthly)["pnl_pct"].median() if monthly else 0
    r=dt["result"].value_counts().to_dict()
    return{"n":n,"wr":wr,"pnl":tp,"pf":pf,"dd":mdd,"mm":mm,"r":r,"monthly":monthly}

def main():
    print("="*100)
    print("BB_Reversal_Martin: FXTF全26通貨ペア H1バックテスト（~2.8年 実データ）")
    print(f"初期: ¥{INITIAL_EQUITY:,} | リスク: {RISK_PCT}% | RR: {RR_RATIO} | H1用パラメータ調整済み")
    print("="*100)

    results=[]
    for pn,info in FXTF_PAIRS.items():
        cfg=pcfg(pn,info["sp"])
        print(f"  {pn:>8s}...",end=" ",flush=True)
        df=fetch(info["t"])
        if df is None or len(df)<200:print("SKIP");continue
        df=calc_ind(df)
        dates=f"{df['time'].iloc[0].date()} ~ {df['time'].iloc[-1].date()}"
        tr,dd,mo=run_bt(df,cfg)
        s=summarize(tr,dd,mo)
        if s:
            results.append({"pair":pn,"sp":info["sp"],**s})
            pf_s=f"{s['pf']:.2f}" if s['pf']<100 else "∞"
            print(f"N={s['n']:>4d} PF={pf_s:>5s} WR={s['wr']:.1f}% DD={s['dd']:.1f}% PnL={s['pnl']:+,.0f} ({dates})")
        else:print("NO SIGNAL")

    if not results:print("No results");return
    df_r=pd.DataFrame(results).sort_values("pf",ascending=False)

    print(f"\n{'='*100}")
    print(f"  【全26通貨ペア 成績一覧】PF降順 / H1 約2.8年")
    print(f"{'='*100}")
    print(f"  {'ペア':>8s}  {'SP':>4s}  {'PF':>5s}  {'勝率':>6s}  {'最大DD':>7s}  {'純損益':>12s}  {'月利中央':>7s}  {'N':>5s}  {'判定'}")
    print(f"  {'-'*75}")
    for _,r in df_r.iterrows():
        pf_s=f"{r['pf']:.2f}" if r['pf']<100 else "∞"
        pnl_s=f"+¥{r['pnl']:,.0f}" if r['pnl']>=0 else f"-¥{abs(r['pnl']):,.0f}"
        if r['pf']>=1.3 and r['dd']<10:v="★稼働推奨"
        elif r['pf']>=1.2 and r['dd']<15:v="◎有望"
        elif r['pf']>=1.1:v="○条件付き"
        elif r['pf']>=1.0:v="△様子見"
        else:v="×見送り"
        print(f"  {r['pair']:>8s}  {r['sp']:>4.1f}  {pf_s:>5s}  {r['wr']:>5.1f}%  {r['dd']:>6.1f}%  {pnl_s:>12s}  {r['mm']:>+6.1f}%  {r['n']:>5d}  {v}")

    # 月利テーブル（上位ペア）
    top=df_r.head(10)
    print(f"\n{'='*100}")
    print(f"  【月利(%) 上位10ペア - 直近12ヶ月】")
    print(f"{'='*100}")
    hdr=f"  {'年月':>7s}"+"".join(f" {r['pair']:>8s}" for _,r in top.iterrows())
    print(hdr)
    print(f"  {'-'*(7+9*len(top))}")

    pair_mo={}
    for _,r in top.iterrows():
        pair_mo[r["pair"]]={m["ym"]:m["pnl_pct"] for m in r["monthly"]} if r.get("monthly") else {}
    ams=sorted(set(m for pm in pair_mo.values() for m in pm))
    # 直近12ヶ月のみ
    recent=ams[-12:] if len(ams)>12 else ams
    for ym in recent:
        row=f"  {ym[0]}/{ym[1]:02d}"
        for _,r in top.iterrows():
            v=pair_mo[r["pair"]].get(ym)
            row+=f" {v:>+7.1f}%" if v is not None else f" {'---':>8s}"
        print(row)

    profitable=df_r[df_r["pnl"]>0]
    print(f"\n{'='*100}")
    print(f"  【サマリー】")
    print(f"{'='*100}")
    print(f"  全ペア数:         {len(df_r)}")
    print(f"  プラス収支:       {len(profitable)} ペア")
    print(f"  PF≥1.3 & DD<10%: {len(df_r[(df_r['pf']>=1.3)&(df_r['dd']<10)])} ペア (稼働推奨)")
    print(f"  PF≥1.2 & DD<15%: {len(df_r[(df_r['pf']>=1.2)&(df_r['dd']<15)])} ペア (有望以上)")
    print(f"  PF≥1.1:          {len(df_r[df_r['pf']>=1.1])} ペア")
    print(f"  合計損益:         ¥{df_r['pnl'].sum():+,.0f}")

main()
