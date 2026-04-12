#!/usr/bin/env python3
"""
BB_Reversal_Martin 固定ロット運用バックテスト
FXTF 1万通貨(0.1lot)までスプレッド0の優遇活用を想定
"""
import numpy as np
import pandas as pd
import yfinance as yf

# EAパラメータ
RISK_PCT=0.8; RR_RATIO=2.0; BE_TRIGGER_RR=1.0; PARTIAL_RR=1.5; PARTIAL_PCT=0.5
TRAIL_ATR_MULT=0.5; MAX_HOLD_BARS=20; MARTIN=[1.0,1.5,2.0]
SL_MULTS={1:2.0,2:1.8,3:1.5}; ATR_FILTER_MULT=2.5
INITIAL_EQUITY=500000

# 固定ロット運用設定
FIXED_LOT=0.1              # 10,000通貨

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
    if j:tv=1000
    elif name in("EURUSD","GBPUSD","AUDUSD","NZDUSD"):tv=1500
    else:tv=1200
    return{"pip":0.01 if j else 0.0001,"pip_mult":100 if j else 10000,"spread":sp,"tick_value":tv}

def fetch(ticker):
    df=yf.download(ticker,period="60d",interval="15m",progress=False)
    if df.empty:return None
    df=df.droplevel("Ticker",axis=1) if isinstance(df.columns,pd.MultiIndex) else df
    df=df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close"})
    df=df[["open","high","low","close"]].dropna()
    df["time"]=df.index;df=df.reset_index(drop=True)
    return df

def calc_ind(df):
    c=df["close"]
    df["sma200"]=c.rolling(200).mean();df["sma50"]=c.rolling(50).mean();df["sma20"]=c.rolling(20).mean()
    df["sma200_up"]=df["sma200"]>df["sma200"].shift(5);df["sma50_up"]=df["sma50"]>df["sma50"].shift(5)
    d=c.diff();g=d.clip(lower=0).ewm(span=10,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(span=10,adjust=False).mean()
    df["rsi"]=100-(100/(1+g/l.replace(0,np.nan)))
    tr=pd.concat([df["high"]-df["low"],(df["high"]-c.shift(1)).abs(),(df["low"]-c.shift(1)).abs()],axis=1).max(axis=1)
    df["atr14"]=tr.rolling(14).mean();df["atr14_ma100"]=df["atr14"].rolling(100).mean()
    bm=c.rolling(20).mean();bs=c.rolling(20).std()
    df["bb_upper"]=bm+2.5*bs;df["bb_lower"]=bm-2.5*bs
    fm=c.rolling(10).mean();fs=c.rolling(10).std()
    df["fbb_upper"]=fm+2.0*fs;df["fbb_lower"]=fm-2.0*fs
    return df.dropna().reset_index(drop=True)

def check_sig(row,prev):
    if row["atr14"]>=row["atr14_ma100"]*ATR_FILTER_MULT:return 0
    c1,c2,o1,rsi=row["close"],prev["close"],row["open"],row["rsi"]
    u2,u5=row["sma200_up"],row["sma50_up"]
    if u2 and c2<=prev["bb_lower"] and c1>row["bb_lower"] and c1>o1 and rsi<42:return 1
    if not u2 and c2>=prev["bb_upper"] and c1<row["bb_upper"] and c1<o1 and rsi>58:return -1
    if u2 and u5 and c1<=row["fbb_lower"] and rsi<48:return 2
    if not u2 and not u5 and c1>=row["fbb_upper"] and rsi>52:return -2
    gap=abs(row["sma20"]-row["sma50"])
    if gap>=row["atr14"]*2:
        lo,hi=min(row["sma20"],row["sma50"]),max(row["sma20"],row["sma50"])
        if u2 and lo<=c1<=hi and 30<=rsi<=50:return 3
        if not u2 and lo<=c1<=hi and 50<=rsi<=70:return -3
    return 0

def run_bt(df,cfg,use_fixed_lot=True,zero_spread=True):
    pm=cfg["pip_mult"];tv=cfg["tick_value"];pu=cfg["pip"]
    eq=float(INITIAL_EQUITY);ep=eq;mdd=0.0;mseq=eq
    ms=0;cl=0;trades=[];monthly=[];cm=None
    i=1
    while i<len(df)-MAX_HOLD_BARS-1:
        row=df.iloc[i];prev=df.iloc[i-1]
        rm=pd.to_datetime(row["time"]).month
        if cm is None:cm=rm;mseq=eq
        elif rm!=cm:monthly.append({"m":cm,"p":(eq-mseq)/mseq*100});cm=rm;mseq=eq
        sig=check_sig(row,prev)
        if sig==0:i+=1;continue
        d=1 if sig>0 else -1;st=abs(sig);atr=row["atr14"]
        sld=atr*SL_MULTS.get(st,1.5);tpd=sld*RR_RATIO;spd=cfg["spread"]*pu
        ep_=row["close"];mm=MARTIN[min(ms,2)]

        # ロット計算（固定 or リスク%ベース）
        if use_fixed_lot:
            lots=round(FIXED_LOT*mm,2)  # マーチン倍率適用
        else:
            ra=eq*RISK_PCT/100.0
            sp_pips=sld*pm
            if sp_pips<=0:i+=1;continue
            raw=(ra*mm)/(sp_pips*tv)
            lots=max(0.01,int(raw/0.01)*0.01)

        slp=ep_-d*sld;tpp=ep_+d*tpd
        be=False;pc=False;rl=lots;rp=0.0;res="timeout";eb=i;pnl=0.0;to=False
        for j in range(1,MAX_HOLD_BARS+1):
            if i+j>=len(df):break
            b=df.iloc[i+j];ba=b["atr14"] if not np.isnan(b["atr14"]) else atr
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

        # スプレッドコスト（zero_spread=Trueで0）
        if not zero_spread:
            pnl-=cfg["spread"]*tv*lots

        if to or res=="BE":cl=0;ms=0
        elif pnl<0:
            cl+=1
            if cl>=3:ms=0;cl=0
            else:ms=min(cl,2)
        else:cl=0;ms=0
        eq+=pnl;ep=max(ep,eq);dd=(ep-eq)/ep*100 if ep>0 else 0;mdd=max(mdd,dd)
        trades.append({"result":res,"pnl":pnl,"lots":lots})
        i=eb+1
    if mseq>0 and cm is not None:monthly.append({"m":cm,"p":(eq-mseq)/mseq*100})
    return trades,mdd,monthly,eq

def summarize(trades,mdd,monthly,final_eq):
    if not trades:return None
    dt=pd.DataFrame(trades);n=len(dt)
    w=dt[dt["pnl"]>0];l=dt[dt["pnl"]<=0]
    wr=len(w)/n*100;tp=dt["pnl"].sum()
    gw=w["pnl"].sum() if len(w)>0 else 0;gl=abs(l["pnl"].sum()) if len(l)>0 else 0
    pf=gw/gl if gl>0 else float("inf")
    mm=pd.DataFrame(monthly)["p"].median() if monthly else 0
    return{"n":n,"wr":wr,"pnl":tp,"pf":pf,"dd":mdd,"mm":mm,"final":final_eq}

def main():
    print("="*100)
    print("BB_Reversal_Martin 固定ロット 0.1 (10,000通貨) 運用バックテスト")
    print(f"初期資金: ¥{INITIAL_EQUITY:,} | 固定LOT: {FIXED_LOT} | FXTF スプレッド0想定")
    print("="*100)

    all_data={}
    for pn,info in FXTF_PAIRS.items():
        cfg=pcfg(pn,info["sp"])
        print(f"  {pn:>8s}... ",end="",flush=True)
        df=fetch(info["t"])
        if df is None or len(df)<300:print("SKIP");continue
        df=calc_ind(df)
        all_data[pn]={"df":df,"cfg":cfg}
        print(f"{len(df)} bars")

    # 3モード比較
    modes=[
        ("変動ロット(通常)", {"use_fixed_lot":False,"zero_spread":False}),
        ("固定0.1lot(spread0)", {"use_fixed_lot":True,"zero_spread":True}),
        ("固定0.1lot(spread有)", {"use_fixed_lot":True,"zero_spread":False}),
    ]

    results={}
    for mode_name,kwargs in modes:
        print(f"\n[モード: {mode_name}]")
        for pn,data in all_data.items():
            tr,dd,mo,eq=run_bt(data["df"],data["cfg"],**kwargs)
            s=summarize(tr,dd,mo,eq)
            results.setdefault(pn,{})[mode_name]=s

    # 表示: 3モード比較
    print(f"\n{'='*100}")
    print(f"  【3モード比較: PF / 月利中央値】")
    print(f"{'='*100}")
    print(f"  {'ペア':>8s}  │ {'変動(通常)':>18s} │ {'固定0.1(SP=0)':>18s} │ {'固定0.1(SP有)':>18s} │ {'推奨'}")
    print(f"  {'-'*96}")
    rows=[]
    for pn in all_data:
        v=results[pn]["変動ロット(通常)"]
        f0=results[pn]["固定0.1lot(spread0)"]
        fs=results[pn]["固定0.1lot(spread有)"]
        def fmt(s):
            if not s:return "   -    /    -   "
            pf=f"{s['pf']:.2f}" if s['pf']<100 else "∞"
            return f"PF{pf:>4s} 月{s['mm']:>+5.1f}%"
        rec="固定0.1(SP=0)" if f0 and f0['pnl']>0 and (not v or f0['pnl']/INITIAL_EQUITY*100>v['pnl']/INITIAL_EQUITY*100*0.5) else ("変動" if v and v['pnl']>0 else "見送り")
        print(f"  {pn:>8s}  │ {fmt(v):>18s} │ {fmt(f0):>18s} │ {fmt(fs):>18s} │ {rec}")
        rows.append({"pair":pn,"v":v,"f0":f0,"fs":fs})

    # サマリー
    def sum_pnl(rows,key):
        return sum(r[key]["pnl"] for r in rows if r[key])
    def sum_monthly(rows,key,filter_plus=False):
        vals=[r[key]["mm"] for r in rows if r[key] and (r[key]["pnl"]>0 if filter_plus else True)]
        return sum(vals)

    print(f"\n{'='*100}")
    print(f"  【全体サマリー（3ヶ月合計）】")
    print(f"{'='*100}")
    for k,label in [("v","変動ロット"),("f0","固定0.1lot(SP=0)"),("fs","固定0.1lot(SP有)")]:
        tot=sum_pnl(rows,k)
        tot_m=sum_monthly(rows,k)/len(rows) if rows else 0
        print(f"  {label:>20s}: 合計¥{tot:+,.0f}  26ペア月利中央値合計 {tot_m:+.1f}%")

    # プラス収支ペアのみ抽出
    plus_pairs_f0=[r["pair"] for r in rows if r["f0"] and r["f0"]["pnl"]>0]
    print(f"\n  固定0.1(SP=0)でプラス: {len(plus_pairs_f0)}/26 ペア")
    print(f"    {', '.join(plus_pairs_f0)}")

    if plus_pairs_f0:
        plus_pnl=sum(r["f0"]["pnl"] for r in rows if r["pair"] in plus_pairs_f0 and r["f0"])
        plus_months_sum=sum(r["f0"]["mm"] for r in rows if r["pair"] in plus_pairs_f0 and r["f0"])
        print(f"  プラスペアだけ合計損益(3ヶ月): +¥{plus_pnl:,.0f}")
        print(f"  プラスペアだけ月利中央値合算: +{plus_months_sum:.1f}% / 月")
        print(f"  → 初期¥500k で推定: +¥{int(INITIAL_EQUITY*plus_months_sum/100):,} / 月 (単利)")

    # 複利シナリオ分析
    print(f"\n{'='*100}")
    print(f"  【複利シナリオ分析 (プラスペアのみ採用)】")
    print(f"{'='*100}")
    if plus_pairs_f0:
        monthly_yield=plus_months_sum  # 月利合算（単利扱い）
        print(f"  想定月利: {monthly_yield:+.1f}% (26ペアの固定0.1lot SP=0 での合算)\n")

        print(f"  {'':>6s}  {'A:固定0.1lot(単利)':>20s}  {'B:リスク%(複利)':>20s}  {'C:EA 2本並列(複利)':>20s}")
        print(f"  {'-'*76}")
        # A: 単利（毎月同じ金額増加）
        # B: 複利（前月残高 × (1+r)）。ただし0.1超のロットはスプレッド発生の影響で月利90%
        # C: EA 2本並列。実質0.2lot相当、どちらも1万以下なのでSP=0維持、月利2倍
        monthly_abs=INITIAL_EQUITY*monthly_yield/100  # A用
        eq_a=INITIAL_EQUITY; eq_b=INITIAL_EQUITY; eq_c=INITIAL_EQUITY
        for month in range(1,13):
            eq_a+=monthly_abs
            # B: lot 0.1を超えたらスプレッド発生で月利の90%に低減と仮定
            ratio_b=1 if eq_b<=INITIAL_EQUITY else 0.9  # 簡易
            eq_b=eq_b*(1+monthly_yield/100*ratio_b)
            # C: EA 2本並列 → 2倍月利（ただし相関あるので1.8倍で近似）
            eq_c=eq_c*(1+monthly_yield/100*1.8)
            if month in[1,3,6,12]:
                print(f"  {month}ヶ月後  ¥{eq_a:>16,.0f}  ¥{eq_b:>18,.0f}  ¥{eq_c:>18,.0f}")

    # おすすめ運用
    print(f"\n{'='*100}")
    print(f"  【推奨運用】")
    print(f"{'='*100}")
    print("  1. まずは固定0.1lotで全プラスペア運用（スプレッド0の恩恵最大化）")
    print("  2. 資金が倍になったら、EAインスタンスを2本並列（各0.1lot固定）で運用")
    print("     → 各1万通貨以下のためスプレッドはずっと0に保たれる")
    print("  3. さらに資金が増えたら並列数を増やす（3本、4本...）")
    print("  ※リスク%ベースはロット>0.1でスプレッドが発生し、FXTFの優遇が消える")

main()
