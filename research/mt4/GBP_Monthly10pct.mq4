//+------------------------------------------------------------------+
//| GBP_Monthly10pct.mq4                                             |
//| GBP/JPY・GBP/USD 月利10%トレンドフォロー + ロンドンBK EA         |
//| 15分足専用 / 2通貨ペア対応                                        |
//+------------------------------------------------------------------+
//| 戦略:                                                             |
//|   A) トレンドフォロー: EMA50/200クロス + ATRトレイリング           |
//|   B) ロンドンBK: アジアレンジBK (07:00 UTC～)                     |
//|   バックテスト最良結果:                                            |
//|     GBP/USD トレンドフォロー EMA50/200 ATR×3.5                    |
//|     → 月利17.82%, MaxDD 19.6%, PF 1.34                           |
//+------------------------------------------------------------------+
#property copyright "sokotsudo research"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| 外部パラメータ                                                    |
//+------------------------------------------------------------------+
// --- 共通 ---
input int    MagicNumber       = 20001;   // MagicNumber
input double MaxRiskPercent    = 2.0;     // 1トレードリスク (%)
input int    MaxPositions      = 2;       // 最大同時ポジション数
input double MonthlyDDLimit    = 15.0;    // 月間DD制限 (%)
input int    ServerGMTOffset   = 9;       // サーバーGMTオフセット (FXTF=9)
input int    Slippage          = 10;      // スリッページ (points)

// --- トレンドフォロー ---
input bool   EnableTrend       = true;    // トレンドフォロー有効
input int    FastEMA           = 50;      // 短期EMA期間
input int    SlowEMA           = 200;     // 長期EMA期間
input double ATR_Trail_Mult    = 3.5;     // ATRトレイリング倍率
input int    ATR_Period        = 14;      // ATR期間
input int    TrendStartHour    = 7;       // トレンド取引開始 (UTC)
input int    TrendEndHour      = 21;      // トレンド取引終了 (UTC)

// --- ロンドンブレイクアウト ---
input bool   EnableLondonBK    = true;    // ロンドンBK有効
input int    AsianStartHour    = 0;       // アジアレンジ開始 (UTC)
input int    AsianEndHour      = 7;       // アジアレンジ終了 (UTC)
input int    LondonEndHour     = 16;      // ロンドンセッション終了 (UTC)
input double BK_TP_Mult        = 1.5;     // TP倍率 (レンジ幅に対して)
input double BK_RiskPercent    = 2.0;     // BKリスク (%)
input double MinRangePips      = 20.0;    // 最小レンジ幅 (pips)
input double MaxRangePips      = 120.0;   // 最大レンジ幅 (pips)

//+------------------------------------------------------------------+
//| 定数                                                              |
//+------------------------------------------------------------------+
#define MAGIC_TREND  0
#define MAGIC_BK     1

//+------------------------------------------------------------------+
//| グローバル変数                                                    |
//+------------------------------------------------------------------+
datetime g_lastBarTime     = 0;
double   g_monthStartEq    = 0;
int      g_currentMonth    = 0;
bool     g_monthStopped    = false;

// ロンドンBK用
double   g_asianHigh       = 0;
double   g_asianLow        = 0;
bool     g_asianRangeSet   = false;
bool     g_bkTriggered     = false;
int      g_bkDate          = 0;

// トレンドフォロー用
double   g_trendTrailStop  = 0;
int      g_trendDirection  = 0;  // 1=long, -1=short, 0=flat(signal)

//+------------------------------------------------------------------+
//| UTC時間取得                                                       |
//+------------------------------------------------------------------+
int GetUTCHour()
{
   datetime utcTime = TimeCurrent() - ServerGMTOffset * 3600;
   MqlDateTime dt;
   TimeToStruct(utcTime, dt);
   return dt.hour;
}

int GetUTCDay()
{
   datetime utcTime = TimeCurrent() - ServerGMTOffset * 3600;
   MqlDateTime dt;
   TimeToStruct(utcTime, dt);
   return dt.day;
}

//+------------------------------------------------------------------+
//| 新バー検知                                                        |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   datetime currentBarTime = iTime(NULL, PERIOD_M15, 0);
   if(currentBarTime != g_lastBarTime)
   {
      g_lastBarTime = currentBarTime;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| pip値取得                                                         |
//+------------------------------------------------------------------+
double GetPipValue()
{
   if(Digits == 3 || Digits == 5)
      return Point * 10;
   return Point;
}

//+------------------------------------------------------------------+
//| 自分のポジション数                                                |
//+------------------------------------------------------------------+
int CountMyOrders(int subMagic = -1)
{
   int count = 0;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(OrderSymbol() != Symbol())
         continue;
      int om = OrderMagicNumber();
      if(subMagic >= 0)
      {
         if(om == MagicNumber + subMagic)
            count++;
      }
      else
      {
         if(om >= MagicNumber && om <= MagicNumber + 10)
            count++;
      }
   }
   return count;
}

//+------------------------------------------------------------------+
//| 自分のポジションチケット取得                                      |
//+------------------------------------------------------------------+
int FindMyOrder(int subMagic)
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber + subMagic)
         return OrderTicket();
   }
   return -1;
}

//+------------------------------------------------------------------+
//| ロットサイズ計算                                                  |
//+------------------------------------------------------------------+
double CalcLotSize(double slDistance, double riskPct)
{
   if(slDistance <= 0) return MarketInfo(Symbol(), MODE_MINLOT);

   double equity     = AccountEquity();
   double riskAmount = equity * riskPct / 100.0;
   double tickValue  = MarketInfo(Symbol(), MODE_TICKVALUE);
   double tickSize   = MarketInfo(Symbol(), MODE_TICKSIZE);
   double minLot     = MarketInfo(Symbol(), MODE_MINLOT);
   double maxLot     = MarketInfo(Symbol(), MODE_MAXLOT);
   double lotStep    = MarketInfo(Symbol(), MODE_LOTSTEP);

   if(tickValue <= 0 || tickSize <= 0) return minLot;

   double slCostPerLot = slDistance / tickSize * tickValue;
   if(slCostPerLot <= 0) return minLot;

   double lots = riskAmount / slCostPerLot;

   // ロットステップに丸める
   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(lots, minLot);
   lots = MathMin(lots, maxLot);

   return lots;
}

//+------------------------------------------------------------------+
//| 月間DD チェック                                                   |
//+------------------------------------------------------------------+
bool CheckMonthlyDD()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int month = dt.mon;

   if(month != g_currentMonth)
   {
      g_currentMonth = month;
      g_monthStartEq = AccountEquity();
      g_monthStopped = false;
   }

   if(g_monthStartEq > 0)
   {
      double dd = (g_monthStartEq - AccountEquity()) / g_monthStartEq * 100;
      if(dd >= MonthlyDDLimit)
      {
         if(!g_monthStopped)
         {
            Print("[DD STOP] Monthly DD ", DoubleToString(dd, 1), "% >= ",
                  DoubleToString(MonthlyDDLimit, 1), "%. Trading stopped for this month.");
            g_monthStopped = true;
         }
         return false;
      }
   }
   return true;
}

//+------------------------------------------------------------------+
//| Init                                                              |
//+------------------------------------------------------------------+
int OnInit()
{
   g_monthStartEq = AccountEquity();
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   g_currentMonth = dt.mon;

   Print("GBP Monthly10% EA started on ", Symbol(),
         " | Trend=", EnableTrend, " LondonBK=", EnableLondonBK,
         " | Risk=", MaxRiskPercent, "%");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!IsNewBar()) return;

   // DD制限チェック
   if(!CheckMonthlyDD()) return;

   int utcHour = GetUTCHour();
   int utcDay  = GetUTCDay();

   // === トレンドフォロー ===
   if(EnableTrend)
      ProcessTrendFollow(utcHour);

   // === ロンドンブレイクアウト ===
   if(EnableLondonBK)
      ProcessLondonBK(utcHour, utcDay);
}

//+------------------------------------------------------------------+
//| トレンドフォロー処理                                              |
//+------------------------------------------------------------------+
void ProcessTrendFollow(int utcHour)
{
   // 取引時間チェック
   if(utcHour < TrendStartHour || utcHour >= TrendEndHour)
      return;

   double emaFast = iMA(NULL, PERIOD_M15, FastEMA, 0, MODE_EMA, PRICE_CLOSE, 1);
   double emaSlow = iMA(NULL, PERIOD_M15, SlowEMA, 0, MODE_EMA, PRICE_CLOSE, 1);
   double atr     = iATR(NULL, PERIOD_M15, ATR_Period, 1);
   double close1  = iClose(NULL, PERIOD_M15, 1);
   double high1   = iHigh(NULL, PERIOD_M15, 1);
   double low1    = iLow(NULL, PERIOD_M15, 1);

   if(atr <= 0) return;

   int signal = 0;
   if(emaFast > emaSlow) signal = 1;
   if(emaFast < emaSlow) signal = -1;

   int ticket = FindMyOrder(MAGIC_TREND);

   // ポジションあり: トレイリングストップ更新
   if(ticket > 0)
   {
      if(!OrderSelect(ticket, SELECT_BY_TICKET)) return;
      int orderType = OrderType();

      if(orderType == OP_BUY)
      {
         double newStop = close1 - atr * ATR_Trail_Mult;
         if(newStop > g_trendTrailStop)
            g_trendTrailStop = newStop;

         // SLヒット or シグナル反転
         if(low1 <= g_trendTrailStop || signal == -1)
         {
            if(!OrderClose(ticket, OrderLots(), Bid, Slippage, clrRed))
               Print("[TREND] Close BUY failed: ", GetLastError());
            else
               Print("[TREND] Closed BUY at ", Bid);
         }
         else
         {
            // SL更新（ブローカーに設定）
            double currentSL = OrderStopLoss();
            double pipVal = GetPipValue();
            if(g_trendTrailStop > currentSL + pipVal)
            {
               if(!OrderModify(ticket, OrderOpenPrice(), NormalizeDouble(g_trendTrailStop, Digits), OrderTakeProfit(), 0))
                  Print("[TREND] Modify SL failed: ", GetLastError());
            }
         }
      }
      else if(orderType == OP_SELL)
      {
         double newStop = close1 + atr * ATR_Trail_Mult;
         if(newStop < g_trendTrailStop || g_trendTrailStop == 0)
            g_trendTrailStop = newStop;

         if(high1 >= g_trendTrailStop || signal == 1)
         {
            if(!OrderClose(ticket, OrderLots(), Ask, Slippage, clrBlue))
               Print("[TREND] Close SELL failed: ", GetLastError());
            else
               Print("[TREND] Closed SELL at ", Ask);
         }
         else
         {
            double currentSL = OrderStopLoss();
            double pipVal = GetPipValue();
            if(g_trendTrailStop < currentSL - pipVal || currentSL == 0)
            {
               if(!OrderModify(ticket, OrderOpenPrice(), NormalizeDouble(g_trendTrailStop, Digits), OrderTakeProfit(), 0))
                  Print("[TREND] Modify SL failed: ", GetLastError());
            }
         }
      }
      return;
   }

   // ポジションなし: 新規エントリー
   if(signal == 0) return;
   if(CountMyOrders() >= MaxPositions) return;

   double slDist = atr * ATR_Trail_Mult;
   double lots = CalcLotSize(slDist, MaxRiskPercent);

   if(signal == 1)
   {
      double sl = Ask - slDist;
      g_trendTrailStop = sl;
      int t = OrderSend(Symbol(), OP_BUY, lots, Ask, Slippage,
                        NormalizeDouble(sl, Digits), 0,
                        "Trend BUY", MagicNumber + MAGIC_TREND, 0, clrBlue);
      if(t > 0)
         Print("[TREND] BUY ", lots, " lots at ", Ask, " SL=", sl);
      else
         Print("[TREND] BUY failed: ", GetLastError());
   }
   else
   {
      double sl = Bid + slDist;
      g_trendTrailStop = sl;
      int t = OrderSend(Symbol(), OP_SELL, lots, Bid, Slippage,
                        NormalizeDouble(sl, Digits), 0,
                        "Trend SELL", MagicNumber + MAGIC_TREND, 0, clrRed);
      if(t > 0)
         Print("[TREND] SELL ", lots, " lots at ", Bid, " SL=", sl);
      else
         Print("[TREND] SELL failed: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| ロンドンブレイクアウト処理                                        |
//+------------------------------------------------------------------+
void ProcessLondonBK(int utcHour, int utcDay)
{
   // 日付が変わったらリセット
   if(utcDay != g_bkDate)
   {
      g_bkDate = utcDay;
      g_asianRangeSet = false;
      g_bkTriggered = false;
      g_asianHigh = 0;
      g_asianLow = 999999;
   }

   // アジアレンジ収集 (AsianStartHour ~ AsianEndHour-1 UTC)
   if(utcHour >= AsianStartHour && utcHour < AsianEndHour)
   {
      double h = iHigh(NULL, PERIOD_M15, 1);
      double l = iLow(NULL, PERIOD_M15, 1);
      if(h > g_asianHigh) g_asianHigh = h;
      if(l < g_asianLow)  g_asianLow = l;
      g_asianRangeSet = true;
      return;
   }

   // ロンドンセッション
   if(!g_asianRangeSet || g_bkTriggered) return;
   if(utcHour < AsianEndHour || utcHour >= LondonEndHour)
   {
      // セッション終了 - ポジションがあれば決済
      if(utcHour >= LondonEndHour)
      {
         int ticket = FindMyOrder(MAGIC_BK);
         if(ticket > 0)
         {
            if(OrderSelect(ticket, SELECT_BY_TICKET))
            {
               if(OrderType() == OP_BUY)
                  OrderClose(ticket, OrderLots(), Bid, Slippage, clrGray);
               else
                  OrderClose(ticket, OrderLots(), Ask, Slippage, clrGray);
               Print("[BK] Session end close");
            }
         }
      }
      return;
   }

   // レンジ幅チェック
   double pipVal = GetPipValue();
   double rangeWidth = g_asianHigh - g_asianLow;
   double rangePips = rangeWidth / pipVal;

   if(rangePips < MinRangePips || rangePips > MaxRangePips) return;
   if(CountMyOrders() >= MaxPositions) return;

   double close1 = iClose(NULL, PERIOD_M15, 1);
   double high1  = iHigh(NULL, PERIOD_M15, 1);
   double low1   = iLow(NULL, PERIOD_M15, 1);
   double spread = Ask - Bid;

   double tp = rangeWidth * BK_TP_Mult;
   double sl = rangeWidth;

   // ロングブレイクアウト
   if(high1 > g_asianHigh + spread)
   {
      double entry   = Ask;
      double slPrice = entry - sl;
      double tpPrice = entry + tp;
      double lots = CalcLotSize(sl, BK_RiskPercent);

      int t = OrderSend(Symbol(), OP_BUY, lots, entry, Slippage,
                        NormalizeDouble(slPrice, Digits),
                        NormalizeDouble(tpPrice, Digits),
                        "BK BUY", MagicNumber + MAGIC_BK, 0, clrGreen);
      if(t > 0)
      {
         Print("[BK] BUY ", lots, " @ ", entry, " SL=", slPrice, " TP=", tpPrice,
               " Range=", rangePips, "pips");
         g_bkTriggered = true;
      }
      else
         Print("[BK] BUY failed: ", GetLastError());
   }
   // ショートブレイクアウト
   else if(low1 < g_asianLow - spread)
   {
      double entry   = Bid;
      double slPrice = entry + sl;
      double tpPrice = entry - tp;
      double lots = CalcLotSize(sl, BK_RiskPercent);

      int t = OrderSend(Symbol(), OP_SELL, lots, entry, Slippage,
                        NormalizeDouble(slPrice, Digits),
                        NormalizeDouble(tpPrice, Digits),
                        "BK SELL", MagicNumber + MAGIC_BK, 0, clrOrange);
      if(t > 0)
      {
         Print("[BK] SELL ", lots, " @ ", entry, " SL=", slPrice, " TP=", tpPrice,
               " Range=", rangePips, "pips");
         g_bkTriggered = true;
      }
      else
         Print("[BK] SELL failed: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| DeInit                                                            |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("GBP Monthly10% EA stopped. Reason: ", reason);
}
//+------------------------------------------------------------------+
