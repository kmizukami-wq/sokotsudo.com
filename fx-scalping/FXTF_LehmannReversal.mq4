//+------------------------------------------------------------------+
//| FXTF_LehmannReversal.mq4                                         |
//| Lehmann 高速逆転 (HFT mean-reversion) 戦略                       |
//| FX TF ランク1 ゼロスプレッド + 手数料0円 環境専用                |
//|                                                                  |
//| 数学的根拠:                                                      |
//|   Lehmann (1990), Lo & MacKinlay (1990) が実証した              |
//|   短期リターンの負の自己相関 ρ(1) ≈ -0.05〜-0.15                |
//|                                                                  |
//| 戦略 (tick数ベース版・バックテスト対応):                         |
//|   1. 過去 InpReversalLookbackTicks (例:10) tick で                |
//|      |move| > InpReversalThresholdPip 動いたら逆方向にエントリ   |
//|   2. 退出:                                                       |
//|      - +InpTakeProfitPip 到達                                    |
//|      - -InpStopLossPip 到達                                      |
//|      - InpMaxHoldSec 経過                                        |
//|                                                                  |
//| 重要: ゼロスプレッド前提。通常スプレッドでは絶対に勝てない       |
//+------------------------------------------------------------------+
#property copyright   "FXTF LehmannReversal"
#property link        ""
#property version     "2.00"
#property strict
#property description "FXTF Rank1 zero-spread HFT mean-reversion (Lehmann 1990)"

//--- Input parameters
input double InpLots                   = 0.01;  // Lot size (0.01=100units / 0.10=1,000units)
input int    InpMagicNumber            = 0;     // Magic (0 = auto-generate from Symbol, base 86000)
input int    InpSlippage               = 0;     // Max slippage (points); 0 = strict

input string _sep1_                    = "===== Lehmann HFT Reversal =====";
input int    InpReversalLookbackTicks  = 10;    // Lookback window (ticks) for move detection
input double InpReversalThresholdPip   = 0.5;   // Min move (pip) in lookback to trigger fade
input double InpTakeProfitPip          = 0.5;   // Take profit (pip)
input double InpStopLossPip            = 0.7;   // Stop loss (pip)
input int    InpMaxHoldSec             = 5;     // Max hold (seconds)
input int    InpMinIntervalTicks       = 3;     // Min ticks between entries (anti-spam)
input int    InpTickBufferSize         = 100;   // Ring buffer size (ticks)

input string _sep2_                    = "===== Risk Guard =====";
input double InpMaxUnits               = 10000; // Rank-1 unit cap (0-fee tier)
input double InpMaxSpreadPoints        = 2;     // Max allowed spread (points)
input bool   InpSkipTokyoMorn          = false; // Skip JST 06-09
input bool   InpSkipNYNewsHrs          = false; // Skip JST 21-24
input int    InpJSTFromBroker          = 6;     // Broker -> JST offset (summer=6 / winter=7)
input bool   InpTradeOnlyEAHrs         = true;  // Enable custom hours
input int    InpStartHour              = 9;     // Start hour JST
input int    InpEndHour                = 3;     // End hour JST (wraps)
input bool   InpVerbose                = false; // Verbose log
input bool   InpLogCSV                 = false; // Append CSV to MQL4/Files/LehmannRev_<Symbol>.csv

//--- Globals: tick ring buffer (mid price only)
double   g_tickMid[];
int      g_tickHead       = 0;   // next write index
int      g_tickCount      = 0;   // valid items
long     g_totalTicks     = 0;   // monotonic tick counter (for min-interval)

double   g_point;
double   g_pipSize;
int      g_effectiveMagic = 0;

//--- Position state
datetime g_entryTime      = 0;
int      g_entrySide      = 0;
double   g_entryPrice     = 0;
double   g_entryMovePip   = 0;
int      g_entryLookback  = 0;
long     g_lastEntryTick  = -1000000;  // 初期値は遠い過去

string   g_csvFile = "";

//+------------------------------------------------------------------+
//| Symbol からマジックナンバーを自動生成 (86000-94999 の範囲)         |
//+------------------------------------------------------------------+
int AutoMagicFromSymbol()
{
   string s = Symbol();
   int hash = 0;
   int len = StringLen(s);
   for(int i = 0; i < len; i++)
   {
      int c = StringGetChar(s, i);
      hash = (hash * 31 + c) & 0x7FFFFFFF;
   }
   return 86000 + (hash % 9000);  // 86000〜94999 (TickScalper 77000-85999 と分離)
}

string SideStr(int s)
{
   if(s > 0) return "BUY";
   if(s < 0) return "SELL";
   return "FLAT";
}

//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   ArrayResize(g_tickMid, InpTickBufferSize);
   ArrayInitialize(g_tickMid, 0.0);
   g_tickHead   = 0;
   g_tickCount  = 0;
   g_totalTicks = 0;
   g_lastEntryTick = -1000000;

   g_point = Point;
   if(Digits == 3 || Digits == 5)
      g_pipSize = g_point * 10;
   else
      g_pipSize = g_point;

   if(InpMagicNumber <= 0)
      g_effectiveMagic = AutoMagicFromSymbol();
   else
      g_effectiveMagic = InpMagicNumber;

   Print("=== FXTF LehmannReversal initialized ===");
   PrintFormat("Symbol=%s Digits=%d Point=%.5f pipSize=%.5f Magic=%d%s",
               Symbol(), Digits, g_point, g_pipSize,
               g_effectiveMagic, (InpMagicNumber<=0 ? " (auto)" : ""));

   datetime brokerT = TimeCurrent();
   int jstHour = (TimeHour(brokerT) + InpJSTFromBroker) % 24;
   int jstMin = TimeMinute(brokerT);
   PrintFormat("Time: broker=%s JST=%02d:%02d (offset=%d)",
               TimeToStr(brokerT, TIME_DATE|TIME_MINUTES),
               jstHour, jstMin, InpJSTFromBroker);
   PrintFormat("Trade hours (JST): %02d:00 to %02d:00 (TradeOnly=%s)",
               InpStartHour, InpEndHour,
               (InpTradeOnlyEAHrs ? "true" : "false"));
   PrintFormat("Strategy: lookback=%dticks threshold=%.2fp TP=%.2fp SL=%.2fp maxHold=%ds",
               InpReversalLookbackTicks, InpReversalThresholdPip,
               InpTakeProfitPip, InpStopLossPip, InpMaxHoldSec);
   PrintFormat("Execution: lots=%.2f slippage=%d maxSpread=%.1fp minInterval=%dticks bufSize=%d",
               InpLots, InpSlippage, InpMaxSpreadPoints,
               InpMinIntervalTicks, InpTickBufferSize);
   PrintFormat("NOTE: Tick-count based. Works in MT4 strategy tester. Zero spread assumed.");

   g_csvFile = StringConcatenate("LehmannRev_", Symbol(), ".csv");
   if(InpLogCSV)
   {
      int h = FileOpen(g_csvFile, FILE_CSV|FILE_READ|FILE_ANSI, ';');
      bool needHeader = (h == INVALID_HANDLE);
      if(h != INVALID_HANDLE) FileClose(h);
      if(needHeader)
      {
         int hw = FileOpen(g_csvFile, FILE_CSV|FILE_WRITE|FILE_ANSI, ';');
         if(hw != INVALID_HANDLE)
         {
            FileWriteString(hw, "timestamp;symbol;event;side;pip;holdSec;movePip;lookbackTicks;spread\r\n");
            FileClose(hw);
         }
      }
      PrintFormat("CSV log -> MQL4/Files/%s", g_csvFile);
   }
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {}

//+------------------------------------------------------------------+
//| CSV writer                                                       |
//+------------------------------------------------------------------+
void LogCSV(string line)
{
   if(!InpLogCSV) return;
   int h = FileOpen(g_csvFile, FILE_CSV|FILE_READ|FILE_WRITE|FILE_ANSI, ';');
   if(h == INVALID_HANDLE)
   {
      h = FileOpen(g_csvFile, FILE_CSV|FILE_WRITE|FILE_ANSI, ';');
      if(h == INVALID_HANDLE) return;
   }
   FileSeek(h, 0, SEEK_END);
   FileWriteString(h, line + "\r\n");
   FileClose(h);
}

//+------------------------------------------------------------------+
//| JST hour                                                         |
//+------------------------------------------------------------------+
int GetJSTHour()
{
   int h = TimeHour(TimeCurrent()) + InpJSTFromBroker;
   h = (h % 24 + 24) % 24;
   return h;
}

bool IsTradeHour()
{
   int jstH = GetJSTHour();
   if(InpSkipTokyoMorn && jstH >= 6 && jstH < 9) return false;
   if(InpSkipNYNewsHrs && jstH >= 21 && jstH < 24) return false;
   if(InpTradeOnlyEAHrs)
   {
      if(InpStartHour <= InpEndHour)
         return (jstH >= InpStartHour && jstH < InpEndHour);
      return (jstH >= InpStartHour || jstH < InpEndHour);
   }
   return true;
}

bool IsSpreadAcceptable()
{
   double spreadPoints = (Ask - Bid) / g_point;
   if(spreadPoints > InpMaxSpreadPoints)
   {
      if(InpVerbose)
         PrintFormat("Spread %.1fp > max %.1fp, skip", spreadPoints, InpMaxSpreadPoints);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Push tick to ring buffer                                         |
//+------------------------------------------------------------------+
void PushTick(double mid)
{
   g_tickMid[g_tickHead] = mid;
   g_tickHead = (g_tickHead + 1) % InpTickBufferSize;
   if(g_tickCount < InpTickBufferSize) g_tickCount++;
   g_totalTicks++;
}

//+------------------------------------------------------------------+
//| Get mid price N ticks ago (0 = now, 1 = 1 tick ago, ...)         |
//| Returns 0 if buffer has insufficient history                     |
//+------------------------------------------------------------------+
double GetMidNTicksAgo(int nTicksAgo)
{
   if(nTicksAgo < 0) return 0;
   if(g_tickCount <= nTicksAgo) return 0;
   // head-1 は最新 tick の index、head-1-N で N tick 前
   int idx = (g_tickHead - 1 - nTicksAgo + InpTickBufferSize) % InpTickBufferSize;
   return g_tickMid[idx];
}

//+------------------------------------------------------------------+
//| Find my open position (-1 if none)                               |
//+------------------------------------------------------------------+
int FindMyPosition()
{
   for(int i = OrdersTotal()-1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() != Symbol()) continue;
      if(OrderMagicNumber() != g_effectiveMagic) continue;
      if(OrderType() != OP_BUY && OrderType() != OP_SELL) continue;
      return OrderTicket();
   }
   return -1;
}

//+------------------------------------------------------------------+
//| Sum same-direction units across all orders on this symbol        |
//+------------------------------------------------------------------+
double SumSameDirectionUnits(int side)
{
   double total = 0;
   double contractSize = MarketInfo(Symbol(), MODE_LOTSIZE);
   for(int i = OrdersTotal()-1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() != Symbol()) continue;
      int t = OrderType();
      bool isBuy  = (t == OP_BUY  || t == OP_BUYLIMIT  || t == OP_BUYSTOP);
      bool isSell = (t == OP_SELL || t == OP_SELLLIMIT || t == OP_SELLSTOP);
      if(side > 0 && isBuy)  total += OrderLots() * contractSize;
      if(side < 0 && isSell) total += OrderLots() * contractSize;
   }
   return total;
}

//+------------------------------------------------------------------+
//| Open position                                                    |
//+------------------------------------------------------------------+
void OpenPosition(int side, double movePip)
{
   double contractSize = MarketInfo(Symbol(), MODE_LOTSIZE);
   double newUnits = InpLots * contractSize;
   double existing = SumSameDirectionUnits(side);
   if(existing + newUnits > InpMaxUnits)
   {
      if(InpVerbose)
         PrintFormat("Guard: existing=%.0f + new=%.0f > max=%.0f, skip",
                     existing, newUnits, InpMaxUnits);
      return;
   }

   int type = (side > 0) ? OP_BUY : OP_SELL;
   double price = (side > 0) ? Ask : Bid;
   string comment = "Lehmann";
   int ticket = OrderSend(Symbol(), type, InpLots, price, InpSlippage,
                          0, 0, comment, g_effectiveMagic, 0, clrNONE);
   if(ticket < 0)
   {
      int err = GetLastError();
      string hint = "";
      switch(err)
      {
         case 129: hint = "invalid price - stale quote"; break;
         case 131: hint = "invalid lot step"; break;
         case 132: hint = "market closed"; break;
         case 133: hint = "trade disabled"; break;
         case 134: hint = "not enough money"; break;
         case 135: hint = "price changed - retry next tick"; break;
         case 136: hint = "off quotes"; break;
         case 138: hint = "requote - raise InpSlippage if frequent"; break;
         case 146: hint = "trade context busy"; break;
         case 149: hint = "hedging prohibited"; break;
         default:  hint = "see MT4 error code reference"; break;
      }
      PrintFormat("OrderSend failed err=%d (%s)", err, hint);
      return;
   }

   g_entryTime     = TimeCurrent();
   g_entrySide     = side;
   g_entryPrice    = price;
   g_entryMovePip  = movePip;
   g_entryLookback = InpReversalLookbackTicks;
   g_lastEntryTick = g_totalTicks;

   double spread = (Ask - Bid) / g_point;
   PrintFormat("ENTRY %s @%.5f ticket=%d movePip=%+.2fp lookback=%dticks spread=%.1fp",
               SideStr(side), price, ticket, movePip,
               InpReversalLookbackTicks, spread);

   if(InpLogCSV)
   {
      string line = StringFormat("%s;%s;ENTRY;%s;;;%+0.2f;%d;%.1f",
                                 TimeToStr(g_entryTime, TIME_DATE|TIME_SECONDS),
                                 Symbol(), SideStr(side),
                                 movePip, InpReversalLookbackTicks, spread);
      LogCSV(line);
   }
}

//+------------------------------------------------------------------+
//| Current pip P/L                                                  |
//+------------------------------------------------------------------+
double CurrentPipPnL(int ticket)
{
   if(!OrderSelect(ticket, SELECT_BY_TICKET)) return 0;
   double entry = OrderOpenPrice();
   double cur   = (OrderType() == OP_BUY) ? Bid : Ask;
   int side     = (OrderType() == OP_BUY) ? 1 : -1;
   return (cur - entry) * side / g_pipSize;
}

//+------------------------------------------------------------------+
//| Close position                                                   |
//+------------------------------------------------------------------+
void ClosePosition(int ticket, string reason)
{
   if(!OrderSelect(ticket, SELECT_BY_TICKET)) return;
   double price = (OrderType() == OP_BUY) ? Bid : Ask;
   double pip   = CurrentPipPnL(ticket);
   int side     = (OrderType() == OP_BUY) ? 1 : -1;
   int holdSec  = (int)(TimeCurrent() - g_entryTime);

   bool ok = OrderClose(ticket, OrderLots(), price, InpSlippage, clrNONE);
   if(!ok)
   {
      PrintFormat("OrderClose failed err=%d", GetLastError());
      return;
   }
   PrintFormat("EXIT[%s] side=%s pip=%+.2f holdSec=%d ticket=%d (entry move=%+.2fp lookback=%dticks)",
               reason, SideStr(side), pip, holdSec, ticket,
               g_entryMovePip, g_entryLookback);

   if(InpLogCSV)
   {
      string evt = "EXIT_" + reason;
      double spread = (Ask - Bid) / g_point;
      string line = StringFormat("%s;%s;%s;%s;%+0.2f;%d;%+0.2f;%d;%.1f",
                                 TimeToStr(TimeCurrent(), TIME_DATE|TIME_SECONDS),
                                 Symbol(), evt, SideStr(side),
                                 pip, holdSec,
                                 g_entryMovePip, g_entryLookback, spread);
      LogCSV(line);
   }
}

//+------------------------------------------------------------------+
//| Manage open position (TP / SL / timeout)                         |
//+------------------------------------------------------------------+
void ManagePosition(int ticket)
{
   double pip = CurrentPipPnL(ticket);
   int holdSec = (int)(TimeCurrent() - g_entryTime);

   // 利確
   if(pip >= InpTakeProfitPip)
   {
      ClosePosition(ticket, "TP");
      return;
   }
   // 損切り
   if(pip <= -InpStopLossPip)
   {
      ClosePosition(ticket, "SL");
      return;
   }
   // タイムアウト
   if(holdSec >= InpMaxHoldSec)
   {
      ClosePosition(ticket, "TIMEOUT");
      return;
   }
}

//+------------------------------------------------------------------+
//| OnTick                                                           |
//+------------------------------------------------------------------+
void OnTick()
{
   double mid = (Bid + Ask) / 2.0;
   PushTick(mid);

   // 既存ポジション管理
   int ticket = FindMyPosition();
   if(ticket >= 0)
   {
      ManagePosition(ticket);
      return;
   }

   // ゲート
   if(!IsTradeHour()) return;
   if(!IsSpreadAcceptable()) return;

   // 連続発火防止 (tick数カウンタで判定)
   if((g_totalTicks - g_lastEntryTick) < InpMinIntervalTicks)
      return;

   // バッファ充填チェック
   if(g_tickCount <= InpReversalLookbackTicks) return;

   // Lehmann シグナル評価
   double pastMid = GetMidNTicksAgo(InpReversalLookbackTicks);
   if(pastMid <= 0) return;
   double movePip = (mid - pastMid) / g_pipSize;

   if(movePip > InpReversalThresholdPip)
   {
      // 急騰直後 → SELL (Lehmann reversal)
      OpenPosition(-1, movePip);
   }
   else if(movePip < -InpReversalThresholdPip)
   {
      // 急落直後 → BUY
      OpenPosition(+1, movePip);
   }
}

//+------------------------------------------------------------------+
