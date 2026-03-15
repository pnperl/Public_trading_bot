# ╔══════════════════════════════════════════════════════════════════╗
# ║         TRADING BOT — GITHUB ACTIONS / TERMINAL                 ║
# ║                                                                  ║
# ║  SETUP:                                                          ║
# ║    1. Upload this file + requirements.txt to GitHub (public repo)║
# ║    2. Add TELEGRAM_TOKEN and CHAT_ID in GitHub → Settings →      ║
# ║       Secrets and variables → Actions                            ║
# ║    3. Create .github/workflows/run_bot.yml (see bottom of file)  ║
# ║    4. Push — bot auto-runs 9 AM–3 PM IST every weekday           ║
# ║                                                                  ║
# ║  ✏️  ONLY EDIT THE SETTINGS SECTION BELOW                        ║
# ╚══════════════════════════════════════════════════════════════════╝

import os
import gc
import time
import logging
import requests
import numpy  as np
import pandas as pd
import yfinance as yf

from datetime        import datetime, timedelta
from zoneinfo        import ZoneInfo
from dotenv          import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Secrets ──────────────────────────────────────────────────────────
load_dotenv()
try:
    from google.colab import userdata
    from IPython.display import clear_output
    TOKEN    = userdata.get("TELEGRAM_TOKEN")
    CHAT_ID  = userdata.get("CHAT_ID")
    IN_COLAB = True
except ImportError:
    TOKEN    = os.getenv("TELEGRAM_TOKEN")
    CHAT_ID  = os.getenv("CHAT_ID")
    IN_COLAB = False
    def clear_output(wait=False):
        print("\n" + "═" * 90)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

IST = ZoneInfo("Asia/Kolkata")


# ════════════════════════════════════════════════════════════════════
# ✏️  SETTINGS — ONLY EDIT THIS SECTION
# ════════════════════════════════════════════════════════════════════

SYMBOLS = [
    # ── Indian Indices ──────────────────────────────────────────
    "^NSEI",           # NIFTY 50
    "^NSEBANK",        # NIFTY Bank
    "^CNXCMDT",        # NIFTY Commodities Index

    # ── Indian Stocks ───────────────────────────────────────────
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "MCX.NS",          # Multi Commodity Exchange stock

    # ── Global Commodity Futures ────────────────────────────────
    "GC=F",            # Gold
    "SI=F",            # Silver
    "CL=F",            # WTI Crude Oil
    "NG=F",            # Natural Gas
    "HG=F",            # Copper

    # ── Crypto ──────────────────────────────────────────────────
    "BTC-USD",
    "ETH-USD",

    # ── Forex ───────────────────────────────────────────────────
    "INR=X",           # USD/INR

    # ── Uncomment to add ────────────────────────────────────────
    # "AAPL",
    # "TSLA",
    # "SOL-USD",
]

INTERVAL          = "5m"   # candle size — do not change
MIN_PROBABILITY   = 35     # 0–100. Raise to reduce signals, lower to increase.
ATR_SL_MULTIPLIER = 2      # SL width. Try 2.0 if SL hits too often.
TP_THRESHOLD      = 0.02   # 2% take profit
MAX_DAILY_LOSS    = 30     # pause symbol after N losses in one day

# ── Parallel fetch settings ─────────────────────────────────────────
# MAX_WORKERS: how many symbols fetch simultaneously
# Rule of thumb: Yahoo Finance allows ~5-8 parallel requests safely.
# DO NOT raise above 8 — causes 429 rate limit errors.
MAX_WORKERS       = 5

# ════════════════════════════════════════════════════════════════════


# ────────────────────────────────────────────────────────────────────
# HOW PARALLEL FETCHING WORKS
# ────────────────────────────────────────────────────────────────────
#
#  OLD (sequential — slow):
#  Symbol 1 ──fetch──► 3s
#                       Symbol 2 ──fetch──► 3s
#                                            Symbol 3 ──fetch──► 3s
#  Total: 3s × 14 symbols = 42s delay
#
#  NEW (parallel — fast):
#  Symbol 1 ──fetch──►|
#  Symbol 2 ──fetch──►| all running simultaneously
#  Symbol 3 ──fetch──►|
#  ...                 |
#  Symbol 5 ──fetch──►|
#                      ↓ done in ~3-5s total (slowest one wins)
#  Then next batch of 5 starts immediately
#
#  Total: ~6-10s for 14 symbols instead of 42s
#  Signal latency: near-zero — all symbols processed within seconds
#  Rate limit safe: max 5 at a time, well under Yahoo's limit
#
# ────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────
# SYMBOL PROFILES
# ────────────────────────────────────────────────────────────────────

def detect_profile(symbol: str) -> dict:
    s = symbol.upper()
    CRYPTO_KW = ["BTC","ETH","BNB","SOL","XRP","DOGE","ADA","MATIC","AVAX"]

    if any(k in s for k in CRYPTO_KW) or s.endswith("-USD"):
        return dict(type="CRYPTO",   tz="UTC",
                    hours=None,              doji=0.15, strike=1,  label="Crypto")

    if s.endswith("=F"):
        return dict(type="FUTURES",  tz="America/New_York",
                    hours=None,              doji=0.18, strike=1,  label="Futures")

    if s.endswith("=X"):
        return dict(type="FOREX",    tz="UTC",
                    hours=None,              doji=0.15, strike=1,  label="Forex")

    if s.endswith(".MCX"):
        return dict(type="MCX",      tz="Asia/Kolkata",
                    hours=("09:00","23:30"), doji=0.20, strike=1,  label="MCX")

    if (any(s.startswith(x) for x in ["^NSE","^BSE","^CNX"])
            or s.endswith((".NS",".BO"))):
        return dict(type="INDIA",    tz="Asia/Kolkata",
                    hours=("09:15","15:30"), doji=0.20, strike=50, label="India")

    if s in ["^GSPC","^DJI","^IXIC","SPY","QQQ","IWM"]:
        return dict(type="US_INDEX", tz="America/New_York",
                    hours=("09:30","16:00"), doji=0.20, strike=5,  label="US Idx")

    return         dict(type="STOCK",    tz="America/New_York",
                    hours=("09:30","16:00"), doji=0.20, strike=1,  label="Stock")


def is_market_open(profile: dict) -> bool:
    tz     = ZoneInfo(profile["tz"])
    now_tz = datetime.now(tz)
    if profile["type"] not in ("CRYPTO", "FOREX") and now_tz.weekday() >= 5:
        return False                          # weekend — closed for non-crypto/forex
    if profile["hours"] is None:
        return True                           # 24×7 asset
    if profile["type"] == "FUTURES":
        return True                           # futures trade ~24h on weekdays
    curr = now_tz.strftime("%H:%M")
    return profile["hours"][0] <= curr <= profile["hours"][1]


# ────────────────────────────────────────────────────────────────────
# TELEGRAM
# ────────────────────────────────────────────────────────────────────

def ist_now() -> str:
    return datetime.now(IST).strftime("%d-%b-%Y %I:%M:%S %p IST")


def send_alert(msg: str):
    if not TOKEN or not CHAT_ID:
        print(f"[ALERT] {msg}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg},
            timeout=5
        )
    except Exception as e:
        print(f"⚠️  Telegram error: {e}")


# ────────────────────────────────────────────────────────────────────
# DATA FETCH  (called in parallel threads)
#
# Returns: (symbol, DataFrame or None)
# The tuple return lets us match results back to symbols after
# parallel execution completes.
#
# Rate limit protection:
#   • MAX_WORKERS=5 caps simultaneous requests
#   • 429 detected → sleep 60s in that thread only, others continue
#   • 3 retries with exponential back-off per symbol
# ────────────────────────────────────────────────────────────────────

def fetch_one(symbol: str, interval: str, retries: int = 3):
    """Fetch data for a single symbol. Returns (symbol, df or None)."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol,
                period      = "1d",
                interval    = interval,
                auto_adjust = False,
                progress    = False,
                threads     = False,    # thread-safe: one HTTP call per thread
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            if df.empty:
                raise ValueError("Empty")

            keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            df   = df[keep].copy()

            for col in ["Open","High","Low","Close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").astype("float32")

            df.dropna(subset=["Open","High","Low","Close"], inplace=True)

            if len(df) < 10:
                raise ValueError(f"Only {len(df)} rows")

            return symbol, df

        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many Requests" in err:
                print(f"⚠️  [{symbol}] Rate limited — sleeping 60s")
                time.sleep(60)
            else:
                wait = 3 * (2 ** (attempt - 1))
                if attempt < retries:
                    time.sleep(wait)

    return symbol, None


def fetch_all_parallel(symbols: list, interval: str) -> dict:
    """
    Fetch all symbols in parallel batches of MAX_WORKERS.
    Returns dict: {symbol: DataFrame or None}

    Why batched and not all at once?
    Sending 14 requests simultaneously could trigger Yahoo's rate limit.
    Batching in groups of 5 keeps us safe while still being fast.

    Time comparison:
      Sequential (old): ~3s × 14 = 42s
      Parallel batch 5: ~3s × 3 batches = ~9s
    """
    results = {}
    batch_size = MAX_WORKERS

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(fetch_one, sym, interval): sym
                for sym in batch
            }
            for future in as_completed(futures):
                try:
                    sym, df = future.result(timeout=30)
                    results[sym] = df
                except Exception as e:
                    sym = futures[future]
                    print(f"⚠️  [{sym}] Thread error: {e}")
                    results[sym] = None

        # Small pause between batches — not within batches
        # This gives Yahoo a brief rest between bursts, not between symbols
        if i + batch_size < len(symbols):
            time.sleep(1.5)

    return results


# ────────────────────────────────────────────────────────────────────
# HEIKIN ASHI
# ────────────────────────────────────────────────────────────────────

def heikin_ashi(df: pd.DataFrame):
    o = df["Open"].values.astype("float32")
    h = df["High"].values.astype("float32")
    l = df["Low"].values.astype("float32")
    c = df["Close"].values.astype("float32")

    ha_c    = (o + h + l + c) / 4
    ha_o    = np.empty_like(o)
    ha_o[0] = (o[0] + c[0]) / 2
    for i in range(1, len(o)):
        ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2

    ha = pd.DataFrame({
        "open":  ha_o,
        "close": ha_c,
        "high":  np.maximum.reduce([ha_o, ha_c, h]),
        "low":   np.minimum.reduce([ha_o, ha_c, l]),
    }, dtype="float32")

    del o, h, l, c, ha_c, ha_o
    return ha


# ────────────────────────────────────────────────────────────────────
# INDICATORS
# ────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"].astype("float32")
    high  = df["High"].astype("float32")
    low   = df["Low"].astype("float32")

    # RSI 14
    delta   = close.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain / loss.replace(0, np.nan)
    rsi_s   = 100 - 100 / (1 + rs)
    rsi     = float(rsi_s.iloc[-2]) if pd.notna(rsi_s.iloc[-2]) else 50.0

    # ATR 14
    pc    = close.shift(1)
    tr    = pd.concat([(high-low),(high-pc).abs(),(low-pc).abs()], axis=1).max(axis=1)
    atr_s = tr.rolling(14).mean()
    atr   = float(atr_s.iloc[-2]) if pd.notna(atr_s.iloc[-2]) else float(tr.mean())

    # EMA 20
    ema_s = close.ewm(span=20, adjust=False).mean()
    ema   = float(ema_s.iloc[-2]) if pd.notna(ema_s.iloc[-2]) else float(close.iloc[-2])

    # Volume ratio
    vol_ratio = 1.0
    if "Volume" in df.columns:
        vol = df["Volume"].astype("float32")
        avg = vol.rolling(20).mean().iloc[-2]
        if pd.notna(avg) and avg > 0:
            vol_ratio = float(vol.iloc[-2] / avg)

    del delta, gain, loss, rs, rsi_s, pc, tr, atr_s, ema_s
    return {"rsi": rsi, "atr": atr, "ema": ema, "vol_ratio": vol_ratio}


# ────────────────────────────────────────────────────────────────────
# SIGNAL
# iloc[-3] = prev confirmed candle
# iloc[-2] = last confirmed candle
# iloc[-1] = live forming candle — NEVER used
# ────────────────────────────────────────────────────────────────────

def not_doji(row, thresh: float) -> bool:
    rng = float(row["high"]) - float(row["low"])
    return rng > 0 and abs(float(row["close"]) - float(row["open"])) > thresh * rng


def check_signal(ha: pd.DataFrame, doji_thresh: float):
    if len(ha) < 4:
        return None
    prev = ha.iloc[-3]
    curr = ha.iloc[-2]

    if (prev["close"] < prev["open"] and curr["close"] > curr["open"]
            and not_doji(prev, doji_thresh) and not_doji(curr, doji_thresh)):
        return "CALL"

    if (prev["close"] > prev["open"] and curr["close"] < curr["open"]
            and not_doji(prev, doji_thresh) and not_doji(curr, doji_thresh)):
        return "PUT"

    return None


# ────────────────────────────────────────────────────────────────────
# PROBABILITY SCORING  (0 – 100)
# ────────────────────────────────────────────────────────────────────

def compute_probability(ha: pd.DataFrame, ind: dict, direction: str) -> tuple:
    score = 0
    breakdown = {}
    curr  = ha.iloc[-2]

    # 1. Candle body strength (0-25)
    rng      = float(curr["high"]) - float(curr["low"])
    body     = abs(float(curr["close"]) - float(curr["open"]))
    strength = (body / rng) if rng > 0 else 0
    pts      = min(round(strength * 25), 25)
    score   += pts
    breakdown["Body Strength"] = f"{pts}/25  ({round(strength*100)}% of range)"

    # 2. RSI zone (0-25)
    rsi = ind["rsi"]
    if direction == "CALL":
        rsi_pts = 25 if 30<=rsi<=60 else (20 if rsi<30 else (10 if rsi<=70 else 0))
    else:
        rsi_pts = 25 if 40<=rsi<=70 else (20 if rsi>70 else (10 if rsi>=30 else 0))
    score += rsi_pts
    breakdown["RSI Zone"] = f"{rsi_pts}/25  (RSI={round(rsi,1)})"

    # 3. Volume surge (0-20)
    vr      = ind["vol_ratio"]
    vol_pts = 20 if vr>=2.0 else (15 if vr>=1.5 else (8 if vr>=1.0 else 0))
    score  += vol_pts
    breakdown["Volume"] = f"{vol_pts}/20  ({round(vr,2)}× avg)"

    # 4. Prior HA trend (0-20)
    if len(ha) >= 7:
        prior     = ha.iloc[-7:-2]
        agree     = int((prior["close"] < prior["open"]).sum()) if direction=="CALL" \
               else int((prior["close"] > prior["open"]).sum())
        trend_pts = round((agree / 5) * 20)
    else:
        agree, trend_pts = 0, 0
    score += trend_pts
    breakdown["Prior Trend"] = f"{trend_pts}/20  ({agree}/5 candles)"

    # 5. Price vs EMA20 (0-10)
    price   = float(curr["close"])
    ema_pts = 10 if (direction=="CALL" and price>ind["ema"]) or \
                    (direction=="PUT"  and price<ind["ema"]) else 0
    score  += ema_pts
    breakdown["EMA20"] = f"{ema_pts}/10  (price={round(price,4)}, EMA={round(ind['ema'],4)})"

    return score, breakdown


# ────────────────────────────────────────────────────────────────────
# STOP LOSS
# ────────────────────────────────────────────────────────────────────

def calc_sl(price: float, atr: float, direction: str) -> float:
    return price-(atr*ATR_SL_MULTIPLIER) if direction=="CALL" \
      else price+(atr*ATR_SL_MULTIPLIER)


def trail_sl(current_sl: float, price: float, atr: float, direction: str) -> float:
    new_sl = calc_sl(price, atr, direction)
    return max(current_sl, new_sl) if direction=="CALL" else min(current_sl, new_sl)


# ────────────────────────────────────────────────────────────────────
# TIMING
# ────────────────────────────────────────────────────────────────────

def seconds_until_next_5min() -> float:
    now       = datetime.now()
    delta     = timedelta(minutes=(5 - now.minute % 5))
    next_time = (now + delta).replace(second=5, microsecond=0)
    return max(5.0, (next_time - now).total_seconds())


# ────────────────────────────────────────────────────────────────────
# PROCESS ONE SYMBOL (signal check + entry/exit)
# Called after all fetches complete — pure logic, no I/O delay
# ────────────────────────────────────────────────────────────────────

def process_symbol(sym: str, df: pd.DataFrame, st: dict) -> None:
    """All signal/entry/exit logic for one symbol. Modifies st in place."""

    # Skip if same candle already processed
    candle_time = df.index[-1]
    if candle_time == st["last_time"]:
        return
    st["last_time"] = candle_time

    ha = heikin_ashi(df)
    if ha is None:
        return

    try:
        ind = compute_indicators(df)
    except Exception as e:
        print(f"⚠️  [{sym}] Indicator error: {e}")
        del ha
        return

    # Confirmed last closed candle price
    price = float(df["Close"].iloc[-2])
    st["latest_price"] = price

    # ── EXIT ──────────────────────────────────────────────────────
    if st["position"]:
        entry = st["entry_price"]
        pos   = st["position"]

        st["trailing_sl"] = trail_sl(st["trailing_sl"], price, ind["atr"], pos)

        p_pct = (price-entry)/entry if pos=="CALL" else (entry-price)/entry

        exit_reason = None
        if   p_pct >= TP_THRESHOLD:                      exit_reason = "✅ TAKE PROFIT HIT"
        elif pos=="CALL" and price < st["trailing_sl"]:  exit_reason = "❌ STOP LOSS HIT"
        elif pos=="PUT"  and price > st["trailing_sl"]:  exit_reason = "❌ STOP LOSS HIT"

        if exit_reason:
            pnl        = (price-entry) if pos=="CALL" else (entry-price)
            st["pnl"] += pnl

            if pnl > 0:
                st["wins"] += 1
            else:
                st["losses"]       += 1
                st["daily_losses"] += 1

            st["best"]  = max(st["best"],  pnl)
            st["worst"] = min(st["worst"], pnl)

            # Log completed trade to history
            st["trade_log"].append({
                "time"  : ist_now(),
                "symbol": sym,
                "dir"   : pos,
                "entry" : round(entry, 4),
                "exit"  : round(price, 4),
                "pnl"   : round(pnl,   4),
                "result": "WIN" if pnl > 0 else "LOSS",
                "reason": "TP" if "PROFIT" in exit_reason else "SL",
            })

            send_alert(
                f"{exit_reason}\n"
                f"{'─'*28}\n"
                f"Symbol    : {sym}\n"
                f"Direction : {pos}\n"
                f"Entry     : {round(entry,4)}\n"
                f"Exit      : {round(price,4)}\n"
                f"P&L       : {round(pnl,4)}\n"
                f"Total P&L : {round(st['pnl'],4)}\n"
                f"Time(IST) : {ist_now()}"
            )
            st["position"] = st["entry_price"] = st["trailing_sl"] = None

    # ── ENTRY ─────────────────────────────────────────────────────
    elif st["position"] is None:
        signal = check_signal(ha, st["profile"]["doji"])

        if signal:
            score, breakdown = compute_probability(ha, ind, signal)
            st["last_prob"]   = f"{score}%"

            if score >= MIN_PROBABILITY:
                st["position"]    = signal
                st["entry_price"] = price
                st["trailing_sl"] = calc_sl(price, ind["atr"], signal)
                strike = st["profile"]["strike"]
                atm    = round(price/strike)*strike if strike > 0 else price

                send_alert(
                    f"📊 {signal} ENTRY\n"
                    f"{'─'*28}\n"
                    f"Symbol     : {sym}\n"
                    f"Type       : {st['profile']['label']}\n"
                    f"Probability: {score}%\n"
                    f"Price      : {round(price,4)}\n"
                    f"ATM Strike : {atm}\n"
                    f"Stop Loss  : {round(st['trailing_sl'],4)}\n"
                    f"ATR        : {round(ind['atr'],4)}\n"
                    f"RSI        : {round(ind['rsi'],1)}\n"
                    f"Time(IST)  : {ist_now()}\n"
                    f"{'─'*28}\n"
                    + "\n".join(f"  {k}: {v}" for k,v in breakdown.items())
                )
            else:
                st["last_prob"] = f"{score}%⚠️"
                print(f"  [{sym}] {signal} skipped — score {score}% < {MIN_PROBABILITY}%")

    del ha, ind
    gc.collect()


# ────────────────────────────────────────────────────────────────────
# DASHBOARD  — 3 sections:
#   1. Live positions table   (refreshes every 5 min)
#   2. Symbol summary table   (stats per symbol)
#   3. Trade performance log  (every completed trade, newest first)
# ────────────────────────────────────────────────────────────────────

def print_dashboard(states: dict, fetch_ms: int, sleep_secs: float):
    clear_output(wait=True)
    now = datetime.now(IST).strftime("%d-%b-%Y  %I:%M:%S %p  IST")
    W   = 110    # total console width

    # ── Header ────────────────────────────────────────────────────
    print(f"\n  ╔{'═'*(W-4)}╗")
    print(f"  ║  🤖  TRADING BOT  ·  {now:<{W-26}}║")
    print(f"  ╚{'═'*(W-4)}╝")
    print(f"  MinProb:{MIN_PROBABILITY}%  ATR×{ATR_SL_MULTIPLIER}  "
          f"TP:{int(TP_THRESHOLD*100)}%  MaxLoss:{MAX_DAILY_LOSS}  "
          f"Workers:{MAX_WORKERS}  FetchTime:{fetch_ms}ms\n")

    # ════════════════════════════════════════════════════════════════
    # TABLE 1 — LIVE POSITIONS
    # Shows every symbol: market status, current position, unrealized P&L
    # ════════════════════════════════════════════════════════════════
    print(f"  {'─'*3} LIVE POSITIONS {'─'*(W-21)}")
    h1 = (f"  {'SYMBOL':<14} {'TYPE':<8} {'MKT':<7} {'POS':<5} "
          f"{'ENTRY':>10} {'PRICE':>12} {'UNREAL':>10} {'PROB':>6} {'SL':>12}")
    print(h1)
    print("  " + "─" * (len(h1)-2))

    for sym, st in states.items():
        mkt     = "OPEN"   if is_market_open(st["profile"]) else "CLOSED"
        typ     = st["profile"]["label"]
        pos     = st["position"] or "—"
        entry   = st["entry_price"]
        price   = st["latest_price"]
        prob    = str(st.get("last_prob","—"))
        sl      = st["trailing_sl"]
        entry_d = f"{round(entry,4)}"  if entry else "—"
        sl_d    = f"{round(sl,4)}"     if sl    else "—"
        pause   = " ⏸" if st["daily_losses"] >= MAX_DAILY_LOSS else ""

        unreal = 0.0
        if pos=="CALL" and entry: unreal = price - entry
        elif pos=="PUT" and entry: unreal = entry - price
        u_icon = "▲" if unreal >= 0 else "▼"
        u_str  = f"{u_icon}{abs(round(unreal,4))}"

        print(f"  {sym:<14} {typ:<8} {mkt:<7} {pos:<5} "
              f"{entry_d:>10} {round(price,4):>12} {u_str:>10} {prob:>6} {sl_d:>12}{pause}")

    # ════════════════════════════════════════════════════════════════
    # TABLE 2 — SYMBOL PERFORMANCE SUMMARY
    # Win rate, total P&L, best/worst trade per symbol
    # ════════════════════════════════════════════════════════════════
    print(f"\n  {'─'*3} SYMBOL PERFORMANCE {'─'*(W-25)}")
    h2 = (f"  {'SYMBOL':<14} {'TYPE':<8} {'TRADES':>6} {'WINS':>5} {'LOSSES':>7} "
          f"{'WIN%':>6} {'REALIZED P&L':>13} {'BEST':>10} {'WORST':>10} {'DAILY L':>8}")
    print(h2)
    print("  " + "─" * (len(h2)-2))

    g = dict(w=0, l=0, pnl=0.0, best=float("-inf"), worst=float("inf"))

    for sym, st in states.items():
        typ   = st["profile"]["label"]
        w, l  = st["wins"], st["losses"]
        tot   = w + l
        wr    = f"{round(w/tot*100,1)}%" if tot > 0 else "—"
        pnl   = st["pnl"]
        best  = round(st["best"],  4) if st["best"]  != float("-inf") else "—"
        worst = round(st["worst"], 4) if st["worst"] != float("inf")  else "—"
        dl    = st["daily_losses"]
        pnl_icon = "+" if pnl >= 0 else ""

        print(f"  {sym:<14} {typ:<8} {tot:>6} {w:>5} {l:>7} "
              f"{wr:>6} {pnl_icon}{round(pnl,4):>12} {str(best):>10} {str(worst):>10} {dl:>8}")

        g["w"]   += w;  g["l"] += l;  g["pnl"] += pnl
        if st["best"]  != float("-inf"): g["best"]  = max(g["best"],  st["best"])
        if st["worst"] != float("inf"):  g["worst"] = min(g["worst"], st["worst"])

    # Summary row
    print("  " + "═" * (len(h2)-2))
    gt   = g["w"] + g["l"]
    gwr  = f"{round(g['w']/gt*100,1)}%" if gt > 0 else "—"
    gb   = round(g["best"],  4) if g["best"]  != float("-inf") else "—"
    gwo  = round(g["worst"], 4) if g["worst"] != float("inf")  else "—"
    icon = "📈" if g["pnl"] >= 0 else "📉"
    pnl_icon = "+" if g["pnl"] >= 0 else ""
    print(f"  {'TOTAL':<14} {'':8} {gt:>6} {g['w']:>5} {g['l']:>7} "
          f"{gwr:>6} {icon}{pnl_icon}{round(g['pnl'],4):>11} {str(gb):>10} {str(gwo):>10}")

    # ════════════════════════════════════════════════════════════════
    # TABLE 3 — TRADE PERFORMANCE LOG
    # Every completed trade across all symbols, newest first
    # ════════════════════════════════════════════════════════════════

    # Collect all trades from all symbols and sort newest first
    all_trades = []
    for st in states.values():
        all_trades.extend(st["trade_log"])

    print(f"\n  {'─'*3} TRADE LOG  ({len(all_trades)} completed trades) {'─'*(W-38)}")

    if not all_trades:
        print("  No completed trades yet this session.\n")
    else:
        h3 = (f"  {'#':>3}  {'TIME (IST)':<26} {'SYMBOL':<14} {'DIR':<5} "
              f"{'ENTRY':>10} {'EXIT':>10} {'P&L':>10} {'%':>6} {'RESULT':<6} {'EXIT BY':<7}")
        print(h3)
        print("  " + "─" * (len(h3)-2))

        # Show newest first — reverse chronological
        for i, t in enumerate(reversed(all_trades), 1):
            entry  = t["entry"]
            exit_p = t["exit"]
            pnl    = t["pnl"]
            # P&L as percentage of entry
            pct    = round((pnl / entry) * 100, 2) if entry != 0 else 0
            result_icon = "✅" if t["result"]=="WIN" else "❌"
            pnl_sign    = "+" if pnl >= 0 else ""

            print(f"  {i:>3}  {t['time']:<26} {t['symbol']:<14} {t['dir']:<5} "
                  f"{entry:>10} {exit_p:>10} {pnl_sign}{pnl:>9} "
                  f"{pnl_sign}{pct:>5}% {result_icon}{t['result']:<5} {t['reason']:<7}")

    print(f"\n  ⏳ Next refresh in {round(sleep_secs)}s  "
          f"·  Fetch took {fetch_ms}ms  ·  Ctrl+C to stop\n")


# ════════════════════════════════════════════════════════════════════
# SYMBOL CATEGORY SELECTOR
# Asks user which categories to trade before bot starts.
# Each category has a fixed symbol list — easy to extend below.
# ════════════════════════════════════════════════════════════════════

CATEGORIES = {
    "1": {
        "name"    : "Indian Indices",
        "symbols" : ["^NSEI", "^NSEBANK", "^CNXCMDT"],
        "desc"    : "NIFTY 50, Bank NIFTY, NIFTY Commodities",
    },
    "2": {
        "name"    : "Indian Stocks",
        "symbols" : ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS",
                     "INFY.NS", "MCX.NS", "ICICIBANK.NS"],
        "desc"    : "Reliance, HDFC Bank, TCS, Infosys, MCX, ICICI Bank",
    },
    "3": {
        "name"    : "Global Commodity Futures",
        "symbols" : ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"],
        "desc"    : "Gold, Silver, Crude Oil, Natural Gas, Copper",
    },
    "4": {
        "name"    : "Crypto",
        "symbols" : ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
        "desc"    : "Bitcoin, Ethereum, Solana, BNB",
    },
    "5": {
        "name"    : "Forex",
        "symbols" : ["INR=X", "EURUSD=X", "GBPUSD=X", "JPYUSD=X"],
        "desc"    : "USD/INR, EUR/USD, GBP/USD, JPY/USD",
    },
    "6": {
        "name"    : "US Stocks",
        "symbols" : ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"],
        "desc"    : "Apple, Tesla, Nvidia, Microsoft, Amazon",
    },
}


def ask_categories() -> list:
    """
    Interactively asks user which categories to monitor.
    Works in both GitHub Actions (non-interactive → uses all) and terminal.
    Returns a deduplicated list of symbols.
    """

    # ── GitHub Actions is non-interactive — no stdin available ──────
    # Detect by checking if we're inside a CI environment.
    # If yes, skip the prompt and use the SYMBOLS list from settings.
    if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
        print("  Running on GitHub Actions — using SYMBOLS from settings.")
        return list(dict.fromkeys(SYMBOLS))

    # ── Interactive prompt (terminal / Colab) ───────────────────────
    print("\n" + "═" * 60)
    print("  🤖  TRADING BOT — SELECT CATEGORIES TO MONITOR")
    print("═" * 60)
    print("\n  Available categories:\n")

    for key, cat in CATEGORIES.items():
        syms_preview = ", ".join(cat["symbols"][:3])
        if len(cat["symbols"]) > 3:
            syms_preview += f" +{len(cat['symbols'])-3} more"
        print(f"    {key}.  {cat['name']:<30}  ({syms_preview})")

    print("\n  Enter numbers separated by commas.")
    print("  Examples:  1        → Indian Indices only")
    print("             1,2      → Indian Indices + Indian Stocks")
    print("             1,2,3,4  → Indices + Stocks + Commodities + Crypto")
    print("             all      → everything\n")

    while True:
        try:
            raw = input("  Your choice: ").strip().lower()
        except EOFError:
            # Non-interactive fallback
            print("  Non-interactive — using all categories.")
            raw = "all"

        if not raw:
            print("  ⚠️  Nothing entered. Please type a number or 'all'.\n")
            continue

        selected_symbols = []

        if raw == "all":
            for cat in CATEGORIES.values():
                selected_symbols.extend(cat["symbols"])
            break

        # Parse comma-separated numbers
        parts   = [p.strip() for p in raw.split(",")]
        invalid = [p for p in parts if p not in CATEGORIES]

        if invalid:
            print(f"  ⚠️  Invalid choices: {', '.join(invalid)}")
            print(f"      Valid options are: {', '.join(CATEGORIES.keys())} or 'all'\n")
            continue

        for p in parts:
            selected_symbols.extend(CATEGORIES[p]["symbols"])

        if not selected_symbols:
            print("  ⚠️  No symbols selected. Try again.\n")
            continue

        break

    # Deduplicate, preserve order
    final = list(dict.fromkeys(selected_symbols))

    # Print confirmation
    print(f"\n  ✅ Selected categories:")
    chosen_keys = [] if raw == "all" else [p.strip() for p in raw.split(",")]
    cats_to_show = CATEGORIES.items() if raw == "all" else \
                   [(k, CATEGORIES[k]) for k in chosen_keys]
    for k, cat in cats_to_show:
        print(f"     {cat['name']}: {', '.join(cat['symbols'])}")
    print(f"\n  Total symbols: {len(final)}")
    print(f"  {', '.join(final)}")
    print()

    return final


# ════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════

def start_bot():

    # ── Ask user which categories to monitor ─────────────────────────
    selected_symbols = ask_categories()

    print(f"\n🚀 Bot starting  ·  {ist_now()}")
    print(f"   {len(selected_symbols)} symbols  ·  {MAX_WORKERS} parallel workers")
    print(f"   MinProb:{MIN_PROBABILITY}%  ATR×{ATR_SL_MULTIPLIER}  "
          f"TP:{int(TP_THRESHOLD*100)}%  MaxLoss:{MAX_DAILY_LOSS}\n")

    send_alert(
        f"🤖 Bot Started\n{'─'*28}\n"
        f"Symbols    : {', '.join(selected_symbols)}\n"
        f"Interval   : {INTERVAL}\n"
        f"Workers    : {MAX_WORKERS} parallel\n"
        f"Min Prob   : {MIN_PROBABILITY}%\n"
        f"ATR SL     : {ATR_SL_MULTIPLIER}×\n"
        f"Take Profit: {int(TP_THRESHOLD*100)}%\n"
        f"Time (IST) : {ist_now()}"
    )

    # Initialise state — use selected symbols, not hardcoded SYMBOLS list
    unique_symbols = selected_symbols
    states = {
        sym: dict(
            position     = None,
            entry_price  = None,
            trailing_sl  = None,
            latest_price = 0.0,
            profile      = detect_profile(sym),
            pnl          = 0.0,
            wins         = 0,
            losses       = 0,
            best         = float("-inf"),
            worst        = float("inf"),
            last_time    = None,
            last_prob    = "—",
            daily_losses = 0,
            last_day     = datetime.now().date(),
            trade_log    = [],               # list of completed trade dicts
        )
        for sym in unique_symbols
    }

    fetch_ms = 0   # track last fetch duration for display

    while True:
        cycle_start = time.time()

        # Reset daily counters at midnight
        today = datetime.now().date()
        for st in states.values():
            if today != st["last_day"]:
                st["daily_losses"] = 0
                st["last_day"]     = today

        # Only fetch symbols whose market is open and not paused
        active = [
            sym for sym, st in states.items()
            if is_market_open(st["profile"])
            and st["daily_losses"] < MAX_DAILY_LOSS
        ]

        if active:
            # ── STEP 1: Fetch ALL active symbols in parallel ─────────
            # This is the slow I/O step — done once, as fast as possible
            t0       = time.time()
            data_map = fetch_all_parallel(active, INTERVAL)
            fetch_ms = int((time.time() - t0) * 1000)

            # ── STEP 2: Process signals — pure logic, instant ─────────
            # No network calls here. All symbols processed immediately
            # after fetch completes. Zero sequential delay.
            for sym in active:
                df = data_map.get(sym)
                if df is None:
                    continue
                try:
                    process_symbol(sym, df, states[sym])
                except Exception as e:
                    print(f"⚠️  [{sym}] Process error: {e}")
                finally:
                    del df
                    data_map[sym] = None   # free immediately after use

            del data_map
            gc.collect()
        else:
            fetch_ms = 0
            print(f"  All markets closed — {ist_now()}")
            time.sleep(60)
            continue

        # ── Sleep until next 5-minute candle ─────────────────────────
        elapsed    = time.time() - cycle_start
        sleep_secs = max(5.0, seconds_until_next_5min() - elapsed)

        print_dashboard(states, fetch_ms, sleep_secs)
        time.sleep(sleep_secs)


# ════════════════════════════════════════════════════════════════════
# START
# ════════════════════════════════════════════════════════════════════
start_bot()


# ════════════════════════════════════════════════════════════════════
# .github/workflows/run_bot.yml — copy into that file in your repo
# ════════════════════════════════════════════════════════════════════
#
# name: Trading Bot
#
# on:
#   schedule:
#     - cron: '30 3 * * 1-5'    # 9:00 AM IST = 3:30 AM UTC, Mon-Fri
#   workflow_dispatch:
#
# jobs:
#   run-bot:
#     runs-on: ubuntu-latest
#     timeout-minutes: 370
#
#     steps:
#       - uses: actions/checkout@v4
#
#       - name: Set up Python
#         uses: actions/setup-python@v5
#         with:
#           python-version: '3.11'
#
#       - name: Install dependencies
#         run: pip install yfinance pandas numpy requests python-dotenv -q
#
#       - name: Run bot
#         env:
#           TELEGRAM_TOKEN: ${{ secrets.TELEGRAM_TOKEN }}
#           CHAT_ID: ${{ secrets.CHAT_ID }}
#         run: python trading_bot_reviewed.py
