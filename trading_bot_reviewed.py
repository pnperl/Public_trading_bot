# ╔══════════════════════════════════════════════════════════════════╗
# ║         TRADING BOT — WORKS IN COLAB & GITHUB ACTIONS           ║
# ║                                                                  ║
# ║  HOW TO USE IN COLAB:                                            ║
# ║    1. Run Cell 1 (install packages)                              ║
# ║    2. Add TELEGRAM_TOKEN and CHAT_ID in Colab 🔑 Secrets panel   ║
# ║    3. Run Cell 2 (bot starts)                                    ║
# ║                                                                  ║
# ║  HOW TO USE ON GITHUB ACTIONS:                                   ║
# ║    1. Upload this file + requirements.txt to GitHub repo         ║
# ║    2. Add TELEGRAM_TOKEN and CHAT_ID in GitHub Secrets           ║
# ║    3. Create .github/workflows/run_bot.yml (see bottom of file)  ║
# ║                                                                  ║
# ║  ✏️  ONLY EDIT THE SETTINGS SECTION — nothing else               ║
# ╚══════════════════════════════════════════════════════════════════╝

# ════════════════════════════════════════════════════════════════════
# CELL 1 — Run this first in Colab (skip if using GitHub Actions)
# ════════════════════════════════════════════════════════════════════
# !pip install yfinance pandas numpy requests python-dotenv -q


# ════════════════════════════════════════════════════════════════════
# CELL 2 — Paste everything below and run
# ════════════════════════════════════════════════════════════════════

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# ── Works in both Colab and GitHub Actions automatically ─────────────
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
        print("\n" + "═" * 80 + "\n")

logging.basicConfig(level=logging.WARNING)
IST = ZoneInfo("Asia/Kolkata")


# ════════════════════════════════════════════════════════════════════
# ✏️  SETTINGS — ONLY EDIT THIS SECTION
# ════════════════════════════════════════════════════════════════════

SYMBOLS = [
    "^NSEI",
    "RELIANCE.NS",
    "HDFCBANK.NS",
    # ── Uncomment to add more ──
    "BTC-USD",
    "ETH-USD",
    # "GOLD.MCX",
    # "CRUDEOIL.MCX",
    # "AAPL",
    # "TSLA",
    "CL=F",
    "^CNXCMDT", #(NIFTY COMMODITIES) 
    "MCX.NS", #(Multi Commodity Exchange of India Ltd)
    "CL=F", #(WTI Crude Future)
    "GC=F", #(Gold Future)
    "SI=F", #(Silver Future)
    "NG=F", #(Natural Gas Future)
    "HG=F", #(Copper Future)
    "INR=X", #(USD/INR)
]

INTERVAL          = "5m"   # candle size — do not change unless testing
MIN_PROBABILITY   = 55     # 0-100. Raise to 65-70 to reduce false signals
ATR_SL_MULTIPLIER = 1.5    # stop loss width. Try 2.0 if hitting SL too often
TP_THRESHOLD      = 0.03   # 3% take profit. Change to 0.02 for 2% etc.
MAX_DAILY_LOSS    = 3      # pause a symbol after this many losses in one day

# ════════════════════════════════════════════════════════════════════


# ────────────────────────────────────────────────────────────────────
# AUTO-DETECT PROFILE
# Reads the symbol name and automatically sets timezone, market hours,
# doji sensitivity, and strike rounding. You never need to edit this.
# ────────────────────────────────────────────────────────────────────

def detect_profile(symbol: str) -> dict:
    s = symbol.upper()
    CRYPTO_KW = ["BTC","ETH","BNB","SOL","XRP","DOGE","ADA","MATIC","AVAX"]

    if any(k in s for k in CRYPTO_KW) or s.endswith("-USD"):
        price = _quick_price(symbol)
        if   price and price > 10_000: strike = 500
        elif price and price > 1_000:  strike = 50
        else:                           strike = 1
        return dict(type="CRYPTO",   tz="UTC",
                    hours=None,               doji=0.15, strike=strike)

    if s.endswith(".MCX"):
        return dict(type="MCX",      tz="Asia/Kolkata",
                    hours=("09:00","23:30"),   doji=0.20, strike=1)

    if any(s.startswith(x) for x in ["^NSE","^BSE"]) or s.endswith((".NS",".BO")):
        return dict(type="INDIA",    tz="Asia/Kolkata",
                    hours=("09:15","15:30"),   doji=0.20, strike=50)

    if s in ["^GSPC","^DJI","^IXIC","SPY","QQQ","IWM"]:
        return dict(type="US_INDEX", tz="America/New_York",
                    hours=("09:30","16:00"),   doji=0.20, strike=5)

    return         dict(type="STOCK",    tz="America/New_York",
                    hours=("09:30","16:00"),   doji=0.20, strike=1)


def _quick_price(symbol: str):
    try:
        df = yf.download(symbol, period="1d", interval="1m",
                         auto_adjust=False, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


# ────────────────────────────────────────────────────────────────────
# MARKET HOURS CHECK
# ────────────────────────────────────────────────────────────────────

def is_market_open(profile: dict) -> bool:
    if profile["hours"] is None:
        return True
    tz     = ZoneInfo(profile["tz"])
    now_tz = datetime.now(tz)
    if now_tz.weekday() >= 5:
        return False                            # Sat=5, Sun=6
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
# DATA FETCH  (3 retries with back-off)
# ────────────────────────────────────────────────────────────────────

def fetch_data(symbol: str, interval: str, retries: int = 3):
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(symbol, period="2d", interval=interval,
                             auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if not df.empty:
                return df
        except Exception as e:
            print(f"⚠️  [{symbol}] Fetch attempt {attempt} failed: {e}")
        time.sleep(3 * attempt)
    return None


# ────────────────────────────────────────────────────────────────────
# HEIKIN ASHI
# Smoothed candles. Each candle averages itself with the previous one.
# Makes trends cleaner and reduces false signals vs regular candles.
# ────────────────────────────────────────────────────────────────────

def heikin_ashi(df: pd.DataFrame):
    ohlc = df[["Open","High","Low","Close"]].apply(
        pd.to_numeric, errors="coerce").dropna()
    if len(ohlc) < 10:
        return None

    o = ohlc["Open"].values.astype(float)
    h = ohlc["High"].values.astype(float)
    l = ohlc["Low"].values.astype(float)
    c = ohlc["Close"].values.astype(float)

    ha_c    = (o + h + l + c) / 4
    ha_o    = np.zeros_like(o)
    ha_o[0] = (o[0] + c[0]) / 2
    for i in range(1, len(o)):
        ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2

    return pd.DataFrame({
        "open":  ha_o,
        "close": ha_c,
        "high":  np.maximum.reduce([ha_o, ha_c, h]),
        "low":   np.minimum.reduce([ha_o, ha_c, l]),
    })


# ────────────────────────────────────────────────────────────────────
# INDICATORS
# RSI  = is market overbought or oversold?
# ATR  = how volatile is the market right now? (sets SL size)
# EMA  = what is the overall trend direction?
# Vol  = is this move backed by real volume?
# ────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> dict:
    close = pd.to_numeric(df["Close"], errors="coerce")
    high  = pd.to_numeric(df["High"],  errors="coerce")
    low   = pd.to_numeric(df["Low"],   errors="coerce")

    # RSI 14
    delta   = close.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rsi_raw = 100 - 100 / (1 + gain / loss)
    rsi     = float(rsi_raw.iloc[-2]) if not np.isnan(rsi_raw.iloc[-2]) else 50.0

    # ATR 14
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_raw = tr.rolling(14).mean()
    atr     = float(atr_raw.iloc[-2]) if not np.isnan(atr_raw.iloc[-2]) else float(tr.mean())

    # EMA 20
    ema_raw = close.ewm(span=20, adjust=False).mean()
    ema     = float(ema_raw.iloc[-2]) if not np.isnan(ema_raw.iloc[-2]) else float(close.iloc[-2])

    # Volume ratio
    vol_ratio = 1.0
    if "Volume" in df.columns:
        vol = pd.to_numeric(df["Volume"], errors="coerce")
        avg = vol.rolling(20).mean().iloc[-2]
        if avg and avg > 0:
            vol_ratio = float(vol.iloc[-2] / avg)

    return {"rsi": rsi, "atr": atr, "ema": ema, "vol_ratio": vol_ratio}


# ────────────────────────────────────────────────────────────────────
# PROBABILITY SCORING  (0 – 100)
#
#  Factor             Max   What it checks
#  ─────────────────────────────────────────────────────
#  Candle body size    25   Is signal candle strong or weak?
#  RSI zone            25   Right momentum zone for direction?
#  Volume surge        20   Is big money behind this move?
#  Prior trend         20   Clean trend before the reversal?
#  Price vs EMA20      10   Does overall market agree?
#  ─────────────────────────────────────────────────────
#  TOTAL              100
#
#  Trade entered only if score >= MIN_PROBABILITY
# ────────────────────────────────────────────────────────────────────

def compute_probability(ha: pd.DataFrame, ind: dict, direction: str) -> tuple:
    score     = 0
    breakdown = {}
    curr      = ha.iloc[-2]    # last confirmed closed candle

    # 1. Candle body strength (0-25)
    rng      = curr["high"] - curr["low"]
    body     = abs(curr["close"] - curr["open"])
    strength = (body / rng) if rng > 0 else 0
    pts      = min(round(strength * 25), 25)
    score   += pts
    breakdown["Body Strength"] = f"{pts}/25  (body={round(strength*100)}% of range)"

    # 2. RSI zone (0-25)
    rsi = ind["rsi"]
    if direction == "CALL":
        if   rsi < 30:  rsi_pts = 20   # oversold → likely to bounce
        elif rsi <= 60: rsi_pts = 25   # ideal range
        elif rsi <= 70: rsi_pts = 10   # a bit stretched
        else:           rsi_pts = 0    # overbought → risky for CALL
    else:
        if   rsi > 70:  rsi_pts = 20
        elif rsi >= 40: rsi_pts = 25
        elif rsi >= 30: rsi_pts = 10
        else:           rsi_pts = 0    # oversold → risky for PUT
    score += rsi_pts
    breakdown["RSI Zone"] = f"{rsi_pts}/25  (RSI={round(rsi,1)})"

    # 3. Volume surge (0-20)
    vr      = ind["vol_ratio"]
    vol_pts = 20 if vr >= 2.0 else (15 if vr >= 1.5 else (8 if vr >= 1.0 else 0))
    score  += vol_pts
    breakdown["Volume"] = f"{vol_pts}/20  ({round(vr,2)}× avg)"

    # 4. Prior HA trend — last 5 candles before flip (0-20)
    prior = ha.iloc[-7:-2]
    if direction == "CALL":
        agree = int((prior["close"] < prior["open"]).sum())   # prior bearish candles
    else:
        agree = int((prior["close"] > prior["open"]).sum())   # prior bullish candles
    trend_pts = round((agree / 5) * 20)
    score    += trend_pts
    breakdown["Prior Trend"] = f"{trend_pts}/20  ({agree}/5 candles)"

    # 5. Price vs EMA20 (0-10)
    price   = float(curr["close"])
    ema_pts = 10 if (direction == "CALL" and price > ind["ema"]) or \
                    (direction == "PUT"  and price < ind["ema"]) else 0
    score  += ema_pts
    breakdown["EMA20"] = f"{ema_pts}/10  (price={round(price,2)}, EMA={round(ind['ema'],2)})"

    return score, breakdown


# ────────────────────────────────────────────────────────────────────
# SIGNAL DETECTION
#
# Heikin Ashi colour flip:
#   Red candle → Green candle = CALL  (downtrend reversing up)
#   Green candle → Red candle = PUT   (uptrend reversing down)
#
# Uses iloc[-3] (prev) and iloc[-2] (curr) — both confirmed closed.
# iloc[-1] = still-forming live candle — intentionally IGNORED.
# ────────────────────────────────────────────────────────────────────

def not_doji(row, thresh: float) -> bool:
    rng = row["high"] - row["low"]
    return rng > 0 and abs(row["close"] - row["open"]) > thresh * rng


def check_signal(ha: pd.DataFrame, doji_thresh: float):
    prev = ha.iloc[-3]
    curr = ha.iloc[-2]

    if (prev["close"] < prev["open"] and
        curr["close"] > curr["open"] and
        not_doji(prev, doji_thresh) and
        not_doji(curr, doji_thresh)):
        return "CALL"

    if (prev["close"] > prev["open"] and
        curr["close"] < curr["open"] and
        not_doji(prev, doji_thresh) and
        not_doji(curr, doji_thresh)):
        return "PUT"

    return None


# ────────────────────────────────────────────────────────────────────
# STOP LOSS  (ATR-based)
# Volatile day = wider SL = fewer false hits
# Calm day     = tighter SL = better risk management
# ────────────────────────────────────────────────────────────────────

def calc_sl(price: float, atr: float, direction: str) -> float:
    return price - (atr * ATR_SL_MULTIPLIER) if direction == "CALL" \
      else price + (atr * ATR_SL_MULTIPLIER)


def trail_sl(current_sl: float, price: float, atr: float, direction: str) -> float:
    new_sl = calc_sl(price, atr, direction)
    return max(current_sl, new_sl) if direction == "CALL" \
      else min(current_sl, new_sl)


# ────────────────────────────────────────────────────────────────────
# TIMING
# ────────────────────────────────────────────────────────────────────

def seconds_until_next_5min() -> float:
    now       = datetime.now()
    delta     = timedelta(minutes=(5 - now.minute % 5))
    next_time = (now + delta).replace(second=5, microsecond=0)
    return max(5.0, (next_time - now).total_seconds())


# ────────────────────────────────────────────────────────────────────
# LIVE DASHBOARD
# ────────────────────────────────────────────────────────────────────

def print_dashboard(states: dict, sleep_secs: float):
    clear_output(wait=True)
    now = datetime.now(IST).strftime("%d-%b-%Y  %I:%M:%S %p  IST")

    print(f"\n  ╔{'═'*72}╗")
    print(f"  ║  🤖  TRADING BOT  ·  {now:<49}║")
    print(f"  ╚{'═'*72}╝")
    print(f"\n  Settings → MinProb:{MIN_PROBABILITY}%  "
          f"ATR×{ATR_SL_MULTIPLIER}  TP:{int(TP_THRESHOLD*100)}%  "
          f"MaxLoss/day:{MAX_DAILY_LOSS}")

    print(f"\n  {'SYMBOL':<14} {'MKT':<7} {'POS':<5} {'ENTRY':>10} "
          f"{'PRICE':>10} {'UNREAL':>10} {'PROB':>6} "
          f"{'T':>4} {'W':>4} {'L':>4} {'WIN%':>6} "
          f"{'P&L':>9} {'BEST':>9} {'WORST':>9}")
    print("  " + "─" * 122)

    g = dict(w=0, l=0, pnl=0.0, best=float("-inf"), worst=float("inf"))

    for sym, st in states.items():
        mkt   = "OPEN" if is_market_open(st["profile"]) else "CLOSED"
        pos   = st["position"] or "—"
        entry = st["entry_price"]
        price = st["latest_price"]
        prob  = str(st.get("last_prob", "—"))
        entry_d = f"{round(entry,2)}" if entry else "—"

        unreal = 0.0
        if st["position"] == "CALL" and entry: unreal = price - entry
        elif st["position"] == "PUT" and entry: unreal = entry - price
        u_str = f"{'▲' if unreal >= 0 else '▼'}{abs(round(unreal,2))}"

        w   = st["wins"]
        l   = st["losses"]
        tot = w + l
        wr  = f"{round(w/tot*100,1)}%" if tot > 0 else "—"

        pnl   = st["pnl"]
        best  = round(st["best"],  2) if st["best"]  != float("-inf") else "—"
        worst = round(st["worst"], 2) if st["worst"] != float("inf")  else "—"

        print(f"  {sym:<14} {mkt:<7} {pos:<5} {entry_d:>10} "
              f"{round(price,2):>10} {u_str:>10} {prob:>6} "
              f"{tot:>4} {w:>4} {l:>4} {wr:>6} "
              f"{round(pnl,2):>9} {str(best):>9} {str(worst):>9}")

        g["w"]   += w
        g["l"]   += l
        g["pnl"] += pnl
        if st["best"]  != float("-inf"): g["best"]  = max(g["best"],  st["best"])
        if st["worst"] != float("inf"):  g["worst"] = min(g["worst"], st["worst"])

    print("  " + "─" * 122)
    gt  = g["w"] + g["l"]
    gwr = f"{round(g['w']/gt*100,1)}%" if gt > 0 else "—"
    gb  = round(g["best"],  2) if g["best"]  != float("-inf") else "—"
    gwo = round(g["worst"], 2) if g["worst"] != float("inf")  else "—"
    icon = "📈" if g["pnl"] >= 0 else "📉"
    print(f"\n  TOTAL → Trades:{gt}  Wins:{g['w']}  Losses:{g['l']}  "
          f"WinRate:{gwr}  {icon} P&L:{round(g['pnl'],2)}  "
          f"Best:{gb}  Worst:{gwo}")
    print(f"\n  ⏳ Next candle in {round(sleep_secs)}s  ·  Ctrl+C to stop\n")


# ════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════

def start_bot():
    print(f"\n🚀 Starting bot...")
    print(f"   Symbols   : {SYMBOLS}")
    print(f"   Min Prob  : {MIN_PROBABILITY}%")
    print(f"   ATR SL    : {ATR_SL_MULTIPLIER}×")
    print(f"   TP        : {int(TP_THRESHOLD*100)}%")
    print(f"   Platform  : {'Google Colab' if IN_COLAB else 'GitHub Actions / Terminal'}")
    print(f"   Time(IST) : {ist_now()}\n")

    send_alert(
        f"🤖 Bot Started\n"
        f"{'─'*28}\n"
        f"Symbols    : {', '.join(SYMBOLS)}\n"
        f"Interval   : {INTERVAL}\n"
        f"Min Prob   : {MIN_PROBABILITY}%\n"
        f"ATR SL     : {ATR_SL_MULTIPLIER}×\n"
        f"Take Profit: {int(TP_THRESHOLD*100)}%\n"
        f"Time (IST) : {ist_now()}"
    )

    # Initialise per-symbol state
    states = {}
    for sym in SYMBOLS:
        states[sym] = dict(
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
        )

    while True:

        for sym in SYMBOLS:
            st = states[sym]

            # Reset daily loss counter at midnight
            today = datetime.now().date()
            if today != st["last_day"]:
                st["daily_losses"] = 0
                st["last_day"]     = today

            # Pause symbol if daily loss limit reached
            if st["daily_losses"] >= MAX_DAILY_LOSS:
                continue

            # Skip if market closed for this symbol
            if not is_market_open(st["profile"]):
                continue

            # Download candles
            df = fetch_data(sym, INTERVAL)
            if df is None:
                continue

            # Skip duplicate candle
            if df.index[-1] == st["last_time"]:
                continue
            st["last_time"] = df.index[-1]

            # Build HA candles
            ha = heikin_ashi(df)
            if ha is None:
                continue

            # Compute indicators
            try:
                ind = compute_indicators(df)
            except Exception as e:
                print(f"⚠️  [{sym}] Indicator error: {e}")
                continue

            # Use iloc[-2] — last confirmed candle price (not live candle)
            price = float(df["Close"].iloc[-2])
            st["latest_price"] = price

            # ── EXIT LOGIC ──────────────────────────────────────────
            if st["position"]:
                entry = st["entry_price"]
                pos   = st["position"]

                # Update trailing SL
                st["trailing_sl"] = trail_sl(st["trailing_sl"], price, ind["atr"], pos)

                # Calculate % profit/loss
                p_pct = (price - entry) / entry if pos == "CALL" \
                   else (entry - price) / entry

                # Check TP first, then SL
                exit_reason = None
                if p_pct >= TP_THRESHOLD:
                    exit_reason = "✅ TAKE PROFIT HIT"
                elif pos == "CALL" and price < st["trailing_sl"]:
                    exit_reason = "❌ STOP LOSS HIT"
                elif pos == "PUT"  and price > st["trailing_sl"]:
                    exit_reason = "❌ STOP LOSS HIT"

                if exit_reason:
                    pnl = (price - entry) if pos == "CALL" else (entry - price)
                    st["pnl"] += pnl

                    if pnl > 0:
                        st["wins"] += 1
                    else:
                        st["losses"]       += 1
                        st["daily_losses"] += 1

                    st["best"]  = max(st["best"],  pnl)
                    st["worst"] = min(st["worst"], pnl)

                    send_alert(
                        f"{exit_reason}\n"
                        f"{'─'*28}\n"
                        f"Symbol    : {sym}\n"
                        f"Direction : {pos}\n"
                        f"Entry     : {round(entry, 2)}\n"
                        f"Exit      : {round(price, 2)}\n"
                        f"P&L       : {round(pnl, 2)} pts\n"
                        f"Total P&L : {round(st['pnl'], 2)} pts\n"
                        f"Time(IST) : {ist_now()}"
                    )
                    st["position"]    = None
                    st["entry_price"] = None
                    st["trailing_sl"] = None

            # ── ENTRY LOGIC ─────────────────────────────────────────
            elif st["position"] is None:
                signal = check_signal(ha, st["profile"]["doji"])

                if signal:
                    score, breakdown = compute_probability(ha, ind, signal)
                    st["last_prob"]   = f"{score}%"

                    if score >= MIN_PROBABILITY:
                        st["position"]    = signal
                        st["entry_price"] = price
                        st["trailing_sl"] = calc_sl(price, ind["atr"], signal)
                        atm = round(price / st["profile"]["strike"]) * st["profile"]["strike"]

                        send_alert(
                            f"📊 {signal} ENTRY\n"
                            f"{'─'*28}\n"
                            f"Symbol     : {sym}\n"
                            f"Probability: {score}%\n"
                            f"Price      : {round(price, 2)}\n"
                            f"ATM Strike : {atm}\n"
                            f"Stop Loss  : {round(st['trailing_sl'], 2)}\n"
                            f"ATR        : {round(ind['atr'], 2)}\n"
                            f"RSI        : {round(ind['rsi'], 1)}\n"
                            f"Time(IST)  : {ist_now()}\n"
                            f"{'─'*28}\n"
                            + "\n".join(f"  {k}: {v}" for k, v in breakdown.items())
                        )
                    else:
                        st["last_prob"] = f"{score}%⚠️"
                        print(f"  [{sym}] {signal} skipped — "
                              f"score {score}% < threshold {MIN_PROBABILITY}%")

        # Dashboard + sleep until next candle
        sleep_secs = seconds_until_next_5min()
        print_dashboard(states, sleep_secs)
        time.sleep(sleep_secs)


# ════════════════════════════════════════════════════════════════════
# START
# ════════════════════════════════════════════════════════════════════
start_bot()


# ════════════════════════════════════════════════════════════════════
# GITHUB ACTIONS WORKFLOW
# ────────────────────────────────────────────────────────────────────
# Save this as:  .github/workflows/run_bot.yml
# in your GitHub repo (create the folders if they don't exist)
# ════════════════════════════════════════════════════════════════════
#
# name: Trading Bot
#
# on:
#   schedule:
#     - cron: '30 3 * * 1-5'    # 9:00 AM IST = 3:30 AM UTC, Mon-Fri only
#   workflow_dispatch:            # adds a manual Run button in GitHub UI
#
# jobs:
#   run-bot:
#     runs-on: ubuntu-latest
#     timeout-minutes: 370        # auto-kills after ~6hrs = around 3:10 PM IST
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
