# ╔══════════════════════════════════════════════════════════════════╗
# ║         TRADING BOT — DUAL STRATEGY COMPARISON                  ║
# ║                                                                  ║
# ║  FILES IN YOUR REPO:                                             ║
# ║    config.py              ← EDIT THIS for symbols & settings    ║
# ║    trading_bot_reviewed.py ← DO NOT EDIT (bot engine)           ║
# ║    requirements.txt       ← Python packages                     ║
# ║    .github/workflows/     ← GitHub Actions scheduler            ║
# ║                                                                  ║
# ║  TWO STRATEGIES RUN SIMULTANEOUSLY:                              ║
# ║    Strategy A — Classic HA reversal (your original logic)       ║
# ║    Strategy B — Smart multi-indicator (new logic)               ║
# ║  Both tracked independently. Telegram shows which strategy      ║
# ║  each signal came from. Dashboard compares performance.         ║
# ╚══════════════════════════════════════════════════════════════════╝

import os
import gc
import time
import logging
import requests
import numpy  as np
import pandas as pd
import yfinance as yf

from datetime           import datetime, timedelta
from zoneinfo           import ZoneInfo
from dotenv             import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Load config ──────────────────────────────────────────────────────
from config import (
    CATEGORIES, STRATEGY_A, STRATEGY_B,
    INTERVAL, MAX_WORKERS, GITHUB_ACTIONS_SYMBOLS
)

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
        print("\n" + "═" * 120)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

IST = ZoneInfo("Asia/Kolkata")

# Convenience references loaded from config
STRATEGIES = {"A": STRATEGY_A, "B": STRATEGY_B}


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
    return     dict(type="STOCK",    tz="America/New_York",
                    hours=("09:30","16:00"), doji=0.20, strike=1,  label="Stock")


def is_market_open(profile: dict) -> bool:
    tz     = ZoneInfo(profile["tz"])
    now_tz = datetime.now(tz)
    if profile["type"] not in ("CRYPTO","FOREX") and now_tz.weekday() >= 5:
        return False
    if profile["hours"] is None:
        return True
    if profile["type"] == "FUTURES":
        return True
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
# DATA FETCH
# ────────────────────────────────────────────────────────────────────

def fetch_one(symbol: str, interval: str, retries: int = 3):
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(symbol, period="2d", interval=interval,
                             auto_adjust=False, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if df.empty:
                raise ValueError("Empty")
            keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            df   = df[keep].copy()
            for col in ["Open","High","Low","Close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").astype("float64")
            df.dropna(subset=["Open","High","Low","Close"], inplace=True)
            if len(df) < 30:
                raise ValueError(f"Only {len(df)} rows")
            return symbol, df
        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many Requests" in err:
                print(f"⚠️  [{symbol}] Rate limited — sleeping 60s")
                time.sleep(60)
            else:
                if attempt < retries:
                    time.sleep(3 * (2 ** (attempt-1)))
    return symbol, None


def fetch_all_parallel(symbols: list, interval: str) -> dict:
    results    = {}
    batch_size = MAX_WORKERS
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(fetch_one, sym, interval): sym for sym in batch}
            for future in as_completed(futures):
                try:
                    sym, df = future.result(timeout=30)
                    results[sym] = df
                except Exception as e:
                    results[futures[future]] = None
        if i + batch_size < len(symbols):
            time.sleep(1.5)
    return results


# ────────────────────────────────────────────────────────────────────
# HEIKIN ASHI
# ────────────────────────────────────────────────────────────────────

def heikin_ashi(df: pd.DataFrame):
    o = df["Open"].values.astype("float64")
    h = df["High"].values.astype("float64")
    l = df["Low"].values.astype("float64")
    c = df["Close"].values.astype("float64")
    ha_c    = (o + h + l + c) / 4
    ha_o    = np.empty_like(o)
    ha_o[0] = (o[0] + c[0]) / 2
    for i in range(1, len(o)):
        ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2
    ha = pd.DataFrame({
        "open":  ha_o, "close": ha_c,
        "high":  np.maximum.reduce([ha_o, ha_c, h]),
        "low":   np.minimum.reduce([ha_o, ha_c, l]),
    })
    del o, h, l, c, ha_c, ha_o
    return ha if len(ha) >= 10 else None


# ────────────────────────────────────────────────────────────────────
# INDICATORS  (shared by both strategies — computed once per symbol)
# ────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"].astype("float64")
    high  = df["High"].astype("float64")
    low   = df["Low"].astype("float64")

    # RSI 14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi_s = 100 - 100 / (1 + rs)
    rsi   = float(rsi_s.iloc[-2]) if pd.notna(rsi_s.iloc[-2]) else 50.0

    # ATR 14
    pc    = close.shift(1)
    tr    = pd.concat([(high-low),(high-pc).abs(),(low-pc).abs()], axis=1).max(axis=1)
    atr_s = tr.rolling(14).mean()
    atr   = float(atr_s.iloc[-2]) if pd.notna(atr_s.iloc[-2]) else float(tr.mean())

    # EMA 20 and EMA 50
    ema20 = float(close.ewm(span=20,adjust=False).mean().iloc[-2])
    ema50 = float(close.ewm(span=50,adjust=False).mean().iloc[-2])

    # MACD (12,26,9)
    macd_line   = close.ewm(span=12,adjust=False).mean() - close.ewm(span=26,adjust=False).mean()
    signal_line = macd_line.ewm(span=9,adjust=False).mean()
    mc, mp      = float(macd_line.iloc[-2]),  float(macd_line.iloc[-3])  if pd.notna(macd_line.iloc[-3])  else 0.0
    sc, sp      = float(signal_line.iloc[-2]),float(signal_line.iloc[-3]) if pd.notna(signal_line.iloc[-3]) else 0.0
    macd_bull   = (mp < sp) and (mc > sc)
    macd_bear   = (mp > sp) and (mc < sc)
    macd_above  = mc > sc
    macd_below  = mc < sc

    # Bollinger Bands (20, 2σ)
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bu       = float((bb_mid + 2*bb_std).iloc[-2])
    bl       = float((bb_mid - 2*bb_std).iloc[-2])
    p        = float(close.iloc[-2])
    bb_pctb  = (p - bl) / (bu - bl) if (bu - bl) > 0 else 0.5

    # Supertrend (ATR×3, period 10)
    st_atr   = tr.rolling(10).mean()
    hl2      = (high + low) / 2
    ub_raw   = hl2 + 3.0 * st_atr
    lb_raw   = hl2 - 3.0 * st_atr
    st_dir   = pd.Series(1, index=close.index)
    st_line  = pd.Series(np.nan, index=close.index)
    for i in range(1, len(close)):
        ub = float(ub_raw.iloc[i])
        lb = float(lb_raw.iloc[i])
        pv_ub = float(ub_raw.iloc[i-1])
        pv_lb = float(lb_raw.iloc[i-1])
        pv_st = float(st_line.iloc[i-1]) if pd.notna(st_line.iloc[i-1]) else lb
        pv_d  = int(st_dir.iloc[i-1])
        cp    = float(close.iloc[i])
        lb    = lb if lb > pv_lb or float(close.iloc[i-1]) < pv_lb else pv_lb
        ub    = ub if ub < pv_ub or float(close.iloc[i-1]) > pv_ub else pv_ub
        if pv_d == -1 and cp > pv_st:   st_dir.iloc[i]=1;  st_line.iloc[i]=lb
        elif pv_d == 1 and cp < pv_st:  st_dir.iloc[i]=-1; st_line.iloc[i]=ub
        else:                            st_dir.iloc[i]=pv_d; st_line.iloc[i]=lb if pv_d==1 else ub
    st_bull = int(st_dir.iloc[-2]) == 1
    st_bear = int(st_dir.iloc[-2]) == -1

    # Volume ratio
    vol_ratio = 1.0
    if "Volume" in df.columns:
        vol = df["Volume"].astype("float64")
        avg = vol.rolling(20).mean().iloc[-2]
        if pd.notna(avg) and avg > 0:
            vol_ratio = float(vol.iloc[-2] / avg)

    candle_range_pct = (atr / p * 100) if p > 0 else 0

    del delta,gain,loss,rs,rsi_s,pc,tr,atr_s
    del macd_line,signal_line,bb_mid,bb_std,st_atr,hl2,ub_raw,lb_raw,st_dir,st_line

    return {
        "rsi":rsi,"atr":atr,"ema":ema20,"ema20":ema20,"ema50":ema50,
        "vol_ratio":vol_ratio,"macd_bull":macd_bull,"macd_bear":macd_bear,
        "macd_above":macd_above,"macd_below":macd_below,
        "bb_pctb":bb_pctb,"st_bull":st_bull,"st_bear":st_bear,
        "candle_range_pct":candle_range_pct,
    }


# ────────────────────────────────────────────────────────────────────
# SIGNAL DETECTION  (same for both strategies)
# ────────────────────────────────────────────────────────────────────

def not_doji(row, thresh: float) -> bool:
    rng = float(row["high"]) - float(row["low"])
    return rng > 0 and abs(float(row["close"]) - float(row["open"])) > thresh * rng

def check_signal(ha: pd.DataFrame, doji_thresh: float):
    if len(ha) < 4: return None
    prev, curr = ha.iloc[-3], ha.iloc[-2]
    if (prev["close"] < prev["open"] and curr["close"] > curr["open"]
            and not_doji(prev,doji_thresh) and not_doji(curr,doji_thresh)):
        return "CALL"
    if (prev["close"] > prev["open"] and curr["close"] < curr["open"]
            and not_doji(prev,doji_thresh) and not_doji(curr,doji_thresh)):
        return "PUT"
    return None


# ────────────────────────────────────────────────────────────────────
# STRATEGY A — Classic HA probability (simple 3-factor, no vetoes)
# This is your original logic — minimal filtering, enters most flips
# ────────────────────────────────────────────────────────────────────

def score_strategy_a(ha: pd.DataFrame, ind: dict, direction: str) -> tuple:
    score = 0
    breakdown = {}
    curr  = ha.iloc[-2]

    # 1. Candle body strength (0-40)
    rng      = float(curr["high"]) - float(curr["low"])
    body     = abs(float(curr["close"]) - float(curr["open"]))
    strength = (body / rng) if rng > 0 else 0
    pts      = min(round(strength * 40), 40)
    score   += pts
    breakdown["Body Strength"] = f"{pts}/40  ({round(strength*100)}% of range)"

    # 2. RSI zone (0-35)
    rsi = ind["rsi"]
    if direction == "CALL":
        rsi_pts = 35 if 30<=rsi<=65 else (20 if rsi<30 else (10 if rsi<=75 else 0))
    else:
        rsi_pts = 35 if 35<=rsi<=70 else (20 if rsi>70 else (10 if rsi>=25 else 0))
    score += rsi_pts
    breakdown["RSI Zone"] = f"{rsi_pts}/35  (RSI={round(rsi,1)})"

    # 3. Prior HA trend — last 4 candles (0-25)
    if len(ha) >= 6:
        prior     = ha.iloc[-6:-2]
        agree     = int((prior["close"]<prior["open"]).sum()) if direction=="CALL" \
               else int((prior["close"]>prior["open"]).sum())
        trend_pts = round((agree/4)*25)
    else:
        agree, trend_pts = 0, 0
    score += trend_pts
    breakdown["Prior Trend"] = f"{trend_pts}/25  ({agree}/4 candles)"

    return min(score, 100), breakdown


# ────────────────────────────────────────────────────────────────────
# STRATEGY B — Smart multi-indicator probability (8 factors + vetoes)
# ────────────────────────────────────────────────────────────────────

def score_strategy_b(ha: pd.DataFrame, ind: dict, direction: str) -> tuple:
    score = 0
    breakdown = {}
    curr  = ha.iloc[-2]
    price = float(curr["close"])

    # ── VETO 1: Chop filter ──────────────────────────────────────
    if ind["candle_range_pct"] < 0.05:
        breakdown["⛔ CHOP VETO"] = f"ATR={round(ind['candle_range_pct'],3)}% — market flat"
        return 0, breakdown

    # ── VETO 2: RSI extreme ──────────────────────────────────────
    rsi = ind["rsi"]
    if direction == "CALL" and rsi > 80:
        breakdown["⛔ RSI VETO"] = f"RSI={round(rsi,1)} overbought"
        return 0, breakdown
    if direction == "PUT" and rsi < 20:
        breakdown["⛔ RSI VETO"] = f"RSI={round(rsi,1)} oversold"
        return 0, breakdown

    # ── VETO 3: Supertrend against direction ─────────────────────
    if direction == "CALL" and ind["st_bear"]:
        breakdown["⛔ ST VETO"] = "Supertrend BEARISH — no CALL"
        return 0, breakdown
    if direction == "PUT" and ind["st_bull"]:
        breakdown["⛔ ST VETO"] = "Supertrend BULLISH — no PUT"
        return 0, breakdown

    # ── Scoring ──────────────────────────────────────────────────
    rng=float(curr["high"])-float(curr["low"]); body=abs(float(curr["close"])-float(curr["open"]))
    strength=(body/rng) if rng>0 else 0
    pts=min(round(strength*15),15); score+=pts
    breakdown["1.Body"] = f"{pts}/15  ({round(strength*100)}%)"

    if direction=="CALL":
        rsi_pts=15 if 35<=rsi<=55 else(12 if rsi<35 else(8 if rsi<=65 else(3 if rsi<=75 else 0)))
    else:
        rsi_pts=15 if 45<=rsi<=65 else(12 if rsi>65 else(8 if rsi>=35 else(3 if rsi>=25 else 0)))
    score+=rsi_pts; breakdown["2.RSI"]=f"{rsi_pts}/15  (RSI={round(rsi,1)})"

    vr=ind["vol_ratio"]; vol_pts=10 if vr>=2.0 else(8 if vr>=1.5 else(5 if vr>=1.0 else 0))
    score+=vol_pts; breakdown["3.Volume"]=f"{vol_pts}/10  ({round(vr,2)}×)"

    if len(ha)>=7:
        prior=ha.iloc[-7:-2]
        agree=int((prior["close"]<prior["open"]).sum()) if direction=="CALL" \
         else int((prior["close"]>prior["open"]).sum())
        trend_pts=round((agree/5)*10)
    else: agree,trend_pts=0,0
    score+=trend_pts; breakdown["4.Trend"]=f"{trend_pts}/10  ({agree}/5)"

    ema20=ind["ema20"]; ema50=ind["ema50"]
    sep=abs(ema20-ema50)/ema50*100 if ema50>0 else 0
    if direction=="CALL":
        ema_pts=15 if price>ema20>ema50 and sep>0.3 else(10 if price>ema20>ema50 else(8 if price>ema20 else(4 if price>ema50 else 0)))
    else:
        ema_pts=15 if price<ema20<ema50 and sep>0.3 else(10 if price<ema20<ema50 else(8 if price<ema20 else(4 if price<ema50 else 0)))
    score+=ema_pts; breakdown["5.EMA"]=f"{ema_pts}/15  (20={round(ema20,2)} 50={round(ema50,2)})"

    if direction=="CALL":
        macd_pts=15 if ind["macd_bull"] else(8 if ind["macd_above"] else 0)
    else:
        macd_pts=15 if ind["macd_bear"] else(8 if ind["macd_below"] else 0)
    score+=macd_pts; breakdown["6.MACD"]=f"{macd_pts}/15"

    st_pts=15 if (direction=="CALL" and ind["st_bull"]) or (direction=="PUT" and ind["st_bear"]) else 0
    score+=st_pts; breakdown["7.SuperTrend"]=f"{st_pts}/15"

    bb=ind["bb_pctb"]
    if direction=="CALL":
        bb_pts=10 if bb<0.2 else(6 if bb<0.5 else(3 if bb<0.8 else 0))
    else:
        bb_pts=10 if bb>0.8 else(6 if bb>0.5 else(3 if bb>0.2 else 0))
    score+=bb_pts; breakdown["8.BB"]=f"{bb_pts}/10  (%B={round(bb,2)})"

    return min(score,100), breakdown


# ────────────────────────────────────────────────────────────────────
# STOP LOSS FUNCTIONS
# ────────────────────────────────────────────────────────────────────

def calc_initial_sl(price: float, atr: float, direction: str, multiplier: float) -> float:
    return price - atr*multiplier if direction=="CALL" else price + atr*multiplier

def fixed_trail_sl(current_sl: float, price: float, atr: float,
                   direction: str, multiplier: float) -> float:
    """Strategy A: simple fixed-distance ATR trail."""
    new_sl = price-atr*multiplier if direction=="CALL" else price+atr*multiplier
    return max(current_sl,new_sl) if direction=="CALL" else min(current_sl,new_sl)

def smart_trail_sl(current_sl: float, price: float, entry: float,
                   atr: float, direction: str, multiplier: float) -> tuple:
    """Strategy B: 3-phase trail. Returns (new_sl, phase_name)."""
    if direction == "CALL":
        pct = (price-entry)/entry*100
        if pct >= 1.0:   new_sl = price-atr*1.0;   phase = "3-LOCK"
        elif pct >= 0.5: new_sl = entry*1.001;      phase = "2-BEVEN"
        else:            new_sl = price-atr*multiplier; phase = "1-PROT"
        return max(current_sl, new_sl), phase
    else:
        pct = (entry-price)/entry*100
        if pct >= 1.0:   new_sl = price+atr*1.0;   phase = "3-LOCK"
        elif pct >= 0.5: new_sl = entry*0.999;      phase = "2-BEVEN"
        else:            new_sl = price+atr*multiplier; phase = "1-PROT"
        return min(current_sl, new_sl), phase


# ────────────────────────────────────────────────────────────────────
# TIMING
# ────────────────────────────────────────────────────────────────────

def seconds_until_next_5min() -> float:
    now       = datetime.now()
    delta     = timedelta(minutes=(5 - now.minute % 5))
    next_time = (now + delta).replace(second=5, microsecond=0)
    return max(5.0, (next_time - now).total_seconds())


# ────────────────────────────────────────────────────────────────────
# MAKE EMPTY STATE for one strategy slot
# ────────────────────────────────────────────────────────────────────

def make_state(sym: str) -> dict:
    return dict(
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
        sl_phase     = "—",
        daily_losses = 0,
        last_day     = datetime.now().date(),
        trade_log    = [],
    )


# ────────────────────────────────────────────────────────────────────
# PROCESS ONE SYMBOL × ONE STRATEGY
# ────────────────────────────────────────────────────────────────────

def process_symbol(sym: str, df: pd.DataFrame,
                   st: dict, ha: pd.DataFrame, ind: dict,
                   strategy_key: str) -> None:
    """
    Runs entry/exit logic for one symbol under one strategy.
    ha and ind are pre-computed and shared between both strategies
    to avoid double computation.
    """
    cfg   = STRATEGIES[strategy_key]
    price = float(df["Close"].iloc[-2])
    st["latest_price"] = price

    # ── EXIT ──────────────────────────────────────────────────────
    if st["position"]:
        entry  = st["entry_price"]
        pos    = st["position"]
        atr    = ind["atr"]
        mult   = cfg["atr_sl_multiplier"]

        if cfg["use_smart_trail"]:
            st["trailing_sl"], st["sl_phase"] = smart_trail_sl(
                st["trailing_sl"], price, entry, atr, pos, mult)
        else:
            st["trailing_sl"] = fixed_trail_sl(
                st["trailing_sl"], price, atr, pos, mult)
            st["sl_phase"] = "FIXED"

        p_pct = (price-entry)/entry if pos=="CALL" else (entry-price)/entry

        exit_reason = None
        if   p_pct >= cfg["tp_threshold"]:               exit_reason = "✅ TAKE PROFIT HIT"
        elif pos=="CALL" and price < st["trailing_sl"]:  exit_reason = "❌ STOP LOSS HIT"
        elif pos=="PUT"  and price > st["trailing_sl"]:  exit_reason = "❌ STOP LOSS HIT"

        if exit_reason:
            pnl = (price-entry) if pos=="CALL" else (entry-price)
            st["pnl"] += pnl
            if pnl > 0: st["wins"] += 1
            else:       st["losses"] += 1; st["daily_losses"] += 1
            st["best"]  = max(st["best"],  pnl)
            st["worst"] = min(st["worst"], pnl)
            st["trade_log"].append({
                "time":ist_now(),"symbol":sym,"strategy":cfg["short_name"],
                "dir":pos,"entry":round(entry,4),"exit":round(price,4),
                "pnl":round(pnl,4),"result":"WIN" if pnl>0 else "LOSS",
                "reason":"TP" if "PROFIT" in exit_reason else "SL",
            })
            pct_str = f"{round(p_pct*100,2)}%"
            send_alert(
                f"{exit_reason}\n"
                f"Strategy  : {cfg['name']}\n"
                f"{'─'*32}\n"
                f"Symbol    : {sym}\n"
                f"Direction : {pos}\n"
                f"Entry     : {round(entry,4)}\n"
                f"Exit      : {round(price,4)}\n"
                f"P&L       : {round(pnl,4)}  ({pct_str})\n"
                f"SL Phase  : {st['sl_phase']}\n"
                f"Total P&L : {round(st['pnl'],4)}\n"
                f"Time(IST) : {ist_now()}"
            )
            st["position"] = st["entry_price"] = st["trailing_sl"] = None

    # ── ENTRY ─────────────────────────────────────────────────────
    elif st["position"] is None:
        signal = check_signal(ha, st["profile"]["doji"])
        if not signal:
            return

        # Score using the strategy's own scoring function
        if strategy_key == "A":
            score, breakdown = score_strategy_a(ha, ind, signal)
        else:
            score, breakdown = score_strategy_b(ha, ind, signal)

        st["last_prob"] = f"{score}%"

        if score < cfg["min_probability"]:
            st["last_prob"] = f"{score}%⚠️"
            return

        mult = cfg["atr_sl_multiplier"]
        st["position"]    = signal
        st["entry_price"] = price
        st["trailing_sl"] = calc_initial_sl(price, ind["atr"], signal, mult)
        st["sl_phase"]    = "1-PROT" if cfg["use_smart_trail"] else "FIXED"

        strike = st["profile"]["strike"]
        atm    = round(price/strike)*strike if strike > 0 else price

        send_alert(
            f"📊 {signal} ENTRY\n"
            f"Strategy   : {cfg['name']}\n"
            f"{'─'*32}\n"
            f"Symbol     : {sym}\n"
            f"Type       : {st['profile']['label']}\n"
            f"Probability: {score}%\n"
            f"Price      : {round(price,4)}\n"
            f"ATM Strike : {atm}\n"
            f"Stop Loss  : {round(st['trailing_sl'],4)}\n"
            f"SL Mode    : {'Smart 3-Phase' if cfg['use_smart_trail'] else 'Fixed ATR'}\n"
            f"ATR        : {round(ind['atr'],4)}\n"
            f"RSI        : {round(ind['rsi'],1)}\n"
            f"Time(IST)  : {ist_now()}\n"
            f"{'─'*32}\n"
            + "\n".join(f"  {k}: {v}" for k,v in breakdown.items())
        )


# ────────────────────────────────────────────────────────────────────
# DASHBOARD
# ────────────────────────────────────────────────────────────────────

def print_dashboard(all_states: dict, fetch_ms: int, sleep_secs: float):
    clear_output(wait=True)
    now = datetime.now(IST).strftime("%d-%b-%Y  %I:%M:%S %p  IST")
    W   = 130

    print(f"\n  ╔{'═'*(W-4)}╗")
    print(f"  ║  🤖  TRADING BOT — DUAL STRATEGY  ·  {now:<{W-42}}║")
    print(f"  ╚{'═'*(W-4)}╝")
    for key, cfg in STRATEGIES.items():
        mode = "Smart 3-Phase SL" if cfg["use_smart_trail"] else "Fixed ATR SL"
        print(f"  [{key}] {cfg['name']:<40}  "
              f"MinProb:{cfg['min_probability']}%  "
              f"ATR×{cfg['atr_sl_multiplier']}  "
              f"TP:{int(cfg['tp_threshold']*100)}%  "
              f"SL:{mode}")
    print(f"  Workers:{MAX_WORKERS}  FetchTime:{fetch_ms}ms\n")

    # ════════════════════════════════════════════════════════════
    # TABLE 1 — LIVE POSITIONS (both strategies side by side)
    # ════════════════════════════════════════════════════════════
    print(f"  {'─'*3} LIVE POSITIONS {'─'*(W-21)}")
    h1 = (f"  {'SYMBOL':<14} {'TYPE':<8} {'MKT':<7} "
          f"│ {'[A] POS':<6} {'ENTRY-A':>10} {'UNREAL-A':>10} {'PROB-A':>7} {'SL-A':>10} {'PH-A':<7} "
          f"│ {'[B] POS':<6} {'ENTRY-B':>10} {'UNREAL-B':>10} {'PROB-B':>7} {'SL-B':>10} {'PH-B':<8}")
    print(h1)
    print("  " + "─" * (len(h1)-2))

    syms = list(all_states["A"].keys())
    for sym in syms:
        sta  = all_states["A"][sym]
        stb  = all_states["B"][sym]
        prof = sta["profile"]
        mkt  = "OPEN" if is_market_open(prof) else "CLOSED"
        typ  = prof["label"]

        def pos_cols(st):
            pos     = st["position"] or "—"
            entry   = st["entry_price"]
            price   = st["latest_price"]
            sl      = st["trailing_sl"]
            phase   = st.get("sl_phase","—") if pos != "—" else "—"
            entry_d = f"{round(entry,2)}" if entry else "—"
            sl_d    = f"{round(sl,2)}"    if sl    else "—"
            prob    = str(st.get("last_prob","—"))
            unreal  = 0.0
            if pos=="CALL" and entry: unreal = price - entry
            elif pos=="PUT" and entry: unreal = entry - price
            u = f"{'▲' if unreal>=0 else '▼'}{abs(round(unreal,2))}"
            return pos, entry_d, u, prob, sl_d, phase

        pa,ea,ua,pra,sla,pha = pos_cols(sta)
        pb,eb,ub,prb,slb,phb = pos_cols(stb)

        print(f"  {sym:<14} {typ:<8} {mkt:<7} "
              f"│ {pa:<6} {ea:>10} {ua:>10} {pra:>7} {sla:>10} {pha:<7} "
              f"│ {pb:<6} {eb:>10} {ub:>10} {prb:>7} {slb:>10} {phb:<8}")

    # ════════════════════════════════════════════════════════════
    # TABLE 2 — STRATEGY COMPARISON SUMMARY
    # ════════════════════════════════════════════════════════════
    print(f"\n  {'─'*3} STRATEGY COMPARISON {'─'*(W-27)}")
    h2 = (f"  {'STRATEGY':<28} {'TRADES':>6} {'WINS':>5} {'LOSSES':>7} "
          f"{'WIN%':>6} {'REALIZED P&L':>14} {'BEST':>10} {'WORST':>10} {'AVG/TRADE':>10}")
    print(h2)
    print("  " + "═" * (len(h2)-2))

    for key, cfg in STRATEGIES.items():
        w=l=0; pnl=0.0; best=float("-inf"); worst=float("inf")
        for sym in syms:
            st = all_states[key][sym]
            w   += st["wins"];  l += st["losses"];  pnl += st["pnl"]
            if st["best"]  != float("-inf"): best  = max(best,  st["best"])
            if st["worst"] != float("inf"):  worst = min(worst, st["worst"])
        tot  = w + l
        wr   = f"{round(w/tot*100,1)}%" if tot > 0 else "—"
        gb   = round(best, 4)  if best  != float("-inf") else "—"
        gwo  = round(worst,4)  if worst != float("inf")  else "—"
        avg  = round(pnl/tot,4) if tot > 0 else "—"
        icon = "📈" if pnl >= 0 else "📉"
        sign = "+" if pnl >= 0 else ""
        print(f"  [{key}] {cfg['short_name']:<24} {tot:>6} {w:>5} {l:>7} "
              f"{wr:>6} {icon}{sign}{round(pnl,4):>12} {str(gb):>10} {str(gwo):>10} {str(avg):>10}")

    # Determine leader
    pnl_a = sum(all_states["A"][s]["pnl"] for s in syms)
    pnl_b = sum(all_states["B"][s]["pnl"] for s in syms)
    if   pnl_a > pnl_b: leader = f"[A] {STRATEGY_A['short_name']} is WINNING 🏆"
    elif pnl_b > pnl_a: leader = f"[B] {STRATEGY_B['short_name']} is WINNING 🏆"
    else:                leader = "Both strategies tied"
    print(f"\n  🏁 Current leader: {leader}")

    # ════════════════════════════════════════════════════════════
    # TABLE 3 — PER-SYMBOL PERFORMANCE (both strategies)
    # ════════════════════════════════════════════════════════════
    print(f"\n  {'─'*3} PER-SYMBOL PERFORMANCE {'─'*(W-29)}")
    h3 = (f"  {'SYMBOL':<14} "
          f"│ {'[A] T':>5} {'W':>3} {'L':>3} {'WR':>5} {'P&L':>10} "
          f"│ {'[B] T':>5} {'W':>3} {'L':>3} {'WR':>5} {'P&L':>10} "
          f"│ {'EDGE':>6}")
    print(h3)
    print("  " + "─" * (len(h3)-2))

    for sym in syms:
        sa, sb = all_states["A"][sym], all_states["B"][sym]
        def fmt(st):
            w=st["wins"]; l=st["losses"]; tot=w+l
            wr=f"{round(w/tot*100)}%" if tot>0 else "—"
            sign="+" if st["pnl"]>=0 else ""
            return f"{tot:>5} {w:>3} {l:>3} {wr:>5} {sign}{round(st['pnl'],2):>9}"
        edge = "A▲" if sa["pnl"]>sb["pnl"] else ("B▲" if sb["pnl"]>sa["pnl"] else "TIE")
        print(f"  {sym:<14} │ {fmt(sa)} │ {fmt(sb)} │ {edge:>6}")

    # ════════════════════════════════════════════════════════════
    # TABLE 4 — TRADE LOG (all strategies, newest first)
    # ════════════════════════════════════════════════════════════
    all_trades = []
    for key in STRATEGIES:
        for st in all_states[key].values():
            all_trades.extend(st["trade_log"])
    all_trades.sort(key=lambda t: t["time"], reverse=True)

    print(f"\n  {'─'*3} TRADE LOG  ({len(all_trades)} trades) {'─'*(W-32)}")
    if not all_trades:
        print("  No completed trades yet.\n")
    else:
        h4 = (f"  {'#':>3}  {'TIME (IST)':<26} {'STRAT':<12} {'SYMBOL':<14} "
              f"{'DIR':<5} {'ENTRY':>10} {'EXIT':>10} {'P&L':>10} {'%':>6} {'R':<5} {'BY':<4}")
        print(h4)
        print("  " + "─" * (len(h4)-2))
        for i, t in enumerate(all_trades[:50], 1):    # show last 50
            entry=t["entry"]; pnl=t["pnl"]
            pct=round(pnl/entry*100,2) if entry!=0 else 0
            ri="✅" if t["result"]=="WIN" else "❌"
            s="+" if pnl>=0 else ""
            print(f"  {i:>3}  {t['time']:<26} {t['strategy']:<12} {t['symbol']:<14} "
                  f"{t['dir']:<5} {entry:>10} {t['exit']:>10} {s}{pnl:>9} "
                  f"{s}{pct:>5}% {ri}{t['result']:<4} {t['reason']:<4}")

    print(f"\n  ⏳ Next refresh in {round(sleep_secs)}s  "
          f"·  Fetch:{fetch_ms}ms  ·  Ctrl+C to stop\n")


# ────────────────────────────────────────────────────────────────────
# CATEGORY SELECTOR
# ────────────────────────────────────────────────────────────────────

def ask_categories() -> list:
    if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
        print(f"  GitHub Actions — using GITHUB_ACTIONS_SYMBOLS from config.py")
        return list(dict.fromkeys(GITHUB_ACTIONS_SYMBOLS))

    print("\n" + "═"*65)
    print("  🤖  DUAL STRATEGY BOT — SELECT CATEGORIES")
    print("═"*65 + "\n")
    for key, cat in CATEGORIES.items():
        prev = ", ".join(cat["symbols"][:3])
        if len(cat["symbols"]) > 3: prev += f" +{len(cat['symbols'])-3} more"
        print(f"    {key}.  {cat['name']:<35} ({prev})")
    print("\n  Enter numbers (e.g. 1,2,3) or 'all'\n")

    while True:
        try:
            raw = input("  Your choice: ").strip().lower()
        except EOFError:
            raw = "all"
        if not raw:
            continue
        if raw == "all":
            syms = [s for cat in CATEGORIES.values() for s in cat["symbols"]]
            break
        parts = [p.strip() for p in raw.split(",")]
        if any(p not in CATEGORIES for p in parts):
            print(f"  ⚠️  Valid options: {', '.join(CATEGORIES.keys())} or 'all'")
            continue
        syms = [s for p in parts for s in CATEGORIES[p]["symbols"]]
        if syms: break

    final = list(dict.fromkeys(syms))
    print(f"\n  ✅ {len(final)} symbols selected: {', '.join(final)}\n")
    return final


# ════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════════

def start_bot():
    selected_symbols = ask_categories()

    print(f"\n🚀 Bot starting  ·  {ist_now()}")
    print(f"   {len(selected_symbols)} symbols  ·  {MAX_WORKERS} workers  ·  2 strategies\n")

    send_alert(
        f"🤖 Dual Strategy Bot Started\n{'─'*32}\n"
        f"Symbols     : {', '.join(selected_symbols)}\n"
        f"Interval    : {INTERVAL}\n"
        f"[A] {STRATEGY_A['name']}\n"
        f"    MinProb:{STRATEGY_A['min_probability']}%  "
        f"ATR×{STRATEGY_A['atr_sl_multiplier']}  "
        f"TP:{int(STRATEGY_A['tp_threshold']*100)}%  SL:Fixed\n"
        f"[B] {STRATEGY_B['name']}\n"
        f"    MinProb:{STRATEGY_B['min_probability']}%  "
        f"ATR×{STRATEGY_B['atr_sl_multiplier']}  "
        f"TP:{int(STRATEGY_B['tp_threshold']*100)}%  SL:Smart 3-Phase\n"
        f"Time (IST)  : {ist_now()}"
    )

    # Initialise independent states for each strategy × symbol
    # Structure: all_states["A"]["^NSEI"] = {...}
    #            all_states["B"]["^NSEI"] = {...}
    all_states = {
        key: {sym: make_state(sym) for sym in selected_symbols}
        for key in STRATEGIES
    }

    fetch_ms = 0

    while True:
        cycle_start = time.time()

        # Reset daily counters at midnight
        today = datetime.now().date()
        for key in STRATEGIES:
            for st in all_states[key].values():
                if today != st["last_day"]:
                    st["daily_losses"] = 0
                    st["last_day"]     = today

        # Active symbols: open market, and at least one strategy not paused
        active = [
            sym for sym in selected_symbols
            if is_market_open(all_states["A"][sym]["profile"])
            and any(
                all_states[key][sym]["daily_losses"] < STRATEGIES[key]["max_daily_loss"]
                for key in STRATEGIES
            )
        ]

        if active:
            # ── Fetch all active symbols in parallel ─────────────
            t0       = time.time()
            data_map = fetch_all_parallel(active, INTERVAL)
            fetch_ms = int((time.time() - t0) * 1000)

            # ── Process each symbol × each strategy ──────────────
            # HA and indicators computed ONCE per symbol, shared
            # between both strategies — avoids double computation.
            for sym in active:
                df = data_map.get(sym)
                if df is None:
                    continue

                # Skip if same candle already processed by ALL strategies
                new_candle = df.index[-1]
                if all(all_states[key][sym]["last_time"] == new_candle
                       for key in STRATEGIES):
                    del df
                    continue

                # Compute HA and indicators once
                try:
                    ha  = heikin_ashi(df)
                    ind = compute_indicators(df) if ha is not None else None
                except Exception as e:
                    print(f"⚠️  [{sym}] Compute error: {e}")
                    del df
                    continue

                if ha is None or ind is None:
                    del df
                    continue

                # Run both strategies on the same data
                for key in STRATEGIES:
                    st = all_states[key][sym]
                    if st["last_time"] == new_candle:
                        continue
                    if st["daily_losses"] >= STRATEGIES[key]["max_daily_loss"]:
                        continue
                    try:
                        process_symbol(sym, df, st, ha, ind, key)
                        st["last_time"] = new_candle
                    except Exception as e:
                        print(f"⚠️  [{sym}][{key}] Error: {e}")

                del ha, ind, df
                data_map[sym] = None

            del data_map
            gc.collect()

        else:
            fetch_ms = 0
            print(f"  All markets closed — {ist_now()}")
            time.sleep(60)
            continue

        # ── Send hourly comparison to Telegram ───────────────────
        now_min = datetime.now().minute
        if now_min == 0:   # top of every hour
            pnl_a = sum(all_states["A"][s]["pnl"] for s in selected_symbols)
            pnl_b = sum(all_states["B"][s]["pnl"] for s in selected_symbols)
            w_a   = sum(all_states["A"][s]["wins"] for s in selected_symbols)
            w_b   = sum(all_states["B"][s]["wins"] for s in selected_symbols)
            l_a   = sum(all_states["A"][s]["losses"] for s in selected_symbols)
            l_b   = sum(all_states["B"][s]["losses"] for s in selected_symbols)
            t_a   = w_a + l_a;  t_b = w_b + l_b
            wr_a  = f"{round(w_a/t_a*100,1)}%" if t_a>0 else "—"
            wr_b  = f"{round(w_b/t_b*100,1)}%" if t_b>0 else "—"
            lead  = "A leads 🏆" if pnl_a>pnl_b else ("B leads 🏆" if pnl_b>pnl_a else "TIE")
            send_alert(
                f"📊 Hourly Strategy Comparison\n{'─'*32}\n"
                f"[A] {STRATEGY_A['short_name']}\n"
                f"    Trades:{t_a}  W:{w_a}  L:{l_a}  WR:{wr_a}  P&L:{round(pnl_a,2)}\n"
                f"[B] {STRATEGY_B['short_name']}\n"
                f"    Trades:{t_b}  W:{w_b}  L:{l_b}  WR:{wr_b}  P&L:{round(pnl_b,2)}\n"
                f"{'─'*32}\n"
                f"Leader  : {lead}\n"
                f"Time    : {ist_now()}"
            )

        elapsed    = time.time() - cycle_start
        sleep_secs = max(5.0, seconds_until_next_5min() - elapsed)
        print_dashboard(all_states, fetch_ms, sleep_secs)
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
