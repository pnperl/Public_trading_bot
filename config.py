# ╔══════════════════════════════════════════════════════════════════╗
# ║                        config.py                                ║
# ║                                                                  ║
# ║  THIS IS THE ONLY FILE YOU EVER NEED TO EDIT.                   ║
# ║                                                                  ║
# ║  Contains:                                                       ║
# ║    • All symbols organised by category                          ║
# ║    • Strategy parameters for both strategies                    ║
# ║    • Bot-wide settings (interval, workers, etc.)                ║
# ╚══════════════════════════════════════════════════════════════════╝


# ════════════════════════════════════════════════════════════════════
# SECTION 1 — SYMBOL CATEGORIES
# Add / remove symbols here. Bot will ask you to pick categories
# at startup when running locally. On GitHub Actions it uses
# GITHUB_ACTIONS_SYMBOLS (see Section 3).
# ════════════════════════════════════════════════════════════════════

CATEGORIES = {

    "1": {
        "name"   : "Indian Indices",
        "symbols": [
            "^NSEI",        # NIFTY 50
            "^NSEBANK",     # NIFTY Bank
            "^CNXCMDT",     # NIFTY Commodities
        ],
    },

    "2": {
        "name"   : "Indian Stocks",
        "symbols": [
            "RELIANCE.NS",
            "HDFCBANK.NS",
            "TCS.NS",
            "INFY.NS",
            "MCX.NS",
            "ICICIBANK.NS",
        ],
    },

    "3": {
        "name"   : "Global Commodity Futures",
        "symbols": [
            "GC=F",         # Gold
            "SI=F",         # Silver
            "CL=F",         # WTI Crude Oil
            "NG=F",         # Natural Gas
            "HG=F",         # Copper
        ],
    },

    "4": {
        "name"   : "US & Global Index Futures",
        "symbols": [
            "YM=F",         # Dow Jones Futures  ← NEW
            "ES=F",         # S&P 500 Futures
            "NQ=F",         # NASDAQ Futures
            "RTY=F",        # Russell 2000 Futures
        ],
    },

    "5": {
        "name"   : "Crypto",
        "symbols": [
            "BTC-USD",
            "ETH-USD",
            "SOL-USD",
            "BNB-USD",
        ],
    },

    "6": {
        "name"   : "Forex",
        "symbols": [
            "INR=X",        # USD/INR
            "EURUSD=X",     # EUR/USD
            "GBPUSD=X",     # GBP/USD
            "JPYUSD=X",     # JPY/USD
        ],
    },

    "7": {
        "name"   : "US Stocks",
        "symbols": [
            "AAPL",
            "TSLA",
            "NVDA",
            "MSFT",
            "AMZN",
        ],
    },

}


# ════════════════════════════════════════════════════════════════════
# SECTION 2 — STRATEGY PARAMETERS
#
# Two strategies run simultaneously on every symbol.
# Each has completely independent entry/exit settings.
# Performance is tracked and compared separately.
# ════════════════════════════════════════════════════════════════════

STRATEGY_A = {
    # ── Identity ────────────────────────────────────────────────
    "name"             : "Strategy-A  [Classic HA]",
    "short_name"       : "A-Classic",
    "description"      : (
        "Original Heikin Ashi reversal — simple 2-candle colour flip. "
        "No probability filter. Fixed ATR stop loss. "
        "Entry on any clean HA reversal candle."
    ),

    # ── Entry filter ────────────────────────────────────────────
    # MIN_PROBABILITY for classic strategy is set low because the
    # classic strategy uses a simpler 3-factor score (no MACD/BB/ST).
    # Setting to 0 means it enters on every clean HA flip.
    "min_probability"  : 0,         # 0 = enter on every HA reversal

    # ── Stop loss ────────────────────────────────────────────────
    # Classic strategy: fixed ATR multiplier SL, no phase trailing
    "atr_sl_multiplier": 1.5,       # SL = ATR × this value from entry
    "use_smart_trail"  : False,     # False = fixed ATR trail only

    # ── Take profit ─────────────────────────────────────────────
    "tp_threshold"     : 0.02,      # 2% take profit

    # ── Risk management ─────────────────────────────────────────
    "max_daily_loss"   : 30,        # max losses per symbol per day
}


STRATEGY_B = {
    # ── Identity ────────────────────────────────────────────────
    "name"             : "Strategy-B  [Smart Multi-Indicator]",
    "short_name"       : "B-Smart",
    "description"      : (
        "Enhanced strategy with 8-factor probability scoring. "
        "3 hard veto conditions (Supertrend, RSI extreme, Chop). "
        "3-phase smart trailing SL (protect → breakeven → lock-in)."
    ),

    # ── Entry filter ────────────────────────────────────────────
    "min_probability"  : 55,        # only enter if score ≥ 55%

    # ── Stop loss ────────────────────────────────────────────────
    "atr_sl_multiplier": 2.0,       # wider initial SL for smart trail
    "use_smart_trail"  : True,      # True = 3-phase trailing SL

    # ── Take profit ─────────────────────────────────────────────
    "tp_threshold"     : 0.02,      # 2% take profit

    # ── Risk management ─────────────────────────────────────────
    "max_daily_loss"   : 30,
}


# ════════════════════════════════════════════════════════════════════
# SECTION 3 — BOT-WIDE SETTINGS
# ════════════════════════════════════════════════════════════════════

# Candle interval — do not change unless you know what you're doing
INTERVAL     = "5m"

# Parallel fetch workers — max 8, Yahoo Finance limit
MAX_WORKERS  = 5

# Symbols used when running on GitHub Actions (non-interactive).
# Copy-paste symbol names from CATEGORIES above.
GITHUB_ACTIONS_SYMBOLS = [
    "^NSEI",
    "^NSEBANK",
    #"^CNXCMDT",
    #"RELIANCE.NS",
    #"HDFCBANK.NS",
    #"GC=F",
    #"SI=F",
    #"CL=F",
    #"YM=F",         # Dow Futures
    #"ES=F",         # S&P 500 Futures
    #"BTC-USD",
    #"ETH-USD",
    #"INR=X",
]
