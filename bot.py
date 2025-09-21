import os
import time
import logging
import threading
from datetime import datetime
from collections import deque

import pandas as pd
import numpy as np
from binance.client import Client
import telebot
from fastapi import FastAPI

# =======================
# ENV VARS (set these on Render)
# =======================
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")         # BotFather token
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")       # e.g. @yourchannel OR -100xxxxxxxxxx
TIMEFRAME        = os.getenv("TIMEFRAME", "5m")        # "5m" or "15m"
MAX_PAIRS        = int(os.getenv("MAX_PAIRS", "300"))  # how many symbols to scan
SLEEP_SECONDS    = int(os.getenv("SLEEP_SECONDS", "15"))
MIN_QVOL         = float(os.getenv("MIN_QVOL", "15000"))  # last-candle quote volume filter
COOLDOWN         = int(os.getenv("COOLDOWN", "3"))        # candles to wait before repeating same-side signal
QUOTE_ASSETS     = os.getenv("QUOTE_ASSETS", "USDT,BUSD").split(",")

# Skip pairs whose BASE coin is a stable/fiat (we want BTC/ETH/SOL etc., not USDT base)
BASE_STABLES = set(
    os.getenv(
        "BASE_STABLES",
        "USDT,USDC,BUSD,FDUSD,TUSD,DAI,USDE,PYUSD,EUR,TRY,BRL"
    ).split(",")
)

# =======================
# BASIC CHECKS
# =======================
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise SystemExit("Please set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables.")

# Public-only client (no trading, no secret needed)
binance = Client()  # uses public endpoints for market data
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("signals")

def utcnow() -> StringError:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# =======================
# INDICATORS
# =======================
def EMA(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def RSI(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-9)
    return 100 - (100 / (1 + rs))

def MACD(s: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ef, es = EMA(s, fast), EMA(s, slow)
    macd = ef - es
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def STOCH(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    fast_k = (close - ll) * 100 / (hh - ll + 1e-9)
    fast_d = fast_k.rolling(d).mean()
    return fast_k, fast_d

def ATR(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# =======================
# DATA FETCH
# =======================
def get_symbols():
    """All spot symbols quoted in QUOTE_ASSETS, with non-stable BASE coin."""
    info = binance.get_exchange_info()
    out = []
    for s in info["symbols"]:
        if s["status"] != "TRADING" or not s.get("isSpotTradingAllowed"):
            continue
        base, quote = s["baseAsset"], s["quoteAsset"]
        if quote not in QUOTE_ASSETS:
            continue
        if base in BASE_STABLES:
            continue
        out.append(s["symbol"])
    return out

def klines_df(symbol: str, interval: str = "5m", limit: int = 200) -> pd.DataFrame:
    kl = binance.get_klines(symbol=symbol, interval=interval, limit=limit)
    if not kl:
        return pd.DataFrame()
    df = pd.DataFrame(
        kl,
        columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qvol', 'num_trades', 'tb_base', 'tb_quote', 'ignore'
        ],
    )
    for c in ['open', 'high', 'low', 'close', 'volume', 'qvol']:
        df[c] = df[c].astype(float)
    return df

# =======================
# STRATEGY
# =======================
def analyze(df: pd.DataFrame):
    """EMA20/50 cross + RSI/MACD/Stoch filters + ATR targets"""
    if df.empty or len(df) < 80:
        return None
    if df['qvol'].iloc[-1] < MIN_QVOL:
        return None

    c = df['close']; h = df['high']; l = df['low']
    ema20, ema50 = EMA(c, 20), EMA(c, 50)
    rsi = RSI(c, 14)
    macd, macds, hist = MACD(c)
    k, d = STOCH(h, l, c)
    atr = ATR(df, 14)

    i = len(df) - 1  # last CLOSED candle
    bull_cross = (ema20.iloc[i-1] < ema50.iloc[i-1]) and (ema20.iloc[i] > ema50.iloc[i])
    bear_cross = (ema20.iloc[i-1] > ema50.iloc[i-1]) and (ema20.iloc[i] < ema50.iloc[i])

    rsi_v = rsi.iloc[i]
    k_v = k.iloc[i]
    macd_v = macd.iloc[i]
    price = c.iloc[i]
    atr_v = max(float(atr.iloc[i]), 1e-8)

    # Filters to avoid overextended entries
    if bull_cross and (rsi_v < 68) and (k_v < 85):
        return {
            "side": "BUY", "price": price,
            "sl": round(price - atr_v, 8),
            "tp1": round(price + atr_v, 8),       # 1:1
            "tp2": round(price + 2 * atr_v, 8),   # 1:2
            "rsi": float(rsi_v), "stoch_k": float(k_v), "macd": float(macd_v),
            "bull": True,
        }
    if bear_cross and (rsi_v > 32) and (k_v > 15):
        return {
            "side": "SELL", "price": price,
            "sl": round(price + atr_v, 8),
            "tp1": round(price - atr_v, 8),       # 1:1
            "tp2": round(price - 2 * atr_v, 8),   # 1:2
            "rsi": float(rsi_v), "stoch_k": float(k_v), "macd": float(macd_v),
            "bull": False,
        }
    return None

# =======================
# TELEGRAM POSTING
# =======================
LAST_SIDE = {}          # symbol -> last side posted
RECENT = {}             # symbol -> deque of recent close_time to avoid duplicates

def reason_text(r: dict) -> str:
    dirw = "bullish" if r["bull"] else "bearish"
    macd_side = ">0" if r["macd"] > 0 else "<0"
    return f"EMA20/50 {dirw} cross | RSI {r['rsi']:.1f} | MACD {macd_side} | Stoch%K {r['stoch_k']:.0f}"

def post_signal(symbol: str, r: dict):
    text = (
        f"📡 <b>SIGNAL (SCALP {TIMEFRAME})</b>\n"
        f"Pair: <code>{symbol}</code>\n"
        f"Side: <b>{r['side']}</b>\n"
        f"Entry: <code>{r['price']}</code>\n"
        f"SL: <code>{r['sl']}</code>\n"
        f"TP1: <code>{r['tp1']}</code> (1:1)   TP2: <code>{r['tp2']}</code> (1:2)\n"
        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Reason: {reason_text(r)}\n"
        f"⚠️ DYOR | Info only | Target RR 1:2"
    )
    try:
        bot.send_message(TELEGRAM_CHAT_ID, text)
        LAST_SIDE[symbol] = r["side"]
    except Exception as e:
        log.error(f"Telegram error: {e}")

# =======================
# MAIN SCAN LOOP
# =======================
def scan_loop():
    # fetch list of symbols once (spot-only)
    try:
        symbols = get_symbols()
        symbols = symbols[:MAX_PAIRS]  # cap for CPU
        log.info(f"Scanning {len(symbols)} symbols on {TIMEFRAME}")
    except Exception as e:
        log.error(f"Exchange info failed, fallback to majors: {e}")
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BCHUSDT"]

    while True:
        t0 = time.time()
        for s in symbols:
            try:
                df = klines_df(s, TIMEFRAME, limit=200)
                if df.empty:
                    continue

                res = analyze(df)
                if not res:
                    continue

                ct = int(df["close_time"].iloc[-1])
                dq = RECENT.get(s, deque(maxlen=COOLDOWN))
                # avoid duplicate for same candle
                if ct in dq:
                    continue
                # avoid spamming the same side repeatedly
                if LAST_SIDE.get(s) == res["side"] and len(dq) < COOLDOWN:
                    dq.append(ct)
                    RECENT[s] = dq
                    continue

                post_signal(s, res)
                dq.append(ct)
                RECENT[s] = dq

            except Exception as e:
                # keep going even if one symbol fails
                log.debug(f"{s} error: {e}")
                continue

        elapsed = time.time() - t0
        time.sleep(max(1, SLEEP_SECONDS - elapsed))

# =======================
# WEB APP (for Render health)
# =======================
app = FastAPI()

@app.get("/")
def health():
    return {"ok": True, "timeframe": TIMEFRAME}

def start_worker():
    th = threading.Thread(target=scan_loop, daemon=True)
    th.start()

start_worker()
