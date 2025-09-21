# bot.py
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

# ============ ENV ============
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")           # BotFather token (paste only in Render env)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")         # e.g. @yourpublicchannel OR -100xxxxxxxxxx
TIMEFRAME        = os.getenv("TIMEFRAME", "5m")          # "5m" or "15m"
MAX_PAIRS        = int(os.getenv("MAX_PAIRS", "300"))    # how many symbols to scan
SLEEP_SECONDS    = int(os.getenv("SLEEP_SECONDS", "15")) # main loop pause
MIN_QVOL         = float(os.getenv("MIN_QVOL", "15000")) # last candle quote-volume filter
COOLDOWN         = int(os.getenv("COOLDOWN", "3"))       # candles to wait before same-side re-signal
QUOTE_ASSETS     = os.getenv("QUOTE_ASSETS", "USDT,BUSD").split(",")

# Skip pairs where BASE is a stable/fiat (we want BTC/ETH/SOL/... not USDT base)
BASE_STABLES = set(os.getenv(
    "BASE_STABLES",
    "USDT,USDC,BUSD,FDUSD,TUSD,DAI,USDE,PYUSD,EUR,TRY,BRL"
).split(","))

# ========= BASIC CHECKS =========
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise SystemExit("Please set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables on Render.")

# public-only client (no trading, no api secret needed)
binance = Client()
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("signals")

def utcnow():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# ========= INDICATORS =========
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

# ========= DATA FETCH =========
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
    df = pd.DataFrame(kl, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qvol','num_trades','tb_base','tb_quote','ignore'
    ])
    for c in ['open','high','low','close','volume','qvol']:
        df[c] = df[c].astype(float)
    return df

# ========= STRATEGY =========
def analyze(df: pd.DataFrame):
    """EMA20/50 cross + RSI/MACD/Stoch filters + ATR-based SL/TP."""
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

    rsi_v = float(rsi.iloc[i])
    k_v   = float(k.iloc[i])
    macd_v= float(macd.iloc[i])
    price = float(c.iloc[i])
    atr_v = max(float(atr.i_
