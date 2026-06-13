"""Data layer — yfinance OHLCV download, feature engineering, trading-day helpers.

The feature set is fixed (FEATURE_COLS) and validated after engineering so the
model never silently trains on the wrong columns if pandas-ta renames an
indicator.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
import streamlit as st

logger = logging.getLogger("market-pulse")

# pandas-ta imports must come after pandas; it monkey-patches the DataFrame.
import pandas_ta as ta  # noqa: E402,F401


# The exact, ordered feature set used everywhere (training, prediction, display).
FEATURE_COLS = [
    "RSI_14",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
    "BBP_5_2.0",
    "ma20_ratio",
    "ma50_ratio",
    "ma_cross",
    "volume_ratio",
    "price_change_5d",
    "sentiment_score",
]

# Crypto trades 24/7 — the NYSE calendar would skip valid days for these.
CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "DOGE-USD"}


# --------------------------------------------------------------------------- #
# Manual indicator fallbacks (used if pandas-ta misbehaves on newer pandas)
# --------------------------------------------------------------------------- #
def _manual_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _manual_macd(close: pd.Series) -> pd.DataFrame:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return pd.DataFrame(
        {
            "MACD_12_26_9": macd,
            "MACDh_12_26_9": hist,
            "MACDs_12_26_9": signal,
        }
    )


def _manual_bbp(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.Series:
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std()
    upper = mid + std * sd
    lower = mid - std * sd
    return (close - lower) / (upper - lower).replace(0, np.nan)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append RSI/MACD/Bollinger %B/SMA columns, with manual fallbacks."""
    # RSI
    try:
        df.ta.rsi(length=14, append=True)
        if "RSI_14" not in df.columns:
            raise AttributeError("RSI_14 missing")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"pandas-ta RSI failed ({e}); using manual RSI.")
        df["RSI_14"] = _manual_rsi(df["Close"])

    # MACD
    try:
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        if "MACD_12_26_9" not in df.columns:
            raise AttributeError("MACD columns missing")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"pandas-ta MACD failed ({e}); using manual MACD.")
        df = df.join(_manual_macd(df["Close"]))

    # Bollinger %B
    try:
        df.ta.bbands(length=20, std=2, append=True)
        if "BBP_5_2.0" not in df.columns:
            # pandas-ta names it BBP_20_2.0 in some versions — normalise.
            alt = [c for c in df.columns if c.startswith("BBP_")]
            if alt:
                df["BBP_5_2.0"] = df[alt[0]]
            else:
                raise AttributeError("BBP missing")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"pandas-ta bbands failed ({e}); using manual %B.")
        df["BBP_5_2.0"] = _manual_bbp(df["Close"])

    # Simple moving averages
    try:
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
    except Exception:  # noqa: BLE001
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
    if "SMA_20" not in df.columns:
        df["SMA_20"] = df["Close"].rolling(20).mean()
    if "SMA_50" not in df.columns:
        df["SMA_50"] = df["Close"].rolling(50).mean()

    return df


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns a MultiIndex (single-ticker) — flatten it."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_engineer(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download OHLCV data and build the full feature set.

    A `sentiment_score` column is added (default 0.0); the live sentiment value
    is injected at prediction time. Raises ValueError on empty/invalid data.
    """
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError("No data returned for ticker")

    df = _flatten_columns(raw.copy())
    df = _add_indicators(df)

    # Derived features
    df["ma20_ratio"] = df["Close"] / df["SMA_20"] - 1
    df["ma50_ratio"] = df["Close"] / df["SMA_50"] - 1
    df["ma_cross"] = (df["SMA_20"] > df["SMA_50"]).astype(int)
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["price_change_5d"] = df["Close"].pct_change(5)

    # Placeholder sentiment column — overwritten with the live score at predict.
    df["sentiment_score"] = 0.0

    # Target: does the next day close higher than today?
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Guard against indicator name drift before we drop rows.
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after engineering: {missing}")

    df = df.dropna(subset=[c for c in FEATURE_COLS if c != "sentiment_score"])
    return df


def get_latest_features(ticker: str, sentiment_score: float = 0.0) -> dict:
    """Return the most recent value of every FEATURE_COL for display/predict."""
    df = fetch_and_engineer(ticker)
    if df.empty:
        raise ValueError("No data returned for ticker")
    last = df.iloc[-1]
    feats = {col: float(last[col]) for col in FEATURE_COLS}
    feats["sentiment_score"] = float(sentiment_score)
    return feats


# --------------------------------------------------------------------------- #
# Trading-day helpers
# --------------------------------------------------------------------------- #
def get_last_trading_day() -> str:
    """Most recent NYSE trading day as YYYY-MM-DD."""
    nyse = mcal.get_calendar("NYSE")
    today = pd.Timestamp.now(tz="US/Eastern").date()
    schedule = nyse.schedule(
        start_date=str(today - pd.Timedelta(days=7)), end_date=str(today)
    )
    if schedule.empty:
        return str(today)
    return str(schedule.index[-1].date())


def get_next_trading_day(from_date: str | None = None) -> str:
    """Next NYSE trading day after `from_date` (default: today)."""
    nyse = mcal.get_calendar("NYSE")
    start = pd.Timestamp(from_date or pd.Timestamp.now(tz="US/Eastern").date())
    schedule = nyse.schedule(
        start_date=str(start + pd.Timedelta(days=1)),
        end_date=str(start + pd.Timedelta(days=10)),
    )
    if schedule.empty:
        return str((start + pd.Timedelta(days=1)).date())
    return str(schedule.index[0].date())


def get_reference_date(ticker: str) -> str:
    """Reference 'today' for training — calendar-aware, crypto-aware."""
    if ticker in CRYPTO_TICKERS:
        return str(pd.Timestamp.now().date())
    return get_last_trading_day()
