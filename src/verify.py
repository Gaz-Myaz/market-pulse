"""Verification of real predictions.

Once enough wall-clock time has passed since a real prediction was made, we
fetch the actual price move and record whether each horizon (24h / 1w / 1m)
was correct. Backtests are not verified here — they already know their outcome.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from src.storage import get_unverified_real_predictions, update_verification

logger = logging.getLogger("market-pulse")

# horizon key -> number of calendar days that must elapse
HORIZONS = {"24h": 1, "1w": 7, "1m": 30}


def get_actual_direction(ticker: str, from_date: str, to_date: str) -> str | None:
    """Compare close prices on/near two dates; return 'UP'/'DOWN' or None.

    Pads the download window on both sides so that if `from_date` or `to_date`
    falls on a weekend/holiday we still find the nearest trading-day close
    (otherwise verification would silently never complete).
    """
    try:
        start = pd.Timestamp(from_date).date() - pd.Timedelta(days=7)
        end = pd.Timestamp(to_date).date() + pd.Timedelta(days=7)
        data = yf.download(
            ticker, start=str(start), end=str(end), auto_adjust=True, progress=False
        )
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        from_ts = pd.Timestamp(from_date)
        to_ts = pd.Timestamp(to_date)
        before = data[data.index <= from_ts]
        after = data[data.index >= to_ts]
        if before.empty or after.empty:
            return None
        from_close = float(before.iloc[-1]["Close"])
        to_close = float(after.iloc[0]["Close"])
        return "UP" if to_close > from_close else "DOWN"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"get_actual_direction failed for {ticker}: {e}")
        return None


def verify_pending_predictions() -> int:
    """Verify any real prediction horizons whose time window has elapsed.

    Returns the number of horizon updates written.
    """
    updates = 0
    try:
        pending = get_unverified_real_predictions()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Could not load pending predictions: {e}")
        return 0

    now = pd.Timestamp.now(tz="UTC")
    for pred in pending:
        created = pred.get("created_at")
        if not created:
            continue
        created_ts = pd.Timestamp(created)
        if created_ts.tzinfo is None:
            created_ts = created_ts.tz_localize("UTC")
        ticker = pred["ticker"]
        # The predicted direction we compare against (XGBoost is the main model).
        predicted = pred.get("xgb_direction")

        # Anchor the price comparison on the reference date the model used
        # (backtest_date), falling back to the creation date for older rows.
        anchor = pd.Timestamp(pred.get("backtest_date") or created_ts.date())

        for horizon, days in HORIZONS.items():
            if pred.get(f"verified_{horizon}") is not None:
                continue
            # Timing gate: only verify once enough real time has elapsed.
            if now < created_ts + pd.Timedelta(days=days):
                continue

            actual = get_actual_direction(
                ticker,
                from_date=str(anchor.date()),
                to_date=str((anchor + pd.Timedelta(days=days)).date()),
            )
            if actual is None:
                continue  # price data not available yet — try again later

            correct = (predicted == actual)
            try:
                update_verification(pred["id"], horizon, correct, actual)
                updates += 1
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Verification update failed for {pred['id']}: {e}")

    return updates
