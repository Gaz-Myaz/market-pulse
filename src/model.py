"""ML pipeline — Logistic Regression (baseline) + XGBoost (main).

Strictly time-ordered: train on the past, test on the most recent slice, never
shuffle. LR runs on standardized features; XGBoost runs on raw features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import streamlit as st

from src.data import FEATURE_COLS, fetch_and_engineer

logger = logging.getLogger("market-pulse")


def get_models() -> tuple[LogisticRegression, XGBClassifier]:
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    return lr, xgb


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Time-split train/test (80/20), fit both models, return models + metrics."""
    data = df.dropna(subset=["target"]).copy()
    X = data[FEATURE_COLS].fillna(0.0)
    y = data["target"].astype(int)

    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr, xgb = get_models()
    lr.fit(X_train_scaled, y_train)
    xgb.fit(X_train, y_train)  # trees don't need scaling

    lr_pred = lr.predict(X_test_scaled)
    xgb_pred = xgb.predict(X_test)

    return {
        "lr_model": lr,
        "xgb_model": xgb,
        "scaler": scaler,
        "lr_metrics": _metrics(y_test, lr_pred),
        "xgb_metrics": _metrics(y_test, xgb_pred),
        "lr_cm": confusion_matrix(y_test, lr_pred, labels=[0, 1]),
        "xgb_cm": confusion_matrix(y_test, xgb_pred, labels=[0, 1]),
        "feature_importance": dict(zip(FEATURE_COLS, xgb.feature_importances_)),
        "test_size": len(X_test),
        "train_size": len(X_train),
    }


def train_on_full_data(df: pd.DataFrame) -> dict:
    """Train both models on ALL rows (used for real predictions).

    Returns the same structure as train_and_evaluate; metrics here are computed
    on a held-out 20% tail so the UI still has something meaningful to show.
    """
    data = df.dropna(subset=["target"]).copy()
    X = data[FEATURE_COLS].fillna(0.0)
    y = data["target"].astype(int)

    # Metrics from a time-split, for display only.
    split = int(len(data) * 0.8)
    scaler_eval = StandardScaler().fit(X.iloc[:split])
    lr_eval, xgb_eval = get_models()
    lr_eval.fit(scaler_eval.transform(X.iloc[:split]), y.iloc[:split])
    xgb_eval.fit(X.iloc[:split], y.iloc[:split])
    lr_pred = lr_eval.predict(scaler_eval.transform(X.iloc[split:]))
    xgb_pred = xgb_eval.predict(X.iloc[split:])
    lr_metrics = _metrics(y.iloc[split:], lr_pred)
    xgb_metrics = _metrics(y.iloc[split:], xgb_pred)
    lr_cm = confusion_matrix(y.iloc[split:], lr_pred, labels=[0, 1])
    xgb_cm = confusion_matrix(y.iloc[split:], xgb_pred, labels=[0, 1])

    # Final models fit on everything.
    scaler = StandardScaler().fit(X)
    lr, xgb = get_models()
    lr.fit(scaler.transform(X), y)
    xgb.fit(X, y)

    return {
        "lr_model": lr,
        "xgb_model": xgb,
        "scaler": scaler,
        "lr_metrics": lr_metrics,
        "xgb_metrics": xgb_metrics,
        "lr_cm": lr_cm,
        "xgb_cm": xgb_cm,
        "feature_importance": dict(zip(FEATURE_COLS, xgb.feature_importances_)),
        "test_size": len(X.iloc[split:]),
        "train_size": len(X),
    }


def train_for_prediction(df: pd.DataFrame) -> dict:
    """Fit only the final models needed to predict (no eval split / metrics).

    Roughly half the work of train_on_full_data — used inside backtests where
    per-day metrics are not displayed.
    """
    data = df.dropna(subset=["target"]).copy()
    X = data[FEATURE_COLS].fillna(0.0)
    y = data["target"].astype(int)

    scaler = StandardScaler().fit(X)
    lr, xgb = get_models()
    lr.fit(scaler.transform(X), y)
    xgb.fit(X, y)
    return {
        "lr_model": lr,
        "xgb_model": xgb,
        "scaler": scaler,
        "feature_importance": dict(zip(FEATURE_COLS, xgb.feature_importances_)),
    }


def predict_direction(models: dict, features: dict, sentiment_score: float) -> dict:
    """Predict next-day direction from current feature values."""
    feats = dict(features)
    feats["sentiment_score"] = float(sentiment_score)
    row = pd.DataFrame([[feats.get(c, 0.0) for c in FEATURE_COLS]], columns=FEATURE_COLS)
    row = row.fillna(0.0)

    scaler = models["scaler"]
    lr = models["lr_model"]
    xgb = models["xgb_model"]

    lr_proba = lr.predict_proba(scaler.transform(row))[0]
    xgb_proba = xgb.predict_proba(row)[0]

    lr_cls = int(np.argmax(lr_proba))
    xgb_cls = int(np.argmax(xgb_proba))

    return {
        "lr_direction": "UP" if lr_cls == 1 else "DOWN",
        "xgb_direction": "UP" if xgb_cls == 1 else "DOWN",
        "lr_confidence": float(lr_proba[lr_cls]),
        "xgb_confidence": float(xgb_proba[xgb_cls]),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_trained_models(ticker: str, reference_date: str) -> dict:
    """Cached full-data training. `reference_date` invalidates the cache daily."""
    df = fetch_and_engineer(ticker)
    return train_on_full_data(df)


# --------------------------------------------------------------------------- #
# Backtesting
# --------------------------------------------------------------------------- #
MIN_TRAIN_ROWS = 252  # ~1 trading year


def backtest_single_date(
    ticker: str, date: str, sentiment_score: float = 0.0
) -> dict:
    """Train on data strictly before `date`, predict, compare to the actual move."""
    df = fetch_and_engineer(ticker)
    if df.empty:
        raise ValueError("No data returned for ticker")

    target_ts = pd.Timestamp(date)
    hist = df[df.index < target_ts]
    if len(hist) < MIN_TRAIN_ROWS:
        raise ValueError(
            "Need at least 1 year of data before the selected date."
        )

    models = train_for_prediction(hist)

    last_row = hist.iloc[-1]
    features = {col: float(last_row[col]) for col in FEATURE_COLS}
    prediction = predict_direction(models, features, sentiment_score)

    # Determine the actual direction: close on/after `date` vs the prior close.
    prev_close = float(last_row["Close"])
    future = df[df.index >= target_ts]
    if future.empty:
        actual_direction = None
        correct_xgb = correct_lr = None
        actual_close = None
    else:
        actual_close = float(future.iloc[0]["Close"])
        actual_direction = "UP" if actual_close > prev_close else "DOWN"
        correct_xgb = prediction["xgb_direction"] == actual_direction
        correct_lr = prediction["lr_direction"] == actual_direction

    return {
        **prediction,
        "date": date,
        "actual_direction": actual_direction,
        "actual_close": actual_close,
        "prev_close": prev_close,
        "xgb_correct": correct_xgb,
        "lr_correct": correct_lr,
        "feature_importance": models["feature_importance"],
        "features": features,
    }


def backtest_range(
    ticker: str,
    start: str,
    end: str,
    sentiment_score: float = 0.0,
    progress_callback=None,
) -> pd.DataFrame:
    """Walk-forward backtest across [start, end]. One retrain per trading day.

    Data is fetched once (not per day) and each step trains a lightweight model
    on the slice before that day. `progress_callback(done, total)` is invoked
    after each step so the UI can show a progress bar.
    """
    df = fetch_and_engineer(ticker)
    if df.empty:
        raise ValueError("No data returned for ticker")

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    window = df[(df.index >= start_ts) & (df.index <= end_ts)]
    total = len(window.index)
    rows = []

    for i, current in enumerate(window.index, start=1):
        hist = df[df.index < current]
        if len(hist) >= MIN_TRAIN_ROWS:
            try:
                models = train_for_prediction(hist)
                last_row = hist.iloc[-1]
                features = {col: float(last_row[col]) for col in FEATURE_COLS}
                pred = predict_direction(models, features, sentiment_score)

                prev_close = float(last_row["Close"])
                actual_close = float(df.loc[current, "Close"])
                actual_direction = "UP" if actual_close > prev_close else "DOWN"
                actual_return = (actual_close / prev_close) - 1.0

                rows.append(
                    {
                        "date": current.date(),
                        "xgb_direction": pred["xgb_direction"],
                        "lr_direction": pred["lr_direction"],
                        "actual_direction": actual_direction,
                        "xgb_correct": pred["xgb_direction"] == actual_direction,
                        "lr_correct": pred["lr_direction"] == actual_direction,
                        # Realised next-day return, for the simulated P&L curve.
                        "return": actual_return,
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Backtest step failed on {current}: {e}")
        if progress_callback is not None:
            try:
                progress_callback(i, total)
            except Exception:  # noqa: BLE001
                pass

    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "xgb_direction",
            "lr_direction",
            "actual_direction",
            "xgb_correct",
            "lr_correct",
            "return",
        ],
    )
