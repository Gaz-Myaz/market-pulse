"""Plotly chart builders. Every figure uses a transparent background and the
warm off-white palette so charts blend into the page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COLORS = {
    "bg": "#F9F7F4",
    "card": "#FFFFFF",
    "text": "#1A1A1A",
    "text_muted": "#6B7280",
    "border": "#E5E3DF",
    "accent": "#2563EB",
    "up": "#059669",
    "down": "#DC2626",
    "neutral": "#D97706",
    "xgb": "#2563EB",
    "lr": "#7C3AED",
}

# Friendly labels for feature names in the importance chart.
FEATURE_LABELS = {
    "RSI_14": "RSI",
    "MACD_12_26_9": "MACD",
    "MACDh_12_26_9": "MACD Hist",
    "MACDs_12_26_9": "MACD Signal",
    "BBP_5_2.0": "BB %B",
    "ma20_ratio": "MA20 Ratio",
    "ma50_ratio": "MA50 Ratio",
    "ma_cross": "MA Cross",
    "volume_ratio": "Volume Ratio",
    "price_change_5d": "5d Change",
    "sentiment_score": "Sentiment",
}


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text"],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    fig.update_xaxes(gridcolor=COLORS["border"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["border"], zeroline=False)
    return fig


def price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick of the last 90 trading days with MA20 / MA50 overlays."""
    d = df.tail(90)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=d.index,
            open=d["Open"],
            high=d["High"],
            low=d["Low"],
            close=d["Close"],
            increasing_line_color=COLORS["up"],
            decreasing_line_color=COLORS["down"],
            name=ticker,
        )
    )
    if "SMA_20" in d.columns:
        fig.add_trace(
            go.Scatter(
                x=d.index, y=d["SMA_20"], name="MA20",
                line=dict(color=COLORS["accent"], width=1.5, dash="dash"),
            )
        )
    if "SMA_50" in d.columns:
        fig.add_trace(
            go.Scatter(
                x=d.index, y=d["SMA_50"], name="MA50",
                line=dict(color=COLORS["neutral"], width=1.5, dash="dash"),
            )
        )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _base_layout(fig, 320)


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI line over the last 90 days with 30/70 reference bands."""
    d = df.tail(90)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d.index, y=d["RSI_14"], name="RSI",
            line=dict(color=COLORS["accent"], width=1.5),
        )
    )
    fig.add_hrect(y0=70, y1=100, fillcolor=COLORS["down"], opacity=0.06, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=COLORS["up"], opacity=0.06, line_width=0)
    fig.add_hline(y=70, line=dict(color=COLORS["down"], width=1, dash="dash"))
    fig.add_hline(y=30, line=dict(color=COLORS["up"], width=1, dash="dash"))
    fig.update_yaxes(range=[0, 100])
    fig = _base_layout(fig, 160)
    fig.update_layout(showlegend=False)
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD + signal lines and a coloured histogram, last 90 days."""
    d = df.tail(90)
    hist = d["MACDh_12_26_9"]
    colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in hist]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d.index, y=hist, name="Histogram", marker_color=colors, opacity=0.5))
    fig.add_trace(
        go.Scatter(x=d.index, y=d["MACD_12_26_9"], name="MACD",
                   line=dict(color=COLORS["accent"], width=1.5))
    )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["MACDs_12_26_9"], name="Signal",
                   line=dict(color=COLORS["down"], width=1.5))
    )
    return _base_layout(fig, 160)


def feature_importance_chart(importance: dict) -> go.Figure:
    """Horizontal bar chart of XGBoost feature importances, sorted descending."""
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    labels = [FEATURE_LABELS.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=values[::-1], y=labels[::-1], orientation="h",
            marker_color=COLORS["accent"],
        )
    )
    fig = _base_layout(fig, 260)
    fig.update_layout(showlegend=False)
    return fig


def backtest_accuracy_chart(results_df: pd.DataFrame) -> go.Figure:
    """Rolling 20-step accuracy for both models over the backtest window."""
    d = results_df.copy()
    fig = go.Figure()
    if not d.empty:
        d["xgb_roll"] = d["xgb_correct"].rolling(20, min_periods=1).mean() * 100
        d["lr_roll"] = d["lr_correct"].rolling(20, min_periods=1).mean() * 100
        fig.add_trace(
            go.Scatter(x=d["date"], y=d["xgb_roll"], name="XGBoost",
                       line=dict(color=COLORS["xgb"], width=2))
        )
        fig.add_trace(
            go.Scatter(x=d["date"], y=d["lr_roll"], name="Log. Reg.",
                       line=dict(color=COLORS["lr"], width=2))
        )
    fig.add_hline(y=50, line=dict(color=COLORS["text_muted"], width=1, dash="dash"))
    fig.update_yaxes(range=[0, 100], title="Rolling accuracy %")
    return _base_layout(fig, 260)


def confusion_matrix_chart(cm: np.ndarray, model_name: str) -> go.Figure:
    """2x2 confusion-matrix heatmap. Rows = actual, cols = predicted."""
    labels = ["DOWN", "UP"]
    cm = np.asarray(cm)
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=16),
        )
    )
    fig.update_layout(title=dict(text=model_name, font=dict(size=13)))
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual", autorange="reversed")
    fig = _base_layout(fig, 220)
    fig.update_layout(showlegend=False)
    return fig
