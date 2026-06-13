"""Market Pulse — Streamlit entry point and page router.

A single-file router (no multi-page folder): `st.session_state["page"]` selects
between the home view (session list) and the session view (analysis + history).
"""

from __future__ import annotations

import logging
from datetime import timezone

import pandas as pd
import streamlit as st

from src.auth import ENABLE_AUTH, require_auth, sign_out
from src.charts import (
    COLORS,
    backtest_accuracy_chart,
    confusion_matrix_chart,
    feature_importance_chart,
    macd_chart,
    price_chart,
    rsi_chart,
)
from src.data import (
    FEATURE_COLS,
    fetch_and_engineer,
    get_latest_features,
    get_next_trading_day,
    get_reference_date,
)
from src.model import (
    backtest_range,
    backtest_single_date,
    get_trained_models,
    predict_direction,
)
from src.sentiment import analyze_sentiment, fetch_headlines
from src.storage import (
    check_db_connection,
    create_session,
    delete_session,
    get_predictions,
    get_session,
    get_all_sessions,
    save_prediction,
    seed_default_sessions,
)
from src.verify import verify_pending_predictions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("market-pulse")

# Ticker validation lives here so the home form can call it without importing
# yfinance everywhere.
import yfinance as yf  # noqa: E402


# --------------------------------------------------------------------------- #
# Page config + global CSS
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Market Pulse", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
    .stApp { background-color: #F9F7F4; }
    .block-container { padding-top: 2rem; max-width: 1200px; }
    h1, h2, h3 { font-weight: 600; color: #1A1A1A; }
    .stButton > button {
        background: #2563EB; color: white;
        border: none; border-radius: 8px;
        padding: 0.5rem 1.25rem; font-weight: 500;
    }
    .stButton > button:hover { background: #1D4ED8; }
    div[data-testid="metric-container"] {
        background: #FFFFFF; border: 1px solid #E5E3DF;
        border-radius: 10px; padding: 0.75rem 1rem;
    }
    .mp-card {
        background: #FFFFFF; border: 1px solid #E5E3DF;
        border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.5rem;
    }
    .mp-badge {
        display: inline-block; background: #EEF2FF; color: #2563EB;
        font-weight: 700; font-size: 0.8rem; padding: 0.15rem 0.55rem;
        border-radius: 6px; letter-spacing: 0.02em;
    }
    .mp-muted { color: #6B7280; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Small display helpers
# --------------------------------------------------------------------------- #
def fmt_datetime(ts: str) -> str:
    """Render a stored UTC timestamp in US/Eastern, human-friendly."""
    try:
        dt = pd.Timestamp(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.tz_convert("US/Eastern").strftime("%b %d, %Y · %H:%M ET")
    except Exception:  # noqa: BLE001
        return str(ts)


def fmt_date(ts: str) -> str:
    try:
        return pd.Timestamp(ts).strftime("%b %d")
    except Exception:  # noqa: BLE001
        return str(ts)


def calc_accuracy(predictions: list[dict], horizon: str):
    """Return (accuracy_float | None, label_string) for verified real preds."""
    verified = [
        p for p in predictions
        if p.get("mode") == "real" and p.get(f"verified_{horizon}") is not None
    ]
    if not verified:
        return None, "—"
    correct = sum(1 for p in verified if p[f"verified_{horizon}"] is True)
    acc = correct / len(verified)
    return acc, f"{acc:.0%} ({correct} of {len(verified)})"


def direction_arrow(direction: str | None) -> str:
    if direction == "UP":
        return "↑ UP"
    if direction == "DOWN":
        return "↓ DOWN"
    return "—"


def validate_ticker(ticker: str) -> bool:
    try:
        info = yf.Ticker(ticker).fast_info
        return info.last_price is not None
    except Exception:  # noqa: BLE001
        return False


def go_home() -> None:
    st.session_state["page"] = "home"
    st.session_state.pop("session_id", None)


def open_session(session_id: str) -> None:
    st.session_state["page"] = "session"
    st.session_state["session_id"] = session_id


# --------------------------------------------------------------------------- #
# Home page
# --------------------------------------------------------------------------- #
def render_home_page() -> None:
    header_l, header_r = st.columns([0.7, 0.3])
    with header_l:
        st.title("📈 Market Pulse")
        st.markdown(
            '<p class="mp-muted">Stock direction prediction with XGBoost + news sentiment.</p>',
            unsafe_allow_html=True,
        )
    with header_r:
        if ENABLE_AUTH and st.session_state.get("user"):
            if st.button("Sign out"):
                sign_out()
                st.rerun()

    try:
        sessions = get_all_sessions()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load sessions: {e}")
        st.error(
            "Database connection failed. Check Supabase credentials in secrets.toml."
        )
        return

    # New-session form.
    with st.expander("➕ New Session"):
        with st.form("new_session_form", clear_on_submit=True):
            name = st.text_input("Name", placeholder="e.g. Apple")
            ticker = st.text_input("Ticker", placeholder="e.g. AAPL")
            description = st.text_area("Description (optional)")
            submitted = st.form_submit_button("Create")
            if submitted:
                ticker_clean = ticker.strip().upper()
                if not name.strip() or not ticker_clean:
                    st.warning("Name and ticker are required.")
                elif not validate_ticker(ticker_clean):
                    st.error(
                        "Ticker not found on Yahoo Finance. Please check the symbol."
                    )
                else:
                    try:
                        create_session(name.strip(), ticker_clean, description.strip())
                        st.success(f"Created session for {ticker_clean}.")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Create session failed: {e}")
                        st.error("Could not create session. Please try again.")

    st.subheader("Sessions")

    if not sessions:
        st.info(
            "No sessions yet. Use **➕ New Session** above to create your first one."
        )
        return

    # Render session cards two per row.
    for i in range(0, len(sessions), 2):
        cols = st.columns(2)
        for col, sess in zip(cols, sessions[i : i + 2]):
            with col:
                _render_session_card(sess)


def _render_session_card(sess: dict) -> None:
    sid = sess["id"]
    with st.container(border=True):
        top_l, top_r = st.columns([0.75, 0.25])
        with top_l:
            st.markdown(
                f'<span class="mp-badge">{sess["ticker"]}</span> '
                f'&nbsp;<b style="font-size:1.05rem">{sess["name"]}</b>',
                unsafe_allow_html=True,
            )
        with top_r:
            if not sess.get("is_default"):
                if st.button("🗑", key=f"del_{sid}", help="Delete session"):
                    st.session_state[f"confirm_del_{sid}"] = True

        # Last prediction summary + accuracy badges.
        try:
            preds = get_predictions(sid)
        except Exception:  # noqa: BLE001
            preds = []

        if preds:
            last = preds[0]
            conf = last.get("xgb_confidence") or 0.0
            st.markdown(
                f'{direction_arrow(last.get("xgb_direction"))} · '
                f'{conf:.0%} confidence',
            )
        else:
            st.markdown('<span class="mp-muted">No predictions yet</span>',
                        unsafe_allow_html=True)

        a24, _ = calc_accuracy(preds, "24h")
        a1w, _ = calc_accuracy(preds, "1w")
        a1m, _ = calc_accuracy(preds, "1m")
        badge = (
            f"24h: {a24:.0%}" if a24 is not None else "24h: —"
        ) + " · " + (
            f"1w: {a1w:.0%}" if a1w is not None else "1w: —"
        ) + " · " + (
            f"1m: {a1m:.0%}" if a1m is not None else "1m: —"
        )
        st.markdown(f'<span class="mp-muted">{badge}</span>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="mp-muted">Created {fmt_date(sess.get("created_at"))}</span>',
            unsafe_allow_html=True,
        )

        if st.button("Open →", key=f"open_{sid}"):
            open_session(sid)
            st.rerun()

        # Inline delete confirmation.
        if st.session_state.get(f"confirm_del_{sid}"):
            st.warning("Delete this session? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Confirm delete", key=f"confirm_{sid}"):
                    try:
                        delete_session(sid)
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Delete failed: {e}")
                    st.session_state.pop(f"confirm_del_{sid}", None)
                    st.rerun()
            with c2:
                if st.button("Cancel", key=f"cancel_{sid}"):
                    st.session_state.pop(f"confirm_del_{sid}", None)
                    st.rerun()


# --------------------------------------------------------------------------- #
# Session page
# --------------------------------------------------------------------------- #
def render_session_page(session_id: str) -> None:
    if st.button("← Back to Sessions"):
        go_home()
        st.rerun()

    try:
        sess = get_session(session_id)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load session: {e}")
        st.error("Could not load this session.")
        return

    if not sess:
        st.error("Session not found.")
        if st.button("Back home"):
            go_home()
            st.rerun()
        return

    ticker = sess["ticker"]
    st.markdown(
        f'## {sess["name"]} &nbsp;<span class="mp-badge">{ticker}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="mp-muted">Created {fmt_date(sess.get("created_at"))}</span>',
        unsafe_allow_html=True,
    )

    # Auto-verify any real predictions whose window has elapsed (once per load).
    if not st.session_state.get(f"verified_{session_id}"):
        try:
            n = verify_pending_predictions()
            if n:
                logger.info(f"Auto-verified {n} prediction horizon(s).")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Auto-verification skipped: {e}")
        st.session_state[f"verified_{session_id}"] = True

    left, right = st.columns([0.42, 0.58])
    with left:
        _render_analysis_panel(sess)
    with right:
        _render_history_and_charts(sess)


def _render_analysis_panel(sess: dict) -> None:
    ticker = sess["ticker"]
    session_id = sess["id"]

    st.markdown("### Analysis")
    mode = st.radio(
        "Mode",
        ["Real Prediction", "Quick Backtest", "Full Backtest"],
        key=f"mode_{session_id}",
    )

    today = pd.Timestamp.now().date()
    quick_date = start_date = end_date = None
    if mode == "Quick Backtest":
        quick_date = st.date_input(
            "Select date",
            value=today - pd.Timedelta(days=30),
            max_value=today - pd.Timedelta(days=1),
            key=f"qdate_{session_id}",
        )
    elif mode == "Full Backtest":
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input(
                "Start",
                value=today - pd.Timedelta(days=90),
                max_value=today - pd.Timedelta(days=2),
                key=f"sdate_{session_id}",
            )
        with c2:
            end_date = st.date_input(
                "End",
                value=today - pd.Timedelta(days=1),
                max_value=today - pd.Timedelta(days=1),
                key=f"edate_{session_id}",
            )
    else:
        next_day = get_next_trading_day()
        st.markdown(
            f'<span class="mp-muted">Predicting for: <b>{next_day}</b></span>',
            unsafe_allow_html=True,
        )

    run_label = "Run Backtest" if mode == "Full Backtest" else "Run Analysis"
    if st.button(run_label, key=f"run_{session_id}"):
        _run_analysis(sess, mode, quick_date, start_date, end_date)

    # Show last result stored in session_state.
    result = st.session_state.get(f"result_{session_id}")
    if result:
        if result["mode"] == "full":
            _render_full_backtest_result(result)
        else:
            _render_result_card(result)
            if result["mode"] == "quick":
                _render_quick_outcome(result)
            if result.get("feature_importance"):
                st.markdown("#### Feature Importance")
                st.plotly_chart(
                    feature_importance_chart(result["feature_importance"]),
                    use_container_width=True,
                )
            _render_news(ticker, result.get("headlines_detail", []),
                         result.get("sentiment_score", 0.0))


def _run_analysis(sess, mode, quick_date, start_date, end_date) -> None:
    ticker = sess["ticker"]
    session_id = sess["id"]

    with st.spinner("Fetching data & running models..."):
        try:
            headlines = fetch_headlines(ticker)
            sentiment_score, headlines_detail = analyze_sentiment(headlines)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Sentiment unavailable: {e}")
            sentiment_score, headlines_detail = 0.0, []

        try:
            if mode == "Real Prediction":
                _run_real(sess, sentiment_score, headlines_detail)
            elif mode == "Quick Backtest":
                if quick_date is None:
                    st.warning("Please choose a date.")
                    return
                res = backtest_single_date(
                    ticker, str(quick_date), sentiment_score
                )
                st.session_state[f"result_{session_id}"] = {
                    "mode": "quick",
                    **res,
                    "sentiment_score": sentiment_score,
                    "headlines_detail": headlines_detail,
                }
            else:  # Full Backtest
                if start_date is None or end_date is None or end_date <= start_date:
                    st.warning("End date must be after start date.")
                    return
                results_df = backtest_range(
                    ticker, str(start_date), str(end_date), sentiment_score
                )
                st.session_state[f"result_{session_id}"] = {
                    "mode": "full",
                    "results_df": results_df,
                }
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:  # noqa: BLE001
            logger.error(f"Analysis failed: {e}")
            st.error("Ticker not found or data unavailable. Please check the symbol.")


def _run_real(sess, sentiment_score, headlines_detail) -> None:
    ticker = sess["ticker"]
    session_id = sess["id"]

    reference_date = get_reference_date(ticker)
    models = get_trained_models(ticker, reference_date)
    features = get_latest_features(ticker, sentiment_score)
    pred = predict_direction(models, features, sentiment_score)

    prediction_data = {
        "ticker": ticker,
        "mode": "real",
        "xgb_direction": pred["xgb_direction"],
        "lr_direction": pred["lr_direction"],
        "xgb_confidence": pred["xgb_confidence"],
        "lr_confidence": pred["lr_confidence"],
        "sentiment_score": sentiment_score,
        "headlines": [h["title"] for h in headlines_detail],
        "features": features,
        "feature_importance": {
            k: float(v) for k, v in models["feature_importance"].items()
        },
    }
    try:
        save_prediction(session_id, prediction_data)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Could not save prediction: {e}")
        st.warning("Prediction made but could not be saved to the database.")

    st.session_state[f"result_{session_id}"] = {
        "mode": "real",
        **pred,
        "features": features,
        "feature_importance": models["feature_importance"],
        "sentiment_score": sentiment_score,
        "headlines_detail": headlines_detail,
        "reference_date": reference_date,
    }


def _render_result_card(result: dict) -> None:
    xgb_dir = result["xgb_direction"]
    bullish = xgb_dir == "UP"
    arrow = "↑" if bullish else "↓"
    word = "BULLISH" if bullish else "BEARISH"
    color = COLORS["up"] if bullish else COLORS["down"]

    with st.container(border=True):
        st.markdown("**PREDICTION**")
        st.markdown(
            f'<div style="font-size:2rem;font-weight:700;color:{color}">'
            f'{arrow} {word}</div>',
            unsafe_allow_html=True,
        )
        xgb_c = result["xgb_confidence"]
        lr_c = result["lr_confidence"]
        st.markdown(f'**XGBoost** — {result["xgb_direction"]} · {xgb_c:.0%}')
        st.progress(min(max(xgb_c, 0.0), 1.0))
        st.markdown(f'**Log. Reg.** — {result["lr_direction"]} · {lr_c:.0%}')
        st.progress(min(max(lr_c, 0.0), 1.0))

        agree = result["xgb_direction"] == result["lr_direction"]
        st.markdown(
            f'Agreement: {"✓ Both agree" if agree else "✗ Models disagree"}'
        )

        feats = result.get("features", {})
        if feats:
            st.markdown("**Key Indicators**")
            rsi = feats.get("RSI_14", 0.0)
            macd = feats.get("MACD_12_26_9", 0.0)
            bbp = feats.get("BBP_5_2.0", 0.0)
            ma_cross = "Bullish" if feats.get("ma_cross", 0) >= 1 else "Bearish"
            vol = feats.get("volume_ratio", 0.0)
            st.markdown(
                f"RSI: {rsi:.1f}  ·  MACD: {macd:+.2f}  ·  BB%: {bbp:.2f}  \n"
                f"MA Cross: {ma_cross}  ·  Vol: {vol:.2f}×  \n"
                f"Sentiment: {result.get('sentiment_score', 0.0):+.2f}"
            )


def _render_quick_outcome(result: dict) -> None:
    actual = result.get("actual_direction")
    if actual is None:
        st.info("Actual outcome not available yet for this date.")
        return
    correct = result.get("xgb_correct")
    icon = "✅" if correct else "❌"
    verdict = "Model was correct" if correct else "Model was wrong"
    st.markdown(f"**Actual outcome (24h):** {icon} {actual} — {verdict}")


def _render_full_backtest_result(result: dict) -> None:
    df = result["results_df"]
    if df is None or df.empty:
        st.warning(
            "No backtest results — the date range may be too short or too early "
            "(need at least 1 year of data before the start date)."
        )
        return

    n = len(df)
    xgb_acc = df["xgb_correct"].mean()
    lr_acc = df["lr_correct"].mean()
    st.markdown(
        f"**XGBoost: {xgb_acc:.1%}** over {n} predictions  |  "
        f"Baseline LR: {lr_acc:.1%}"
    )
    st.plotly_chart(backtest_accuracy_chart(df), use_container_width=True)

    display = df.copy()
    display["date"] = display["date"].astype(str)
    display["xgb_correct"] = display["xgb_correct"].map({True: "✅", False: "❌"})
    display["lr_correct"] = display["lr_correct"].map({True: "✅", False: "❌"})
    display = display.rename(
        columns={
            "date": "Date",
            "xgb_direction": "XGB",
            "lr_direction": "LR",
            "actual_direction": "Actual",
            "xgb_correct": "XGB ✓",
            "lr_correct": "LR ✓",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_news(ticker: str, headlines_detail: list[dict], sentiment_score: float) -> None:
    st.markdown(f"#### Latest News · {ticker}")
    if not headlines_detail:
        st.markdown(
            '<span class="mp-muted">No recent news found for this ticker.</span>',
            unsafe_allow_html=True,
        )
        return
    for h in headlines_detail:
        score = h["score"]
        icon = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
        st.markdown(f"{icon} {h['title']}  ·  **{score:+.2f}**")

    if sentiment_score > 0.1:
        mood = "Slightly Positive" if sentiment_score < 0.4 else "Positive"
    elif sentiment_score < -0.1:
        mood = "Slightly Negative" if sentiment_score > -0.4 else "Negative"
    else:
        mood = "Neutral"
    st.markdown(f"**Overall sentiment: {sentiment_score:+.2f} ({mood})**")


def _render_history_and_charts(sess: dict) -> None:
    ticker = sess["ticker"]
    session_id = sess["id"]

    try:
        preds = get_predictions(session_id)
    except Exception:  # noqa: BLE001
        preds = []

    # Accuracy metrics (real, verified predictions only).
    st.markdown("### Track Record")
    m1, m2, m3 = st.columns(3)
    for col, horizon, title in [
        (m1, "24h", "24h Accuracy"),
        (m2, "1w", "1w Accuracy"),
        (m3, "1m", "1m Accuracy"),
    ]:
        acc, label = calc_accuracy(preds, horizon)
        with col:
            st.metric(title, f"{acc:.0%}" if acc is not None else "—",
                      help=label)
            st.markdown(
                f'<span class="mp-muted">{label if acc is not None else "no data"}</span>',
                unsafe_allow_html=True,
            )

    # Charts + model metrics need engineered data.
    try:
        df = fetch_and_engineer(ticker)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Could not load chart data for {ticker}: {e}")
        st.error("Could not load market data for this ticker.")
        df = None

    if df is not None and not df.empty:
        st.markdown("#### Price")
        st.plotly_chart(price_chart(df, ticker), use_container_width=True)
        st.markdown("#### RSI")
        st.plotly_chart(rsi_chart(df), use_container_width=True)
        st.markdown("#### MACD")
        st.plotly_chart(macd_chart(df), use_container_width=True)

        # Model metrics from cached trained models.
        try:
            reference_date = get_reference_date(ticker)
            models = get_trained_models(ticker, reference_date)
            _render_model_metrics(models)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Model metrics unavailable: {e}")

    _render_history_table(preds)

    if st.button("🔄 Verify Predictions", key=f"verify_{session_id}"):
        with st.spinner("Checking outcomes..."):
            try:
                n = verify_pending_predictions()
                st.success(f"Verified {n} prediction horizon(s).")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Manual verification failed: {e}")
                st.error("Verification failed. Please try again later.")
        st.rerun()


def _render_model_metrics(models: dict) -> None:
    st.markdown("#### Model Metrics")
    if models.get("test_size", 0) < 30:
        st.warning("Not enough data for reliable metrics (small test set).")

    xgb_m = models["xgb_metrics"]
    lr_m = models["lr_metrics"]
    metrics_df = pd.DataFrame(
        {
            "XGBoost": [
                f"{xgb_m['accuracy']:.1%}",
                f"{xgb_m['precision']:.1%}",
                f"{xgb_m['recall']:.1%}",
                f"{xgb_m['f1']:.1%}",
            ],
            "Log. Reg.": [
                f"{lr_m['accuracy']:.1%}",
                f"{lr_m['precision']:.1%}",
                f"{lr_m['recall']:.1%}",
                f"{lr_m['f1']:.1%}",
            ],
        },
        index=["Accuracy", "Precision", "Recall", "F1"],
    )
    st.dataframe(metrics_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            confusion_matrix_chart(models["xgb_cm"], "XGBoost"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            confusion_matrix_chart(models["lr_cm"], "Log. Reg."),
            use_container_width=True,
        )


def _verify_icon(pred: dict, horizon: str) -> str:
    val = pred.get(f"verified_{horizon}")
    if val is True:
        return "✅"
    if val is False:
        return "❌"
    # Backtests don't apply to 1w/1m.
    if pred.get("mode") != "real" and horizon in ("1w", "1m"):
        return "—"
    return "⏳"


def _render_history_table(preds: list[dict]) -> None:
    st.markdown("#### Prediction History")
    if not preds:
        st.info("Run your first analysis to see history.")
        return

    rows = []
    for p in preds[:50]:
        agree = p.get("xgb_direction") == p.get("lr_direction")
        mode_label = "Real" if p.get("mode") == "real" else "Backtest"
        rows.append(
            {
                "Date": fmt_date(p.get("created_at")),
                "Mode": mode_label,
                "Direction": direction_arrow(p.get("xgb_direction")),
                "XGB": f'{direction_arrow(p.get("xgb_direction"))} '
                       f'{(p.get("xgb_confidence") or 0):.0%}',
                "LR": f'{direction_arrow(p.get("lr_direction"))} '
                      f'{(p.get("lr_confidence") or 0):.0%}',
                "Agree": "✓" if agree else "✗",
                "24h": _verify_icon(p, "24h"),
                "1w": _verify_icon(p, "1w"),
                "1m": _verify_icon(p, "1m"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Boot
# --------------------------------------------------------------------------- #
def main() -> None:
    require_auth()

    # DB health check on first load.
    if not st.session_state.get("db_checked"):
        with st.spinner("Starting Market Pulse..."):
            ok = check_db_connection()
        if not ok:
            st.error(
                "Database is waking up — this can take up to 30 seconds on the "
                "first visit. Please refresh. If this persists, check your "
                "Supabase credentials in secrets.toml."
            )
            st.stop()
        try:
            seed_default_sessions()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Seeding skipped: {e}")
        st.session_state["db_checked"] = True

    page = st.session_state.get("page", "home")
    if page == "session" and st.session_state.get("session_id"):
        render_session_page(st.session_state["session_id"])
    else:
        render_home_page()


if __name__ == "__main__":
    main()
