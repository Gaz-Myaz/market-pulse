"""Market Pulse — Streamlit entry point and page router.

A single-file router (no multi-page folder): `st.session_state["page"]` selects
between the home view (session list) and the session view (analysis + history).
"""

from __future__ import annotations

import logging

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
from src.i18n import LANGUAGES, get_lang, t
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

# Full backtests retrain a model per trading day. Cap the range to protect
# free-tier hosting; warn well before the cap.
MAX_BACKTEST_DAYS = 31      # hard limit (calendar days)
RECOMMENDED_BACKTEST_DAYS = 7  # soft "keep it short" threshold

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
    /* Streamlit's top header is a 60px opaque bar. Make it blend with the page
       and push content below it so nothing (e.g. the Back button) is hidden. */
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 4.5rem; max-width: 1200px; }
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
    /* Language switcher: compact, right-aligned, themed. */
    div[data-testid="stSegmentedControl"] { display: flex; justify-content: flex-end; }
    div[data-testid="stSegmentedControl"] button { padding: 0.15rem 0.7rem; }
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
    """Return (accuracy_float | None, count_label) for verified real preds."""
    verified = [
        p for p in predictions
        if p.get("mode") == "real" and p.get(f"verified_{horizon}") is not None
    ]
    if not verified:
        return None, t("dir_none")
    correct = sum(1 for p in verified if p[f"verified_{horizon}"] is True)
    acc = correct / len(verified)
    return acc, t("acc_count_fmt", correct=correct, total=len(verified))


def direction_arrow(direction: str | None) -> str:
    if direction == "UP":
        return t("dir_up")
    if direction == "DOWN":
        return t("dir_down")
    return t("dir_none")


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
            f'<p class="mp-muted">{t("app_tagline")}</p>',
            unsafe_allow_html=True,
        )
    with header_r:
        if ENABLE_AUTH and st.session_state.get("user"):
            if st.button(t("sign_out")):
                sign_out()
                st.rerun()

    try:
        sessions = get_all_sessions()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load sessions: {e}")
        st.error(t("db_error"))
        return

    # New-session form.
    with st.expander(t("new_session")):
        with st.form("new_session_form", clear_on_submit=True):
            name = st.text_input(t("field_name"), placeholder="e.g. Apple")
            ticker = st.text_input(t("field_ticker"), placeholder="e.g. AAPL")
            description = st.text_area(t("field_description"))
            submitted = st.form_submit_button(t("create"))
            if submitted:
                ticker_clean = ticker.strip().upper()
                if not name.strip() or not ticker_clean:
                    st.warning(t("name_ticker_required"))
                elif not validate_ticker(ticker_clean):
                    st.error(t("ticker_not_found"))
                else:
                    try:
                        create_session(name.strip(), ticker_clean, description.strip())
                        st.success(t("session_created", ticker=ticker_clean))
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Create session failed: {e}")
                        st.error(t("create_failed"))

    st.subheader(t("sessions"))

    if not sessions:
        st.info(t("no_sessions"))
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
                if st.button("🗑", key=f"del_{sid}", help=t("delete_help")):
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
                t(
                    "conf_fmt",
                    dir=direction_arrow(last.get("xgb_direction")),
                    pct=f"{conf:.0%}",
                )
            )
        else:
            st.markdown(f'<span class="mp-muted">{t("no_predictions_yet")}</span>',
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
            f'<span class="mp-muted">{t("created", date=fmt_date(sess.get("created_at")))}</span>',
            unsafe_allow_html=True,
        )

        if st.button(t("open"), key=f"open_{sid}"):
            open_session(sid)
            st.rerun()

        # Inline delete confirmation.
        if st.session_state.get(f"confirm_del_{sid}"):
            st.warning(t("delete_confirm"))
            c1, c2 = st.columns(2)
            with c1:
                if st.button(t("confirm_delete"), key=f"confirm_{sid}"):
                    try:
                        delete_session(sid)
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Delete failed: {e}")
                    st.session_state.pop(f"confirm_del_{sid}", None)
                    st.rerun()
            with c2:
                if st.button(t("cancel"), key=f"cancel_{sid}"):
                    st.session_state.pop(f"confirm_del_{sid}", None)
                    st.rerun()


# --------------------------------------------------------------------------- #
# Session page
# --------------------------------------------------------------------------- #
def render_session_page(session_id: str) -> None:
    if st.button(t("back_to_sessions")):
        go_home()
        st.rerun()

    try:
        sess = get_session(session_id)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load session: {e}")
        st.error(t("could_not_load_session"))
        return

    if not sess:
        st.error(t("session_not_found"))
        if st.button(t("back_home")):
            go_home()
            st.rerun()
        return

    ticker = sess["ticker"]
    st.markdown(
        f'## {sess["name"]} &nbsp;<span class="mp-badge">{ticker}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="mp-muted">{t("created", date=fmt_date(sess.get("created_at")))}</span>',
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

    st.markdown(f"### {t('analysis')}")
    # Stable mode keys; labels are translated via format_func.
    mode = st.radio(
        t("mode"),
        ["real", "quick", "full"],
        format_func=lambda k: t(f"mode_{k}"),
        key=f"mode_{session_id}",
    )
    st.caption(t(f"mode_help_{mode}"))

    today = pd.Timestamp.now().date()
    quick_date = start_date = end_date = None
    if mode == "quick":
        quick_date = st.date_input(
            t("select_date"),
            value=today - pd.Timedelta(days=30),
            max_value=today - pd.Timedelta(days=1),
            key=f"qdate_{session_id}",
        )
    elif mode == "full":
        # Keep the default window short — a long range retrains a model per day.
        st.warning(t("backtest_warning"))
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input(
                t("start"),
                value=today - pd.Timedelta(days=8),
                max_value=today - pd.Timedelta(days=2),
                key=f"sdate_{session_id}",
            )
        with c2:
            end_date = st.date_input(
                t("end"),
                value=today - pd.Timedelta(days=1),
                max_value=today - pd.Timedelta(days=1),
                key=f"edate_{session_id}",
            )
        # Live feedback on how heavy the chosen range is.
        if start_date and end_date and end_date > start_date:
            span = (end_date - start_date).days
            if span > MAX_BACKTEST_DAYS:
                st.error(t("range_too_long", days=span, max=MAX_BACKTEST_DAYS))
            elif span > RECOMMENDED_BACKTEST_DAYS:
                st.info(t("backtest_long_notice", days=span))
    else:
        next_day = get_next_trading_day()
        st.markdown(
            f'<span class="mp-muted">{t("predicting_for", date=next_day)}</span>',
            unsafe_allow_html=True,
        )

    run_label = t("run_backtest") if mode == "full" else t("run_analysis")
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
                st.markdown(f"#### {t('feature_importance')}")
                st.plotly_chart(
                    feature_importance_chart(result["feature_importance"]),
                    use_container_width=True,
                )
            _render_news(ticker, result.get("headlines_detail", []),
                         result.get("sentiment_score", 0.0))


def _safe_sentiment(ticker: str):
    """Fetch headlines + sentiment, degrading to neutral on any failure."""
    try:
        headlines = fetch_headlines(ticker)
        return analyze_sentiment(headlines)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Sentiment unavailable: {e}")
        return 0.0, []


def _run_analysis(sess, mode, quick_date, start_date, end_date) -> None:
    ticker = sess["ticker"]
    session_id = sess["id"]

    # Full backtest gets its own progress bar (it can run many retrains) instead
    # of a generic spinner.
    if mode == "full":
        _run_full_backtest(sess, start_date, end_date)
        return

    with st.spinner(t("spinner_running")):
        sentiment_score, headlines_detail = _safe_sentiment(ticker)
        try:
            if mode == "real":
                _run_real(sess, sentiment_score, headlines_detail)
            else:  # quick backtest
                if quick_date is None:
                    st.warning(t("choose_date"))
                    return
                res = backtest_single_date(ticker, str(quick_date), sentiment_score)
                st.session_state[f"result_{session_id}"] = {
                    "mode": "quick",
                    **res,
                    "sentiment_score": sentiment_score,
                    "headlines_detail": headlines_detail,
                }
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:  # noqa: BLE001
            logger.error(f"Analysis failed: {e}")
            st.error(t("ticker_unavailable"))


def _run_full_backtest(sess, start_date, end_date) -> None:
    ticker = sess["ticker"]
    session_id = sess["id"]

    if start_date is None or end_date is None or end_date <= start_date:
        st.warning(t("end_after_start"))
        return
    span = (end_date - start_date).days
    if span > MAX_BACKTEST_DAYS:
        st.error(t("range_too_long", days=span, max=MAX_BACKTEST_DAYS))
        return

    sentiment_score, _ = _safe_sentiment(ticker)
    progress = st.progress(0.0, text=t("backtest_progress", done=0, total=span))

    def _cb(done: int, total: int) -> None:
        frac = (done / total) if total else 1.0
        progress.progress(
            min(frac, 1.0), text=t("backtest_progress", done=done, total=total)
        )

    try:
        results_df = backtest_range(
            ticker, str(start_date), str(end_date), sentiment_score,
            progress_callback=_cb,
        )
        progress.empty()
        st.session_state[f"result_{session_id}"] = {
            "mode": "full",
            "results_df": results_df,
        }
    except ValueError as e:
        progress.empty()
        st.warning(str(e))
    except Exception as e:  # noqa: BLE001
        progress.empty()
        logger.error(f"Backtest failed: {e}")
        st.error(t("ticker_unavailable"))


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
        st.warning(t("prediction_saved_no_db"))

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
    word = t("bullish") if bullish else t("bearish")
    color = COLORS["up"] if bullish else COLORS["down"]

    with st.container(border=True):
        st.markdown(f"**{t('prediction')}**")
        st.markdown(
            f'<div style="font-size:2rem;font-weight:700;color:{color}">'
            f'{arrow} {word}</div>',
            unsafe_allow_html=True,
        )
        xgb_c = result["xgb_confidence"]
        lr_c = result["lr_confidence"]
        st.markdown(
            f'**{t("xgboost")}** — {direction_arrow(result["xgb_direction"])} · {xgb_c:.0%}'
        )
        st.progress(min(max(xgb_c, 0.0), 1.0))
        st.markdown(
            f'**{t("logreg")}** — {direction_arrow(result["lr_direction"])} · {lr_c:.0%}'
        )
        st.progress(min(max(lr_c, 0.0), 1.0))

        agree = result["xgb_direction"] == result["lr_direction"]
        st.markdown(
            f'{t("agreement")}: {t("both_agree") if agree else t("models_disagree")}'
        )

        feats = result.get("features", {})
        if feats:
            st.markdown(f"**{t('key_indicators')}**")
            rsi = feats.get("RSI_14", 0.0)
            macd = feats.get("MACD_12_26_9", 0.0)
            bbp = feats.get("BBP_5_2.0", 0.0)
            ma_cross = (
                t("ma_cross_bullish") if feats.get("ma_cross", 0) >= 1
                else t("ma_cross_bearish")
            )
            vol = feats.get("volume_ratio", 0.0)
            st.markdown(
                f"RSI: {rsi:.1f}  ·  MACD: {macd:+.2f}  ·  BB%: {bbp:.2f}  \n"
                f"MA Cross: {ma_cross}  ·  Vol: {vol:.2f}×  \n"
                f"{t('sentiment')}: {result.get('sentiment_score', 0.0):+.2f}"
            )


def _render_quick_outcome(result: dict) -> None:
    actual = result.get("actual_direction")
    if actual is None:
        st.info(t("outcome_unavailable"))
        return
    correct = result.get("xgb_correct")
    icon = "✅" if correct else "❌"
    verdict = t("model_correct") if correct else t("model_wrong")
    st.markdown(
        f"**{t('actual_outcome_24h')}:** {icon} {direction_arrow(actual)} — {verdict}"
    )


def _render_full_backtest_result(result: dict) -> None:
    df = result["results_df"]
    if df is None or df.empty:
        st.warning(t("backtest_no_results"))
        return

    n = len(df)
    xgb_acc = df["xgb_correct"].mean()
    lr_acc = df["lr_correct"].mean()
    st.markdown(
        t("backtest_summary", xgb=f"{xgb_acc:.1%}", n=n, lr=f"{lr_acc:.1%}")
    )
    st.plotly_chart(backtest_accuracy_chart(df), use_container_width=True)

    display = df.copy()
    display["date"] = display["date"].astype(str)
    display["xgb_direction"] = display["xgb_direction"].map(
        lambda d: direction_arrow(d)
    )
    display["lr_direction"] = display["lr_direction"].map(lambda d: direction_arrow(d))
    display["actual_direction"] = display["actual_direction"].map(
        lambda d: direction_arrow(d)
    )
    display["xgb_correct"] = display["xgb_correct"].map({True: "✅", False: "❌"})
    display["lr_correct"] = display["lr_correct"].map({True: "✅", False: "❌"})
    display = display.rename(
        columns={
            "date": t("col_date"),
            "xgb_direction": "XGB",
            "lr_direction": "LR",
            "actual_direction": t("col_actual"),
            "xgb_correct": "XGB ✓",
            "lr_correct": "LR ✓",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_news(ticker: str, headlines_detail: list[dict], sentiment_score: float) -> None:
    st.markdown(f"#### {t('latest_news', ticker=ticker)}")
    if not headlines_detail:
        st.markdown(
            f'<span class="mp-muted">{t("no_news")}</span>',
            unsafe_allow_html=True,
        )
        return
    for h in headlines_detail:
        score = h["score"]
        icon = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "⚪")
        st.markdown(f"{icon} {h['title']}  ·  **{score:+.2f}**")

    if sentiment_score > 0.1:
        mood = t("mood_slightly_positive") if sentiment_score < 0.4 else t("mood_positive")
    elif sentiment_score < -0.1:
        mood = t("mood_slightly_negative") if sentiment_score > -0.4 else t("mood_negative")
    else:
        mood = t("mood_neutral")
    st.markdown(
        f"**{t('overall_sentiment', score=f'{sentiment_score:+.2f}', mood=mood)}**"
    )


def _render_history_and_charts(sess: dict) -> None:
    ticker = sess["ticker"]
    session_id = sess["id"]

    try:
        preds = get_predictions(session_id)
    except Exception:  # noqa: BLE001
        preds = []

    # Accuracy metrics (real, verified predictions only).
    st.markdown(f"### {t('track_record')}")
    m1, m2, m3 = st.columns(3)
    for col, horizon, title in [
        (m1, "24h", t("acc_24h")),
        (m2, "1w", t("acc_1w")),
        (m3, "1m", t("acc_1m")),
    ]:
        acc, label = calc_accuracy(preds, horizon)
        with col:
            st.metric(title, f"{acc:.0%}" if acc is not None else "—",
                      help=label)
            st.markdown(
                f'<span class="mp-muted">{label if acc is not None else t("no_data")}</span>',
                unsafe_allow_html=True,
            )

    # Charts + model metrics need engineered data and trained models — the
    # heaviest part of the page. Load both under one spinner so the navigation
    # shows clear "loading" feedback instead of a silent greyed-out delay.
    df = None
    models = None
    with st.spinner(t("loading_market")):
        try:
            df = fetch_and_engineer(ticker)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Could not load chart data for {ticker}: {e}")
            st.error(t("could_not_load_market"))
        if df is not None and not df.empty:
            try:
                models = get_trained_models(ticker, get_reference_date(ticker))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Model metrics unavailable: {e}")

    if df is not None and not df.empty:
        st.markdown(f"#### {t('price')}")
        st.plotly_chart(price_chart(df, ticker), use_container_width=True)
        st.markdown(f"#### {t('rsi')}")
        st.plotly_chart(rsi_chart(df), use_container_width=True)
        st.markdown(f"#### {t('macd')}")
        st.plotly_chart(macd_chart(df), use_container_width=True)

        if models is not None:
            _render_model_metrics(models)

    _render_history_table(preds)

    if st.button(t("verify_predictions"), key=f"verify_{session_id}"):
        with st.spinner(t("checking_outcomes")):
            try:
                n = verify_pending_predictions()
                st.success(t("verified_n", n=n))
            except Exception as e:  # noqa: BLE001
                logger.error(f"Manual verification failed: {e}")
                st.error(t("verification_failed"))
        st.rerun()


def _render_model_metrics(models: dict) -> None:
    st.markdown(f"#### {t('model_metrics')}")
    if models.get("test_size", 0) < 30:
        st.warning(t("not_enough_metrics"))

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
    st.markdown(f"#### {t('prediction_history')}")
    if not preds:
        st.info(t("run_first"))
        return

    rows = []
    for p in preds[:50]:
        agree = p.get("xgb_direction") == p.get("lr_direction")
        mode_label = t("real_short") if p.get("mode") == "real" else t("backtest_short")
        rows.append(
            {
                t("col_date"): fmt_date(p.get("created_at")),
                t("col_mode"): mode_label,
                t("col_direction"): direction_arrow(p.get("xgb_direction")),
                "XGB": f'{direction_arrow(p.get("xgb_direction"))} '
                       f'{(p.get("xgb_confidence") or 0):.0%}',
                "LR": f'{direction_arrow(p.get("lr_direction"))} '
                      f'{(p.get("lr_confidence") or 0):.0%}',
                t("col_agree"): "✓" if agree else "✗",
                "24h": _verify_icon(p, "24h"),
                "1w": _verify_icon(p, "1w"),
                "1m": _verify_icon(p, "1m"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Language selector
# --------------------------------------------------------------------------- #
def _render_language_selector() -> None:
    """Compact EN/RU switcher, right-aligned under the header on every page."""
    st.session_state.setdefault("lang", "en")
    codes = list(LANGUAGES.keys())  # ["en", "ru"]
    current = get_lang()
    _, right = st.columns([0.78, 0.22])
    with right:
        if hasattr(st, "segmented_control"):
            choice = st.segmented_control(
                t("language"),
                options=codes,
                format_func=lambda c: LANGUAGES[c],
                default=current,
                key="lang_select",
                label_visibility="collapsed",
            )
        else:  # fallback for older Streamlit
            choice = st.radio(
                t("language"),
                options=codes,
                index=codes.index(current),
                format_func=lambda c: LANGUAGES[c],
                horizontal=True,
                key="lang_select",
                label_visibility="collapsed",
            )
        # Keep the previous language if the control was deselected (None).
        if choice and choice != st.session_state.get("lang"):
            st.session_state["lang"] = choice
            st.rerun()


# --------------------------------------------------------------------------- #
# Boot
# --------------------------------------------------------------------------- #
def main() -> None:
    require_auth()

    _render_language_selector()

    # DB health check on first load.
    if not st.session_state.get("db_checked"):
        with st.spinner(t("db_starting")):
            ok = check_db_connection()
        if not ok:
            st.error(t("db_error"))
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
