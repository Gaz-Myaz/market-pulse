"""Lightweight i18n for Market Pulse — English and Russian.

Usage:
    from src.i18n import t
    st.title(t("app_title"))
    st.write(t("created", date="Jun 13"))

The active language lives in st.session_state["lang"] (set by the language
selector in app.py). t() falls back to English, then to the key itself, so a
missing translation never crashes the UI.
"""

from __future__ import annotations

import streamlit as st

LANGUAGES = {"en": "English", "ru": "Русский"}
DEFAULT_LANG = "en"


def get_lang() -> str:
    lang = st.session_state.get("lang", DEFAULT_LANG)
    return lang if lang in LANGUAGES else DEFAULT_LANG


def t(key: str, **kwargs) -> str:
    """Translate `key` for the active language, formatting with kwargs."""
    table = TRANSLATIONS.get(get_lang(), {})
    text = table.get(key)
    if text is None:
        text = TRANSLATIONS[DEFAULT_LANG].get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:  # noqa: BLE001 - never let formatting break the UI
            pass
    return text


TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Home
        "app_tagline": "Stock direction prediction with XGBoost + news sentiment.",
        "sign_out": "Sign out",
        "new_session": "➕ New Session",
        "field_name": "Name",
        "field_ticker": "Ticker",
        "field_description": "Description (optional)",
        "create": "Create",
        "name_ticker_required": "Name and ticker are required.",
        "ticker_not_found": "Ticker not found on Yahoo Finance. Please check the symbol.",
        "session_created": "Created session for {ticker}.",
        "create_failed": "Could not create session. Please try again.",
        "sessions": "Sessions",
        "no_sessions": "No sessions yet. Use **➕ New Session** above to create your first one.",
        "no_predictions_yet": "No predictions yet",
        "conf_fmt": "{dir} · {pct} confidence",
        "created": "Created {date}",
        "open": "Open →",
        "delete_confirm": "Delete this session? This cannot be undone.",
        "confirm_delete": "Confirm delete",
        "cancel": "Cancel",
        "delete_help": "Delete session",
        # Session page
        "back_to_sessions": "← Back to Sessions",
        "session_not_found": "Session not found.",
        "back_home": "Back home",
        "could_not_load_session": "Could not load this session.",
        "analysis": "Analysis",
        "mode": "Mode",
        "mode_real": "Real Prediction",
        "mode_quick": "Quick Backtest",
        "mode_full": "Full Backtest",
        "mode_help_real": "Predict the next trading day's direction and save it to history.",
        "mode_help_quick": "Pick one past date and see how the model would have called it.",
        "mode_help_full": "Replay a date range to measure overall accuracy.",
        "select_date": "Select date",
        "start": "Start",
        "end": "End",
        "predicting_for": "Predicting for: {date}",
        "run_analysis": "Run Analysis",
        "run_backtest": "Run Backtest",
        "spinner_running": "Fetching data & running models...",
        "choose_date": "Please choose a date.",
        "end_after_start": "End date must be after start date.",
        "need_year": "Need at least 1 year of data before the selected date.",
        "ticker_unavailable": "Ticker not found or data unavailable. Please check the symbol.",
        # Backtest safety
        "backtest_warning": "⚠️ Each day in the range retrains both models. Long ranges are slow and can crash free-tier hosting — keep it to about a week (≤ 7 days).",
        "range_too_long": "Range too long ({days} days). Please select at most {max} days to avoid timeouts on free-tier hosting.",
        "backtest_progress": "Running backtest — {done} / {total} days…",
        "backtest_long_notice": "This range spans {days} days and may take a while. Consider shortening it if the app feels slow.",
        # Result card
        "prediction": "PREDICTION",
        "bullish": "BULLISH",
        "bearish": "BEARISH",
        "xgboost": "XGBoost",
        "logreg": "Log. Reg.",
        "agreement": "Agreement",
        "both_agree": "✓ Both agree",
        "models_disagree": "✗ Models disagree",
        "key_indicators": "Key Indicators",
        "ma_cross_bullish": "Bullish",
        "ma_cross_bearish": "Bearish",
        "sentiment": "Sentiment",
        "actual_outcome_24h": "Actual outcome (24h)",
        "model_correct": "Model was correct",
        "model_wrong": "Model was wrong",
        "outcome_unavailable": "Actual outcome not available yet for this date.",
        "backtest_no_results": "No backtest results — the date range may be too short or too early (need at least 1 year of data before the start date).",
        "backtest_summary": "**XGBoost: {xgb}** over {n} predictions  |  Baseline LR: {lr}",
        "prediction_saved_no_db": "Prediction made but could not be saved to the database.",
        # News
        "latest_news": "Latest News · {ticker}",
        "no_news": "No recent news found for this ticker.",
        "overall_sentiment": "Overall sentiment: {score} ({mood})",
        "mood_positive": "Positive",
        "mood_slightly_positive": "Slightly Positive",
        "mood_negative": "Negative",
        "mood_slightly_negative": "Slightly Negative",
        "mood_neutral": "Neutral",
        # Right column
        "track_record": "Track Record",
        "acc_24h": "24h Accuracy",
        "acc_1w": "1w Accuracy",
        "acc_1m": "1m Accuracy",
        "acc_count_fmt": "{correct} of {total}",
        "no_data": "no data",
        "loading_market": "Loading market data & models…",
        "could_not_load_market": "Could not load market data for this ticker.",
        "price": "Price",
        "rsi": "RSI",
        "macd": "MACD",
        "feature_importance": "Feature Importance",
        "model_metrics": "Model Metrics",
        "not_enough_metrics": "Not enough data for reliable metrics (small test set).",
        "prediction_history": "Prediction History",
        "run_first": "Run your first analysis to see history.",
        "verify_predictions": "🔄 Verify Predictions",
        "checking_outcomes": "Checking outcomes…",
        "verified_n": "Verified {n} prediction horizon(s).",
        "verification_failed": "Verification failed. Please try again later.",
        # Table headers
        "col_date": "Date",
        "col_mode": "Mode",
        "col_direction": "Direction",
        "col_agree": "Agree",
        "col_actual": "Actual",
        "real_short": "Real",
        "backtest_short": "Backtest",
        # Directions
        "dir_up": "↑ UP",
        "dir_down": "↓ DOWN",
        "dir_none": "—",
        # Boot
        "db_starting": "Starting Market Pulse…",
        "db_error": "Database is waking up — this can take up to 30 seconds on the first visit. Please refresh. If this persists, check your Supabase credentials in secrets.toml.",
        "language": "Language",
    },
    "ru": {
        # Home
        "app_tagline": "Прогноз направления рынка с XGBoost и анализом новостей.",
        "sign_out": "Выйти",
        "new_session": "➕ Новая сессия",
        "field_name": "Название",
        "field_ticker": "Тикер",
        "field_description": "Описание (необязательно)",
        "create": "Создать",
        "name_ticker_required": "Укажите название и тикер.",
        "ticker_not_found": "Тикер не найден на Yahoo Finance. Проверьте символ.",
        "session_created": "Сессия для {ticker} создана.",
        "create_failed": "Не удалось создать сессию. Попробуйте ещё раз.",
        "sessions": "Сессии",
        "no_sessions": "Сессий пока нет. Нажмите **➕ Новая сессия** выше, чтобы создать первую.",
        "no_predictions_yet": "Прогнозов пока нет",
        "conf_fmt": "{dir} · уверенность {pct}",
        "created": "Создано {date}",
        "open": "Открыть →",
        "delete_confirm": "Удалить эту сессию? Это действие необратимо.",
        "confirm_delete": "Удалить",
        "cancel": "Отмена",
        "delete_help": "Удалить сессию",
        # Session page
        "back_to_sessions": "← К сессиям",
        "session_not_found": "Сессия не найдена.",
        "back_home": "На главную",
        "could_not_load_session": "Не удалось загрузить сессию.",
        "analysis": "Анализ",
        "mode": "Режим",
        "mode_real": "Реальный прогноз",
        "mode_quick": "Быстрый бэктест",
        "mode_full": "Полный бэктест",
        "mode_help_real": "Спрогнозировать направление на следующий торговый день и сохранить в историю.",
        "mode_help_quick": "Выберите одну прошедшую дату и посмотрите, как сработала бы модель.",
        "mode_help_full": "Прогон по диапазону дат для оценки общей точности.",
        "select_date": "Выберите дату",
        "start": "Начало",
        "end": "Конец",
        "predicting_for": "Прогноз на: {date}",
        "run_analysis": "Запустить анализ",
        "run_backtest": "Запустить бэктест",
        "spinner_running": "Загрузка данных и запуск моделей…",
        "choose_date": "Пожалуйста, выберите дату.",
        "end_after_start": "Дата конца должна быть позже даты начала.",
        "need_year": "Нужен минимум 1 год данных до выбранной даты.",
        "ticker_unavailable": "Тикер не найден или данные недоступны. Проверьте символ.",
        # Backtest safety
        "backtest_warning": "⚠️ Каждый день диапазона заново обучает обе модели. Длинные диапазоны медленные и могут уронить бесплатный сервер — держитесь в пределах недели (≤ 7 дней).",
        "range_too_long": "Диапазон слишком длинный ({days} дн.). Выберите не более {max} дней, чтобы избежать таймаутов на бесплатном хостинге.",
        "backtest_progress": "Идёт бэктест — {done} / {total} дней…",
        "backtest_long_notice": "Диапазон охватывает {days} дн. и может занять время. Сократите его, если приложение работает медленно.",
        # Result card
        "prediction": "ПРОГНОЗ",
        "bullish": "РОСТ",
        "bearish": "ПАДЕНИЕ",
        "xgboost": "XGBoost",
        "logreg": "Лог. рег.",
        "agreement": "Согласие",
        "both_agree": "✓ Обе модели согласны",
        "models_disagree": "✗ Модели расходятся",
        "key_indicators": "Ключевые индикаторы",
        "ma_cross_bullish": "Бычий",
        "ma_cross_bearish": "Медвежий",
        "sentiment": "Настроение",
        "actual_outcome_24h": "Фактический исход (24ч)",
        "model_correct": "Модель была права",
        "model_wrong": "Модель ошиблась",
        "outcome_unavailable": "Фактический исход для этой даты пока недоступен.",
        "backtest_no_results": "Нет результатов бэктеста — диапазон слишком мал или слишком ранний (нужен минимум 1 год данных до даты начала).",
        "backtest_summary": "**XGBoost: {xgb}** на {n} прогнозах  |  База LR: {lr}",
        "prediction_saved_no_db": "Прогноз сделан, но не сохранён в базе данных.",
        # News
        "latest_news": "Последние новости · {ticker}",
        "no_news": "Свежих новостей по этому тикеру не найдено.",
        "overall_sentiment": "Общее настроение: {score} ({mood})",
        "mood_positive": "Позитивное",
        "mood_slightly_positive": "Скорее позитивное",
        "mood_negative": "Негативное",
        "mood_slightly_negative": "Скорее негативное",
        "mood_neutral": "Нейтральное",
        # Right column
        "track_record": "История точности",
        "acc_24h": "Точность 24ч",
        "acc_1w": "Точность 1 нед",
        "acc_1m": "Точность 1 мес",
        "acc_count_fmt": "{correct} из {total}",
        "no_data": "нет данных",
        "loading_market": "Загрузка рыночных данных и моделей…",
        "could_not_load_market": "Не удалось загрузить рыночные данные для тикера.",
        "price": "Цена",
        "rsi": "RSI",
        "macd": "MACD",
        "feature_importance": "Важность признаков",
        "model_metrics": "Метрики модели",
        "not_enough_metrics": "Недостаточно данных для надёжных метрик (маленькая тестовая выборка).",
        "prediction_history": "История прогнозов",
        "run_first": "Запустите первый анализ, чтобы увидеть историю.",
        "verify_predictions": "🔄 Проверить прогнозы",
        "checking_outcomes": "Проверка исходов…",
        "verified_n": "Проверено прогнозов: {n}.",
        "verification_failed": "Не удалось проверить. Попробуйте позже.",
        # Table headers
        "col_date": "Дата",
        "col_mode": "Режим",
        "col_direction": "Направление",
        "col_agree": "Согласие",
        "col_actual": "Факт",
        "real_short": "Реальный",
        "backtest_short": "Бэктест",
        # Directions
        "dir_up": "↑ ВВЕРХ",
        "dir_down": "↓ ВНИЗ",
        "dir_none": "—",
        # Boot
        "db_starting": "Запуск Market Pulse…",
        "db_error": "База данных просыпается — при первом запуске это может занять до 30 секунд. Обновите страницу. Если ошибка не исчезает, проверьте учётные данные Supabase в secrets.toml.",
        "language": "Язык",
    },
}
