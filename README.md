# Market Pulse

Stock market direction prediction platform — XGBoost + news sentiment, with session management, backtesting, and prediction verification over time.

![screenshot](screenshot.png)

## Features

- **Sessions** — organize predictions per asset (SPY, GLD, BTC-USD, UUP seeded by default; add your own).
- **Real predictions** — predict the next trading day's direction with XGBoost (main) and Logistic Regression (baseline).
- **Backtesting** — quick single-date backtest or full walk-forward backtest across a date range, with a rolling accuracy chart and a **simulated P&L equity curve** (long/short strategy vs buy & hold).
- **Honest signals** — low-conviction calls (models disagree or low confidence) surface as "Uncertain" instead of a forced direction.
- **CSV export** — download a session's full prediction history.
- **News sentiment** — Yahoo Finance RSS headlines scored by FinBERT via the HuggingFace Inference API (set `HF_TOKEN` in secrets; without it the app degrades gracefully to neutral sentiment).
- **Verification** — real predictions are automatically checked at 24h / 1 week / 1 month and scored against the actual move.
- **Track record** — per-session accuracy metrics, model metrics, confusion matrices, and a full prediction history table.
- **Bilingual UI** — switch between English and Russian from the top-right toggle.

> **Backtesting note:** the full backtest retrains both models for every trading day in the range, so long ranges are slow and can exhaust free-tier hosting. Keep ranges short (≈1 week); the app warns past 7 days and blocks ranges over ~1 month.

## Tech Stack

Python 3.11 · Streamlit · yfinance · pandas-ta · scikit-learn · XGBoost · FinBERT (HuggingFace Inference API) · feedparser · Supabase (PostgreSQL) · Plotly · pandas-market-calendars

## Local Setup

No Docker and no local database required — the app connects directly to cloud Supabase.

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/market-pulse.git
cd market-pulse

# 2. Virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your secrets file (never commit this)
#    .streamlit/secrets.toml — see the template already in the repo and
#    fill in your Supabase URL and anon key.

# 5. Run
streamlit run app.py
```

The app opens at http://localhost:8501.

### `.streamlit/secrets.toml`

```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "eyJ..."   # anon/public (publishable) key — NOT the service_role key
HF_TOKEN     = "hf_..."   # HuggingFace token — required for live news sentiment
ENABLE_AUTH  = false      # set true to enable per-user login
```

## Supabase Setup

1. Create a project at [supabase.com](https://supabase.com) (free tier is fine).
2. Open the **SQL Editor**, paste and run the entire [`supabase_setup.sql`](supabase_setup.sql) script. It creates the tables, **grants table access to the API role** (required — without the grants the app gets `permission denied for table sessions`), and configures demo-mode access (RLS disabled). It is safe to re-run.
3. Go to **Connect → Use a client library → Python** and copy the **Project URL** and **anon/public (publishable) API key** into `.streamlit/secrets.toml`.

> To switch to per-user login later, set `ENABLE_AUTH = true` in secrets and run section **3b** (RLS + policies) of `supabase_setup.sql` instead of 3a.

<details>
<summary>Schema reference (also in <code>supabase_setup.sql</code>)</summary>

```sql
-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    ticker TEXT NOT NULL,
    description TEXT,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Predictions table
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('real', 'quick_backtest', 'full_backtest')),
    backtest_date DATE,
    backtest_start DATE,
    backtest_end DATE,
    created_at TIMESTAMPTZ DEFAULT now(),
    xgb_direction TEXT CHECK (xgb_direction IN ('UP', 'DOWN')),
    lr_direction TEXT CHECK (lr_direction IN ('UP', 'DOWN')),
    xgb_confidence FLOAT,
    lr_confidence FLOAT,
    sentiment_score FLOAT,
    headlines JSONB,
    features JSONB,
    feature_importance JSONB,
    actual_24h TEXT CHECK (actual_24h IN ('UP', 'DOWN')),
    actual_1w TEXT CHECK (actual_1w IN ('UP', 'DOWN')),
    actual_1m TEXT CHECK (actual_1m IN ('UP', 'DOWN')),
    verified_24h BOOLEAN,
    verified_1w BOOLEAN,
    verified_1m BOOLEAN,
    verified_24h_at TIMESTAMPTZ,
    verified_1w_at TIMESTAMPTZ,
    verified_1m_at TIMESTAMPTZ
);

-- Multi-user support (auth is feature-flagged; defaults to shared "public")
ALTER TABLE sessions    ADD COLUMN user_id TEXT NOT NULL DEFAULT 'public';
ALTER TABLE predictions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'public';

-- Enable these only when running with ENABLE_AUTH = true
-- ALTER TABLE sessions    ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "users see own sessions"
--   ON sessions FOR ALL USING (user_id = auth.uid()::text OR user_id = 'public');
-- CREATE POLICY "users see own predictions"
--   ON predictions FOR ALL USING (user_id = auth.uid()::text OR user_id = 'public');
```

</details>

> **Note:** Supabase free-tier projects pause after ~7 days of inactivity. The first request after a pause may fail while the database wakes up (≈30 seconds). The app detects this on startup and asks you to refresh — just reload the page.

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick your repo, branch, and `app.py`.
3. Under **Advanced settings → Secrets**, paste the same contents as your local `.streamlit/secrets.toml`.
4. Deploy. The app uses the HuggingFace Inference API for sentiment, so it stays within the free-tier RAM limit (no local FinBERT).

## Project Structure

```
market-pulse/
├── app.py                  # Entry point + page router
├── src/
│   ├── data.py             # yfinance fetch + feature engineering
│   ├── model.py            # Logistic Regression + XGBoost pipeline
│   ├── sentiment.py        # Yahoo RSS + FinBERT (HF Inference API)
│   ├── storage.py          # Supabase CRUD
│   ├── charts.py           # Plotly chart builders
│   ├── verify.py           # Prediction verification logic
│   └── auth.py             # Feature-flagged Supabase Auth
├── .streamlit/
│   ├── config.toml         # Theme
│   └── secrets.toml        # Credentials (not committed)
├── requirements.txt
├── .gitignore
└── README.md
```

## Disclaimer

Market Pulse is an educational project. Its predictions are not financial advice.
