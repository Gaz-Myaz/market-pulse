-- Market Pulse — Supabase setup.
-- Run this ENTIRE script in the Supabase SQL Editor (it is safe to re-run).
-- It creates the tables (if missing), grants access to the API role used by the
-- anon/publishable key, and configures access for demo mode (ENABLE_AUTH = false).

-- --------------------------------------------------------------------------- --
-- 1. Tables
-- --------------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    ticker TEXT NOT NULL,
    description TEXT,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    user_id TEXT NOT NULL DEFAULT 'public'
);

CREATE TABLE IF NOT EXISTS predictions (
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
    verified_1m_at TIMESTAMPTZ,
    user_id TEXT NOT NULL DEFAULT 'public'
);

-- --------------------------------------------------------------------------- --
-- 2. Grant access to the API roles (fixes "permission denied for table", 42501)
-- --------------------------------------------------------------------------- --
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.sessions    TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.predictions TO anon, authenticated;

-- --------------------------------------------------------------------------- --
-- 3a. DEMO MODE (ENABLE_AUTH = false) — shared "public" data, no login.
--     Disabling RLS is the simplest setup for the demo.
-- --------------------------------------------------------------------------- --
ALTER TABLE public.sessions    DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions DISABLE ROW LEVEL SECURITY;

-- --------------------------------------------------------------------------- --
-- 3b. PRODUCTION MODE (ENABLE_AUTH = true) — per-user isolation.
--     Comment out section 3a above and uncomment this block instead.
-- --------------------------------------------------------------------------- --
-- ALTER TABLE public.sessions    ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
--
-- DROP POLICY IF EXISTS "users see own sessions" ON public.sessions;
-- CREATE POLICY "users see own sessions"
--   ON public.sessions FOR ALL
--   USING (user_id = auth.uid()::text OR user_id = 'public')
--   WITH CHECK (user_id = auth.uid()::text OR user_id = 'public');
--
-- DROP POLICY IF EXISTS "users see own predictions" ON public.predictions;
-- CREATE POLICY "users see own predictions"
--   ON public.predictions FOR ALL
--   USING (user_id = auth.uid()::text OR user_id = 'public')
--   WITH CHECK (user_id = auth.uid()::text OR user_id = 'public');
