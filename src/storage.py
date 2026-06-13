"""Supabase storage layer — connection, CRUD, and default-session seeding.

All write/read operations go through here. Queries are user-scoped when the
auth feature flag is enabled (see src/auth.py); otherwise everything lives
under the shared "public" user.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import streamlit as st
from supabase import create_client, Client

from src.auth import ENABLE_AUTH, get_user_id

logger = logging.getLogger("market-pulse")


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string PostgREST can cast to timestamptz."""
    return datetime.now(timezone.utc).isoformat()


@st.cache_resource
def get_client() -> Client:
    """Create (and cache) the Supabase client from Streamlit secrets."""
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


def check_db_connection() -> bool:
    """Lightweight ping used on startup to detect a paused/unreachable DB."""
    try:
        get_client().table("sessions").select("id").limit(1).execute()
        return True
    except Exception as e:  # noqa: BLE001 - surface any failure as "down"
        logger.error(f"DB connection failed: {e}")
        return False


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def get_all_sessions() -> list[dict]:
    """Return all sessions, most recently updated first."""
    db = get_client()
    query = db.table("sessions").select("*")
    if ENABLE_AUTH:
        query = query.eq("user_id", get_user_id())
    return query.order("updated_at", desc=True).execute().data


def get_session(session_id: str) -> dict | None:
    """Return a single session by id, or None if it does not exist."""
    db = get_client()
    res = db.table("sessions").select("*").eq("id", session_id).limit(1).execute()
    return res.data[0] if res.data else None


def create_session(
    name: str,
    ticker: str,
    description: str = "",
    is_default: bool = False,
) -> dict:
    """Insert a new session and return the created row."""
    db = get_client()
    row = {
        "name": name,
        "ticker": ticker.upper(),
        "description": description,
        "is_default": is_default,
        "user_id": get_user_id() or "public",
    }
    res = db.table("sessions").insert(row).execute()
    return res.data[0]


def delete_session(session_id: str) -> None:
    """Delete a session (predictions cascade-delete via the FK)."""
    db = get_client()
    db.table("sessions").delete().eq("id", session_id).execute()


def seed_default_sessions() -> None:
    """Create the four default sessions if the sessions table is empty."""
    try:
        existing = get_all_sessions()
        if existing:
            return
        defaults = [
            ("S&P 500 Index", "SPY", "SPDR S&P 500 ETF Trust"),
            ("Gold ETF", "GLD", "SPDR Gold Shares"),
            ("Bitcoin", "BTC-USD", "Bitcoin vs US Dollar"),
            ("US Dollar ETF", "UUP", "Invesco DB US Dollar Index"),
        ]
        for name, ticker, description in defaults:
            create_session(name, ticker, description, is_default=True)
        logger.info("Seeded default sessions.")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to seed default sessions: {e}")


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def save_prediction(session_id: str, prediction_data: dict) -> dict:
    """Persist a prediction row and bump the parent session's updated_at."""
    db = get_client()
    row = dict(prediction_data)
    row["session_id"] = session_id
    row["user_id"] = get_user_id() or "public"
    res = db.table("predictions").insert(row).execute()
    # Touch the session so it sorts to the top of the home list.
    try:
        db.table("sessions").update({"updated_at": _utc_now_iso()}).eq(
            "id", session_id
        ).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not bump session updated_at: {e}")
    return res.data[0]


def get_predictions(session_id: str, limit: int = 50) -> list[dict]:
    """Return up to `limit` predictions for a session, newest first."""
    db = get_client()
    query = db.table("predictions").select("*").eq("session_id", session_id)
    if ENABLE_AUTH:
        query = query.eq("user_id", get_user_id())
    return query.order("created_at", desc=True).limit(limit).execute().data


def update_verification(
    prediction_id: str,
    horizon: str,
    verified: bool,
    actual: str,
) -> None:
    """Write verification results for a single horizon (24h / 1w / 1m)."""
    db = get_client()
    patch = {
        f"verified_{horizon}": verified,
        f"actual_{horizon}": actual,
        f"verified_{horizon}_at": _utc_now_iso(),
    }
    db.table("predictions").update(patch).eq("id", prediction_id).execute()


def get_unverified_real_predictions() -> list[dict]:
    """Return real predictions that still have at least one unverified horizon."""
    db = get_client()
    query = db.table("predictions").select("*").eq("mode", "real")
    if ENABLE_AUTH:
        query = query.eq("user_id", get_user_id())
    rows = query.execute().data
    return [
        r
        for r in rows
        if r.get("verified_24h") is None
        or r.get("verified_1w") is None
        or r.get("verified_1m") is None
    ]
