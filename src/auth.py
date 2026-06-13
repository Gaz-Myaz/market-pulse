"""Feature-flagged authentication.

Auth is OFF by default. When disabled, every request maps to the shared
"public" user and the app behaves like a single-tenant demo. When the
ENABLE_AUTH secret is true, users sign in via Supabase Auth and each user
sees only their own sessions/predictions (enforced by RLS in the DB plus
user_id filtering in src/storage.py).
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger("market-pulse")

ENABLE_AUTH = bool(st.secrets.get("ENABLE_AUTH", False))


def get_user_id() -> str | None:
    """Return the current user_id. 'public' when auth is disabled.

    Returns None when auth is enabled but no user is signed in yet.
    """
    if not ENABLE_AUTH:
        return "public"
    user = st.session_state.get("user")
    if not user:
        return None
    # supabase-py returns a User object; support both object and dict.
    return getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)


def require_auth() -> None:
    """Render a login wall and halt the run if auth is on and user is absent."""
    if not ENABLE_AUTH:
        return
    if not st.session_state.get("user"):
        render_login_page()
        st.stop()


def render_login_page() -> None:
    """Minimal email/password sign-in + sign-up form."""
    # Imported lazily to avoid a circular import with storage.py.
    from src.storage import get_client

    st.title("Sign in to Market Pulse")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sign in"):
            try:
                client = get_client()
                res = client.auth.sign_in_with_password(
                    {"email": email, "password": password}
                )
                st.session_state["user"] = res.user
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Login failed: {e}")
    with col2:
        if st.button("Sign up"):
            try:
                client = get_client()
                client.auth.sign_up({"email": email, "password": password})
                st.success("Check your email to confirm signup.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Signup failed: {e}")


def sign_out() -> None:
    """Sign the current user out and clear session state."""
    try:
        from src.storage import get_client

        get_client().auth.sign_out()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Sign out error: {e}")
    st.session_state.pop("user", None)
