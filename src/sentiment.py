"""News sentiment — Yahoo Finance RSS headlines + FinBERT via HF Inference API.

FinBERT is NOT loaded locally (Streamlit Cloud free tier has ~1GB RAM). We call
the hosted HuggingFace Inference API instead and degrade gracefully to neutral
sentiment on any failure.
"""

from __future__ import annotations

import logging
import time

import feedparser
import requests
import streamlit as st

logger = logging.getLogger("market-pulse")

# HuggingFace retired the old `api-inference.huggingface.co` host in 2025. The
# current serverless Inference endpoint is the router below. Anonymous access is
# no longer free — a token (HF_TOKEN in secrets) is required for live sentiment;
# without one the API returns 401 and the app falls back to neutral sentiment.
HF_API_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"


def _hf_token() -> str:
    try:
        return st.secrets.get("HF_TOKEN", "")
    except Exception:  # noqa: BLE001 - secrets may be absent in tests
        return ""


def fetch_headlines(ticker: str, max_items: int = 5) -> list[str]:
    """Return up to `max_items` recent headline titles for the ticker.

    Never raises — returns [] on any failure or empty feed.
    """
    try:
        url = (
            "https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={ticker}&region=US&lang=en-US"
        )
        feed = feedparser.parse(url)
        titles = [entry.title for entry in feed.entries[:max_items] if entry.get("title")]
        return titles
    except Exception as e:  # noqa: BLE001
        logger.warning(f"RSS fetch failed for {ticker}: {e}")
        return []


def call_hf_api(headline: str, retries: int = 3) -> list:
    """POST a single headline to the HF Inference API with backoff on cold start.

    Returns the parsed JSON scores list, or [] on persistent failure.
    """
    token = _hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": headline},
                timeout=10,
            )
        except Exception as e:  # noqa: BLE001 - network/DNS error, retry
            logger.warning(f"HF API call failed (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            try:
                return response.json()
            except Exception:  # noqa: BLE001
                return []
        if response.status_code == 503:
            # Model is loading (cold start) — back off and retry.
            time.sleep(2 ** attempt)
            continue
        # 401/403/4xx/5xx won't be fixed by retrying (e.g. missing HF_TOKEN).
        logger.warning(
            f"HF API returned {response.status_code}; using neutral sentiment."
        )
        return []
    return []


def analyze_sentiment(headlines: list[str]) -> tuple[float, list[dict]]:
    """Score each headline with FinBERT; return (avg_score, per-headline list).

    Score is positive_prob - negative_prob, in [-1, 1]. Neutral on failure.
    """
    if not headlines:
        return 0.0, []

    # The HF Inference API requires a token. With none set, every call 401s, so
    # skip the network entirely and show the headlines with neutral scores.
    token = _hf_token()
    if not token:
        return 0.0, [
            {"title": h, "score": 0.0, "label": "neutral"} for h in headlines
        ]

    results: list[dict] = []
    for headline in headlines:
        scores = call_hf_api(headline)
        if isinstance(scores, list) and scores:
            # API returns [[{label, score}, ...]] for a single input.
            inner = scores[0] if isinstance(scores[0], list) else scores
            try:
                label_scores = {s["label"].lower(): s["score"] for s in inner}
                score = label_scores.get("positive", 0.0) - label_scores.get(
                    "negative", 0.0
                )
                label = max(label_scores, key=label_scores.get)
                results.append(
                    {"title": headline, "score": round(score, 3), "label": label}
                )
                continue
            except (KeyError, TypeError, AttributeError):
                pass
        # Fallback: neutral.
        results.append({"title": headline, "score": 0.0, "label": "neutral"})

    if not results:
        return 0.0, []
    avg_score = sum(r["score"] for r in results) / len(results)
    return round(avg_score, 3), results
