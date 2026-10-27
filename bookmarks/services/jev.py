"""
Jev-based auto tagging.

Uses the OpenRouter Decisions API to classify a bookmarked page against the
user's tags. Each existing tag is turned into a "noul" (yes/no) question that
the Jev decision model answers with a probability between 0 and 1. The
resulting scores can be used to recommend the most relevant tags for a page,
sorted by score.

The OpenRouter endpoint is a fixed third-party URL (not user-supplied), so we
use plain `requests` here - the same approach as fetching the latest version
in the settings view. The SSRF-protected HTTP client in `http_client` is
meant for user-supplied URLs only.
"""

import logging
from functools import lru_cache
from time import time

import requests  # noqa: TID251 - fixed OpenRouter API URL, not a user-supplied URL
from django.conf import settings
from django.core import signing

logger = logging.getLogger(__name__)

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

# Default threshold for recommending a tag. Only tags with a score equal to or
# above this value are recommended.
RECOMMENDATION_THRESHOLD = 0.5


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for storage in the database."""
    if not api_key:
        return ""
    return signing.dumps(api_key)


def decrypt_api_key(stored_api_key: str) -> str:
    """Decrypt an API key that was stored with `encrypt_api_key`."""
    if not stored_api_key:
        return ""
    try:
        return signing.loads(stored_api_key)
    except (signing.BadSignature, signing.SignatureExpired):
        logger.error("Failed to decrypt OpenRouter API key")
        return ""


def _dedupe_tag_names(tag_names: list[str]) -> list[str]:
    """Dedupe tag names case-insensitively, preserving order of first occurrence."""
    seen = set()
    result = []
    for tag_name in tag_names:
        key = tag_name.lower()
        if key not in seen:
            seen.add(key)
            result.append(tag_name)
    return result


def build_questions(tag_names: list[str]) -> dict:
    """
    Build the Decisions API questions for a set of tag names.

    Each tag becomes a "noul" question - a yes/no question that is answered
    with a probability between 0 and 1. Tag names are quoted verbatim so that
    the model matches on the tag name itself, regardless of its language.
    """
    return {
        tag_name: {
            "type": "noul",
            "instructions": f'Is the bookmarked page about "{tag_name}"?',
        }
        for tag_name in tag_names
    }


def get_recommendations(
    url: str,
    title: str,
    description: str,
    tag_names: list[str],
    api_key: str,
) -> list[dict]:
    """
    Classify a page against the given tags and return recommended tags,
    sorted by score (highest first), filtered by the recommendation threshold.

    Returns a list of dicts with `name` and `score` keys. Raises an exception
    if the Decisions API call fails - callers are expected to handle that.
    """
    tag_names = _dedupe_tag_names(tag_names)
    if not tag_names or not api_key:
        return []

    payload = {
        "model": settings.LD_OPENROUTER_MODEL,
        "state": {
            "url": url,
            "title": title,
            "description": description,
        },
        "questions": build_questions(tag_names),
    }

    response = requests.post(
        OPENROUTER_DECISIONS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    answers = data.get("answers", {})

    results = []
    for tag_name in tag_names:
        answer = answers.get(tag_name, {})
        score = answer.get("noul")
        if isinstance(score, (int, float)):
            results.append({"name": tag_name, "score": score})

    results.sort(key=lambda result: result["score"], reverse=True)
    return [result for result in results if result["score"] >= RECOMMENDATION_THRESHOLD]


def get_ttl_hash(seconds: int = 3600) -> int:
    """Return the same value within `seconds` time period, to invalidate caches."""
    return round(time() / seconds)


@lru_cache(maxsize=256)
def _get_recommendations_cached(
    url: str,
    title: str,
    description: str,
    tag_names: tuple,
    api_key: str,
    ttl_hash: int,
) -> tuple:
    return tuple(get_recommendations(url, title, description, list(tag_names), api_key))


def get_recommendations_cached(
    url: str,
    title: str,
    description: str,
    tag_names: list[str],
    api_key: str,
) -> list[dict]:
    """
    Like `get_recommendations`, but caches results per page and tag set for one
    hour to keep the number of paid API calls low.
    """
    tag_names = _dedupe_tag_names(tag_names)
    if not tag_names or not api_key:
        return []
    return list(
        _get_recommendations_cached(
            url,
            title,
            description,
            tuple(tag_names),
            api_key,
            get_ttl_hash(),
        )
    )
