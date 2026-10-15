"""Live-network tests: the two keyless sources must return real articles.

These are marked ``live`` and are skipped by ``-m "not live"``. They exist to
prove the pipeline really talks to the network and to surface feed rot.
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.tools import fetch_article_excerpt, fetch_rss_feed, search_hacker_news

pytestmark = pytest.mark.live

TECHCRUNCH_AI_FEED = "https://techcrunch.com/category/artificial-intelligence/feed/"


def test_hacker_news_search_returns_real_articles(settings: Settings) -> None:
    articles = search_hacker_news("AI agent", hits=10, since_days=90, settings=settings)

    assert articles, "HN Algolia returned no stories for 'AI agent'"
    for article in articles:
        assert article.title
        assert article.url.startswith("http")
        assert article.source == "Hacker News"


def test_rss_feed_returns_real_articles(settings: Settings) -> None:
    articles = fetch_rss_feed(TECHCRUNCH_AI_FEED, limit=10, timeout=settings.request_timeout)

    assert articles, "the TechCrunch AI feed returned no entries"
    for article in articles:
        assert article.title
        assert article.url.startswith("http")
        assert article.source


def test_at_least_one_configured_feed_is_live(settings: Settings) -> None:
    """Feeds rot; this reports which ones still work and fails only if none do."""
    results: dict[str, str] = {}

    for feed_url in settings.rss_feed_urls:
        try:
            count = len(fetch_rss_feed(feed_url, limit=5, timeout=settings.request_timeout))
            results[feed_url] = f"{count} entries"
        except Exception as exc:  # noqa: BLE001 - reporting, not asserting
            results[feed_url] = f"FAILED: {type(exc).__name__}: {exc}"

    print("\nConfigured RSS feeds:")
    for feed_url, outcome in results.items():
        print(f"  {outcome:<40} {feed_url}")

    assert any("entries" in outcome for outcome in results.values()), (
        "none of the configured RSS feeds responded"
    )


def test_fetch_article_excerpt_reads_a_real_page(settings: Settings) -> None:
    articles = fetch_rss_feed(TECHCRUNCH_AI_FEED, limit=1, timeout=settings.request_timeout)
    if not articles:
        pytest.skip("the feed returned no entries to follow")

    try:
        text = fetch_article_excerpt(
            articles[0].url, max_chars=4000, timeout=settings.request_timeout
        )
    except Exception as exc:  # noqa: BLE001 - paywalls/bot-blocks are expected sometimes
        pytest.skip(f"article page not fetchable right now: {type(exc).__name__}: {exc}")

    assert len(text) > 200, "extracted text is suspiciously short"
    assert "<" not in text[:200], "HTML tags leaked into the extracted text"
