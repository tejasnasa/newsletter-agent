"""Live article fetching: Hacker News (Algolia) and RSS feeds.

Both sources are keyless. Every per-source call is individually guarded by the
caller so one dead feed can never fail a run.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from html import unescape

import feedparser
import httpx

from src.config import Settings
from src.models import Article
from src.tools.article import html_to_text

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
USER_AGENT = "newsletter-agent/0.1 (educational assignment; keyless news research)"
SNIPPET_CHARS = 600

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def search_hacker_news(
    query: str,
    *,
    hits: int = 20,
    since_days: int = 14,
    settings: Settings,
) -> list[Article]:
    """Search HN stories by date and return real :class:`Article` objects."""
    cutoff = int((datetime.now(UTC) - timedelta(days=since_days)).timestamp())
    params = {
        "query": query,
        "tags": "story",
        "hitsPerPage": str(hits),
        "numericFilters": f"created_at_i>{cutoff}",
    }

    with httpx.Client(
        timeout=settings.request_timeout, headers={"User-Agent": USER_AGENT}
    ) as client:
        response = client.get(HN_SEARCH_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    articles: list[Article] = []
    for hit in payload.get("hits", []):
        article = _hn_hit_to_article(hit, query)
        if article is not None:
            articles.append(article)
    return articles


def _hn_hit_to_article(hit: dict, query: str) -> Article | None:
    title = _clean_text(hit.get("title") or hit.get("story_title") or "")
    if not title:
        return None

    object_id = hit.get("objectID")
    url = hit.get("url") or (
        f"https://news.ycombinator.com/item?id={object_id}" if object_id else ""
    )
    if not url:
        return None

    snippet = _clean_text(hit.get("story_text") or hit.get("comment_text") or "")
    return Article(
        title=title,
        url=url,
        source="Hacker News",
        published=_iso_from_timestamp(hit.get("created_at_i")),
        snippet=snippet[:SNIPPET_CHARS],
        points=hit.get("points"),
        found_by=f"HN search: {query}",
    )


def fetch_rss_feed(feed_url: str, *, limit: int = 20, timeout: float = 15.0) -> list[Article]:
    """Fetch and parse one RSS/Atom feed over httpx.

    Handing the raw bytes to ``feedparser`` (rather than a URL) means we control
    the HTTP layer: timeouts, redirects, and the User-Agent.
    """
    with httpx.Client(
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        response = client.get(feed_url)
        response.raise_for_status()
        content = response.content

    parsed = feedparser.parse(content)
    feed_title = _clean_text(parsed.feed.get("title", "")) or _host_of(feed_url)
    if getattr(parsed, "bozo", 0) and not parsed.entries:
        raise ValueError(f"feed did not parse and contained no entries: {feed_url}")

    articles: list[Article] = []
    for entry in parsed.entries[:limit]:
        title = _clean_text(entry.get("title", ""))
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        published = entry.get("published") or entry.get("updated") or ""
        articles.append(
            Article(
                title=title,
                url=link,
                source=feed_title,
                published=_iso_from_struct(
                    entry.get("published_parsed") or entry.get("updated_parsed")
                )
                or published.strip()
                or None,
                snippet=_clean_text(entry.get("summary") or entry.get("description") or "")[
                    :SNIPPET_CHARS
                ],
                found_by=f"RSS: {feed_title}",
            )
        )
    return articles


def fetch_recent_articles(
    queries: list[str],
    feed_urls: list[str],
    *,
    settings: Settings,
    max_items: int = 60,
) -> list[Article]:
    """Fan out over every source, then merge and deduplicate.

    Never raises for a single bad source: a dead feed or a failed query is
    simply skipped, so the run continues if *any* source responds.
    """
    collected: list[Article] = []

    for query in queries:
        try:
            collected.extend(
                search_hacker_news(
                    query,
                    hits=settings.hn_hits_per_query,
                    since_days=settings.timeframe_days,
                    settings=settings,
                )
            )
        except Exception:  # noqa: BLE001 - one bad source must not kill the run
            continue

    for feed_url in feed_urls:
        try:
            collected.extend(fetch_rss_feed(feed_url, timeout=settings.request_timeout))
        except Exception:  # noqa: BLE001 - one bad feed must not kill the run
            continue

    return dedupe_articles(collected)[:max_items]


def dedupe_articles(articles: list[Article]) -> list[Article]:
    """Drop repeats by normalised URL first, then by normalised title."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[Article] = []

    for article in articles:
        url_key = normalize_url(article.url)
        title_key = normalize_title(article.title)
        if url_key in seen_urls or (title_key and title_key in seen_titles):
            continue
        seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        unique.append(article)

    return unique


def normalize_url(url: str) -> str:
    """Collapse a URL to a comparable key (scheme, www, query, fragment gone)."""
    text = (url or "").strip().lower()
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    text = text.split("#", 1)[0].split("?", 1)[0]
    return text.rstrip("/")


def normalize_title(title: str) -> str:
    """Reduce a headline to lowercase alphanumerics for duplicate detection."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def keywords_from(goal: str, *extra: str) -> list[str]:
    """Extract coarse keywords used to score relevance before the LLM sees the pool."""
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "to",
        "of",
        "in",
        "on",
        "with",
        "our",
        "create",
        "make",
        "write",
        "weekly",
        "newsletter",
        "latest",
        "news",
        "send",
        "it",
        "its",
        "this",
        "that",
        "about",
        "from",
        "into",
        "using",
        "use",
        "some",
        "subscribers",
        "please",
        "should",
        "must",
        "will",
        "can",
        "top",
        "best",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+.-]{2,}", f"{goal} {' '.join(extra)}".lower())
    keywords: list[str] = []
    for word in words:
        if word in stopwords or word in keywords:
            continue
        keywords.append(word)
    return keywords


def relevance_score(article: Article, keywords: list[str]) -> float:
    """Cheap on-topic score: keyword overlap, recency, and HN points."""
    title = article.title.lower()
    snippet = article.snippet.lower()

    score = 0.0
    for keyword in keywords:
        if keyword in title:
            score += 2.0
        elif keyword in snippet:
            score += 1.0

    age_days = age_in_days(article.published)
    if age_days is not None:
        score += max(0.0, 3.0 - age_days / 7.0)
    if article.points:
        score += min(article.points / 100.0, 1.5)

    return round(score, 3)


def age_in_days(published: str | None) -> float | None:
    """Age of an article in days, or ``None`` when the date is unusable."""
    if not published:
        return None
    text = published.strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - moment).total_seconds() / 86400.0)


def rank_articles(articles: list[Article], keywords: list[str]) -> list[Article]:
    """Attach relevance scores and sort best-first."""
    scored = [
        article.model_copy(update={"relevance": relevance_score(article, keywords)})
        for article in articles
    ]
    return sorted(scored, key=lambda item: (item.relevance, item.published or ""), reverse=True)


def filter_recent(articles: list[Article], timeframe_days: int) -> list[Article]:
    """Drop items we can date and that fall outside the window.

    Undated items are kept: RSS feeds frequently omit dates, and dropping them
    would lose real articles.
    """
    fresh: list[Article] = []
    for article in articles:
        age = age_in_days(article.published)
        if age is None or age <= timeframe_days:
            fresh.append(article)
    return fresh


def _iso_from_timestamp(value: object) -> str | None:
    try:
        moment = datetime.fromtimestamp(int(value), tz=UTC)  # type: ignore[arg-type]
    except TypeError, ValueError, OSError:
        return None
    return moment.replace(microsecond=0).isoformat()


def _iso_from_struct(struct: object) -> str | None:
    if not struct:
        return None
    try:
        moment = datetime(*struct[:6], tzinfo=UTC)  # type: ignore[index]
    except TypeError, ValueError:
        return None
    return moment.replace(microsecond=0).isoformat()


def _clean_text(value: str) -> str:
    text = _TAG_RE.sub(" ", str(value or ""))
    text = unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _host_of(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url).split("/", 1)[0] or url


__all__ = [
    "dedupe_articles",
    "fetch_recent_articles",
    "fetch_rss_feed",
    "filter_recent",
    "html_to_text",
    "keywords_from",
    "normalize_title",
    "normalize_url",
    "rank_articles",
    "relevance_score",
    "search_hacker_news",
]
