"""The agent's seven tools.

1. ``search_hacker_news``      — live HN Algolia search
2. ``fetch_rss_feed``          — live RSS fetch + parse
3. ``fetch_article_excerpt``   — live article text extraction
4. ``summarize_article``       — LLM summary per article
5. ``write_newsletter_intro``  — LLM subject/intro/outro
6. ``render_newsletter``       — Jinja2 + Markdown rendering
7. ``send_newsletter``         — simulated send to ``output/`` and stdout
"""

from src.tools.article import fetch_article_excerpt, html_to_text
from src.tools.render import render_newsletter, to_markdown
from src.tools.send import send_newsletter
from src.tools.sources import (
    dedupe_articles,
    fetch_recent_articles,
    fetch_rss_feed,
    filter_recent,
    keywords_from,
    rank_articles,
    relevance_score,
    search_hacker_news,
)
from src.tools.summarize import (
    fallback_intro,
    fallback_item,
    fallback_subject,
    summarize_article,
    write_newsletter_intro,
)

__all__ = [
    "dedupe_articles",
    "fallback_intro",
    "fallback_item",
    "fallback_subject",
    "fetch_article_excerpt",
    "fetch_recent_articles",
    "fetch_rss_feed",
    "filter_recent",
    "html_to_text",
    "keywords_from",
    "rank_articles",
    "relevance_score",
    "render_newsletter",
    "search_hacker_news",
    "send_newsletter",
    "summarize_article",
    "to_markdown",
    "write_newsletter_intro",
]
