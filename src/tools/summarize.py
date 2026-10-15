"""LLM-backed writing tools: per-article summaries and the issue intro.

Anti-fabrication: the model only ever supplies headline/summary/why-it-matters
text. ``url``, ``source``, ``published`` and ``article_id`` are copied from the
real :class:`Article` object here, in code, so a hallucinated link cannot reach
the newsletter.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from src import llm
from src.config import Settings
from src.models import Article, NewsletterItem
from src.prompts import INTRO_SYSTEM, INTRO_USER, SUMMARIZE_SYSTEM, SUMMARIZE_USER

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
MAX_EXCERPT_IN_PROMPT = 4000


class ArticleSummary(BaseModel):
    """What the model is allowed to write about one article."""

    title: str = ""
    summary: str = ""
    why_it_matters: str = ""


class NewsletterIntro(BaseModel):
    """What the model is allowed to write about the issue as a whole."""

    subject: str = ""
    intro: str = ""
    outro: str = ""


def summarize_article(
    article: Article,
    *,
    goal: str,
    excerpt: str,
    settings: Settings,
) -> NewsletterItem:
    """Write one :class:`NewsletterItem` for one real article."""
    source_text = (excerpt or article.snippet or "").strip()
    messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {
            "role": "user",
            "content": SUMMARIZE_USER.format(
                goal=goal,
                title=article.title,
                source=article.source,
                published=article.published or "unknown",
                excerpt=source_text[:MAX_EXCERPT_IN_PROMPT] or "(no article text available)",
            ),
        },
    ]

    result = llm.chat_json(messages, ArticleSummary, temperature=settings.llm_temperature)
    return item_from_summary(article, result, source_text)


def item_from_summary(
    article: Article, summary: ArticleSummary, source_text: str
) -> NewsletterItem:
    """Attach model-written text to the real article's identifiers."""
    return NewsletterItem(
        title=summary.title.strip() or article.title,
        url=article.url,
        source=article.source,
        published=article.published,
        summary=summary.summary.strip() or first_sentences(source_text, limit=3),
        why_it_matters=summary.why_it_matters.strip(),
        article_id=article.id,
    )


def fallback_item(article: Article, excerpt: str, reason: str) -> NewsletterItem:
    """Deterministic entry used when the LLM cannot write one."""
    source_text = (excerpt or article.snippet or "").strip()
    summary = first_sentences(source_text, limit=2) or f"Coverage from {article.source}."
    return NewsletterItem(
        title=article.title,
        url=article.url,
        source=article.source,
        published=article.published,
        summary=summary,
        why_it_matters=(
            f"Included because the agent could not summarise it automatically ({reason})."
        ),
        article_id=article.id,
    )


def write_newsletter_intro(
    items: list[NewsletterItem],
    *,
    goal: str,
    settings: Settings,
) -> tuple[str, str, str]:
    """Return ``(subject, intro, outro)`` for the issue."""
    listing = (
        "\n".join(f"- {item.title} ({item.source}): {item.summary}" for item in items)
        or "(no items)"
    )

    messages = [
        {"role": "system", "content": INTRO_SYSTEM},
        {"role": "user", "content": INTRO_USER.format(goal=goal, items=listing)},
    ]

    result = llm.chat_json(messages, NewsletterIntro, temperature=settings.llm_temperature)
    subject = result.subject.strip() or fallback_subject(goal)
    intro = result.intro.strip() or fallback_intro(items)
    outro = result.outro.strip() or FALLBACK_OUTRO
    return subject, intro, outro


FALLBACK_OUTRO = "That's all for this issue — see you next week."


def fallback_subject(goal: str) -> str:
    """A publishable subject line derived from the goal, with no model involved."""
    words = [word for word in re.findall(r"[A-Za-z0-9+.-]+", goal) if word]
    topic = " ".join(words[:8]) or "AI agent news"
    subject = f"{topic[0].upper()}{topic[1:]}"
    return subject[:80]


def fallback_intro(items: list[NewsletterItem]) -> str:
    count = len(items)
    noun = "story" if count == 1 else "stories"
    return f"This issue covers the {count} AI agent {noun} that mattered most this week."


def first_sentences(text: str, *, limit: int = 2) -> str:
    """Truncate text to its first ``limit`` sentences (or first 320 characters)."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    sentences = _SENTENCE_RE.split(cleaned)[:limit]
    return " ".join(sentences)[:320]
