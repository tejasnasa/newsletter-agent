"""Shared test fixtures.

Two kinds of test live here:

* offline tests (the default) swap in :class:`FakeLLM` and fake sources, so
  they are deterministic and need no credentials;
* live tests are marked ``@pytest.mark.live`` and hit the real network.

The fake LLM works by monkeypatching ``src.llm.chat_json``. Every call site
reaches the model through the ``src.llm`` module (``from src import llm`` ->
``llm.chat_json(...)``), so patching that one attribute fakes the whole agent.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from src import llm, nodes
from src.config import Settings
from src.models import (
    Article,
    Critique,
    CurationResult,
    NewsletterDraft,
    NewsletterItem,
    ResearchPlan,
    RevisedDraft,
    RevisedItem,
)
from src.tools.summarize import ArticleSummary, NewsletterIntro

# So a live test can see a key the developer put in .env.
load_dotenv(override=False)

GOAL = "Create a weekly newsletter on the latest AI agent news and send it to our subscribers."


class FakeLLM:
    """A deterministic stand-in for the LLM.

    Records every schema it was asked for, so tests can assert that the critique
    step really ran and the revise step really ran.
    """

    def __init__(
        self,
        *,
        verdict: str = "approve",
        score: int = 8,
        selected_ids: list[str] | None = None,
        plan_queries: list[str] | None = None,
        feed_urls: list[str] | None = None,
        extra_selected_ids: list[str] | None = None,
    ) -> None:
        self.verdict = verdict
        self.score = score
        self.selected_ids = selected_ids or [f"A{i}" for i in range(1, 7)]
        self.extra_selected_ids = extra_selected_ids or []
        self.plan_queries = plan_queries or ["AI agent"]
        self.feed_urls = feed_urls or []

        self.calls: list[str] = []
        self.critiques = 0
        self.revisions = 0

    # -- the seam ------------------------------------------------------------ #

    def chat_json(self, messages, schema, *, temperature=0.0, retries=1):
        self.calls.append(schema.__name__)

        if schema is ResearchPlan:
            return ResearchPlan(
                topic="AI agent news",
                hn_queries=self.plan_queries,
                feed_urls=self.feed_urls,
                timeframe_days=14,
                target_items=6,
                angle="What shipped this week",
            )
        if schema is CurationResult:
            return CurationResult(
                selected_ids=self.selected_ids + self.extra_selected_ids,
                reasoning="fake selection",
            )
        if schema is ArticleSummary:
            return ArticleSummary(
                title="",
                summary="Fake summary grounded in the excerpt.",
                why_it_matters="Fake reason it matters.",
            )
        if schema is NewsletterIntro:
            return NewsletterIntro(
                subject="Fake subject line",
                intro="Fake intro.",
                outro="Fake outro.",
            )
        if schema is Critique:
            self.critiques += 1
            return Critique(
                score=self.score,
                issues=["fake issue"] if self.verdict == "revise" else [],
                verdict=self.verdict,
                notes="fake critique",
            )
        if schema is RevisedDraft:
            self.revisions += 1
            return RevisedDraft(
                subject="Revised subject line",
                intro="Revised intro.",
                outro="Revised outro.",
                items=[
                    RevisedItem(article_id=article_id, summary=f"Revised summary for {article_id}.")
                    for article_id in self.selected_ids
                ],
            )
        raise AssertionError(f"FakeLLM got an unexpected schema: {schema}")


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointed at a throwaway output directory, with a dummy key."""
    return Settings(
        llm_api_key="test-key",
        min_items=5,
        max_items=7,
        max_revisions=2,
        output_dir=tmp_path / "output",
        recipients=("test@example.com",),
    )


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    """Install a :class:`FakeLLM` in place of the real client."""
    fake = FakeLLM()
    monkeypatch.setattr(llm, "chat_json", fake.chat_json)
    return fake


@pytest.fixture
def live_llm_available() -> bool:
    return bool(os.environ.get("LLM_API_KEY"))


@pytest.fixture
def articles() -> list[Article]:
    """Eight real-looking candidates, as a live source would return them."""
    return [
        Article(
            title=f"Real article {index}: agents in production",
            url=f"https://example.com/story-{index}",
            source="Example Feed",
            published="2026-10-10T09:00:00+00:00",
            snippet=f"Snippet for article {index} about AI agents.",
            points=100 - index,
        )
        for index in range(1, 9)
    ]


@pytest.fixture
def fake_sources(monkeypatch, articles) -> list[Article]:
    """Replace the network-touching nodes with in-memory fakes."""

    def fake_fetch_recent(queries, feed_urls, *, settings, max_items=60):
        return list(articles)

    monkeypatch.setattr(nodes, "fetch_recent_articles", fake_fetch_recent)
    monkeypatch.setattr(nodes, "fetch_article_excerpt", lambda url, **kwargs: "excerpt " * 100)
    return articles


@pytest.fixture
def draft() -> NewsletterDraft:
    """A finished draft, used by the render/send tests."""
    return NewsletterDraft(
        subject="AI agents this week",
        intro="Three things worth knowing.",
        items=[
            NewsletterItem(
                title="Agents ship to production",
                url="https://example.com/agents",
                source="Example Feed",
                published="2026-10-10T09:00:00+00:00",
                summary="A team put agents in production.",
                why_it_matters="Production experience is the useful signal.",
                article_id="A1",
            ),
            NewsletterItem(
                title="A new agent framework",
                url="https://example.com/framework",
                source="Other Feed",
                summary="Another framework appeared.",
                why_it_matters="Framework churn is slowing down.",
                article_id="A2",
            ),
        ],
        outro="That's all for this issue.",
    )
