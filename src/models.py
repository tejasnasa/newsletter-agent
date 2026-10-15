"""Shared pydantic models.

Every layer (tools, nodes, runner, web) exchanges these types, so the data
contract lives in exactly one place.

Two models exist purely as a fabrication guardrail:

* the LLM is only ever asked to return :class:`CurationResult` (candidate *IDs*)
  or :class:`RevisedItem` (text only, keyed by ``article_id``);
* :class:`NewsletterItem.url` is therefore always copied from a real
  :class:`Article` object, never written by the model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def utc_now_iso() -> str:
    """Timestamp used on rendered newsletters and send receipts."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class RunMode(StrEnum):
    """Fully autonomous vs human-in-the-loop."""

    AUTONOMOUS = "autonomous"
    HUMAN_IN_THE_LOOP = "hitl"

    @classmethod
    def parse(cls, value: RunMode | str) -> RunMode:
        """Accept a RunMode or its string value, case-insensitively."""
        if isinstance(value, RunMode):
            return value
        text = str(value).strip().lower().replace("_", "-")
        aliases = {
            "auto": cls.AUTONOMOUS,
            "autonomous": cls.AUTONOMOUS,
            "hitl": cls.HUMAN_IN_THE_LOOP,
            "human-in-the-loop": cls.HUMAN_IN_THE_LOOP,
            "human": cls.HUMAN_IN_THE_LOOP,
        }
        if text not in aliases:
            raise ValueError(f"unknown run mode {value!r}; expected 'autonomous' or 'hitl'")
        return aliases[text]


class Article(BaseModel):
    """One real news item discovered by a live source.

    ``url`` always comes from the source (HN Algolia or an RSS feed) — it is
    never written by the LLM.
    """

    id: str = ""
    title: str
    url: str
    source: str
    published: str | None = None
    snippet: str = ""
    points: int | None = None
    found_by: str | None = None
    relevance: float = 0.0


class ResearchPlan(BaseModel):
    """What ``plan_node`` decides before any searching happens."""

    topic: str
    hn_queries: list[str] = Field(default_factory=list)
    feed_urls: list[str] = Field(default_factory=list)
    timeframe_days: int = 14
    target_items: int = 6
    angle: str = ""

    @field_validator("timeframe_days")
    @classmethod
    def _clamp_timeframe(cls, value: int) -> int:
        return max(1, min(int(value), 30))

    @field_validator("target_items")
    @classmethod
    def _clamp_target(cls, value: int) -> int:
        return max(4, min(int(value), 8))

    @field_validator("hn_queries", "feed_urls")
    @classmethod
    def _clean_list(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]


class CurationResult(BaseModel):
    """The LLM's selection: candidate IDs only, never URLs or article text."""

    selected_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""


class NewsletterItem(BaseModel):
    """A finished newsletter entry.

    ``url``, ``source``, ``published`` and ``article_id`` are copied from the
    real :class:`Article`; only ``summary``/``why_it_matters`` are model-written.
    """

    title: str
    url: str
    source: str
    published: str | None = None
    summary: str
    why_it_matters: str = ""
    article_id: str = ""


class NewsletterDraft(BaseModel):
    """The full newsletter before critique."""

    subject: str
    intro: str = ""
    items: list[NewsletterItem] = Field(default_factory=list)
    outro: str = ""
    revision: int = 0


class RevisedItem(BaseModel):
    """A rewritten item, keyed back to its real article by ``article_id``."""

    article_id: str
    title: str = ""
    summary: str = ""
    why_it_matters: str = ""


class RevisedDraft(BaseModel):
    """The shape ``revise_node`` asks the LLM for."""

    subject: str = ""
    intro: str = ""
    outro: str = ""
    items: list[RevisedItem] = Field(default_factory=list)


class Critique(BaseModel):
    """The agent's self-evaluation of its own draft."""

    score: int = 0
    issues: list[str] = Field(default_factory=list)
    verdict: Literal["approve", "revise"] = "revise"
    fabricated_urls: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, value: object) -> int:
        try:
            number = int(float(value))  # models sometimes reply "7.0" or "7"
        except TypeError, ValueError:
            return 0
        return max(0, min(number, 10))

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalise_verdict(cls, value: object) -> str:
        text = str(value or "").strip().lower()
        if text in {"approve", "approved", "ok", "accept", "pass", "good"}:
            return "approve"
        return "revise"

    @property
    def has_fabrications(self) -> bool:
        return bool(self.fabricated_urls)


class ReviewDecision(BaseModel):
    """A human's answer at the approval gate."""

    action: Literal["approve", "revise"] = "approve"
    feedback: str = ""

    @field_validator("action", mode="before")
    @classmethod
    def _normalise_action(cls, value: object) -> str:
        text = str(value or "").strip().lower()
        return "revise" if text.startswith(("r", "change", "edit")) else "approve"


class ReviewRequest(BaseModel):
    """What the human is shown when the graph suspends."""

    thread_id: str
    goal: str
    mode: RunMode
    draft: NewsletterDraft
    critique: Critique
    item_count: int
    revision_count: int
    remaining_revisions: int
    can_request_changes: bool


class RenderedNewsletter(BaseModel):
    """Markdown + HTML output of ``render_node``."""

    subject: str
    markdown: str
    html: str
    generated_at: str = Field(default_factory=utc_now_iso)


class SendReceipt(BaseModel):
    """Result of the simulated send."""

    to: list[str]
    subject: str
    body: str
    html_path: str
    email_path: str
    simulated: bool = True
    sent_at: str = Field(default_factory=utc_now_iso)


class NewsletterResult(BaseModel):
    """The final answer returned by :func:`src.agent.run_newsletter_agent`."""

    goal: str
    mode: RunMode
    subject: str
    items: list[NewsletterItem]
    markdown: str
    html: str
    email_body: str
    html_path: str
    email_path: str
    critiques: list[Critique] = Field(default_factory=list)
    revisions: int = 0
    generated_at: str = Field(default_factory=utc_now_iso)
    log: list[str] = Field(default_factory=list)
