"""Graph state.

``log`` uses an additive reducer so every node can append progress lines
without reading and rewriting the whole list.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from src.models import (
    Article,
    Critique,
    NewsletterDraft,
    RenderedNewsletter,
    ResearchPlan,
    ReviewDecision,
    RunMode,
    SendReceipt,
)


class NewsletterState(TypedDict, total=False):
    """The shared state passed along every edge of the graph."""

    goal: str
    mode: RunMode
    plan: ResearchPlan
    candidates: list[Article]
    selected: list[Article]
    excerpts: dict[str, str]
    draft: NewsletterDraft
    critique: Critique
    critique_history: list[Critique]
    revision_count: int
    human_decision: ReviewDecision | None
    human_feedback: str
    rendered: RenderedNewsletter | None
    receipt: SendReceipt | None
    log: Annotated[list[str], operator.add]
