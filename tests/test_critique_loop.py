"""Offline tests for both loops: the self-critique loop and the human loop.

These are the tests that would catch a breaking ``langgraph`` upgrade: they
assert that ``interrupt()`` really suspends the graph and that ``Command(resume=...)``
really continues it.
"""

from __future__ import annotations

from src import nodes
from src.config import Settings
from src.models import (
    Article,
    Critique,
    NewsletterDraft,
    NewsletterItem,
    ReviewDecision,
    RevisedDraft,
    RevisedItem,
    RunMode,
)
from src.runner import Runner


def make_state(**overrides) -> dict:
    """A minimal state for exercising the routers directly."""
    base = {
        "goal": "goal",
        "mode": RunMode.AUTONOMOUS,
        "revision_count": 0,
        "critique": Critique(score=5, verdict="approve"),
    }
    base.update(overrides)
    return base


# -- routers ---------------------------------------------------------------- #


def test_route_after_critique_revises_when_verdict_says_so(settings: Settings) -> None:
    state = make_state(critique=Critique(score=4, verdict="revise"), revision_count=0)

    assert nodes.route_after_critique(state, settings) == "revise"


def test_route_after_critique_stops_at_the_revision_cap(settings: Settings) -> None:
    at_cap = make_state(
        critique=Critique(score=4, verdict="revise"), revision_count=settings.max_revisions
    )

    assert nodes.route_after_critique(at_cap, settings) == "human_review"


def test_route_after_critique_advances_on_approval(settings: Settings) -> None:
    state = make_state(critique=Critique(score=9, verdict="approve"))

    assert nodes.route_after_critique(state, settings) == "human_review"


def test_route_after_human_review_only_loops_for_change_requests(settings: Settings) -> None:
    request_changes = make_state(
        human_decision=ReviewDecision(action="revise", feedback="tighten it")
    )
    approve = make_state(human_decision=ReviewDecision(action="approve"))

    assert nodes.route_after_human_review(request_changes, settings) == "revise"
    assert nodes.route_after_human_review(approve, settings) == "render"


def test_route_after_human_review_respects_the_cap(settings: Settings) -> None:
    exhausted = make_state(
        human_decision=ReviewDecision(action="revise", feedback="again"),
        revision_count=settings.max_revisions,
    )

    assert nodes.route_after_human_review(exhausted, settings) == "render"


# -- the self-critique loop, end to end through the graph ------------------- #


def test_critique_loop_revises_exactly_max_revisions_times(
    monkeypatch, settings: Settings, fake_sources, fake_llm
) -> None:
    """A critique that never approves must stop at the cap, not loop forever."""
    fake_llm.verdict = "revise"
    fake_llm.score = 3

    runner = Runner(settings)
    snapshot = runner.start("weekly AI agent news", RunMode.AUTONOMOUS)

    assert snapshot.status == "done", snapshot.error
    result = snapshot.result
    assert fake_llm.critiques == settings.max_revisions + 1, (
        "one critique per draft, plus the first"
    )
    assert fake_llm.revisions == settings.max_revisions
    assert result.revisions == settings.max_revisions
    assert len(result.critiques) == settings.max_revisions + 1
    assert result.subject == "Revised subject line", "the revision must reach the final newsletter"


def test_critique_loop_does_not_revise_when_the_draft_is_approved(
    fake_sources, fake_llm, settings: Settings
) -> None:
    runner = Runner(settings)

    snapshot = runner.start("weekly AI agent news", RunMode.AUTONOMOUS)

    assert snapshot.status == "done"
    assert fake_llm.revisions == 0
    assert snapshot.result.revisions == 0
    assert snapshot.result.subject == "Fake subject line"


# -- the human loop: a real checkpoint-and-resume round trip ---------------- #


def test_hitl_pauses_at_human_review(fake_sources, fake_llm, settings: Settings) -> None:
    runner = Runner(settings)

    snapshot = runner.start("weekly AI agent news", RunMode.HUMAN_IN_THE_LOOP)

    assert snapshot.status == "paused", snapshot.error
    assert snapshot.next_node == "human_review"
    assert snapshot.review is not None
    assert snapshot.review.mode == RunMode.HUMAN_IN_THE_LOOP
    assert snapshot.review.item_count == 6
    assert snapshot.review.thread_id == runner.thread_id
    assert snapshot.result is None, "nothing after human_review may have run yet"
    assert "send" not in " ".join(snapshot.log), "the send step must not have run"


def test_hitl_human_feedback_drives_a_revision_then_returns_for_approval(
    fake_sources, fake_llm, settings: Settings
) -> None:
    runner = Runner(settings)
    paused = runner.start("weekly AI agent news", RunMode.HUMAN_IN_THE_LOOP)
    assert paused.review.can_request_changes

    second = runner.resume(
        ReviewDecision(action="revise", feedback="Lead with the enterprise angle.")
    )

    assert second.status == "paused", second.error
    assert second.review.revision_count == 1
    assert second.review.draft.subject == "Revised subject line"
    assert fake_llm.revisions == 1

    third = runner.resume(ReviewDecision(action="approve"))

    assert third.status == "done", third.error
    assert third.result.revisions == 1
    assert third.result.items[0].url.startswith("https://example.com/")


def test_hitl_autonomous_mode_never_pauses(fake_sources, fake_llm, settings: Settings) -> None:
    runner = Runner(settings)

    snapshot = runner.start("weekly AI agent news", RunMode.AUTONOMOUS)

    assert snapshot.status == "done"
    assert snapshot.review is None


# -- the anti-fabrication guardrail ---------------------------------------- #


def test_fabrication_check_forces_a_revision() -> None:
    selected = [Article(id="A1", title="Real", url="https://example.com/real", source="s")]
    draft = NewsletterDraft(
        subject="s",
        items=[
            NewsletterItem(
                title="Real",
                url="https://example.com/real",
                source="s",
                summary="x",
                article_id="A1",
            ),
            NewsletterItem(
                title="Invented",
                url="https://example.com/invented",
                source="s",
                summary="x",
                article_id="A2",
            ),
        ],
    )

    flagged = nodes.apply_fabrication_check(Critique(score=9, verdict="approve"), draft, selected)

    assert flagged.verdict == "revise"
    assert flagged.fabricated_urls == ["https://example.com/invented"]
    assert flagged.has_fabrications


def test_render_refuses_a_fabricated_url() -> None:
    selected = [Article(id="A1", title="Real", url="https://example.com/real", source="s")]
    draft = NewsletterDraft(
        subject="s",
        items=[
            NewsletterItem(
                title="Invented", url="https://example.com/invented", source="s", summary="x"
            )
        ],
    )

    try:
        nodes.assert_no_fabricated_urls(draft, selected)
    except nodes.FabricationError as exc:
        assert "not present in the fetched candidate pool" in str(exc)
    else:  # pragma: no cover - would mean the guardrail is gone
        raise AssertionError("render_node must refuse an untraceable URL")


def test_revision_cannot_add_articles(settings: Settings) -> None:
    """rebuild_draft keeps exactly the selected articles, whatever the model says."""
    selected = [
        Article(id="A1", title="Real one", url="https://example.com/1", source="s"),
        Article(id="A2", title="Real two", url="https://example.com/2", source="s"),
    ]
    previous = NewsletterDraft(
        subject="old",
        items=[
            NewsletterItem(
                title="Real one",
                url="https://example.com/1",
                source="s",
                summary="s1",
                article_id="A1",
            ),
            NewsletterItem(
                title="Real two",
                url="https://example.com/2",
                source="s",
                summary="s2",
                article_id="A2",
            ),
        ],
    )

    revised = RevisedDraft(
        subject="new",
        items=[
            RevisedItem(article_id="A1", summary="better one"),
            RevisedItem(article_id="A9", summary="invented article"),
        ],
    )

    rebuilt = nodes.rebuild_draft(revised, selected, previous, revision=1)

    assert [item.article_id for item in rebuilt.items] == ["A1", "A2"]
    assert rebuilt.items[0].url == "https://example.com/1"
    assert rebuilt.items[0].summary == "better one"
    assert rebuilt.items[1].summary == "s2", "a forgotten article keeps its previous text"
    assert rebuilt.subject == "new"


def test_fake_llm_is_actually_installed(fake_llm) -> None:
    """Guard against the fixtures silently stopping patching the LLM."""
    from src import llm

    assert llm.chat_json.__self__ is fake_llm
