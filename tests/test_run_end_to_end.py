"""End-to-end tests.

The offline tests use :class:`FakeLLM` plus fake sources, so they need no
credentials and no network. The ``live`` test at the bottom runs the real
thing and is skipped when ``LLM_API_KEY`` is not set.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from src import llm
from src.agent import NewsletterAgentError, run_newsletter_agent
from src.config import ConfigError, Settings, load_settings
from src.llm import LLMError
from src.models import RunMode
from src.tools.sources import normalize_url

GOAL = "Create a weekly newsletter on the latest AI agent news and send it to our subscribers."


def test_run_newsletter_agent_produces_five_to_seven_items(
    settings: Settings, fake_llm, fake_sources, tmp_path
) -> None:
    result = run_newsletter_agent(GOAL, mode=RunMode.AUTONOMOUS, settings=settings)

    assert 5 <= len(result.items) <= 7, f"expected 5-7 items, got {len(result.items)}"
    for item in result.items:
        assert item.title, "every item needs a title"
        assert item.url.startswith("https://"), "every item needs a real link"
        assert item.summary, "every item needs a summary"

    assert result.subject
    assert result.goal == GOAL
    assert result.mode is RunMode.AUTONOMOUS


def test_every_url_traces_to_a_fetched_article(settings: Settings, fake_llm, fake_sources) -> None:
    """The anti-fabrication guardrail, asserted on the final artifact."""
    fetched = {normalize_url(article.url) for article in fake_sources}

    result = run_newsletter_agent(GOAL, mode=RunMode.AUTONOMOUS, settings=settings)

    for item in result.items:
        assert normalize_url(item.url) in fetched, f"{item.url} is not a fetched article"


def test_pipeline_writes_both_artifacts_and_prints_the_email(
    settings: Settings, fake_llm, fake_sources, capsys
) -> None:
    result = run_newsletter_agent(GOAL, mode=RunMode.AUTONOMOUS, settings=settings)

    html_path = Path(result.html_path)
    email_path = Path(result.email_path)
    assert html_path.exists() and html_path.suffix == ".html"
    assert email_path.exists() and email_path.suffix == ".txt"
    assert html_path.parent == settings.output_dir

    email_text = email_path.read_text(encoding="utf-8")
    assert f"Subject: {result.subject}" in email_text
    assert result.items[0].url in email_text
    assert "<html" in html_path.read_text(encoding="utf-8")

    printed = capsys.readouterr().out
    assert f"Subject: {result.subject}" in printed


def test_the_agent_critiques_itself_and_revises(settings: Settings, fake_llm, fake_sources) -> None:
    """Acceptance criterion 7: the critique step is real and drives a revision."""
    fake_llm.verdict = "revise"

    result = run_newsletter_agent(GOAL, mode=RunMode.AUTONOMOUS, settings=settings)

    assert "Critique" in fake_llm.calls, "the agent must evaluate its own draft"
    assert "RevisedDraft" in fake_llm.calls, "a revise verdict must trigger a rewrite"
    assert result.revisions >= 1
    assert result.critiques, "the result must carry its critique history"
    assert any(critique.verdict == "revise" for critique in result.critiques)
    assert "revise" in " ".join(result.log)


def test_selection_is_validated_against_the_real_pool(
    settings: Settings, fake_llm, fake_sources
) -> None:
    """IDs the model invents are dropped; the count still lands in range."""
    fake_llm.extra_selected_ids = ["A99", "A100"]

    result = run_newsletter_agent(GOAL, mode=RunMode.AUTONOMOUS, settings=settings)

    assert 5 <= len(result.items) <= 7
    assert "A99" not in {item.article_id for item in result.items}
    assert any("dropped" in line for line in result.log), "unknown ids must be reported"


def test_item_count_is_enforced_in_code_not_by_the_model(
    settings: Settings, fake_llm, fake_sources
) -> None:
    """Even if the model returns nothing, the deterministic top-up fills the issue."""
    fake_llm.selected_ids = []

    result = run_newsletter_agent(GOAL, mode=RunMode.AUTONOMOUS, settings=settings)

    assert 5 <= len(result.items) <= 7


def test_hitl_mode_pauses_and_resumes_through_the_public_api(
    settings: Settings, fake_llm, fake_sources
) -> None:
    """The single public function still pauses, by driving the pause loop itself."""
    seen: list[int] = []

    def responder(request):
        from src.models import ReviewDecision

        seen.append(request.revision_count)
        return ReviewDecision(
            action="revise" if len(seen) == 1 else "approve", feedback="Tighten the intro."
        )

    result = run_newsletter_agent(
        GOAL, mode=RunMode.HUMAN_IN_THE_LOOP, responder=responder, settings=settings
    )

    assert seen == [0, 1], "the agent must have been shown the draft twice"
    assert result.mode is RunMode.HUMAN_IN_THE_LOOP
    assert result.revisions == 1
    assert 5 <= len(result.items) <= 7


def test_hitl_without_a_responder_uses_the_blocking_stdin_prompt(
    settings: Settings, fake_llm, fake_sources, monkeypatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "approve")

    result = run_newsletter_agent(GOAL, mode=RunMode.HUMAN_IN_THE_LOOP, settings=settings)

    assert result.subject


def test_load_settings_requires_a_key(monkeypatch, tmp_path) -> None:
    """No key must fail at configuration time, not 40 seconds into a run."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ConfigError) as excinfo:
        load_settings(env_file=tmp_path / "does-not-exist.env", require_key=True)

    assert "LLM_API_KEY" in str(excinfo.value)


def test_run_degrades_to_deterministic_fallbacks_when_the_llm_is_unreachable(
    settings: Settings, fake_sources, monkeypatch
) -> None:
    """Every node has a fallback, so an LLM outage produces a real issue, not a crash."""

    def unreachable(*args, **kwargs):
        raise LLMError("simulated provider outage")

    monkeypatch.setattr(llm, "chat_json", unreachable)

    result = run_newsletter_agent(GOAL, mode=RunMode.AUTONOMOUS, settings=settings)

    assert 5 <= len(result.items) <= 7, "the deterministic top-up must fill the issue"
    assert all(item.summary and item.url for item in result.items)
    assert result.subject
    assert any("fallback" in line.lower() for line in result.log)


def test_agent_reports_source_failures(settings: Settings, fake_llm, monkeypatch) -> None:
    """When every live source is down the agent says so instead of inventing news."""

    def dead(queries, feed_urls, *, settings, max_items=60):
        return []

    monkeypatch.setattr("src.nodes.fetch_recent_articles", dead)
    monkeypatch.setattr("src.nodes.search_hacker_news", lambda *a, **k: [])

    with pytest.raises(NewsletterAgentError) as excinfo:
        run_newsletter_agent(GOAL, mode=RunMode.AUTONOMOUS, settings=settings)

    assert "source" in str(excinfo.value).lower()


# --------------------------------------------------------------------------- #
# live: the real network and the real model
# --------------------------------------------------------------------------- #


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="needs LLM_API_KEY in .env")
def test_live_end_to_end(tmp_path) -> None:
    """The real thing: HN + RSS + a live LLM + render + simulated send."""
    settings = load_settings(require_key=True)
    settings = dataclasses.replace(settings, output_dir=tmp_path / "output")

    result = run_newsletter_agent(GOAL, mode=RunMode.AUTONOMOUS, settings=settings)

    assert 5 <= len(result.items) <= 7
    assert result.subject
    assert all(item.url.startswith("http") for item in result.items)
    assert Path(result.email_path).exists()
    assert result.critiques, "the live run must produce a real critique"
