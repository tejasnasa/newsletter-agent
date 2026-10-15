"""The single public entry point: :func:`run_newsletter_agent`.

One call does the whole job. In HITL mode the pause loop lives here, so the
caller still sees one function while the graph genuinely suspends and resumes.
"""

from __future__ import annotations

from collections.abc import Callable

from src.config import Settings, load_settings
from src.models import NewsletterResult, ReviewDecision, ReviewRequest, RunMode
from src.runner import Runner

Responder = Callable[[ReviewRequest], ReviewDecision]


class NewsletterAgentError(RuntimeError):
    """Raised when a run ends in an error state."""


def run_newsletter_agent(
    goal: str,
    *,
    mode: RunMode | str = RunMode.AUTONOMOUS,
    responder: Responder | None = None,
    settings: Settings | None = None,
    on_log: Callable[[list[str]], None] | None = None,
) -> NewsletterResult:
    """Research, write, self-critique, and "send" a newsletter for ``goal``.

    Args:
        goal: plain-English goal, e.g. "Create a weekly newsletter on the
            latest AI agent news and send it to our subscribers."
        mode: :class:`RunMode` (or its string value). ``hitl`` pauses at the
            approval gate until ``responder`` answers.
        responder: called with the :class:`ReviewRequest` whenever the graph
            suspends. Defaults to a blocking stdin prompt.
        settings: defaults to :func:`src.config.load_settings`.
        on_log: optional callback receiving the progress log as it grows.

    Returns:
        The finished :class:`NewsletterResult`.

    Raises:
        NewsletterAgentError: the run failed (missing key, dead network, ...).
    """
    resolved = settings or load_settings()
    runner = Runner(resolved, on_log=on_log)

    snapshot = runner.start(goal, mode)
    while snapshot.status == "paused":
        handler = responder or stdin_responder
        if snapshot.review is None:
            raise NewsletterAgentError(
                f"run paused at {snapshot.next_node!r} without a review request; cannot continue"
            )
        snapshot = runner.resume(handler(snapshot.review))

    if snapshot.status == "error" or snapshot.result is None:
        raise NewsletterAgentError(
            snapshot.error or "the run finished without producing a newsletter"
        )

    return snapshot.result


def stdin_responder(request: ReviewRequest) -> ReviewDecision:
    """Blocking reviewer prompt used by the CLI in HITL mode.

    This really blocks on ``input()`` until the user answers, which is what
    "genuinely pauses for input" means for a terminal workflow.
    """
    print_review_request(request)

    while True:
        try:
            answer = input("Approve this newsletter? [a]pprove, or [r]evise <feedback>: ").strip()
        except EOFError:  # non-interactive stdin: do not hang the demo
            print("(no input available — approving so the run can finish)")
            return ReviewDecision(action="approve", feedback="")

        if not answer:
            continue
        if answer[0].lower() == "a":
            return ReviewDecision(action="approve", feedback="")
        if answer[0].lower() == "r":
            return ReviewDecision(action="revise", feedback=answer[1:].strip())
        print("Please answer 'a' or 'r <feedback>'.")


def print_review_request(request: ReviewRequest) -> None:
    """Show the reviewer the draft and the agent's own critique."""
    line = "=" * 72
    print(f"\n{line}\nREVIEW REQUIRED — {request.goal}\n{line}")
    print(f"Subject: {request.draft.subject}")
    if request.draft.intro:
        print(f"\n{request.draft.intro}")
    for index, item in enumerate(request.draft.items, start=1):
        print(f"\n{index}. {item.title}\n   {item.url}\n   {item.summary}")
        if item.why_it_matters:
            print(f"   Why it matters: {item.why_it_matters}")

    print(f"\nAgent self-critique: {request.critique.score}/10, verdict={request.critique.verdict}")
    for issue in request.critique.issues:
        print(f"  - {issue}")
    print(
        f"Revisions used: {request.revision_count}/"
        f"{request.revision_count + request.remaining_revisions}"
    )
    if not request.can_request_changes:
        print("(revision budget exhausted — approving is the only way forward)")


__all__ = ["NewsletterAgentError", "Responder", "print_review_request", "run_newsletter_agent"]
