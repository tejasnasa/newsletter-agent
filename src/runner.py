"""The Runner: owns the compiled graph, the checkpointer and one thread id.

``start()`` runs until the graph finishes or suspends; ``resume()`` feeds a
:class:`~src.models.ReviewDecision` back in with ``Command(resume=...)``. The
web layer drives these two calls from two separate HTTP requests, so a HITL
pause is real, HTTP-visible state rather than a blocked socket.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from src import models
from src.config import Settings, load_settings
from src.graph import build_graph
from src.models import (
    Critique,
    NewsletterResult,
    ReviewDecision,
    ReviewRequest,
    RunMode,
)
from src.tools.summarize import ArticleSummary, NewsletterIntro

#: The concrete types this project writes into checkpoints. Listing them keeps
#: deserialisation explicit instead of relying on LangGraph's permissive
#: default, and silences the "unregistered type" warnings.
CHECKPOINTED_MODELS: tuple[type, ...] = (
    models.Article,
    models.Critique,
    models.CurationResult,
    models.NewsletterDraft,
    models.NewsletterItem,
    models.NewsletterResult,
    models.RenderedNewsletter,
    models.ResearchPlan,
    models.ReviewDecision,
    models.ReviewRequest,
    models.RevisedDraft,
    models.RevisedItem,
    models.RunMode,
    models.SendReceipt,
    ArticleSummary,
    NewsletterIntro,
)


#: ``idle`` and ``running`` are live states; ``paused``, ``done`` and ``error``
#: are the states the graph itself settles into.
RunStatus = Literal["idle", "running", "paused", "done", "error"]

LogCallback = Callable[[list[str]], None]


@dataclass
class RunSnapshot:
    """A point-in-time view of one run."""

    thread_id: str
    status: RunStatus
    next_node: str | None = None
    review: ReviewRequest | None = None
    result: NewsletterResult | None = None
    log: list[str] = field(default_factory=list)
    error: str | None = None


class Runner:
    """Runs one newsletter job on one LangGraph thread."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        thread_id: str | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        on_log: LogCallback | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.thread_id = thread_id or f"run-{uuid.uuid4().hex[:12]}"
        self.on_log = on_log

        self._lock = threading.Lock()
        self._log: list[str] = []
        self._status: RunStatus = "idle"
        self._error: str | None = None

        # In-memory checkpointing: a paused run does not survive a process
        # restart. Documented in the README as a deliberate limitation.
        self.checkpointer: BaseCheckpointSaver = checkpointer or build_checkpointer()
        self.graph = build_graph(self.settings, checkpointer=self.checkpointer)

    # -- public API ---------------------------------------------------------- #

    def start(self, goal: str, mode: RunMode | str = RunMode.AUTONOMOUS) -> RunSnapshot:
        """Run from the entry point until the graph finishes or suspends."""
        run_mode = RunMode.parse(mode)
        initial_state: dict[str, Any] = {
            "goal": goal,
            "mode": run_mode,
            "revision_count": 0,
            "critique_history": [],
            "human_feedback": "",
            "human_decision": None,
            "log": [f"start: mode={run_mode.value}, goal={goal!r}"],
        }
        return self._execute(initial_state)

    def resume(self, decision: ReviewDecision | dict[str, Any]) -> RunSnapshot:
        """Continue a suspended run with the reviewer's decision."""
        payload = decision.model_dump() if isinstance(decision, ReviewDecision) else decision
        return self._execute(Command(resume=payload))

    def snapshot(self) -> RunSnapshot:
        """The current state of the run."""
        with self._lock:
            status, error, log = self._status, self._error, list(self._log)

        if status == "error":
            return RunSnapshot(self.thread_id, "error", log=log, error=error)
        if status in {"idle", "running"}:
            return RunSnapshot(self.thread_id, status, log=log)
        return self._read_state_snapshot(fallback_log=log)

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def log(self) -> list[str]:
        with self._lock:
            return list(self._log)

    # -- internals ----------------------------------------------------------- #

    def _execute(self, graph_input: Any) -> RunSnapshot:
        with self._lock:
            self._status = "running"
            self._error = None

        try:
            self._stream(graph_input)
        except Exception as exc:  # noqa: BLE001 - surfaced through the snapshot
            with self._lock:
                self._status = "error"
                self._error = f"{type(exc).__name__}: {exc}"
            return self.snapshot()

        finished = self._read_state_snapshot(fallback_log=self.log)
        with self._lock:
            self._status = finished.status
        return finished

    def _stream(self, graph_input: Any) -> None:
        """Walk the graph value-by-value so the log grows live during a run."""
        for values in self.graph.stream(graph_input, self.config, stream_mode="values"):
            lines = list(values.get("log", []))
            with self._lock:
                self._log = lines
            if self.on_log is not None:
                self.on_log(lines)

    def _read_state_snapshot(self, *, fallback_log: list[str]) -> RunSnapshot:
        """Read the checkpoint: either suspended at a node, or finished."""
        state = self.graph.get_state(self.config)
        merged_log = list(state.values.get("log", fallback_log)) or fallback_log
        next_node = state.next[0] if state.next else None

        if next_node:
            return RunSnapshot(
                self.thread_id,
                "paused",
                next_node=next_node,
                review=self._review_from(state),
                log=merged_log,
            )
        return RunSnapshot(
            self.thread_id, "done", result=self._result_from(state.values), log=merged_log
        )

    def _review_from(self, state: Any) -> ReviewRequest | None:
        interrupts = getattr(state.tasks[0], "interrupts", ()) if state.tasks else ()
        if not interrupts:
            return None
        request = ReviewRequest.model_validate(interrupts[0].value)
        return request.model_copy(update={"thread_id": self.thread_id})

    def _result_from(self, values: dict[str, Any]) -> NewsletterResult:
        draft = values.get("draft")
        rendered = values.get("rendered")
        receipt = values.get("receipt")

        return NewsletterResult(
            goal=values.get("goal", ""),
            mode=RunMode.parse(values.get("mode", RunMode.AUTONOMOUS)),
            subject=getattr(draft, "subject", "") or getattr(rendered, "subject", ""),
            items=list(getattr(draft, "items", []) or []),
            markdown=getattr(rendered, "markdown", ""),
            html=getattr(rendered, "html", ""),
            email_body=getattr(receipt, "body", ""),
            html_path=getattr(receipt, "html_path", ""),
            email_path=getattr(receipt, "email_path", ""),
            critiques=list(values.get("critique_history", []) or []),
            revisions=int(values.get("revision_count", 0) or 0),
            log=list(values.get("log", [])),
        )


__all__ = ["Critique", "RunSnapshot", "RunStatus", "Runner", "build_checkpointer"]


def build_checkpointer() -> InMemorySaver:
    """An in-memory saver with our model types explicitly allowed."""
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            (type_.__module__, type_.__name__) for type_ in CHECKPOINTED_MODELS
        ]
    )
    return InMemorySaver(serde=serde)
