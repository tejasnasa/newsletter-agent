"""FastAPI front end: start a run, poll it, approve or request changes.

A run is executed on a background thread (network + LLM calls take 30-90s), so
``POST /api/runs`` returns immediately with ``status: "running"`` and the page
polls ``GET /api/runs/{thread_id}`` once a second for logs and status.

HITL is real here: the graph suspends inside ``interrupt()`` and the pause
survives across two separate HTTP requests.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import Settings, load_settings
from src.models import ReviewDecision, RunMode
from src.runner import Runner, RunSnapshot

STATIC_DIR = Path(__file__).resolve().parent / "static"


def current_settings() -> Settings:
    """Read configuration at request time.

    ``require_key=False`` so the page still loads without a key (the health
    badge says so); starting a run without a key then fails fast and clearly
    instead of 40 seconds into the pipeline.
    """
    return load_settings(require_key=False)


class StartRunRequest(BaseModel):
    goal: str = Field(min_length=3)
    mode: str = "autonomous"


class ResumeRequest(BaseModel):
    action: str = "approve"
    feedback: str = ""


@dataclass
class RunRecord:
    """One run, plus the thread executing it."""

    thread_id: str
    goal: str
    mode: RunMode
    runner: Runner
    status: str = "running"
    snapshot: RunSnapshot | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


REGISTRY: dict[str, RunRecord] = {}
REGISTRY_LOCK = threading.Lock()

app = FastAPI(title="Newsletter Agent", version="0.1.0")


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = current_settings()
    return {
        "ok": True,
        "provider": settings.llm_base_url,
        "model": settings.llm_model,
        "has_api_key": settings.has_api_key,
        "default_mode": settings.default_mode.value,
        "max_revisions": settings.max_revisions,
    }


@app.post("/api/runs")
def start_run(request: StartRunRequest) -> dict[str, Any]:
    settings = current_settings()
    if not settings.has_api_key:
        raise HTTPException(
            status_code=400,
            detail=(
                "LLM_API_KEY is not set. Copy .env.example to .env, set your provider's "
                "API key, and start the run again."
            ),
        )

    mode = _parse_mode(request.mode)
    runner = Runner(settings)
    record = RunRecord(thread_id=runner.thread_id, goal=request.goal, mode=mode, runner=runner)

    with REGISTRY_LOCK:
        REGISTRY[record.thread_id] = record

    _spawn(record, lambda: runner.start(request.goal, mode))
    return payload(record)


@app.get("/api/runs/{thread_id}")
def get_run(thread_id: str) -> dict[str, Any]:
    return payload(_get_record(thread_id))


@app.post("/api/runs/{thread_id}/resume")
def resume_run(thread_id: str, request: ResumeRequest) -> dict[str, Any]:
    record = _get_record(thread_id)

    with record.lock:
        if record.status != "paused":
            raise HTTPException(
                status_code=409,
                detail=f"run is '{record.status}', not paused; nothing to resume",
            )
        record.status = "running"

    decision = ReviewDecision(action=request.action, feedback=request.feedback)
    _spawn(record, lambda: record.runner.resume(decision))
    return payload(record)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def payload(record: RunRecord) -> dict[str, Any]:
    """The JSON shape every run route returns."""
    with record.lock:
        snapshot = record.snapshot
        status = record.status
        error = record.error

    log = record.runner.log or (snapshot.log if snapshot else [])

    body: dict[str, Any] = {
        "thread_id": record.thread_id,
        "goal": record.goal,
        "mode": record.mode.value,
        "status": status,
        "next_node": snapshot.next_node if snapshot else None,
        "log": log,
        "review": None,
        "result": None,
        "error": error or (snapshot.error if snapshot else None),
    }

    if status == "paused" and snapshot and snapshot.review is not None:
        body["review"] = snapshot.review.model_dump(mode="json")

    if status == "done" and snapshot and snapshot.result is not None:
        body["result"] = _trim_result(snapshot.result.model_dump(mode="json"))

    return body


def _trim_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop the raw log (the UI already shows it) to keep polls small."""
    result = dict(result)
    result.pop("log", None)
    return result


def _spawn(record: RunRecord, job) -> None:
    """Run ``job`` on a background thread and record how it ended."""

    def worker() -> None:
        try:
            snapshot = job()
            with record.lock:
                record.snapshot = snapshot
                record.status = snapshot.status
                record.error = snapshot.error
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never crashes the server
            with record.lock:
                record.status = "error"
                record.error = f"{type(exc).__name__}: {exc}"

    threading.Thread(target=worker, daemon=True, name=f"run-{record.thread_id}").start()


def _get_record(thread_id: str) -> RunRecord:
    with REGISTRY_LOCK:
        record = REGISTRY.get(thread_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {thread_id!r}")
    return record


def _parse_mode(value: str) -> RunMode:
    try:
        return RunMode.parse(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
