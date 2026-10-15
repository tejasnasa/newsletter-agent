"""Offline tests for the front end's HTTP surface.

Added beyond the plan's list because acceptance criterion 6 is "the front end
starts and can run a job against a goal typed into it", which deserves a real
test. Uses FastAPI's TestClient against the actual app: the run executes on the
app's background thread, exactly as it does in the browser.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from src import nodes
from src.web import app as web_app

GOAL = "Create a weekly newsletter on the latest AI agent news."


@pytest.fixture
def client(monkeypatch, tmp_path, fake_llm, fake_sources) -> TestClient:
    """The real app, with a fake LLM, fake sources, and a key in the environment."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    return TestClient(web_app.app)


def wait_for(client: TestClient, thread_id: str, wanted: set[str], timeout: float = 15.0) -> dict:
    """Poll like the browser does, until the run reaches a state we want."""
    deadline = time.monotonic() + timeout
    payload: dict = {}

    while time.monotonic() < deadline:
        payload = client.get(f"/api/runs/{thread_id}").json()
        if payload["status"] in wanted:
            return payload
        time.sleep(0.02)

    raise AssertionError(f"run never reached {wanted}; last payload was {payload}")


def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Newsletter Agent" in response.text
    assert "Human-in-the-Loop" in response.text


def test_health_reports_the_provider_and_key_state(client: TestClient) -> None:
    body = client.get("/api/health").json()

    assert body["ok"] is True
    assert body["model"]
    assert body["provider"]
    assert body["has_api_key"] is True


def test_starting_a_run_without_a_key_fails_fast(
    monkeypatch, tmp_path, fake_llm, fake_sources
) -> None:
    # Neutralise .env so the test cannot pick up a developer's real key.
    monkeypatch.setattr("src.config.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    client = TestClient(web_app.app)

    response = client.post("/api/runs", json={"goal": GOAL, "mode": "autonomous"})

    assert response.status_code == 400
    assert "LLM_API_KEY" in response.json()["detail"]


def test_a_full_autonomous_run_through_the_api(client: TestClient) -> None:
    started = client.post("/api/runs", json={"goal": GOAL, "mode": "autonomous"})
    assert started.status_code == 200
    assert started.json()["status"] in {"running", "done"}

    finished = wait_for(client, started.json()["thread_id"], {"done", "error"})

    assert finished["status"] == "done", finished["error"]
    result = finished["result"]
    assert 5 <= len(result["items"]) <= 7
    assert result["subject"]
    assert result["html"] and result["markdown"]
    assert result["html_path"] and result["email_path"]
    assert any("critique" in line for line in finished["log"])


def test_a_hitl_run_pauses_then_resumes_through_the_api(client: TestClient) -> None:
    started = client.post("/api/runs", json={"goal": GOAL, "mode": "hitl"}).json()

    paused = wait_for(client, started["thread_id"], {"paused", "error", "done"})

    assert paused["status"] == "paused", paused["error"]
    assert paused["next_node"] == "human_review"
    assert paused["review"]["item_count"] >= 5
    assert paused["review"]["thread_id"] == started["thread_id"]
    assert paused["result"] is None

    resumed = client.post(
        f"/api/runs/{started['thread_id']}/resume",
        json={"action": "revise", "feedback": "Lead with the enterprise angle."},
    )
    assert resumed.status_code == 200

    again = wait_for(client, started["thread_id"], {"paused", "done", "error"})
    assert again["status"] == "paused", again["error"]
    assert again["review"]["revision_count"] == 1

    approved = client.post(
        f"/api/runs/{started['thread_id']}/resume", json={"action": "approve", "feedback": ""}
    ).json()
    assert approved["status"] in {"running", "paused", "done"}

    finished = wait_for(client, started["thread_id"], {"done", "error"})
    assert finished["status"] == "done", finished["error"]
    assert finished["result"]["revisions"] == 1


def test_resuming_a_finished_run_is_rejected(client: TestClient) -> None:
    started = client.post("/api/runs", json={"goal": GOAL, "mode": "hitl"}).json()
    wait_for(client, started["thread_id"], {"paused"})

    # Approve twice: the run finishes on the first resume.
    client.post(f"/api/runs/{started['thread_id']}/resume", json={"action": "approve"})
    wait_for(client, started["thread_id"], {"done", "error"})

    second = client.post(f"/api/runs/{started['thread_id']}/resume", json={"action": "approve"})

    assert second.status_code == 409
    assert "paused" in second.json()["detail"]


def test_unknown_run_is_a_404(client: TestClient) -> None:
    assert client.get("/api/runs/run-does-not-exist").status_code == 404


def test_invalid_mode_is_rejected(client: TestClient) -> None:
    response = client.post("/api/runs", json={"goal": GOAL, "mode": "telepathic"})

    assert response.status_code == 422


def test_a_failing_run_surfaces_the_error_instead_of_hanging(
    client: TestClient, monkeypatch
) -> None:
    def dead(queries, feed_urls, *, settings, max_items=60):
        return []

    monkeypatch.setattr(nodes, "fetch_recent_articles", dead)
    monkeypatch.setattr(nodes, "search_hacker_news", lambda *args, **kwargs: [])

    started = client.post("/api/runs", json={"goal": GOAL, "mode": "autonomous"}).json()
    failed = wait_for(client, started["thread_id"], {"error", "done"})

    assert failed["status"] == "error"
    assert "source" in failed["error"].lower()
