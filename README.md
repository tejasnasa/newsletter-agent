# Newsletter Agent

An autonomous agent that researches the latest AI agent news and produces a
newsletter. You give it a plain-English goal; it plans, searches live sources,
curates 5-7 articles, writes summaries, **critiques its own draft and revises
it**, optionally pauses for human approval, renders the email, and "sends" it.

One function does the whole job:

```python
from src import run_newsletter_agent

result = run_newsletter_agent(
    "Create a weekly newsletter on the latest AI agent news and send it to our subscribers."
)
print(result.subject, len(result.items))  # -> 'AI agents this week' 6
```

---

## 1. Quick start

```powershell
# 1. Install (Python 3.14 via uv)
uv sync

# 2. Configure — choose your OpenAI-compatible provider, URL and key
Copy-Item .env.example .env
notepad .env                       # set LLM_BASE_URL, LLM_MODEL and LLM_API_KEY
#    Git Bash: cp .env.example .env

# 3. End-to-end, Fully Autonomous
uv run python -m src.cli run --goal "Create a weekly newsletter on the latest AI agent news and send it to our subscribers." --mode autonomous

# 4. Same job, Human-in-the-Loop — it stops and waits at the approval prompt
uv run python -m src.cli run --goal "Create a weekly newsletter on the latest AI agent news and send it to our subscribers." --mode hitl

# 5. Front end
uv run python -m src.cli serve --port 8000       # then open http://127.0.0.1:8000

# 6. Tests — offline suite first, then the live-network suite
uv run pytest -q -m "not live"
uv run pytest -q -m live
```

`--json` prints the whole result (including the HTML) as JSON. `--mode` defaults
to `DEFAULT_MODE` from `.env`.

---

## 2. Architecture

Pure **LangGraph `StateGraph`** — no LangChain chains — with the model called
directly through the OpenAI SDK (`openai` package, custom `base_url`).

```
START
  -> plan            LLM turns the goal into a ResearchPlan (queries, feeds, window, target)
  -> research        every query + feed, merged and deduplicated       [live network]
  -> curate          LLM picks candidate IDs; validated; 5-7 enforced in code; excerpts fetched
  -> write_draft     one LLM call per article + one for subject/intro/outro
  -> critique        self-evaluation against a rubric + the fabrication check
  -> [route_after_critique]
        "revise" ----------> revise -> critique                      <- LOOP 1 (self-critique)
        "human_review"
  -> [route_after_human_review]
        "revise" ----------> revise -> critique -> human_review      <- LOOP 2 (human)
        "render"
  -> render          Markdown + Jinja2 email HTML
  -> send            simulated send: output/*.html, output/*.txt, and stdout
  -> END
```

**Loop 1 (self-critique).** After `critique`, the router returns `"revise"` while
`critique.verdict == "revise"` *and* `revision_count < MAX_REVISIONS` (default 2).
`revise` writes a new draft and control returns to `critique`. The cap is what
stops it looping forever; the count is checked in the router, not trusted to the
model.

**Loop 2 (human-requested changes).** After `human_review`, the router returns
`"revise"` only when the reviewer asked for changes *and* the revision budget is
not exhausted; otherwise `"render"`. The human's words go into `human_feedback`,
which `revise_node` consumes (and then clears), so the rewrite is genuinely
driven by the feedback, then re-critiqued, then re-presented for approval.

Both loops share the same `revise -> critique` edge, so a human-requested change
is held to exactly the same standard as a self-critique revision.

### File layout

```
src/
  config.py     Settings dataclass, .env loading, feed list, limits
  models.py     every pydantic model shared across layers
  state.py      NewsletterState TypedDict + append-only `log` reducer
  prompts.py    one prompt constant group per node
  llm.py        the only module that touches the LLM SDK: chat(), chat_json()
  tools/        sources.py | article.py | summarize.py | render.py | send.py
  nodes.py      plan/research/curate/write_draft/critique/revise/human_review/render/send + 2 routers
  graph.py      build_graph(settings) -> compiled StateGraph
  runner.py     Runner: checkpointer + thread id; start() / resume() / snapshot()
  agent.py      run_newsletter_agent(goal, ...) — the single public entry point
  cli.py        `run` and `serve` subcommands
  templates/    newsletter.html.j2
  web/          FastAPI app + static/index.html, app.js, style.css
tests/          conftest.py + 5 test modules (see section 7)
output/         the rendered newsletter and the simulated email
```

### The seven tools

| Tool | Kind | What it does |
| --- | --- | --- |
| `search_hacker_news` | live network | HN Algolia `search_by_date`, recency-filtered |
| `fetch_rss_feed` | live network | httpx GET -> `feedparser.parse(bytes)` |
| `fetch_article_excerpt` | live network | fetches the article URL, extracts text with stdlib `html.parser` |
| `summarize_article` | LLM | one summary + why-it-matters per selected article |
| `write_newsletter_intro` | LLM | subject, intro, outro |
| `render_newsletter` | offline | Markdown in code + Jinja2 email HTML |
| `send_newsletter` | offline | simulated send to `output/` and stdout |

---

## 3. Fully Autonomous vs Human-in-the-Loop

One `RunMode` enum. The **only** place the mode changes behaviour is
`human_review_node`:

* **`autonomous`** — auto-approves and the run never stops.
* **`hitl`** — calls LangGraph's `interrupt(payload)`. That *checkpoints and
  suspends* the graph: `snapshot.next` is populated, nothing after that node
  runs, and execution continues only when the caller sends
  `graph.invoke(Command(resume={...}), config)`. This is a real suspension, not a
  print-and-continue.

Three ways to choose: `DEFAULT_MODE` in `.env`, `--mode` on the CLI, or the radio
button in the UI.

* **CLI + HITL** prints the draft and the agent's own critique, then blocks on
  `input()` for `approve` / `revise <feedback>`.
* **Web + HITL** uses `Runner.start()` and `Runner.resume()` across two separate
  HTTP requests, so the pause is visible state rather than a blocked socket. The
  page shows an Approve / Request-changes box. When the revision budget is spent
  the Request-changes button is disabled and says why.
* **`run_newsletter_agent(..., responder=...)`** keeps the public API to one
  function: it drives the pause loop for you, calling your `responder` each time
  the graph suspends.

---

## 4. The anti-fabrication guardrail

The agent never invents article data, and that is enforced structurally rather
than by asking the model nicely:

1. **The LLM only ever selects candidate _IDs_.** `curate_node` sends a numbered
   pool (`A1 | headline | source | date | snippet`) and receives
   `selected_ids`. Unknown IDs are dropped and logged.
2. **URLs are copied in code.** `NewsletterItem.url`, `.source` and `.published`
   are copied from the real `Article` object in `summarize_article` and again in
   `rebuild_draft`. No prompt ever asks for a URL, and the revision step returns
   text keyed by `article_id`, so a revision cannot add an article or a link.
3. **The critique step flags untraceable URLs.** `apply_fabrication_check` forces
   `verdict="revise"` if any draft URL is not in `selected`.
4. **Render refuses to proceed.** `render_node` calls
   `assert_no_fabricated_urls`, which raises `FabricationError` unless every item
   URL appears in the fetched pool.

The same principle covers the plan: feed URLs returned by the LLM are intersected
with the allow-list in `config.py`, so the model cannot cause a fetch from an
invented address.

---

## 5. Sources (keyless, verified)

* **Hacker News Algolia API** — `search_by_date`, recency-filtered, no key.
* **RSS feeds**, fetched with `httpx` and parsed with `feedparser`. Every feed
  below was fetched successfully before being listed in `config.py`. Feeds rot,
  so each one is individually try/except'd: a dead feed is skipped and the run
  continues. `pytest -m live` prints a live/dead report for all of them.

  TechCrunch AI · OpenAI News · Google AI Blog · VentureBeat AI · The Verge AI ·
  MarkTechPost · MIT Technology Review · hnrss (AI agent, 10+ points)

To reach the 5-7 item floor on a quiet news day, `research_node` widens the
search with broader fallback queries and a doubled time window, then re-ranks.

---

## 6. Front end

`src/web/app.py` (FastAPI + uvicorn, plain static files, no Node):

| Route | Body | Returns |
| --- | --- | --- |
| `GET /` | – | `index.html` |
| `GET /api/health` | – | `{ok, provider, model, has_api_key, default_mode, max_revisions}` |
| `POST /api/runs` | `{goal, mode}` | `{thread_id, status, log, review?, result?}` |
| `GET /api/runs/{id}` | – | same shape, polled every second for logs and status |
| `POST /api/runs/{id}/resume` | `{action, feedback}` | same shape |

A run executes on a background thread (network + LLM calls take 30-90s), so
`POST /api/runs` returns immediately with `status: "running"` and the page polls
once a second. Deliberately simple — no websockets, no extra dependency, robust
on Windows. The newsletter HTML renders in a sandboxed `<iframe>`
(`sandbox=""`), and all model-provided text is inserted with `textContent`, so
the page cannot execute agent output.

---

## 7. Tests

```powershell
uv run pytest -q -m "not live"    # 43 tests, no credentials, no network
uv run pytest -q -m live          # real HN + real RSS + real article fetch
uv run pytest -q                  # everything (the live LLM test self-skips without a key)
```

| File | What it proves |
| --- | --- |
| `test_sources.py` | **live**: HN Algolia, the RSS feeds, and article extraction return real articles (also reports dead feeds) |
| `test_tools.py` | render + simulated send write the expected files and print the email; HTML is escaped; dedupe/relevance behave |
| `test_critique_loop.py` | both routers; the self-critique loop stops at `MAX_REVISIONS`; **a pause/resume round trip through `interrupt()`/`Command`**; the fabrication guardrail; a revision cannot add articles |
| `test_run_end_to_end.py` | the full graph yields 5-7 items with titles, links and summaries; every URL traces to a fetched article; artifacts are written; the critique genuinely drives a revision; HITL pauses via the public API; graceful degradation; source failure is reported |
| `test_web_api.py` | the real FastAPI app over HTTP: page served, background run, polling, HITL pause/resume across requests, 400 without a key, 409 on a late resume, 404/422 handling |

The offline tests install a `FakeLLM` by monkeypatching `src.llm.chat_json`.
That works because every call site reaches the model through the `src.llm`
module, which is also why the SDK is confined to one file.

`test_critique_loop.py` deliberately asserts the `interrupt`/`Command` round trip:
`langgraph` is on 1.x and these APIs have moved between releases, so a breaking
upgrade fails a test instead of the demo. `langgraph` and `openai` are both
pinned exactly in `pyproject.toml`.

---

## 8. Design notes and known limitations

* **Provider.** Any OpenAI-compatible provider. Set `LLM_BASE_URL`, `LLM_MODEL`
  and `LLM_API_KEY` in `.env`; the defaults point at OpenAI. All SDK contact is
  confined to `src/llm.py`, so changing provider is a `.env` change with no code
  edit — which is also what keeps the project portable across providers.
* **Pure LangGraph, not LangChain.** The assignment permits either. This is the
  graph engine with the provider SDK called directly.
* **In-memory checkpointer.** A paused HITL run lives in process memory: it
  survives across HTTP requests, but restarting the server loses it. Fine for
  this assignment; a SQLite saver is the one-line upgrade
  (`build_checkpointer` in `src/runner.py`).
* **`MAX_REVISIONS` is shared by both loops.** A critique that asks for two
  revisions consumes the budget before the human is asked, and the UI then says
  the budget is exhausted. That is what the plan specifies; raise
  `MAX_REVISIONS` if you want the reviewer to have room after a demanding
  self-critique.
* **Every node has a deterministic fallback.** If the model is unreachable, the
  run still produces a real newsletter from real articles (ranked top-N,
  headline-as-title, feed summary) and the log says exactly which step fell back.
  Article data is never invented to fill a gap — if every source fails, the run
  stops with an error instead.
* **`send_newsletter(simulate=False)`** raises `NotImplementedError`: real
  delivery is out of scope and the stub says so rather than pretending.
* **Cost.** A run is roughly 10-20 LLM calls: excerpts are fetched only for the
  selected items, and never for the whole candidate pool.

---

## 9. Verification status

Run on Windows, Python 3.14.3, `uv 0.11.3`, with a live OpenAI-compatible
provider configured in `.env`:

| Command | Result |
| --- | --- |
| `uv sync` | 59 packages resolved, installed |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 29 files already formatted |
| `uv run pytest -q -m "not live"` | **43 passed** |
| `uv run pytest -q -m live` | **4 passed, 1 skipped** |
| `uv run python -m src.cli run ... --mode autonomous` | full pipeline; 7 items, 2 revisions; wrote `output/newsletter-*.html` and `output/email-*.txt` |
| `uv run python -m src.cli run ... --mode hitl` | printed the review gate, **blocked on input**, resumed on `approve`, rendered and sent |
| `uv run python -m src.cli run ...` (no key) | exit 1, `error: LLM_API_KEY is not set. Copy .env.example to .env ...` |
| `uv run python -m src.cli serve --port 8000` | page, static assets, health, and a **full background run driven over HTTP** through `POST /api/runs` and polled to completion |

Both modes were exercised against a live model rather than a fake. The autonomous
run and the HITL run each produced a 7-item newsletter in which every URL traced
back to an article that was actually fetched.

### The anti-fabrication guardrail, probed directly

Beyond the test suite, the guardrail was attacked on purpose:

| Probe | Result |
| --- | --- |
| Clean draft renders | passes |
| URL differing only by a trailing slash | tolerated — URL normalization working |
| URL absent from the fetched pool | `FabricationError` raised at render |
| Critique sees a fabricated URL | verdict forced to `revise` |
| Duplicate article URLs in the draft | verdict forced to `revise` |
| Clean draft sent through the critique | verdict left alone — no false positives |

### Known behaviour worth reading

`MAX_REVISIONS` binds every run. In the live runs the critique returned
`verdict="revise"` on every pass, so the loop terminated on the cap rather than on
an approval, and the score did not move between passes. The cause is structural
and deliberate: an item whose source article carries no real content cannot be
improved by rewriting, and the revision step is not permitted to add or drop
articles, because that restriction is exactly what makes the guardrail airtight.
Allowing a revision to *drop* an item the critique flags as unsourceable is the
obvious next improvement.

```powershell
uv run pytest -q -m live tests/test_run_end_to_end.py::test_live_end_to_end
```
