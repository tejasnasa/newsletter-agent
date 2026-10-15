"""Graph nodes and routers.

Each node is a small function ``(state, settings) -> partial state update``.
``build_graph`` binds ``settings`` with :func:`functools.partial`, which keeps
every node directly testable on its own.

The anti-fabrication guardrail lives in three places:

* prompts only ever ask for candidate **IDs** or text keyed by ``article_id``;
* :func:`apply_fabrication_check` flags any item whose URL is not in
  ``selected`` and forces a revision;
* :func:`assert_no_fabricated_urls` re-checks every URL against the real
  candidate pool immediately before rendering.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Literal

from langgraph.types import interrupt

from src import llm
from src.config import Settings
from src.models import (
    Article,
    Critique,
    CurationResult,
    NewsletterDraft,
    NewsletterItem,
    ResearchPlan,
    ReviewDecision,
    ReviewRequest,
    RevisedDraft,
    RunMode,
)
from src.prompts import (
    CRITIQUE_SYSTEM,
    CRITIQUE_USER,
    CURATE_SYSTEM,
    CURATE_USER,
    PLAN_SYSTEM,
    PLAN_USER,
    REVISE_SYSTEM,
    REVISE_USER,
)
from src.state import NewsletterState
from src.tools import (
    fallback_intro,
    fallback_item,
    fallback_subject,
    fetch_article_excerpt,
    fetch_recent_articles,
    filter_recent,
    keywords_from,
    rank_articles,
    render_newsletter,
    search_hacker_news,
    send_newsletter,
    summarize_article,
    write_newsletter_intro,
)
from src.tools.sources import normalize_url

#: Excerpts shorter than this are treated as unusable and the RSS snippet is
#: used instead (paywalls and JS-only pages produce almost no text).
MIN_EXCERPT_CHARS = 200

#: How many ranked candidates are shown to the curation prompt.
MAX_PROMPT_CANDIDATES = 40

#: Characters of each snippet included in the curation prompt.
PROMPT_SNIPPET_CHARS = 200


class ResearchError(RuntimeError):
    """Raised when every live source failed and there is nothing to write about."""


class FabricationError(RuntimeError):
    """Raised when a draft URL cannot be traced to a real fetched article."""


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #


def plan_node(state: NewsletterState, settings: Settings) -> dict:
    """Turn the English goal into a :class:`ResearchPlan`."""
    goal = state["goal"]
    plan = build_plan(goal, settings)

    return {
        "plan": plan,
        "log": [
            f"plan: topic={plan.topic!r}, {len(plan.hn_queries)} HN queries, "
            f"{len(plan.feed_urls)} feeds, target={plan.target_items} items, "
            f"window={plan.timeframe_days}d"
        ],
    }


def build_plan(goal: str, settings: Settings) -> ResearchPlan:
    """Ask the LLM for a plan; fall back to a safe default on any failure."""
    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {
            "role": "user",
            "content": PLAN_USER.format(
                goal=goal,
                today=date.today().isoformat(),
                timeframe_days=settings.timeframe_days,
                feeds="\n".join(f"- {url}" for url in settings.rss_feed_urls),
            ),
        },
    ]

    try:
        plan = llm.chat_json(messages, ResearchPlan, temperature=0.0)
    except Exception:  # noqa: BLE001 - a bad plan must not stop the run
        return default_plan(goal, settings)

    return sanitize_plan(plan, settings)


def default_plan(goal: str, settings: Settings) -> ResearchPlan:
    """The deterministic plan used when the LLM is unavailable or unusable."""
    queries = list(settings.fallback_hn_queries)
    for extra in ("MCP protocol agent", "AI agents enterprise adoption"):
        if extra not in queries:
            queries.append(extra)

    return ResearchPlan(
        topic=goal.strip()[:80] or "AI agent news",
        hn_queries=queries[:5],
        feed_urls=list(settings.rss_feed_urls),
        timeframe_days=settings.timeframe_days,
        target_items=min(max(settings.min_items, 6), settings.max_items),
        angle="What AI agents shipped this week, and what it means for teams building them.",
    )


def sanitize_plan(plan: ResearchPlan, settings: Settings) -> ResearchPlan:
    """Clamp the plan into ranges we can actually execute.

    Feed URLs are intersected with the configured allow-list, so the model
    cannot cause a fetch from an invented URL.
    """
    known_feeds = [url for url in plan.feed_urls if url in settings.rss_feed_urls]
    queries = _dedupe(plan.hn_queries)[:5] or list(settings.fallback_hn_queries)

    return plan.model_copy(
        update={
            "hn_queries": queries,
            "feed_urls": known_feeds or list(settings.rss_feed_urls),
            "timeframe_days": max(1, min(plan.timeframe_days, 30)),
            "target_items": max(settings.min_items, min(plan.target_items, settings.max_items)),
            "topic": plan.topic.strip() or "AI agent news",
        }
    )


# --------------------------------------------------------------------------- #
# research
# --------------------------------------------------------------------------- #


def research_node(state: NewsletterState, settings: Settings) -> dict:
    """Run every planned query and feed, then merge into ``candidates``."""
    plan = state["plan"]
    goal = state["goal"]
    keywords = keywords_from(goal, plan.topic, plan.angle)

    fetched = fetch_recent_articles(
        plan.hn_queries,
        plan.feed_urls,
        settings=settings,
        max_items=settings.max_candidates * 2,
    )
    log = [
        f"research: {len(fetched)} unique articles from {len(plan.hn_queries)} HN queries "
        f"and {len(plan.feed_urls)} RSS feeds"
    ]

    fresh = filter_recent(fetched, plan.timeframe_days)
    ranked = rank_articles(fresh or fetched, keywords)
    candidates = assign_candidate_ids(ranked[: settings.max_candidates])

    if len(candidates) < settings.min_items:
        candidates, top_up_log = top_up_candidates(candidates, goal, settings)
        log.append(top_up_log)

    if not candidates:
        raise ResearchError(
            "every live source failed or returned nothing; check network access "
            "and the configured feeds before retrying"
        )

    log.append(
        f"research: pool = {len(candidates)} candidates "
        f"(top relevance {candidates[0].relevance:.1f}: {candidates[0].title[:70]!r})"
    )
    return {"candidates": candidates, "log": log}


def assign_candidate_ids(articles: list[Article]) -> list[Article]:
    """Give each candidate a short stable id (``A1``, ``A2``, ...).

    These ids are the only thing the LLM is allowed to choose between.
    """
    return [
        article.model_copy(update={"id": f"A{index}"})
        for index, article in enumerate(articles, start=1)
    ]


def top_up_candidates(
    candidates: list[Article],
    goal: str,
    settings: Settings,
) -> tuple[list[Article], str]:
    """Broaden the search when the pool is too small to fill the newsletter."""
    keywords = keywords_from(goal)
    extra: list[Article] = []

    for query in settings.fallback_hn_queries:
        if len(candidates) + len(extra) >= settings.min_items:
            break
        try:
            extra.extend(
                search_hacker_news(
                    query,
                    hits=settings.hn_hits_per_query,
                    since_days=settings.timeframe_days * 2,
                    settings=settings,
                )
            )
        except Exception:  # noqa: BLE001 - the top-up is best-effort
            continue

    known = {normalize_url(article.url) for article in candidates}
    merged = candidates + [article for article in extra if normalize_url(article.url) not in known]
    ranked = rank_articles(filter_recent(merged, settings.timeframe_days * 2) or merged, keywords)
    return (
        assign_candidate_ids(ranked[: settings.max_candidates]),
        f"research: pool below {settings.min_items}, "
        f"widened the search to {len(ranked)} candidates",
    )


# --------------------------------------------------------------------------- #
# curate
# --------------------------------------------------------------------------- #


def curate_node(state: NewsletterState, settings: Settings) -> dict:
    """Select 5-7 articles by ID, validate them, then fetch their excerpts."""
    pool = state["candidates"]
    plan = state["plan"]

    selected, chosen_ids, note = choose_items(pool, state["goal"], plan, settings)

    excerpts, excerpt_log = fetch_excerpts(selected, settings)
    log = [
        f"curate: LLM chose {len(chosen_ids)} ids, validated {len(selected)} against the real pool "
        f"({note})"
    ]
    log.append(excerpt_log)
    return {"selected": selected, "excerpts": excerpts, "log": log}


def choose_items(
    pool: list[Article],
    goal: str,
    plan: ResearchPlan,
    settings: Settings,
) -> tuple[list[Article], list[str], str]:
    """Pick the final articles. Returns ``(selected, chosen_ids, note)``.

    The deterministic top-up means the item count can never depend on the model
    behaving: if the LLM is unavailable or answers with junk, this still returns
    the best-ranked real candidates.
    """
    by_id = {article.id: article for article in pool}
    # Only the best-ranked candidates reach the prompt: that is the noise filter.
    prompt_pool = pool[:MAX_PROMPT_CANDIDATES]
    target = min(max(settings.min_items, plan.target_items), settings.max_items)

    requested_ids, note = request_selection(prompt_pool, goal, plan, target, settings)

    chosen: list[Article] = []
    unknown: list[str] = []
    for raw in requested_ids:
        key = str(raw).strip().upper()
        article = by_id.get(key)
        if article is not None and article not in chosen:
            chosen.append(article)
        elif article is None:
            unknown.append(key)

    if unknown:
        note = f"{note}; dropped {len(unknown)} unknown id(s) {unknown[:4]}"

    # Deterministic floor and ceiling: never trust the model with the count.
    if len(chosen) < target:
        for article in pool:
            if article not in chosen:
                chosen.append(article)
            if len(chosen) >= target:
                break

    return chosen[: settings.max_items], [article.id for article in chosen], note


def request_selection(
    pool: list[Article],
    goal: str,
    plan: ResearchPlan,
    target: int,
    settings: Settings,
) -> tuple[list[str], str]:
    """Ask the LLM for candidate IDs. Returns ``([], reason)`` on failure."""
    messages = [
        {
            "role": "system",
            "content": CURATE_SYSTEM.format(
                target_items=settings.min_items, max_items=settings.max_items
            ),
        },
        {
            "role": "user",
            "content": CURATE_USER.format(
                goal=goal,
                angle=plan.angle or "latest AI agent news",
                count=len(pool),
                candidates="\n".join(format_candidate(article) for article in pool),
            ),
        },
    ]

    try:
        result = llm.chat_json(messages, CurationResult, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 - fall back to ranked top-N
        return [], f"LLM selection unavailable ({type(exc).__name__}), used ranked top-{target}"

    return result.selected_ids, result.reasoning.strip() or "LLM selection"


def format_candidate(article: Article) -> str:
    """One line of the curation prompt: id, headline, source, date, snippet."""
    snippet = " ".join(article.snippet.split())[:PROMPT_SNIPPET_CHARS]
    return (
        f"{article.id} | {article.title} | {article.source} | "
        f"{article.published or 'undated'} | {snippet or '(no summary)'}"
    )


def fetch_excerpts(articles: list[Article], settings: Settings) -> tuple[dict[str, str], str]:
    """Fetch article text for the selected items only, in parallel."""
    if not articles:
        return {}, "curate: no items selected, no excerpts fetched"

    with ThreadPoolExecutor(max_workers=min(settings.max_workers, len(articles))) as executor:
        results = list(executor.map(lambda article: _excerpt_for(article, settings), articles))

    excerpts = {article.id: text for article, text in results if text}
    log = (
        f"curate: fetched excerpts for {len(excerpts)}/{len(articles)} selected articles "
        f"(the rest fall back to their feed summary)"
    )
    return excerpts, log


def _excerpt_for(article: Article, settings: Settings) -> tuple[Article, str]:
    """Best-effort excerpt for one article; never raises."""
    try:
        text = fetch_article_excerpt(
            article.url,
            max_chars=settings.excerpt_max_chars,
            timeout=settings.request_timeout,
        )
    except Exception:  # noqa: BLE001 - one bad page must not stop the run
        return article, ""
    return article, text if len(text) >= MIN_EXCERPT_CHARS else ""


# --------------------------------------------------------------------------- #
# write_draft
# --------------------------------------------------------------------------- #


def write_draft_node(state: NewsletterState, settings: Settings) -> dict:
    """One LLM call per article, plus one for subject/intro/outro."""
    articles = state["selected"]
    excerpts = state.get("excerpts", {})
    items: list[NewsletterItem] = []
    failures = 0

    for article in articles:
        excerpt = excerpts.get(article.id, "")
        try:
            items.append(
                summarize_article(article, goal=state["goal"], excerpt=excerpt, settings=settings)
            )
        except Exception as exc:  # noqa: BLE001 - degrade to a deterministic item
            failures += 1
            items.append(fallback_item(article, excerpt, type(exc).__name__))

    subject, intro, outro, intro_note = build_intro(items, state["goal"], settings)
    draft = NewsletterDraft(
        subject=subject,
        intro=intro,
        items=items,
        outro=outro,
        revision=state.get("revision_count", 0),
    )

    log = [f"write_draft: wrote {len(items)} items ({failures} via deterministic fallback)"]
    if intro_note:
        log.append(f"write_draft: {intro_note}")
    return {"draft": draft, "log": log}


def build_intro(
    items: list[NewsletterItem],
    goal: str,
    settings: Settings,
) -> tuple[str, str, str, str]:
    """Return ``(subject, intro, outro, note)``; ``note`` is empty on success."""
    try:
        subject, intro, outro = write_newsletter_intro(items, goal=goal, settings=settings)
        return subject, intro, outro, ""
    except Exception as exc:  # noqa: BLE001 - deterministic subject/intro/outro
        return (
            fallback_subject(goal),
            fallback_intro(items),
            "That's all for this issue — see you next week.",
            f"LLM intro unavailable ({type(exc).__name__}), used deterministic fallback copy",
        )


# --------------------------------------------------------------------------- #
# critique + revise (the self-improvement loop)
# --------------------------------------------------------------------------- #


def critique_node(state: NewsletterState, settings: Settings) -> dict:
    """Self-evaluate the draft, then apply the code-level fabrication check."""
    draft = state["draft"]
    critique = apply_fabrication_check(
        evaluate_draft(draft, state["goal"], settings), draft, state["selected"]
    )

    history = [*state.get("critique_history", []), critique]
    log = [
        f"critique: score {critique.score}/10, verdict={critique.verdict}, "
        f"issues={len(critique.issues)}"
    ]
    if critique.fabricated_urls:
        log.append(
            f"critique: GUARDRAIL tripped — {len(critique.fabricated_urls)} untraceable URL(s)"
        )

    return {"critique": critique, "critique_history": history, "log": log}


def evaluate_draft(draft: NewsletterDraft, goal: str, settings: Settings) -> Critique:
    """Ask the model to grade its own draft against the rubric."""
    messages = [
        {"role": "system", "content": CRITIQUE_SYSTEM},
        {"role": "user", "content": format_critique_prompt(draft, goal)},
    ]
    try:
        return llm.chat_json(messages, Critique, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 - never block the run on the critique call
        return Critique(
            score=5,
            issues=[f"self-critique unavailable: {type(exc).__name__}"],
            verdict="approve",
            notes="Deterministic checks only; the critique model could not be reached.",
        )


def format_critique_prompt(draft: NewsletterDraft, goal: str) -> str:
    items = "\n".join(
        f"{index}. {item.title} — {item.summary} (why it matters: {item.why_it_matters or 'n/a'})"
        for index, item in enumerate(draft.items, start=1)
    )
    return CRITIQUE_USER.format(
        goal=goal,
        subject=draft.subject,
        intro=draft.intro,
        items=items or "(no items)",
        outro=draft.outro,
    )


def apply_fabrication_check(
    critique: Critique, draft: NewsletterDraft, selected: list[Article]
) -> Critique:
    """Force a revision when a draft URL is not traceable to a real article.

    With the guardrail intact this never fires — every URL is copied in code
    from an :class:`Article`. It exists so that a future change which lets the
    model write a URL fails loudly instead of shipping a dead link.
    """
    fabricated = find_fabricated_urls(draft, selected)
    duplicate_urls = find_duplicate_urls(draft)

    if not fabricated and not duplicate_urls:
        return critique

    issues = list(critique.issues)
    if fabricated:
        issues.insert(0, f"{len(fabricated)} item URL(s) cannot be traced to a fetched article")
    if duplicate_urls:
        issues.append(f"{len(duplicate_urls)} duplicate article URL(s) in the draft")

    return critique.model_copy(
        update={
            "verdict": "revise",
            "issues": issues,
            "fabricated_urls": fabricated,
            "notes": (critique.notes + " Guardrail: revision forced.").strip(),
        }
    )


def find_fabricated_urls(draft: NewsletterDraft, selected: list[Article]) -> list[str]:
    """Return draft URLs that do not appear in the real selected articles."""
    allowed = {normalize_url(article.url) for article in selected}
    return [item.url for item in draft.items if normalize_url(item.url) not in allowed]


def find_duplicate_urls(draft: NewsletterDraft) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in draft.items:
        key = normalize_url(item.url)
        if key in seen:
            duplicates.append(item.url)
        seen.add(key)
    return duplicates


def revise_node(state: NewsletterState, settings: Settings) -> dict:
    """Rewrite the draft using the critique and any human feedback."""
    selected = state["selected"]
    previous = state["draft"]
    critique = state["critique"]
    human_feedback = (state.get("human_feedback") or "").strip()
    revision = state.get("revision_count", 0) + 1

    revised, note = request_revision(
        previous,
        critique,
        human_feedback,
        selected,
        state.get("excerpts", {}),
        state["goal"],
        settings,
    )
    draft = rebuild_draft(revised, selected, previous, revision)

    log = [
        f"revise: revision {revision} ({note})"
        + (", driven by human feedback" if human_feedback else ", driven by self-critique")
    ]
    return {
        "draft": draft,
        "revision_count": revision,
        "human_feedback": "",
        "human_decision": None,
        "log": log,
    }


def request_revision(
    draft: NewsletterDraft,
    critique: Critique,
    human_feedback: str,
    selected: list[Article],
    excerpts: dict[str, str],
    goal: str,
    settings: Settings,
) -> tuple[RevisedDraft, str]:
    """Ask the LLM for a rewritten draft. Returns ``(revised, note)``."""
    messages = [
        {"role": "system", "content": REVISE_SYSTEM},
        {
            "role": "user",
            "content": format_revise_prompt(
                draft, critique, human_feedback, selected, excerpts, goal
            ),
        },
    ]

    try:
        revised = llm.chat_json(messages, RevisedDraft, temperature=settings.llm_temperature)
    except Exception as exc:  # noqa: BLE001 - keep the previous text instead of failing
        return (
            RevisedDraft(),
            f"LLM revision unavailable ({type(exc).__name__}), kept previous text",
        )

    return revised, "rewritten by the model"


def format_revise_prompt(
    draft: NewsletterDraft,
    critique: Critique,
    human_feedback: str,
    selected: list[Article],
    excerpts: dict[str, str],
    goal: str,
) -> str:
    items = "\n".join(
        f"{item.article_id}: {item.title} — {item.summary} "
        f"(why it matters: {item.why_it_matters or 'n/a'})"
        for item in draft.items
    )
    excerpt_block = "\n\n".join(
        f"[{article.id}] {article.title}\n{excerpts.get(article.id, article.snippet)[:1200]}"
        for article in selected
    )
    return REVISE_USER.format(
        goal=goal,
        score=critique.score,
        issues="\n".join(f"- {issue}" for issue in critique.issues) or "- none",
        notes=critique.notes or "n/a",
        human_feedback=human_feedback or "(none — this revision comes from the self-critique)",
        subject=draft.subject,
        intro=draft.intro,
        outro=draft.outro,
        items=items or "(no items)",
        article_ids=", ".join(article.id for article in selected),
        excerpts=excerpt_block or "(no excerpts available)",
    )


def rebuild_draft(
    revised: RevisedDraft,
    selected: list[Article],
    previous: NewsletterDraft,
    revision: int,
) -> NewsletterDraft:
    """Re-attach model-written text to the real articles.

    The item set is exactly ``selected``: ids the model invented are dropped,
    and articles it forgot keep their previous text. URLs, sources and dates are
    copied from the :class:`Article`, never from the model.
    """
    previous_by_id = {item.article_id: item for item in previous.items}
    valid_ids = {article.id for article in selected}
    revised_by_id = {}
    for item in revised.items:
        key = item.article_id.strip().upper()
        if key in valid_ids:
            revised_by_id[key] = item

    items: list[NewsletterItem] = []
    for article in selected:
        fresh = revised_by_id.get(article.id)
        old = previous_by_id.get(article.id)
        items.append(
            NewsletterItem(
                title=(fresh.title.strip() if fresh else "")
                or (old.title if old else article.title),
                url=article.url,
                source=article.source,
                published=article.published,
                summary=(fresh.summary.strip() if fresh else "")
                or (old.summary if old else article.title),
                why_it_matters=(fresh.why_it_matters.strip() if fresh else "")
                or (old.why_it_matters if old else ""),
                article_id=article.id,
            )
        )

    return NewsletterDraft(
        subject=(revised.subject or "").strip() or previous.subject,
        intro=(revised.intro or "").strip() or previous.intro,
        items=items,
        outro=(revised.outro or "").strip() or previous.outro,
        revision=revision,
    )


# --------------------------------------------------------------------------- #
# human_review (the HITL gate)
# --------------------------------------------------------------------------- #


def human_review_node(state: NewsletterState, settings: Settings) -> dict:
    """Autonomous: auto-approve. HITL: ``interrupt()`` and genuinely suspend.

    Nothing before ``interrupt`` may have side effects: LangGraph re-runs this
    node from the top when the run is resumed.
    """
    if RunMode.parse(state["mode"]) is RunMode.AUTONOMOUS:
        return {
            "human_decision": ReviewDecision(action="approve", feedback=""),
            "log": ["human_review: auto-approved (autonomous mode)"],
        }

    request = build_review_request(state, settings)
    answer = interrupt(request.model_dump(mode="json"))
    decision = ReviewDecision.model_validate(answer or {})

    log = [
        f"human_review: reviewer chose '{decision.action}'"
        + (" with feedback" if decision.feedback else "")
    ]
    return {
        "human_decision": decision,
        "human_feedback": decision.feedback if decision.action == "revise" else "",
        "log": log,
    }


def build_review_request(state: NewsletterState, settings: Settings) -> ReviewRequest:
    used = state.get("revision_count", 0)
    remaining = max(0, settings.max_revisions - used)
    return ReviewRequest(
        thread_id="",
        goal=state["goal"],
        mode=RunMode.parse(state["mode"]),
        draft=state["draft"],
        critique=state["critique"],
        item_count=len(state["draft"].items),
        revision_count=used,
        remaining_revisions=remaining,
        can_request_changes=remaining > 0,
    )


# --------------------------------------------------------------------------- #
# render + send
# --------------------------------------------------------------------------- #


def render_node(state: NewsletterState, settings: Settings) -> dict:
    """Assert every URL traces to a real article, then render Markdown + HTML."""
    draft = state["draft"]
    assert_no_fabricated_urls(draft, state["selected"])

    rendered = render_newsletter(draft, templates_dir=settings.templates_dir)
    return {
        "rendered": rendered,
        "log": [
            f"render: {len(draft.items)} items, {len(rendered.markdown)} chars markdown, "
            f"{len(rendered.html)} chars html (all URLs verified against the pool)"
        ],
    }


def assert_no_fabricated_urls(draft: NewsletterDraft, selected: list[Article]) -> None:
    """Fail loudly if the draft contains a URL that was never fetched."""
    fabricated = find_fabricated_urls(draft, selected)
    if fabricated:
        raise FabricationError(
            "refusing to render: these URLs are not present in the fetched candidate pool: "
            + ", ".join(fabricated[:5])
        )


def send_node(state: NewsletterState, settings: Settings) -> dict:
    """Simulated send: write the artifacts and print the email."""
    rendered = state["rendered"]
    receipt = send_newsletter(
        rendered,
        to=list(settings.recipients),
        out_dir=settings.output_dir,
    )
    return {
        "receipt": receipt,
        "log": [
            f"send: simulated send complete, wrote {receipt.html_path} and {receipt.email_path}"
        ],
    }


# --------------------------------------------------------------------------- #
# routers
# --------------------------------------------------------------------------- #


def route_after_critique(
    state: NewsletterState, settings: Settings
) -> Literal["revise", "human_review"]:
    """Self-critique loop: revise while the verdict says so and the cap allows."""
    critique = state.get("critique")
    if critique is None:
        return "human_review"
    if critique.verdict == "revise" and revisions_used(state) < settings.max_revisions:
        return "revise"
    return "human_review"


def route_after_human_review(
    state: NewsletterState, settings: Settings
) -> Literal["revise", "render"]:
    """Human loop: only a reviewer asking for changes can go back to ``revise``."""
    decision = state.get("human_decision")
    if (
        decision is not None
        and decision.action == "revise"
        and revisions_used(state) < settings.max_revisions
    ):
        return "revise"
    return "render"


def revisions_used(state: NewsletterState) -> int:
    return int(state.get("revision_count", 0) or 0)


def _dedupe(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen
