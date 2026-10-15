"""Every prompt the agent sends, one constant group per node.

Kept out of ``nodes.py`` so the prompts can be read and tuned without wading
through control flow.

Guardrail note: the curation and revision prompts only ever ask for candidate
*IDs* or text keyed by ``article_id``. No prompt asks the model for a URL.
"""

from __future__ import annotations

PLAN_SYSTEM = """You are the planning step of a newsletter research agent.
Break the user's goal into a concrete research plan for finding the LATEST news.
Reply with a single JSON object using exactly these keys:
  "topic": short topic label,
  "hn_queries": 3-5 short search queries for Hacker News, most specific first,
  "feed_urls": the subset of the allowed RSS feeds most relevant to the goal,
  "timeframe_days": whole number of days back to search (1-30),
  "target_items": how many articles the newsletter should contain (5-7),
  "angle": one sentence describing the editorial angle.
Choose search queries that return recent, on-topic stories rather than generic ones."""

PLAN_USER = """Goal:
{goal}

The current date is {today}. Prefer coverage from the last {timeframe_days} days.

Allowed RSS feeds (choose from these only, do not invent feed URLs):
{feeds}
"""

CURATE_SYSTEM = """You are the curation step of a newsletter agent.
You are given a numbered list of real candidate articles. Choose the best
{target_items} to {max_items} for the reader.

Rules:
- Select by article id only, using the exact ids shown (for example "A3").
- Never invent an id, a title, or a URL. Only ids from the list are valid.
- Prefer recent, substantive, on-topic items about AI agents and agentic systems.
- Drop duplicates, press-release noise, and thinly-sourced posts.

Reply with a single JSON object:
  "selected_ids": array of ids, best first,
  "reasoning": one short sentence explaining the selection."""

CURATE_USER = """Goal:
{goal}

Editorial angle: {angle}

Candidates ({count} total):
{candidates}
"""

SUMMARIZE_SYSTEM = """You write one newsletter entry about one real article.
You are given the article's real headline and an excerpt of its text.
Reply with a single JSON object with exactly these keys:
  "title": the article headline, cleaned up (no trailing site name, no clickbait),
  "summary": 2-3 sentences describing what the article actually says,
  "why_it_matters": 1-2 sentences on why this matters to people building AI agents.
Rules:
- Use only facts present in the excerpt or headline. If the excerpt is thin, say
  less rather than guessing.
- Do not include URLs, links, or markdown in any field.
- Do not mention the excerpt itself."""

SUMMARIZE_USER = """Newsletter goal:
{goal}

Article headline: {title}
Source: {source}
Published: {published}
Excerpt:
\"\"\"
{excerpt}
\"\"\"
"""

INTRO_SYSTEM = """You are the editor of a short AI-agents newsletter.
Write the email subject line, a short intro and a short outro.

Reply with a single JSON object:
  "subject": a specific, non-hypey subject line under 80 characters,
  "intro": 2-3 sentences setting up this issue,
  "outro": 1-2 sentences signing off.
Rules:
- Ground everything in the items given; never invent facts, names, or numbers.
- Plain text only: no URLs, no markdown, no placeholders like [Name]."""

INTRO_USER = """Goal:
{goal}

Items in this issue:
{items}
"""

CRITIQUE_SYSTEM = """You are a demanding but fair editor reviewing your own newsletter draft.
Grade it against this rubric: relevance to the goal, factual grounding,
specificity (no vague filler), and readability.

Reply with a single JSON object:
  "score": integer 0-10,
  "issues": array of concrete problems (empty array if none),
  "verdict": "approve" if the draft is publishable as-is, otherwise "revise",
  "notes": one short sentence summarising your judgement.
Rules:
- Judge only the draft text you are given.
- Use "revise" when any issue would be visible to a reader."""

CRITIQUE_USER = """Goal:
{goal}

Subject: {subject}

Intro: {intro}

Items:
{items}

Outro: {outro}
"""

REVISE_SYSTEM = """You are the revision step of a newsletter agent.
Rewrite the draft so that every listed issue is fixed.

Reply with a single JSON object:
  "subject": the revised subject,
  "intro": the revised intro,
  "outro": the revised outro,
  "items": an array of objects, one per article id given below, each with
           "article_id" (copied exactly), "title", "summary", "why_it_matters"."""

REVISE_USER = """Goal:
{goal}

Editor's critique (fix all of this):
score: {score}
issues:
{issues}
notes: {notes}

Human feedback to honour:
{human_feedback}

Current draft:
Subject: {subject}
Intro: {intro}
Outro: {outro}
Items:
{items}

You must return exactly these article ids, using them verbatim:
{article_ids}
Rules:
- Never invent an article id, headline, or URL. Keep the same set of articles.
- Do not add facts that are not already in the draft or the excerpt below.
- Plain text only: no URLs, no markdown.

Excerpts for reference:
{excerpts}
"""
