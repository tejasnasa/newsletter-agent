"""Rendering a draft into Markdown and email HTML. No network access."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import TEMPLATES_DIR
from src.models import NewsletterDraft, RenderedNewsletter

TEMPLATE_NAME = "newsletter.html.j2"

_environment: Environment | None = None
_environment_dir: Path | None = None


def render_newsletter(
    draft: NewsletterDraft,
    *,
    templates_dir: Path | None = None,
) -> RenderedNewsletter:
    """Build the Markdown body and the HTML email for ``draft``."""
    return RenderedNewsletter(
        subject=draft.subject,
        markdown=to_markdown(draft),
        html=_render_html(draft, templates_dir or TEMPLATES_DIR),
    )


def to_markdown(draft: NewsletterDraft) -> str:
    """Plain-text/Markdown body, assembled in code (no templating)."""
    parts: list[str] = [f"# {draft.subject}", ""]
    if draft.intro:
        parts += [draft.intro, ""]

    for index, item in enumerate(draft.items, start=1):
        parts.append(f"## {index}. [{item.title}]({item.url})")
        meta = item.source if not item.published else f"{item.source} · {item.published}"
        parts += [f"*{meta}*", "", item.summary, ""]
        if item.why_it_matters:
            parts += [f"**Why it matters:** {item.why_it_matters}", ""]

    if draft.outro:
        parts += [draft.outro, ""]

    return "\n".join(parts).strip() + "\n"


def _render_html(draft: NewsletterDraft, templates_dir: Path) -> str:
    template = _get_environment(templates_dir).get_template(TEMPLATE_NAME)
    return template.render(
        subject=draft.subject, intro=draft.intro, items=draft.items, outro=draft.outro
    )


def _get_environment(templates_dir: Path) -> Environment:
    """Cache one Jinja environment per template directory."""
    global _environment, _environment_dir

    resolved = Path(templates_dir).resolve()
    if _environment is None or _environment_dir != resolved:
        _environment = Environment(
            loader=FileSystemLoader(str(resolved)),
            autoescape=select_autoescape(default_for_string=True, default=True),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        _environment_dir = resolved
    return _environment
