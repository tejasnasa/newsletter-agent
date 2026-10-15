"""Configuration: ``.env`` loading, run limits, and the keyless source list.

Credentials are never hardcoded. They are read from the environment (populated
from ``.env`` via ``python-dotenv``) and a clear error is raised *early* when
the key is missing, rather than 40 seconds into a run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.models import RunMode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# Keyless sources only. Every feed below was fetched successfully before being
# added here (see the "Sources" section of the README).
RSS_FEEDS: tuple[str, ...] = (
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://openai.com/news/rss.xml",
    "https://blog.google/technology/ai/rss/",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://www.marktechpost.com/feed/",
    "https://www.technologyreview.com/feed/",
    "https://hnrss.org/newest?q=AI+agent&points=10",
)

# Queries used when the planner's own queries fail, or when the pool is too
# small to reach the 5-7 item floor.
FALLBACK_HN_QUERIES: tuple[str, ...] = (
    "AI agent",
    "agentic AI",
    "LLM agent framework",
)


class ConfigError(RuntimeError):
    """Raised when the environment is missing something we cannot invent."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything tunable in one immutable object."""

    llm_api_key: str = ""
    llm_base_url: str = DEFAULT_BASE_URL
    llm_model: str = DEFAULT_MODEL
    llm_temperature: float = 0.2

    default_mode: RunMode = RunMode.AUTONOMOUS
    max_revisions: int = 2

    min_items: int = 5
    max_items: int = 7
    timeframe_days: int = 14

    hn_hits_per_query: int = 20
    max_candidates: int = 60
    min_relevance: float = 0.0

    excerpt_max_chars: int = 4000
    request_timeout: float = 15.0
    max_workers: int = 8

    recipients: tuple[str, ...] = ("subscribers@example.com",)
    output_dir: Path = DEFAULT_OUTPUT_DIR
    templates_dir: Path = TEMPLATES_DIR
    rss_feed_urls: tuple[str, ...] = RSS_FEEDS
    fallback_hn_queries: tuple[str, ...] = FALLBACK_HN_QUERIES

    @property
    def has_api_key(self) -> bool:
        return bool(self.llm_api_key.strip())


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_settings(
    *,
    env_file: str | Path | None = ".env",
    require_key: bool = True,
) -> Settings:
    """Load ``.env`` into the environment and build a :class:`Settings`.

    Args:
        env_file: path to the dotenv file; missing files are ignored.
        require_key: raise :class:`ConfigError` when ``LLM_API_KEY`` is absent.
            Tests and the offline test-suite pass ``False``.
    """
    if env_file is not None:
        load_dotenv(env_file, override=False)

    api_key = _env_str("LLM_API_KEY", "")
    if require_key and not api_key:
        raise ConfigError(
            "LLM_API_KEY is not set. Copy .env.example to .env and set your "
            "provider's API key (see README.md, step 2)."
        )

    settings = Settings(
        llm_api_key=api_key,
        llm_base_url=_env_str("LLM_BASE_URL", DEFAULT_BASE_URL),
        llm_model=_env_str("LLM_MODEL", DEFAULT_MODEL),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        default_mode=RunMode.parse(_env_str("DEFAULT_MODE", RunMode.AUTONOMOUS.value)),
        max_revisions=_env_int("MAX_REVISIONS", 2),
        min_items=_env_int("MIN_ITEMS", 5),
        max_items=_env_int("MAX_ITEMS", 7),
        timeframe_days=_env_int("TIMEFRAME_DAYS", 14),
        hn_hits_per_query=_env_int("HN_HITS_PER_QUERY", 20),
        max_candidates=_env_int("MAX_CANDIDATES", 60),
        recipients=_env_list("RECIPIENTS", ("subscribers@example.com",)),
        output_dir=Path(_env_str("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
        rss_feed_urls=_env_list("RSS_FEEDS", RSS_FEEDS),
        fallback_hn_queries=_env_list("FALLBACK_HN_QUERIES", FALLBACK_HN_QUERIES),
    )
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    if settings.min_items < 1:
        raise ConfigError("MIN_ITEMS must be at least 1")
    if settings.max_items < settings.min_items:
        raise ConfigError("MAX_ITEMS must be greater than or equal to MIN_ITEMS")
    if settings.max_revisions < 0:
        raise ConfigError("MAX_REVISIONS cannot be negative")
