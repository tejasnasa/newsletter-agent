"""Command line interface: ``run`` and ``serve``.

uv run python -m src.cli run --goal "..." --mode autonomous
uv run python -m src.cli run --goal "..." --mode hitl
uv run python -m src.cli serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys

from src.agent import NewsletterAgentError, run_newsletter_agent
from src.config import ConfigError, load_settings
from src.models import NewsletterResult, RunMode

DEFAULT_GOAL = (
    "Create a weekly newsletter on the latest AI agent news and send it to our subscribers."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newsletter-agent",
        description="Research AI agent news and produce a newsletter.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one newsletter job end to end")
    run_parser.add_argument("--goal", default=DEFAULT_GOAL, help="plain-English goal")
    run_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RunMode],
        default=None,
        help="autonomous (default) or hitl; falls back to DEFAULT_MODE in .env",
    )
    run_parser.add_argument("--json", action="store_true", help="print the result as JSON")
    run_parser.set_defaults(handler=run_command)

    serve_parser = subparsers.add_parser("serve", help="start the web front end")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    serve_parser.set_defaults(handler=serve_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ConfigError, NewsletterAgentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 130


def run_command(args: argparse.Namespace) -> int:
    settings = load_settings()
    mode = RunMode.parse(args.mode) if args.mode else settings.default_mode

    print(f"goal: {args.goal}")
    print(f"mode: {mode.value}   model: {settings.llm_model}   provider: {settings.llm_base_url}")
    print("-" * 72)

    seen = 0

    def print_new_log_lines(lines: list[str]) -> None:
        nonlocal seen
        for line in lines[seen:]:
            print(f"  {line}")
        seen = len(lines)

    result = run_newsletter_agent(
        args.goal,
        mode=mode,
        settings=settings,
        on_log=print_new_log_lines,
    )

    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
    else:
        print_run_summary(result)
    return 0


def print_run_summary(result: NewsletterResult) -> None:
    """Print the critique score, the revisions, and the artifacts written."""
    print("-" * 72)
    print(f"subject: {result.subject}")
    print(f"items:   {len(result.items)}")
    print(f"revisions: {result.revisions}")

    if result.critiques:
        print("critique history:")
        for index, critique in enumerate(result.critiques, start=1):
            print(
                f"  {index}. score {critique.score}/10 "
                f"verdict={critique.verdict} issues={len(critique.issues)}"
            )
            for issue in critique.issues:
                print(f"       - {issue}")

    print("files written:")
    print(f"  {result.html_path}")
    print(f"  {result.email_path}")


def serve_command(args: argparse.Namespace) -> int:
    import uvicorn  # imported lazily so `run` stays fast

    print(f"starting front end on http://{args.host}:{args.port}")
    uvicorn.run("src.web.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
