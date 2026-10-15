"""Fetching and cleaning the text of a single article."""

from __future__ import annotations

import re
from html.parser import HTMLParser

import httpx

USER_AGENT = "newsletter-agent/0.1 (educational assignment; keyless news research)"

_SKIP_TAGS = {"script", "style", "noscript", "head", "svg", "template", "form"}
_BLOCK_TAGS = {
    "p",
    "div",
    "br",
    "li",
    "ul",
    "ol",
    "section",
    "article",
    "header",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "tr",
    "table",
    "main",
    "nav",
}
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$")


class _TextExtractor(HTMLParser):
    """Collect human-readable text from HTML, skipping scripts and styles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def html_to_text(html: str) -> str:
    """Convert HTML to plain text using only the standard library."""
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()

    text = extractor.text().replace("\r\n", "\n").replace("\r", "\n")
    lines = [_TRAILING_SPACE_RE.sub("", line).strip() for line in text.split("\n")]
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def fetch_article_excerpt(url: str, *, max_chars: int = 4000, timeout: float = 15.0) -> str:
    """Fetch an article's URL and return its readable text, truncated.

    Raises on network/HTTP failure; callers fetch per-article and tolerate
    individual failures so one paywalled page cannot sink a run.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"not a fetchable URL: {url!r}")

    with httpx.Client(
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if content_type and not any(kind in content_type for kind in ("html", "text", "xml")):
            raise ValueError(f"unsupported content-type {content_type!r} for {url}")

    return html_to_text(response.text)[:max_chars]
