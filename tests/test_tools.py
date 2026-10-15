"""Offline tests for the non-network tools: dedupe, relevance, render and send."""

from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.models import Article, NewsletterDraft, NewsletterItem
from src.tools import (
    dedupe_articles,
    html_to_text,
    rank_articles,
    relevance_score,
    render_newsletter,
    send_newsletter,
)
from src.tools.sources import keywords_from, normalize_url


def test_normalize_url_removes_noise() -> None:
    assert normalize_url("https://www.Example.com/Story/?utm=1#part") == "example.com/story"
    assert normalize_url("http://example.com/story") == normalize_url(
        "https://www.example.com/story/"
    )


def test_dedupe_drops_repeated_urls_and_titles() -> None:
    articles = [
        Article(title="One", url="https://example.com/a", source="s"),
        Article(
            title="One", url="https://example.com/a?ref=x", source="s"
        ),  # same URL after normalising
        Article(
            title="One!", url="https://other.com/b", source="s"
        ),  # same title after normalising
        Article(title="Two", url="https://example.com/c", source="s"),
    ]

    unique = dedupe_articles(articles)

    assert [article.title for article in unique] == ["One", "Two"]


def test_relevance_prefers_on_topic_and_recent() -> None:
    on_topic = Article(
        title="AI agent framework ships",
        url="https://example.com/1",
        source="s",
        published="2026-10-14T00:00:00+00:00",
    )
    off_topic = Article(
        title="Fridge hack",
        url="https://example.com/2",
        source="s",
        published="2026-01-01T00:00:00+00:00",
    )

    keywords = keywords_from("newsletter about AI agent news")
    assert relevance_score(on_topic, keywords) > relevance_score(off_topic, keywords)
    assert rank_articles([off_topic, on_topic], keywords)[0] is not None


def test_html_to_text_strips_markup_and_scripts() -> None:
    text = html_to_text(
        "<html><head><style>b{}</style></head><body>"
        "<h1>Title</h1><script>alert(1)</script><p>Body &amp; more</p></body></html>"
    )

    assert "Title" in text
    assert "Body & more" in text
    assert "alert" not in text
    assert "<p>" not in text


def test_render_newsletter_includes_every_link(draft: NewsletterDraft) -> None:
    rendered = render_newsletter(draft)

    assert rendered.subject == draft.subject
    for item in draft.items:
        assert item.url in rendered.markdown
        assert item.title in rendered.markdown
        assert item.url in rendered.html
        assert item.summary in rendered.html


def test_render_newsletter_escapes_html(draft: NewsletterDraft) -> None:
    draft.items[0].title = "<script>alert('xss')</script>"

    rendered = render_newsletter(draft)

    assert "<script>alert" not in rendered.html
    assert "&lt;script&gt;" in rendered.html


def test_send_newsletter_writes_both_files(
    draft: NewsletterDraft, settings: Settings, tmp_path, capsys
) -> None:
    rendered = render_newsletter(draft)

    receipt = send_newsletter(rendered, to=list(settings.recipients), out_dir=tmp_path / "out")

    html_path, email_path = receipt.html_path, receipt.email_path
    assert html_path.endswith(".html") and email_path.endswith(".txt")

    email_text = Path(email_path).read_text(encoding="utf-8")
    assert f"Subject: {draft.subject}" in email_text
    assert draft.items[0].url in email_text
    assert "test@example.com" in email_text
    assert "<html" in Path(html_path).read_text(encoding="utf-8")

    printed = capsys.readouterr().out
    assert f"Subject: {draft.subject}" in printed, "the simulated send must print the subject"
    assert draft.items[0].title in printed, "the simulated send must print the body"

    assert receipt.simulated is True
    assert receipt.to == ["test@example.com"]


def test_send_newsletter_real_delivery_is_an_honest_stub(draft: NewsletterDraft, tmp_path) -> None:
    rendered = render_newsletter(draft)

    try:
        send_newsletter(rendered, to=["a@b.c"], out_dir=tmp_path, simulate=False)
    except NotImplementedError as exc:
        assert "not implemented" in str(exc).lower()
    else:  # pragma: no cover - would mean a real transport appeared
        raise AssertionError("simulate=False should not silently succeed")


def test_render_newsletter_uses_the_item_links_verbatim() -> None:
    item = NewsletterItem(
        title="T",
        url="https://example.com/only-real-url",
        source="s",
        summary="x",
        article_id="A1",
    )
    draft = NewsletterDraft(subject="s", items=[item])

    rendered = render_newsletter(draft)

    assert rendered.markdown.count("https://example.com/only-real-url") == 1
