"""The simulated send step: writes the artifacts and prints the email.

No network is involved and no credentials are required. ``simulate=False`` is an
honest stub: wiring a real SMTP/SES transport is out of scope for the
assignment, and the README says so.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.models import RenderedNewsletter, SendReceipt


def send_newsletter(
    rendered: RenderedNewsletter,
    *,
    to: list[str],
    out_dir: Path,
    simulate: bool = True,
) -> SendReceipt:
    """Write ``output/newsletter-YYYY-MM-DD.html`` and ``output/email-YYYY-MM-DD.txt``.

    The subject and the full body are also printed to stdout, which is the
    "simulated send" required by the assignment.
    """
    if not simulate:
        raise NotImplementedError(
            "Real delivery is not implemented. This project simulates the send; "
            "set simulate=True (the default) or wire an SMTP transport here."
        )

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).date().isoformat()
    html_path = directory / f"newsletter-{stamp}.html"
    email_path = directory / f"email-{stamp}.txt"

    html_path.write_text(rendered.html, encoding="utf-8")

    recipients = ", ".join(to)
    email_text = f"To: {recipients}\nSubject: {rendered.subject}\n\n{rendered.markdown}"
    email_path.write_text(email_text, encoding="utf-8")

    print(f"To: {recipients}")
    print(f"Subject: {rendered.subject}")
    print()
    print(rendered.markdown)

    return SendReceipt(
        to=list(to),
        subject=rendered.subject,
        body=rendered.markdown,
        html_path=str(html_path),
        email_path=str(email_path),
        simulated=True,
    )
