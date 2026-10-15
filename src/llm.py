"""The only module that touches the LLM SDK.

Any OpenAI-compatible provider is reached through the ``openai`` client with a
custom ``base_url``. Everything is read from the environment; nothing is
hardcoded. Because all SDK contact lives here, switching provider is a ``.env``
change (``LLM_BASE_URL`` / ``LLM_MODEL``).

Call sites import the *module* (``from src import llm``) and call
``llm.chat_json(...)``, so the test-suite can substitute a fake LLM with
``monkeypatch.setattr(llm, "chat_json", fake)``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

Message = dict[str, str]

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

#: Extra attempts after the first one, for transient network/API failures.
REQUEST_RETRIES = 2
_BACKOFF_SECONDS = 1.5


class LLMError(RuntimeError):
    """Raised when the LLM is unreachable, unconfigured, or unparseable."""


_client: OpenAI | None = None
_client_signature: tuple[str, str] | None = None


def model_name() -> str:
    """The model id currently configured in the environment."""
    load_dotenv(override=False)
    return os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL


def base_url() -> str:
    """The provider base URL currently configured in the environment."""
    load_dotenv(override=False)
    return os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL


def api_key() -> str:
    """The API key currently configured in the environment (may be empty)."""
    load_dotenv(override=False)
    return os.environ.get("LLM_API_KEY", "").strip()


def get_client() -> OpenAI:
    """Return a cached OpenAI-compatible client for the configured provider."""
    key, url = api_key(), base_url()
    if not key:
        raise LLMError(
            "LLM_API_KEY is not set. Copy .env.example to .env and set your "
            "provider's API key before asking the agent to write anything."
        )

    global _client, _client_signature
    signature = (key, url)
    if _client is None or _client_signature != signature:
        _client = OpenAI(api_key=key, base_url=url)
        _client_signature = signature
    return _client


def chat(
    messages: list[Message],
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """One ``chat.completions`` round trip. Retries transient failures."""
    client = get_client()
    request: dict[str, Any] = {
        "model": model_name(),
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if json_mode:
        request["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            response = client.chat.completions.create(**request)
            return response.choices[0].message.content or ""
        except (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            APIStatusError,
        ) as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(_BACKOFF_SECONDS * (attempt + 1))

    raise LLMError(f"LLM request failed after {REQUEST_RETRIES + 1} attempts: {last_error}")


def chat_json[T: BaseModel](
    messages: list[Message],
    schema: type[T],
    *,
    temperature: float = 0.0,
    retries: int = 1,
) -> T:
    """Ask for JSON and validate it with ``schema``.

    Models wrap JSON in code fences or add prose, so the first ``{...}`` block is
    extracted. On a parse/validation failure the error is fed back and the call
    is retried once; if that also fails, :class:`LLMError` is raised and the
    calling node falls back to deterministic behaviour.
    """
    conversation = list(messages)
    last_error: Exception | None = None

    for _ in range(retries + 1):
        raw = chat(conversation, temperature=temperature, json_mode=True)
        try:
            return schema.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as exc:
            last_error = exc
            conversation = [
                *messages,
                {"role": "assistant", "content": raw[:2000]},
                {
                    "role": "user",
                    "content": (
                        f"That reply was not valid for the required JSON schema ({exc}). "
                        "Reply with ONLY one JSON object, no prose and no code fences."
                    ),
                },
            ]

    raise LLMError(f"could not parse JSON response as {schema.__name__}: {last_error}")


def extract_json(text: str) -> Any:
    """Return the first balanced ``{...}`` object found in ``text``.

    Brace matching is string-aware, so braces inside quoted values do not end
    the object early.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise ValueError("unbalanced JSON object in response")
