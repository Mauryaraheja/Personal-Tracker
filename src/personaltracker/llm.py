"""
The one place in this app that talks to Groq.

Every feature used to repeat the same few lines: build the request, name
the model, read the reply out of the response object, parse the JSON, and
raise if that failed. Seven call sites, six copies of the same parsing.

They live here now, so anything about *how* we call the model -- the
model name, a retry, a token count, a different provider one day -- is
changed once instead of seven times. The features keep what is actually
theirs: the prompt, and what the answer has to contain.
"""

import json

from .clients import groq_client

MODEL = "openai/gpt-oss-120b"


def ask_groq(prompt: str, max_tokens: int | None = None) -> str:
    """Send one prompt and return the reply as plain text."""
    return _create(prompt, max_tokens=max_tokens)


def ask_groq_for_json(prompt: str, what: str, max_tokens: int | None = None) -> dict:
    """Send one prompt in JSON mode and return the reply as a dict.

    JSON mode guarantees valid JSON, not the right shape, so callers
    still check what came back. `what` names the step in the error
    message, e.g. "grading" -> "Groq's grading reply wasn't JSON".
    """
    raw_text = _create(prompt, max_tokens=max_tokens, json_mode=True)

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Groq's {what} reply wasn't JSON: {raw_text[:200]!r}") from err


def _create(prompt: str, max_tokens: int | None = None, json_mode: bool = False) -> str:
    """The actual API call. Optional settings are only sent when asked
    for, so the request looks exactly like it did before."""
    options = {}
    if max_tokens:
        options["max_tokens"] = max_tokens
    if json_mode:
        options["response_format"] = {"type": "json_object"}

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        **options,
    )
    return response.choices[0].message.content
