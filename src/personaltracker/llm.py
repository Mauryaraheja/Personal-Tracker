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


def ask_groq(prompt: str, max_tokens: int | None = None,
             temperature: float | None = None) -> str:
    """Send one prompt and return the reply as plain text."""
    return _create(prompt, max_tokens=max_tokens, temperature=temperature)


def ask_groq_for_json(prompt: str, what: str, max_tokens: int | None = None,
                      temperature: float | None = None) -> dict:
    """Send one prompt in JSON mode and return the reply as a dict.

    JSON mode guarantees valid JSON, not the right shape, so callers
    still check what came back. `what` names the step in the error
    message, e.g. "grading" -> "Groq's grading reply wasn't JSON".
    """
    raw_text = _create(prompt, max_tokens=max_tokens, json_mode=True,
                       temperature=temperature)

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Groq's {what} reply wasn't JSON: {raw_text[:200]!r}") from err


def _create(prompt: str, max_tokens: int | None = None, json_mode: bool = False,
            temperature: float | None = None) -> str:
    """The actual API call. Optional settings are only sent when asked
    for, so the request looks exactly like it did before."""
    options = {}
    if max_tokens:
        options["max_tokens"] = max_tokens
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    # `is not None`, not a truthiness check: temperature=0 is the whole
    # point of this parameter and 0 is falsy, so `if temperature:` would
    # silently drop the one value callers actually ask for.
    if temperature is not None:
        options["temperature"] = temperature

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        **options,
    )
    return response.choices[0].message.content



# Whisper, on Groq. It lives here for the same reason every other Groq
# call does: one place decides which model we use and how we talk to the
# API. "turbo" is the cheap, fast one -- an interview answer is a minute
# of clear speech, not a noisy two-hour recording.
TRANSCRIBE_MODEL = "whisper-large-v3-turbo"


def transcribe(audio_bytes: bytes, filename: str = "answer.wav") -> str:
    """Turn a recorded answer into text.

    Silence raises instead of returning "". An empty answer is already
    meaningful to the grader -- it marks every key point missed and
    scores zero -- so a failed microphone would be graded as a candidate
    who had nothing to say. Those are different things, and only one of
    them belongs in the score.

    `filename` is only a label: the Groq API uses the extension to work
    out the audio format, and nothing is written to disk.
    """
    response = groq_client.audio.transcriptions.create(
        model=TRANSCRIBE_MODEL,
        file=(filename, audio_bytes),
    )

    text = response.text.strip()
    if not text:
        raise ValueError(
            "Nothing was heard in that recording -- record it again, or type your answer."
        )
    return text
