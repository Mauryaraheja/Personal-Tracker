"""Tests for the one place that talks to Groq.

Speech is the first call here that isn't chat: it goes to a different
endpoint, so conftest's fake_groq (which only fills in .chat.completions)
can't stand in for it.
"""

from types import SimpleNamespace

import pytest
from personaltracker import llm


def fake_audio_client(transcript):
    """A stand-in for groq_client that answers audio transcriptions.

    It remembers how it was called, so a test can check the REQUEST --
    which model, and the file tuple the API needs -- and not just the
    answer.
    """

    def create(**kwargs):
        create.calls.append(kwargs)
        return SimpleNamespace(text=transcript)

    create.calls = []
    return SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )


def test_a_recording_comes_back_as_text(monkeypatch):
    monkeypatch.setattr(llm, "groq_client", fake_audio_client("  I used Postgres.  "))

    assert llm.transcribe(b"fake-wav-bytes") == "I used Postgres."


@pytest.mark.parametrize("silence", ["", "   ", "\n\t"])
def test_silence_is_an_error_not_an_empty_answer(monkeypatch, silence):
    monkeypatch.setattr(llm, "groq_client", fake_audio_client(silence))

    with pytest.raises(ValueError, match="Nothing was heard"):
        llm.transcribe(b"fake-wav-bytes")


def test_the_recording_is_sent_as_a_named_file(monkeypatch):
    client = fake_audio_client("hello")
    monkeypatch.setattr(llm, "groq_client", client)

    llm.transcribe(b"wav", filename="answer_qst_001.wav")

    sent = client.audio.transcriptions.create.calls[0]
    assert sent["file"] == ("answer_qst_001.wav", b"wav")
    assert sent["model"] == llm.TRANSCRIBE_MODEL
