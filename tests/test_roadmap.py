"""Tests for how build_roadmap handles Groq's reply.

Groq's JSON mode promises valid JSON, not the right shape. A reply with
the wrong key used to become an empty list: nothing got saved, and the
app showed nothing -- no error at all. These pin down that it now fails
loudly instead.
"""

from types import SimpleNamespace

import pytest
from personaltracker import roadmap
from personaltracker import llm


class FakeTavily:
    def search(self, **kwargs):
        return {"results": [{"url": "https://example.com/job", "content": "Needs RAG."}]}

@pytest.fixture(autouse=True)
def no_real_search(monkeypatch):
    monkeypatch.setattr(roadmap, "tavily_client", FakeTavily())


def test_a_good_reply_becomes_skills_with_ids(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq(
        {"skills": [{"name": "RAG"}, {"name": "Vector databases"}]}
    ))

    skills = roadmap.build_roadmap("Generative AI Engineer")

    assert [s["id"] for s in skills] == ["skl_001", "skl_002"]


def test_a_reply_with_the_wrong_key_raises(monkeypatch,fake_groq):
    """Valid JSON, wrong shape -- the case JSON mode can't prevent."""
    monkeypatch.setattr(llm, "groq_client", fake_groq({"roadmap": [{"name": "RAG"}]}))

    with pytest.raises(ValueError):
        roadmap.build_roadmap("Generative AI Engineer")


def test_an_empty_skill_list_raises(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"skills": []}))

    with pytest.raises(ValueError):
        roadmap.build_roadmap("Generative AI Engineer")


def test_a_reply_that_is_not_json_raises(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq('{"skills": [{"name": "RA'))

    with pytest.raises(ValueError):
        roadmap.build_roadmap("Generative AI Engineer")


# ---------------------------------------------------------------------------
# refine_role: the same input must give the same job title
# ---------------------------------------------------------------------------

class RecordingGroq:
    """Like the fake_groq fixture, but it remembers how it was called.

    conftest's fake_groq only controls the reply. These tests are about
    the request -- which settings reach the API -- so they need the
    kwargs, not the answer.
    """

    def __init__(self, reply="Graphics Programmer"):
        self.calls = []

        def create(**kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_refine_role_asks_for_a_deterministic_answer(monkeypatch):
    """A roadmap is cached forever under whatever title comes back here.

    At the API default this call was a coin flip between "Graphics
    Programmer" and "Graphics Engineer" -- so the same typed role could
    give two different roadmaps depending on the day.
    """
    recorder = RecordingGroq()
    monkeypatch.setattr(llm, "groq_client", recorder)

    roadmap.refine_role("graphics programming")

    assert recorder.calls[0]["temperature"] == 0


def test_temperature_zero_is_actually_sent(monkeypatch):
    """Guards the falsy-zero trap: `if temperature:` would drop it."""
    recorder = RecordingGroq()
    monkeypatch.setattr(llm, "groq_client", recorder)

    llm.ask_groq("anything", temperature=0)

    assert "temperature" in recorder.calls[0], "temperature=0 must reach the API"
    assert recorder.calls[0]["temperature"] == 0


def test_a_call_that_asks_for_no_temperature_sends_none(monkeypatch):
    """Every other call site must look exactly as it did before."""
    recorder = RecordingGroq()
    monkeypatch.setattr(llm, "groq_client", recorder)

    llm.ask_groq("anything")

    assert "temperature" not in recorder.calls[0]


def test_refine_role_tells_the_model_to_prefer_the_specific_title(monkeypatch):
    """The rule that flips "Graphics Engineer" -> "Graphics Programmer".

    Pinned as text because it is the whole fix, and because it must stay
    general -- no field-specific example belongs in this prompt.
    """
    recorder = RecordingGroq()
    monkeypatch.setattr(llm, "groq_client", recorder)

    roadmap.refine_role("graphics programming")

    prompt = recorder.calls[0]["messages"][0]["content"]
    assert "two different jobs in different fields" in prompt
    assert "more specific title" in prompt
