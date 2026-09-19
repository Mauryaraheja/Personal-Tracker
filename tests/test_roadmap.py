"""Tests for how build_roadmap handles Groq's reply.

Groq's JSON mode promises valid JSON, not the right shape. A reply with
the wrong key used to become an empty list: nothing got saved, and the
app showed nothing -- no error at all. These pin down that it now fails
loudly instead.
"""

import json
from types import SimpleNamespace

import pytest

from personaltracker import roadmap


class FakeTavily:
    def search(self, **kwargs):
        return {"results": [{"url": "https://example.com/job", "content": "Needs RAG."}]}


def fake_groq(reply):
    """A stand-in for groq_client that always answers with `reply`."""
    content = reply if isinstance(reply, str) else json.dumps(reply)
    message = SimpleNamespace(content=content)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=message)])
    )))


@pytest.fixture(autouse=True)
def no_real_search(monkeypatch):
    monkeypatch.setattr(roadmap, "tavily_client", FakeTavily())


def test_a_good_reply_becomes_skills_with_ids(monkeypatch):
    monkeypatch.setattr(roadmap, "groq_client", fake_groq(
        {"skills": [{"name": "RAG"}, {"name": "Vector databases"}]}
    ))

    skills = roadmap.build_roadmap("Generative AI Engineer")

    assert [s["id"] for s in skills] == ["skl_001", "skl_002"]


def test_a_reply_with_the_wrong_key_raises(monkeypatch):
    """Valid JSON, wrong shape -- the case JSON mode can't prevent."""
    monkeypatch.setattr(roadmap, "groq_client", fake_groq({"roadmap": [{"name": "RAG"}]}))

    with pytest.raises(ValueError):
        roadmap.build_roadmap("Generative AI Engineer")


def test_an_empty_skill_list_raises(monkeypatch):
    monkeypatch.setattr(roadmap, "groq_client", fake_groq({"skills": []}))

    with pytest.raises(ValueError):
        roadmap.build_roadmap("Generative AI Engineer")


def test_a_reply_that_is_not_json_raises(monkeypatch):
    monkeypatch.setattr(roadmap, "groq_client", fake_groq('{"skills": [{"name": "RA'))

    with pytest.raises(ValueError):
        roadmap.build_roadmap("Generative AI Engineer")