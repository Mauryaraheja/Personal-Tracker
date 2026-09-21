"""Shared test fixtures.

Every test in this suite runs without touching Groq, Tavily, or the real
personaltracker.db. That's deliberate and it's what makes the suite
useful: it runs in under a second, costs nothing, works offline, and
can't be broken by a rate limit or a bad API key.

The logic being tested here is the part that decides what's *true* --
which postings are duplicates, how many postings mention a skill, which
roadmap skills a match backs up. None of that is the model's job; it's
plain Python, so it can be pinned down exactly.
"""

import json
import os
from types import SimpleNamespace

import pytest

# The package builds its Groq and Tavily clients the moment it's imported,
# and both refuse to start without a key -- even though no test ever calls
# them. Placeholder keys let the suite run on a fresh clone with no .env.
# setdefault() never overwrites a key that's already set.
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("TAVILY_API_KEY", "test-key-not-used")

from personaltracker import tracker  # after the keys, or the import fails


@pytest.fixture(autouse=True)
def never_the_real_database(tmp_path, monkeypatch):
    """Point every test at a throwaway database, not personaltracker.db.

    The promise at the top of this file used to hold by accident: only
    tracker.py talked to SQLite, and only tests that asked for the `db`
    fixture reached it. Then the market check began caching postings, so
    a test that merely runs an extraction writes rows -- and three of
    them landed in the real personaltracker.db before this existed.

    autouse makes the promise hold for tests that do not know they touch
    the database at all. The `db` fixture below still works: it sets
    DB_PATH again and runs init_db(), and the last setattr wins.
    """
    monkeypatch.setattr(tracker, "DB_PATH", tmp_path / "auto.db")


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point tracker at a throwaway database for one test.

    tracker.DB_PATH is read inside _connect() on every call rather than
    captured at import, so swapping the module attribute is enough to
    redirect every query in the module.
    """
    monkeypatch.setattr(tracker, "DB_PATH", tmp_path / "test.db")
    tracker.init_db()
    return tracker


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the two LLM-backed calls tracker depends on.

    Returns a record of what was asked, so a test can assert not just
    that the right answer came back but that the API was called the
    right number of times -- which is the whole point of the caching.
    """

    class Recorder:
        def __init__(self):
            self.refine_calls = []
            self.build_calls = []
            # what refine_role should return, keyed by lowercased input
            self.refinements = {}
            self.default_refinement = "Generative AI Engineer"

        def refine(self, role):
            self.refine_calls.append(role)
            return self.refinements.get(role.strip().lower(), self.default_refinement)

        def build(self, refined_role):
            self.build_calls.append(refined_role)
            return [
                {
                    "id": f"skl_{i:03d}",
                    "name": f"{refined_role} skill {i}",
                    "description": f"description {i}",
                    "why_it_matters": f"why {i}",
                    "priority": "high",
                    "level_required": "intermediate",
                    "source_url": f"https://example.com/{i}",
                }
                for i in range(1, 4)
            ]

    recorder = Recorder()
    monkeypatch.setattr(tracker, "refine_role", recorder.refine)
    monkeypatch.setattr(tracker, "build_roadmap", recorder.build)
    return recorder


@pytest.fixture
def fake_groq():
    """A stand-in for groq_client, shared by every test file.

    It gives the test a function: fake_groq(reply) builds a fake client
    that always answers with `reply` -- a dict (sent back as JSON), or a
    plain string for replies that aren't valid JSON.
    """

    def build(reply):
        content = reply if isinstance(reply, str) else json.dumps(reply)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=message)])
        )))

    return build
