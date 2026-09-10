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

import pytest

from personaltracker import tracker


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
