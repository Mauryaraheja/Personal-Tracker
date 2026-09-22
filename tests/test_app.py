"""Tests for app.py, the Streamlit layer.

app.py had no tests at all until a missing import survived a fully green
suite: nothing failed, and the mistake only appeared when the Rebuild
button was clicked -- where the broad `except Exception` reported it as
"usually a Groq or Tavily issue".

streamlit.testing.v1.AppTest runs the script headlessly and lets a test
click a button, so the parts of the UI that actually decide something can
be pinned down: what a confirm step costs, what a rebuild clears, and
what a failed rebuild leaves behind.

The autouse fixture in conftest points tracker at a throwaway database,
and fake_llm replaces both API-backed calls, so none of this touches the
real personaltracker.db, Groq, or Tavily.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _skill(skill_id, name):
    return {
        "id": skill_id,
        "name": name,
        "description": "d",
        "why_it_matters": "w",
        "priority": "high",
        "level_required": "intermediate",
        "source_url": None,
    }


def _click(at, label, handled_error=False):
    """Click a button by its label.

    handled_error=True for a click the app is supposed to fail on: the
    error expander calls st.exception(), which AppTest records in
    at.exception even though the app caught it and carried on.
    """
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button labelled {label!r}; found {[b.label for b in at.button]}"
    matches[0].click().run()
    if not handled_error:
        assert not at.exception, at.exception


def _open(db, role="Gen AI", **state):
    """Open the app mid-session, with the roadmap for `role` on screen."""
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["role"] = role
    at.session_state["skills"] = db.get_or_create_roadmap(role)
    at.session_state["tracker"] = db.get_tracker_items(role)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.fixture
def app(db, fake_llm):
    """The app mid-session: a role chosen and its roadmap on screen."""
    return _open(db, cv_text="my CV", market_insights={"stale": True})


def test_the_page_renders(app):
    """The cheapest guard against the missing-import class of mistake."""
    assert [b.label for b in app.button].count("Rebuild this roadmap") == 1


def test_asking_to_rebuild_spends_nothing_until_confirmed(app, fake_llm):
    before = len(fake_llm.build_calls)

    _click(app, "Rebuild this roadmap")

    assert app.warning, "the confirm step must say what it costs"
    assert len(fake_llm.build_calls) == before, "confirming is what spends credits"


def test_cancelling_spends_nothing(app, fake_llm):
    before = len(fake_llm.build_calls)

    _click(app, "Rebuild this roadmap")
    _click(app, "Cancel")

    assert len(fake_llm.build_calls) == before
    assert [b.label for b in app.button].count("Rebuild this roadmap") == 1


def test_confirming_rebuilds_and_clears_stale_results(app, db, fake_llm, monkeypatch):
    monkeypatch.setattr(db, "build_roadmap", lambda title: [_skill("skl_001", "Vulkan")])

    _click(app, "Rebuild this roadmap")
    _click(app, "Yes, rebuild")

    assert [s["name"] for s in app.session_state["skills"]] == ["Vulkan"]
    # Everything worked out from the OLD skills describes skills that may
    # no longer exist; the CV is the user's file, not a result.
    assert app.session_state["market_insights"] is None
    assert app.session_state["cv_text"] == "my CV"


def test_a_rebuilt_skill_shows_its_own_progress_not_the_old_one(db, fake_llm, monkeypatch):
    """The widget-key trap: a rebuild reuses skill ids.

    Without bumping skills_key, Streamlit keeps whatever was selected for
    "status_skl_001" and shows it for a brand new skill -- measured: the
    dropdown said "completed" while the database said not_started.

    The progress is set before the app opens on purpose. Once a widget
    has rendered, Streamlit keeps its value -- which is exactly the
    behaviour under test, so it must not be part of the setup.
    """
    db.get_or_create_roadmap("Gen AI")
    db.update_tracker_status("Gen AI", "skl_001", "completed", "old note")

    app = _open(db)
    assert app.selectbox[0].value == "completed"

    monkeypatch.setattr(db, "build_roadmap", lambda title: [_skill("skl_001", "Vulkan")])
    _click(app, "Rebuild this roadmap")
    _click(app, "Yes, rebuild")

    assert app.session_state["skills"][0]["name"] == "Vulkan"
    assert db.get_tracker_items("Gen AI")[0]["status"] == "not_started"
    assert app.selectbox[0].value == "not_started", "the dropdown must match the database"


def test_a_failed_rebuild_keeps_the_old_roadmap(app, db, monkeypatch):
    """A rate limit must not cost the user the roadmap they already had."""
    before = list(app.session_state["skills"])

    def boom(title):
        raise RuntimeError("Groq is rate limited")

    monkeypatch.setattr(db, "build_roadmap", boom)

    _click(app, "Rebuild this roadmap")
    _click(app, "Yes, rebuild", handled_error=True)

    assert app.error, "the user must be told it failed"
    assert app.session_state["skills"] == before
    assert db.get_or_create_roadmap("Gen AI") == before
