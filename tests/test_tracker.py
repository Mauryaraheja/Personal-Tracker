"""Tests for role identity and progress persistence.

The bug these exist to prevent: two spellings of the same role quietly
becoming two roadmaps with two separate sets of progress, and the user
appearing to lose everything they'd tracked.
"""

import pytest

from personaltracker.tracker import _normalize_role

# ---------------------------------------------------------------------------
# _normalize_role -- casing and spacing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "typed",
    ["Gen AI", "gen ai", "GEN AI", "  Gen AI  ", "Gen  AI", "\tGen\nAI "],
)
def test_normalize_role_collapses_spelling_variants(typed):
    assert _normalize_role(typed) == "gen ai"


def test_normalize_role_keeps_genuinely_different_roles_apart():
    assert _normalize_role("Backend Engineer") != _normalize_role("Frontend Engineer")


@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n  \n"])
def test_normalize_role_rejects_blank(blank):
    # A blank role would create a nameless row that nothing can ever find.
    with pytest.raises(ValueError):
        _normalize_role(blank)


# ---------------------------------------------------------------------------
# Roadmap caching and role aliasing
# ---------------------------------------------------------------------------

def test_roadmap_is_generated_once_then_served_from_db(db, fake_llm):
    first = db.get_or_create_roadmap("Gen AI")
    second = db.get_or_create_roadmap("Gen AI")

    assert first == second
    assert len(fake_llm.build_calls) == 1, "roadmap should only be built once"


def test_spelling_variants_share_one_roadmap(db, fake_llm):
    db.get_or_create_roadmap("Gen AI")

    for variant in ["gen ai", "GEN AI", "  Gen  AI  "]:
        assert db.get_or_create_roadmap(variant) == db.get_or_create_roadmap("Gen AI")

    assert len(fake_llm.build_calls) == 1, "variants must not rebuild the roadmap"


def test_variants_cost_no_extra_api_calls(db, fake_llm):
    db.get_or_create_roadmap("Gen AI")
    calls_after_first = len(fake_llm.refine_calls)

    db.get_or_create_roadmap("gen ai")
    db.get_or_create_roadmap("GEN AI")

    assert len(fake_llm.refine_calls) == calls_after_first, (
        "a spelling that normalizes to a known one must not hit the LLM"
    )


def test_different_wording_reaches_the_same_roadmap(db, fake_llm):
    """The point of role_aliases: only the LLM knows these are one job."""
    fake_llm.refinements = {
        "gen ai": "Generative AI Engineer",
        "generative ai engineer": "Generative AI Engineer",
    }

    from_shorthand = db.get_or_create_roadmap("Gen AI")
    from_full_title = db.get_or_create_roadmap("Generative AI Engineer")

    assert from_shorthand == from_full_title
    assert len(fake_llm.build_calls) == 1, "should reuse, not rebuild"


def test_roadmap_is_built_from_the_refined_title(db, fake_llm):
    """Searching for "Gen AI" returns listicles; the job title returns jobs."""
    fake_llm.refinements = {"gen ai": "Generative AI Engineer"}

    db.get_or_create_roadmap("Gen AI")

    assert fake_llm.build_calls == ["Generative AI Engineer"]


def test_genuinely_different_roles_stay_separate(db, fake_llm):
    fake_llm.refinements = {
        "backend engineer": "Backend Engineer",
        "data scientist": "Data Scientist",
    }

    backend = db.get_or_create_roadmap("Backend Engineer")
    data = db.get_or_create_roadmap("Data Scientist")

    assert backend != data
    assert len(fake_llm.build_calls) == 2


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def test_every_skill_starts_untracked(db, fake_llm):
    skills = db.get_or_create_roadmap("Gen AI")
    items = db.get_tracker_items("Gen AI")

    assert len(items) == len(skills)
    assert {i["status"] for i in items} == {"not_started"}


def test_status_change_persists(db, fake_llm):
    db.get_or_create_roadmap("Gen AI")

    db.update_tracker_status("Gen AI", "skl_002", "completed")

    statuses = {i["skill_id"]: i["status"] for i in db.get_tracker_items("Gen AI")}
    assert statuses["skl_002"] == "completed"
    assert statuses["skl_001"] == "not_started", "only the named skill should change"


def test_progress_survives_a_different_spelling(db, fake_llm):
    """The user-facing bug: capitalize differently, appear to lose progress."""
    db.get_or_create_roadmap("Gen AI")
    db.update_tracker_status("Gen AI", "skl_001", "completed")

    statuses = {i["skill_id"]: i["status"] for i in db.get_tracker_items("gen ai")}
    assert statuses["skl_001"] == "completed"


def test_tracker_reads_and_writes_never_call_the_llm(db, fake_llm):
    db.get_or_create_roadmap("Gen AI")
    calls_before = len(fake_llm.refine_calls)

    db.get_tracker_items("Gen AI")
    db.update_tracker_status("gen ai", "skl_001", "in_progress")
    db.get_tracker_items("GEN AI")

    assert len(fake_llm.refine_calls) == calls_before, (
        "marking a skill complete must not cost an API call"
    )


def test_progress_is_scoped_per_role(db, fake_llm):
    fake_llm.refinements = {
        "backend engineer": "Backend Engineer",
        "data scientist": "Data Scientist",
    }
    db.get_or_create_roadmap("Backend Engineer")
    db.get_or_create_roadmap("Data Scientist")

    db.update_tracker_status("Backend Engineer", "skl_001", "completed")

    other = {i["skill_id"]: i["status"] for i in db.get_tracker_items("Data Scientist")}
    assert other["skl_001"] == "not_started", "roles must not share progress"


def test_notes_are_kept_when_only_status_changes(db, fake_llm):
    db.get_or_create_roadmap("Gen AI")
    db.update_tracker_status("Gen AI", "skl_001", "in_progress", notes="started the course")

    db.update_tracker_status("Gen AI", "skl_001", "completed")

    notes = {i["skill_id"]: i["notes"] for i in db.get_tracker_items("Gen AI")}
    assert notes["skl_001"] == "started the course"


# ---------------------------------------------------------------------------
# Writes that go nowhere
# ---------------------------------------------------------------------------

def test_invalid_status_is_rejected(db, fake_llm):
    db.get_or_create_roadmap("Gen AI")

    with pytest.raises(ValueError):
        db.update_tracker_status("Gen AI", "skl_001", "done")


def test_unknown_skill_raises_instead_of_silently_doing_nothing(db, fake_llm):
    """An UPDATE matching zero rows is not an error in SQL -- it just does
    nothing. Accepting that silently is how a tracker ends up disagreeing
    with what the user sees on screen."""
    db.get_or_create_roadmap("Gen AI")

    with pytest.raises(KeyError):
        db.update_tracker_status("Gen AI", "skl_999", "completed")


def test_unknown_role_raises(db, fake_llm):
    db.get_or_create_roadmap("Gen AI")

    with pytest.raises(KeyError):
        db.update_tracker_status("Underwater Basket Weaver", "skl_001", "completed")


def test_tracker_items_for_an_unknown_role_is_empty_not_an_error(db, fake_llm):
    assert db.get_tracker_items("Never Tracked") == []


# ---------------------------------------------------------------------------
# The saved job title -- what the market check searches for
# ---------------------------------------------------------------------------

def test_refined_title_is_read_back_without_calling_the_llm(db, fake_llm):
    fake_llm.refinements = {"gen ai": "Generative AI Engineer"}
    db.get_or_create_roadmap("Gen AI")
    calls_before = len(fake_llm.refine_calls)

    assert db.get_refined_title("gen ai") == "Generative AI Engineer"
    assert len(fake_llm.refine_calls) == calls_before, (
        "reading the saved title must not cost an API call"
    )


def test_refined_title_for_an_unknown_role_raises(db, fake_llm):
    with pytest.raises(KeyError):
        db.get_refined_title("Never Tracked")

# ---------------------------------------------------------------------------
# Rebuilding a saved roadmap
# ---------------------------------------------------------------------------

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


def test_rebuild_generates_the_roadmap_again(db, fake_llm):
    db.get_or_create_roadmap("Gen AI")
    assert len(fake_llm.build_calls) == 1

    db.rebuild_roadmap("Gen AI")

    assert len(fake_llm.build_calls) == 2, "rebuild must call the LLM again"
    assert len(db.get_or_create_roadmap("Gen AI")) == 3, "and the new one must be saved"


def test_rebuild_replaces_the_old_skills(db, fake_llm, monkeypatch):
    db.get_or_create_roadmap("Gen AI")

    monkeypatch.setattr(db, "build_roadmap", lambda title: [_skill("skl_001", "Vulkan")])
    db.rebuild_roadmap("Gen AI")

    names = [s["name"] for s in db.get_or_create_roadmap("Gen AI")]
    assert names == ["Vulkan"], "the old roadmap must be gone, not appended to"


def test_rebuild_keeps_progress_for_a_skill_that_survives(db, fake_llm, monkeypatch):
    """Progress follows the skill NAME, not its id -- the same skill can
    come back at a different position in the new roadmap."""
    db.get_or_create_roadmap("Gen AI")
    db.update_tracker_status("Gen AI", "skl_002", "completed", "finished the course")

    monkeypatch.setattr(db, "build_roadmap", lambda title: [
        _skill("skl_001", "Generative AI Engineer skill 2"),  # same skill, new id
        _skill("skl_002", "Vulkan"),                          # brand new
    ])
    db.rebuild_roadmap("Gen AI")

    items = {i["skill_id"]: i for i in db.get_tracker_items("Gen AI")}
    assert items["skl_001"]["status"] == "completed"
    assert items["skl_001"]["notes"] == "finished the course"
    assert items["skl_002"]["status"] == "not_started"
    assert items["skl_002"]["notes"] is None


def test_rebuild_drops_progress_for_a_skill_that_disappears(db, fake_llm, monkeypatch):
    db.get_or_create_roadmap("Gen AI")
    db.update_tracker_status("Gen AI", "skl_001", "completed", "gone soon")

    monkeypatch.setattr(db, "build_roadmap", lambda title: [_skill("skl_001", "Vulkan")])
    db.rebuild_roadmap("Gen AI")

    items = db.get_tracker_items("Gen AI")
    assert [i["name"] for i in items] == ["Vulkan"]
    assert items[0]["notes"] is None, "notes must not land on a different skill"


def test_a_spelling_difference_still_counts_as_the_same_skill(db, fake_llm, monkeypatch):
    db.get_or_create_roadmap("Gen AI")
    db.update_tracker_status("Gen AI", "skl_001", "in_progress", "halfway")

    monkeypatch.setattr(db, "build_roadmap", lambda title: [
        _skill("skl_001", "  generative ai engineer   SKILL 1 "),
    ])
    db.rebuild_roadmap("Gen AI")

    items = db.get_tracker_items("Gen AI")
    assert items[0]["status"] == "in_progress"
    assert items[0]["notes"] == "halfway"


def test_a_failed_rebuild_leaves_the_old_roadmap_alone(db, fake_llm, monkeypatch):
    """A rate limit must not cost the user the roadmap they already had."""
    before = db.get_or_create_roadmap("Gen AI")
    db.update_tracker_status("Gen AI", "skl_001", "completed", "keep me")

    def boom(title):
        raise RuntimeError("Groq is rate limited")

    monkeypatch.setattr(db, "build_roadmap", boom)

    with pytest.raises(RuntimeError):
        db.rebuild_roadmap("Gen AI")

    assert db.get_or_create_roadmap("Gen AI") == before
    assert db.get_tracker_items("Gen AI")[0]["notes"] == "keep me"
