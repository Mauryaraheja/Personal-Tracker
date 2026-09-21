"""Tests for how get_skill_gaps handles Groq's reply.

Same rule as build_roadmap: a reply with the wrong shape used to become
an empty list, and the app then quietly skipped "Next Steps". It must
fail loudly instead.
"""

import pytest
from personaltracker import gap_analysis
from personaltracker import llm

SKILLS = [
    {"id": "skl_001", "name": "RAG", "description": "Retrieval-augmented generation",
     "level_required": "advanced"},
]


def test_a_good_reply_becomes_gaps_with_ids(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"gaps": [
        {"skill_id": "skl_001", "status": "missing", "evidence_from_cv": None,
         "suggestion": "Build a small RAG demo"},
    ]}))

    gaps = gap_analysis.get_skill_gaps(SKILLS, "my CV text")

    assert [g["id"] for g in gaps] == ["gap_001"]


def test_a_reply_with_the_wrong_key_raises(monkeypatch,fake_groq):
    """Valid JSON, wrong shape -- the case JSON mode can't prevent."""
    monkeypatch.setattr(llm, "groq_client", fake_groq({"results": []}))

    with pytest.raises(ValueError):
        gap_analysis.get_skill_gaps(SKILLS, "my CV text")


def test_an_empty_gap_list_raises(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"gaps": []}))

    with pytest.raises(ValueError):
        gap_analysis.get_skill_gaps(SKILLS, "my CV text")


def test_a_reply_that_is_not_json_raises(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq('{"gaps": [{"skill_'))

    with pytest.raises(ValueError):
        gap_analysis.get_skill_gaps(SKILLS, "my CV text")


# ---------------------------------------------------------------------------
# Every skill gets exactly one gap -- the quiet version of the bug
# ---------------------------------------------------------------------------

TWO_SKILLS = SKILLS + [
    {"id": "skl_002", "name": "Vector databases", "description": "Similarity search",
     "level_required": "intermediate"},
]


def gap(skill_id):
    return {"skill_id": skill_id, "status": "missing", "evidence_from_cv": None,
            "suggestion": "Practise it"}


def test_gaps_can_come_back_in_any_order(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq(
        {"gaps": [gap("skl_002"), gap("skl_001")]}
    ))

    gaps = gap_analysis.get_skill_gaps(TWO_SKILLS, "my CV text")

    assert len(gaps) == 2


def test_a_skipped_skill_raises(monkeypatch,fake_groq):
    """Nothing crashes -- the skill would just vanish from Next Steps."""
    monkeypatch.setattr(llm, "groq_client", fake_groq({"gaps": [gap("skl_001")]}))

    with pytest.raises(ValueError):
        gap_analysis.get_skill_gaps(TWO_SKILLS, "my CV text")


def test_a_gap_for_an_unknown_skill_raises(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"gaps": [gap("skl_999")]}))

    with pytest.raises(ValueError):
        gap_analysis.get_skill_gaps(SKILLS, "my CV text")


def test_a_skill_covered_twice_raises(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq(
        {"gaps": [gap("skl_001"), gap("skl_001")]}
    ))

    with pytest.raises(ValueError):
        gap_analysis.get_skill_gaps(SKILLS, "my CV text")
