"""Tests for the parts of the market pipeline that decide what's true.

The model's job is reading one job posting and naming the technologies in
it. Everything downstream of that -- which postings are duplicates, how
many postings mention a skill, which roadmap skills a match backs up --
is plain Python, and that's exactly the part worth pinning down.

This matters because the original version asked the model to count, and
it returned "1" for everything. These tests are what makes the fix
provable rather than merely believed.
"""

import json
from types import SimpleNamespace

import pytest

from personaltracker import market_validation as mv


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def groq_reply(payload):
    """Shape a fake Groq response the way the real client returns one."""
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class FakeTavily:
    """Returns canned results per domain, or raises for chosen domains."""

    def __init__(self, results_by_domain, failing_domains=()):
        self.results_by_domain = results_by_domain
        self.failing_domains = set(failing_domains)
        self.queries = []

    def search(self, **kwargs):
        domain = kwargs["include_domains"][0]
        self.queries.append(domain)
        if domain in self.failing_domains:
            raise RuntimeError(f"boom: {domain}")
        return {"results": self.results_by_domain.get(domain, [])}


def posting(url, raw_content="some job text"):
    return {"url": url, "raw_content": raw_content, "content": raw_content}


@pytest.fixture
def no_sleep(monkeypatch):
    """The pipeline sleeps 4s between calls to respect the token ceiling.

    Correct in production, pointless in a test -- 12 postings would mean
    a 44-second test that proves nothing extra.
    """
    monkeypatch.setattr(mv, "time", SimpleNamespace(sleep=lambda _seconds: None))


# ---------------------------------------------------------------------------
# search_job_postings -- deduplication
# ---------------------------------------------------------------------------
# Duplicates silently inflate mention_count, which is the number the whole
# feature exists to report. They have to be caught here.

def test_same_posting_over_http_and_https_counts_once(monkeypatch):
    monkeypatch.setattr(mv, "tavily_client", FakeTavily({
        "greenhouse.io": [posting("http://boards.greenhouse.io/acme/jobs/1")],
        "lever.co": [posting("https://boards.greenhouse.io/acme/jobs/1")],
    }))

    results = mv.search_job_postings("Gen AI")

    assert len(results) == 1


def test_query_strings_do_not_create_duplicates(monkeypatch):
    monkeypatch.setattr(mv, "tavily_client", FakeTavily({
        "greenhouse.io": [posting("https://x.com/jobs/1?utm_source=a")],
        "lever.co": [posting("https://x.com/jobs/1?utm_source=b")],
    }))

    assert len(mv.search_job_postings("Gen AI")) == 1


def test_trailing_slash_does_not_create_a_duplicate(monkeypatch):
    monkeypatch.setattr(mv, "tavily_client", FakeTavily({
        "greenhouse.io": [posting("https://x.com/jobs/1/")],
        "lever.co": [posting("https://x.com/jobs/1")],
    }))

    assert len(mv.search_job_postings("Gen AI")) == 1


def test_genuinely_different_postings_are_all_kept(monkeypatch):
    monkeypatch.setattr(mv, "tavily_client", FakeTavily({
        "greenhouse.io": [posting("https://x.com/jobs/1")],
        "lever.co": [posting("https://x.com/jobs/2")],
    }))

    assert len(mv.search_job_postings("Gen AI")) == 2


# ---------------------------------------------------------------------------
# search_job_postings -- filtering, interleaving, capping
# ---------------------------------------------------------------------------

def test_wellfound_aggregator_pages_are_excluded(monkeypatch):
    """Wellfound serves real postings and multi-role listing pages on the
    same domain, so include_domains can't tell them apart -- only the URL
    path can."""
    monkeypatch.setattr(mv, "tavily_client", FakeTavily({
        "wellfound.com": [
            posting("https://wellfound.com/role/r/ai-engineer"),
            posting("https://wellfound.com/jobs/12345-ai-engineer"),
        ],
    }))

    results = mv.search_job_postings("Gen AI")

    assert [r["url"] for r in results] == ["https://wellfound.com/jobs/12345-ai-engineer"]


def test_results_are_interleaved_across_domains(monkeypatch):
    """A pooled search is dominated by whichever board has most content.
    Interleaving is what guarantees the smaller boards appear at all."""
    monkeypatch.setattr(mv, "tavily_client", FakeTavily({
        "greenhouse.io": [posting("https://g.com/1"), posting("https://g.com/2")],
        "lever.co": [posting("https://l.com/1"), posting("https://l.com/2")],
    }))

    urls = [r["url"] for r in mv.search_job_postings("Gen AI")]

    assert urls == ["https://g.com/1", "https://l.com/1",
                    "https://g.com/2", "https://l.com/2"]


def test_cap_takes_a_slice_across_domains_not_just_the_first(monkeypatch):
    monkeypatch.setattr(mv, "tavily_client", FakeTavily({
        "greenhouse.io": [posting(f"https://g.com/{i}") for i in range(4)],
        "lever.co": [posting(f"https://l.com/{i}") for i in range(4)],
    }))

    results = mv.search_job_postings("Gen AI", max_total=4)

    assert len(results) == 4
    hosts = {r["url"].split("/")[2] for r in results}
    assert hosts == {"g.com", "l.com"}, "the cap must not starve a domain"


def test_one_failing_domain_does_not_kill_the_search(monkeypatch):
    monkeypatch.setattr(mv, "tavily_client", FakeTavily(
        {
            "greenhouse.io": [posting("https://g.com/1")],
            "lever.co": [posting("https://l.com/1")],
        },
        failing_domains=["greenhouse.io"],
    ))

    results = mv.search_job_postings("Gen AI")

    assert [r["url"] for r in results] == ["https://l.com/1"]


def test_every_domain_is_searched_separately(monkeypatch):
    fake = FakeTavily({})
    monkeypatch.setattr(mv, "tavily_client", fake)

    mv.search_job_postings("Gen AI")

    assert fake.queries == mv.MARKET_DOMAINS


def test_no_postings_found_returns_empty(monkeypatch):
    monkeypatch.setattr(mv, "tavily_client", FakeTavily({}))

    assert mv.search_job_postings("Gen AI") == []


# ---------------------------------------------------------------------------
# extract_market_skills -- the counting
# ---------------------------------------------------------------------------

def stub_extraction(monkeypatch, skills_by_url, groups=()):
    monkeypatch.setattr(
        mv, "extract_skills_from_posting",
        lambda role, p, **kw: skills_by_url[p["url"]],
    )
    monkeypatch.setattr(mv, "consolidate_skill_mentions", lambda mentions: list(groups))


def test_mention_count_is_the_number_of_postings(monkeypatch, no_sleep):
    stub_extraction(monkeypatch, {
        "A": ["Python", "Spark"],
        "B": ["Python"],
        "C": ["Docker"],
    })

    skills = mv.extract_market_skills("Gen AI", [posting("A"), posting("B"), posting("C")])
    counts = {s["skill_name"]: s["mention_count"] for s in skills}

    assert counts == {"Python": 2, "Spark": 1, "Docker": 1}


def test_a_skill_named_twice_in_one_posting_counts_once(monkeypatch, no_sleep):
    """mention_count answers "how many postings ask for this", not "how
    many times was the word written"."""
    stub_extraction(monkeypatch, {"A": ["Python", "Python", "Python"]})

    skills = mv.extract_market_skills("Gen AI", [posting("A")])

    assert skills[0]["mention_count"] == 1


def test_counting_is_case_insensitive(monkeypatch, no_sleep):
    stub_extraction(monkeypatch, {"A": ["Python"], "B": ["python"], "C": ["PYTHON"]})

    skills = mv.extract_market_skills("Gen AI", [posting("A"), posting("B"), posting("C")])

    assert len(skills) == 1
    assert skills[0]["mention_count"] == 3


def test_aliases_are_counted_as_one_skill(monkeypatch, no_sleep):
    stub_extraction(
        monkeypatch,
        {"A": ["K8s"], "B": ["Kubernetes"]},
        groups=[{"canonical_name": "Kubernetes", "aliases_seen": ["K8s", "Kubernetes"]}],
    )

    skills = mv.extract_market_skills("Gen AI", [posting("A"), posting("B")])

    assert len(skills) == 1
    assert skills[0]["skill_name"] == "Kubernetes"
    assert skills[0]["mention_count"] == 2


def test_skills_with_no_alias_group_still_appear(monkeypatch, no_sleep):
    """consolidate only returns groups of 2+, so singletons are added in
    Python -- if that loop breaks, most skills silently vanish."""
    stub_extraction(
        monkeypatch,
        {"A": ["K8s", "Airflow"], "B": ["Kubernetes"]},
        groups=[{"canonical_name": "Kubernetes", "aliases_seen": ["K8s", "Kubernetes"]}],
    )

    skills = mv.extract_market_skills("Gen AI", [posting("A"), posting("B")])
    names = {s["skill_name"] for s in skills}

    assert names == {"Kubernetes", "Airflow"}


def test_source_urls_are_recorded_for_every_skill(monkeypatch, no_sleep):
    stub_extraction(monkeypatch, {"A": ["Python"], "B": ["Python"]})

    skills = mv.extract_market_skills("Gen AI", [posting("A"), posting("B")])

    assert skills[0]["source_urls"] == ["A", "B"]
    assert len(skills[0]["source_urls"]) == skills[0]["mention_count"]


def test_every_skill_gets_an_id(monkeypatch, no_sleep):
    stub_extraction(monkeypatch, {"A": ["Python", "Spark"]})

    skills = mv.extract_market_skills("Gen AI", [posting("A")])

    assert [s["id"] for s in skills] == ["mkt_001", "mkt_002"]


def test_progress_callback_reports_each_posting(monkeypatch, no_sleep):
    stub_extraction(monkeypatch, {"A": [], "B": [], "C": []})
    seen = []

    mv.extract_market_skills(
        "Gen AI",
        [posting("A"), posting("B"), posting("C")],
        progress_callback=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]


# ---------------------------------------------------------------------------
# compare_to_roadmap -- reattaching real numbers to the model's matches
# ---------------------------------------------------------------------------

ROADMAP = [{"id": "skl_001", "name": "Vector Databases"}]

MARKET = [
    {"skill_name": "Chroma", "mention_count": 2, "source_urls": ["u1", "u2"]},
    {"skill_name": "Pinecone", "mention_count": 2, "source_urls": ["u2", "u3"]},
]


def test_confirmed_unions_postings_instead_of_summing_them(monkeypatch):
    """One posting naming both Chroma and Pinecone is ONE posting asking
    for vector databases, not two. Summing would report 4 postings when
    only 3 exist."""
    monkeypatch.setattr(mv, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: groq_reply({
            "confirmed": [{
                "roadmap_skill_id": "skl_001",
                "roadmap_skill_name": "Vector Databases",
                "matched_market_skills": ["Chroma", "Pinecone"],
            }],
        })))
    ))

    result = mv.compare_to_roadmap(ROADMAP, MARKET)
    confirmed = result["confirmed"][0]

    assert confirmed["mention_count"] == 3, "u1, u2, u3 -- not 2 + 2"
    assert confirmed["source_urls"] == ["u1", "u2", "u3"]


def test_counts_come_from_python_not_from_the_model(monkeypatch):
    """If the model invents a count, it must be overwritten, not trusted."""
    monkeypatch.setattr(mv, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: groq_reply({
            "confirmed": [{
                "roadmap_skill_id": "skl_001",
                "roadmap_skill_name": "Vector Databases",
                "matched_market_skills": ["Chroma"],
                "mention_count": 99,
            }],
        })))
    ))

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert result["confirmed"][0]["mention_count"] == 2


def test_a_match_the_model_invented_is_ignored(monkeypatch):
    monkeypatch.setattr(mv, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: groq_reply({
            "confirmed": [{
                "roadmap_skill_id": "skl_001",
                "roadmap_skill_name": "Vector Databases",
                "matched_market_skills": ["Chroma", "Nonexistent DB"],
            }],
        })))
    ))

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert result["confirmed"][0]["mention_count"] == 2, "unknown names contribute nothing"


def test_suggested_additions_get_their_real_counts(monkeypatch):
    monkeypatch.setattr(mv, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: groq_reply({
            "suggested_additions": [{"skill_name": "Pinecone"}],
        })))
    ))

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert result["suggested_additions"][0]["mention_count"] == 2
    assert result["suggested_additions"][0]["source_urls"] == ["u2", "u3"]


def test_all_three_buckets_always_exist(monkeypatch):
    """The model omits keys for empty buckets; app.py would KeyError."""
    monkeypatch.setattr(mv, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: groq_reply({"confirmed": []})
        ))
    ))

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert set(result) >= {"confirmed", "suggested_additions", "weak_signal"}


def test_invalid_json_returns_empty_buckets_not_a_crash(monkeypatch):
    monkeypatch.setattr(mv, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: groq_reply("this is not json")
        ))
    ))

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert result == {"confirmed": [], "suggested_additions": [], "weak_signal": []}


def test_consolidation_survives_invalid_json(monkeypatch):
    monkeypatch.setattr(mv, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: groq_reply("not json either")
        ))
    ))

    assert mv.consolidate_skill_mentions([{"skill_name": "Python", "source_url": "A"}]) == []
