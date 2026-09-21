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

from personaltracker import llm
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

    def __init__(self, results_by_domain, failing_domains=(), extracts=None,
                 extract_fails=False):
        self.results_by_domain = results_by_domain
        self.failing_domains = set(failing_domains)
        self.queries = []
        # what extract() can recover, keyed by url; and a record of what
        # it was asked for, so a test can assert it wasn't called at all
        self.extracts = extracts or {}
        self.extract_fails = extract_fails
        self.extracted = []

    def search(self, **kwargs):
        domain = kwargs["include_domains"][0]
        self.queries.append(domain)
        if domain in self.failing_domains:
            raise RuntimeError(f"boom: {domain}")
        return {"results": self.results_by_domain.get(domain, [])}

    def extract(self, urls, **kwargs):
        self.extracted.append(list(urls))
        if self.extract_fails:
            raise RuntimeError("boom: extract")
        return {"results": [{"url": u, "raw_content": self.extracts.get(u, "")}
                            for u in urls]}


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


def test_wellfound_listing_pages_are_excluded_whatever_their_path(monkeypatch):
    """/role/r/ was not the only listing shape. A Backend Engineer search
    returned /hire/back-end-developers -- a marketing page that put
    PyTorch, TensorFlow and Power BI into a backend role's market skills.
    Only /jobs/ is one posting about one job, so that is what we keep."""
    monkeypatch.setattr(mv, "tavily_client", FakeTavily({
        "wellfound.com": [
            posting("https://wellfound.com/hire/back-end-developers"),
            posting("https://wellfound.com/role/r/backend-engineer"),
            posting("https://wellfound.com/company/acme/openings"),
            posting("https://wellfound.com/jobs/4584948-backend-engineer"),
        ],
    }))

    results = mv.search_job_postings("Backend Engineer")

    assert [r["url"] for r in results] == [
        "https://wellfound.com/jobs/4584948-backend-engineer"
    ]


def test_other_boards_are_not_path_filtered(monkeypatch):
    """The /jobs/ rule is about Wellfound only. Greenhouse, Lever, Ashby
    and Workable serve one posting per URL, so their paths are left
    alone -- Lever's, for instance, carries no /jobs/ segment at all."""
    monkeypatch.setattr(mv, "tavily_client", FakeTavily({
        "lever.co": [posting("https://jobs.lever.co/acme/abc-123")],
        "jobs.ashbyhq.com": [posting("https://jobs.ashbyhq.com/acme/def-456")],
    }))

    results = mv.search_job_postings("Backend Engineer")

    assert [r["url"] for r in results] == [
        "https://jobs.lever.co/acme/abc-123",
        "https://jobs.ashbyhq.com/acme/def-456",
    ]


def test_a_page_the_search_returned_empty_is_fetched_again(monkeypatch):
    """Tavily often returns raw_content blank for boards that build their
    postings in the browser. A "Graphics Programmer" search lost 8 of 12
    postings that way; extract() recovered most of them."""
    fake = FakeTavily(
        {"lever.co": [posting("https://jobs.lever.co/a/1", raw_content=""),
                      posting("https://jobs.lever.co/a/2")]},
        extracts={"https://jobs.lever.co/a/1": "Vulkan, DX12 and RenderDoc"},
    )
    monkeypatch.setattr(mv, "tavily_client", fake)

    results = mv.search_job_postings("Graphics Programmer")

    assert fake.extracted == [["https://jobs.lever.co/a/1"]]   # only the missing one
    assert results[0]["raw_content"] == "Vulkan, DX12 and RenderDoc"
    assert all(mv.was_fetched(r) for r in results)


def test_nothing_is_re_fetched_when_every_page_arrived(monkeypatch):
    """extract() costs credits, so it only runs when a page is missing."""
    fake = FakeTavily({"lever.co": [posting("https://jobs.lever.co/a/1")]})
    monkeypatch.setattr(mv, "tavily_client", fake)

    mv.search_job_postings("Graphics Programmer")

    assert fake.extracted == []


def test_a_failed_re_fetch_still_returns_the_postings_that_arrived(monkeypatch):
    """A failed repair is not a failed search -- the pages that did come
    back are still worth checking, and was_fetched drops the rest."""
    fake = FakeTavily(
        {"lever.co": [posting("https://jobs.lever.co/a/1", raw_content=""),
                      posting("https://jobs.lever.co/a/2")]},
        extract_fails=True,
    )
    monkeypatch.setattr(mv, "tavily_client", fake)

    results = mv.search_job_postings("Graphics Programmer")

    assert [r["url"] for r in results] == ["https://jobs.lever.co/a/1",
                                           "https://jobs.lever.co/a/2"]
    assert [r["url"] for r in results if mv.was_fetched(r)] == ["https://jobs.lever.co/a/2"]


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
        mv, "extract_skills_from_postings",
        lambda role, batch, **kw: {p["url"]: skills_by_url[p["url"]] for p in batch},
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
# extract_market_skills -- the posting cache
# ---------------------------------------------------------------------------

def counting_extraction(monkeypatch, skills, calls):
    def extract(role, batch, **kw):
        for p in batch:
            calls.append(p["url"])
        return {p["url"]: list(skills) for p in batch}

    monkeypatch.setattr(mv, "extract_skills_from_postings", extract)
    monkeypatch.setattr(mv, "consolidate_skill_mentions", lambda mentions: [])


def test_a_posting_read_before_is_not_sent_to_groq_again(monkeypatch, no_sleep):
    """A job posting is a fixed document, unlike a roadmap, which is the
    model's opinion. Reading the same page twice costs a call and cannot
    produce a better answer."""
    calls = []
    counting_extraction(monkeypatch, ["Python", "Docker"], calls)

    mv.extract_market_skills("Gen AI", [posting("https://x.com/jobs/1")])
    again = mv.extract_market_skills("Gen AI", [posting("https://x.com/jobs/1")])

    assert calls == ["https://x.com/jobs/1"]
    assert {s["skill_name"] for s in again} == {"Python", "Docker"}
    assert {s["mention_count"] for s in again} == {1}


def test_the_cache_ignores_http_https_and_capitalisation(monkeypatch, no_sleep):
    """Tavily returns the same posting under http:// and https://, and one
    run gave .../MachinaLabs/... where the next gave .../machinalabs/....
    Keying on the raw URL would re-pay for a page already read."""
    calls = []
    counting_extraction(monkeypatch, ["Rust"], calls)

    mv.extract_market_skills("Gen AI", [posting("https://jobs.lever.co/MachinaLabs/7")])
    mv.extract_market_skills("Gen AI", [posting("http://jobs.lever.co/machinalabs/7/")])

    assert calls == ["https://jobs.lever.co/MachinaLabs/7"]


def test_changing_the_prompt_retires_the_cached_answers(monkeypatch, no_sleep):
    """The prompt decides what counts as a skill, so an answer written
    under an older one is not an answer to the current question. Without
    this, improving the prompt would change nothing already cached."""
    calls = []
    counting_extraction(monkeypatch, ["Go"], calls)

    mv.extract_market_skills("Gen AI", [posting("https://x.com/jobs/1")])
    monkeypatch.setattr(mv, "EXTRACTION_VERSION", "a-different-prompt")
    mv.extract_market_skills("Gen AI", [posting("https://x.com/jobs/1")])

    assert calls == ["https://x.com/jobs/1", "https://x.com/jobs/1"]


def test_an_empty_result_is_never_cached(monkeypatch, no_sleep):
    """extract_skills_from_postings returns nothing both for a posting that
    names no technology and for a reply it could not read. The two are
    indistinguishable here, so caching one would turn a single bad reply
    into a permanent "this posting asks for nothing"."""
    calls = []
    counting_extraction(monkeypatch, [], calls)

    mv.extract_market_skills("Gen AI", [posting("https://x.com/jobs/1")])
    mv.extract_market_skills("Gen AI", [posting("https://x.com/jobs/1")])

    assert calls == ["https://x.com/jobs/1", "https://x.com/jobs/1"]


# ---------------------------------------------------------------------------
# extract_skills_from_postings -- batching, and who said what
# ---------------------------------------------------------------------------

def test_postings_are_read_in_batches_not_one_at_a_time(monkeypatch, no_sleep):
    """The instructions are ~460 tokens and identical for every posting,
    so sending them once per posting was most of the bill."""
    batches = []

    def extract(role, batch, **kw):
        batches.append([p["url"] for p in batch])
        return {p["url"]: ["Python"] for p in batch}

    monkeypatch.setattr(mv, "extract_skills_from_postings", extract)
    monkeypatch.setattr(mv, "consolidate_skill_mentions", lambda mentions: [])
    monkeypatch.setattr(mv, "BATCH_SIZE", 4)

    nine = [posting(f"https://x.com/jobs/{n}") for n in range(9)]
    mv.extract_market_skills("Gen AI", nine)

    assert sorted((len(b) for b in batches), reverse=True) == [4, 4, 1]
    assert sum(len(b) for b in batches) == 9      # every posting read exactly once


def test_each_posting_keeps_only_its_own_skills(monkeypatch, fake_groq):
    """The whole risk of sharing a call. If the model's numbering slips,
    a skill lands on a posting that never mentioned it and every
    mention_count downstream is wrong, with nothing to show for it."""
    monkeypatch.setattr(llm, "groq_client", fake_groq(
        {"postings": {"1": ["Rust"], "2": ["Kafka", "Redis"]}}
    ))

    found = mv.extract_skills_from_postings(
        "Backend Engineer", [posting("https://a.com/1"), posting("https://b.com/2")]
    )

    assert found == {"https://a.com/1": ["Rust"], "https://b.com/2": ["Kafka", "Redis"]}


def test_a_posting_the_model_skips_is_left_out_not_guessed(monkeypatch, fake_groq):
    """Groq answered for posting 1 and said nothing about posting 2.
    Posting 2 contributes nothing -- exactly as an unreadable single
    posting used to. Filling it in from its neighbour would invent
    evidence about a job nobody read."""
    monkeypatch.setattr(llm, "groq_client", fake_groq({"postings": {"1": ["Rust"]}}))

    found = mv.extract_skills_from_postings(
        "Backend Engineer", [posting("https://a.com/1"), posting("https://b.com/2")]
    )

    assert found == {"https://a.com/1": ["Rust"]}


def test_a_number_that_was_never_sent_is_dropped(monkeypatch, fake_groq):
    """Posting 7 does not exist in a batch of one. There is no posting to
    attach those skills to, so they are dropped rather than landing on
    whatever happens to be at that index."""
    monkeypatch.setattr(llm, "groq_client", fake_groq(
        {"postings": {"1": ["Rust"], "7": ["Invented"]}}
    ))

    found = mv.extract_skills_from_postings("Backend Engineer", [posting("https://a.com/1")])

    assert found == {"https://a.com/1": ["Rust"]}


def test_an_unreadable_batch_reply_loses_the_batch_not_the_check(monkeypatch, fake_groq):
    """A reply that isn't the right shape skips those postings and lets
    the rest of the market check carry on -- the same trade the
    per-posting version made, which is why BATCH_SIZE stays small."""
    monkeypatch.setattr(llm, "groq_client", fake_groq({"wrong": "shape"}))

    found = mv.extract_skills_from_postings(
        "Backend Engineer", [posting("https://a.com/1"), posting("https://b.com/2")]
    )

    assert found == {}


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
    monkeypatch.setattr(llm, "groq_client", SimpleNamespace(
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
    monkeypatch.setattr(llm, "groq_client", SimpleNamespace(
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
    monkeypatch.setattr(llm, "groq_client", SimpleNamespace(
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
    monkeypatch.setattr(llm, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: groq_reply({
            "suggested_additions": [{"skill_name": "Pinecone"}],
        })))
    ))

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert result["suggested_additions"][0]["mention_count"] == 2
    assert result["suggested_additions"][0]["source_urls"] == ["u2", "u3"]


def test_all_three_buckets_always_exist(monkeypatch):
    """The model omits keys for empty buckets; app.py would KeyError."""
    monkeypatch.setattr(llm, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: groq_reply({"confirmed": []})
        ))
    ))

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert set(result) >= {"confirmed", "suggested_additions", "weak_signal"}


def test_invalid_json_returns_empty_buckets_not_a_crash(monkeypatch):
    monkeypatch.setattr(llm, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: groq_reply("this is not json")
        ))
    ))

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert result == {"confirmed": [], "suggested_additions": [], "weak_signal": []}


def test_consolidation_survives_invalid_json(monkeypatch):
    monkeypatch.setattr(llm, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: groq_reply("not json either")
        ))
    ))

    assert mv.consolidate_skill_mentions([{"skill_name": "Python", "source_url": "A"}]) == []


# ---------------------------------------------------------------------------
# get_market_validation -- searches for the saved title, never asks Groq
# ---------------------------------------------------------------------------

def test_market_check_uses_the_title_it_is_given_without_asking_groq(monkeypatch):
    """The roadmap was built for the title saved in role_aliases. Asking
    Groq again costs a request and could return a different job title."""

    def groq_must_not_be_called(**kwargs):
        raise AssertionError("the market check asked Groq for a job title")

    no_groq = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=groq_must_not_be_called))
    )
    monkeypatch.setattr(llm, "groq_client", no_groq)  # every Groq call goes through llm

    searched_for = []

    def fake_search(title):
        searched_for.append(title)
        return []  # no postings, so the check stops before reading any

    monkeypatch.setattr(mv, "search_job_postings", fake_search)

    mv.get_market_validation("Generative AI Engineer", ROADMAP)

    assert searched_for == ["Generative AI Engineer"]


# ---------------------------------------------------------------------------
# compare_to_roadmap -- Python decides the buckets, not the model
# ---------------------------------------------------------------------------

TWO_ROADMAP_SKILLS = ROADMAP + [{"id": "skl_002", "name": "Prompt Engineering"}]


def model_answers(monkeypatch, payload):
    """Make compare_to_roadmap's one Groq call return `payload`."""
    monkeypatch.setattr(llm, "groq_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: groq_reply(payload)))
    ))


def test_a_skill_the_model_forgot_is_still_listed_as_not_confirmed(monkeypatch):
    """The model confirmed one skill and simply left the other one out."""
    model_answers(monkeypatch, {
        "confirmed": [{"roadmap_skill_id": "skl_001", "roadmap_skill_name": "Vector Databases",
                       "matched_market_skills": ["Chroma"]}],
        "weak_signal": [],
    })

    result = mv.compare_to_roadmap(TWO_ROADMAP_SKILLS, MARKET)

    assert [w["roadmap_skill_id"] for w in result["weak_signal"]] == ["skl_002"]


def test_a_skill_is_never_both_confirmed_and_not_confirmed(monkeypatch):
    model_answers(monkeypatch, {
        "confirmed": [{"roadmap_skill_id": "skl_001", "roadmap_skill_name": "Vector Databases",
                       "matched_market_skills": ["Chroma"]}],
        "weak_signal": [{"roadmap_skill_id": "skl_001", "roadmap_skill_name": "Vector Databases"}],
    })

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert result["weak_signal"] == []


def test_a_confirmation_with_no_real_market_skill_is_dropped(monkeypatch):
    """Confirmed by nothing real would read as 'mentioned in 0 of 12 postings'."""
    model_answers(monkeypatch, {
        "confirmed": [{"roadmap_skill_id": "skl_001", "roadmap_skill_name": "Vector Databases",
                       "matched_market_skills": ["Nonexistent DB"]}],
    })

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert result["confirmed"] == []
    assert [w["roadmap_skill_id"] for w in result["weak_signal"]] == ["skl_001"]


def test_a_confirmation_for_a_skill_not_in_the_roadmap_is_dropped(monkeypatch):
    model_answers(monkeypatch, {
        "confirmed": [{"roadmap_skill_id": "skl_999", "roadmap_skill_name": "Made Up",
                       "matched_market_skills": ["Chroma"]}],
    })

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert result["confirmed"] == []


def test_a_suggested_skill_the_model_made_up_is_dropped(monkeypatch):
    """It would otherwise show under 'Mentioned in only one posting'."""
    model_answers(monkeypatch, {
        "suggested_additions": [{"skill_name": "Made Up Tool"}, {"skill_name": "Pinecone"}],
    })

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert [s["skill_name"] for s in result["suggested_additions"]] == ["Pinecone"]


def test_a_skill_backing_a_confirmation_is_not_also_suggested(monkeypatch):
    """Chroma confirms Vector Databases, so the roadmap isn't missing it."""
    model_answers(monkeypatch, {
        "confirmed": [{"roadmap_skill_id": "skl_001", "roadmap_skill_name": "Vector Databases",
                       "matched_market_skills": ["Chroma"]}],
        "suggested_additions": [{"skill_name": "Chroma"}, {"skill_name": "Pinecone"}],
    })

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert [s["skill_name"] for s in result["suggested_additions"]] == ["Pinecone"]


def test_a_suggested_skill_is_listed_only_once(monkeypatch):
    model_answers(monkeypatch, {
        "suggested_additions": [{"skill_name": "Pinecone"}, {"skill_name": "Pinecone"}],
    })

    result = mv.compare_to_roadmap(ROADMAP, MARKET)

    assert [s["skill_name"] for s in result["suggested_additions"]] == ["Pinecone"]
