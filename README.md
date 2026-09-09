# NextRole

An AI career coach that answers four questions about a target job role:
**what should I learn, does my CV show it, am I making progress, and does
the market actually ask for this?**

Give it a role like "Backend Engineer" or "Gen AI". It builds a skill
roadmap grounded in real web sources, checks your CV against that roadmap
skill by skill, tracks your progress in a local database, and then
validates the whole roadmap against **live job postings** — so every claim
about what employers want is backed by real URLs you can open yourself.

> The repo is named `Personal-Tracker`; the app is **NextRole**.

---

## What it does

**1. Skill roadmap, grounded in sources**

Vague input is rewritten into a real job title first ("Gen AI" →
"Generative AI Engineer"), because searching for a *field* returns
"AI skills for everyone" listicles instead of what a *job* requires. The
roadmap is then synthesized only from fetched sources, and every skill
carries the `source_url` it came from.

**2. CV gap analysis**

Upload a PDF CV and each roadmap skill is marked `missing` / `partial` /
`met`, with a paraphrase of the supporting CV evidence and a concrete
next step. Text-based PDFs are parsed directly; scanned CVs fall back to
OCR automatically — detected by measuring how much text came back, not by
catching an exception, because a scanned PDF doesn't error, it just
returns almost nothing.

**3. Progress tracking**

Each skill gets a status (`not_started` / `in_progress` / `completed`)
persisted to SQLite. Progress is scoped per role, so tracking two career
paths doesn't merge them into one list.

**4. Market validation**

The roadmap is LLM-generated, which makes it plausible — not verified.
This checks it against real, currently-open job postings and sorts every
skill into three buckets:

| Bucket | Meaning |
|---|---|
| **Confirmed** | Roadmap skill found in real postings, with the count and source URLs |
| **Suggested additions** | Skills employers ask for that the roadmap missed |
| **Not confirmed** | Roadmap skills this batch of postings didn't back up |

Every count is traceable to specific postings. Nothing is asserted
without a URL behind it.

---

## How the market check works

The interesting part isn't the LLM calls — it's the division of labour
between the model and Python.

```
search_job_postings()           one Tavily search per job board, interleaved,
        │                       deduplicated by normalized URL
        ▼
extract_skills_from_posting()   one LLM call per posting → skill names only
        │                       (deliberately: no counting)
        ▼
consolidate_skill_mentions()    one LLM call → groups alias variants
        │                       ("K8s"/"Kubernetes"), reports no numbers
        ▼
extract_market_skills()         Python counts distinct source URLs per group
        │
        ▼
compare_to_roadmap()            one LLM call decides WHICH market skills match
                                which roadmap skills; Python reattaches the
                                real counts and URLs
```

**The model never counts anything.** The first version asked a single call
to count skill frequency across eight concatenated job descriptions. It
returned `mention_count: 1` for everything — because that is not a task
language models do reliably. The fix was to split the work by what each
side is genuinely good at: the model reads one document at a time and
extracts names, Python adds the integers. Counts are now exact by
construction rather than by hope.

---

## Setup

**Prerequisites**

- Python 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- API keys for [Groq](https://console.groq.com) and [Tavily](https://tavily.com) — both have free tiers
- *Optional, only for scanned-PDF CVs:* [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) installed and on `PATH`. Ordinary text-based PDFs work without them.

**Install and run**

```bash
git clone https://github.com/Mauryaraheja/Personal-Tracker.git
cd Personal-Tracker
uv sync
```

Then create your `.env` from the template and add both keys:

```bash
cp .env.example .env
```

```bash
uv run streamlit run app.py
```

The SQLite database is created automatically on first use, tables
included. It's gitignored — your roadmaps and progress stay local.

---

## Project structure

```
src/personaltracker/
├── clients.py            # shared Groq + Tavily clients, constructed once
├── roadmap.py            # role refinement, source search, roadmap generation
├── cv_parser.py          # PDF text extraction with OCR fallback
├── gap_analysis.py       # CV vs. roadmap comparison
├── tracker.py            # SQLite persistence for skills and progress
└── market_validation.py  # live job-posting validation pipeline
app.py                    # Streamlit UI
```

Two tables, deliberately kept separate so a skill's description lives in
exactly one place instead of being duplicated into every progress row:

```sql
skills        (role, skill_id, name, description, why_it_matters,
               priority, level_required, source_url)

tracker_items (id, role, skill_id, status, notes, updated_at)
```

`tracker_items` is keyed on `(role, skill_id)` rather than a generated
id, because that pair *is* the real uniqueness rule: one progress row per
skill per role.

---

## Design decisions

**Raw API calls — no LangChain, no ORM.** Every Groq call and every SQL
statement is hand-written. At this scale the abstractions would cost more
in indirection than they save in code, and writing them by hand means
being able to explain exactly what happens on every call.

**Roadmaps are cached per role; market data never is.** LLM output isn't
deterministic, so regenerating a roadmap can silently change the skill
IDs your saved progress points at — which would quietly corrupt the
tracker. Roadmaps are therefore written to SQLite once and reused
forever. Job postings are the opposite case: they go stale in weeks, so
the market check always runs fresh, on demand, and is never folded into
the cached object.

**A curated allowlist of job boards, not open web search.** Postings come
only from Greenhouse, Lever, Ashby, Workable, and Wellfound — platforms
individually verified to serve real single-posting content. The honest
trade-off: this can't reach university career portals or companies' own
custom career pages. That's a chosen scope, not an oversight.

**LinkedIn, Glassdoor, and Indeed are excluded.** The first two on
terms-of-service and consent grounds — having a search API fetch them on
your behalf raises the same problem as scraping them directly. Indeed was
tested and dropped on quality: it returns hiring guides and
job-description *templates* rather than real postings.

**Matching is deliberately conservative.** "Deep Learning" appearing in a
posting does not confirm a roadmap skill of "Model Fine-Tuning of LLMs" —
related is not equivalent. The consequence is that **Confirmed** looks
sparse on a small sample. That's the intended trade: a sparse honest
result beats a full-looking one built on loose matches.

---

## Known limitations

Stated rather than hidden:

- **Results vary between runs.** Different posting samples plus
  non-deterministic matching mean a skill confirmed at 4 mentions in one
  run can fall to "not confirmed" in the next. The UI describes its own
  output as a directional signal, not a market survey.
- **Sample size is small** — roughly 12 postings per check.
- **Groq's free tier caps at 8,000 tokens/minute**, so a full market check
  is deliberately throttled and takes ~1.5–2 minutes. That's a tier
  constraint, not a code problem.
- **Occasional loose matches survive** in the Confirmed bucket, and alias
  consolidation sometimes picks a version-specific canonical name
  ("GPT-4" over "GPT"). Counts stay correct; labels are imperfect.
- **PDF only** for CV upload.

---

## Roadmap

- [x] Sourced skill roadmap generation
- [x] CV parsing with OCR fallback
- [x] Gap analysis
- [x] SQLite progress tracking
- [x] Live job-posting market validation
- [ ] Test suite for the pure-Python logic — URL dedup, count computation, match reattachment
- [ ] Mock interview loop with a scoring rubric and per-answer feedback
- [ ] Multi-user support with authentication
- [ ] Deployment

---

## Status

Built as a portfolio project, with the explicit goal of understanding
every decision rather than shipping fast. The reasoning above is as much
the point as the code is.
