"""
Market validation: search real, live job postings for a role and extract
the technical skills actually mentioned in them, then compare that
against the LLM-generated roadmap to surface skills the roadmap might be
missing (or skills it has that this batch of postings doesn't back up).

Deliberately NOT wired into get_or_create_roadmap()'s cache-or-generate
flow. The roadmap is evergreen and correctly cached in SQLite; job
postings go stale in weeks, not months, so mixing the two into one
cached object would misrepresent how fresh the market data actually is.

This runs as an explicit, separate action instead (see app.py).

Searches trusted job-posting sources (currently Greenhouse, Lever, Ashby,
Workable, and Wellfound) to minimize blog articles and hiring guides
while keeping the results focused on real job descriptions. Wellfound
hosts both genuine single postings (/jobs/{id}) and aggregator/category
pages (/role/r/{slug}) on the same domain -- the aggregator shape is
filtered out by URL path below, since include_domains only restricts by
domain and can't distinguish the two.

LinkedIn and Glassdoor are deliberately excluded -- having Tavily fetch
their pages on our behalf raises the same ToS/consent problem as
scraping them directly.
"""

import json
from .clients import groq_client, tavily_client
from .roadmap import refine_role
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from groq import RateLimitError

DEBUG = False

MARKET_DOMAINS = [
    "greenhouse.io",
    "lever.co",
    "jobs.ashbyhq.com",
    "apply.workable.com",
    "wellfound.com"
]


def search_job_postings(role: str, per_domain: int = 4, max_total: int = 12) -> list[dict]:
    """Fetch real, live job postings for a role -- one search per domain
    rather than one pooled search across all of them.

    Tavily ranks a pooled multi-domain search by relevance alone, and
    Greenhouse has far more indexed postings than the others, so a single
    call returns almost entirely Greenhouse results and the smaller
    platforms never appear. Searching each domain separately guarantees
    every source actually gets represented.

    Results are interleaved round-robin before the total cap is applied,
    so trimming to max_total takes a slice across all domains rather than
    exhausting the first domain's results first.

    Excludes Wellfound's aggregator/category pages (/role/r/{slug}) --
    those list multiple unrelated roles on one page, unlike the genuine
    single-posting pages (/jobs/{id}) on the same domain.

    Deduplicates by normalized URL -- Tavily can return the same posting
    twice under http:// vs https:// or with a different query string,
    which would otherwise silently inflate mention_count downstream."""

    query = f"{role} job opening responsibilities requirements qualifications"
    per_domain_results = []

    for domain in MARKET_DOMAINS:
        try:
            response = tavily_client.search(
                query=query,
                max_results=per_domain,
                search_depth="advanced",
                include_domains=[domain],
                include_domains_mode="filter",
                include_raw_content="text",
            )
            per_domain_results.append(response.get("results", []))
        except Exception as e:
            # One bad domain shouldn't kill the whole market check --
            # log it and carry on with whatever the others return.
            print(f"Search failed for {domain}: {e}")
            per_domain_results.append([])

    # Round-robin interleave: first result from each domain, then the
    # second from each, and so on.
    interleaved = []
    for i in range(per_domain):
        for domain_results in per_domain_results:
            if i < len(domain_results):
                interleaved.append(domain_results[i])

    interleaved = [
        r for r in interleaved if "wellfound.com/role/" not in r.get("url", "")
    ]

    seen = set()
    deduped = []
    for r in interleaved:
        url = r.get("url", "")
        normalized = url.split("://", 1)[-1].split("?")[0].rstrip("/")
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(r)

    return deduped[:max_total]

def format_postings_for_prompt(postings: list[dict]) -> str:
    """Turn raw postings into text the LLM can read, one block per posting."""
    blocks = []
    for p in postings:
        snippet = p.get("content", "")[:1500]
        blocks.append(f"Source: {p.get('url')}\n{snippet}")
    return "\n\n".join(blocks)

def extract_skills_from_posting(role: str, posting: dict, max_retries: int = 3) -> list[str]:
    """
    Extract technical skills from ONE job posting.
    Returns only skill names.
    """

    text = posting.get("raw_content") or posting.get("content", "")

    prompt = f"""
    The text below is ONE real job posting for the role: {role}.

    Read the job description carefully and identify every technical skill,
    framework, library, programming language, database, cloud platform,
    machine learning framework, analytics tool, infrastructure tool,
    or technology that is explicitly mentioned.

    URL:
    {posting.get("url")}

    Content:
    {text[:10000]}

    Examples of the kinds of skills to extract include:
    - Python
    - SQL
    - PyTorch
    - TensorFlow
    - Scikit-learn
    - Spark
    - Hadoop
    - Airflow
    - dbt
    - Snowflake
    - Docker
    - Kubernetes
    - AWS
    - Azure
    - GCP
    - Tableau
    - Power BI

    Only include technologies that are explicitly mentioned in the text.
    Do NOT guess or infer technologies that are not written.
        Prefer specific, named technologies over broad categories -- extract
    "PyTorch" or "scikit-learn" rather than "Machine Learning"; extract
    "AWS" or "GCP" rather than "cloud computing". Only include a broad
    category term (e.g. "Machine Learning", "Internet of Things") if the
    posting calls it out as a specific requirement and does not name any
    more specific technology for it.

    Return JSON only in the following format:

    {{
        "skills": [
            "Python",
            "Spark",
            "AWS"
        ]
    }}
    """
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            break
        except RateLimitError:
            if attempt == max_retries - 1:
                raise  # out of retries -- let it surface, don't pretend it succeeded
            time.sleep(3 * (attempt + 1))  # 3s, 6s, 9s

    raw = response.choices[0].message.content

    if DEBUG:
        print("=" * 80)
        print(posting.get("url"))
        print(posting.get("content", "")[:5000])
        print("=" * 80)

    if DEBUG:
        print("\n========== RAW SKILL EXTRACTION ==========")
        print(raw)
        print("=========================================\n")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("Invalid JSON while extracting posting skills.")
        print(raw)
        return []

    return data.get("skills", [])

def consolidate_skill_mentions(raw_mentions: list[dict]) -> list[dict]:
    """
    Group different names that refer to the same technology.
    """

    unique_skills = sorted(
        {mention["skill_name"] for mention in raw_mentions}
    )

    prompt = f"""
    Below is a list of technical skill names extracted from job postings.

    Group names that refer to the same technology.

    For example:

    Spark -> Apache Spark

    K8s -> Kubernetes

    Amazon Web Services -> AWS

    Return JSON only.

    {{
        "skill_groups": [
            {{
                "canonical_name": "Apache Spark",
                "aliases_seen": [
                    "Spark",
                    "Apache Spark"
                ]
            }}
        ]
    }}

    Skills:

    {json.dumps(unique_skills, indent=2)}
    """

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(raw)
        return []

    return data.get("skill_groups", [])


def extract_market_skills(role: str, postings: list[dict], progress_callback=None) -> list[dict]:
    """
    Extract skills from each posting individually.
    Extract technical skills from multiple job postings,
    consolidate aliases, and compute mention counts based
    on distinct postings. """
      
    raw_mentions = []
    completed = 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_posting = {
            executor.submit(extract_skills_from_posting, role, posting): posting
            for posting in postings
        }

        for future in as_completed(future_to_posting):
            posting = future_to_posting[future]
            completed += 1
            if progress_callback:
                progress_callback(completed, len(postings))

            skills = future.result()

            if DEBUG:
                print(f"\n{posting['url']}")
                print(skills)

            for skill in skills:
                raw_mentions.append({"skill_name": skill, "source_url": posting["url"]})


    if DEBUG:
        print("\n========== RAW MENTIONS ==========")
        print(raw_mentions)

    groups = consolidate_skill_mentions(raw_mentions)

    if DEBUG:
        print("\n========== GROUPS ==========")
        print(groups)  

    seen_aliases = {
    alias.strip().lower()
    for group in groups
    for alias in group["aliases_seen"]
}

    for mention in raw_mentions:
        if mention["skill_name"].strip().lower() not in seen_aliases:
            groups.append(
            {
                "canonical_name": mention["skill_name"],
                "aliases_seen": [mention["skill_name"]],
            }

        )
            seen_aliases.add(
        mention["skill_name"].strip().lower()
        )

    market_skills = []

    for group in groups:
        aliases = {
        alias.strip().lower()
        for alias in group["aliases_seen"]
        }

        matched_urls = {
        mention["source_url"]
        for mention in raw_mentions
        if mention["skill_name"].strip().lower() in aliases
    }

        market_skills.append(
        {
            "skill_name": group["canonical_name"],
            "mention_count": len(matched_urls),
            "source_urls": sorted(matched_urls),
        }
    )

    for i, skill in enumerate(market_skills, start=1):
        skill["id"] = f"mkt_{i:03d}"

    return market_skills


def compare_to_roadmap(roadmap_skills: list[dict], market_skills: list[dict]) -> dict:
    """Compare the LLM-generated roadmap against skills actually found in
    real job postings. Single call across the whole roadmap -- same
    reasoning as gap_analysis.get_skill_gaps: cheap at this scale, and
    lets the model reason about near-duplicate names consistently
    instead of skill-by-skill.

    A roadmap skill can be confirmed by MULTIPLE market skills (e.g.
    "Vector Databases" backed by Chroma, Pinecone, and Weaviate all at
    once) -- the model only decides WHICH market skills apply; the real
    mention_count and source_urls are computed in Python afterward from
    market_skills, not trusted from the model's own arithmetic."""

    roadmap_text = "\n".join(
        f"- id: {s['id']}, name: {s['name']}" for s in roadmap_skills
    )
    market_text = "\n".join(
        f"- {m['skill_name']} (mentioned in {m['mention_count']} postings)"
        for m in market_skills
    )

    prompt = f"""
    Roadmap skills (generated by an LLM from general web sources):
    {roadmap_text}

    Skills actually found in real job postings for this role:
    {market_text}

    Match these two lists semantically -- the same skill may be named
    differently in each, and a single roadmap skill can reasonably be
    confirmed by MORE THAN ONE market skill. For example, if the roadmap
    has "Vector Databases" and the market skills include "Chroma",
    "Pinecone", and "Weaviate", all three should be listed as backing
    that one roadmap skill -- do not pick only the closest single match.

    Being able to list multiple matches does NOT mean being loose about
    what counts as a match. Each individual market skill you list must be
    a genuinely closer or more specific name for the SAME underlying
    skill -- not merely a related or adjacent topic. For example, "Deep
    Learning" or "Large Language Models" should NOT be treated as
    confirming "Model Fine-Tuning of LLMs" just because they're in the
    same general domain -- fine-tuning itself would need to be explicitly
    named or clearly implied. When in doubt, leave a roadmap skill in
    weak_signal rather than force a loose match.

    Return a JSON object with exactly these three keys:

    "confirmed": roadmap skills backed up by one or more market skills.
    List of objects with "roadmap_skill_id", "roadmap_skill_name",
    "matched_market_skills" (a list of one or more market skill names).

    "suggested_additions": market skills with no close match in the
    roadmap -- real skills employers ask for that the roadmap is missing.
    List of objects with "skill_name".

    "weak_signal": roadmap skills with no market skill backing them up.
    This does NOT mean they're wrong -- postings often omit foundational
    or assumed skills -- just that this batch didn't confirm them. List
    of objects with "roadmap_skill_id", "roadmap_skill_name".

    Respond with JSON only, no extra text.
    """

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        print("The model didn't return valid JSON. Raw response:")
        print(raw_text)
        return {"confirmed": [], "suggested_additions": [], "weak_signal": []}

    market_by_name = {m["skill_name"]: m for m in market_skills}

    for item in data.get("confirmed", []):
        urls = set()
        for name in item.get("matched_market_skills", []):
            source = market_by_name.get(name)
            if source:
                urls.update(source["source_urls"])
        item["mention_count"] = len(urls)
        item["source_urls"] = sorted(urls)

    for item in data.get("suggested_additions", []):
        source = market_by_name.get(item.get("skill_name"))
        if source:
            item["mention_count"] = source["mention_count"]
            item["source_urls"] = source["source_urls"]

    return data


def get_market_validation(role: str, roadmap_skills: list[dict], progress_callback=None) -> dict:
    """Full pipeline: refine the role, pull real postings, extract skills
    actually asked for, compare against the existing roadmap. Called
    explicitly from the UI -- not part of get_or_create_roadmap()."""

    refined_role = refine_role(role)

    postings = search_job_postings(refined_role)
    if not postings:
        print("Warning: no job postings found -- market validation skipped.")
        return {
            "confirmed": [],
            "suggested_additions": [],
            "weak_signal": [],
            "market_skills": [],
            "total_postings_scanned": 0,
        }

    market_skills = extract_market_skills(refined_role, postings, progress_callback=progress_callback)
    comparison = compare_to_roadmap(roadmap_skills, market_skills)
    comparison["market_skills"] = market_skills
    comparison["total_postings_scanned"] = len(postings)

    return comparison