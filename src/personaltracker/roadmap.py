"""
Skill roadmap generation: given a role or interest, search the web for
real, current info about the role, then have the LLM synthesize a
structured, sourced skill list from those sources.
"""

import json
from .clients import groq_client, tavily_client


def refine_role(role: str) -> str:
    """
    Turn a vague interest into a specific, well-defined job title before
    searching. Without this, ambiguous input like "Gen AI" pulls generic
    "AI skills for everyone" content instead of skills for an actual job --
    this is the same problem RAG systems call "query rewriting."
    """
    prompt = f"""
    Someone said they're interested in: "{role}"

    If this is already a specific job title, return it unchanged.
    If it's vague (a field, technology, or interest rather than a job
    title), rewrite it as the single most likely specific job title a
    company would actually post -- e.g. "Gen AI" becomes
    "Generative AI Engineer".

    Respond with ONLY the job title, nothing else.
    """
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    refined = response.choices[0].message.content.strip()
    return refined if refined else role


def search_job_info(role: str, max_results: int = 5) -> list[dict]:
    """Fetch real, current info about what a role actually requires."""
    response = tavily_client.search(
        query=f"{role} technical skills required job description",
        max_results=max_results,
        search_depth="advanced",
    )
    return response.get("results", [])


def format_sources_for_prompt(search_results: list[dict]) -> str:
    """Turn raw search results into a block of text the LLM can read."""
    blocks = []
    for r in search_results:
        snippet = r.get("content", "")[:1500]
        blocks.append(f"Source: {r.get('url')}\n{snippet}")
    return "\n\n".join(blocks)


def get_skill_roadmap(role: str) -> list[dict]:
    """Search for real sources, then ask the LLM to build a roadmap from them."""

    refined_role = refine_role(role)
    if refined_role != role:
        print(f"Interpreting '{role}' as: {refined_role}")

    search_results = search_job_info(refined_role)

    if not search_results:
        print("Warning: no search results found -- roadmap will be less grounded.")

    sources_text = format_sources_for_prompt(search_results)

    prompt = f"""
    Here are real, current sources about the role: {refined_role}

    {sources_text}

    Based ONLY on the sources above, return a JSON object with a single key
    "skills", containing a list of 5 skills someone needs for this role.

    Focus on the specific, technical skills that separate a genuinely
    strong candidate from someone with only bare-minimum knowledge --
    named tools, frameworks, and techniques (e.g. "Retrieval-Augmented
    Generation", "LangChain/LangGraph", "vector databases", "prompt
    engineering") rather than skills that apply to nearly any tech role
    (e.g. "Python", "communication", "AI ethics"). Only include a broad
    or generic skill if the sources specifically call it out as critical
    or unusually important for this exact role -- do not include it just
    because it's mentioned in passing.

    Each skill should be an object with these exact keys:
    - "name": short skill name
    - "description": 1-2 sentences on what this skill actually involves
    - "why_it_matters": one sentence, referencing what the sources say
      about why THIS role specifically needs it
    - "priority": one of "high", "medium", "low"
    - "level_required": one of "beginner", "intermediate", "advanced"
    - "source_url": the URL of the source this skill came from

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
        return []

    skills = data.get("skills", [])

    for i, skill in enumerate(skills, start=1):
        skill["id"] = f"skl_{i:03d}"

    return skills