"""
Step 3 of the AI Career Coach project.

Adds the "agentic" part: before asking the LLM for a skill roadmap, we
first search the web for real, current info about the role. The LLM
then has to base its answer on those real sources instead of whatever
it remembers from training -- which might be outdated or generic.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


def search_job_info(role: str, max_results: int = 5) -> list[dict]:
    """Fetch real, current info about what a role actually requires."""
    response = tavily_client.search(
        query=f"skills required for {role} job",
        max_results=max_results,
    )
    return response.get("results", [])


def format_sources_for_prompt(search_results: list[dict]) -> str:
    """Turn raw search results into a block of text the LLM can read."""
    blocks = []
    for r in search_results:
        # content is often long -- truncate so the prompt doesn't get huge
        snippet = r.get("content", "")[:500]
        blocks.append(f"Source: {r.get('url')}\n{snippet}")
    return "\n\n".join(blocks)


def get_skill_roadmap(role: str) -> list[dict]:
    """Search for real sources, then ask the LLM to build a roadmap from them."""

    search_results = search_job_info(role)

    if not search_results:
        print("Warning: no search results found -- roadmap will be less grounded.")

    sources_text = format_sources_for_prompt(search_results)

    prompt = f"""
    Here are real, current sources about the role: {role}

    {sources_text}

    Based ONLY on the sources above, return a JSON object with a single key
    "skills", containing a list of 5 skills someone needs for this role.

    Each skill should be an object with these exact keys:
    - "name": short skill name
    - "why_it_matters": one sentence, referencing what the sources say
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
        skill["id"] = f"skl_{i:02d}"

    return skills


if __name__ == "__main__":
    role = input("What role or interest are you exploring? ")

    print(f"\nSearching for real, current info on '{role}'...\n")
    skills = get_skill_roadmap(role)

    print("--- Skill roadmap (grounded in real sources) ---\n")
    for skill in skills:
        print(f"[{skill['id']}] {skill['name']} ({skill['priority']} priority, {skill['level_required']})")
        print(f"    {skill['why_it_matters']}")
        print(f"    Source: {skill.get('source_url', 'n/a')}\n")