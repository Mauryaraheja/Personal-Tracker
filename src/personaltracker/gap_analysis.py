"""
Gap analysis: compare a candidate's parsed CV text against a skill
roadmap, skill-by-skill, and produce structured Gap objects.

Done as a single LLM call covering all skills at once (rather than one
call per skill) -- with a small roadmap (~5 skills) this is cheaper and
lets the model reason across the whole CV consistently. Worth revisiting
if the roadmap ever grows much larger.
"""

import json
from .clients import groq_client


def format_skills_for_prompt(skills: list[dict]) -> str:
    """Turn the skill roadmap into text the LLM can reference by id."""
    blocks = []
    for s in skills:
        blocks.append(
            f"- id: {s['id']}\n"
            f"  name: {s['name']}\n"
            f"  description: {s['description']}\n"
            f"  level_required: {s['level_required']}"
        )
    return "\n".join(blocks)


def get_skill_gaps(skills: list[dict], cv_text: str) -> list[dict]:
    """Compare CV text against each skill in the roadmap, return Gap objects."""

    skills_text = format_skills_for_prompt(skills)

    prompt = f"""
    Here is a candidate's CV:

    {cv_text}

    Here is the skill roadmap for the role they're targeting:

    {skills_text}

    For EACH skill listed above, decide the candidate's status on it based
    ONLY on what the CV actually shows -- do not assume skills the CV
    doesn't mention.

    status must be one of:
    - "missing": no evidence of this skill anywhere in the CV
    - "partial": some evidence, but limited/underdeveloped relative to
      the skill's level_required (e.g. mentioned once, no depth, or below
      the required proficiency)
    - "met": clear, solid evidence the candidate has this skill at or
      above the level_required

    evidence_from_cv: if status is "partial" or "met", a short paraphrase
    (in your own words, not a direct quote) of what in the CV shows this.
    If status is "missing", this must be null.

    suggestion: one concrete, actionable next step to close or strengthen
    this gap -- e.g. a specific project idea, certification, or practice
    exercise. This should make sense given the status (e.g. "missing"
    suggestions should be more foundational than "partial" ones).

    Return a JSON object with a single key "gaps", a list of objects,
    one per skill above, each with exactly these keys:
    - "skill_id": the skill's id from the roadmap above (use these exact
      ids, do not invent new ones)
    - "status": one of "missing", "partial", "met"
    - "evidence_from_cv": string or null
    - "suggestion": string

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

    gaps = data.get("gaps", [])

    for i, gap in enumerate(gaps, start=1):
        gap["id"] = f"gap_{i:03d}"

    return gaps