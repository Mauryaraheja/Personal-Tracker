"""
Gap analysis: compare a candidate's parsed CV text against a skill
roadmap, skill-by-skill, and produce structured Gap objects.

Done as a single LLM call covering all skills at once (rather than one
call per skill) -- with a small roadmap (~5 skills) this is cheaper and
lets the model reason across the whole CV consistently. Worth revisiting
if the roadmap ever grows much larger.
"""

from .llm import ask_groq_for_json
from .models import GapsReply


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

    data = ask_groq_for_json(prompt, "gap-analysis")

    # Same rule as build_roadmap: fail loudly instead of returning []. An
    # empty list would make app.py quietly skip "Next Steps" -- the user
    # would click Analyze Gaps, see nothing happen, and get no error.
    gaps = [gap.model_dump() for gap in GapsReply.model_validate(data).gaps]


    # One gap per roadmap skill -- no more, no fewer. A skill the model
    # skipped would simply vanish from "Next Steps": the quiet version of
    # the empty-list bug above.
    expected_ids = {s["id"] for s in skills}
    returned_ids = [g["skill_id"] for g in gaps]
    if len(returned_ids) != len(expected_ids) or set(returned_ids) != expected_ids:
        raise ValueError(
            f"Groq's gap analysis didn't cover each skill exactly once: "
            f"expected {sorted(expected_ids)}, got {returned_ids}"
        )

    
    for i, gap in enumerate(gaps, start=1):
        gap["id"] = f"gap_{i:03d}"

    return gaps
