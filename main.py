"""
Entry point for the AI Career Coach CLI.
"""

from personaltracker import get_skill_roadmap, extract_text_from_pdf, get_skill_gaps


def print_roadmap(skills: list[dict]) -> None:
    print("--- Skill roadmap (grounded in real sources) ---\n")
    for i, skill in enumerate(skills, start=1):
        print(f"{i}. {skill['name']} ({skill['priority']} priority, {skill['level_required']})")
        print(f"   {skill['description']}")
        print(f"   Why it matters: {skill['why_it_matters']}")
        print(f"    Source: {skill.get('source_url', 'n/a')}\n")


def print_gaps(gaps: list[dict], skills: list[dict]) -> None:
    # map skill_id -> name, so the display is readable instead of raw ids
    name_by_id = {s["id"]: s["name"] for s in skills}

    print("--- Gap analysis ---\n")
    for i, gap in enumerate(gaps, start=1):
        skill_name = name_by_id.get(gap["skill_id"], gap["skill_id"])
        print(f"{i}. {skill_name} -- {gap['status'].upper()}")
        if gap["evidence_from_cv"]:
            print(f"   Evidence in CV: {gap['evidence_from_cv']}")
        print(f"   Suggestion: {gap['suggestion']}\n")


if __name__ == "__main__":
    role = input("What role or interest are you exploring? ")
    print(f"\nSearching for real, current info on '{role}'...\n")
    skills = get_skill_roadmap(role)
    print_roadmap(skills)

    cv_path = input("\nPath to your CV (PDF): ").strip()
    print("\nExtracting text from your CV...\n")
    cv_text = extract_text_from_pdf(cv_path)
    print(f"Extracted {len(cv_text)} characters total.\n")

    print("Analyzing gaps against the roadmap...\n")
    gaps = get_skill_gaps(skills, cv_text)
    print_gaps(gaps, skills)