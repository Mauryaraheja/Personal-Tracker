"""
Entry point for the AI Career Coach CLI.

The real logic now lives in src/personaltracker/ -- the project has
enough distinct pieces (skill roadmap generation, CV parsing) to
benefit from being split into modules instead of one growing file.
"""

from personaltracker import get_skill_roadmap, extract_text_from_pdf


def print_roadmap(skills: list[dict]) -> None:
    print("--- Skill roadmap (grounded in real sources) ---\n")
    for i, skill in enumerate(skills, start=1):
        print(f"{i}. {skill['name']} ({skill['priority']} priority, {skill['level_required']})")
        print(f"   {skill['description']}")
        print(f"   Why it matters: {skill['why_it_matters']}")
        print(f"    Source: {skill.get('source_url', 'n/a')}\n")


if __name__ == "__main__":
    role = input("What role or interest are you exploring? ")
    print(f"\nSearching for real, current info on '{role}'...\n")
    skills = get_skill_roadmap(role)
    print_roadmap(skills)

    cv_path = input("\nPath to your CV (PDF): ").strip()
    print("\nExtracting text from your CV...\n")
    cv_text = extract_text_from_pdf(cv_path)

    print(f"\nExtracted {len(cv_text)} characters total.")
    print("--- Extracted CV text (preview) ---\n")
    print(cv_text[:1000])
    if len(cv_text) > 1000:
        print("\n...(truncated)\n")