from personaltracker.market_validation import search_job_postings, extract_market_skills

postings = search_job_postings("Data Scientist")
print(f"Got {len(postings)} postings\n")

for p in postings:

    print("=" * 80)
    print("URL:", p.get("url"))

    content = p.get("content") or ""
    raw = p.get("raw_content") or ""

    print(f"content length: {len(content)}")
    print(f"raw_content length: {len(raw)}")

    print("\nCONTENT:")
    print(repr(content[:500]))

    print("\nRAW CONTENT:")
    print(repr(raw[:500]))

    print("=" * 80)

skills = extract_market_skills("Data Scientist", postings)

print(f"\nExtracted {len(skills)} market skills")
print(skills)