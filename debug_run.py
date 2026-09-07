from personaltracker.market_validation import search_job_postings, extract_market_skills

postings = search_job_postings("Data Scientist")
print(f"Got {len(postings)} postings\n")
for p in postings:
    print(p.get("url"))
    print(repr(p.get("content", "")[:300]))
    print("---")

skills = extract_market_skills("Data Scientist", postings)
print(f"\nExtracted {len(skills)} market skills")
print(skills)