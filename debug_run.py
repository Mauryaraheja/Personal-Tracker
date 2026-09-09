from personaltracker.market_validation import search_job_postings

postings = search_job_postings("Generative AI Engineer")
print(f"Got {len(postings)} postings\n")
for p in postings:
    print(p.get("url"))