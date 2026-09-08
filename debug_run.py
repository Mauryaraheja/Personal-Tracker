from personaltracker.roadmap import refine_role
from personaltracker.market_validation import search_job_postings

refined = refine_role("Gen Ai")
print(f"Refined role: {refined}")

postings = search_job_postings(refined)
print(f"Got {len(postings)} postings\n")
for p in postings:
    print(p.get("url"))
    print(p.get("content", "")[:200])
    print("---")