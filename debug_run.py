from collections import Counter
from personaltracker.market_validation import search_job_postings

postings = search_job_postings("Generative AI Engineer")
print(f"Got {len(postings)} postings\n")

for p in postings:
    print(p.get("url"))

print("\n========== DOMAIN SPREAD ==========")
domains = Counter(p.get("url", "").split("/")[2] for p in postings)
for domain, count in domains.most_common():
    print(f"{domain}: {count}")