from personaltracker.market_validation import search_job_postings

postings = search_job_postings("Generative AI Engineer", max_results=40)
print(f"Got {len(postings)} postings\n")

from collections import Counter
domains = Counter(p.get("url", "").split("/")[2] for p in postings)
for domain, count in domains.most_common():
    print(f"{domain}: {count}")