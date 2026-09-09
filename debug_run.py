from personaltracker import get_or_create_roadmap
from personaltracker.market_validation import get_market_validation

role = "Gen Ai"
roadmap_skills = get_or_create_roadmap(role)
result = get_market_validation(role=role, roadmap_skills=roadmap_skills)

print("\n========== CONFIRMED ==========")
print(result["confirmed"])

print("\n========== SUGGESTED ADDITIONS ==========")
print(result["suggested_additions"])

print("\n========== WEAK SIGNAL ==========")
print(result["weak_signal"])