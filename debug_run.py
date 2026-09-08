from personaltracker import get_or_create_roadmap
from personaltracker.market_validation import get_market_validation

role = "Gen Ai"  # matches what's already tracked in your DB
roadmap_skills = get_or_create_roadmap(role)  # DB read, no API cost

result = get_market_validation(role=role, roadmap_skills=roadmap_skills)

print("\n========== MARKET SKILLS ==========")
print(result["market_skills"])

print("\n========== CONFIRMED ==========")
print(result["confirmed"])

print("\n========== SUGGESTED ADDITIONS ==========")
print(result["suggested_additions"])

print("\n========== WEAK SIGNAL ==========")
print(result["weak_signal"])