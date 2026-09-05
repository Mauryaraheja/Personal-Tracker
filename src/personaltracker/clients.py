"""
Shared API clients, initialized once and imported wherever needed.

Both the skill-roadmap pipeline and the (future) CV gap-analysis pipeline
need a Groq client, so it lives here instead of each module re-loading
.env and constructing its own client.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))