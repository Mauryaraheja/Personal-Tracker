"""
Shared API clients, initialized once and imported wherever needed.

Every module that talks to Groq or Tavily imports its client from here,
instead of each one re-loading .env and constructing its own.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()

# The groq library retries rate-limit errors (429) by itself, waiting as
# long as Groq's retry-after header asks. Its default of 2 retries wasn't
# enough: a market check sends ~14 calls in about a minute, right at the
# free tier's 8,000 tokens per minute, and one 429 on the grouping step
# threw away the whole check. 6 retries gives it up to about 20 seconds
# of waiting before it gives up.
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"), max_retries=6)
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
