"""
Step 1 of the AI Career Coach project — the smallest possible version
of "the core loop" that everything else gets built on top of:

    take some text  -->  send it to an LLM  -->  print what comes back

Nothing fancy here on purpose. No web framework, no UI, no database.
Just proving the connection works before adding anything on top of it.
"""

import os
from dotenv import load_dotenv
from groq import Groq

# load_dotenv() reads the .env file in this folder and makes GROQ_API_KEY
# available via os.environ, without you ever typing the key into this file.
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def ask_llm(prompt: str) -> str:
    """Send one prompt to the LLM and return its text response."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # current production model on Groq
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # (python main.py) — it's the standard entry point for a script.
    user_input = input("What role or interest are you exploring? ")

    prompt = f"List 5 key skills someone would need for: {user_input}. Keep it short."

    print("\nAsking the LLM...\n")
    answer = ask_llm(prompt)

    print("--- LLM response ---\n")
    print(answer)
