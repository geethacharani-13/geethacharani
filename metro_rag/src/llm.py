"""
Thin wrapper around Groq's chat completions API.

Groq is free-tier friendly and fast, which matters for iterating during
development. Swap the model/base_url here if you'd rather use Gemini or
another provider -- the rest of the app doesn't need to change.
"""

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the Delhi Metro "
    "network. Only use the CONTEXT provided to answer -- do not invent "
    "station names, distances, or connections that aren't in the context. "
    "If the context doesn't contain the answer, say so plainly. Keep "
    "answers concise (2-4 sentences)."
)


def generate_answer(api_key: str, question: str, context: str) -> str:
    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
