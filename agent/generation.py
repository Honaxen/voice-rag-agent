"""
agent/generation.py — Answer generation using Ollama.
"""

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma3:12b"


def generate(query: str, context: list[str], model: str = DEFAULT_MODEL) -> str:
    """Generate a grounded answer from query and retrieved context."""
    context_text = "\n\n".join(context)
    prompt = (
        f"Answer the question based only on the context below.\n"
        f"Keep your answer concise — 2-3 sentences maximum.\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\nAnswer:"
    )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return data["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach Ollama. Run: ollama serve\nError: {e}")