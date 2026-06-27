# Voice RAG Agent

A fully local voice-enabled RAG agent — speak your question, get a spoken answer.
No API key, no internet, no cloud. Everything runs on your machine.

---

## How It Works

```
🎤 You speak
      ↓
Whisper (speech-to-text, local)
      ↓
Hybrid search — FAISS (dense) + BM25 (sparse) + RRF fusion
      ↓
Ollama — gemma3:12b (answer generation)
      ↓
pyttsx3 (text-to-speech, local)
      ↓
🔊 Agent speaks back
```

---

## What Makes This Different

Every other project in this portfolio is text-in, text-out.
This one is voice-in, voice-out — a fully conversational agent that runs entirely offline.

---

## Project Structure

```
voice-rag-agent/
├── agent/
│   ├── __init__.py
│   ├── stt.py          — Whisper speech-to-text
│   ├── tts.py          — pyttsx3 text-to-speech
│   ├── ingestion.py    — document loading and chunking
│   ├── retrieval.py    — hybrid search (FAISS + BM25 + RRF)
│   ├── generation.py   — Ollama answer generation
│   └── pipeline.py     — connects all components
├── tests/
│   └── test_pipeline.py  — 8/8 passing
├── main.py             — CLI voice loop
├── app.py              — Gradio UI with mic input
├── requirements.txt
└── .gitignore
```

---

## Getting Started

```bash
pip install -r requirements.txt
ollama serve
ollama pull gemma3:12b
```

### Run CLI (text mode — no mic needed)

```bash
python3 main.py --doc your_document.pdf --text
```

### Run CLI (voice mode)

```bash
python3 main.py --doc your_document.pdf --duration 5
```

### Run Gradio UI

```bash
python3 app.py
```

Open: http://localhost:7860

### Run Tests

```bash
pytest tests/ -v
```

---

## Whisper Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny  | 39M  | fastest | lowest |
| base  | 74M  | fast    | good ← default |
| small | 244M | medium  | better |
| medium| 769M | slow    | best |

```bash
python3 main.py --doc doc.pdf --whisper small
```

---

## Stack

Python · Whisper · pyttsx3 · FAISS · BM25 · Ollama · Gradio · pytest

---

## What I Learned

Voice adds a constraint text doesn't have.
The answer has to be short enough to listen to.
This changed how I wrote the generation prompt — "2-3 sentences maximum" became a hard requirement, not a suggestion.

Local TTS is good enough for demos.
pyttsx3 uses the OS voice engine — no latency, no API, works offline.
The quality is not human-level but perfectly usable.

Whisper is remarkably robust.
It handles accents, background noise, and incomplete sentences better than expected for a ~74M parameter model.

Pipeline composition matters.
Each component (STT → retrieval → generation → TTS) can fail independently.
Wrapping each in clear interfaces made debugging much easier.

---

## Related Projects

- [document-agent](https://github.com/Honaxen/document-agent) — text-based RAG agent (base for this)
- [multi-tool-agent](https://github.com/Honaxen/multi-tool-agent) — ReAct agent with tools
- [rag-evaluation-framework](https://github.com/Honaxen/rag-evaluation-framework) — evaluate RAG quality

---

## Author

[Honaxen](https://github.com/Honaxen)