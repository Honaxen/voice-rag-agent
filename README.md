# Voice RAG Agent

A fully local voice-enabled RAG agent — speak your question, get a spoken answer.
No API key, no internet, no cloud. Everything runs on your machine.

🚧 **Work in progress.**

---

## How It Works

```
🎤 You speak
      ↓
Whisper (speech-to-text, local)
      ↓
RAG pipeline (FAISS + BM25 hybrid search)
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
This one is voice-in, voice-out — a fully conversational agent.

---

## Build Checklist

- [ ] `agent/__init__.py`
- [ ] `.gitignore`
- [ ] `agent/stt.py` — Whisper speech-to-text
- [ ] `agent/tts.py` — pyttsx3 text-to-speech
- [ ] `agent/ingestion.py` — document loading and chunking
- [ ] `agent/retrieval.py` — hybrid search (FAISS + BM25 + RRF)
- [ ] `agent/generation.py` — Ollama answer generation
- [ ] `agent/pipeline.py` — connects all components
- [ ] `main.py` — CLI voice loop
- [ ] `app.py` — Gradio UI with mic input
- [ ] `tests/test_pipeline.py`
- [ ] `requirements.txt`
- [ ] Test end-to-end
- [ ] Replace this README with full documentation

---

## Planned Structure

```
voice-rag-agent/
├── agent/
│   ├── __init__.py
│   ├── stt.py
│   ├── tts.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── generation.py
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py
├── main.py
├── app.py
├── requirements.txt
└── .gitignore
```

---

## Stack

Python · Whisper · pyttsx3 · FAISS · BM25 · Ollama · Gradio

---

## Related Projects

- [document-agent](https://github.com/Honaxen/document-agent) — text-based RAG agent (base for this project)
- [multi-tool-agent](https://github.com/Honaxen/multi-tool-agent) — ReAct agent with tools

---

## Author

[Honaxen](https://github.com/Honaxen)