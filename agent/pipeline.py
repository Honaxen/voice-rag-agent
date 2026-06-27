"""
agent/pipeline.py — The full voice RAG pipeline.

Flow:
  1. Ingest document → build hybrid index
  2. Record voice → Whisper transcription
  3. Retrieve relevant chunks
  4. Generate answer with Ollama
  5. Speak answer with pyttsx3
"""

from dataclasses import dataclass

from agent.ingestion import ingest
from agent.retrieval import VectorStore
from agent.generation import generate
from agent.stt import SpeechToText
from agent.tts import TextToSpeech


@dataclass
class TurnResult:
    question: str
    answer: str
    context: list[str]


class VoiceRAGPipeline:
    """End-to-end voice RAG agent."""

    def __init__(
        self,
        model: str = "gemma3:12b",
        whisper_model: str = "base",
        tts_rate: int = 180,
    ):
        self.model = model
        self.vector_store = VectorStore()
        self.stt = SpeechToText(model_size=whisper_model)
        self.tts = TextToSpeech(rate=tts_rate)
        self._loaded = False

    def load_document(self, path: str) -> int:
        """Ingest a document and build the search index."""
        print(f"  [Pipeline] Ingesting: {path}")
        chunks = ingest(path)
        self.vector_store.build(chunks)
        self._loaded = True
        print(f"  [Pipeline] {self.vector_store.size} chunks indexed")
        return self.vector_store.size

    def ask_text(self, question: str, top_k: int = 3) -> TurnResult:
        """
        Text-in, voice-out.
        Ask a question as text, get a spoken answer.
        """
        if not self._loaded:
            raise ValueError("No document loaded. Call load_document() first.")

        context = self.vector_store.retrieve(question, top_k=top_k)
        answer = generate(question, context, model=self.model)
        self.tts.speak(answer)

        return TurnResult(question=question, answer=answer, context=context)

    def ask_voice(self, duration: int = 5, top_k: int = 3) -> TurnResult:
        """
        Voice-in, voice-out.
        Record a question, get a spoken answer.
        """
        if not self._loaded:
            raise ValueError("No document loaded. Call load_document() first.")

        question = self.stt.from_microphone(duration=duration)
        print(f"  [STT] Heard: {question}")

        return self.ask_text(question, top_k=top_k)

    def ask_file(self, audio_path: str, top_k: int = 3) -> TurnResult:
        """
        Audio file in, voice-out.
        Transcribe an audio file and answer.
        """
        question = self.stt.from_file(audio_path)
        print(f"  [STT] Transcribed: {question}")
        return self.ask_text(question, top_k=top_k)