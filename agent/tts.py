"""
agent/tts.py — Text-to-speech using pyttsx3 (local, no internet).

pyttsx3 uses the OS's built-in TTS engine:
    macOS   → NSSpeechSynthesizer (Alex, Samantha, etc.)
    Windows → SAPI5
    Linux   → eSpeak

Usage:
    tts = TextToSpeech(rate=180, volume=0.9)
    tts.speak("Hello! How can I help you?")
    tts.save("output.mp3", "Hello!")
"""

import pyttsx3


class TextToSpeech:
    """Convert text to speech using local pyttsx3 engine."""

    def __init__(self, rate: int = 180, volume: float = 0.9, voice_index: int = 0):
        """
        Args:
            rate: words per minute (default 180)
            volume: 0.0 to 1.0 (default 0.9)
            voice_index: which system voice to use (0 = first available)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        voices = self.engine.getProperty("voices")
        if voices and voice_index < len(voices):
            self.engine.setProperty("voice", voices[voice_index].id)

    def speak(self, text: str) -> None:
        """Speak text out loud (blocking)."""
        self.engine.say(text)
        self.engine.runAndWait()

    def save(self, path: str, text: str) -> None:
        """Save speech to an audio file."""
        self.engine.save_to_file(text, path)
        self.engine.runAndWait()

    def list_voices(self) -> list[str]:
        """Return available voice names."""
        voices = self.engine.getProperty("voices")
        return [v.name for v in voices]