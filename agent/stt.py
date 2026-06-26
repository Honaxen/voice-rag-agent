"""
agent/stt.py — Speech-to-text using OpenAI Whisper (local).

Whisper runs entirely on your machine — no API key, no internet.
Supports microphone input and audio file transcription.

Models by size (trade-off: speed vs accuracy):
    tiny   — fastest, least accurate (~39M params)
    base   — good balance (~74M params)  ← default
    small  — better accuracy (~244M params)
    medium — near-human accuracy (~769M params)

Usage:
    stt = SpeechToText(model_size="base")
    text = stt.from_microphone(duration=5)
    text = stt.from_file("audio.wav")
"""

import tempfile
import os
import wave
import numpy as np

import whisper


class SpeechToText:
    """Transcribe speech using local Whisper model."""

    def __init__(self, model_size: str = "base"):
        print(f"  [STT] Loading Whisper {model_size}...")
        self.model = whisper.load_model(model_size)
        self.model_size = model_size

    def from_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file.

        Args:
            audio_path: path to .wav, .mp3, .m4a, or .flac file

        Returns:
            Transcribed text
        """
        result = self.model.transcribe(audio_path, fp16=False)
        return result["text"].strip()

    def from_microphone(self, duration: int = 5, sample_rate: int = 16000) -> str:
        """
        Record from microphone and transcribe.

        Args:
            duration: recording duration in seconds
            sample_rate: audio sample rate (Whisper expects 16000 Hz)

        Returns:
            Transcribed text
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "pip install sounddevice to use microphone input"
            )

        print(f"  [STT] Recording for {duration}s... (speak now)")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print(f"  [STT] Transcribing...")

        audio_data = audio.flatten()

        # Save to temp file and transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with wave.open(tmp_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            text = self.from_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        return text