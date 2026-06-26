"""
agent/ingestion.py — Document loading, cleaning, and chunking.
Supports .txt and .pdf files.
"""

import re
from pathlib import Path


def load_document(path: str) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix == ".txt":
        return path.read_text(encoding="utf-8")

    elif path.suffix == ".pdf":
        try:
            import PyPDF2
            text = ""
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("pip install pypdf2 to load PDF files")

    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .txt or .pdf")


def clean_text(text: str) -> str:
    text = re.sub(r"\x00", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def chunk_document(text: str, sentences_per_chunk: int = 4, min_length: int = 50) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if len(chunk.strip()) >= min_length:
            chunks.append(chunk.strip())
    return chunks


def ingest(path: str, sentences_per_chunk: int = 4) -> list[str]:
    """Full pipeline: load → clean → chunk."""
    raw = load_document(path)
    cleaned = clean_text(raw)
    return chunk_document(cleaned, sentences_per_chunk)