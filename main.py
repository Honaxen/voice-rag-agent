"""
main.py — CLI voice loop for the voice RAG agent.

Usage:
    python3 main.py --doc data/document.pdf
    python3 main.py --doc data/document.txt --duration 7
    python3 main.py --doc data/document.pdf --text  # text mode (no mic)
"""

import argparse
import sys

from agent.pipeline import VoiceRAGPipeline

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
DIM    = "\033[2m"
YELLOW = "\033[33m"


def voice_loop(pipeline: VoiceRAGPipeline, duration: int):
    print(f"\n{BOLD}Voice RAG Agent{RESET}")
    print(f"{DIM}Speak your question when prompted. Press Ctrl+C to exit.{RESET}\n")

    while True:
        try:
            input(f"{CYAN}Press Enter to start recording...{RESET}")
            result = pipeline.ask_voice(duration=duration)
            print(f"\n{YELLOW}You asked:{RESET} {result.question}")
            print(f"{GREEN}Answer:{RESET} {result.answer}\n")
        except KeyboardInterrupt:
            print("\nBye.")
            break
        except Exception as e:
            print(f"Error: {e}")


def text_loop(pipeline: VoiceRAGPipeline):
    print(f"\n{BOLD}Voice RAG Agent (text mode){RESET}")
    print(f"{DIM}Type your question. Answer will be spoken. Type 'quit' to exit.{RESET}\n")

    while True:
        try:
            question = input(f"{CYAN}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        try:
            result = pipeline.ask_text(question)
            print(f"{GREEN}Agent:{RESET} {result.answer}\n")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Voice RAG Agent")
    parser.add_argument("--doc",      required=True, help="Path to document (.txt or .pdf)")
    parser.add_argument("--model",    default="gemma3:12b", help="Ollama model")
    parser.add_argument("--whisper",  default="base", help="Whisper model size")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds")
    parser.add_argument("--text",     action="store_true", help="Text input mode (no mic)")
    args = parser.parse_args()

    try:
        pipeline = VoiceRAGPipeline(
            model=args.model,
            whisper_model=args.whisper,
        )
        pipeline.load_document(args.doc)
    except ConnectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.text:
        text_loop(pipeline)
    else:
        voice_loop(pipeline, args.duration)


if __name__ == "__main__":
    main()