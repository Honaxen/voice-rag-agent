"""
app.py — Gradio web UI for the voice RAG agent.

Features:
- Upload a document (PDF or TXT)
- Record voice or type a question
- See and hear the answer

Run:
    python3 app.py
Then open: http://localhost:7860
"""

import gradio as gr
from agent.pipeline import VoiceRAGPipeline

pipeline = VoiceRAGPipeline()
_doc_loaded = False


def load_doc(file):
    global _doc_loaded
    if file is None:
        return "No file uploaded."
    try:
        n = pipeline.load_document(file.name)
        _doc_loaded = True
        return f"✅ Loaded {n} chunks from {file.name.split('/')[-1]}"
    except Exception as e:
        return f"❌ Error: {e}"


def answer_text(question: str):
    if not _doc_loaded:
        return "Please upload a document first.", None
    if not question.strip():
        return "Please enter a question.", None
    try:
        result = pipeline.ask_text(question)
        # Save spoken answer to temp file for Gradio audio output
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        pipeline.tts.save(tmp_path, result.answer)
        return result.answer, tmp_path
    except Exception as e:
        return f"Error: {e}", None


def answer_voice(audio_path: str):
    if not _doc_loaded:
        return "Please upload a document first.", None
    if audio_path is None:
        return "No audio recorded.", None
    try:
        result = pipeline.ask_file(audio_path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        pipeline.tts.save(tmp_path, result.answer)
        return f"**You asked:** {result.question}\n\n**Answer:** {result.answer}", tmp_path
    except Exception as e:
        return f"Error: {e}", None


with gr.Blocks(title="Voice RAG Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Voice RAG Agent\nAsk questions about your document — by voice or text.")

    with gr.Row():
        doc_input = gr.File(label="Upload document (.txt or .pdf)", file_types=[".txt", ".pdf"])
        doc_status = gr.Textbox(label="Status", interactive=False)

    doc_input.change(fn=load_doc, inputs=doc_input, outputs=doc_status)

    gr.Markdown("---")

    with gr.Tab("🎤 Voice Input"):
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your question")
        voice_btn = gr.Button("Ask", variant="primary")
        voice_answer = gr.Markdown(label="Answer")
        voice_audio = gr.Audio(label="Spoken answer", autoplay=True)
        voice_btn.click(fn=answer_voice, inputs=audio_input, outputs=[voice_answer, voice_audio])

    with gr.Tab("⌨️ Text Input"):
        text_input = gr.Textbox(label="Your question", placeholder="Ask anything about the document...")
        text_btn = gr.Button("Ask", variant="primary")
        text_answer = gr.Textbox(label="Answer", interactive=False)
        text_audio = gr.Audio(label="Spoken answer", autoplay=True)
        text_btn.click(fn=answer_text, inputs=text_input, outputs=[text_answer, text_audio])
        text_input.submit(fn=answer_text, inputs=text_input, outputs=[text_answer, text_audio])

if __name__ == "__main__":
    demo.launch(server_port=7860)