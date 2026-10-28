"""Gradio interface.

    python ui/gradio_app.py            # http://localhost:7860
    # or mount it behind FastAPI:
    #   import gradio as gr; from ui.gradio_app import build_demo
    #   app = gr.mount_gradio_app(app, build_demo(), path="/ui")

The UI is deliberately thin: it calls the same `answer()` the CLI and the API
use, so there is one code path and one set of behaviours to trust.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXAMPLES = [
    "How does Self-RAG decide when retrieval is needed?",
    "What did the graph RAG paper find about global questions?",
    "What is the cost of the graph RAG indexing stage?",
    "What is reciprocal rank fusion used for in hybrid search?",
]


def answer_question(question: str):
    """Returns (answer, sources_markdown, trace_markdown)."""
    from main import answer

    if not question or not question.strip():
        return "Ask a question about the indexed arXiv corpus.", "", ""

    result = answer(question.strip())
    sources = "\n".join(
        f"- **[{c['number']}]** {c['title']} — [{c['paper_id']}]({c['url']}) "
        f"({c['published']})" for c in result["citations"]
    ) or "_No verified citations._"

    meta = result["meta"]
    trace = "\n".join(f"- `{step['agent']}` {step['message']}"
                      for step in result["trace"])
    header = (f"writer: `{meta.get('writer')}` · grounding: "
              f"`{meta.get('grounding', 0):.2f}` · cache: `{meta.get('cache')}` · "
              f"engine: `{meta.get('engine')}`")
    return result["answer"], f"{header}\n\n{sources}", trace


def build_demo():
    import gradio as gr

    with gr.Blocks(title="QueryMesh") as demo:
        gr.Markdown("## QueryMesh\nFive agents over an arXiv corpus: hybrid retrieval, "
                    "graded evidence, verified citations.")
        question = gr.Textbox(label="Question", value=EXAMPLES[0], lines=2)
        with gr.Row():
            ask_button = gr.Button("Ask", variant="primary")
            clear_button = gr.ClearButton([question])
        answer_box = gr.Textbox(label="Answer", lines=8)
        sources_box = gr.Markdown(label="Sources")
        with gr.Accordion("Agent trace", open=False):
            trace_box = gr.Markdown()

        gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[question])

        ask_button.click(answer_question, inputs=question,
                         outputs=[answer_box, sources_box, trace_box])
        question.submit(answer_question, inputs=question,
                        outputs=[answer_box, sources_box, trace_box])
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=7860)
