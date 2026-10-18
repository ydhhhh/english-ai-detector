"""English public-Space adapter. No fitting, remote generator calls or hardware API."""
import importlib
import os
from pathlib import Path
import tempfile
import threading

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(tempfile.gettempdir()) / "english_detector_uploads"))
os.environ.setdefault("GRADIO_RUN_HISTORY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Import the real HF decorator even on CPU. It is effect-free off ZeroGPU;
# a fake/no-op substitute must not disguise a broken ZeroGPU installation.
import spaces
import gradio as gr
from huggingface_hub import hf_hub_download
from space_core import (Config, InputError, MODEL_ID, AI_CUTOFF, HUMAN_CUTOFF,
    MAX_UPLOAD_BYTES, download_bundle, pasted_text, document_text, formatted_result)

LOCK = threading.Lock()


def load_backend(config):
    bundle = download_bundle(config, hf_hub_download)
    library = importlib.import_module(config.library_package)
    # Startup placement, outside @spaces.GPU: ZeroGPU emulates CUDA here.
    # Never allocate/move models inside a decorated request or train a model.
    device = "cuda" if config.backend == "zerogpu" else "cpu"
    detector = library.EnglishDetector.load(bundle, device=device, batch_size=4)
    if detector.model_id != MODEL_ID or any(detector.source_assertions[k] for k in ("valid", "family_enabled", "model_enabled")):
        raise ValueError("wrong model or enabled source assertions")
    if detector.thresholds["AI"]["threshold"] != AI_CUTOFF or detector.thresholds["human"]["threshold"] != HUMAN_CUTOFF:
        raise ValueError("frozen thresholds differ")
    import torch
    torch.set_num_threads(2)

    def predict(text):
        return detector.predict(text)

    runner = spaces.GPU(duration=30)(predict) if config.backend == "zerogpu" else predict
    return runner, library.extract_document, bundle


def build_app(config=None, runner=None, parser=None, initialization_error=None):
    configured = config is not None and runner is not None and parser is not None
    backend = config.backend if configured else "not initialized"
    upload_root = Path(os.environ["GRADIO_TEMP_DIR"])

    def analyze(text):
        if not configured:
            raise gr.Error("The model is not initialized. The Space owner must complete its repository/library and hardware configuration.")
        try:
            text = pasted_text(text)
            with LOCK:
                result = runner(text)
            return formatted_result(result)
        except InputError as exc:
            raise gr.Error(str(exc)) from None
        except Exception:
            # Do not expose traceback, input content, server paths or tokens.
            suffix = " ZeroGPU may be unavailable or its visitor quota exhausted. Try later or ask the owner to explicitly enable CPU mode." if backend == "zerogpu" else " Try a smaller document or contact the Space owner."
            raise gr.Error("Detection could not complete; no result was substituted and no hardware upgrade was requested." + suffix) from None

    def analyze_text(text):
        return analyze(text)

    def analyze_upload(filename):
        if not configured:
            return analyze("")
        try:
            text = document_text(filename, parser, upload_root)
        except InputError as exc:
            raise gr.Error(str(exc)) from None
        return analyze(text)

    with gr.Blocks(title="English AI Text Detector", analytics_enabled=False, delete_cache=(60, 60)) as demo:
        gr.Markdown("# English AI Text Detector\nHuman/AI probabilities and conditional source candidates. **Source family and version assertions are disabled.**")
        gr.Markdown("AI includes substantive AI rewriting of human originals. Short texts (<128 word-like units) and unsupported languages receive **inconclusive** decisions; their probabilities are exploratory. This is not an authorship certificate.")
        gr.Markdown(f"Model: **{MODEL_ID if configured else 'not loaded'}** · Compute: **{backend}**")
        if initialization_error:
            gr.Markdown("**Owner setup required.** Model loading did not complete. Check the pinned Model Hub revision, installed detector library, and selected hardware. No paid upgrade or login was attempted by this app.")
        with gr.Tabs():
            with gr.Tab("Paste text"):
                text = gr.Textbox(label="English passage", lines=10, placeholder="Paste your English passage here.")
                text_button = gr.Button("Analyze text", variant="primary", interactive=configured)
            with gr.Tab("Upload document"):
                file = gr.File(label="TXT, DOCX or DOC · maximum 8 MiB", file_types=[".txt", ".docx", ".doc"], type="filepath")
                file_button = gr.Button("Analyze document", variant="primary", interactive=configured)
        status = gr.Textbox(label="Decision", interactive=False)
        conclusion = gr.Textbox(label="Confidence explanation", interactive=False)
        gr.Markdown("## 1 · Human / AI probabilities\nThese are calibrated estimates, not the percentage of AI-written words.")
        binary = gr.Dataframe(headers=["Category", "Probability"], datatype=["str", "str"], interactive=False)
        gr.Markdown("## 2 · Model-family candidates\nConditional on AI generation. A high rank does not confirm a source.")
        families = gr.Dataframe(headers=["Family", "Probability"], datatype=["str", "str"], interactive=False)
        gr.Markdown("## 3 · Nominal version candidates\nConditional on AI generation, including other-version/family buckets. Publisher labels do not authenticate actual model weights or the last editor.")
        leaves = gr.Dataframe(headers=["Nominal label", "Family", "Probability"], datatype=["str", "str", "str"], interactive=False)
        warnings = gr.Textbox(label="Scope and uncertainty notes", lines=3, interactive=False)
        quality = gr.Textbox(label="Input coverage", interactive=False)
        outputs = [status, conclusion, binary, families, leaves, warnings, quality]
        text_button.click(analyze_text, inputs=text, outputs=outputs, concurrency_id="detector", concurrency_limit=1, api_visibility="private")
        file_button.click(analyze_upload, inputs=file, outputs=outputs, concurrency_id="detector", concurrency_limit=1, api_visibility="private")
        gr.ClearButton([text, file, *outputs])
        gr.Markdown("This hosted demo sends your text/files to this Space. Gradio can create temporary upload files; cleanup is scheduled after 60 seconds. No examples or run history are cached by this app. Avoid confidential material.\n\nMaximum request: 200,000 characters. Over-limit inputs are rejected, never truncated. TXT accepts UTF-8 and BOM-marked UTF-16; image-only scans need OCR. Accuracy and error rates depend on domain, generator and editing style.")
    return demo.queue(default_concurrency_limit=1, max_size=16, api_open=False)


def main():
    config = runner = parser = bundle = None
    error = None
    try:
        config = Config.from_environment()
        runner, parser, bundle = load_backend(config)
    except Exception as exc:
        error = type(exc).__name__
        print("Model initialization incomplete: " + error + "; consult owner configuration. No hardware API was called.", flush=True)
    demo = build_app(config, runner, parser, error)
    blocked = [str(bundle)] if bundle is not None else []
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), share=False,
        show_error=False, max_file_size=MAX_UPLOAD_BYTES, blocked_paths=blocked,
        enable_monitoring=False, run_history=False, footer_links=[], ssr_mode=False)


if __name__ == "__main__":
    main()
