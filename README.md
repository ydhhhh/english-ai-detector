# English AI Detector

Calibrated English human-versus-AI text detection with complete long-document window coverage, a local browser interface, a Python API, and exploratory model-family/version rankings.

The frozen v2 binary ensemble uses **75% ModernBERT + 25% MiniLM standardized margins**, followed by a separately fitted Platt calibration. Source ranking additionally uses an inherited MiniLM component and a statistical model. These four components are distributed separately on the [Model Hub](https://huggingface.co/ydh0822/english-ai-detector).

**Human/AI detection met the preregistered practical test targets. Family and exact-version assertions remain disabled: their probabilities are conditional candidate rankings, not reliable evidence of actual API usage.** Texts shorter than 128 original word-like units, or outside the supported English scope, receive an inconclusive decision.

## Install

Python 3.12 is required. Install from this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

`requirements.txt` records the exact versions used in the validated local environment. ZeroGPU uses a separately documented platform lock and requires its own deployment checks. No weights or corpus are downloaded by package installation.

## Download the model

The Model Hub distributes one complete `english_ai_detector_v2_bundle.zip`, containing all four components under `bundle/`. The download example verifies the ZIP SHA256 before parsing, bounds and validates the exact inventory, checks every extracted file, and prints the verified local directory. It downloads publicly without a token. No `bundle/*` tree is assumed to exist separately on the Hub.

```bash
BUNDLE_PATH="$(python examples/download_model.py --revision 4a98aa247f3cf227c81eec54596b9c0f8d7c358e)"
```

The command and example default pin the verified immutable Model Hub release. The example additionally pins the approved archive bytes, so a different archive fails closed. Only load the trusted project bundle: Python joblib serialization can execute code when loading untrusted files. The model card and `LICENSE-MODEL.md` define the separate model-use scope.

## Use the browser tool

```bash
english-ai-detector serve --bundle "$BUNDLE_PATH" --device cpu
```

Open <http://127.0.0.1:8765>. Paste English text or upload DOCX, DOC, or TXT. The included HTML can also be opened directly and connected to your local server. On compatible Apple Silicon, `--device mps` is available. HTTP defaults are 200,000 characters, 8 MiB, one simultaneous detection and six requests per minute; oversized requests fail explicitly rather than silently truncating the document. Legacy DOC needs `antiword` on Linux or `textutil` on macOS.

The separate Hugging Face Space target is `ydh0822/english-ai-detector`. Its English ZeroGPU code is prepared for an independent Space repository, but this publication account currently has compute-Space creation disabled in the actual HF UI. No live Space, subscription purchase, paid fallback or CUDA equivalence is claimed. Deployment files with fixed source/model revisions are available in [deployment/space](deployment/space/README.md). See [publication status](docs/publication.md).

## Python and command line

```python
from english_ai_detector import EnglishDetector

detector = EnglishDetector.load("/path/printed/by/the/download/example", device="cpu")
result = detector.predict("Your complete English document goes here.")
print(result["status"], result["probabilities"])
```

```bash
english-ai-detector predict --bundle "$BUNDLE_PATH" --file example.docx
python -m english_ai_detector predict --bundle "$BUNDLE_PATH" --stdin < example.txt
```

## Frozen final evaluation

| Cohort | Distinct text rows | AUROC | Online AI recall | Human-to-AI false positives | Decision coverage |
|---|---:|---:|---:|---:|---:|
| Main | 3,000 | 0.997829 | 97.60% | 1.40% | 97.77% |
| Natural Questions source-pipeline holdout | 800 | 0.999388 | 99.50% | 3.50% | 95.38% |
| Modern generators with human controls | 2,000 | 0.997897 | 97.80% | 0.70% | 97.45% |
| Paraphrase stress with human controls | 1,800 | 0.995828 | 95.17% | 0.17% | 96.22% |

These are 7,600 distinct held-out rows, with correlated parent groups handled jointly. Rejected inputs remain in online recall denominators. The preregistered point criteria passed; this is not a guarantee for arbitrary domains or future generators. The weakest disclosed news/human-original-to-AI-rewrite cell has 90.67% AI recall. Confidence intervals, both error directions, accuracy definitions and correlated rewrite views are in the [model card](docs/model_card.md) and [aggregate report](docs/reports/METRICS_FOR_MODEL_CARD.json). Forced binary accuracy is different from accuracy conditional on the system issuing a decision.

## Development and reproducibility

```bash
python -m unittest discover -s tests -v
```

See [training and evaluation](docs/training.md), [data sources](docs/data_sources.md), [HTTP API](docs/api.md), [license scope](docs/licensing.md) and [publication checks](docs/publication.md). The `src/experiments` compatibility namespace preserves the frozen numeric algorithms and joblib class paths; `src/english_ai_detector` is the public interface. Moving source files does not retrain or recalibrate the model. Full data reproduction requires separately obtained upstream material and recorded historical evidence; cloning this repository alone does not reconstruct the full corpus.

## License

Original public source and documentation: **Apache-2.0**, with the legacy MIT notice retained in [NOTICE](NOTICE) and [third_party/LICENSE-MIT-legacy.txt](third_party/LICENSE-MIT-legacy.txt). The separately distributed trained model has its own research-use notice, not an unrestricted Apache weight license. Wikipedia/CMU/Natural Questions attribution and ShareAlike conditions, historical S2ORC noncommercial terms, BBC research/noncommercial terms, upstream generation terms and inherited training-history limitations remain separately disclosed. No corpus text, weights, caches or checkpoint history are included here.
