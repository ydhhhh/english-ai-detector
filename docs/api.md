# HTTP and Python interface

Start `english-ai-detector serve --bundle MODEL_SNAPSHOT/bundle --device cpu`. The default listener is loopback-only, port 8765. `GET /` serves the included English HTML and `GET /health` reports readiness, the loaded model identifier, limits and legacy DOC converter availability.

`POST /detect` uses `Content-Type: application/json` and an object with a nonempty `text` string. `POST /detect-file` accepts the original document bytes with `Content-Type: application/octet-stream` and a URL-encoded basename in `X-Filename`. Supported inputs are UTF-8/BOM/UTF-16 TXT, DOCX body/table text and legacy binary DOC through an available local converter. Scanned images/OCR and PDF are not supported.

```bash
curl http://127.0.0.1:8765/health
curl --data-binary @document.docx -H 'Content-Type: application/octet-stream' -H 'X-Filename: document.docx' http://127.0.0.1:8765/detect-file
```

The response contains `model_id`, `status`, `probabilities.human`, `probabilities.AI`, conditional `family_probabilities` and `model_probabilities`, `quality`, `warnings`, `source_assertions`, and elapsed time. `status` is `AI-generated`, `likely human`, or `inconclusive`. This release always leaves `accepted_family` and `accepted_model` null. The source probabilities condition on the text being AI generated; they are not measured probabilities of actual provider calls. The `other_version` and `other_family` buckets do not establish open-world source accuracy.

The AI decision requires P(AI) >= 0.7233173320180257; the human decision requires P(human) >= 0.9049688379295407. English support and at least 128 original word-like units are also required. Everything else is inconclusive, even if an exploratory probability is high. These mixed-calibration thresholds do not guarantee a fixed population error rate for each domain.

Default HTTP limits are 200,000 characters, 8 MiB upload/request bytes, six requests per minute and one simultaneous model call. Oversized inputs receive an explicit error; model inference covers all accepted encoder tokens in overlapping windows and does not silently discard the tail. The service does not log request bodies, document names or predictions. These storage and logging statements describe the included local HTTP service.

The Python interface exports `EnglishDetector`, `extract_document`, `extract_text`, and `DocumentError` from `english_ai_detector`. `EnglishDetector.load(path, device="cpu", batch_size=16)` validates the manifest, files, calibration and language-rule identity. `predict(text)` returns one response; `predict_many(texts)` returns one per string. CPU and supported Apple MPS are exposed by the local CLI. Only load the project's trusted statistical joblib artifact.

For detailed evaluation and limitations see [model_card.md](model_card.md).
