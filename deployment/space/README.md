---
title: English AI Text Detector
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12.12
app_file: app.py
license: apache-2.0
short_description: English human/AI detection with conditional source rankings
tags:
  - text-classification
  - ai-detection
  - english
---

An English interface for the immutable `english-ai-detector-v2-practical` bundle. Paste text or upload TXT/DOCX/DOC. The original library supplies full-window inference, English/core gates, calibrated probabilities and both frozen cutoffs. Source outputs are **candidate rankings only**, not authenticated model identities. AI includes substantive AI rewriting. Short/unsupported inputs abstain; probabilities are exploratory. No training or calibration occurs here.

This is a **publication candidate**, not a claim that a hosted build or NVIDIA parity test has passed. The local frozen release was checked on CPU/MPS; this candidate uses a separate Linux/CUDA environment.

The public model repository is committed and verified at `4a98aa247f3cf227c81eec54596b9c0f8d7c358e`. This candidate defaults to that immutable revision and installs the runtime from GitHub commit `0835c57e49e9a44fb3414064ed0f2e93ec864434`. The actual owner account's current creation page did not expose eligible ZeroGPU hosting; no Space or paid replacement was created. The source is prepared for a future eligible ZeroGPU host.

## Owner configuration

When deploying from the GitHub mirror, copy the **contents of `deployment/space/`** to the root of a separate Hugging Face Space repository. `README.md`, `app.py`, `requirements.txt` and `packages.txt` must be at that repository root. Select ZeroGPU only on an eligible host.

1. Publish `english_ai_detector_v2_bundle.zip` at the Model Hub repository root beside the English model card and license. It is the **complete four-component bundle**, with 32 files rooted at `bundle/`, not an individual encoder checkpoint. The approved archive is 780,023,143 bytes with SHA256 `da04e4314bdc4571e1dfff8414d203865ac03ad6d14aeff2a182ac070b5be164`; its inner manifest remains `f588dc69355cc96105ea627e0aa20bdcd9bed78709038884466abf008a4271cc`. The app downloads this one archive at an immutable revision and verifies its SHA **before opening the ZIP**. It rejects unsafe paths, symlinks, duplicate names, excess sizes/counts and extra/missing files, then extracts the exact inventory to a verified content-addressed cache. Each inner file's size/SHA and frozen policy are verified before model loading. No `bundle/` directory is assumed to exist in the Hub repository.
2. Keep the pinned runtime requirement: `english-ai-detector @ git+https://github.com/ydhhhh/english-ai-detector.git@0835c57e49e9a44fb3414064ed0f2e93ec864434`. It is installed during the Space build; no request executes pip/git. The public `english_ai_detector` module exposes `EnglishDetector.load(...).predict(...)` and `extract_document(...)`; compatibility modules include `experiments.ntl_v1.baseline` and all original runtime dependencies. No working-directory, vendor or data-folder fallback is provided. A renamed package may be selected through `DETECTOR_LIBRARY_PACKAGE` only after the publisher verifies equivalent code/API.
3. Set Space **Variables**, not hardcoded credentials:

| Variable | Value |
| --- | --- |
| `MODEL_REPO_ID` | `ydh0822/english-ai-detector` (default) |
| `MODEL_REVISION` | `4a98aa247f3cf227c81eec54596b9c0f8d7c358e` (default immutable public model commit) |
| `BUNDLE_SUBDIRECTORY` | Fixed internal archive root `bundle`; omit it or leave `bundle` |
| `INFERENCE_BACKEND` | `zerogpu` (default); `cpu` only when the owner explicitly selects it |
| `DETECTOR_LIBRARY_PACKAGE` | Optional; defaults to the installed public API `english_ai_detector` |

Public model downloads use `token=False`; no login/token input is needed. Missing or inconsistent configuration leaves the app uninitialized, not silently pointed at another model.

`bundle_archive.py` is a standard-library-only helper shared with local download examples. Its `extract_bundle_archive(path, expected_archive_sha256=..., expected_manifest_sha256=..., member_names=None, cache_directory=None)` returns the verified extracted `bundle/` path. Optional `member_names` adds caller schema checks; default extraction already validates the pinned manifest inventory. It never imports or executes model code.

## Hosting and cost boundary

Checked against official documentation on 2026-10-18. Eligible free personal accounts (verified email and older than 30 days) may host two ZeroGPU Spaces; PRO permits ten, organization hosting requires Team/Enterprise. ZeroGPU requires Gradio. Hosting eligibility must be confirmed in the actual account: none is assumed here. [Official ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu).

CPU Basic has no hourly charge, but the current rules require a subscription to create ordinary Gradio/Docker compute Spaces; it is **not an automatic free-account fallback**. Static hosting alone cannot run this model. [Official Spaces overview](https://huggingface.co/docs/hub/spaces-overview).

Select eligible ZeroGPU hardware explicitly in Space Settings. The README does not request a hardware flavor: `suggested_hardware` would only be a recommendation and cannot assign it. This repository never requests an upgrade, billing or paid endpoint. If the account cannot host the required free setup, stop and report that limitation; do not automatically choose paid hardware. [Configuration reference](https://huggingface.co/docs/hub/spaces-config-reference).

### CPU and ZeroGPU assembly

- `cpu`: loads the same full bundle on CPU at startup; it makes no GPU request. Owners may explicitly use this computational fallback on an available, authorized host. Changing backend does not change model probabilities, cutoffs or source policy.
- `zerogpu`: select eligible ZeroGPU hardware in Settings **and** set the variable. The real `spaces` package is imported before Torch. CUDA model placement occurs during startup, outside the decorated request; `@spaces.GPU(duration=30)` wraps only inference. The statistical component stays on CPU. There is no `torch.compile`, per-request model transfer or fake decorator fallback.
- Queues/quota failure produces an explicit error. The app does not silently change hardware or substitute probabilities. An owner can explicitly restart in CPU mode, subject to hosting eligibility. GPU quota and queue availability still limit use.

The candidate pins Python 3.12.12 / Torch 2.13.0 from the documented ZeroGPU support list. The frozen local environment used Torch 2.14.1; do not claim the old CPU/MPS checks establish CUDA/new-environment equivalence. Before declaring this Space ready, build it, verify all package versions, and compare fixed synthetic CPU/CUDA outputs, decisions, hierarchy and long-window coverage without recalibrating. DOC uses Linux `antiword`; verify an actual DOC upload after build.

HF applies per-visitor GPU quotas. Its documentation also describes automatic use of existing prepaid credits for eligible subscribers after included quota. This application does not configure or control that platform billing behavior; users seeking zero spending must check their own HF quota/credit settings. No credits or subscription are purchased here. [Quota and credit rules](https://huggingface.co/docs/hub/spaces-zerogpu#usage-tiers).

## User-facing scope and privacy

Requests are limited to 200,000 characters and 8 MiB; oversized content is rejected without truncation. TXT supports UTF-8 and BOM-marked UTF-16; DOCX body/table paragraphs and convertible DOC are supported, not scanned images. Both decisions retain the original ≥128-word-like English guard and thresholds: P(AI)≥0.7233173320180257 or P(human)≥0.9049688379295407. Other inputs are inconclusive. Family/model assertions remain disabled, with all seven family/25 leaf candidates displayed conditionally on AI.

Unlike the local app, this demo sends text/files to the hosted Space. Gradio may create temporary uploads. This app disables run history, example caching and analytics and schedules cache cleanup; HF infrastructure has its own policies. Do not submit confidential documents. Results are not an AI-word percentage or an authorship certificate.

The four registered final cohorts and correlated rewrite views are summarized separately in the accompanying model card/metrics artifact; no pooled “accuracy” over all rows is claimed. The model and training-data terms in the Model Hub card and third-party notices remain applicable. `license: apache-2.0` here covers original application code, not relicensing every pretrained/trained artifact or source text.

## Candidate checks

`python checks.py` runs 16 constructed archive, transport, integrity, display and UI-wiring checks without a model, Hub request, login, GPU or service. The archive cases include SHA-before-open, traversal/symlink/duplicate/wrong-root rejection, size/count limits, exact inventory, cache tampering and refusal before model import. These checks do not replace an actual HF build/inference smoke. Model and runtime commits are pinned; actual hosting eligibility and new-environment inference still need verification.
