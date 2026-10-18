# Publication and engineering evidence

Public source: https://github.com/ydhhhh/english-ai-detector . Public model: https://huggingface.co/ydh0822/english-ai-detector . The two services have distinct repository owners. The model repository distributes one complete `english_ai_detector_v2_bundle.zip` plus the English model card and model-use notice. It does not distribute corpus bodies or training checkpoints. The source repository contains neither the model ZIP nor raw weights/data/caches.

The ZIP has 32 regular files under `bundle/`, 777,567,035 bytes compressed, SHA256 `8d040a57f38739ec10c8eec9b5930d94a6bcea303232a7b80a53fbf3a47877c4`. Its internal bundle manifest is `e0ba21adeb0bdd39cff9c7375edfa0c651150bf7ae1584c61a859cd24fa11c65`. Every numeric component, blend, calibration and decision-protocol member matches the frozen research distribution. English embedded documentation and publication-license/provenance packaging, followed by a local-use documentation refresh, produce a different manifest from the original research bundle `6194953ec27387b483c3c7d203d8a17eda84150c56a8480ae1f700293e7baa17`; this is not retraining or threshold selection. Frozen configuration may contain historical provenance paths; runtime does not read those paths.

Hugging Face's current large-file backend is Xet, with its Git/LFS-compatible object tracking. Browser upload uses the logged-in owner's official web uploader and the Hub's large-file storage. No browser cookies or tokens are exported to a command-line client. See the [official repository guide](https://huggingface.co/docs/hub/repositories-getting-started). A host-side upload is only complete when its committed object identity matches the recorded artifact; creating an empty repository is not a complete model release.

## Verified public release

The initial complete source release is commit `0835c57e49e9a44fb3414064ed0f2e93ec864434` ([immutable source](https://github.com/ydhhhh/english-ai-detector/tree/0835c57e49e9a44fb3414064ed0f2e93ec864434)): 154 reviewed files, 3,342,643 bytes. Public recursive-tree metadata matched every expected file size and Git blob identity, with no missing or extra files. The publisher retains external source-verification receipts. Later source commits update documentation and local-use publication records without changing the frozen inference modules.

The Model Hub release is commit `d54db158b99d4b408f950a99a37664f0c5f3d81b` ([immutable model](https://huggingface.co/ydh0822/english-ai-detector/tree/d54db158b99d4b408f950a99a37664f0c5f3d81b)). Public unauthenticated Hub metadata confirms the archive size and SHA256 stated above; pinned raw model-card/license bytes also match the reviewed local artifacts. The publisher retains the external Model Hub verification receipt. The current English root model card and model archive were updated together; the separate research-use notice is unchanged. No authentication token is required to download these public artifacts.

## Local engineering evidence

The local-use documentation refresh changed four embedded documents and their integrity metadata. All 19 numerical members are byte-identical to the prior public bundle. The current archive passed actual verified extraction and four-component CPU integrity loading through the installed public runtime. [Integrity-load report](reports/local_use_bundle_runtime_integrity.json). The existing prediction-parity report below concerns the earlier documentation packaging; this refresh does not claim a new accuracy or forward-prediction test.

- A separately built wheel was installed without dependencies into an isolated target and tested from the existing clean Python 3.12 environment. Public API/CLI/HTML resources and all 13 original numeric/serialization modules match their declared identities. Original-source serialization class paths are deliberately preserved. No pretrained or corpus directory fallback is required.
- An actual trusted joblib load exposed a missing inherited style-refinement module in an early package candidate. Both original serialized-module dependencies were added unchanged before the successful actual check; a constructed fixture was not treated as proof of real-artifact loading.
- Actual original-source/original-bundle versus installed-public-source/public-bundle CPU predictions on core English, short English, unsupported language and long-tail synthetic cases had **maximum absolute probability difference 0.0**, with equal decisions, quality/coverage and source abstention. This is engineering equivalence, not four additional accuracy samples. [Report](reports/public_bundle_runtime_parity.json).
- The translated document parser passed 43 actual checks, including a genuine legacy DOC through the available macOS converter. Source HTTP/unit tests exercised errors and recovery without real accuracy claims. [Parser report](reports/source_parser_checks.json).
- The example archive extractor verifies the approved archive before parsing, bounds total/member sizes, rejects traversal/symlinks/duplicates, matches the full inventory and checks each file before returning a path. A real extraction of the earlier approved ZIP passed; the current local-use archive retains all 19 numerical members and has separately verified integrity/loading evidence. [Documentation refresh report](reports/local_use_bundle_packaging.json).

The original frozen release also has CPU/MPS and independently relocated CLI/HTTP evidence. Those historical checks do not establish equivalence on other execution platforms. The public packaging's actual CPU check above is separately identified. The frozen four-cohort performance and confidence intervals remain the statistical evidence; reformatting/packaging and synthetic QA do not increase detection accuracy.

## Local use only

The project provides a browser tool connected to its local HTTP API. Hosted
inference demos and deployment plans are withdrawn; no online inference service
is provided or planned. Hugging Face is used only to distribute the downloadable
model archive. This change does not alter the model weights, probabilities,
calibration, thresholds or local browser/API functionality.

## Reproduction and report scope

[Reproduction source](../reproduction/README.md) retains separate historical training/audit scripts and a guarded dispatcher, rather than pretending that moving them under `src/` preserves their original project-root contracts. Historical metadata are reference-only, with origin hashes and path redactions. New prepared data and code must receive new reviewed protocol bindings; old QA cannot certify a new run. Upstream access/reconstruction gaps remain in the data and training documents.

Public aggregate reports redact home/workspace paths while retaining source hashes and numerical metrics. They contain no corpus bodies or per-row predictions, and their public byte identities differ from original private-path reports. Source/model/transport manifests are separate; a file cannot contain its own future upload verification. Check the actual public commit history and publisher's external release receipt for the finally committed revisions.
