# Reproducing the registered research pipeline

This directory contains source, not datasets, checkpoints, cached API responses,
raw predictions or permission to reuse a historical final set. The public
`src/english_ai_detector` package is for deployment. The separate
`legacy_source` layout keeps each research script at its original relative path:
for example, `experiments/english_v2/train_registered.py` resolves
`Path(__file__).parents[2]` to the legacy project root.

## What is preserved

`source_manifest.json` lists 80 original Python files and four generated package
markers. Seventy-six original files are byte-identical. Four publication copies
have explicitly recorded adaptations, with both original and shipped SHA256:

- `lexical_signature.py` and `bundle_checks.py` use Unicode escapes for synthetic
  non-English test strings. Their parsed Python AST and runtime string values are
  unchanged.
- `documents.py` and `serve.py` translate user-facing errors to English; their
  extraction, resource-limit and serving algorithms are unchanged.

The deployed package's thirteen exact numeric/integrity files are separate and
unchanged. No original workspace file, frozen bundle, threshold or trained weight
was modified. An adapted source SHA is never presented as the historical SHA.

The copied research runtime may probe optional directories under the chosen
legacy project root. No vendors, caches or third-party package implementations
are included here; install the declared runtime dependencies normally.

## Read-only inspection

From the repository root:

```bash
python scripts/reproduce.py check
python scripts/reproduce.py show --stage train
python scripts/reproduce.py --help
```

These commands read source and metadata only. They do not import legacy model
modules, load data or weights, download anything, allocate data roles or train.
An inventory pass proves source identity, not that a dataset or run is ready.

Inspect the actual research CLI without starting a job:

```bash
python scripts/reproduce.py run   --project-root reproduction/legacy_source --stage train -- --help
```

`train` exposes the historical coordinator's `prepare` and `run` subcommands.
The dispatcher also names prepare-data, prepare-init, export, export-release,
develop, select, finalize, evaluate-compatible and acceptance. `show` only prints
an argument vector. Arguments after `--` are forwarded unchanged; the wrapper
does not alter sampling, optimization, calibration or decision parameters.

## Prepare an external project before real execution

Create a separate directory and copy the *contents* of `legacy_source` into it.
It must contain `experiments/` and `research/` at its root. The four generated
`__init__.py` markers keep this local research namespace ahead of the installed
inference compatibility package. The dispatcher sets that external directory as
both the working directory and Python module search root.

Before dispatching a real stage, independently provide:

1. Authorized canonical data, all raw lineage/index/cache inputs needed by the
   data recipe, and original file revisions/hashes. Public historical APIs are
   not guaranteed to reproduce old bytes. Code or ID/hash receipts cannot replace
   a missing corpus; no automatic acquisition occurs in this launcher.
2. The actual warm-start parent checkpoints, tokenizer/config artifacts and
   their checksums. Downloaded base encoders are not substitutes for the trained
   v1 parents. Checkpoint and data rights retain their upstream conditions.
3. New connected-component role assignments and independent QA tied to your
   actual inputs, plus the required exact data manifest/code registry/protocol.
   Observed historical final groups must not become fresh blind evaluation.
4. A Python environment and resource/device plan compatible with your intended
   stage. Original recipes were executed on the documented Apple Silicon setup;
   a source-inventory check does not establish another platform's equivalence.

Four adapted publication files change source fingerprints. New data caches,
code registries, QA and frozen protocols must bind your actual source/input
hashes. Existing historical hard-coded safety guards are not bypassed: where a
new frozen protocol is necessary, review and register it explicitly. Do not edit
an old receipt to make a different run look identical.

Use an external root only after those prerequisites are resolved. For example,
the following displays a command but does not execute it:

```bash
python scripts/reproduce.py show   --project-root /path/to/prepared-legacy-project --stage train --   run --protocol /path/to/new-reviewed-protocol.json --sha256 YOUR_PROTOCOL_SHA256
```

A real `run` additionally requires `--acknowledge-external-inputs`. This is an
explicit operator assertion, not validation or approval supplied by this
repository. The wrapper refuses real execution inside the repository snapshot
and refuses its sanitized historical reference files as current protocols.

## Historical recipe references

`reference_metadata/` contains five sanitized, wrapped protocol/config/launch
references. Each records the original file's SHA256 and relative path, and
places historical content under `payload`. `${PROJECT_ROOT}`,
`${PYTHON_EXECUTABLE}` and `${EXTERNAL_PATH}` are placeholders rather than usable
machine paths. The wrappers are not original exact protocols or new QA seals.

They document the completed recipe and command parameters; they cannot authorize
new training, restore unavailable inputs, establish redistribution permission or
turn a cold clone into a one-command reproduction. Training, model selection,
calibration and final evaluation remain separate roles and explicit stages.
