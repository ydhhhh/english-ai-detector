"""One-shot train-only metadata audit of the two unchanged weight functions.

This is a diagnostic helper, never a trainer/coordinator startup gate. It
reads only the explicitly named training manifest entry and Parquet metadata
columns below. It never projects text or opens development/calibration/final
corpora, predictions, checkpoints, tokenizers or an optimizer. No GPU API is
called. The actual registered train is audited once, using a new JSON output.

After the real data build (not while this is still a proposal):
  python -B -m experiments.english_v2.audit_training_weights \
    --input-sha256 TRAIN_SHA --manifest-sha256 MANIFEST_SHA \
    --neural-sha256 FROZEN_NEURAL_SHA --output NEW_REPORT_JSON
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiments.english_v1 import runtime  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ("id", "group_id", "y_ai", "source", "domain", "collector", "new_vs_replay", "wordlike_tokens",
           "variant", "rewrite_kind", "original_generator", "last_editor", "split")
DIMENSIONS = ("source", "domain", "collector", "new_vs_replay", "length_bin", "variant", "rewrite_kind",
              "original_generator", "last_editor")
MODES = ("class_domain_v3", "class_normalized_v2")


def sha(p):
    with Path(p).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def path(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT/p).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def length_bin(value):
    for upper in (64, 128, 256, 512, 1024):
        if value < upper:
            lower = {64: 0, 128: 64, 256: 128, 512: 256, 1024: 512}[upper]
            return f"{lower}-{upper-1}"
    return "1024+"


def value_label(value):
    # Preserve unknown metadata explicitly instead of inventing origin labels.
    return "[missing]" if value is None or value == "" else str(value)


def aggregate(rows, weights):
    y = np.asarray([r["y_ai"] for r in rows], dtype=np.int64)
    n, total = len(rows), float(weights.sum())
    class_mass = {label: float(weights[y == label].sum()) for label in (0, 1)}
    def counts(indices):
        ii = np.asarray(indices, dtype=np.int64)
        mass = float(weights[ii].sum())
        labels = set(y[ii].tolist())
        label = next(iter(labels)) if len(labels) == 1 else None
        raw_class = int((y == label).sum()) if label is not None else None
        return {"rows": len(ii), "raw_fraction_total": len(ii)/n,
            "raw_fraction_within_class": len(ii)/raw_class if raw_class else None,
            "weight_mass": mass, "fraction_of_total_weight": mass/total,
            "fraction_of_class_weight": mass/class_mass[label] if label is not None else None,
            "mean_weight": float(weights[ii].mean()), "minimum_weight": float(weights[ii].min()),
            "maximum_weight": float(weights[ii].max()), "connected_groups": len({rows[i]["group_id"] for i in ii})}
    output = {"rows": n, "total_weight": total,
        "classes": [{"y_ai": label, **counts(np.flatnonzero(y == label))} for label in (0, 1)],
        "dimensions": {}, "joint_source_domain_collector_replay_origin_y": []}
    for dimension in DIMENSIONS:
        buckets = defaultdict(list)
        for i, row in enumerate(rows):
            buckets[(value_label(row[dimension]), row["y_ai"])].append(i)
        output["dimensions"][dimension] = [{dimension: value, "y_ai": label, **counts(indices)}
            for (value, label), indices in sorted(buckets.items())]
    joint = defaultdict(list)
    fields = ("source", "domain", "collector", "new_vs_replay", "variant", "y_ai")
    for i, row in enumerate(rows):
        joint[tuple(row[field] if field == "y_ai" else value_label(row[field]) for field in fields)].append(i)
    output["joint_source_domain_collector_replay_origin_y"] = [{**dict(zip(fields, keys)), **counts(indices)}
        for keys, indices in sorted(joint.items())]
    return output


def audit(input_path, expected_input_sha, manifest_path, expected_manifest_sha, expected_neural_sha, *, expected_rows=60000):
    import pyarrow.parquet as pq
    input_path, manifest_path = path(input_path), path(manifest_path)
    neural_path = ROOT/"experiments/english_v1/neural.py"
    before = {"input_sha256": sha(input_path), "manifest_sha256": sha(manifest_path), "neural_sha256": sha(neural_path)}
    require(before == {"input_sha256": expected_input_sha, "manifest_sha256": expected_manifest_sha, "neural_sha256": expected_neural_sha},
            "explicit registered train/manifest/frozen-neural SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["output_files"]["train_registered"]
    require(path(entry["path"]) == input_path and entry["sha256"] == expected_input_sha and entry["rows"] == expected_rows and entry["role"] == "train",
            "only the registered training manifest entry may be audited")
    source = pq.ParquetFile(input_path)
    require(source.metadata.num_rows == expected_rows and set(COLUMNS) <= set(source.schema_arrow.names), "training metadata count or named columns differ")
    rows = source.read(columns=list(COLUMNS)).to_pylist()
    require(all(r["split"] == "train" and isinstance(r["y_ai"], int) and not isinstance(r["y_ai"], bool) and
        r["y_ai"] in (0, 1) and r["new_vs_replay"] in {"new", "replay"} and
        isinstance(r["wordlike_tokens"], int) and not isinstance(r["wordlike_tokens"], bool) and r["wordlike_tokens"] >= 0 and
        all(isinstance(r[key], str) and r[key].strip() for key in ("id", "group_id", "source", "domain", "collector")) for r in rows),
        "explicit train-only binary/source/domain/collector/replay/length metadata required")
    require(len({r["id"] for r in rows}) == len(rows) and {r["y_ai"] for r in rows} == {0, 1}, "unique IDs and both binary classes required")
    for row in rows:
        row["length_bin"] = length_bin(row["wordlike_tokens"])
    # Import only the immutable implementation. No model class is instantiated,
    # and no torch/CUDA/MPS device or checkpoint API is invoked here.
    from experiments.english_v1.neural import training_weights
    reports = {}
    for mode in MODES:
        weights = training_weights(rows, mode=mode)
        require(weights.shape == (len(rows),) and np.isfinite(weights).all() and (weights > 0).all(), "invalid original weight-function output")
        summary = aggregate(rows, weights)
        y = np.asarray([r["y_ai"] for r in rows])
        wp = np.asarray([r["domain"].casefold() == "wp" for r in rows])
        summary["WP_class_mass"] = [{"y_ai": label, "rows": int(((y == label) & wp).sum()),
            "class_weight_mass": float(weights[y == label].sum()), "WP_weight_mass": float(weights[(y == label) & wp].sum()),
            "WP_fraction_within_class": float(weights[(y == label) & wp].sum()/weights[y == label].sum()),
            "registered_fraction": .3 if mode == "class_domain_v3" else None,
            "matches_30pct_within_class": bool(np.isclose(weights[(y == label) & wp].sum()/weights[y == label].sum(), .3, atol=1e-12, rtol=0)) if mode == "class_domain_v3" else None}
            for label in (0, 1)]
        reports[mode] = summary
    after = {"input_sha256": sha(input_path), "manifest_sha256": sha(manifest_path), "neural_sha256": sha(neural_path)}
    require(after == before, "immutable train/manifest/neural bytes changed during diagnostic audit")
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "train_only_metadata_weight_audit_not_a_launch_gate", "audit_completed": True,
        "input": {"path": str(input_path), "sha256": expected_input_sha, "rows": len(rows)},
        "manifest": {"path": str(manifest_path), "sha256": expected_manifest_sha, "only_entry_used": "output_files.train_registered"},
        "columns_projected": list(COLUMNS), "length_unit": "word_like_tokens_regex", "raw_metadata_counts": aggregate(rows, np.ones(len(rows))),
        "unchanged_original_weight_modes": reports, "frozen_neural_sha256": expected_neural_sha, "audit_code_sha256": sha(__file__),
        "registered_sources_unchanged_during_audit": True, "text_or_nontrain_corpus_columns_read": False,
        "predictions_or_models_or_optimizer_opened": False, "GPU_API_called": False,
        "interpretation": ["Global sample-weight masses are descriptive loss weights, not a generative or deployment class prior.",
            "Mini4 class_domain_v3 assigns WP30% and non-WP70% within each binary class; non-WP includes both new data and replay, which are separately tabulated.",
            "The unchanged trainer divides each batch loss by that batch's sampled weight mass. Shuffling, finite batches, correlated documents and changing gradients mean global mass fractions do not equal exact minibatch optimization or gradient fractions.",
            "Counts by variant/rewrite_kind preserve publisher/data-builder origin metadata; missing origin is shown explicitly rather than inferred from model scores.",
            "This report is a train-only diagnostic run once after real role construction, and neither starts training nor changes weights, roles, thresholds or release criteria."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/english_v2/train_registered.parquet")
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--manifest", default="data/english_v2/manifest.json")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--neural-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, default=60000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "one-shot diagnostic output must be new; never overwrite or silently re-audit")
    result = audit(args.input, args.input_sha256, args.manifest, args.manifest_sha256, args.neural_sha256, expected_rows=args.expected_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"audit_completed": True, "output": str(args.output.resolve()), "train_only": True, "training_started": False}))
