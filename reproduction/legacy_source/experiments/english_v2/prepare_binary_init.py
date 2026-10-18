"""Prepare a new binary-only warm-start directory without altering v1 files.

This performs file/hash/metadata operations only. It never imports torch,
loads model tensors, runs inference, fits a model, or copies optimizer or
calibration state. Execute only after the v2 data/recipe has been frozen.

The immutable v1 trainer checks objective coefficients as architecture fields.
Only architecture.family_loss_weight and architecture.version_loss_weight in
the new detector.json are changed to 0.0. The tensor file, tokenizer, encoder
configuration, class order, head shapes and all other detector metadata remain
unchanged. Historical training_config/provenance inside detector.json describe
the parent training; manifest.json discloses this initialization-only operation.

Example (supply the frozen parent fingerprints, not placeholder strings)::

    .venv/bin/python -B -m experiments.english_v2.prepare_binary_init \
      --source models/candidates/minilm_v3_epoch2 \
      --output models/v2_initializers/minilm4_binary \
      --expected-weights-sha256 PARENT_WEIGHTS_SHA256 \
      --expected-metadata-sha256 PARENT_DETECTOR_JSON_SHA256

Then use the unchanged english_v1.neural CLI with --init-checkpoint OUTPUT,
--family-loss-weight 0 --version-loss-weight 0 and class_normalized_v2.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS, MODEL_FAMILIES

LOSS_FIELDS = ("family_loss_weight", "version_loss_weight")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint inventories refuse symlinks")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha(path)
    return result


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def prepare(source, output, *, expected_weights_sha256, expected_metadata_sha256):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output.exists():
        raise ValueError("source must exist and output must be a new, nonexistent directory")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("parent checkpoint and new initializer directories must not overlap")
    for value in (expected_weights_sha256, expected_metadata_sha256):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("explicit frozen parent SHA256 values are required")
    before = inventory(source)
    if before.get("detector.safetensors") != expected_weights_sha256 or before.get("detector.json") != expected_metadata_sha256:
        raise ValueError("parent checkpoint differs from its frozen fingerprints")
    if not any(key.startswith("encoder_config/") for key in before) or not any(key.startswith("tokenizer/") for key in before):
        raise ValueError("local encoder configuration and tokenizer are required")
    original = json.loads((source / "detector.json").read_text())
    if original.get("weights_sha256") != expected_weights_sha256:
        raise ValueError("parent detector metadata does not bind its actual tensor file")
    if (original.get("family_labels") != FAMILIES or original.get("model_labels") != MODEL_LABELS or
            original.get("model_families") != MODEL_FAMILIES):
        raise ValueError("parent head/catalog order differs from the frozen v1 architecture")
    for field in LOSS_FIELDS:
        value = original.get("architecture", {}).get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("parent has an invalid objective coefficient")
    derived = copy.deepcopy(original)
    changes = {}
    for field in LOSS_FIELDS:
        changes["architecture." + field] = {"parent": derived["architecture"][field], "initializer": 0.0}
        derived["architecture"][field] = 0.0
    copied = {key: value for key, value in before.items() if key == "detector.safetensors" or
              key.startswith("encoder_config/") or key.startswith("tokenizer/")}
    # Exclusive creation prevents overwriting a previous initializer, partial
    # preparation, checkpoint, or historical result. Partial failure is retained
    # for inspection; a new output path is required for another attempt.
    output.mkdir(parents=True, exist_ok=False)
    for key in copied:
        target = output / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / key, target)
    write_new(output / "detector.json", derived)
    after_copy = inventory(output)
    if any(after_copy.get(key) != value for key, value in copied.items()):
        raise ValueError("a copied tensor/tokenizer/encoder file changed")
    if set(after_copy) != set(copied) | {"detector.json"}:
        raise ValueError("unexpected file copied to the initializer")
    new_metadata = json.loads((output / "detector.json").read_text())
    if new_metadata != derived:
        raise ValueError("initializer metadata differs from the two-field objective edit")
    if inventory(source) != before:
        raise ValueError("the parent checkpoint changed during preparation")
    manifest = {
        "schema_version": 1, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "byte-identical checkpoint initialization with binary-only objective configuration",
        "parent": {"path": str(source), "weights_sha256": expected_weights_sha256,
            "detector_metadata_sha256": expected_metadata_sha256, "all_original_files_sha256": before},
        "initializer": {"path": str(output), "weights_sha256": after_copy["detector.safetensors"],
            "detector_metadata_sha256": after_copy["detector.json"], "all_inference_files_sha256": after_copy},
        "detector_json_changes": changes, "only_two_objective_fields_changed": True,
        "all_copied_files_byte_identical": True, "parent_all_files_unchanged": True,
        "parameter_tensors_and_head_shapes_unchanged": True,
        "head_shape_evidence": "The entire safetensors file, including its shape header and every tensor byte, has the parent SHA256.",
        "training_config_and_provenance_scope": "Unchanged historical parent training metadata; this operation has no training/optimizer state.",
        "optimizer_or_scheduler_or_rng_copied": False, "calibration_copied": False,
        "model_forward_or_training_performed": False,
        "fine_labels_and_trust_masks_changed": False,
        "future_fine_head_use": "Unused after binary adaptation; source rankings must use the separate frozen v1 Mini3+stat2 components.",
        "prepare_code_sha256": sha(__file__),
    }
    write_new(output / "manifest.json", manifest)
    # Recheck the original after manifest writing as well. No parent writes are
    # performed anywhere in this function.
    if inventory(source) != before:
        raise ValueError("the parent checkpoint changed before completion")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--expected-metadata-sha256", required=True)
    args = parser.parse_args()
    value = prepare(args.source, args.output, expected_weights_sha256=args.expected_weights_sha256,
                    expected_metadata_sha256=args.expected_metadata_sha256)
    print(json.dumps({"prepared": True, "output": str(args.output.resolve()),
        "weights_sha256": value["initializer"]["weights_sha256"],
        "metadata_sha256": value["initializer"]["detector_metadata_sha256"],
        "parent_unchanged": True, "training_performed": False}, indent=2))


if __name__ == "__main__":
    main()
