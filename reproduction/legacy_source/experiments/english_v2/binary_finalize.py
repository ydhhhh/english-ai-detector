"""Binary-only v2 calibration with frozen v1 source rankings.

No training or source-temperature fitting exists in this module. The CLI fits
only a BinaryPlatt and AI/human cutoffs using a hash-bound calibration role.
The existing v1 source hierarchy, refinement and inactive historical source
cutoffs are copied without editing. Offline assertions are always disabled,
matching EnglishDetector's no-gate online path.

Protocol schema1 (all fields required; no implicit mixture/operating point):
  status: "FROZEN_BEFORE_V2_TRAINING_AND_SCORES"
  source_mode: "rankings_only"
  source_calibration_sha256: <original v1 calibration JSON SHA256>
  language_rule_sha256: <current frozen language_guard.py SHA256>
  deployment_decision_protocol_sha256: <frozen amendment07 SHA256>
  calibration_input_sha256: <registered mixed calibration Parquet SHA256>
  calibration_mixture: <nonempty description/object frozen by root>
  calibration_role_history: <nonempty object describing fresh sources and
    observed historical CAL replay: source SHA, selected rows, previous/current
    CAL role continuity; it is not a wholly fresh calibration cohort>
  binary_calibration:
    probability_fit_subset: "supported_all_lengths" | "supported_core"
    threshold_subset: "supported_core"
    minimum_word_like_tokens: 128
    AI_target_fpr: <frozen value, e.g.0.01>
    human_target_fpr: <frozen value, e.g.0.01>
  final_input_sha256s: [<fresh final SHA256>, ...]  # required for evaluate final

fit example (calibration NPZ files must have original export sidecars):
  python -m experiments.english_v2.binary_finalize fit --blend NEW_BLEND \
    --source-calibration V1_CAL_JSON --protocol V2_PROTOCOL \
    --protocol-sha256 FROZEN_PROTOCOL_SHA --input REGISTERED_CAL_PARQUET \
    --input-sha256 FROZEN_INPUT_SHA --predictions new_binary=RAW_CAL_NPZ \
    --output-dir NEW_CAL_DIRECTORY

evaluate example (all selected binary+old Mini3+stat2 components required):
  python -m experiments.english_v2.binary_finalize evaluate --blend NEW_BLEND \
    --calibration NEW_CAL_JSON --protocol V2_PROTOCOL \
    --protocol-sha256 FROZEN_PROTOCOL_SHA --purpose final --input FRESH_FINAL \
    --input-sha256 FROZEN_INPUT_SHA --predictions name=RAW_NPZ ... \
    --output-dir NEW_EVALUATION_DIRECTORY --n-bootstrap 500

The blend must explicitly bind the old source component fingerprints as well
as new binary components before calibration is opened. Pack with the unchanged
package_bundle.package using no source_release_gate. New outputs never replace
existing directories. This source does not freeze a real v2 data/recipe policy.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from experiments.english_v1 import runtime  # noqa: F401
import numpy as np
from experiments.english_v1.calibration import BinaryPlatt, prediction_fingerprint
from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS
from experiments.english_v1.evaluate import _binary_metrics, evaluate_npz_payload, fit_ai_threshold
from experiments.english_v1.finalize import (
    _decision_protocol_sha256, _majority_floor, apply_calibration,
    apply_source_release,
)
from experiments.english_v1.select_binary import blend_margin

PROJECT = Path(__file__).resolve().parents[2]
SOURCE_NAME = "minilm_v3_epoch2"
STAT_NAME = "statistical_v2"
SOURCE_CALIBRATION_SHA256 = "1b3cac20a654f1b0cb27f2784be172b9e59fa21b4858f1e5d2cd253337521ff4"
SOURCE_WEIGHTS = {
    SOURCE_NAME: "9d2ca9d251aeb2f2067478c12c564be863c478b91c579e4ea312f5170c04aabb",
    STAT_NAME: "32a35bcfa2fc09a49d841bf2f2ad67f3d0d5fa926590c282f4dd14363c41fc45",
}
OFF = {"valid": False, "family_enabled": False, "model_enabled": False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def language_sha():
    return sha(PROJECT / "experiments/english_v1/language_guard.py")


def source_objects(value):
    """Exact serialized numeric objects retained from the old source artifact."""
    return {"hierarchy": value["hierarchy"], "version_refinement": value["version_refinement"],
            "family_threshold": value["thresholds"]["family"], "model_threshold": value["thresholds"]["model"]}


def validate_protocol(protocol, protocol_sha256, source_sha256):
    require(is_sha(protocol_sha256) and is_sha(source_sha256), "explicit protocol/source SHA256 required")
    require(protocol.get("schema_version") == 1 and protocol.get("source_mode") == "rankings_only" and
            protocol.get("status") == "FROZEN_BEFORE_V2_TRAINING_AND_SCORES", "pre-frozen v2rankings-only schema1 policy required")
    require(source_sha256 == SOURCE_CALIBRATION_SHA256 and protocol.get("source_calibration_sha256") == source_sha256,
            "source probabilities must inherit the original frozen v1 calibration")
    require(protocol.get("language_rule_sha256") == language_sha() and
            protocol.get("deployment_decision_protocol_sha256") == _decision_protocol_sha256(), "v2 must explicitly retain the frozen language/07decision algorithms")
    require(is_sha(protocol.get("calibration_input_sha256")) and bool(protocol.get("calibration_mixture")) and
            isinstance(protocol.get("calibration_role_history"), dict) and bool(protocol["calibration_role_history"]),
            "registered calibration identity/mixture and explicit nonempty role history required")
    spec = protocol.get("binary_calibration", {})
    require(spec.get("probability_fit_subset") in {"supported_all_lengths", "supported_core"} and
            spec.get("threshold_subset") == "supported_core" and spec.get("minimum_word_like_tokens") == 128,
            "an explicit supported-English probability subset and>=128core threshold subset required")
    for key in ("AI_target_fpr", "human_target_fpr"):
        value = spec.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and 0 < value < 1,
                "both operating-point error budgets must be frozen explicitly")
    return spec


def validate_source(source):
    p = source["provenance"]
    require(p["attribution_model"] == SOURCE_NAME and source["version_refinement"]["statistical_model_name"] == STAT_NAME,
            "the original Mini3/stat2 source pair is required")
    require(p.get("language_rule_sha256") == language_sha() and
            p.get("deployment_decision_protocol_sha256") == _decision_protocol_sha256(), "inherited source artifact used different deployment semantics")
    require(source["hierarchy"]["family_classes"] == FAMILIES and source["hierarchy"]["leaf_classes"] == MODEL_LABELS,
            "source catalog order changed")
    for name, expected in SOURCE_WEIGHTS.items():
        require(p["component_weights_sha256"].get(name) == expected and is_sha(p["component_metadata_sha256"].get(name)), "source component fingerprint changed")


def binary_names(blend):
    require(blend.get("fit_split") == "development_only" and blend.get("calibration_opened") is False and
            blend.get("final_test_opened") is False, "candidate/blend selection must precede calibration and final scores")
    names, weights = blend["names"], np.asarray(blend["selected"]["weights"], dtype=float)
    require(names and len(set(names)) == len(names) and weights.shape == (len(names),) and
            np.isfinite(weights).all() and (weights >= 0).all() and (weights > 0).any(), "valid selected binary component weights required")
    for key in ("mean", "scale"):
        values = np.asarray(blend[key], dtype=float)
        require(values.shape == weights.shape and np.isfinite(values).all() and (key != "scale" or (values > 0).all()), "invalid frozen binary standardization")
    require(np.isfinite(blend["selected"]["intercept"]), "finite frozen blend intercept required")
    return [name for name, weight in zip(names, weights) if weight != 0]


def identity(payload):
    output = {}
    for field, key in (("component_weights_sha256", "weights_sha256"), ("component_metadata_sha256", "metadata_sha256")):
        a = np.asarray(payload.get(field, ""))
        require(a.ndim == 0 and is_sha(str(a.item())), "fingerprinted scalar raw component identity required")
        output[key] = str(a.item())
    return output


def validate_raw_rows(payload):
    flag = np.asarray(payload.get("calibrated", None))
    require(flag.ndim == 0 and flag.dtype.kind == "b" and flag.item() is False,
            "raw calibrated must be an explicit scalar bool False")
    n = len(payload["ids"])
    groups, lengths, scoped = (np.asarray(payload[key]) for key in ("group_ids", "lengths", "scope_word_like_tokens"))
    require(groups.shape == (n,) and groups.dtype.kind == "U" and all(value.strip() for value in groups),
            "aligned nonempty Unicode connected-group IDs required")
    require(lengths.shape == (n,) and lengths.dtype.kind in "iu" and np.isfinite(lengths).all() and (lengths >= 0).all(),
            "aligned finite nonnegative integer word-like counts required")
    require(scoped.shape == lengths.shape and scoped.dtype == lengths.dtype and scoped.tobytes() == lengths.tobytes(),
            "language-scope word counts must match original counts without conversion")


def aligned_binary(blend, arrays, source):
    active = binary_names(blend)
    require(set(arrays) == set(active), "fit requires exactly the nonzero binary raw components; no source fitting inputs")
    ref = arrays[active[0]]
    n = len(ref["ids"])
    ids = np.asarray(ref["ids"])
    require(ids.shape == (n,) and ids.dtype.kind == "U" and len(set(ids.tolist())) == n and n > 0, "unique Unicode calibration IDs required")
    require(np.isin(ref["y_ai"], [0, 1]).all() and np.asarray(ref["y_ai"]).shape == (n,), "aligned binary truth required")
    require(np.asarray(ref["lengths"]).shape == (n,) and (np.asarray(ref["lengths"]) >= 0).all(), "aligned original word-like counts required")
    support = np.asarray(ref["language_supported"])
    require(support.shape == (n,) and np.isin(support, [0, 1]).all(), "aligned explicit language scope required")
    component_ids = {}
    for name, a in arrays.items():
        validate_raw_rows(a)
        for key in ("ids", "group_ids", "y_ai", "lengths", "language_supported", "scope_word_like_tokens"):
            require(np.array_equal(a[key], ref[key]), "binary raw cohorts are not aligned")
        require(np.array_equal(a["scope_word_like_tokens"], a["lengths"]) and
                np.asarray(a["language_rule_sha256"]).ndim == 0 and str(a["language_rule_sha256"].item()) == language_sha(), "raw language rule/word counts differ")
        raw = np.asarray(a["raw_binary"])
        require(raw.shape == (n, 2) and np.isfinite(raw).all(), "finite N-by2binary logits required")
        component_ids[name] = identity(a)
        require(blend.get("components", {}).get(name) == component_ids[name], "raw binary model differs from frozen development selection")
    for name in SOURCE_WEIGHTS:
        inherited = {"weights_sha256": source["provenance"]["component_weights_sha256"][name],
                     "metadata_sha256": source["provenance"]["component_metadata_sha256"][name]}
        require(blend.get("components", {}).get(name) == inherited, "the blend must prebind the unchanged source components")
        if name in component_ids:
            require(component_ids[name] == inherited, "a binary/source name collision changed the source encoder")
        component_ids[name] = inherited
    raw_inputs = {name: arrays[name]["raw_binary"] if name in arrays else np.zeros((n, 2)) for name in blend["names"]}
    return ref, blend_margin(blend, raw_inputs), component_ids


def fit_binary_only(blend, binary_arrays, source_calibration, *, blend_sha256, source_calibration_sha256,
                    protocol, protocol_sha256, calibration_input_sha256, raw_provenance=None):
    """Pure numeric core; CLI verifies every file/receipt before calling it.

    This fits synthetic arrays in checks or the registered mixed calibration.
    No source-fit function is called. Original fine trust masks are irrelevant
    to binary fitting and are never rewritten.
    """
    spec = validate_protocol(protocol, protocol_sha256, source_calibration_sha256)
    validate_source(source_calibration)
    require(is_sha(blend_sha256) and calibration_input_sha256 == protocol["calibration_input_sha256"], "registered calibration identity differs from the predeclared v2 policy")
    ref, margin, components = aligned_binary(blend, binary_arrays, source_calibration)
    support = np.asarray(ref["language_supported"], dtype=bool)
    core = support & (np.asarray(ref["lengths"]) >= 128)
    fit = core if spec["probability_fit_subset"] == "supported_core" else support
    require(fit.any() and core.any() and len(np.unique(ref["y_ai"][fit])) == len(np.unique(ref["y_ai"][core])) == 2,
            "both labels required in the frozen probability/threshold subsets")
    binary = BinaryPlatt().fit(margin[fit], ref["y_ai"][fit], split="calibration")
    p = binary.predict(margin)
    threshold_id = prediction_fingerprint(ref["ids"][core].tolist())
    ai = _majority_floor(fit_ai_threshold(ref["y_ai"][core], p[core], target_fpr=spec["AI_target_fpr"], calibration_id=threshold_id), ref["y_ai"][core], p[core])
    human = _majority_floor(fit_ai_threshold(1-ref["y_ai"][core], 1-p[core], target_fpr=spec["human_target_fpr"], calibration_id=threshold_id), 1-ref["y_ai"][core], 1-p[core])
    human.update(task="human_detection", negative_class="AI", decision_regions_exclusive_by_boolean_gate=True)
    for threshold in (ai, human):
        threshold["minimum_word_like_tokens"] = 128
        threshold["cohort"] = "Registered v2mixed calibration (fresh plus observed historical CAL replay), supported English core; empirical mixture error budget, not a per-domain guarantee"
    inherited = copy.deepcopy(source_objects(source_calibration))
    provenance = {
        "attribution_model": SOURCE_NAME, "blend_sha256": blend_sha256,
        "language_rule_sha256": language_sha(), "deployment_decision_protocol_sha256": _decision_protocol_sha256(),
        "v2_protocol_sha256": protocol_sha256, "calibration_input_sha256": calibration_input_sha256,
        "component_weights_sha256": {name: value["weights_sha256"] for name, value in components.items()},
        "component_metadata_sha256": {name: value["metadata_sha256"] for name, value in components.items()},
        "binary_calibration": {"input_sha256": calibration_input_sha256, "rows": len(ref["ids"]),
            "all_ids_sha256": prediction_fingerprint(ref["ids"].tolist()), "fit_subset": spec["probability_fit_subset"],
            "fit_rows": int(fit.sum()), "fit_ids_sha256": prediction_fingerprint(ref["ids"][fit].tolist()),
            "threshold_subset": "supported_core", "threshold_rows": int(core.sum()), "threshold_ids_sha256": threshold_id,
            "mixture": copy.deepcopy(protocol["calibration_mixture"]),
            "calibration_role_history": copy.deepcopy(protocol["calibration_role_history"]), "raw_exports": raw_provenance or {}},
        "source_calibration_inheritance": {"artifact_sha256": source_calibration_sha256,
            "original_calibration_id": source_calibration["provenance"].get("calibration_id"),
            "objects_sha256": digest(inherited), "refitted_on_v2_calibration": False,
            "historical_source_cutoffs_active": False, "source_assertions_enabled": False},
        "final_test_opened": False, "selection_after_calibration": False,
        "only_binary_fitted_on_registered_calibration": True,
    }
    value = {"schema_version": 1, "binary": asdict(binary), "hierarchy": inherited["hierarchy"],
        "version_refinement": inherited["version_refinement"],
        "thresholds": {"AI": ai, "human": human, "family": inherited["family_threshold"], "model": inherited["model_threshold"]},
        "provenance": provenance,
        "source_policy": {"mode": "rankings_only", "family_enabled": False, "model_enabled": False,
            "source_release_gate": None, "historical_cutoffs_are_inactive": True},
        "probability_scope": "Binary: registered v2mixture/prior including fresh sources and observed historical CAL replay; not wholly fresh or a per-domain FPR guarantee. Source conditional rankings: unchanged v1source temperatures/refinement, not revalidated on v2. No authorship authenticity guarantee."}
    require(digest(source_objects(value)) == digest(source_objects(source_calibration)), "an inherited source object changed")
    return value


def apply_rankings_only(blend, calibration, arrays):
    """Reuse frozen v1 arithmetic; disable only assertions, never probabilities."""
    validate_source(calibration)
    require(calibration.get("source_policy") == {"mode": "rankings_only", "family_enabled": False,
        "model_enabled": False, "source_release_gate": None, "historical_cutoffs_are_inactive": True}, "fixed v2rankings-only policy required")
    inherited = calibration["provenance"]["source_calibration_inheritance"]
    require(inherited["artifact_sha256"] == SOURCE_CALIBRATION_SHA256 and
            digest(source_objects(calibration)) == inherited["objects_sha256"] and inherited["refitted_on_v2_calibration"] is False and
            inherited["historical_source_cutoffs_active"] is False and inherited["source_assertions_enabled"] is False,
            "frozen source calibration objects changed")
    needed = set(binary_names(blend)) | set(SOURCE_WEIGHTS)
    require(set(arrays) == needed, "apply needs exactly the selected binary and frozen source components")
    reference = arrays[SOURCE_NAME]
    require(list(reference["family_labels"]) == FAMILIES and list(reference["model_labels"]) == MODEL_LABELS,
            "reference source class order changed")
    for a in arrays.values():
        validate_raw_rows(a)
        for key in ("ids", "group_ids", "y_ai", "lengths"):
            require(np.array_equal(a[key], reference[key]), "application component rows are not aligned")
    return apply_source_release(apply_calibration(blend, calibration, arrays, SOURCE_NAME), OFF)


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_json_bound(path, expected):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "explicit frozen JSON SHA256 mismatch")
    return json.loads(data)


def load_exports(named, purpose, input_path, input_sha256, *, blend_sha256=None, calibration_sha256=None):
    import pyarrow.parquet as pq
    require(sha(input_path) == input_sha256 and is_sha(input_sha256), "prepared input Parquet changed")
    source = pq.ParquetFile(input_path)
    columns = ["id", "y_ai", "split"] + (["eligible"] if "eligible" in source.schema.names else [])
    rows = source.read(columns=columns).to_pylist()
    roles = {r["split"] for r in rows}
    allowed_dev = {"validation", "dev", "dev_challenge", "dev_challenge_source_domain", "dev_backup", "dev_backup_allowed"}
    require((purpose == "calibration" and roles == {"calibration"}) or
            (purpose == "development" and roles and roles <= allowed_dev) or
            (purpose == "final" and roles and all(isinstance(r, str) and (r == "test" or r.startswith("challenge_") or r.startswith("final_sealed")) for r in roles)), "input role does not match the explicit operation")
    rows = [r for r in rows if r.get("eligible", True)]
    row_y = {r["id"]: r["y_ai"] for r in rows}
    require(len(row_y) == len(rows), "prepared eligible IDs must be unique")
    arrays, evidence = {}, {}
    for name, path in named.items():
        path = Path(path)
        receipt_path = path.with_suffix(".provenance.json")
        receipt = json.loads(receipt_path.read_text())
        checks = {"purpose": purpose, "input_sha256": input_sha256, "output_sha256": sha(path),
                  "eligible_export_rows": len(rows), "calibration_fitted": False}
        if purpose == "final":
            checks.update(frozen_blend_sha256=blend_sha256, frozen_calibration_sha256=calibration_sha256)
        require(all(receipt.get(key) == value for key, value in checks.items()), "raw receipt differs from frozen input/phase/artifacts")
        with np.load(path, allow_pickle=False) as archive:
            a = {key: archive[key] for key in archive.files}
        validate_raw_rows(a)
        require(str(np.asarray(a.get("export_purpose", "")).item()) == purpose, "raw array role differs from receipt")
        require(set(a["ids"].tolist()) == set(row_y) and len(a["ids"]) == len(row_y) and
                all(row_y[value] == label for value, label in zip(a["ids"], a["y_ai"])), "raw IDs/truth differ from prepared metadata")
        ids = identity(a)
        require(receipt.get("weights_sha256") == ids["weights_sha256"] and
                receipt.get("component_metadata_sha256") == ids["metadata_sha256"] and
                receipt.get("deployment_scope", {}).get("rule_sha256") == language_sha(), "raw receipt/model/scope identity differs")
        arrays[name], evidence[name] = a, {"path": str(path.resolve()), "sha256": checks["output_sha256"],
            "sidecar_sha256": sha(receipt_path), **ids}
    return arrays, evidence


def parse_named(values):
    pairs = [value.split("=", 1) for value in values]
    require(all(len(pair) == 2 and pair[0] and pair[1] for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs), "unique name=NPZ pairs required")
    return dict(pairs)


def evaluate_fixed(payload, calibration, *, n_boot):
    """Use the same frozen probabilities/cutoffs for a parent or registered view."""
    report = evaluate_npz_payload(payload, threshold=calibration["thresholds"]["AI"],
        human_threshold=calibration["thresholds"]["human"], n_boot=n_boot, seed=42)
    report["binary_at_probability_half"] = {
        "threshold": .5, "threshold_source": "constant default probability cutoff, never fitted",
        "all_lengths": _binary_metrics(payload["y_ai"], payload["p_ai"], .5, np.ones(len(payload["y_ai"]))),
        "core_ge_128_word_like_tokens": _binary_metrics(payload["y_ai"], payload["p_ai"], .5,
            (payload["lengths"] >= 128).astype(float))}
    report["source_policy"] = {"rankings_only": True, "source_assertions_enabled": False,
        "inherited_source_temperatures_revalidated_on_v2": False,
        "candidate_masks": "Inactive historical cutoff diagnostics only, not v2source-aim results."}
    return report


def registered_view_reports(protocol, input_path, input_sha256, result, calibration, *, n_boot):
    """Read only preregistered boolean metadata; never corpus text or new scores.

    Human controls may occur in multiple views. Each view retains its own whole
    connected groups and all abstentions; its counts/intervals cannot be added
    to another view or interpreted as independent evidence.
    """
    matches = [spec for spec in protocol.get("necessary_final_suites", {}).values()
               if spec.get("input_sha256") == input_sha256 and spec.get("required_views")]
    if not matches:
        return {}  # No extra Parquet columns are opened without a declaration.
    require(len(matches) == 1, "registered-view parent input must identify exactly one necessary suite")
    specs = matches[0]["required_views"]
    require(isinstance(specs, dict) and bool(specs) and n_boot == 500,
            "registered final views require a nonempty mapping and fixed500 bootstrap replicates")
    import pyarrow as pa
    import pyarrow.parquet as pq
    require(sha(input_path) == input_sha256, "registered-view parent Parquet changed")
    source = pq.ParquetFile(input_path)
    columns = []
    for name, spec in specs.items():
        column = spec.get("column")
        require(isinstance(name, str) and bool(name.strip()) and isinstance(column, str) and
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column) is not None and
                column in source.schema_arrow.names and pa.types.is_boolean(source.schema_arrow.field(column).type),
                "each registered view must name an existing boolean metadata column")
        columns.append(column)
    projection = ["id", "y_ai"] + (["eligible"] if "eligible" in source.schema.names else []) + list(dict.fromkeys(columns))
    rows = source.read(columns=projection).to_pylist()
    require(sha(input_path) == input_sha256, "registered-view Parquet changed during metadata projection")
    rows = [r for r in rows if r.get("eligible", True)]
    by_id = {r["id"]: r for r in rows}
    require(len(by_id) == len(rows) == len(result["ids"]) and set(by_id) == set(result["ids"]),
            "view metadata IDs differ from the complete parent shipping rows")
    require(all(by_id[row_id]["y_ai"] == y for row_id, y in zip(result["ids"], result["y_ai"])),
            "view metadata truth differs from parent shipping rows")
    row_keys = ("ids", "y_ai", "group_ids", "sources", "domains", "lengths", "family_true", "p_family", "mask_family",
                "model_true", "p_model", "mask_model", "family_accept", "model_accept", "language_supported", "p_ai")
    output = {}
    for name, spec in specs.items():
        flags = [by_id[row_id][spec["column"]] for row_id in result["ids"]]
        require(all(isinstance(flag, bool) for flag in flags), "registered boolean view membership must not be null")
        selected = np.asarray(flags, dtype=bool)
        groups = np.asarray(result["group_ids"])[selected]
        core = selected & (np.asarray(result["lengths"]) >= 128)
        counts = {"n": int(core.sum()), "human_n": int((core & (result["y_ai"] == 0)).sum()),
                  "AI_n": int((core & (result["y_ai"] == 1)).sum()),
                  "groups": int(len(np.unique(result["group_ids"][core])))}
        require(int(selected.sum()) == spec.get("rows") and len(np.unique(groups)) == spec.get("groups") and
                counts == spec.get("core_counts") and counts["human_n"] > 0 and counts["AI_n"] > 0,
                "registered-view rows/groups/core class counts differ from the frozen policy")
        payload = {key: np.asarray(result[key])[selected] for key in row_keys if key in result}
        payload.update({key: result[key] for key in ("family_labels", "model_labels") if key in result})
        report = evaluate_fixed(payload, calibration, n_boot=n_boot)
        report["registered_view"] = {"column": spec["column"], "parent_input_sha256": input_sha256,
            "rows": int(selected.sum()), "groups": int(len(np.unique(groups))), "core_counts": counts,
            "membership_source": "pre-registered boolean column in the SHA-bound parent Parquet; aligned by ID",
            "shared_human_controls_not_independent_or_additive": True,
            "probabilities_or_thresholds_refitted": False}
        output[name] = report
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "evaluate"))
    parser.add_argument("--blend", required=True, type=Path)
    parser.add_argument("--source-calibration", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--purpose", choices=("development", "final"))
    parser.add_argument("--n-bootstrap", type=int, default=500)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "outputs must use a new nonexistent directory")
    require(args.n_bootstrap >= 0, "nonnegative bootstrap count required")
    protocol = load_json_bound(args.protocol, args.protocol_sha256)
    validate_protocol(protocol, args.protocol_sha256, SOURCE_CALIBRATION_SHA256)
    blend = json.loads(args.blend.read_text())
    binary_names(blend)
    if args.command == "fit":
        require(args.source_calibration is not None and args.calibration is None and args.purpose is None, "fit requires only the inherited source calibration artifact")
        require(args.input_sha256 == protocol["calibration_input_sha256"], "registered calibration input differs from frozen v2policy")
        source = load_json_bound(args.source_calibration, SOURCE_CALIBRATION_SHA256)
        arrays, evidence = load_exports(parse_named(args.predictions), "calibration", args.input, args.input_sha256)
        result = fit_binary_only(blend, arrays, source, blend_sha256=sha(args.blend),
            source_calibration_sha256=SOURCE_CALIBRATION_SHA256, protocol=protocol, protocol_sha256=args.protocol_sha256,
            calibration_input_sha256=args.input_sha256, raw_provenance=evidence)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_new(args.output_dir / "calibration.json", result)
        manifest = {"operation": "only_new_binary_probability_and_two_decision_cutoffs_fitted", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_sha256": sha(args.output_dir / "calibration.json"), "calibration_input_sha256": args.input_sha256,
            "v2_protocol_sha256": args.protocol_sha256, "old_source_calibration_sha256": SOURCE_CALIBRATION_SHA256,
            "source_objects_sha256": digest(source_objects(result)), "source_objects_unchanged": True,
            "source_assertions_enabled": False, "code_sha256": sha(__file__), "raw_exports": evidence}
        write_new(args.output_dir / "fit_manifest.json", manifest)
    else:
        require(args.calibration is not None and args.source_calibration is None and args.purpose is not None, "evaluate requires the frozen new calibration and explicit purpose")
        if args.purpose == "final":
            require(args.input_sha256 in protocol.get("final_input_sha256s", []), "final input must be preregistered in the frozen v2policy")
        calibration = json.loads(args.calibration.read_text())
        require(calibration["provenance"]["v2_protocol_sha256"] == args.protocol_sha256 and
                calibration["provenance"]["blend_sha256"] == sha(args.blend), "calibration differs from frozen v2policy/blend")
        named = parse_named(args.predictions)
        arrays, evidence = load_exports(named, args.purpose, args.input, args.input_sha256,
            blend_sha256=sha(args.blend), calibration_sha256=sha(args.calibration))
        needed = set(binary_names(blend)) | set(SOURCE_WEIGHTS)
        require(set(arrays) == needed, "evaluation needs exactly the new binary and original source components")
        result = apply_rankings_only(blend, calibration, arrays)
        report = evaluate_fixed(result, calibration, n_boot=args.n_bootstrap)
        report["registered_views"] = registered_view_reports(protocol, args.input, args.input_sha256,
            result, calibration, n_boot=args.n_bootstrap)
        report["provenance"] = {"blend_sha256": sha(args.blend), "calibration_sha256": sha(args.calibration),
            "input_sha256": args.input_sha256, "v2_protocol_sha256": args.protocol_sha256,
            "old_source_calibration_sha256": SOURCE_CALIBRATION_SHA256,
            "thresholds_refitted_on_evaluation": False, "raw_exports": evidence, "code_sha256": sha(__file__)}
        args.output_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(args.output_dir / "shipping.npz", **result)
        write_new(args.output_dir / "report.json", report)
    print(json.dumps({"completed": True, "operation": args.command, "output": str(args.output_dir.resolve()),
                      "only_binary_calibration_fitted": args.command == "fit", "source_assertions_enabled": False}))


if __name__ == "__main__":
    main()
