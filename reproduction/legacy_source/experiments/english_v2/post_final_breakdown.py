"""Fixed descriptive slices AFTER all four necessary final reports complete.

No fitting, threshold selection, source assertion, model loading or text reads.
Only original shipping probabilities, prepared metadata and bound raw exports
are used. These overlapping slices are diagnostics, never new acceptance gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiments.english_v1 import runtime  # noqa: F401
import numpy as np
import pyarrow.parquet as pq
from experiments.english_v2 import binary_finalize as bf

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("fresh_main", "fresh_source_domain", "fresh_modern_with_human_controls", "fresh_paraphrase_with_human_controls")
MODERN, PARA = SUITES[2:]
META = ("id", "group_id", "native_parent_id", "y_ai", "wordlike_tokens", "eligible", "split",
        "source", "domain", "collector", "variant", "config", "original_generator",
        "view_human_origin_rewrite", "view_AI_origin_rewrite")
BINARY_ROWS = ("ids", "y_ai", "group_ids", "sources", "domains", "lengths", "language_supported", "p_ai")
CONTRACT = {
    "schema_version": 1,
    "purpose": "post-final descriptive diagnostics only; no additional acceptance gate or tuning",
    "execution": "Once, only after all four necessary report.json plus shipping.npz and raw receipts are complete; new output directory, no overwrite",
    "modern_axes": ["nominal_original_generator", "AI_configuration", "source_domain"],
    "generator_categories": "All nonempty author nominal original_generator values among prepared final-modern AI rows, sorted by string; never grouped by score",
    "configuration_categories": {"MGB_direct": {"config": "MGB", "variant": "direct"},
        "MAGA_fused_alignment": {"config": "MAGA", "variant": "fused_alignment"}},
    "source_domain_categories": "All (source,domain) pairs among prepared final-modern AI rows, sorted; no score-based regrouping",
    "paraphrase_cells": [{"domain": d, "origin": o} for d in ("books", "news")
                         for o in ("human_origin_DIPPER", "AI_origin_DIPPER")],
    "controls": "Define each stratum by its AI rows, then add the unique original y=0 row with the same native_parent_id. Never filter the human control by its own model/config tag.",
    "missing_controls": "All selected AI remain in all-AI recall. Paired AUC/FPR uses only AI with their actual native-parent H controls. Unpaired AI receive recall/abstention diagnostics only, never AUC/FPR/accuracy.",
    "bootstrap": {"n_boot": 500, "seed": 42, "unit": "unchanged whole connected group_id", "CI": "percentile 95%; conditional on the frozen predictions, calibration and sampled groups"},
    "scope": "Unsupported English and short rows stay in all-row denominators; unchanged core>=128 and language/exclusive decisions are computed by the existing evaluator",
    "nonindependence": "Modern axes and rewrite origins reuse controls/AI/groups; never sum counts or intervals across slices. No multiplicity-adjusted or population guarantee is claimed.",
    "source_labels": "Publisher nominal generator and configuration claims only; aligned custom weights are not authenticated stock versions",
    "required_views": "The two registered paraphrase acceptance views remain exclusively the original finalizer/acceptance responsibility",
    "prohibited": ["refit", "drop_language_unsupported_rows", "select_model_or_blend", "change_cutoffs", "new_acceptance_pass_fail", "replace_required_reports"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read_bound(path, expected=None):
    p = resolve(path)
    if expected is not None:
        require(bf.is_sha(expected) and sha(p) == expected, "frozen file fingerprint changed")
    return json.loads(p.read_text())


def _scalar(a, name):
    value = np.asarray(a[name])
    require(value.shape == (), "scalar header required: " + name)
    return value.item()


def _same_array(left, right, key):
    a, b = np.asarray(left), np.asarray(right)
    require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), "shipping/raw replay mismatch: " + key)


def thin(payload, mask):
    return {key: np.asarray(payload[key])[mask] for key in BINARY_ROWS}


def recall_only(report):
    """An AI-only set cannot produce a binary accuracy, AUC or human FPR."""
    output = {"scope": "AI-only recall/abstention; no human FPR, AUC or accuracy estimate",
              "threshold": report["threshold"], "length_unit": report["length_unit"]}
    for section in ("all_lengths", "core_ge_128_tokens"):
        point = report[section]
        b, d = point["binary"], point["deployed_decision_audit"]
        output[section] = {
            "fixed_AI_threshold": {k: b[k] for k in ("n", "human_n", "AI_n", "tp", "fn", "tpr")},
            "deployed_decision_audit": {k: d[k] for k in ("n", "AI_n", "language_supported_n", "language_unsupported_n",
                "correct_AI_n", "correct_AI_recall", "AI_misclassified_as_human_n", "AI_misclassified_as_human_rate",
                "inconclusive_AI_n", "inconclusive_coverage", "predicted_AI_n", "predicted_human_n", "inconclusive_n")}}
    bootstrap = report["group_bootstrap"]
    allowed = ("binary.tpr", "deployed_decision_audit.correct_AI_recall",
               "deployed_decision_audit.AI_misclassified_as_human_rate", "deployed_decision_audit.inconclusive_coverage")
    output["group_bootstrap"] = {key: value for key, value in bootstrap.items() if key != "intervals95"}
    output["group_bootstrap"]["intervals95"] = {key: value for key, value in bootstrap["intervals95"].items()
        if key in allowed or key.startswith("core_ge_128_tokens.") and key[len("core_ge_128_tokens."):] in allowed}
    half = report["binary_at_probability_half"]
    output["AI_recall_at_probability_half"] = {key: {k: half[key][k] for k in ("n", "human_n", "AI_n", "tp", "fn", "tpr")}
        for key in ("all_lengths", "core_ge_128_word_like_tokens")}
    return output


def slice_definitions(records, suite):
    """Only metadata defines cuts; scores, language support and length are absent."""
    positives = [r for r in records if r["y_ai"] == 1]
    if suite == MODERN:
        require(all(isinstance(r["original_generator"], str) and r["original_generator"].strip() for r in positives), "nominal generator metadata required")
        result = []
        for label in sorted({r["original_generator"] for r in positives}):
            result.append({"axis": "nominal_original_generator", "value": label,
                           "AI_ids": {r["id"] for r in positives if r["original_generator"] == label}})
        for label, spec in CONTRACT["configuration_categories"].items():
            result.append({"axis": "AI_configuration", "value": label,
                "AI_ids": {r["id"] for r in positives if r["config"] == spec["config"] and r["variant"] == spec["variant"]}})
        require(all(any(r["config"] == spec["config"] and r["variant"] == spec["variant"]
                    for spec in CONTRACT["configuration_categories"].values()) for r in positives), "unknown modern AI configuration/variant combination")
        for source, domain in sorted({(r["source"], r["domain"]) for r in positives}):
            result.append({"axis": "source_domain", "value": {"source": source, "domain": domain},
                "AI_ids": {r["id"] for r in positives if (r["source"], r["domain"]) == (source, domain)}})
        return result
    require(suite == PARA, "only registered modern/paraphrase suites have these diagnostic slices")
    require(all(r["domain"] in ("books", "news") and r["variant"] in ("human_origin_DIPPER", "AI_origin_DIPPER") for r in positives), "unknown paraphrase origin/domain")
    return [{"axis": "domain_by_rewrite_origin", "value": cell,
        "AI_ids": {r["id"] for r in positives if r["domain"] == cell["domain"] and r["variant"] == cell["origin"]}}
        for cell in CONTRACT["paraphrase_cells"]]


def describe_slices(records, payload, calibration, suite):
    """Pure fixed-probability arithmetic; always500/42 bootstrap, never fit."""
    by_id = {r["id"]: r for r in records}
    require(len(by_id) == len(records) and set(by_id) == set(payload["ids"].tolist()), "diagnostic metadata/shipping IDs differ")
    controls = {}
    for r in records:
        if r["y_ai"] == 0:
            require(r["native_parent_id"] not in controls, "multiple original human controls for one native parent")
            controls[r["native_parent_id"]] = r
    result = []
    ids = np.asarray(payload["ids"])
    for spec in slice_definitions(records, suite):
        selected = spec["AI_ids"]
        public = {k: v for k, v in spec.items() if k != "AI_ids"}
        if not selected:
            result.append({**public, "status": "empty_metadata_stratum", "AI_n": 0, "human_n": 0})
            continue
        paired_AI, human_ids = set(), set()
        for identifier in selected:
            row = by_id[identifier]
            human = controls.get(row["native_parent_id"])
            if human is not None:
                require(human["group_id"] == row["group_id"] and (human["source"], human["domain"]) == (row["source"], row["domain"]), "native parent human/control group or source differs")
                paired_AI.add(identifier)
                human_ids.add(human["id"])
        unpaired = selected - paired_AI
        all_AI_mask = np.isin(ids, sorted(selected))
        all_AI = recall_only(bf.evaluate_fixed(thin(payload, all_AI_mask), calibration, n_boot=500))
        comparison = None
        if paired_AI:
            comparison = bf.evaluate_fixed(thin(payload, np.isin(ids, sorted(paired_AI | human_ids))), calibration, n_boot=500)
        unmatched = recall_only(bf.evaluate_fixed(thin(payload, np.isin(ids, sorted(unpaired))), calibration, n_boot=500)) if unpaired else None
        result.append({**public, "status": "paired" if not unpaired else "partially_paired" if paired_AI else "AI_only_no_human_controls",
            "AI_n": len(selected), "human_n": len(human_ids), "paired_AI_n": len(paired_AI), "unpaired_AI_n": len(unpaired),
            "native_parent_n": len({by_id[i]["native_parent_id"] for i in selected}),
            "selected_AI_ids_sha256": json_hash(sorted(selected)), "selected_H_ids_sha256": json_hash(sorted(human_ids)),
            "small_group_diagnostic": len({by_id[i]["group_id"] for i in selected}) < 30,
            "paired_controls_report": comparison, "all_AI_recall": all_AI, "unpaired_AI_recall": unmatched,
            "not_an_acceptance_gate": True, "controls_and_intervals_not_additive_across_slices": True})
    return result


def prepare(protocol_path, expected_protocol_sha, blend_path, expected_blend_sha, calibration_path, expected_cal_sha, reports):
    """Validate all FOUR completion reports before loading any score arrays."""
    protocol = read_bound(protocol_path, expected_protocol_sha)
    bf.validate_protocol(protocol, expected_protocol_sha, bf.SOURCE_CALIBRATION_SHA256)
    require(set(reports) == set(SUITES), "all four necessary final reports must already exist")
    require(all(resolve(reports[suite]).is_file() and resolve(reports[suite]).with_name("shipping.npz").is_file()
                for suite in SUITES), "all four report/shipping pairs must exist before inspecting a final report")
    blend, cal = read_bound(blend_path, expected_blend_sha), read_bound(calibration_path, expected_cal_sha)
    require(cal["provenance"]["blend_sha256"] == expected_blend_sha and cal["provenance"]["v2_protocol_sha256"] == expected_protocol_sha,
            "frozen blend/calibration/protocol mismatch")
    require(cal["provenance"]["language_rule_sha256"] == protocol["language_rule_sha256"] and
            cal["provenance"]["deployment_decision_protocol_sha256"] == protocol["deployment_decision_protocol_sha256"] and
            cal["source_policy"]["mode"] == "rankings_only" and cal["source_policy"]["family_enabled"] is False and
            cal["source_policy"]["model_enabled"] is False, "frozen scope/decision/rankings-only policy differs")
    code = protocol["code_registry"]
    for name in ("experiments/english_v2/binary_finalize.py", "experiments/english_v1/evaluate.py", "experiments/english_v1/finalize.py"):
        require(sha(ROOT / name) == code[name], "frozen evaluation producer changed")
    prepared = read_bound(protocol["prepared_manifest"]["path"], protocol["prepared_manifest"]["sha256"])
    mapping = read_bound(protocol["dataset_key_mapping"]["path"], protocol["dataset_key_mapping"]["sha256"])
    needed = set(bf.binary_names(blend)) | set(bf.SOURCE_WEIGHTS)
    require(set(cal["provenance"]["component_weights_sha256"]) == set(cal["provenance"]["component_metadata_sha256"]) == needed,
            "calibration component directory differs")
    for name in needed:
        require(blend["components"][name] == {"weights_sha256": cal["provenance"]["component_weights_sha256"][name],
            "metadata_sha256": cal["provenance"]["component_metadata_sha256"][name]}, "selected component identity drift")
    evidence, artifacts = {}, {}
    for suite in SUITES:
        report_path = resolve(reports[suite])
        shipping_path = report_path.with_name("shipping.npz")
        require(report_path.is_file() and shipping_path.is_file(), "four completed report/shipping pairs required before diagnostics")
        report = read_bound(report_path)
        spec = protocol["necessary_final_suites"][suite]
        entry = prepared["output_files"][mapping["final"][suite]]
        require(entry["sha256"] == spec["input_sha256"] and entry["rows"] == spec["rows"] and sha(resolve(entry["path"])) == entry["sha256"], "prepared final input identity differs")
        prov = report["provenance"]
        require(all(prov.get(k) == v for k, v in {"blend_sha256": expected_blend_sha, "calibration_sha256": expected_cal_sha,
            "v2_protocol_sha256": expected_protocol_sha, "input_sha256": spec["input_sha256"], "thresholds_refitted_on_evaluation": False,
            "old_source_calibration_sha256": bf.SOURCE_CALIBRATION_SHA256, "code_sha256": code["experiments/english_v2/binary_finalize.py"]}.items()), "necessary report is not the frozen v2 finalizer output")
        require(report.get("source_policy", {}).get("source_assertions_enabled") is False and report["threshold"] == cal["thresholds"]["AI"], "report changed source mode or frozen threshold")
        boot = report["group_bootstrap"]
        require(boot["n_boot"] == 500 and boot["seed"] == 42 and boot["n_groups"] == spec["groups"] and boot["core_groups"] == spec["core_counts"]["groups"], "all necessary reports require original whole-group500/42 intervals")
        require(set(prov["raw_exports"]) == needed, "necessary report raw component set differs")
        raw_paths, raw_evidence = {}, {}
        for name, proof in prov["raw_exports"].items():
            raw = resolve(proof["path"])
            side_path = raw.with_suffix(".provenance.json")
            require(sha(raw) == proof["sha256"] and sha(side_path) == proof["sidecar_sha256"], "raw evidence changed after final report")
            side = read_bound(side_path)
            expected = {"purpose": "final", "input_sha256": entry["sha256"], "output_sha256": proof["sha256"],
                "original_rows": entry["rows"], "eligible_export_rows": entry["rows"], "excluded_ineligible_rows": 0,
                "calibration_fitted": False, "frozen_blend_sha256": expected_blend_sha, "frozen_calibration_sha256": expected_cal_sha,
                "weights_sha256": cal["provenance"]["component_weights_sha256"][name],
                "component_metadata_sha256": cal["provenance"]["component_metadata_sha256"][name]}
            require(all(side.get(k) == v for k, v in expected.items()) and side["deployment_scope"]["rule_sha256"] == protocol["language_rule_sha256"], "raw final role/scope/model/denominator/freeze differs")
            require(proof.get("weights_sha256") == expected["weights_sha256"] and
                    proof.get("metadata_sha256") == expected["component_metadata_sha256"], "report component fingerprint declarations differ")
            require(side["code_sha256"] == {n: code["experiments/english_v1/" + n] for n in
                ("export_component.py", "neural.py", "statistical.py", "catalog.py", "language_guard.py")}, "raw exporter source registry differs")
            raw_paths[name] = raw
            raw_evidence[name] = {"sha256": proof["sha256"], "sidecar_sha256": proof["sidecar_sha256"],
                "weights_sha256": expected["weights_sha256"], "metadata_sha256": expected["component_metadata_sha256"]}
            artifacts[str(raw)] = proof["sha256"]
            artifacts[str(side_path)] = proof["sidecar_sha256"]
        evidence[suite] = {"report_path": report_path, "shipping_path": shipping_path, "report": report,
            "input": resolve(entry["path"]), "spec": spec, "raw_paths": raw_paths,
            "hashes": {"report_sha256": sha(report_path), "shipping_sha256": sha(shipping_path),
                       "input_sha256": entry["sha256"], "raw_exports": raw_evidence}}
        artifacts.update({str(report_path): sha(report_path), str(shipping_path): sha(shipping_path), str(resolve(entry["path"])): entry["sha256"]})
    artifacts.update({str(resolve(protocol_path)): expected_protocol_sha, str(resolve(blend_path)): expected_blend_sha,
        str(resolve(calibration_path)): expected_cal_sha, str(resolve(protocol["prepared_manifest"]["path"])): protocol["prepared_manifest"]["sha256"],
        str(resolve(protocol["dataset_key_mapping"]["path"])): protocol["dataset_key_mapping"]["sha256"]})
    return {"protocol": protocol, "blend": blend, "calibration": cal, "calibration_sha256": expected_cal_sha,
            "evidence": evidence, "artifacts": artifacts}


def load_shipping(context, suite):
    evidence, cal = context["evidence"][suite], context["calibration"]
    records = pq.read_table(evidence["input"], columns=list(META)).to_pylist()
    by_id = {r["id"]: r for r in records}
    require(len(by_id) == len(records) == evidence["spec"]["rows"] and all(r["eligible"] is True and r["split"] == "final_sealed" for r in records), "complete frozen final metadata required")
    arrays, _ = bf.load_exports(evidence["raw_paths"], "final", evidence["input"], evidence["spec"]["input_sha256"],
        blend_sha256=cal["provenance"]["blend_sha256"], calibration_sha256=context["calibration_sha256"])
    replay = bf.apply_rankings_only(context["blend"], cal, arrays)
    with np.load(evidence["shipping_path"], allow_pickle=False) as archive:
        shipping = {key: archive[key] for key in archive.files}
    require(set(shipping) == set(replay), "shipping array keys differ from immutable raw replay")
    for key in replay:
        _same_array(shipping[key], replay[key], key)
    require(_scalar(shipping, "calibrated") is True and _scalar(shipping, "export_purpose") == "final", "completed calibrated shipping output required")
    ids = np.asarray(shipping["ids"])
    require(ids.shape == (len(records),) and ids.dtype.kind == "U" and set(ids.tolist()) == set(by_id), "shipping IDs differ from complete metadata")
    for key, field in (("group_ids", "group_id"), ("y_ai", "y_ai"), ("lengths", "wordlike_tokens"), ("sources", "source"), ("domains", "domain")):
        require(np.array_equal(shipping[key], np.asarray([by_id[i][field] for i in ids])), "shipping prepared row alignment differs: " + key)
    require(np.array_equal(shipping["lengths"], shipping["scope_word_like_tokens"]) and
            np.asarray(shipping["language_supported"]).dtype.kind == "b" and
            _scalar(shipping, "language_rule_sha256") == context["protocol"]["language_rule_sha256"], "shipping scope/length/rule differs")
    require(not shipping["family_accept"].any() and not shipping["model_accept"].any(), "descriptive diagnostics cannot enable source assertions")
    point = bf.evaluate_fixed(thin(shipping, np.ones(len(ids), dtype=bool)), cal, n_boot=0)
    for section in ("all_lengths", "core_ge_128_tokens"):
        for head in ("binary", "deployed_decision_audit"):
            require(point[section][head] == evidence["report"][section][head], "necessary report point metrics differ from shipping arithmetic")
    return records, shipping


def build(protocol_path, protocol_sha, blend_path, blend_sha, calibration_path, calibration_sha, reports):
    started = datetime.now(timezone.utc).isoformat()
    context = prepare(protocol_path, protocol_sha, blend_path, blend_sha, calibration_path, calibration_sha, reports)
    diagnostics = {}
    for suite in SUITES:
        records, shipping = load_shipping(context, suite)
        if suite in (MODERN, PARA):
            diagnostics[suite] = describe_slices(records, shipping, context["calibration"], suite)
    require(all(sha(path) == expected for path, expected in context["artifacts"].items()), "input/report/raw artifacts changed during diagnostic execution")
    return {"schema_version": 1, "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "descriptive_only": True, "new_acceptance_gate": False, "no_parameters_or_thresholds_fitted": True,
        "all_four_necessary_reports_complete_before_reading_scores": True, "contract": CONTRACT,
        "diagnostics": diagnostics, "provenance": {"v2_protocol_sha256": protocol_sha, "blend_sha256": blend_sha,
            "calibration_sha256": calibration_sha, "language_rule_sha256": context["protocol"]["language_rule_sha256"],
            "deployment_decision_protocol_sha256": context["protocol"]["deployment_decision_protocol_sha256"],
            "helper_code_sha256": sha(__file__), "contract_sha256": json_hash(CONTRACT),
            "all_four_final_artifact_evidence": {suite: evidence["hashes"] for suite, evidence in context["evidence"].items()}},
        "limitations": ["Nominal generator/configuration strata are descriptive dataset labels, not source authenticity or stock-weight certification.",
            "Multiple overlapping slices and controls are not independent/additive; no multiplicity correction or population guarantee is claimed.",
            "Intervals condition on the fixed selected models, calibration and evaluation groups, excluding training/selection/calibration uncertainty.",
            "These diagnostics cannot replace, relax or reinterpret any original necessary-suite or registered rewrite-view acceptance result."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--blend-sha256", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--reports", nargs="+", required=True, help="all four suite=report.json pairs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "one new diagnostic output directory required; never overwrite")
    result = build(args.protocol, args.protocol_sha256, args.blend, args.blend_sha256,
        args.calibration, args.calibration_sha256, bf.parse_named(args.reports))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bf.write_new(args.output_dir / "breakdown.json", result)
    print(json.dumps({"completed": True, "descriptive_only": True, "output": str((args.output_dir / "breakdown.json").resolve())}))
