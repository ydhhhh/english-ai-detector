"""Check every preregistered necessary v2 final suite independently.

This reads bound aggregate reports and receipts only, never labels or scores.
No failed modern/rewriting suite can be hidden by a pooled average. This is
project evidence under the registered point criteria, not a population claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from experiments.english_v1.release_acceptance import (
    _binary_consistency, _count, _get, _hash, _json, _online_consistency, _prob, _sha,
)

REQUIRED_SUITES = ("fresh_main", "fresh_source_domain", "fresh_modern_with_human_controls", "fresh_paraphrase_with_human_controls")
BOOTSTRAP = {"n_boot": 500, "seed": 42,
    "method": "percentile whole-group bootstrap; same resampling weights across tasks; fixed thresholds",
    "minimum_valid_fraction": 0.95}
METRICS = {
    "core_AUROC_min": ("binary", "auroc", ">="),
    "core_balanced_accuracy_at_probability_half_min": ("half", "balanced_accuracy", ">="),
    "fixed_AI_human_FPR_max": ("binary", "fpr", "<="),
    "fixed_AI_recall_min": ("binary", "tpr", ">="),
    "online_human_as_AI_rate_max": ("online", "human_misclassified_as_AI_rate", "<="),
    "online_AI_as_human_rate_max": ("online", "AI_misclassified_as_human_rate", "<="),
    "online_correct_AI_recall_min": ("online", "correct_AI_recall", ">="),
    "online_correct_human_recall_min": ("online", "correct_human_recall", ">="),
    "online_decided_coverage_min": ("online", "decided_coverage", ">="),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aggregate_checks(spec, report, calibration, boot_spec, name):
    require(set(spec["targets"]) == set(METRICS) and all(_prob(x) for x in spec["targets"].values()), "complete necessary metric targets required")
    require(all(_count(spec.get(key)) and spec[key] > 0 for key in ("rows", "groups")) and
        all(_count(spec.get("core_counts", {}).get(key)) and spec["core_counts"][key] > 0 for key in ("n", "human_n", "AI_n", "groups")), "positive registered row/group/core denominators required")
    errors = []
    if report.get("length_unit") not in {"word_like_tokens", "word_like_tokens_regex"}:
        errors.append("original word-like token length units required")
    if report.get("source_policy", {}).get("source_assertions_enabled") is not False:
        errors.append("report did not apply rankings-only source closure")
    core, all_lengths = report.get("core_ge_128_tokens", {}), report.get("all_lengths", {})
    _binary_consistency(core, errors, name + ":core")
    _online_consistency(core, errors, name + ":core")
    _binary_consistency(all_lengths, errors, name + ":all")
    _online_consistency(all_lengths, errors, name + ":all")
    if _get(all_lengths, "binary", "n") != spec["rows"]:
        errors.append("complete frozen input denominator missing")
    for key in ("n", "human_n", "AI_n"):
        if _get(core, "binary", key) != spec["core_counts"][key]:
            errors.append("frozen core denominator differs: " + key)
    for point in (core, all_lengths):
        for head in ("family", "model"):
            audit = _get(point, head, "assertion_audit") or {}
            if any(audit.get(key) != 0 for key in ("asserted_on_humans_n", "asserted_on_trusted_AI_n", "asserted_on_AI_without_trusted_source_label_n")):
                errors.append("source assertions were not globally disabled: " + head)
    if report.get("threshold") != calibration["thresholds"]["AI"]:
        errors.append("report changed the common frozen AI threshold")
    half = report.get("binary_at_probability_half", {})
    if half.get("threshold") != 0.5 or half.get("threshold_source") != "constant default probability cutoff, never fitted":
        errors.append("fixed probability-half metric missing")
    half_core = half.get("core_ge_128_word_like_tokens", {})
    _binary_consistency({"binary": half_core}, errors, name + ":half")
    if any(half_core.get(key) != _get(core, "binary", key) for key in ("n", "human_n", "AI_n")):
        errors.append("probability-half denominator differs")
    points = {"binary": core.get("binary", {}), "online": core.get("deployed_decision_audit", {}), "half": half_core}
    checks = []
    for metric, (section, key, op) in METRICS.items():
        value, target = points[section].get(key), spec["targets"][metric]
        passed = _prob(value) and (value >= target if op == ">=" else value <= target)
        checks.append({"name": metric, "actual": value, "target": target, "comparison": op, "pass": bool(passed)})
    # Independently recompute the extra directional rates, with every true
    # class example retained even when the deployed tool abstains.
    online = points["online"]
    for rate, numerator, denominator in (("AI_misclassified_as_human_rate", "AI_misclassified_as_human_n", "AI_n"),
        ("correct_human_recall", "correct_human_n", "human_n"), ("decided_coverage", None, "n")):
        den = online.get(denominator)
        num = online.get(numerator) if numerator else (online.get("predicted_AI_n", 0) + online.get("predicted_human_n", 0))
        if not _count(den) or not den or not _count(num) or not _prob(online.get(rate)) or not math.isclose(online[rate], num / den, abs_tol=1e-10):
            errors.append("online rate does not use all true-class examples: " + rate)
    bootstrap = report.get("group_bootstrap") or {}
    if any(bootstrap.get(key) != boot_spec[key] for key in ("n_boot", "seed", "method")):
        errors.append("registered whole-component uncertainty method missing")
    if bootstrap.get("n_groups") != spec["groups"] or bootstrap.get("core_groups") != spec["core_counts"]["groups"]:
        errors.append("bootstrap group denominator differs from frozen role metadata")
    for metric, (section, key, _) in METRICS.items():
        if section == "half":
            continue  # Fixed-half BA is a point criterion; no interval is fabricated.
        report_key = "binary" if section == "binary" else "deployed_decision_audit"
        interval = bootstrap.get("intervals95", {}).get(f"core_ge_128_tokens.{report_key}.{key}") or {}
        value = points[section].get(key)
        valid = interval.get("valid_replicates")
        if (not all(_prob(interval.get(k)) for k in ("point", "low", "high")) or
                interval.get("point") != value or interval["low"] > interval["high"] or
                not _count(valid) or not math.ceil(boot_spec["n_boot"] * boot_spec["minimum_valid_fraction"]) <= valid <= boot_spec["n_boot"]):
            errors.append("required conditional 95% group interval missing/invalid: " + metric)
    return checks, errors


def summarize(protocol_path, expected_protocol_sha, blend_path, calibration_path, report_paths):
    require(_hash(expected_protocol_sha) and _sha(protocol_path) == expected_protocol_sha, "preregistered protocol SHA mismatch")
    protocol, calibration = _json(protocol_path), _json(calibration_path)
    require(protocol.get("status") == "FROZEN_BEFORE_V2_TRAINING_AND_SCORES", "pretraining frozen protocol required")
    require(protocol.get("acceptance", {}).get("criterion") == "point_estimates_all_checks_with_group_CI_disclosed", "explicit frozen point-versus-CI criteria required")
    require(protocol["acceptance"].get("bootstrap") == BOOTSTRAP, "the fixed 500-replicate whole-component uncertainty contract is required")
    require(set(protocol["necessary_final_suites"]) == set(REQUIRED_SUITES) and set(report_paths) == set(REQUIRED_SUITES), "all four necessary suites must be reported")
    blend_sha, cal_sha = _sha(blend_path), _sha(calibration_path)
    require(calibration["provenance"]["blend_sha256"] == blend_sha and calibration["provenance"]["v2_protocol_sha256"] == expected_protocol_sha, "calibration belongs to another selection/protocol")
    require(calibration["provenance"].get("only_binary_fitted_on_registered_calibration") is True and
        calibration["source_policy"]["mode"] == "rankings_only" and not calibration["source_policy"]["family_enabled"] and not calibration["source_policy"]["model_enabled"], "fixed rankings-only calibration required")
    result = {}
    for name in REQUIRED_SUITES:
        spec, report = protocol["necessary_final_suites"][name], _json(report_paths[name])
        errors = []
        require(spec.get("required_for_practical_acceptance") is True, "necessary suite cannot become optional")
        require(set(spec["targets"]) == set(METRICS) and all(_prob(x) for x in spec["targets"].values()), "complete necessary metric targets required")
        for key, expected in (("blend_sha256", blend_sha), ("calibration_sha256", cal_sha),
                              ("v2_protocol_sha256", expected_protocol_sha), ("input_sha256", spec["input_sha256"]),
                              ("thresholds_refitted_on_evaluation", False)):
            if report.get("provenance", {}).get(key) != expected:
                errors.append("evaluation provenance mismatch: " + key)
        checks, aggregate_errors = aggregate_checks(spec, report, calibration, protocol["acceptance"]["bootstrap"], name)
        errors.extend(aggregate_errors)
        receipts = report.get("provenance", {}).get("raw_exports", {})
        if set(receipts) != set(calibration["provenance"]["component_weights_sha256"]):
            errors.append("raw export component set differs from frozen calibration")
        for component, evidence in receipts.items():
            raw_path = Path(evidence["path"])
            side_path = raw_path.with_suffix(".provenance.json")
            if _sha(side_path) != evidence["sidecar_sha256"]:
                errors.append("raw sidecar changed: " + component)
                continue
            side = _json(side_path)
            expected = {"purpose": "final", "input_sha256": spec["input_sha256"], "original_rows": spec["rows"],
                "eligible_export_rows": spec["rows"], "excluded_ineligible_rows": 0, "calibration_fitted": False,
                "frozen_blend_sha256": blend_sha, "frozen_calibration_sha256": cal_sha, "output_sha256": evidence["sha256"]}
            if any(side.get(k) != v for k, v in expected.items()):
                errors.append("raw phase/input/artifact/denominator mismatch: " + component)
            if _get(side, "deployment_scope", "rule_sha256") != protocol["language_rule_sha256"]:
                errors.append("raw language rule differs: " + component)
            for raw_key, cal_key in (("weights_sha256", "component_weights_sha256"), ("component_metadata_sha256", "component_metadata_sha256")):
                if side.get(raw_key) != calibration["provenance"][cal_key].get(component):
                    errors.append("raw checkpoint identity mismatch: " + component)
        view_results = {}
        view_specs, view_reports = spec.get("required_views", {}), report.get("registered_views", {})
        require(isinstance(view_specs, dict), "required_views must be a mapping when declared")
        if not isinstance(view_reports, dict) or set(view_reports) != set(view_specs):
            errors.append("registered view set differs from frozen parent policy")
            view_reports = view_reports if isinstance(view_reports, dict) else {}
        for view_name, view_spec in view_specs.items():
            view_report = view_reports.get(view_name, {})
            view_checks, view_errors = aggregate_checks(view_spec, view_report, calibration,
                protocol["acceptance"]["bootstrap"], name + ":" + view_name)
            expected_metadata = {"column": view_spec["column"], "parent_input_sha256": spec["input_sha256"],
                "rows": view_spec["rows"], "groups": view_spec["groups"], "core_counts": view_spec["core_counts"],
                "shared_human_controls_not_independent_or_additive": True, "probabilities_or_thresholds_refitted": False}
            metadata = view_report.get("registered_view", {})
            if any(metadata.get(k) != v for k, v in expected_metadata.items()):
                view_errors.append("registered boolean membership/parent/count/nonindependence declaration differs")
            view_results[view_name] = {"pass": not view_errors and all(c["pass"] for c in view_checks),
                "checks": view_checks, "provenance_errors": view_errors, "metadata": metadata,
                "group_bootstrap": view_report.get("group_bootstrap")}
        result[name] = {"pass": not errors and all(c["pass"] for c in checks) and all(v["pass"] for v in view_results.values()),
            "checks": checks, "provenance_errors": errors, "registered_views": view_results,
            "core_counts": spec["core_counts"], "all_rows": spec["rows"], "group_bootstrap": report.get("group_bootstrap"),
            "report": {"path": str(Path(report_paths[name]).resolve()), "sha256": _sha(report_paths[name])}}
    passed = all(item["pass"] for item in result.values())
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preregistered_practical_targets_pass" if passed else "preregistered_practical_targets_not_met",
        "all_necessary_suites_pass": passed, "suites": result,
        "source_assertions_enabled": False, "source_ranking_accuracy_is_not_covered_by_binary_pass": True,
        "protocol_sha256": expected_protocol_sha, "blend_sha256": blend_sha, "calibration_sha256": cal_sha,
        "acceptance_code_sha256": _sha(__file__), "parameters_or_thresholds_fitted_here": False,
        "scope": "Registered fresh final suites and point criteria only; confidence intervals and limitations must accompany any release. Old observed finals are never substituted or pooled."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--reports", nargs="+", required=True, help="suite=report.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pairs = [x.split("=", 1) for x in args.reports]
    require(all(len(x) == 2 for x in pairs) and len({x[0] for x in pairs}) == len(pairs), "unique named reports required")
    result = summarize(args.protocol, args.protocol_sha256, args.blend, args.calibration, dict(pairs))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "all_necessary_suites_pass": result["all_necessary_suites_pass"]}))
