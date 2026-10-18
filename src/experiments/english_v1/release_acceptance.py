"""Summarize frozen final-report JSONs; never load predictions or refit anything.

Only JSON reports, export sidecars and the frozen data/protocol manifests are
opened. Input NPZ hashes are cross-checked as declared in the evaluator report
and raw-export sidecar; NPZ arrays and corpus files are never opened here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from experiments.english_v1.catalog import FAMILIES, NOMINAL_MODELS

PROJECT = Path(__file__).resolve().parents[2]
DECISION_PROTOCOL_SHA256 = "0e335238cb5b66c77ef633cc9c09494357e23c4cb986651f10887df2bfdaa0fb"
FROZEN_BINARY_TARGETS = {"core_test_AUROC_min": .97, "core_test_balanced_accuracy_at_0_5_min": .90,
                       "operating_threshold_target_calibration_human_FPR": .01,
                       "core_test_human_FPR_max": .02, "core_test_AI_recall_min_at_fixed_threshold": .80,
                       "final_source_domain_AUROC_min": .90, "final_source_domain_human_FPR_max_at_fixed_threshold": .05}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _json(path):
    path = Path(path)
    if path.suffix.casefold() != ".json":
        raise ValueError("acceptance reads JSON only")
    value = json.loads(path.read_text(), object_pairs_hook=_object,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + x)))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _get(value, *keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _prob(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def _path(value, root):
    path = Path(value)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def _cohort(point):
    return {key: _get(point, "binary", key) for key in ("n", "human_n", "AI_n")}


def _check(name, actual, target, op, counts, *, required=True, note=None):
    valid = _prob(actual)
    passed = (actual >= target if op == ">=" else actual <= target) if valid and target is not None else None
    return {"name": name, "actual": actual if valid else None, "target": target,
            "comparison": op, "pass": passed,
            "status": "missing" if actual is None else "invalid" if not valid else "diagnostic" if target is None else "pass" if passed else "fail",
            "required_for_binary_target_pass": required, "cohort": counts,
            **({"note": note} if note else {})}


def _binary_consistency(point, errors, prefix):
    b = _get(point, "binary")
    if not isinstance(b, dict):
        errors.append(prefix + ":missing binary counts")
        return
    keys = ("n", "human_n", "AI_n", "tp", "fp", "tn", "fn")
    if not all(_count(b.get(k)) for k in keys) or not b["human_n"] or not b["AI_n"]:
        errors.append(prefix + ":invalid or single-class binary counts")
        return
    if not (b["n"] == b["human_n"] + b["AI_n"] and b["tp"] + b["fn"] == b["AI_n"] and b["tn"] + b["fp"] == b["human_n"]):
        errors.append(prefix + ":binary count denominators disagree")
        return
    for key, expected in (("fpr", b["fp"] / b["human_n"]), ("tpr", b["tp"] / b["AI_n"]),
                          ("balanced_accuracy", (b["tp"] / b["AI_n"] + b["tn"] / b["human_n"]) / 2)):
        if not _prob(b.get(key)) or not math.isclose(b[key], expected, abs_tol=1e-10):
            errors.append(prefix + ":" + key + " disagrees with raw counts")


def _online_consistency(point, errors, prefix):
    audit = _get(point, "deployed_decision_audit")
    b = _get(point, "binary") or {}
    if not isinstance(b, dict) or not isinstance(audit, dict) or audit.get("status") != "evaluated":
        errors.append(prefix + ":missing online audit")
        return
    required = ("n", "human_n", "AI_n", "predicted_AI_n", "predicted_human_n", "inconclusive_n",
                "human_misclassified_as_AI_n", "AI_misclassified_as_human_n", "correct_AI_n", "correct_human_n",
                "inconclusive_human_n", "inconclusive_AI_n", "language_supported_n", "language_unsupported_n")
    if not all(_count(audit.get(k)) for k in required):
        errors.append(prefix + ":invalid online raw counts")
        return
    if any(audit[k] != b.get(k) for k in ("n", "human_n", "AI_n")):
        errors.append(prefix + ":online denominator dropped rejected examples")
    if audit["correct_AI_n"] > b.get("tp", -1) or audit["human_misclassified_as_AI_n"] > b.get("fp", -1):
        errors.append(prefix + ":online AI assertions exceed the fixed AI region")
    if not (audit["predicted_AI_n"] + audit["predicted_human_n"] + audit["inconclusive_n"] == audit["n"]
            and audit["correct_AI_n"] + audit["human_misclassified_as_AI_n"] == audit["predicted_AI_n"]
            and audit["correct_human_n"] + audit["AI_misclassified_as_human_n"] == audit["predicted_human_n"]
            and audit["inconclusive_human_n"] + audit["inconclusive_AI_n"] == audit["inconclusive_n"]
            and audit["correct_AI_n"] + audit["AI_misclassified_as_human_n"] + audit["inconclusive_AI_n"] == audit["AI_n"]
            and audit["correct_human_n"] + audit["human_misclassified_as_AI_n"] + audit["inconclusive_human_n"] == audit["human_n"]
            and audit["language_supported_n"] + audit["language_unsupported_n"] == audit["n"]):
        errors.append(prefix + ":online decision partition disagrees")
    for key, numerator, denominator in (("human_misclassified_as_AI_rate", "human_misclassified_as_AI_n", "human_n"),
                                         ("correct_AI_recall", "correct_AI_n", "AI_n")):
        if not audit[denominator] or not _prob(audit.get(key)) or not math.isclose(audit[key], audit[numerator] / audit[denominator], abs_tol=1e-10):
            errors.append(prefix + ":" + key + " does not use all true-class examples")


def _provenance(report, role, manifest, root):
    errors, sides = [], {}
    p = report.get("provenance") or {}
    if not isinstance(p, dict):
        raise ValueError("provenance JSON object required")
    if report.get("length_unit") not in {"word_like_tokens", "word_like_tokens_regex"}:
        errors.append("core length must use original-text word-like units")
    for key in ("blend_sha256", "calibration_sha256"):
        if not _hash(p.get(key)):
            errors.append("missing/invalid " + key)
    if p.get("thresholds_refitted_on_evaluation") is not False:
        errors.append("evaluation must explicitly declare thresholds were not refitted")
    if p.get("deployment_decision_protocol_sha256") != DECISION_PROTOCOL_SHA256:
        errors.append("report does not bind the frozen amendment07 decision policy")
    if not _hash(p.get("language_rule_sha256")):
        errors.append("report must bind its deployment language rule")
    expected = _get(manifest, "output_files", "test" if role == "main" else "challenge_source_domain")
    if manifest.get("frozen") is not True or not isinstance(expected, dict) or not _hash(expected.get("sha256")) or not _count(expected.get("rows")):
        errors.append("frozen main/Yelp input identity missing from data manifest")
    if not isinstance(expected, dict):
        expected = {}
    inputs = p.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        errors.append("component input provenance missing")
        inputs = {}
    for name, info in inputs.items():
        try:
            if not isinstance(info, dict) or not _hash(info.get("sha256")):
                raise ValueError("invalid report component input hash")
            path = _path(info["path"], root)
            sidepath = path.with_suffix(".provenance.json")
            side = _json(sidepath)
            sides[name] = {"path": str(sidepath), "sha256": _sha(sidepath), "content": side}
            if side.get("purpose") != "final" or side.get("calibration_fitted") is not False:
                raise ValueError("component is not a final raw export")
            if side.get("output_sha256") != info["sha256"] or _path(side["output"], root) != path:
                raise ValueError("sidecar output identity differs from evaluated input")
            if any(side.get("frozen_" + key) != p.get(key) for key in ("blend_sha256", "calibration_sha256")):
                raise ValueError("sidecar uses another blend/calibration freeze")
            if not _hash(side.get("weights_sha256")) or not _hash(side.get("component_metadata_sha256")) or not _hash(_get(side, "deployment_scope", "rule_sha256")):
                raise ValueError("component/config/language fingerprint missing")
            if _get(side, "deployment_scope", "rule_sha256") != p.get("language_rule_sha256"):
                raise ValueError("sidecar language rule differs from evaluated decision audit")
            if not expected or side.get("input_sha256") != expected.get("sha256") or _path(side["input"], root) != _path(expected["path"], root):
                raise ValueError("input is not the exact frozen " + role + " corpus")
            if side.get("original_rows") != expected.get("rows") or side.get("eligible_export_rows") != expected.get("rows") or side.get("excluded_ineligible_rows") != 0:
                raise ValueError("main/Yelp export must cover its complete frozen input")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(str(name) + ":" + str(exc))
    if expected and _get(report, "all_lengths", "binary", "n") != expected.get("rows"):
        errors.append("report all-length count differs from complete frozen corpus")
    threshold = report.get("threshold") or {}
    if not isinstance(threshold, dict):
        raise ValueError("fixed threshold JSON object required")
    if threshold.get("fit_split") != "calibration" or threshold.get("task") != "AI_detection" or threshold.get("comparison") != ">=":
        errors.append("fixed calibration AI threshold identity missing")
    if threshold.get("target_fpr") != .01 or not _hash(threshold.get("calibration_prediction_sha256")):
        errors.append("original empirical calibration1%FPR target/evidence missing")
    if not _prob(_get(threshold, "calibration_metrics", "fpr")) or _get(threshold, "calibration_metrics", "fpr") > .01:
        errors.append("calibration empirical1%humanFPR condition not evidenced")
    cutoff = threshold.get("threshold")
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff) or cutoff <= .5:
        errors.append("AI cutoff must be finite and require strict majority")
    return {"passed": not errors, "errors": errors, "blend_sha256": p.get("blend_sha256"),
            "calibration_sha256": p.get("calibration_sha256"), "input_sidecars": sides,
            "hash_verification_scope": "JSON declarations cross-checked; prediction/corpus bytes never opened by acceptance"}


def _aims(report):
    core = _get(report, "supported_catalog_attribution", "core_ge_128_tokens") or {}
    general = report.get("core_ge_128_tokens") or {}
    if not isinstance(core, dict):core = {}
    if not isinstance(general, dict):general = {}
    out = {}
    for head, accuracy_target, coverage_target, known in (("family", .85, .50, set(FAMILIES[:-1])), ("model", .75, .20, set(NOMINAL_MODELS))):
        value = core.get(head) or {}
        if not isinstance(value, dict):value = {}
        supported = value.get("supported_AI_n")
        accepted = _get(value, "accepted_conditional", "n")
        audit = value.get("assertion_audit") or {}
        support = _get(value, "conditional", "class_support")
        errors = []
        if not _count(supported) or not supported or not _count(accepted) or accepted > supported:
            errors.append("no valid core trusted named-catalog cohort")
        if not isinstance(support, dict) or any(not _count(n) for n in support.values()) or sum(support.values()) != supported or any(n for label, n in support.items() if label not in known):
            errors.append("Other/unknown/human examples cannot inflate named-catalog denominator")
        accepted_support = _get(value, "accepted_conditional", "class_support")
        if accepted and (not isinstance(accepted_support, dict) or any(not _count(n) for n in accepted_support.values()) or sum(accepted_support.values()) != accepted or any(n for label, n in accepted_support.items() if label not in known)):
            errors.append("accepted selective cohort includes Other/unknown/human or inconsistent counts")
        coverage = value.get("deployed_attribution_coverage_on_supported_AI")
        accuracy = _get(value, "accepted_conditional", "accuracy")
        if _count(supported) and supported and _count(accepted):
            if not _prob(coverage) or not math.isclose(coverage, accepted / supported, abs_tol=1e-10):
                errors.append("selective coverage denominator inconsistent")
            if audit.get("asserted_on_trusted_AI_n") != accepted:
                errors.append("accepted cohort differs from actual deployed trusted-AI assertions")
            correct = audit.get("correct_asserted_on_trusted_AI_n")
            if accepted and (not _count(correct) or correct > accepted or not _prob(accuracy) or not math.isclose(accuracy, correct / accepted, abs_tol=1e-10)):
                errors.append("selective accuracy denominator inconsistent")
        checks = [_check("selective_accuracy", accuracy, accuracy_target, ">=", {"trusted_AI_n": supported, "accepted_trusted_AI_n": accepted}, required=False),
                  _check("deployed_coverage_on_supported_AI", coverage, coverage_target, ">=", {"trusted_AI_n": supported, "accepted_trusted_AI_n": accepted}, required=False)]
        met = not errors and all(c["pass"] is True for c in checks)
        out[head] = {"aim_met": met, "status": "aim_met" if met else "missing" if not supported else "aim_not_met",
                     "checks": checks, "cohort_errors": errors,
                     "catalog_assertion_audit": audit, "all_trusted_label_assertion_audit": _get(general, head, "assertion_audit"),
                     "interpretation": "Core trusted AI in named6families/nominal18 only; true Other/unknown leaves do not boost accuracy. Human false assertions remain in separate audit; lower confidence can be presented as candidate rankings."}
    return out


def summarize(main_report, yelp_report, *, extra_reports=None, protocol_path=None, data_manifest=None, project_root=PROJECT):
    """Return an acceptance artifact from JSON paths; never mutate inputs."""
    protocol_path = Path(protocol_path or PROJECT / "experiments/english_v1/acceptance_protocol.json")
    data_manifest = Path(data_manifest or PROJECT / "data/english_v1/manifest.json")
    protocol, manifest = _json(protocol_path), _json(data_manifest)
    targets = protocol["binary_acceptance_targets"]
    if any(targets.get(key) != value for key, value in FROZEN_BINARY_TARGETS.items()):
        raise ValueError("required binary targets differ from frozen acceptance protocol")
    datasets, checks, integrity_errors = {}, [], []
    for role, path in (("main", main_report), ("yelp", yelp_report)):
        if path is None or not Path(path).is_file():
            datasets[role] = {"status": "missing", "report_path": str(path) if path else None}
            checks.append({"name": role + ".final_report", "actual": None, "target": "required frozen final report", "pass": None, "status": "missing", "required_for_binary_target_pass": True})
            continue
        try:
            report = _json(path)
            evidence = _provenance(report, role, manifest, project_root)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            datasets[role] = {"status": "invalid", "report_path": str(path), "errors": [str(exc)]}
            integrity_errors.append(role + ":" + str(exc))
            continue
        core = report.get("core_ge_128_tokens") or {}
        errors = list(evidence["errors"])
        _binary_consistency(core, errors, role + ".core")
        _online_consistency(core, errors, role + ".core")
        counts = _cohort(core)
        metric_plan = [("AUROC", _get(core, "binary", "auroc"), targets["core_test_AUROC_min"] if role == "main" else targets["final_source_domain_AUROC_min"], ">="),
                       ("fixed_AI_human_FPR", _get(core, "binary", "fpr"), targets["core_test_human_FPR_max"] if role == "main" else targets["final_source_domain_human_FPR_max_at_fixed_threshold"], "<="),
                       ("online_human_as_AI_rate", _get(core, "deployed_decision_audit", "human_misclassified_as_AI_rate"), targets["core_test_human_FPR_max"] if role == "main" else targets["final_source_domain_human_FPR_max_at_fixed_threshold"], "<=")]
        if role == "main":
            half = _get(report, "binary_at_probability_half", "core_ge_128_word_like_tokens")
            _binary_consistency({"binary": half}, errors, "main.half")
            if _cohort({"binary": half}) != counts or _get(report, "binary_at_probability_half", "threshold") != .5:
                errors.append("main0.5metric uses another cohort or threshold")
            metric_plan.extend([( "balanced_accuracy_at_0.5", _get(half, "balanced_accuracy"), targets["core_test_balanced_accuracy_at_0_5_min"], ">="),
                                ("fixed_AI_recall", _get(core, "binary", "tpr"), targets["core_test_AI_recall_min_at_fixed_threshold"], ">="),
                                ("online_correct_AI_recall", _get(core, "deployed_decision_audit", "correct_AI_recall"), targets["core_test_AI_recall_min_at_fixed_threshold"], ">=")])
        else:
            metric_plan.append(("online_correct_AI_recall", _get(core, "deployed_decision_audit", "correct_AI_recall"), None, ">="))
        local = [_check(role + "." + name, actual, target, op, counts, required=target is not None) for name, actual, target, op in metric_plan]
        checks.extend(local)
        integrity_errors.extend(role + ":" + error for error in errors)
        evidence["passed"] = not errors
        evidence["errors"] = errors
        datasets[role] = {"status": "verified" if not errors else "invalid", "report_path": str(Path(path).resolve()), "report_sha256": _sha(path),
                          "core_counts": counts, "provenance": evidence, "binary_checks": local,
                          "online_core_decision_audit": core.get("deployed_decision_audit"), "attribution_aims": _aims(report),
                          "group_bootstrap": report.get("group_bootstrap"), "frozen_AI_threshold": report.get("threshold")}
    if all(datasets.get(k, {}).get("provenance") for k in ("main", "yelp")):
        a, b = datasets["main"], datasets["yelp"]
        if any(a["provenance"][k] != b["provenance"][k] for k in ("blend_sha256", "calibration_sha256")) or a["frozen_AI_threshold"] != b["frozen_AI_threshold"]:
            integrity_errors.append("main/Yelp report blend, calibration or fixed AI threshold differs")
        ca, cb = a["provenance"]["input_sidecars"], b["provenance"]["input_sidecars"]
        if set(ca) != set(cb):
            integrity_errors.append("main/Yelp component sets differ")
        for name in set(ca) & set(cb):
            sa, sb = ca[name]["content"], cb[name]["content"]
            if any(sa.get(k) != sb.get(k) for k in ("weights_sha256", "component_metadata_sha256")) or _get(sa, "deployment_scope", "rule_sha256") != _get(sb, "deployment_scope", "rule_sha256"):
                integrity_errors.append("main/Yelp component/config/language fingerprint differs:" + name)
    required = [item for item in checks if item["required_for_binary_target_pass"]]
    if integrity_errors or any(item["pass"] is False or item["status"] == "invalid" for item in required):
        status = "failed"
    elif any(item["pass"] is not True for item in required):
        status = "limited"
    else:
        status = "target_pass"
    extra = {}
    for name, path in (extra_reports or {}).items():
        try:
            report = _json(path)
            extra[name] = {"diagnostic_only": True, "report_path": str(Path(path).resolve()), "report_sha256": _sha(path),
                           "core": report.get("core_ge_128_tokens"), "provenance": report.get("provenance"), "can_replace_required_Yelp": False,
                           "provenance_verified_for_acceptance": False, "note": "Preserved diagnostic JSON; never substitutes for exact main/Yelp acceptance evidence."}
        except (OSError, ValueError) as exc:
            extra[name] = {"diagnostic_only": True, "status": "missing_or_invalid", "error": str(exc), "can_replace_required_Yelp": False}
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
            "binary_targets_passed": status == "target_pass", "checks": checks, "integrity_errors": integrity_errors, "datasets": datasets,
            "extra_diagnostics": extra, "protocol": {"path": str(protocol_path.resolve()), "sha256": _sha(protocol_path)},
            "data_manifest": {"path": str(data_manifest.resolve()), "sha256": _sha(data_manifest)},
            "interpretation": "Target pass requires all original necessary binary targets and the additional main online recall/FPR and Yelp online FPR gates. Yelp online recall has no preregistered minimum and is disclosed with coverage. Binary success does not certify generic attribution, population/domain FPR or high Yelp recall.",
            "attribution_policy": "Family/model thresholds are aims; report named-catalog core accuracy plus coverage and human assertion errors separately. Failed/missing aims may be described as candidate rankings. This summary does not change deployed assertions or select a new configuration.",
            "mutations_or_refits": False, "prediction_arrays_opened": False,
            "failure_rule": "Record missing/failed targets honestly. Do not tune any model, blend or threshold to this final report; future changes require new development and independent final evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path)
    parser.add_argument("--yelp", type=Path)
    parser.add_argument("--extra", nargs="*", default=[], help="diagnostic name=report.json only")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protected = [args.main,args.yelp,args.protocol or PROJECT/"experiments/english_v1/acceptance_protocol.json",args.data_manifest or PROJECT/"data/english_v1/manifest.json"]
    protected += [item.split("=",1)[1] for item in args.extra if "=" in item]
    if args.output.suffix.casefold()!=".json" or any(path is not None and args.output.resolve()==Path(path).resolve() for path in protected):
        parser.error("write a new JSON summary path; never replace an input report/protocol/manifest")
    extra = dict(item.split("=", 1) for item in args.extra)
    result = summarize(args.main, args.yelp, extra_reports=extra, protocol_path=args.protocol, data_manifest=args.data_manifest, project_root=args.project_root)
    sidecars=[side["path"] for value in result["datasets"].values() for side in value.get("provenance",{}).get("input_sidecars",{}).values()]
    if any(args.output.resolve()==Path(path).resolve() for path in sidecars):
        parser.error("never replace an input provenance sidecar")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "binary_targets_passed": result["binary_targets_passed"]}))


if __name__ == "__main__":
    main()
