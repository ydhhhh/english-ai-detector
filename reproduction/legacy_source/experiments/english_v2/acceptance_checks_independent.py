"""Independent aggregate/receipt fixtures; no real final, scores or model I/O.

All JSONs and inert raw-file placeholders are synthetic TemporaryDirectory
fixtures. This does not establish accuracy, simulate a model, or fit anything.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

from experiments.english_v2 import acceptance as a


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fixture(root, *, boot_override=None):
    root.mkdir()
    blend = root/"blend.json"; write(blend, {"synthetic_only": True})
    targets = {"core_AUROC_min": .90, "core_balanced_accuracy_at_probability_half_min": .90,
        "fixed_AI_human_FPR_max": .05, "fixed_AI_recall_min": .80,
        "online_human_as_AI_rate_max": .05, "online_AI_as_human_rate_max": .05,
        "online_correct_AI_recall_min": .80, "online_correct_human_recall_min": .80,
        "online_decided_coverage_min": .80}
    boot = {"n_boot": 500, "seed": 42,
        "method": "percentile whole-group bootstrap; same resampling weights across tasks; fixed thresholds", "minimum_valid_fraction": .95}
    boot.update(boot_override or {})
    suites = {name: {"required_for_practical_acceptance": True, "input_sha256": digest("synthetic_input:"+name),
        "rows": 200, "groups": 100, "core_counts": {"n": 200, "human_n": 100, "AI_n": 100, "groups": 100},
        "targets": copy.deepcopy(targets)} for name in a.REQUIRED_SUITES}
    protocol = root/"protocol.json"
    write(protocol, {"schema_version": 1, "status": "FROZEN_BEFORE_V2_TRAINING_AND_SCORES", "language_rule_sha256": digest("synthetic_language_rule"),
        "acceptance": {"criterion": "point_estimates_all_checks_with_group_CI_disclosed", "bootstrap": boot}, "necessary_final_suites": suites})
    models = {name: digest("synthetic_weights:"+name) for name in ("new_binary", "minilm_v3_epoch2", "statistical_v2")}
    configs = {name: digest("synthetic_metadata:"+name) for name in models}
    threshold = {"task": "AI_detection", "fit_split": "calibration", "threshold": .8, "comparison": ">="}
    calibration = root/"calibration.json"
    write(calibration, {"provenance": {"blend_sha256": a._sha(blend), "v2_protocol_sha256": a._sha(protocol),
        "only_binary_fitted_on_registered_calibration": True, "component_weights_sha256": models, "component_metadata_sha256": configs},
        "thresholds": {"AI": threshold, "human": {**threshold, "task": "human_detection"}},
        "source_policy": {"mode": "rankings_only", "family_enabled": False, "model_enabled": False}})
    binary = {"n": 200, "human_n": 100, "AI_n": 100, "tp": 95, "fn": 5, "fp": 1, "tn": 99,
              "fpr": .01, "tpr": .95, "balanced_accuracy": .97, "auroc": .99}
    online = {"status": "evaluated", "n": 200, "human_n": 100, "AI_n": 100,
        "predicted_AI_n": 93, "predicted_human_n": 97, "inconclusive_n": 10,
        "human_misclassified_as_AI_n": 1, "AI_misclassified_as_human_n": 2, "correct_AI_n": 92, "correct_human_n": 95,
        "inconclusive_human_n": 4, "inconclusive_AI_n": 6, "language_supported_n": 200, "language_unsupported_n": 0,
        "human_misclassified_as_AI_rate": .01, "AI_misclassified_as_human_rate": .02,
        "correct_AI_recall": .92, "correct_human_recall": .95, "decided_coverage": .95}
    source_off = {"assertion_audit": {key: 0 for key in ("asserted_on_humans_n", "asserted_on_trusted_AI_n", "asserted_on_AI_without_trusted_source_label_n")}}
    point = {"binary": binary, "deployed_decision_audit": online, "family": source_off, "model": source_off}
    paths = {}
    for name, suite in suites.items():
        intervals = {}
        for _, (section, key, _) in a.METRICS.items():
            if section == "half":
                continue
            head = "binary" if section == "binary" else "deployed_decision_audit"
            value = point[head][key]
            intervals[f"core_ge_128_tokens.{head}.{key}"] = {"point": value, "low": max(0., value-.02),
                "high": min(1., value+.02), "valid_replicates": boot["n_boot"]}
        receipts = {}
        for component in models:
            raw = root/(name+"_"+component+".npz")
            raw.write_bytes(b"SYNTHETIC_INERT_PLACEHOLDER_NOT_A_REAL_PREDICTION_FILE")
            side = raw.with_suffix(".provenance.json")
            write(side, {"purpose": "final", "input_sha256": suite["input_sha256"], "original_rows": 200,
                "eligible_export_rows": 200, "excluded_ineligible_rows": 0, "calibration_fitted": False,
                "frozen_blend_sha256": a._sha(blend), "frozen_calibration_sha256": a._sha(calibration),
                "output_sha256": a._sha(raw), "weights_sha256": models[component], "component_metadata_sha256": configs[component],
                "deployment_scope": {"rule_sha256": digest("synthetic_language_rule")}})
            receipts[component] = {"path": str(raw), "sha256": a._sha(raw), "sidecar_sha256": a._sha(side)}
        report = {"all_lengths": copy.deepcopy(point), "core_ge_128_tokens": copy.deepcopy(point), "threshold": threshold,
            "length_unit": "word_like_tokens", "source_policy": {"source_assertions_enabled": False},
            "binary_at_probability_half": {"threshold": .5, "threshold_source": "constant default probability cutoff, never fitted",
                "all_lengths": binary, "core_ge_128_word_like_tokens": binary},
            "group_bootstrap": {"n_groups": 100, "core_groups": 100, "n_boot": boot["n_boot"], "seed": boot["seed"],
                "method": boot["method"], "intervals95": intervals},
            "provenance": {"blend_sha256": a._sha(blend), "calibration_sha256": a._sha(calibration), "v2_protocol_sha256": a._sha(protocol),
                "input_sha256": suite["input_sha256"], "thresholds_refitted_on_evaluation": False, "raw_exports": receipts}}
        paths[name] = root/(name+".json"); write(paths[name], report)
    return protocol, blend, calibration, paths


def run():
    source_sha = a._sha(a.__file__)
    results = []
    with tempfile.TemporaryDirectory(prefix="synthetic-v2-acceptance-") as temp:
        root = Path(temp)
        p, b, c, paths = fixture(root/"case")
        def evaluate(chosen=None):
            try:
                return a.summarize(p, a._sha(p), b, c, paths if chosen is None else chosen)["all_necessary_suites_pass"]
            except (ValueError, KeyError, OSError):
                return False
        def test(name, expected, observed):
            results.append({"name": name, "expected_pass": expected, "actual_pass": observed, "passed": expected == observed})
        test("consistent complete four-suite receipt fixture", True, evaluate())
        test("missing necessary suite", False, evaluate({k:v for k,v in paths.items() if k != a.REQUIRED_SUITES[-1]}))
        chosen = a.REQUIRED_SUITES[2]
        original = json.loads(paths[chosen].read_text())
        def mutate(name, change):
            value = copy.deepcopy(original); change(value); write(paths[chosen], value)
            test(name, False, evaluate()); write(paths[chosen], original)
        def weak_modern(v):
            for key in ("core_ge_128_tokens", "all_lengths"):
                v[key]["binary"]["auroc"] = .85
            interval = v["group_bootstrap"]["intervals95"]["core_ge_128_tokens.binary.auroc"]
            interval.update(point=.85, low=.80, high=.90)
        assert (.99*3+.85)/4 > .90
        mutate("failed modern suite despite high pooled AUROC", weak_modern)
        mutate("tampered frozen denominator", lambda v:v["core_ge_128_tokens"]["binary"].update(n=199))
        mutate("drop abstentions from reported AI recall denominator", lambda v:v["core_ge_128_tokens"]["deployed_decision_audit"].update(correct_AI_recall=92/94))
        mutate("conditional decided coverage substituted for all-row coverage", lambda v:v["core_ge_128_tokens"]["deployed_decision_audit"].update(decided_coverage=1.0))
        mutate("wrong calibration fingerprint", lambda v:v["provenance"].update(calibration_sha256="0"*64))
        mutate("wrong final input fingerprint", lambda v:v["provenance"].update(input_sha256="0"*64))
        mutate("missing required CI", lambda v:v["group_bootstrap"]["intervals95"].pop("core_ge_128_tokens.binary.auroc"))
        mutate("impossible valid replicate count above n_boot", lambda v:v["group_bootstrap"]["intervals95"]["core_ge_128_tokens.binary.auroc"].update(valid_replicates=501))
        receipt = original["provenance"]["raw_exports"]["new_binary"]
        side = Path(receipt["path"]).with_suffix(".provenance.json")
        old_side = json.loads(side.read_text()); changed_side = copy.deepcopy(old_side)
        changed_side["frozen_calibration_sha256"] = "0"*64
        write(side, changed_side)
        changed_report = copy.deepcopy(original)
        changed_report["provenance"]["raw_exports"]["new_binary"]["sidecar_sha256"] = a._sha(side)
        write(paths[chosen], changed_report)
        test("wrong cal in internally rehashed raw receipt", False, evaluate())
        write(side, old_side); write(paths[chosen], original)
        p0, b0, c0, paths0 = fixture(root/"bad_boot", boot_override={"n_boot": 0, "minimum_valid_fraction": 0})
        try:
            observed = a.summarize(p0, a._sha(p0), b0, c0, paths0)["all_necessary_suites_pass"]
        except (ValueError, KeyError, OSError):
            observed = False
        test("zero-replicate bootstrap protocol rejected", False, observed)
    stable = a._sha(a.__file__) == source_sha
    return {"passed": stable and all(r["passed"] for r in results), "source_stable_during_check": stable,
        "aggregate_receipt_fixtures_only": True, "real_calibration_final_or_scores_read": False, "no_model_training_or_GPU": True,
        "acceptance_sha256": source_sha, "checks_sha256": a._sha(__file__), "checks": results,
        "hash_scope": "Production acceptance cross-checks report/sidecar declarations, matching v1 JSON-only scope; inert raw placeholders are synthetic fixtures."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(json.dumps(value, indent=2))
    raise SystemExit(0 if value["passed"] else 1)
