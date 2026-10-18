"""Independent synthetic-only input-boundary and selection oracles for v2.

No real dataset, checkpoint, calibration, final score, or final report is read.
All NPZ payloads and the mock protocol are created in a temporary directory.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from experiments.english_v2 import select_binary as selector
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
NAMES = ["constructed_spiky", "constructed_balanced", "constructed_noise", "constructed_weak"]


def fitting(groups):
    salt = selector.SELECTOR["fit_selection_group_salt"]
    return np.array([int(hashlib.sha256((salt + str(g)).encode()).hexdigest()[:8], 16) % 2 == 0 for g in groups])


def factory():
    rng = np.random.default_rng(20261017)
    payloads = {}
    components = {name: {"weights_sha256": hashlib.sha256((name + ":weights").encode()).hexdigest(),
                         "metadata_sha256": hashlib.sha256((name + ":metadata").encode()).hexdigest()} for name in NAMES}
    for ci, cohort in enumerate(selector.COHORTS):
        group_n = 96 + 16 * ci
        n = 4 * (group_n + 16)
        y = np.tile([0, 1, 0, 1], group_n + 16).astype(np.int8)
        signed = 2 * y.astype(float) - 1
        groups = np.repeat([f"constructed-{cohort}-parent-{j}" for j in range(group_n + 16)], 4)
        lengths = np.r_[np.full(4 * group_n, 200), np.full(64, 30)].astype(np.int32)
        # Spiky excels on all but rewrite, whereas balanced has useful quality everywhere.
        spiky = (3 if cohort != "paraphrase" else -3) * signed + rng.normal(0, .5, n)
        balanced = 1.2 * signed + rng.normal(0, 1.0, n)
        noise = rng.normal(0, 1.0, n)
        weak = .25 * signed + rng.normal(0, 1.4, n)
        base = {"ids": np.array([f"constructed-{cohort}-document-{i}" for i in range(n)]),
                "group_ids": groups, "y_ai": y, "lengths": lengths,
                "export_purpose": np.array("development"), "calibrated": np.array(False)}
        payloads[cohort] = {}
        for name, margin in zip(NAMES, (spiky, balanced, noise, weak)):
            payloads[cohort][name] = {**{k: v.copy() for k, v in base.items()},
                "raw_binary": np.column_stack([-margin / 2, margin / 2]),
                "component_weights_sha256": np.array(components[name]["weights_sha256"]),
                "component_metadata_sha256": np.array(components[name]["metadata_sha256"])}
    return payloads, components


def run_payloads(payloads, components, folder, *, mutate_plan=None, capture_objective=False):
    folder.mkdir()
    protocol = {"status": "FROZEN_BEFORE_V2_TRAINING_AND_SCORES", "candidate_names": NAMES,
                "binary_selector": copy.deepcopy(selector.SELECTOR), "constructed_fixture_only": True}
    pp = folder / "constructed_protocol.json"
    pp.write_text(json.dumps(protocol))
    plan = {"protocol": {"path": str(pp), "sha256": selector.sha(pp)}, "names": NAMES,
            "components": components, "calibration_opened": False, "final_test_opened": False,
            "cohorts": {}}
    for cohort, candidates in payloads.items():
        plan["cohorts"][cohort] = {}
        for name, payload in candidates.items():
            path = folder / f"constructed_{cohort}_{name}.npz"
            np.savez_compressed(path, **payload)
            plan["cohorts"][cohort][name] = {"path": str(path), "sha256": selector.sha(path)}
    if mutate_plan:
        mutate_plan(plan)
    plan_path = folder / "constructed_plan.json"
    plan_path.write_text(json.dumps(plan))
    out = folder / "constructed_selection.json"
    if not capture_objective:
        return selector.fit_selection(plan_path, out), plan_path, out
    original = selector.minimize
    captured = {}

    def observe(fun, theta, **kwargs):
        captured["loss_zero"], captured["gradient_zero"] = fun(np.zeros_like(theta))
        return original(fun, theta, **kwargs)

    with patch.object(selector, "minimize", observe):
        artifact = selector.fit_selection(plan_path, out)
    return artifact, plan_path, out, captured


def main():
    code_sha = selector.sha(selector.__file__)
    checks = []

    def ok(name, condition=True):
        if not condition:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    def reject(name, payloads, components, root, *, mutate_plan=None, message=None):
        try:
            run_payloads(payloads, components, root / name, mutate_plan=mutate_plan)
        except (ValueError, KeyError, FileExistsError) as exc:
            if message and message not in str(exc):
                raise AssertionError(f"{name}: unexpected rejection: {exc}") from exc
            ok(name)
        else:
            raise AssertionError(name + " was accepted")

    payloads, components = factory()
    with tempfile.TemporaryDirectory(prefix="constructed_v2_selector_") as tmp:
        root = Path(tmp)
        baseline, plan_path, out, captured = run_payloads(payloads, components, root / "baseline", capture_objective=True)
        ok("six_cohorts_including_observed_retention", set(selector.COHORTS) == {"general", "source_domain", "modern", "paraphrase", "legacy_main", "legacy_creative"})
        ok("all_synthetic_candidate_metrics_have_all_cohorts", all(set(r["metrics"]) == set(selector.COHORTS) for r in baseline["candidates"]))
        try:
            selector.fit_selection(plan_path, out)
        except FileExistsError:
            ok("existing_selection_cannot_be_overwritten")
        else:
            raise AssertionError("overwrite allowed")
        # Reconstruct only group membership and fitting design, independent of artifact internals.
        xfit, yfit, wfit = [], [], []
        for cohort in selector.COHORTS:
            ref = payloads[cohort][NAMES[0]]
            fit = fitting(ref["group_ids"])
            core = ref["lengths"] >= 128
            for group in set(ref["group_ids"]):
                ok_group = np.unique(fit[ref["group_ids"] == group])
                if len(ok_group) != 1:
                    raise AssertionError("one parent reached both halves")
            mask = fit & core
            xfit.append(np.stack([payloads[cohort][name]["raw_binary"][mask, 1] - payloads[cohort][name]["raw_binary"][mask, 0] for name in NAMES], axis=1))
            y = ref["y_ai"][mask]
            yfit.append(y)
            wfit.append(np.array([1 / len(selector.COHORTS) / 2 / np.sum(y == value) for value in y]))
        xx, yy, ww = np.concatenate(xfit), np.concatenate(yfit), np.concatenate(wfit)
        mean, scale = xx.mean(0), np.maximum(xx.std(0), 1e-6)
        ok("parent_group_never_crosses_fit_selection")
        ok("standardization_uses_core_fitting_rows_only", np.array_equal(baseline["mean"], mean) and np.array_equal(baseline["scale"], scale))
        ok("balanced_fit_objective_zero_logit_oracle", abs(captured["loss_zero"] - np.log(2)) < 1e-12 and abs(captured["gradient_zero"][-1]) < 1e-12)
        expected_gradient = (((xx - mean) / scale).T @ ((.5 - yy) * ww))
        ok("equal_cohort_and_within_class_fit_gradient", np.allclose(captured["gradient_zero"][:-1], expected_gradient, atol=1e-12, rtol=0))
        by_name = {r["name"]: r for r in baseline["candidates"]}
        a, b = by_name[NAMES[0]], by_name[NAMES[1]]
        ok("constructed_spiky_mean_exceeds_balanced_but_worst_is_lower", a["mean_cohort_quality"] > b["mean_cohort_quality"] and a["minimum_cohort_quality"] < b["minimum_cohort_quality"])
        quality_key = lambda r: (min(v["quality"] for v in r["metrics"].values()), np.mean([v["quality"] for v in r["metrics"].values()]), -np.mean([v["nll"] for v in r["metrics"].values()]), -np.count_nonzero(r["weights"]))
        ok("maximin_winner_matches_independent_priority_oracle", baseline["selected"]["name"] == max(baseline["candidates"], key=quality_key)["name"] and baseline["selected"]["name"] != NAMES[0])
        # Source and fine-identity arrays must be completely irrelevant to binary selection.
        changed = copy.deepcopy(payloads)
        for candidates in changed.values():
            for p in candidates.values():
                p["sources"] = np.array(["constructed-leak-" + str(x) for x in p["y_ai"]])
                p["family_true"] = np.array(["constructed-family" for _ in p["y_ai"]])
        source_result, _, _ = run_payloads(changed, components, root / "source_free")
        ok("source_and_fine_arrays_are_not_features", source_result["selected"] == baseline["selected"] and source_result["mean"] == baseline["mean"])
        changed = copy.deepcopy(payloads)
        for candidates in changed.values():
            for p in candidates.values():
                p["raw_binary"][p["lengths"] < 128] *= -1000
        short_result, _, _ = run_payloads(changed, components, root / "shorts")
        core_view = lambda r: {k: v for k, v in r.items() if k != "all_lengths_diagnostic_only"}
        ok("short_rows_never_affect_selection_or_fitting", core_view(short_result["selected"]) == core_view(baseline["selected"]) and short_result["mean"] == baseline["mean"] and short_result["scale"] == baseline["scale"])
        changed = copy.deepcopy(payloads)
        for candidates in changed.values():
            for p in candidates.values():
                mask = ~fitting(p["group_ids"])
                p["y_ai"][mask] = 1 - p["y_ai"][mask]
                p["raw_binary"][mask] *= -2
        selection_result, _, _ = run_payloads(changed, components, root / "selection_only")
        get_logistic = lambda artifact: next((r for r in artifact["candidates"] if r["name"] == "development_nonnegative_logistic"), None)
        la, lb = get_logistic(baseline), get_logistic(selection_result)
        ok("selection_half_does_not_change_fitted_optimizer_or_scale", selection_result["mean"] == baseline["mean"] and selection_result["scale"] == baseline["scale"] and la is not None and lb is not None and la["weights"] == lb["weights"] and la["intercept"] == lb["intercept"])

        def case(name, mutate, *, every=False, message=None):
            changed = copy.deepcopy(payloads)
            if every:
                for p in changed[selector.COHORTS[0]].values():
                    mutate(p)
            else:
                mutate(changed[selector.COHORTS[0]][NAMES[0]])
            reject(name, changed, components, root, message=message)

        case("calibrated_true_rejected", lambda p: p.update(calibrated=np.array(True)), message="uncalibrated")
        case("missing_calibrated_rejected", lambda p: p.pop("calibrated"), message="uncalibrated")
        case("integer_calibrated_flag_rejected", lambda p: p.update(calibrated=np.array(0)), message="uncalibrated")
        case("calibration_named_payload_rejected", lambda p: p.update(calibration_sha256=np.array("f" * 64)), message="calibration-bound")
        for purpose in ["calibration", "final", "test"]:
            case(purpose + "_purpose_rejected", lambda p, value=purpose: p.update(export_purpose=np.array(value)), message="development")
        case("fingerprint_drift_rejected", lambda p: p.update(component_weights_sha256=np.array("f" * 64)), message="identity")
        case("non_sha_fingerprint_rejected", lambda p: p.update(component_weights_sha256=np.array("foo")), message="SHA256")
        case("nan_raw_rejected", lambda p: p["raw_binary"].__setitem__((0, 0), np.nan), message="finite N")
        case("reordered_candidate_ids_rejected", lambda p: p.update(ids=p["ids"][::-1]), message="aligned")
        case("fractional_truth_rejected_before_cast", lambda p: p.update(y_ai=np.r_[.9, p["y_ai"][1:]]), every=True, message="binary truth")
        case("string_truth_rejected", lambda p: p.update(y_ai=p["y_ai"].astype(str)), every=True, message="binary truth")
        for name, value in [("nan", np.nan), ("infinite", np.inf), ("negative", -1), ("fractional", 128.5)]:
            # np.array_equal also rejects aligned NaNs before the reference-length check.
            case(name + "_length_rejected", lambda p, v=value: p.update(lengths=np.r_[v, p["lengths"][1:]]), every=True, message=None if name == "nan" else "word counts")
        case("blank_group_rejected", lambda p: p["group_ids"].__setitem__(0, " "), every=True, message="nonempty")
        case("blank_id_rejected", lambda p: p["ids"].__setitem__(0, ""), every=True, message="nonempty")
        case("byte_ids_rejected", lambda p: p.update(ids=p["ids"].astype("S")), every=True, message="nonempty")
        case("within_cohort_duplicate_ids_rejected", lambda p: p["ids"].__setitem__(1, p["ids"][0]), every=True, message="duplicate document")
        changed = copy.deepcopy(payloads)
        for p in changed[selector.COHORTS[1]].values():
            p["group_ids"][0] = payloads[selector.COHORTS[0]][NAMES[0]]["group_ids"][0]
        reject("cross_cohort_group_rejected", changed, components, root, message="parent components")
        changed = copy.deepcopy(payloads)
        for p in changed[selector.COHORTS[1]].values():
            p["ids"][0] = payloads[selector.COHORTS[0]][NAMES[0]]["ids"][0]
        reject("cross_cohort_id_rejected_even_with_different_group", changed, components, root, message="IDs must be disjoint")
        for flag in ["calibration_opened", "final_test_opened"]:
            reject(flag + "_plan_rejected", payloads, components, root, mutate_plan=lambda p, key=flag: p.update({key: True}), message="precede calibration")
        reject("bound_raw_sha_drift_rejected", payloads, components, root,
               mutate_plan=lambda p: p["cohorts"][selector.COHORTS[0]][NAMES[0]].update(sha256="0" * 64), message="bound file changed")
        reject("missing_necessary_cohort_rejected", payloads, components, root,
               mutate_plan=lambda p: p["cohorts"].pop(selector.COHORTS[-1]), message="development cohorts")
        # Strict threshold handles ties conservatively, and exactly separated tails as expected.
        y = np.r_[np.zeros(100, int), np.ones(10, int)]
        tied = selector.metrics(y, np.r_[np.zeros(100), np.ones(10)])
        ok("empirical_fpr_ties_are_conservative", tied["AI_recall_at_empirical_human_FPR_2pct"] == 1.0)
        edge = selector.metrics(y, np.r_[np.arange(100), np.full(5, 97.0), np.full(5, 98.0)])
        ok("empirical_fpr_order_statistic_strict_edge_oracle", edge["AI_recall_at_empirical_human_FPR_2pct"] == .5)
    ok("selector_source_did_not_change_during_checks", code_sha == selector.sha(selector.__file__))
    report = {"schema_version": 1, "passed": True, "constructed_only": True,
              "real_data_or_real_scores_read": False, "calibration_or_final_artifacts_opened": False,
              "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "selector_code_sha256": code_sha, "test_code_sha256": selector.sha(__file__),
              "cohorts": list(selector.COHORTS), "checks_passed": len(checks), "checks": checks}
    output = PROJECT / "research/english_v2_selector_checks.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ["passed", "constructed_only", "checks_passed", "selector_code_sha256"]}))
    return report


if __name__ == "__main__":
    main()
