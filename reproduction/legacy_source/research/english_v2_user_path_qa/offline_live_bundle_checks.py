"""Future v2 fixed-bundle engineering QA; preparation now is source/policy only.

Without --execute this prints the preregistered policy and binding-plan schema,
opening no bundle, corpus or scores. Actual execution requires root approval,
a hash-bound final artifact plan, and the final-export PID plus all recognized
MPS producers to have exited. It never fits, selects, changes or repairs a model.

  python -B research/english_v2_user_path_qa/offline_live_bundle_checks.py
  python -B research/english_v2_user_path_qa/offline_live_bundle_checks.py \
    --execute --root-approved --bindings FINAL_QA_PLAN --bindings-sha256 SHA \
    --final-export-pid PID --output NEW_DIRECTORY

This adds a source dependency on the unchanged v1 offline/live QA helper, in
addition to unchanged bundle_checks.run. Both must be in the source archive.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[2]
LEGACY_PATH = PROJECT/"research/english_v1_user_path_qa/offline_live_bundle_checks.py"
LEGACY_SHA = "e3f02b313a5c1eb7732692f4f653d8a06bd41ef604ed5bd2c4fb5b873ecd05fa"
BUNDLE_CHECKS_SHA = "04861add6a61718a75efd559c4fbecb67b49174243c9a564cdb31acb73012d5f"
PROTOCOL_SHA = "276c4757774c0f1867470a040c0a303209a6a2444e8481d4402e4693ce221da4"
SUITES = ("fresh_main", "fresh_source_domain", "fresh_modern_with_human_controls", "fresh_paraphrase_with_human_controls")
spec = importlib.util.spec_from_file_location("unchanged_v1_offline_live_qa", LEGACY_PATH)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
require, sha, read, resolve = legacy.require, legacy.sha, legacy.read, legacy.resolve
POLICY = copy.deepcopy(legacy.POLICY)
POLICY.update(cohort="registered v2 fresh_main, complete eligible final_sealed rows", seed=20261017,
    comparison="same fixed v2 binary calibration and inherited source rankings; no source gate; AI+7family+25leaf",
    source_policy="rankings_only; no family/model assertions online or offline",
    selection_inputs="SHA-bound final ID/truth/role/eligibility and existing shipping lengths only; no probability-based choice",
    empty_stratum="record absent, including expected empty short strata in a core-only cohort; never fabricate or replace",
    execution_gate="root-approved actual bundle ready; exact plan SHA; final export PID exited; no MPS producer",
    pretrained_model_or_recipe_changes=False)
BINDING_SCHEMA = {
    "schema_version": 1, "status": "READY_FOR_ROOT_AUTHORIZED_FIXED_BUNDLE_QA",
    "protocol": {"path": "registered protocol", "sha256": PROTOCOL_SHA},
    "data_plan": {"path": "registered release data plan", "sha256": "registered plan SHA"},
    "selection_freeze": {"path": "development_output/freeze.json", "sha256": "frozen selection SHA"},
    "selection_completed": {"path": "development_output/COMPLETED.json", "sha256": "completion SHA"},
    "calibration": {"path": "actual new calibration.json", "sha256": "calibration SHA"},
    "raw_export_plan": {"path": "raw_final/export_plan.json", "sha256": "complete plan SHA"},
    "raw_export_progress": {"path": "raw_final/export_progress.json", "sha256": "closed completed progress SHA"},
    "suite_reports": {name: {"report": {"path": "suite/report.json", "sha256": "report SHA"},
        "shipping": {"path": "suite/shipping.npz", "sha256": "shipping SHA"}} for name in SUITES},
    "acceptance": {"path": "closed acceptance JSON (pass or failure retained)", "sha256": "acceptance SHA"},
    "bundle": {"path": "actual complete bundle directory", "manifest_sha256": "bundle.json SHA"},
}


def bound(item, bindings):
    p = resolve(item["path"])
    require(sha(p) == item["sha256"], "root-approved artifact hash differs")
    bindings[str(p)] = item["sha256"]
    return p


def prepare(plan_path, expected_plan_sha, final_export_pid):
    """Future execution only: complete receipts/hashes, no corpus bodies/arrays."""
    evidence = legacy.idle(final_export_pid)
    require(sha(LEGACY_PATH) == LEGACY_SHA and sha(PROJECT/"experiments/english_v1/bundle_checks.py") == BUNDLE_CHECKS_SHA,
        "unchanged original QA dependencies differ")
    bindings = {}
    plan_path = bound({"path": str(plan_path), "sha256": expected_plan_sha}, bindings)
    plan = read(plan_path)
    require(plan.get("schema_version") == 1 and plan.get("status") == BINDING_SCHEMA["status"] and plan["protocol"]["sha256"] == PROTOCOL_SHA,
        "explicit root-ready v2 artifact plan required")
    artifacts = {key: bound(plan[key], bindings) for key in ("protocol", "data_plan", "selection_freeze", "selection_completed",
        "calibration", "raw_export_plan", "raw_export_progress", "acceptance")}
    protocol, data_plan, freeze, seal, cal, job, progress, acceptance = (read(artifacts[key]) for key in
        ("protocol", "data_plan", "selection_freeze", "selection_completed", "calibration", "raw_export_plan", "raw_export_progress", "acceptance"))
    require(protocol["status"] == "FROZEN_BEFORE_V2_TRAINING_AND_SCORES" and protocol["source_mode"] == "rankings_only" and
        protocol["release_data_plan"]["sha256"] == sha(artifacts["data_plan"]), "registered v2 data/decision policy differs")
    require(not (artifacts["selection_freeze"].parent/"FAILED.json").exists() and seal["freeze_sha256"] == sha(artifacts["selection_freeze"]) and
        seal.get("all_24_exports_complete") is True and seal.get("no_exports_running") is True and
        freeze["status"] == "development_selection_frozen" and freeze["protocol_sha256"] == PROTOCOL_SHA and
        freeze["calibration_opened"] is False and freeze["final_test_opened"] is False, "complete v2 development freeze required")
    blend_path = bound(freeze["artifacts"]["binary_blend"], bindings)
    blend_sha, cal_sha = sha(blend_path), sha(artifacts["calibration"])
    require(cal["provenance"]["v2_protocol_sha256"] == PROTOCOL_SHA and cal["provenance"]["blend_sha256"] == blend_sha and
        cal["provenance"]["only_binary_fitted_on_registered_calibration"] is True and
        cal["provenance"]["source_calibration_inheritance"]["artifact_sha256"] == protocol["source_calibration_sha256"] and
        cal["source_policy"] == {"mode": "rankings_only", "family_enabled": False, "model_enabled": False,
            "source_release_gate": None, "historical_cutoffs_are_inactive": True}, "new binary plus unchanged inactive source rankings required")
    required = set(freeze["required_components"])
    require(job["phase"] == "final" and job["selection_freeze_sha256"] == sha(artifacts["selection_freeze"]) and
        job["data_plan_sha256"] == sha(artifacts["data_plan"]) and job["blend_sha256"] == blend_sha and
        job["calibration_sha256"] == cal_sha and set(job["required_components"]) == required,
        "final raw-export plan differs from v2 freeze/calibration")
    require(progress.get("completed_at_utc") and progress["plan_sha256"] == sha(artifacts["raw_export_plan"]) and
        len(progress["completed"]) == len(job["tasks"]) == len(required)*len(SUITES), "every selected-component final raw export must complete")
    commits = {(r["dataset"], r["component"]): r for r in progress["completed"]}
    tasks = {(r["dataset"], r["name"]): r for r in job["tasks"]}
    expected_pairs = {(suite, component) for suite in SUITES for component in required}
    require(set(commits) == set(tasks) == expected_pairs and len(commits) == len(progress["completed"]) and
        len(tasks) == len(job["tasks"]), "raw tasks/commits duplicate, omit or add a suite/component")
    require(set(plan["suite_reports"]) == set(acceptance["suites"]) == set(protocol["necessary_final_suites"]) == set(SUITES) and
        acceptance["protocol_sha256"] == PROTOCOL_SHA and acceptance["blend_sha256"] == blend_sha and
        acceptance["calibration_sha256"] == cal_sha and acceptance["source_assertions_enabled"] is False,
        "all four reports and unchanged acceptance identities required; accuracy failures are retained")
    data = {r["name"]: r for r in data_plan["final_datasets"]}
    require(set(data) == set(SUITES), "exactly four registered parent final cohorts required")
    reports, shipping = {}, {}
    from experiments.english_v1.run_release_exports import complete
    for suite in SUITES:
        record = plan["suite_reports"][suite]
        report_path, shipping[suite] = bound(record["report"], bindings), bound(record["shipping"], bindings)
        reports[suite] = report = read(report_path)
        expected = protocol["necessary_final_suites"][suite]
        require(data[suite]["input_sha256"] == expected["input_sha256"] and data[suite]["expected_export_rows"] == expected["rows"] and
            acceptance["suites"][suite]["report"]["sha256"] == sha(report_path), "suite/acceptance input/count/report binding differs")
        require(report["provenance"]["input_sha256"] == expected["input_sha256"] and
            report["provenance"]["v2_protocol_sha256"] == PROTOCOL_SHA and report["provenance"]["blend_sha256"] == blend_sha and
            report["provenance"]["calibration_sha256"] == cal_sha and report["provenance"]["thresholds_refitted_on_evaluation"] is False and
            report["source_policy"]["source_assertions_enabled"] is False and report["all_lengths"]["binary"]["n"] == expected["rows"], "frozen full-parent shipping report required")
        require(set(report["provenance"]["raw_exports"]) == required, "shipping raw component identity set differs")
        for component in required:
            task, commit = tasks[(suite, component)], commits[(suite, component)]
            require(task["input_sha256"] == expected["input_sha256"] and task["expected_rows"] == expected["rows"] and complete(task, job), "a raw final export is incomplete or changed")
            raw = resolve(task["output"]); side = raw.with_suffix(".provenance.json"); provenance = report["provenance"]["raw_exports"][component]
            require(commit["output_sha256"] == provenance["sha256"] == sha(raw) and
                commit["sidecar_sha256"] == provenance["sidecar_sha256"] == sha(side), "raw commit and shipping evidence differ")
            bindings[str(raw)], bindings[str(side)] = sha(raw), sha(side)
    bundle = resolve(plan["bundle"]["path"])
    manifest_path = bound({"path": str(bundle/"bundle.json"), "sha256": plan["bundle"]["manifest_sha256"]}, bindings)
    manifest = read(manifest_path)
    require((bundle/"bundle.sha256").read_text().split()[0] == sha(manifest_path), "bundle checksum differs")
    for entry in manifest["integrity"]:
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe bundle member")
        bound({"path": str(bundle/relative), "sha256": entry["sha256"]}, bindings)
    require(manifest.get("source_release_gate") is None and manifest.get("source_release_protocol") is None and
        manifest.get("source_assertions") == {"valid": False, "family_enabled": False, "model_enabled": False} and
        set(manifest["models"]) == required and manifest["attribution_model"] == freeze["attribution_model"] and
        sha(bundle/manifest["binary_blend"]) == blend_sha and sha(bundle/manifest["calibration"]) == cal_sha, "actual v2 rankings-only bundle differs")
    from experiments.english_v1.fingerprints import metadata_fingerprint
    for name, component in manifest["models"].items():
        cp = bundle/component["path"]; weight = cp/"detector.safetensors" if component["type"] == "neural" else cp
        require(sha(weight) == component["weights_sha256"] == cal["provenance"]["component_weights_sha256"][name] and
            metadata_fingerprint(component["type"], cp) == cal["provenance"]["component_metadata_sha256"][name], "packaged component fingerprint differs")
    main = data["fresh_main"]
    input_path = bound({"path": main["input"], "sha256": main["input_sha256"]}, bindings)
    bound({"path": main["manifest"], "sha256": main["manifest_sha256"]}, bindings)
    for filename, expected in protocol["code_registry"].items():
        bound({"path": filename, "sha256": expected}, bindings)
    for p in (LEGACY_PATH, PROJECT/"experiments/english_v1/bundle_checks.py", bundle/"bundle.sha256", Path(__file__).resolve()):
        bindings[str(p)] = sha(p)
    evidence.update(protocol_sha256=PROTOCOL_SHA, binding_plan_sha256=expected_plan_sha,
        bundle_manifest_sha256=sha(manifest_path), input_sha256=main["input_sha256"], shipping_npz_sha256=sha(shipping["fresh_main"]),
        shipping_report_sha256=sha(resolve(plan["suite_reports"]["fresh_main"]["report"]["path"])),
        binary_blend_sha256=blend_sha, calibration_sha256=cal_sha, source_gate_sha256=None, source_mode="rankings_only",
        all_four_necessary_reports_and_acceptance_bound=True, model_id=manifest["model_id"])
    return bundle, input_path, shipping["fresh_main"], manifest, cal, bindings, evidence


def choose_ids(metadata, lengths):
    """Pure hash/length policy: no score or probability enters selection."""
    require(metadata and len({r["id"] for r in metadata}) == len(metadata) and set(lengths) == {r["id"] for r in metadata} and
        all(r["split"] == "final_sealed" and r.get("eligible", True) is True for r in metadata), "complete final_sealed IDs required")
    selected, absent = [], []
    for role, label, core in (("core_human", 0, True), ("core_AI", 1, True), ("short_human", 0, False), ("short_AI", 1, False)):
        pool = [r["id"] for r in metadata if r["y_ai"] == label and (lengths[r["id"]] >= 128) == core]
        if pool:
            selected.append((min(pool, key=legacy.ordering), role))
        else:
            absent.append(role)
    selected.append((min(lengths, key=lambda value: (-int(lengths[value]), *legacy.ordering(value))), "longest"))
    roles = {}
    for row_id, role in selected:
        roles.setdefault(row_id, []).append(role)
    require(0 < len(roles) <= 5, "fixed selection exceeds five unique rows")
    return roles, absent


def select_rows(input_path, shipping):
    import numpy as np
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS
    keys = ("ids", "y_ai", "lengths", "p_ai", "p_family", "p_model", "family_labels", "model_labels", "decision_AI", "decision_human",
        "decision_inconclusive", "family_accept", "model_accept", "language_supported", "encoder_tokens", "encoder_coverage", "window_counts")
    with np.load(shipping, allow_pickle=False) as archive:
        reference = {key: archive[key] for key in keys}
    require(list(reference["family_labels"]) == FAMILIES and list(reference["model_labels"]) == MODEL_LABELS and
        not reference["family_accept"].any() and not reference["model_accept"].any(), "shipping catalog or global source closure differs")
    ids = list(reference["ids"]); require(all(isinstance(v, str) and v for v in ids) and len(set(ids)) == len(ids), "unique shipping IDs required")
    metadata = pq.ParquetFile(input_path).read(columns=["id", "y_ai", "split", "eligible"]).to_pylist()
    require(len(metadata) == len(ids), "full parent denominator differs")
    index = {value: i for i, value in enumerate(ids)}
    require(all(r["id"] in index and r["y_ai"] == reference["y_ai"][index[r["id"]]] for r in metadata), "final/shipping truth differs")
    roles, absent = choose_ids(metadata, {value: int(reference["lengths"][i]) for i, value in enumerate(ids)})
    # Only the fixed <=5 selected bodies are exposed in Python, after all gates.
    chosen = ds.dataset(input_path, format="parquet").to_table(columns=["id", "text"], filter=ds.field("id").isin(list(roles))).to_pylist()
    by_id = {row["id"]: row["text"] for row in chosen}; require(len(chosen) == len(roles) and set(by_id) == set(roles), "selected text lookup differs")
    selected = list(roles); texts = [by_id[row_id] for row_id in selected]
    require(all(isinstance(text, str) and text.strip() for text in texts), "selected body missing")
    reference = {key: value if key in {"family_labels", "model_labels"} else value[[index[row_id] for row_id in selected]] for key, value in reference.items()}
    require(all(len(re.findall(r"\b\w+\b", text)) == reference["lengths"][i] for i, text in enumerate(texts)), "original full-text counts differ")
    return texts, reference, [{"id_sha256": legacy.hash_id(row_id), "selection_roles": roles[row_id]} for row_id in selected], absent


def run(args):
    require(args.root_approved, "root actual-bundle-ready approval required")
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from experiments.english_v1 import runtime  # noqa: F401
    destination = resolve(args.output); require(not destination.exists(), "QA output must be a new directory")
    # Do not reserve an output directory while another GPU producer is running.
    bundle, input_path, shipping, manifest, cal, bindings, evidence = prepare(args.bindings, args.bindings_sha256, args.final_export_pid)
    destination.mkdir(parents=True, exist_ok=False)
    legacy.new_json(destination/"policy.json", POLICY)
    report = {"started_at_utc": legacy.utc(), "passed": False, "qa_code_sha256": sha(__file__), "policy_sha256": sha(destination/"policy.json"),
        "legacy_comparison_helper_sha256": LEGACY_SHA, "provenance": evidence, "bindings": bindings,
        "contains_text": False, "fits_or_threshold_changes": False, "fixed_model_engineering_only": True, "stage": "synthetic_bundle_checks"}
    try:
        from experiments.english_v1.bundle_checks import run as bundle_run
        synthetic = bundle_run(bundle, destination/"synthetic_bundle_checks.json", device="cpu", compare_device="mps", tolerance=1e-4)
        require(synthetic.get("passed") is True, "original CPU/MPS synthetic bundle checks failed")
        report["synthetic_bundle_checks_sha256"] = sha(destination/"synthetic_bundle_checks.json")
        report["stage"] = "fixed_final_selection"
        texts, reference, selected, absent = select_rows(input_path, shipping)
        report.update(selected=selected, missing_strata=absent, selected_rows=len(texts))
        from experiments.english_v1.inference import EnglishDetector
        import numpy as np
        import torch
        vectors, comparisons = {}, []
        for device, tolerance in (("cpu", 1e-5), ("mps", 1e-4)):
            report["stage"] = "live_"+device
            detector = EnglishDetector.load(bundle, device=device, batch_size=4)
            results = detector.predict_many(texts)
            require(all(r["model_id"] == manifest["model_id"] and r["accepted_family"] is None and r["accepted_model"] is None for r in results), "actual no-gate runtime differs")
            items, vectors[device] = legacy.check_live(results, reference, selected, cal, device, tolerance)
            comparisons.extend(items)
            del detector, results
            gc.collect()
            if device == "mps":
                torch.mps.synchronize(); torch.mps.empty_cache()
        del texts
        cross = np.abs(vectors["cpu"]-vectors["mps"])
        report.update(comparisons=comparisons, cpu_mps_max_absolute_probability_difference=float(cross.max()), cpu_mps_probability_atol=1e-4,
            probability_equivalence_passed=all(r["probabilities_passed"] for r in comparisons) and bool((cross <= 1e-4).all()),
            decision_equivalence_passed=all(r["decisions_match"] for r in comparisons), quality_equivalence_passed=all(r["quality_match"] and r["full_token_coverage"] for r in comparisons))
        report["stage"] = "immutable_recheck"
        for filename, expected in bindings.items():
            require(sha(filename) == expected, "bound artifact changed during engineering QA")
        report["frozen_artifacts_unchanged"] = True
        report["passed"] = all(report[k] for k in ("probability_equivalence_passed", "decision_equivalence_passed", "quality_equivalence_passed"))
        report["stage"] = "completed"
    except BaseException as error:
        report["error_type"] = type(error).__name__  # Never log exception/input text.
    finally:
        report["completed_at_utc"] = legacy.utc()
        legacy.new_json(destination/"report.json", report)
    print(json.dumps({"passed": report["passed"], "stage": report["stage"], "selected_rows": report.get("selected_rows", 0), "report": str(destination/"report.json")}))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-approved", action="store_true")
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--bindings-sha256")
    parser.add_argument("--final-export-pid", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"policy": POLICY, "future_binding_plan_schema": BINDING_SCHEMA, "actual_execute_performed": False}, indent=2))
        return 0
    if not args.root_approved or any(v is None for v in (args.bindings, args.bindings_sha256, args.final_export_pid, args.output)):
        parser.error("execute requires root approval, exact binding-plan SHA, exited export PID and new output")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
