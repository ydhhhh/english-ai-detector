"""Fixed release engineering QA; never fit, select or alter a threshold.

PREPARATION (no bundle, corpus or probability file is opened)::

    .venv/bin/python -B research/english_v1_user_path_qa/offline_live_bundle_checks.py

EXECUTE ONLY after root confirms the actual bundle is packaged and the final
MPS export coordinator and its children have exited::

    .venv/bin/python -B research/english_v1_user_path_qa/offline_live_bundle_checks.py \
      --execute --bundle BUNDLE --final-export-pid PID --output NEW_QA_DIRECTORY

The existing bundle_checks.run CPU/MPS synthetic checks are retained unchanged
at atol=1e-4. Then two real detector instances (one per device, sequential)
each predict the identical fixed <=5 final texts once. No text, raw ID, group
ID, or probability vector is printed or saved. Engineering differences and
decision-boundary evidence are saved, without changing deployment artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
SEED = 20261017
POLICY = {
    "schema_version": 1,
    "purpose": "Fixed-model engineering consistency, not an accuracy estimate or a model/threshold selection step.",
    "seed": SEED,
    "cohort": "main final, eligible exported rows only",
    "selection": ["core human", "core AI", "short human", "short AI", "longest eligible row"],
    "core_definition": r"at least 128 original-text word-like tokens, regex \b\w+\b",
    "stratum_order": ["core_human", "core_AI", "short_human", "short_AI", "longest"],
    "within_stratum": "minimum SHA256(UTF8('20261017:' + exact string ID)); exact ID breaks a SHA tie",
    "longest": "maximum exported original word-like length; same SHA/ID order breaks ties",
    "deduplication": "by exact ID, preserve first selection role; at most five texts",
    "empty_stratum": "record absent; do not replace or search by model prediction",
    "selection_inputs": "projected final ID/truth/role/eligibility plus existing shipping lengths; no probabilities affect selection",
    "comparison": "same texts and frozen calibration/blend/source gate; AI + all 7 family + all 25 leaf probabilities",
    "cpu_probability_atol": 1e-5,
    "mps_probability_atol": 1e-4,
    "cpu_mps_probability_atol": 1e-4,
    "probability_rtol": 0,
    "decisions": "status and accepted_family/model must match exactly; every difference is disclosed even within probability tolerance",
    "quality": "word counts, complete encoder coverage, attribution encoder token/window counts, language support",
    "synthetic_checks": "existing experiments.english_v1.bundle_checks.run(cpu, compare_device=mps, tolerance=1e-4), unchanged",
    "data_retention": "no text, raw ID/group ID, or probability vectors; SHA identifiers and engineering comparisons only",
    "prohibited": ["training", "calibration fitting", "threshold adjustment", "candidate selection", "resampling after results"],
}
PINNED = {
    "research/english_v1_finalization_plan.json": "afb1c17db12a1e50cb485472fe88b593b4b64ab6e1d2b31ea0fcd277ea053afd",
    "results/english_v1/selection_frozen_v1/freeze.json": "6283139a43393f316693d34cd1647aa08cf2c121e189e1aa4d2a648ddbe230d2",
    "results/english_v1/release_v1/calibration.json": "1b3cac20a654f1b0cb27f2784be172b9e59fa21b4858f1e5d2cd253337521ff4",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT / value).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def new_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def hash_id(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def ordering(value):
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).digest(), value


def idle(final_export_pid):
    require(final_export_pid > 0 and final_export_pid != os.getpid(), "a real prior final-export PID is required")
    try:
        os.kill(final_export_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("the recorded final export PID has not exited")
    # The coordinator must have exited and no orphan neural/export GPU child
    # may remain. Do not emit the process listing or match our own QA command.
    listing = subprocess.run(["ps", "-axo", "pid=,command="], check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    active = []
    for line in listing.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        command = fields[1]
        producer = any(name in command for name in (
            "experiments.english_v1.export_component", "experiments.english_v1.neural",
            "experiments.english_v1.run_release_exports"))
        if producer and re.search(r"--device(?:=|\s+)mps(?:\s|$)", command):
            active.append(int(fields[0]))
    require(not active, "a neural/export MPS process is still active")
    return {"recorded_final_export_pid": final_export_pid, "recorded_pid_exited": True,
            "recognized_mps_producers_running": False}


def prepare(bundle, final_export_pid):
    """Receipt/hash checks only; no corpus text or NPZ arrays are opened here."""
    evidence = idle(final_export_pid)
    bindings = {str(PROJECT / path): value for path, value in PINNED.items()}
    for path, expected in bindings.items():
        require(sha(path) == expected, "frozen release identity changed")
    release = PROJECT / "results/english_v1/release_v1"
    plan_path = PROJECT / "research/english_v1_finalization_plan.json"
    plan = read(plan_path)
    matches = [item for item in plan["final_datasets"] if item["name"] == "main"]
    require(len(matches) == 1, "one fixed main-final cohort is required")
    main = matches[0]
    input_path, manifest_path = resolve(main["input"]), resolve(main["manifest"])
    require(sha(input_path) == main["input_sha256"] and sha(manifest_path) == main["manifest_sha256"],
            "main final input/manifest changed")
    for path in (input_path, manifest_path):
        bindings[str(path)] = sha(path)
    execution = release / "evaluation_execution"
    require(not (execution / "FAILED.json").exists(), "release evaluation has a failure marker")
    seal, state = read(execution / "COMPLETED.json"), read(execution / "progress.json")
    require(seal["execution_sha256"] == sha(execution / "progress.json") and
            seal["protocol_sha256"] == sha(execution / "protocol.json"), "evaluation seal changed")
    require(state.get("phase") == "completed" and state.get("frozen_bindings_unchanged") is True and
            len(state["candidate_completed"]) == len(state["shipping_completed"]) == 10,
            "all ten release evaluations must be complete")
    raw_plan_path = release / "raw_final/export_plan.json"
    raw_progress_path = release / "raw_final/export_progress.json"
    job, progress = read(raw_plan_path), read(raw_progress_path)
    require(job.get("phase") == "final" and job.get("data_plan_sha256") == sha(plan_path) and
            job.get("selection_freeze_sha256") == PINNED["results/english_v1/selection_frozen_v1/freeze.json"],
            "raw export plan identity changed")
    require(progress.get("completed_at_utc") and progress.get("plan_sha256") == sha(raw_plan_path) and
            len(progress["completed"]) == len(job["tasks"]) == 30 and
            state["raw_export_progress_sha256"] == sha(raw_progress_path), "all final raw exports must be complete")
    committed = {(v["dataset"], v["component"]): v for v in progress["completed"]}
    require(len(committed) == 30, "raw export progress contains duplicate commits")
    bundle = resolve(bundle)
    manifest = read(bundle / "bundle.json")
    require((bundle / "bundle.sha256").read_text().split()[0] == sha(bundle / "bundle.json"), "bundle checksum mismatch")
    for item in manifest["integrity"]:
        require(sha(bundle / item["path"]) == item["sha256"], "bundle integrity changed")
        bindings[str(bundle / item["path"])] = item["sha256"]
    require(manifest.get("source_release_gate"), "the actual shipping source gate must be bundled")
    require(sha(bundle / manifest["binary_blend"]) == job["blend_sha256"] and
            sha(bundle / manifest["calibration"]) == job["calibration_sha256"] and
            sha(bundle / manifest["source_release_gate"]) == state["source_gate_sha256"], "bundle differs from shipping artifacts")
    shipping = release / "evaluation_shipping/main.npz"
    shipping_report = release / "evaluation_shipping/main.json"
    events = [e for e in state["events"] if e.get("phase") == "shipping" and e.get("cohort") == "main"]
    require(len(events) == 1, "one immutable main shipping evaluation is required")
    event = events[0]
    require(sha(shipping) == event["npz_sha256"] and sha(shipping_report) == event["report_sha256"], "shipping payload/report changed")
    provenance = read(shipping_report)["provenance"]
    require(provenance["blend_sha256"] == job["blend_sha256"] and
            provenance["calibration_sha256"] == job["calibration_sha256"] and
            provenance["source_release_gate_sha256"] == state["source_gate_sha256"] and
            provenance.get("thresholds_refitted_on_evaluation") is False, "shipping calibration/gate binding differs")
    require(manifest["attribution_model"] == job["attribution_model"] and
            set(manifest["models"]) == set(job["required_components"]), "bundle component set differs")
    from experiments.english_v1.run_release_exports import complete
    from experiments.english_v1.fingerprints import metadata_fingerprint
    cal = read(bundle / manifest["calibration"])
    for name in job["required_components"]:
        tasks = [t for t in job["tasks"] if t["dataset"] == "main" and t["name"] == name]
        require(len(tasks) == 1, "duplicate or missing main export task")
        task = tasks[0]
        require(task["input_sha256"] == main["input_sha256"] and
                task["expected_rows"] == main["expected_export_rows"] and complete(task, job), "main raw export is not independently complete")
        raw = resolve(task["output"])
        receipt = committed[("main", name)]
        require(receipt["output_sha256"] == sha(raw) and
                receipt["sidecar_sha256"] == sha(raw.with_suffix(".provenance.json")) and
                provenance["inputs"][name]["sha256"] == sha(raw), "main raw commit/shipping identity differs")
        component = manifest["models"][name]
        path = bundle / component["path"]
        weight_path = path / "detector.safetensors" if component["type"] == "neural" else path
        require(component["type"] == task["type"] and component["weights_sha256"] == task["weights_sha256"] and
                sha(weight_path) == task["weights_sha256"] and
                metadata_fingerprint(component["type"], path) == task["metadata_sha256"] and
                cal["provenance"]["component_weights_sha256"][name] == task["weights_sha256"], "packaged component differs from frozen inference")
    for path in (raw_plan_path, raw_progress_path, execution / "COMPLETED.json", execution / "progress.json",
                 execution / "protocol.json", bundle / "bundle.json", bundle / "bundle.sha256", shipping, shipping_report):
        bindings[str(path)] = sha(path)
    code = PROJECT / "experiments/english_v1"
    for name in ("inference.py", "bundle_checks.py", "neural.py", "statistical.py", "calibration.py",
                 "language_guard.py", "catalog.py", "select_binary.py", "version_refinement.py", "source_release_gate.py"):
        bindings[str(code / name)] = sha(code / name)
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    evidence.update({"bundle_manifest_sha256": sha(bundle / "bundle.json"),
        "input_sha256": main["input_sha256"], "shipping_npz_sha256": sha(shipping),
        "shipping_report_sha256": sha(shipping_report), "evaluation_completion_sha256": sha(execution / "COMPLETED.json"),
        "calibration_sha256": job["calibration_sha256"], "binary_blend_sha256": job["blend_sha256"],
        "source_gate_sha256": state["source_gate_sha256"], "model_id": manifest["model_id"]})
    return bundle, input_path, shipping, manifest, cal, bindings, evidence


def select_rows(input_path, shipping):
    import numpy as np
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS
    needed = ("ids", "y_ai", "lengths", "p_ai", "p_family", "p_model", "family_labels", "model_labels",
        "decision_AI", "decision_human", "decision_inconclusive", "family_accept", "model_accept",
        "language_supported", "encoder_tokens", "encoder_coverage", "window_counts")
    with np.load(shipping, allow_pickle=False) as archive:
        reference = {key: archive[key] for key in needed}
    require(list(reference["family_labels"]) == FAMILIES and list(reference["model_labels"]) == MODEL_LABELS,
            "shipping class order differs from runtime")
    ids = list(reference["ids"])
    require(all(isinstance(value, str) and value for value in ids) and len(set(ids)) == len(ids), "shipping IDs are not unique strings")
    source = pq.ParquetFile(input_path)
    columns = ["id", "y_ai", "split"] + (["eligible"] if "eligible" in source.schema.names else [])
    # Only these metadata columns enter selection. Corpus text is never scanned
    # to inspect examples or choose probabilities close to a threshold.
    metadata = source.read(columns=columns).to_pylist()
    require(all(r["split"] == "test" for r in metadata), "main corpus must have only its frozen test role")
    metadata = [r for r in metadata if r.get("eligible", True)]
    require(len({r["id"] for r in metadata}) == len(metadata) and set(r["id"] for r in metadata) == set(ids), "main and shipping eligible IDs differ")
    index = {value: i for i, value in enumerate(ids)}
    require(all(r["y_ai"] == reference["y_ai"][index[r["id"]]] for r in metadata), "main/shipping binary truth differs")
    selected, absent = [], []
    for role, label, core in (("core_human", 0, True), ("core_AI", 1, True),
                              ("short_human", 0, False), ("short_AI", 1, False)):
        pool = [r["id"] for r in metadata if r["y_ai"] == label and (reference["lengths"][index[r["id"]]] >= 128) == core]
        if pool:
            selected.append((min(pool, key=ordering), role))
        else:
            absent.append(role)
    selected.append((min(ids, key=lambda value: (-int(reference["lengths"][index[value]]), *ordering(value))), "longest"))
    roles = {}
    for value, role in selected:
        roles.setdefault(value, []).append(role)
    require(0 < len(roles) <= 5, "fixed selection must contain at most five rows")
    # Arrow applies an ID filter and exposes only the selected text values to
    # Python. No full-corpus body list is created or written.
    chosen = ds.dataset(input_path, format="parquet").to_table(columns=["id", "text"],
        filter=ds.field("id").isin(list(roles))).to_pylist()
    by_id = {r["id"]: r["text"] for r in chosen}
    require(len(chosen) == len(roles) and set(by_id) == set(roles), "selected text retrieval differs")
    values = list(roles)
    texts = [by_id[value] for value in values]
    require(all(isinstance(text, str) and text.strip() for text in texts), "selected final row has no usable text")
    selected_reference = {key: (value[[index[row] for row in values]] if key not in {"family_labels", "model_labels"} else value)
                          for key, value in reference.items()}
    for j, text in enumerate(texts):
        require(len(re.findall(r"\b\w+\b", text)) == selected_reference["lengths"][j], "original and shipping word-like lengths differ")
    return texts, selected_reference, [{"id_sha256": hash_id(value), "selection_roles": roles[value]} for value in values], absent


def decisions(reference, i):
    import numpy as np
    from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS
    statuses = [bool(reference[key][i]) for key in ("decision_AI", "decision_human", "decision_inconclusive")]
    require(sum(statuses) == 1, "shipping decisions are not exclusive/exhaustive")
    return {"status": ("AI-generated", "likely human", "inconclusive")[statuses.index(True)],
        "accepted_family": FAMILIES[int(np.argmax(reference["p_family"][i]))] if reference["family_accept"][i] else None,
        "accepted_model": MODEL_LABELS[int(np.argmax(reference["p_model"][i]))] if reference["model_accept"][i] else None}


def vector(result):
    import numpy as np
    from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS
    family = {r["label"]: r["probability"] for r in result["family_probabilities"]}
    leaves = {r["label"]: r["probability"] for r in result["model_probabilities"]}
    require(len(family) == 7 and set(family) == set(FAMILIES) and len(leaves) == 25 and set(leaves) == set(MODEL_LABELS), "live labels differ")
    return np.asarray([result["probabilities"]["AI"], *[family[k] for k in FAMILIES], *[leaves[k] for k in MODEL_LABELS]], dtype=np.float64)


def boundary(vector_value, thresholds):
    import numpy as np
    from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS
    ai, family, leaves = vector_value[0], vector_value[1:8], vector_value[8:]
    fi, mi = int(np.argmax(family)), int(np.argmax(leaves))
    return {"AI_minus_cutoff": float(ai - thresholds["AI"]["threshold"]),
        "human_minus_cutoff": float(1 - ai - thresholds["human"]["threshold"]),
        "family_argmax": FAMILIES[fi], "family_max_minus_cutoff": float(family[fi] - thresholds["family"]["threshold"]),
        "leaf_argmax": MODEL_LABELS[mi], "leaf_max_minus_cutoff": float(leaves[mi] - thresholds["model"]["threshold"])}


def check_live(results, reference, selected, cal, device, atol):
    import numpy as np
    from experiments.english_v1.bundle_checks import validate_result
    require(len(results) == len(selected), "live output row count differs")
    comparisons, vectors = [], []
    for i, result in enumerate(results):
        validate_result(result)
        live = vector(result)
        expected = np.concatenate(([reference["p_ai"][i]], reference["p_family"][i], reference["p_model"][i])).astype(np.float64)
        require(np.isfinite(expected).all() and expected.shape == (33,), "shipping probabilities are invalid")
        diff = np.abs(expected - live)
        expected_decisions = decisions(reference, i)
        actual_decisions = {key: result[key] for key in expected_decisions}
        quality = result["quality"]
        expected_quality = {"word_like_tokens": int(reference["lengths"][i]),
            "core_length_supported": bool(reference["lengths"][i] >= 128),
            "encoder_tokens": int(reference["encoder_tokens"][i]), "encoder_coverage": float(reference["encoder_coverage"][i]),
            "window_counts": int(reference["window_counts"][i]),
            "language": "en" if bool(reference["language_supported"][i]) else "unsupported"}
        actual_quality = {key: quality[key] for key in expected_quality}
        equal_decisions = actual_decisions == expected_decisions
        item = {**selected[i], "device": device, "probability_atol": atol,
            "AI_absolute_error": float(diff[0]), "family_max_absolute_error": float(diff[1:8].max()),
            "leaf_max_absolute_error": float(diff[8:].max()), "probabilities_passed": bool((diff <= atol).all()),
            "offline_decisions": expected_decisions, "live_decisions": actual_decisions, "decisions_match": equal_decisions,
            "offline_quality": expected_quality, "live_quality": actual_quality,
            "quality_match": expected_quality == actual_quality,
            "full_token_coverage": expected_quality["encoder_coverage"] == quality["encoder_coverage"] == 1.0}
        if not equal_decisions:
            item["decision_difference_disclosure"] = {"offline": boundary(expected, cal["thresholds"]),
                "live": boundary(live, cal["thresholds"]),
                "interpretation": "No thresholds changed. Probability agreement alone does not establish decision agreement at a boundary."}
        comparisons.append(item)
        vectors.append(live)
    return comparisons, np.stack(vectors)


def run(args):
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from experiments.english_v1 import runtime  # noqa: F401
    destination = resolve(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    new_json(destination / "policy.json", POLICY)
    report = {"started_at_utc": utc(), "passed": False, "policy_sha256": sha(destination / "policy.json"),
              "qa_code_sha256": sha(__file__), "contains_text": False, "fits_or_threshold_changes": False,
              "fixed_model_engineering_only": True, "stage": "readiness"}
    try:
        bundle, input_path, shipping, manifest, cal, bindings, evidence = prepare(args.bundle, args.final_export_pid)
        report["provenance"] = evidence
        report["bindings"] = bindings
        report["stage"] = "existing_synthetic_bundle_checks"
        from experiments.english_v1.bundle_checks import run as bundle_run
        synthetic = bundle_run(bundle, destination / "synthetic_bundle_checks.json", device="cpu",
                               compare_device="mps", tolerance=1e-4)
        require(synthetic.get("passed") is True, "existing CPU/MPS synthetic checks failed")
        report["synthetic_bundle_checks_sha256"] = sha(destination / "synthetic_bundle_checks.json")
        report["stage"] = "mechanical_final_selection"
        texts, reference, selected, absent = select_rows(input_path, shipping)
        report.update({"selected": selected, "missing_strata": absent, "selected_rows": len(texts)})
        from experiments.english_v1.inference import EnglishDetector
        import numpy as np
        import torch
        vectors, comparisons = {}, []
        for device, tolerance in (("cpu", 1e-5), ("mps", 1e-4)):
            report["stage"] = "live_" + device
            detector = EnglishDetector.load(bundle, device=device, batch_size=4)
            results = detector.predict_many(texts)
            require(all(r["model_id"] == manifest["model_id"] for r in results), "live model ID differs")
            items, vectors[device] = check_live(results, reference, selected, cal, device, tolerance)
            comparisons.extend(items)
            del detector, results
            gc.collect()
            if device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
        del texts
        report["comparisons"] = comparisons
        cross = np.abs(vectors["cpu"] - vectors["mps"])
        report["cpu_mps_max_absolute_probability_difference"] = float(cross.max())
        report["cpu_mps_probability_atol"] = 1e-4
        report["probability_equivalence_passed"] = all(r["probabilities_passed"] for r in comparisons) and bool((cross <= 1e-4).all())
        report["decision_equivalence_passed"] = all(r["decisions_match"] for r in comparisons)
        report["quality_equivalence_passed"] = all(r["quality_match"] and r["full_token_coverage"] for r in comparisons)
        report["stage"] = "recheck_immutable_artifacts"
        for path, expected in bindings.items():
            require(sha(path) == expected, "a frozen artifact changed during engineering QA")
        report["frozen_artifacts_unchanged"] = True
        report["passed"] = all(report[key] for key in ("probability_equivalence_passed", "decision_equivalence_passed", "quality_equivalence_passed"))
        report["stage"] = "completed"
    except BaseException as error:
        # Deliberately omit exception strings: external libraries could include
        # an input value. Source/stage/type suffice to investigate privately.
        report["error_type"] = type(error).__name__
    finally:
        report["completed_at_utc"] = utc()
        new_json(destination / "report.json", report)
    print(json.dumps({"passed": report["passed"], "stage": report["stage"], "report": str(destination / "report.json"),
                      "selected_rows": report.get("selected_rows", 0)}, allow_nan=False))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Only after root announces actual bundle ready and final MPS exports exited")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--final-export-pid", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(POLICY, indent=2))
        return 0
    if args.bundle is None or args.final_export_pid is None or args.output is None:
        parser.error("--execute requires --bundle, --final-export-pid and a new --output directory")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
