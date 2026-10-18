"""Export four registered binary candidates on six development cohorts, then freeze.

This coordinator never opens calibration/final data or fits calibration. It
requires the exact pretraining protocol SHA and the completed serial-training
seal, exports FP32 raw scores through the unchanged v1 wrapper, and runs only
the registered v2 binary selector. The old source pair/version-selection file
is bound without running source selection or changing its probabilities.

Protocol development fields: output_dir (new), cohorts (six path/bytes/SHA/rows
records), candidate_checkpoints (name:job/epoch/path), fixed_source_components
(type/checkpoint/weights_sha256/metadata_sha256), inherited_version_selection
(path/SHA), and code_sha256 (project-relative Python source registry).

Usage: python -m experiments.english_v2.run_development --protocol FILE \
  --sha256 FROZEN_SHA [--preflight-only]
No valid frozen/completed protocol means no output directory or subprocess.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from experiments.english_v1 import runtime  # noqa: F401
from experiments.english_v1.fingerprints import metadata_fingerprint
from experiments.english_v2 import select_binary as selector, train_registered as tr
from experiments.english_v2.binary_finalize import SOURCE_CALIBRATION_SHA256, SOURCE_NAME, SOURCE_WEIGHTS, STAT_NAME

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("minilm4_epoch1", "minilm4_epoch2", "modern3_epoch1", "modern3_epoch2")
CHECKPOINTS = (("Mini4", 1), ("Mini4", 2), ("Modern3", 1), ("Modern3", 2))
EXPORT_FILES = ("export_component.py", "neural.py", "statistical.py", "catalog.py", "language_guard.py")
REQUIRED_CODE = tuple("experiments/english_v1/"+name for name in (*EXPORT_FILES, "fingerprints.py", "runtime.py")) + (
    "experiments/english_v2/run_development.py", "experiments/english_v2/select_binary.py")
ROLES = {"validation", "dev", "dev_challenge", "dev_challenge_source_domain", "dev_backup", "dev_backup_allowed"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return tr.sha(path)


def path(value):
    value = Path(value)
    return (value if value.is_absolute() else ROOT/value).resolve()


def read(value):
    return json.loads(Path(value).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(destination, value):
    tr.write_new(destination, value)


def bound(spec):
    p = path(spec["path"])
    need(sha(p) == spec["sha256"] and ("bytes" not in spec or p.stat().st_size == spec["bytes"]), "registered file size/SHA changed")
    return p


def pid_exists(pid):
    need(isinstance(pid, int) and not isinstance(pid, bool) and pid > 0, "explicit exited training PID required")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def verify_code(protocol):
    registry = protocol["development"]["code_sha256"]
    need(set(REQUIRED_CODE) <= set(registry), "complete frozen development/export source registry required")
    for name, expected in registry.items():
        p = Path(name)
        need(not p.is_absolute() and ".." not in p.parts and p.suffix == ".py" and sha(path(name)) == expected,
            "registered development source changed")
        if name in protocol["training"]["code_sha256"]:
            need(protocol["training"]["code_sha256"][name] == expected, "training/development shared source differs")
    need(registry["experiments/english_v1/language_guard.py"] == protocol["language_rule_sha256"], "registered language rule differs")
    need(all(sha(path(name)) == expected for name, expected in protocol["training"]["code_sha256"].items()), "training source changed before development")
    return registry


def metadata_rows(spec, *, allowed_roles):
    import pyarrow.parquet as pq
    need({"path", "bytes", "sha256", "rows"} <= set(spec) and
        isinstance(spec["bytes"], int) and not isinstance(spec["bytes"], bool) and spec["bytes"] > 0 and
        isinstance(spec["rows"], int) and not isinstance(spec["rows"], bool) and spec["rows"] > 0,
        "prepared inputs require explicit positive integer byte and row counts")
    p = bound(spec)
    source = pq.ParquetFile(p)
    need(source.metadata.num_rows == spec["rows"], "registered metadata row count changed")
    columns = ["id", "group_id", "y_ai", "split"] + (["eligible"] if "eligible" in source.schema.names else [])
    rows = source.read(columns=columns).to_pylist()
    need(rows and all(r["split"] in allowed_roles and r.get("eligible", True) is True for r in rows), "development refuses nondevelopment roles or silent eligibility removal")
    need(all(isinstance(r["id"], str) and r["id"].strip() and isinstance(r["group_id"], str) and r["group_id"].strip() and
        isinstance(r["y_ai"], int) and not isinstance(r["y_ai"], bool) and r["y_ai"] in (0, 1) for r in rows), "explicit IDs/groups/binary truth required")
    need(len({r["id"] for r in rows}) == len(rows), "duplicate metadata document IDs")
    return rows


def preflight(protocol_path, expected_sha, *, exited=pid_exists):
    pp = path(protocol_path)
    need(sha(pp) == expected_sha, "explicit protocol SHA mismatch")
    protocol = read(pp)
    need(protocol.get("schema_version") == 1 and protocol.get("status") == "FROZEN_BEFORE_V2_TRAINING_AND_SCORES" and
        protocol.get("candidate_names") == list(NAMES) and protocol.get("binary_selector") == selector.SELECTOR and
        protocol.get("source_mode") == "rankings_only" and protocol.get("source_calibration_sha256") == SOURCE_CALIBRATION_SHA256,
        "exact pretraining candidate/selector/rankings-only policy required")
    registry = verify_code(protocol)
    training, development = protocol["training"], protocol["development"]
    work = path(training["work_dir"])
    completed = read(work/"COMPLETED.json")
    need(not (work/"FAILED.json").exists() and completed.get("protocol_sha256") == expected_sha and
        completed.get("all_children_exited") is True and completed.get("GPU_released_by_process_exit") is True and
        completed.get("calibration_final_opened") is False and set(completed["jobs"]) == {"Mini4", "Modern3"},
        "both serial training jobs must complete without failure")
    started, prepared = read(work/"STARTED.json"), read(work/"prepared.json")
    need(not exited(started["coordinator_pid"]) and started["protocol_sha256"] == prepared["protocol_sha256"] == expected_sha and
        started["prepared_sha256"] == sha(work/"prepared.json") and
        tr.inventory(work/"source_snapshot") == prepared["source_snapshot_sha256"] == training["code_sha256"], "training coordinator not exited or prepared source evidence differs")
    need(set(development["candidate_checkpoints"]) == set(NAMES) and set(development["cohorts"]) == set(selector.COHORTS), "four candidates and all six cohorts required")
    components = {}
    for name, (job_name, epoch) in zip(NAMES, CHECKPOINTS):
        spec = development["candidate_checkpoints"][name]
        job = next(j for j in training["jobs"] if j["name"] == job_name)
        need(spec["job"] == job_name and spec["epoch"] == epoch and path(spec["path"]) == path(job["snapshots"][str(epoch)]), "candidate does not identify the registered epoch snapshot")
        record = completed["jobs"][job_name]
        launch = read(work/(job_name+"_launch.json"))
        need(record.get("returncode") == 0 and record.get("pid_exited") is True and not exited(launch["pid"]) and
            read(work/(job_name+"_COMPLETED.json")) == record and launch["protocol_sha256"] == expected_sha,
            "training child did not exit with matching completion evidence")
        snap = record["snapshots"][str(epoch)]
        cp = path(spec["path"])
        files = tr.inventory(cp)
        need(files == snap["all_files_sha256"] and str(cp) == snap["path"] and snap["epoch"] == epoch and
            sha(cp/"detector.safetensors") == snap["weights_sha256"] and sha(cp/"detector.json") == snap["detector_metadata_sha256"],
            "epoch snapshot bytes differ from the committed training record")
        meta = read(cp/"detector.json")
        need(meta["weights_sha256"] == snap["weights_sha256"] and meta["architecture"]["family_loss_weight"] == meta["architecture"]["version_loss_weight"] == 0,
            "candidate is not the committed binary-only detector")
        components[name] = {"type": "neural", "checkpoint": str(cp), "weights_sha256": snap["weights_sha256"],
            "metadata_sha256": metadata_fingerprint("neural", cp), "detector_metadata_sha256": snap["detector_metadata_sha256"]}
    need(set(development["fixed_source_components"]) == set(SOURCE_WEIGHTS), "unchanged Mini3/stat2 source pair required")
    for name, spec in development["fixed_source_components"].items():
        kind = "statistical" if name == STAT_NAME else "neural"
        cp = path(spec["checkpoint"])
        weight = cp if kind == "statistical" else cp/"detector.safetensors"
        need(spec["type"] == kind and sha(weight) == spec["weights_sha256"] == SOURCE_WEIGHTS[name] and
            metadata_fingerprint(kind, cp) == spec["metadata_sha256"], "original source weights/configuration changed")
        components[name] = {**spec, "checkpoint": str(cp)}
    version_path = bound(development["inherited_version_selection"])
    need(read(version_path).get("enabled") is True, "the original active known-version selection must be inherited without selection")
    meta = {}
    seen_ids, seen_groups = set(), set()
    training_rows = metadata_rows(training["data"]["train"], allowed_roles={"train"})
    train_ids, train_groups = {r["id"] for r in training_rows}, {r["group_id"] for r in training_rows}
    for cohort in selector.COHORTS:
        rows = metadata_rows(development["cohorts"][cohort], allowed_roles=ROLES)
        ids, groups = {r["id"] for r in rows}, {r["group_id"] for r in rows}
        need(not ids & seen_ids and not groups & seen_groups and not ids & train_ids and not groups & train_groups,
            "development cohorts and actual training must have disjoint observable IDs/groups")
        seen_ids.update(ids); seen_groups.update(groups)
        meta[cohort] = rows
    # Output must never contain any bound source or be inside a source snapshot.
    out = path(development["output_dir"])
    protected = [work, *[path(s["checkpoint"]) for s in components.values()], version_path, pp,
        *[path(s["path"]) for s in development["cohorts"].values()]]
    need(not out.exists() and all(out != p and not out.is_relative_to(p) and not p.is_relative_to(out) for p in protected), "new independent output directory required")
    need(sha(pp) == expected_sha, "protocol changed during preflight")
    return {"protocol": protocol, "protocol_path": str(pp), "protocol_sha256": expected_sha, "components": components,
        "rows": meta, "training_completion_sha256": sha(work/"COMPLETED.json"), "code_sha256": registry,
        "inherited_version_selection": {"path": str(version_path), "sha256": sha(version_path)}}


def export_argv(evidence, name, cohort, target):
    p = evidence["protocol"]
    python = Path(p["training"]["python"])
    python = python.absolute() if python.is_absolute() else (ROOT/python).absolute()
    return [str(python), "-B", "-m", "experiments.english_v1.export_component", "--type", "neural",
        "--checkpoint", evidence["components"][name]["checkpoint"], "--input", str(path(p["development"]["cohorts"][cohort]["path"])),
        "--output", str(target), "--purpose", "development", "--device", "mps", "--batch-size", "16"]


def verify_export(evidence, name, cohort, target):
    import numpy as np
    target = Path(target)
    side = target.with_suffix(".provenance.json")
    value = read(side)
    component = evidence["components"][name]
    spec = evidence["protocol"]["development"]["cohorts"][cohort]
    expected = {"purpose": "development", "type": "neural", "checkpoint": component["checkpoint"],
        "weights_sha256": component["weights_sha256"], "component_metadata_sha256": component["metadata_sha256"],
        "input_sha256": spec["sha256"], "original_rows": spec["rows"], "eligible_export_rows": spec["rows"],
        "excluded_ineligible_rows": 0, "output_sha256": sha(target), "calibration_fitted": False,
        "frozen_blend_sha256": None, "frozen_calibration_sha256": None,
        "code_sha256": {n: evidence["code_sha256"]["experiments/english_v1/"+n] for n in EXPORT_FILES}}
    need(all(value.get(k) == v for k, v in expected.items()) and value.get("deployment_scope", {}).get("rule_sha256") == evidence["protocol"]["language_rule_sha256"],
        "raw sidecar role/input/count/model/exporter/guard fingerprints differ")
    rows = evidence["rows"][cohort]
    with np.load(target, allow_pickle=False) as archive:
        need(not any("calibration" in k.casefold() for k in archive.files), "development arrays must not be calibration-bound")
        need(archive["calibrated"].shape == () and archive["calibrated"].dtype.kind == "b" and archive["calibrated"].item() is False and
            archive["export_purpose"].shape == () and archive["export_purpose"].item() == "development", "explicit uncalibrated development arrays required")
        for key, field in (("ids", "id"), ("group_ids", "group_id"), ("y_ai", "y_ai")):
            values = archive[key]
            need(values.shape == (len(rows),) and values.tolist() == [r[field] for r in rows] and
                (key == "y_ai" or values.dtype.kind == "U"), "exported IDs/groups/truth do not match complete prepared metadata order")
        for key, expected_value in (("component_weights_sha256", component["weights_sha256"]), ("component_metadata_sha256", component["metadata_sha256"]),
                ("language_rule_sha256", evidence["protocol"]["language_rule_sha256"])):
            need(archive[key].shape == () and str(archive[key].item()) == expected_value, "raw scalar fingerprint differs")
        lengths = archive["lengths"]
        scope_lengths = archive["scope_word_like_tokens"]
        need(lengths.shape == (len(rows),) and lengths.dtype.kind in "iu" and (lengths >= 0).all() and
            scope_lengths.shape == lengths.shape and scope_lengths.dtype == lengths.dtype and scope_lengths.tobytes() == lengths.tobytes(), "scope word counts changed or misaligned")
        support = archive["language_supported"]
        need(support.shape == (len(rows),) and support.dtype.kind == "b" and int(support.sum()) == value["deployment_scope"]["supported_rows"], "language support must retain every original row")
        for key in ("language_probability", "ascii_letter_ratio"):
            values = archive[key]
            need(values.shape == (len(rows),) and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "aligned finite deployment language evidence required")
        raw = archive["raw_binary"]
        need(raw.shape == (len(rows), 2) and np.isfinite(raw).all(), "finite complete N-by2 raw scores required")
    return {"path": str(target), "sha256": sha(target), "provenance_path": str(side), "provenance_sha256": sha(side),
        "precision": "fp32", "rows": len(rows)}


def recheck_committed(evidence, protocol_path, expected_sha, exports):
    """Close the byte-identity chain on both sides of the selector call."""
    protocol = evidence["protocol"]
    verify_code(protocol)
    work = path(protocol["training"]["work_dir"])
    need(sha(path(protocol_path)) == expected_sha and not (work/"FAILED.json").exists() and
        sha(work/"COMPLETED.json") == evidence["training_completion_sha256"], "protocol/training completion changed")
    for spec in protocol["development"]["cohorts"].values():
        bound(spec)
    need(bound(protocol["development"]["inherited_version_selection"]) == Path(evidence["inherited_version_selection"]["path"]), "inherited source selection changed")
    for component in evidence["components"].values():
        cp = Path(component["checkpoint"])
        need(sha(cp if component["type"] == "statistical" else cp/"detector.safetensors") == component["weights_sha256"] and
            metadata_fingerprint(component["type"], cp) == component["metadata_sha256"], "frozen new/old component changed")
    for cohort in selector.COHORTS:
        for name in NAMES:
            need(verify_export(evidence, name, cohort, exports[cohort][name]["path"]) == exports[cohort][name], "committed raw export changed")


def run(protocol_path, expected_sha, *, guard=tr.mps_pids, exited=pid_exists, poll_seconds=1.0):
    evidence = preflight(protocol_path, expected_sha, exited=exited)
    need(not guard(), "another MPS process is active; no development subprocess launched")
    protocol = evidence["protocol"]
    out = path(protocol["development"]["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"STARTED.json", {"started_at_utc": now(), "coordinator_pid": os.getpid(), "protocol_sha256": expected_sha,
        "training_completion_sha256": evidence["training_completion_sha256"], "code_sha256": evidence["code_sha256"],
        "phase": "development_only", "precision": "fp32", "batch_size": 16})
    for name in evidence["code_sha256"]:
        import shutil
        destination = out/"source_snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path(name), destination)
    child, exports = None, {cohort: {} for cohort in selector.COHORTS}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupt(signum, frame):
        raise KeyboardInterrupt("development coordinator interrupted")
    for sig in previous:
        signal.signal(sig, interrupt)
    try:
        for name in NAMES:
            for cohort in selector.COHORTS:
                verify_code(protocol)
                need(sha(path(protocol_path)) == expected_sha and not guard(), "protocol changed or another MPS child remains")
                bound(protocol["development"]["cohorts"][cohort])
                cp = Path(evidence["components"][name]["checkpoint"])
                need(sha(cp/"detector.safetensors") == evidence["components"][name]["weights_sha256"] and
                    metadata_fingerprint("neural", cp) == evidence["components"][name]["metadata_sha256"], "candidate changed between exports")
                target = out/"raw"/name/(cohort+".npz")
                target.parent.mkdir(parents=True, exist_ok=True)
                need(not target.exists() and not target.with_suffix(".provenance.json").exists(), "raw output already has a writer")
                command = export_argv(evidence, name, cohort, target)
                log = out/"logs"/(name+"_"+cohort+".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("xb") as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false"),
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    write_new(log.with_suffix(".launch.json"), {"pid": child.pid, "argv": command, "started_at_utc": now(),
                        "protocol_sha256": expected_sha, "input_sha256": protocol["development"]["cohorts"][cohort]["sha256"]})
                    last_emit = time.monotonic()
                    while child.poll() is None:
                        if time.monotonic()-last_emit >= 45:
                            print(json.dumps({"phase": "development_export_running", "candidate": name, "cohort": cohort, "pid": child.pid}), flush=True)
                            last_emit = time.monotonic()
                        time.sleep(poll_seconds)
                    need(child.wait() == 0, "development child failed; later exports/selection will not run")
                record = verify_export(evidence, name, cohort, target)
                exports[cohort][name] = record
                child = None
                temporary = out/"progress.json.part"
                temporary.write_text(json.dumps({"protocol_sha256": expected_sha, "completed": exports, "last_completed_at_utc": now()}, indent=2)+"\n")
                temporary.replace(out/"progress.json")
                print(json.dumps({"phase": "development_export_committed", "candidate": name, "cohort": cohort,
                    "completed": sum(len(v) for v in exports.values()), "total": 24}), flush=True)
        need(not guard(), "MPS subprocess remains after raw exports")
        # Recheck all immutable sources, prepared inputs and every committed raw
        # file before selection. No calibration/final data is consulted.
        recheck_committed(evidence, protocol_path, expected_sha, exports)
        plan = {"protocol": {"path": str(path(protocol_path)), "sha256": expected_sha}, "names": list(NAMES),
            "components": {name: {key: evidence["components"][name][key] for key in ("weights_sha256", "metadata_sha256")} for name in NAMES},
            "cohorts": {cohort: {name: {key: exports[cohort][name][key] for key in ("path", "sha256")} for name in NAMES} for cohort in selector.COHORTS},
            "calibration_opened": False, "final_test_opened": False}
        write_new(out/"binary_selection_plan.json", plan)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):
            selection = selector.fit_selection(out/"binary_selection_plan.json", out/"binary_selection.json")
        recheck_committed(evidence, protocol_path, expected_sha, exports)
        blend = copy.deepcopy(selection)
        blend["components"].update({name: {key: evidence["components"][name][key] for key in ("weights_sha256", "metadata_sha256")} for name in SOURCE_WEIGHTS})
        blend["source_inheritance"] = {"source_calibration_sha256": SOURCE_CALIBRATION_SHA256,
            "attribution_model": SOURCE_NAME, "statistical_model_name": STAT_NAME, "mode": "rankings_only",
            "version_selection": evidence["inherited_version_selection"], "source_selection_refitted_on_v2": False}
        need(all(blend[key] == selection[key] for key in selection if key != "components"), "source binding changed binary selection mathematics")
        write_new(out/"binary_blend.json", blend)
        bound(protocol["development"]["inherited_version_selection"])
        needed = sorted({name for name, weight in zip(blend["names"], blend["selected"]["weights"]) if weight != 0} | set(SOURCE_WEIGHTS))
        artifacts = {"binary_blend": {"path": str(out/"binary_blend.json"), "sha256": sha(out/"binary_blend.json")},
            "binary_selection": {"path": str(out/"binary_selection.json"), "sha256": sha(out/"binary_selection.json")},
            "version_selection": evidence["inherited_version_selection"]}
        freeze = {"schema_version": 1, "created_at_utc": now(), "status": "development_selection_frozen",
            "components": evidence["components"], "required_components": needed, "attribution_model": SOURCE_NAME,
            "statistical_model_name": STAT_NAME, "version_statistics_name": STAT_NAME, "artifacts": artifacts,
            "protocol_sha256": expected_sha, "training_completion_sha256": evidence["training_completion_sha256"],
            "development_inputs": protocol["development"]["cohorts"], "development_exports": exports,
            "code_sha256": evidence["code_sha256"], "source_mode": "rankings_only", "source_calibration_sha256": SOURCE_CALIBRATION_SHA256,
            "version_selection_inherited_only": True, "calibration_opened": False, "final_test_opened": False}
        verify_code(protocol)
        need(sha(path(protocol_path)) == expected_sha and not guard() and tr.inventory(out/"source_snapshot") == evidence["code_sha256"],
            "source changed, source snapshot differs, or MPS process still active at seal")
        write_new(out/"freeze.json", freeze)
        seal = {"completed_at_utc": now(), "freeze_sha256": sha(out/"freeze.json"), "protocol_sha256": expected_sha,
            "all_24_exports_complete": True, "no_exports_running": True, "source_selection_refitted": False,
            "calibration_final_opened": False, "GPU_released_by_process_exit": True}
        write_new(out/"COMPLETED.json", seal)
        return seal
    except BaseException as error:
        tr.stop_child(child)
        write_new(out/"FAILED.json", {"failed_at_utc": now(), "error_type": type(error).__name__, "error": str(error),
            "active_child": child is not None and child.poll() is None, "later_exports_or_selection_will_not_start": True})
        raise
    finally:
        tr.stop_child(child)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight(args.protocol, args.sha256)
        print(json.dumps({"ready": True, "exports_started": False, "protocol_sha256": result["protocol_sha256"]}))
    else:
        print(json.dumps(run(args.protocol, args.sha256), indent=2))
