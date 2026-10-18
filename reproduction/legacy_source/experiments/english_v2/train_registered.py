"""Register and serialize exactly Mini4 then Modern3; no unfrozen launch.

prepare/run both require --protocol and its independently supplied --sha256.
prepare verifies the frozen data/QA/code/parent contract, creates two NEW
binary initializers, hashes them, and writes an exclusive prepared receipt.
run rechecks it, launches the immutable v1 trainer sequentially, snapshots both
epochs, and stops. No model selection, calibration or final export occurs.

CLI:
  python -m experiments.english_v2.train_registered prepare --protocol FILE --sha256 SHA
  python -m experiments.english_v2.train_registered run     --protocol FILE --sha256 SHA

Protocol schema1, status=FROZEN_BEFORE_V2_TRAINING_AND_SCORES, training:
  work_dir: NEW_AUDIT_DIRECTORY
  python: .venv/bin/python
  data: {train, validation, dev_challenge: {path,bytes,sha256,rows}}
  data_qa: {path,bytes,sha256}
  code_sha256: {project-relative Python path:SHA,...} including REQUIRED_CODE
  jobs: exactly ordered Mini4 then Modern3, each:
    name; parent:{path,weights_sha256,detector_metadata_sha256};
    initializer; run_dir; snapshots:{"1":NEW_DIR,"2":NEW_DIR};
    hyperparameters: exactly HP_FLAGS keys (see below).
QA JSON must contain passed:true and training_inputs_sha256 mapping all three
data roles to their exact SHA. Parent/QA/data roles are verified without text
inspection. The original trainer performs its existing text disjoint checks.

Fixed constraints:2epochs; Mini4 B32 FP32/bert/class_domain_v3,
Modern3 B16 BF16/modernbert/class_normalized_v2; binary1/family0/version0;
512/448;sdpa. All remaining
hyperparameters are explicitly registered and passed unchanged, with no hidden
max_train/resume option. Initializer/training outputs must be new paths.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
STATUS = "FROZEN_BEFORE_V2_TRAINING_AND_SCORES"
REQUIRED_CODE = ("experiments/english_v1/neural.py", "experiments/english_v1/catalog.py",
    "experiments/english_v2/prepare_binary_init.py", "experiments/english_v2/train_registered.py")
HP_FLAGS = ("epochs", "batch_size", "precision", "lr", "weight_mode", "family_loss_weight", "version_loss_weight",
    "max_length", "stride", "attention", "seed", "weight_decay", "warmup_ratio", "clip_grad", "cpu_threads", "save_every")


def now():
    return datetime.now(timezone.utc).isoformat()


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def path(value):
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def read(value):
    return json.loads(Path(value).read_text())


def write_new(destination, value):
    with Path(destination).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inventory(directory):
    return {p.relative_to(directory).as_posix(): sha(p) for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def bound_file(spec):
    target = path(spec["path"])
    need(isinstance(spec["bytes"], int) and spec["bytes"] > 0 and target.is_file() and
        target.stat().st_size == spec["bytes"] and sha(target) == spec["sha256"], "registered file size/SHA mismatch: " + str(target))
    return target


def verified(protocol_path, expected_sha256, *, initializers_exist=False):
    protocol_path = path(protocol_path)
    need(re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or "") and sha(protocol_path) == expected_sha256, "explicit protocol SHA mismatch")
    protocol = read(protocol_path)
    need(protocol.get("schema_version") == 1 and protocol.get("status") == STATUS, "no frozen v2protocol; refusing preparation or launch")
    training = protocol["training"]
    need(set(REQUIRED_CODE) <= set(training["code_sha256"]), "registered frozen trainer/catalog/adapter/coordinator code hashes required")
    for name, expected in training["code_sha256"].items():
        relative = Path(name)
        need(not relative.is_absolute() and ".." not in relative.parts and relative.suffix == ".py" and relative.as_posix() == name,
            "code registry must use canonical safe project-relative Python paths")
        need(sha(path(name)) == expected, "registered execution source changed: " + name)
    need(set(training["data"]) == {"train", "validation", "dev_challenge"}, "only the registered train/two development roles may be opened")
    inputs = {role: bound_file(spec) for role, spec in training["data"].items()}
    qa = read(bound_file(training["data_qa"]))
    need(qa.get("passed") is True and qa.get("training_inputs_sha256") == {role: spec["sha256"] for role, spec in training["data"].items()}, "independent data QA seal must pass and bind all three input roles")
    from experiments.english_v1 import runtime  # noqa: F401
    import pyarrow.parquet as pq
    role_sets = {"train": {"train"}, "validation": {"validation", "dev"}, "dev_challenge": {"dev_challenge", "dev_challenge_source_domain"}}
    identifiers, groups = {}, {}
    for role, target in inputs.items():
        spec = training["data"][role]
        source = pq.ParquetFile(target)
        need(isinstance(spec["rows"], int) and spec["rows"] > 0 and source.metadata.num_rows == spec["rows"], "registered row count changed")
        rows = source.read(columns=["id", "group_id", "split"]).to_pylist()
        need(rows and all(r["split"] in role_sets[role] for r in rows), "registered Parquet has a wrong training/development role")
        need(all(isinstance(r["id"], str) and r["id"].strip() and isinstance(r["group_id"], str) and r["group_id"].strip() for r in rows), "explicit nonempty IDs/groups required")
        identifiers[role], groups[role] = {r["id"] for r in rows}, {r["group_id"] for r in rows}
        need(len(identifiers[role]) == len(rows), "duplicate prepared row IDs")
    for i, role in enumerate(inputs):
        for other in list(inputs)[i+1:]:
            need(not identifiers[role] & identifiers[other] and not groups[role] & groups[other], "train/development observable IDs/groups overlap")
    jobs = training["jobs"]
    need([j["name"] for j in jobs] == ["Mini4", "Modern3"], "fixed Mini4 then Modern3 order required")
    output_paths = [path(training["work_dir"])]
    for job, batch, precision, architecture, weight_mode in zip(jobs, (32, 16), ("fp32", "bf16"),
            ("bert", "modernbert"), ("class_domain_v3", "class_normalized_v2")):
        hp = job["hyperparameters"]
        need(set(hp) == set(HP_FLAGS), "all and only supported hyperparameters must be registered")
        expected = {"epochs": 2, "batch_size": batch, "precision": precision, "weight_mode": weight_mode,
            "family_loss_weight": 0, "version_loss_weight": 0, "max_length": 512, "stride": 448, "attention": "sdpa"}
        need(all(hp[k] == v for k, v in expected.items()), "registered binary-only architecture/batch/precision changed")
        need(all(isinstance(hp[k], (int, float)) and not isinstance(hp[k], bool) and math.isfinite(hp[k]) for k in ("lr", "weight_decay", "warmup_ratio", "clip_grad")) and
            hp["lr"] > 0 and hp["weight_decay"] >= 0 and 0 <= hp["warmup_ratio"] < 1 and hp["clip_grad"] > 0, "invalid registered optimizer hyperparameters")
        need(all(isinstance(hp[k], int) and not isinstance(hp[k], bool) for k in ("epochs", "batch_size", "max_length", "stride", "seed", "cpu_threads", "save_every")) and
            0 <= hp["seed"] < 2**32 and hp["cpu_threads"] > 0 and hp["save_every"] > 0 and
            all(isinstance(hp[k], (int, float)) and not isinstance(hp[k], bool) for k in ("family_loss_weight", "version_loss_weight")), "explicit integer sizes/seed/threads and numeric loss coefficients required")
        parent = path(job["parent"]["path"])
        need(sha(parent / "detector.safetensors") == job["parent"]["weights_sha256"] and
            sha(parent / "detector.json") == job["parent"]["detector_metadata_sha256"], "registered parent weights/metadata changed")
        meta = read(parent / "detector.json")
        need(meta["weights_sha256"] == job["parent"]["weights_sha256"] and read(parent / "encoder_config/config.json")["model_type"] == architecture, "wrong registered parent architecture")
        need(meta["architecture"]["head_hidden"] == 256 and meta["architecture"]["dropout"] == .1 and
            all(meta["architecture"][k] == hp[h] for k, h in (("max_length", "max_length"), ("stride", "stride"), ("attention_implementation", "attention"))), "parent architecture/window differs from the fixed trainer CLI")
        output_paths.extend([path(job["initializer"]), path(job["run_dir"]), *[path(job["snapshots"][str(e)]) for e in (1, 2)]])
        for target in output_paths:
            need(target != parent and not target.is_relative_to(parent) and not parent.is_relative_to(target), "outputs must not touch parent directories")
    need(len(set(output_paths)) == len(output_paths), "all output directories must be distinct")
    for i, target in enumerate(output_paths):
        for other in output_paths[i+1:]:
            need(not target.is_relative_to(other) and not other.is_relative_to(target), "output directories must not contain one another")
    for job in jobs:
        if initializers_exist:
            verify_initializer(job)
        else:
            need(not path(job["initializer"]).exists(), "initializer already exists; do not overwrite")
        need(not path(job["run_dir"]).exists() and all(not path(job["snapshots"][str(e)]).exists() and
            not path(job["snapshots"][str(e)]).with_name(path(job["snapshots"][str(e)]).name+".copying").exists() for e in (1, 2)), "run/snapshot directory already exists")
    need(path(training["python"]).is_file(), "registered Python executable missing")
    return protocol


def verify_initializer(job):
    initial, parent = path(job["initializer"]), path(job["parent"]["path"])
    manifest = read(initial / "manifest.json")
    expected = copy.deepcopy(read(parent / "detector.json"))
    expected["architecture"].update(family_loss_weight=0.0, version_loss_weight=0.0)
    need(read(initial / "detector.json") == expected, "initializer changed more than the two objective coefficients")
    actual = inventory(initial)
    saved = manifest["initializer"]["all_inference_files_sha256"]
    need(set(actual) == set(saved) | {"manifest.json"} and all(actual[k] == v for k, v in saved.items()) and
        actual["detector.safetensors"] == job["parent"]["weights_sha256"] and inventory(parent) == manifest["parent"]["all_original_files_sha256"], "initializer/parent copy hashes changed")
    need(manifest["parent"]["weights_sha256"] == job["parent"]["weights_sha256"] and
        manifest["parent"]["detector_metadata_sha256"] == job["parent"]["detector_metadata_sha256"] and
        manifest["only_two_objective_fields_changed"] is True and manifest["parent_all_files_unchanged"] is True, "initializer provenance differs")
    return {"manifest_sha256": actual["manifest.json"], "files_sha256": actual,
        "weights_sha256": actual["detector.safetensors"], "detector_metadata_sha256": actual["detector.json"]}


def argv(training, job):
    # Preserve the venv entry symlink. Resolving it to the base interpreter can
    # silently lose pyvenv.cfg and its installed packages.
    python = Path(training["python"])
    python = python.absolute() if python.is_absolute() else (ROOT/python).absolute()
    values = [str(python), "-B", "-m", "experiments.english_v1.neural", "train"]
    for role, flag in (("train", "--train"), ("validation", "--validation"), ("dev_challenge", "--dev-challenge")):
        values.extend([flag, str(path(training["data"][role]["path"]))])
    values.extend(["--init-checkpoint", str(path(job["initializer"])), "--output", str(path(job["run_dir"]))])
    for name in HP_FLAGS:
        values.extend(["--"+name.replace("_", "-"), str(job["hyperparameters"][name])])
    return values + ["--device", "mps"]


def prepare(protocol_path, expected_sha256):
    protocol = verified(protocol_path, expected_sha256)
    training = protocol["training"]
    work = path(training["work_dir"])
    work.mkdir(parents=True, exist_ok=False)
    sources = work / "source_snapshot"
    for name in training["code_sha256"]:
        target = sources / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path(name), target)
        need(sha(target) == training["code_sha256"][name], "execution source snapshot differs")
    from experiments.english_v2.prepare_binary_init import prepare as make_init
    initializers = {}
    for job in training["jobs"]:
        make_init(path(job["parent"]["path"]), path(job["initializer"]),
            expected_weights_sha256=job["parent"]["weights_sha256"], expected_metadata_sha256=job["parent"]["detector_metadata_sha256"])
        initializers[job["name"]] = verify_initializer(job)
    verified(protocol_path, expected_sha256, initializers_exist=True)
    receipt = {"prepared_at_utc": now(), "protocol_sha256": expected_sha256,
        "initializers": initializers, "source_snapshot_sha256": inventory(sources),
        "jobs": {job["name"]: {"argv": argv(training, job), "steps_per_epoch": math.ceil(training["data"]["train"]["rows"]/job["hyperparameters"]["batch_size"])} for job in training["jobs"]},
        "training_or_forward_performed": False, "calibration_final_opened": False}
    write_new(work / "prepared.json", receipt)
    return receipt


def complete_progress(destination):
    if not Path(destination).exists():
        return []
    value = Path(destination).read_text()
    if value and not value.endswith("\n"):
        value = value[:value.rfind("\n")+1] if "\n" in value else ""
    return [json.loads(line) for line in value.splitlines() if line.strip()]


def mps_pids():
    result = subprocess.run(["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True)
    found = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and int(fields[0]) != os.getpid() and re.search(r"--device(?:=|\s+)mps(?:\s|$)", fields[1]):
            found.append(int(fields[0]))
    return found


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)


def snapshot(job, epoch, event, steps, initializer):
    source = path(job["run_dir"]) / ("best" if epoch == 1 else "last")
    target = path(job["snapshots"][str(epoch)])
    temporary = target.with_name(target.name+".copying")
    need(not target.exists() and not temporary.exists(), "immutable epoch snapshot already exists")
    if epoch == 1:
        selection = read(source / "selection.json")
        need(selection["epoch"] == 1 and selection["step"] == steps, "missed/incorrect epoch1 snapshot; never substitute epoch2")
    else:
        state = read(source / "resume.json")
        need(state["epoch"] == 2 and state["global_step"] == 2*steps, "epoch2last checkpoint incomplete")
        selection = {"epoch": 2, "step": 2*steps, "metadata_origin": "epoch_complete event and last/resume.json", "development_event": event}
    meta = read(source / "detector.json")
    need(sha(source / "detector.safetensors") == meta["weights_sha256"] and
        meta["architecture"] == read(path(job["initializer"]) / "detector.json")["architecture"], "snapshot weight/config identity differs")
    for key in HP_FLAGS:
        need(meta["training_config"].get(key) == job["hyperparameters"][key], "snapshot registered training option changed")
    origin = meta["provenance"]["initial_checkpoint"]
    need(origin == {"path": str(path(job["initializer"])), "detector_weights_sha256": initializer["weights_sha256"],
        "detector_metadata_sha256": initializer["detector_metadata_sha256"]}, "snapshot does not come from the registered fresh warm start")
    temporary.mkdir(parents=True, exist_ok=False)
    copied = [source / "detector.safetensors", source / "detector.json"]
    copied += [p for name in ("encoder_config", "tokenizer") for p in sorted((source/name).rglob("*")) if p.is_file()]
    hashes = {p.relative_to(source).as_posix(): sha(p) for p in copied}
    for name in hashes:
        destination = temporary / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    if epoch == 1:
        hashes["selection.json"] = sha(source/"selection.json")
        shutil.copyfile(source/"selection.json", temporary/"selection.json")
    else:
        write_new(temporary/"selection.json", selection)
    need(all(sha(source/name) == digest == sha(temporary/name) for name, digest in hashes.items()), "source checkpoint changed while copying the epoch snapshot")
    files = inventory(temporary)
    temporary.rename(target)
    return {"epoch": epoch, "path": str(target), "weights_sha256": files["detector.safetensors"],
        "detector_metadata_sha256": files["detector.json"], "all_files_sha256": files,
        "source_copied_files_sha256": hashes, "source_bytes_unchanged_during_copy": True}


def run(protocol_path, expected_sha256, *, guard=mps_pids, poll_seconds=1.0):
    protocol = verified(protocol_path, expected_sha256, initializers_exist=True)
    training = protocol["training"]
    work = path(training["work_dir"])
    prepared = read(work / "prepared.json")
    need(prepared["protocol_sha256"] == expected_sha256 and prepared["source_snapshot_sha256"] == inventory(work/"source_snapshot") == training["code_sha256"], "prepared receipt/source snapshot differs")
    for job in training["jobs"]:
        need(prepared["initializers"][job["name"]] == verify_initializer(job) and
            prepared["jobs"][job["name"]]["argv"] == argv(training, job), "prepared initializer/registered command changed")
    need(not guard(), "another MPS job is active; coordinator does not launch")
    write_new(work / "STARTED.json", {"started_at_utc": now(), "coordinator_pid": os.getpid(), "protocol_sha256": expected_sha256,
        "prepared_sha256": sha(work/"prepared.json")})
    records = {}
    child = None
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupt(signum, frame):
        raise KeyboardInterrupt("coordinator interrupted")
    for sig in previous_handlers:
        signal.signal(sig, interrupt)
    try:
        for job in training["jobs"]:
            # Full data/code/parent hashes are checked between jobs, not once
            # per second while large weights are in active training.
            if records:
                for spec in [*training["data"].values(), training["data_qa"]]:
                    bound_file(spec)
                need(sha(path(protocol_path)) == expected_sha256 and not guard(), "protocol changed or previous GPU child still running")
                for name, expected in training["code_sha256"].items():
                    need(sha(path(name)) == expected, "execution source changed between jobs")
            initial = verify_initializer(job)
            need(not path(job["run_dir"]).exists() and all(not path(job["snapshots"][str(e)]).exists() and
                not path(job["snapshots"][str(e)]).with_name(path(job["snapshots"][str(e)]).name+".copying").exists() for e in (1, 2)),
                "another writer created a registered run/snapshot before launch")
            steps = prepared["jobs"][job["name"]]["steps_per_epoch"]
            log_path = work / (job["name"]+".log")
            env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
            with log_path.open("xb") as log:
                child = subprocess.Popen(argv(training, job), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write_new(work/(job["name"]+"_launch.json"), {"pid": child.pid, "coordinator_pid": os.getpid(), "started_at_utc": now(),
                    "argv": argv(training, job), "protocol_sha256": expected_sha256, "initializer": initial,
                    "data_qa_sha256": training["data_qa"]["sha256"], "data": training["data"], "code_sha256": training["code_sha256"]})
                snapshots, ready = {}, False
                last_emit = 0.0
                while True:
                    observed_code = child.poll()
                    rows = complete_progress(path(job["run_dir"])/"progress.jsonl")
                    for event in rows:
                        if event.get("phase") == "train_ready":
                            need(event.get("global_step") == 0 and event.get("optimizer_state_entries") == 0 and
                                event.get("initialization_mode") == "init_checkpoint_new_training" and
                                event.get("steps_per_epoch") == steps and event.get("total_steps") == 2*steps,
                                "child did not start with fresh optimizer/scheduler/global step0")
                            need(all(event["inputs"][role]["sha256"] == spec["sha256"] and event["inputs"][role]["rows"] == spec["rows"] for role, spec in training["data"].items()), "child actually opened different data")
                            ready = True
                        if event.get("phase") == "epoch_complete" and event["epoch"] not in snapshots:
                            epoch = event["epoch"]
                            need(ready and epoch in (1, 2) and (epoch == 1 or 1 in snapshots), "invalid epoch progression")
                            snapshots[epoch] = snapshot(job, epoch, event, steps, initial)
                            write_new(work/f"{job['name']}_epoch{epoch}.json", snapshots[epoch])
                            print(json.dumps({"job": job["name"], "epoch_snapshot": epoch, "weights_sha256": snapshots[epoch]["weights_sha256"]}), flush=True)
                    if rows and time.monotonic()-last_emit >= 45:
                        latest = rows[-1]
                        print(json.dumps({"job": job["name"], "utc": now(), **{k:latest[k] for k in ("phase", "epoch", "step", "elapsed_seconds", "loss") if k in latest}}), flush=True)
                        last_emit = time.monotonic()
                    if observed_code is not None:
                        break
                    time.sleep(poll_seconds)
                code = child.wait()
            need(code == 0 and ready and set(snapshots) == {1, 2} and rows and rows[-1].get("phase") == "training_complete" and rows[-1].get("step") == 2*steps,
                "child did not complete both registered epochs successfully")
            records[job["name"]] = {"returncode": code, "pid_exited": True, "snapshots": snapshots, "completed_at_utc": now()}
            write_new(work/(job["name"]+"_COMPLETED.json"), records[job["name"]])
            child = None
        # Validate original parents/init and inputs again without requiring the
        # now intentionally existing run/snapshot directories to be absent.
        for job in training["jobs"]:
            verify_initializer(job)
        for spec in [*training["data"].values(), training["data_qa"]]:
            bound_file(spec)
        need(sha(path(protocol_path)) == expected_sha256 and not guard(), "bound protocol changed or an MPS child remains")
        need(all(sha(path(name)) == value for name, value in training["code_sha256"].items()), "registered source changed during training")
        value = {"completed_at_utc": now(), "protocol_sha256": expected_sha256, "jobs": records,
            "all_children_exited": True, "GPU_released_by_process_exit": True, "calibration_final_opened": False,
            "selection_or_calibration_automatically_run": False}
        write_new(work/"COMPLETED.json", value)
        return value
    except BaseException as error:
        stop_child(child)
        write_new(work/"FAILED.json", {"failed_at_utc": now(), "error_type": type(error).__name__, "error": str(error),
            "child_running": child is not None and child.poll() is None, "completed_jobs": list(records), "later_jobs_will_not_start": True})
        raise
    finally:
        stop_child(child)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    value = prepare(args.protocol, args.sha256) if args.command == "prepare" else run(args.protocol, args.sha256)
    print(json.dumps({"operation": args.command, "completed": True, "training_launched": args.command == "run",
        "calibration_final_opened": False, "protocol_sha256": value["protocol_sha256"]}))


if __name__ == "__main__":
    main()
