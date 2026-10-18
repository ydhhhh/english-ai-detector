"""Fake-executable/metadata checks; never load real weights or run a forward.

All Parquet rows and parent files live in TemporaryDirectory. Fake tensor files
are not a usable model. The CPU-only fake child receives the actual generated
argv and emits the frozen trainer's metadata/progress/checkpoint schema.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from experiments.english_v1 import runtime  # noqa: F401
from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS, MODEL_FAMILIES
from experiments.english_v2 import train_registered as tr


def spec(path, rows=None):
    value = {"path": str(path), "bytes": path.stat().st_size, "sha256": tr.sha(path)}
    if rows is not None:
        value["rows"] = rows
    return value


def fixture(folder, *, scenario="success"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    folder.mkdir()
    data = {}
    for role, split in (("train", "train"), ("validation", "validation"), ("dev_challenge", "dev_challenge")):
        destination = folder / (role+".parquet")
        pq.write_table(pa.Table.from_pylist([{"id": f"synthetic:{role}:{i}", "group_id": f"synthetic:parent:{role}:{i}", "split": split} for i in range(64)]), destination)
        data[role] = spec(destination, 64)
    qa = folder / "qa.json"
    tr.write_new(qa, {"passed": True, "training_inputs_sha256": {role: value["sha256"] for role, value in data.items()}, "synthetic_only": True})
    jobs = []
    for name, model_type, batch, precision in (("Mini4", "bert", 32, "fp32"), ("Modern3", "modernbert", 16, "bf16")):
        parent = folder / (name+"_fake_parent")
        (parent/"encoder_config").mkdir(parents=True)
        (parent/"tokenizer").mkdir()
        (parent/"detector.safetensors").write_bytes(b"SYNTHETIC_ONLY_NOT_A_USABLE_MODEL:"+name.encode())
        tr.write_new(parent/"encoder_config/config.json", {"model_type": model_type})
        tr.write_new(parent/"tokenizer/tokenizer.json", {"synthetic_only": True})
        tr.write_new(parent/"detector.json", {"weights_sha256": tr.sha(parent/"detector.safetensors"),
            "family_labels": FAMILIES, "model_labels": MODEL_LABELS, "model_families": MODEL_FAMILIES,
            "architecture": {"head_hidden": 256, "dropout": .1, "max_length": 512, "stride": 448,
                "family_loss_weight": .3, "version_loss_weight": .3, "attention_implementation": "sdpa"}})
        hp = {"epochs": 2, "batch_size": batch, "precision": precision, "lr": 5e-6,
            "weight_mode": "class_domain_v3" if name == "Mini4" else "class_normalized_v2", "family_loss_weight": 0., "version_loss_weight": 0.,
            "max_length": 512, "stride": 448, "attention": "sdpa", "seed": 20261017,
            "weight_decay": .01, "warmup_ratio": .06, "clip_grad": 1., "cpu_threads": 6, "save_every": 1000}
        jobs.append({"name": name, "parent": {"path": str(parent), "weights_sha256": tr.sha(parent/"detector.safetensors"),
            "detector_metadata_sha256": tr.sha(parent/"detector.json")}, "initializer": str(folder/(name+"_init")),
            "run_dir": str(folder/(name+"_run")), "snapshots": {str(e): str(folder/f"{name}_epoch{e}") for e in (1, 2)}, "hyperparameters": hp})
    protocol_path = folder/"protocol.json"
    executable = folder/"fake_python"
    child_source = '''#!INTERPRETER
import hashlib,json,pathlib,shutil,sys,time
P=pathlib.Path(PROTOCOL)
v=json.loads(P.read_text()); training=v["training"]
args=sys.argv[1:]; flags={args[i]:args[i+1] for i in range(len(args)-1) if args[i].startswith("--")}
job=next(j for j in training["jobs"] if pathlib.Path(j["run_dir"]).resolve()==pathlib.Path(flags["--output"]).resolve())
trace=P.parent/"order.jsonl"
def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value)+"\\n")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(value):
 with trace.open("a") as f:f.write(json.dumps(value)+"\\n")
record({"name":job["name"],"state":"start","pid":__import__("os").getpid()})
run=pathlib.Path(job["run_dir"]);run.mkdir(); progress=run/"progress.jsonl"
def event(value):
 with progress.open("a") as f:f.write(json.dumps(value)+"\\n");f.flush()
initial=pathlib.Path(flags["--init-checkpoint"]); parent=json.loads((initial/"detector.json").read_text())
steps=(training["data"]["train"]["rows"]+job["hyperparameters"]["batch_size"]-1)//job["hyperparameters"]["batch_size"]
event({"phase":"train_ready","global_step":0,"optimizer_state_entries":0,"initialization_mode":"init_checkpoint_new_training", "steps_per_epoch":steps,"total_steps":2*steps,
 "inputs":{k:{"sha256":s["sha256"],"rows":s["rows"]} for k,s in training["data"].items()}})
if v["synthetic_scenario"]=="fail_Mini4" and job["name"]=="Mini4":sys.exit(13)
if v["synthetic_scenario"]=="bad_progress" and job["name"]=="Mini4":
 with progress.open("a") as f:f.write("{bad completed json}\\n");f.flush()
 time.sleep(60);sys.exit(19)
def checkpoint(destination,epoch):
 destination.mkdir(exist_ok=True)
 for name in ("encoder_config","tokenizer"):
  if (destination/name).exists():shutil.rmtree(destination/name)
  shutil.copytree(initial/name,destination/name)
 (destination/"detector.safetensors").write_bytes((initial/"detector.safetensors").read_bytes()+f"|synthetic_epoch{epoch}".encode())
 meta=json.loads(json.dumps(parent));meta["weights_sha256"]=sha(destination/"detector.safetensors")
 meta["training_config"]=job["hyperparameters"]
 meta["provenance"]={"initial_checkpoint":{"path":str(initial),"detector_weights_sha256":sha(initial/"detector.safetensors"),"detector_metadata_sha256":sha(initial/"detector.json")}}
 write(destination/"detector.json",meta)
checkpoint(run/"best",1);write(run/"best/selection.json",{"epoch":1,"step":steps})
event({"phase":"epoch_complete","epoch":1,"selection_loss":0.,"development":{"synthetic_only":True}})
time.sleep(.25)
checkpoint(run/"last",2);write(run/"last/resume.json",{"epoch":2,"global_step":2*steps})
with progress.open("a") as f:f.write("{\\"phase\\": ");f.flush()
time.sleep(.1)
with progress.open("a") as f:f.write("\\"epoch_complete\\", \\"epoch\\": 2, \\"development\\": {\\"synthetic_only\\": true}}\\n");f.flush()
event({"phase":"training_complete","step":2*steps})
record({"name":job["name"],"state":"end"})
'''.replace("INTERPRETER", sys.executable).replace("PROTOCOL", repr(str(protocol_path)))
    executable.write_text(child_source)
    executable.chmod(0o700)
    protocol = {"schema_version": 1, "status": tr.STATUS, "synthetic_scenario": scenario,
        "training": {"work_dir": str(folder/"audit"), "python": str(executable), "data": data,
            "data_qa": spec(qa), "code_sha256": {name: tr.sha(tr.ROOT/name) for name in tr.REQUIRED_CODE}, "jobs": jobs}}
    tr.write_new(protocol_path, protocol)
    return protocol_path, protocol


def rejected(call):
    try:
        call()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("unsafe registration should be rejected")


def run():
    original = {name: tr.sha(tr.ROOT/name) for name in tr.REQUIRED_CODE}
    checks = []
    with tempfile.TemporaryDirectory(prefix="synthetic-v2-coordinator-") as root:
        root = Path(root)
        p, protocol = fixture(root/"main")
        draft = copy.deepcopy(protocol)
        draft["status"] = "DRAFT"
        d = root/"draft.json"; tr.write_new(d, draft)
        rejected(lambda: tr.prepare(d, tr.sha(d)))
        rejected(lambda: tr.prepare(p, "0"*64))
        assert not Path(protocol["training"]["work_dir"]).exists()
        checks.append("No frozen status/exact expected protocol SHA: no initializer or audit created")
        wrong = copy.deepcopy(protocol)
        wrong["training"]["data"]["train"]["bytes"] += 1
        d = root/"wrong_size.json"; tr.write_new(d, wrong)
        rejected(lambda: tr.prepare(d, tr.sha(d)))
        checks.append("Registered file size/SHA fail before preparation")
        for i, wrong_mode in ((0, "class_normalized_v2"), (1, "class_domain_v3")):
            wrong = copy.deepcopy(protocol)
            wrong["training"]["jobs"][i]["hyperparameters"]["weight_mode"] = wrong_mode
            d = root/f"wrong_weight_mode_{i}.json"; tr.write_new(d, wrong)
            rejected(lambda: tr.prepare(d, tr.sha(d)))
        assert not Path(protocol["training"]["work_dir"]).exists()
        checks.append("Mini4 class_domain_v3 and Modern3 class_normalized_v2 are exact per-job constraints, not interchangeable")
        prepared = tr.prepare(p, tr.sha(p))
        assert prepared["training_or_forward_performed"] is False
        rejected(lambda: tr.prepare(p, tr.sha(p)))
        result = tr.run(p, tr.sha(p), guard=lambda: [], poll_seconds=.02)
        assert result["all_children_exited"] is True and list(result["jobs"]) == ["Mini4", "Modern3"]
        order = [json.loads(line) for line in (p.parent/"order.jsonl").read_text().splitlines()]
        assert [(r["name"],r["state"]) for r in order] == [("Mini4","start"),("Mini4","end"),("Modern3","start"),("Modern3","end")]
        for job in protocol["training"]["jobs"]:
            for e in (1, 2):
                record = result["jobs"][job["name"]]["snapshots"][e]
                assert tr.sha(Path(job["snapshots"][str(e)])/"detector.safetensors") == record["weights_sha256"]
                assert tr.sha(Path(job["snapshots"][str(e)])/"detector.json") == record["detector_metadata_sha256"]
            assert tr.verify_initializer(job) == prepared["initializers"][job["name"]]
        checks.append("Fake executable receives exact argv; Mini4 exits before Modern3; fresh step0; two immutable snapshots each")
        rejected(lambda: tr.run(p, tr.sha(p), guard=lambda: []))
        checks.append("Existing initializers/runs/snapshots cannot be overwritten or implicitly resumed")
        for case in ("fail_Mini4", "bad_progress"):
            p2, proto2 = fixture(root/case, scenario=case)
            tr.prepare(p2, tr.sha(p2))
            try:
                tr.run(p2, tr.sha(p2), guard=lambda: [], poll_seconds=.02)
            except (ValueError, json.JSONDecodeError):
                pass
            else:
                raise AssertionError("failed fake child must stop queue")
            failed = tr.read(Path(proto2["training"]["work_dir"])/"FAILED.json")
            launch = tr.read(Path(proto2["training"]["work_dir"])/"Mini4_launch.json")
            assert failed["child_running"] is False and not Path(proto2["training"]["jobs"][1]["run_dir"]).exists()
            try:os.kill(launch["pid"],0)
            except ProcessLookupError:pass
            else:raise AssertionError("fake child orphaned")
        checks.append("Child failure and completed malformed JSON both stop queue, terminate active child, and never start Modern3")
        half = root/"partial.jsonl"
        half.write_text('{"phase":"train"}\n{"unfinished":')
        assert tr.complete_progress(half) == [{"phase":"train"}]
        checks.append("Only final unfinished JSON line ignored; complete malformed rows fail")
    assert {name:tr.sha(tr.ROOT/name) for name in tr.REQUIRED_CODE} == original
    return {"passed": True, "fake_executable_and_synthetic_metadata_only": True, "real_parent_weights_or_train_dev_read": False,
        "real_model_forward_training_or_GPU_used": False, "v1_frozen_sources_unchanged": True,
        "check_count": len(checks), "checks": checks, "coordinator_sha256": tr.sha(Path(tr.__file__)), "checks_sha256": tr.sha(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tr.write_new(args.output, value)
    print(json.dumps(value, indent=2))
