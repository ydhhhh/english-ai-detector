"""CPU fake-process/metadata checks for the v2 development coordinator.

Every tensor file is deliberately unusable; raw scores are mathematical test
fixtures. No actual checkpoint, corpus, model forward, calibration or final is
opened. This proves orchestration and receipts, not detector quality.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from experiments.english_v1 import runtime  # noqa: F401
from experiments.english_v1.fingerprints import metadata_fingerprint
from experiments.english_v2 import run_development as rd, train_registered as tr
from experiments.english_v2.train_registered_checks import fixture as training_fixture, spec


def write(p, value):
    p.write_text(json.dumps(value, indent=2)+"\n")


def fixture(folder, *, failure=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pp, protocol = training_fixture(folder)
    # The training fixture itself has no text. Add only binary metadata.
    for item in protocol["training"]["data"].values():
        destination = Path(item["path"])
        rows = pq.read_table(destination).to_pylist()
        for i, row in enumerate(rows):
            row["y_ai"] = i%2
        pq.write_table(pa.Table.from_pylist(rows), destination)
        item.update(spec(destination, len(rows)))
    qa = Path(protocol["training"]["data_qa"]["path"])
    write(qa, {"passed": True, "training_inputs_sha256": {role: item["sha256"] for role, item in protocol["training"]["data"].items()}})
    protocol["training"]["data_qa"] = spec(qa)
    cohorts = {}
    for cohort in rd.selector.COHORTS:
        rows = [{"id": f"constructed:{cohort}:row:{i}", "group_id": f"constructed:{cohort}:parent:{i//4}",
            "y_ai": i%2, "split": "dev", "eligible": True} for i in range(512)]
        destination = folder/(cohort+"_dev.parquet")
        pq.write_table(pa.Table.from_pylist(rows), destination)
        cohorts[cohort] = spec(destination, len(rows))
    source = folder/"source_minilm"
    import shutil
    shutil.copytree(protocol["training"]["jobs"][0]["parent"]["path"], source)
    stat = folder/"source_stat.joblib"; stat.write_bytes(b"FAKE_NOT_A_USABLE_STATISTICAL_MODEL")
    sources = {rd.SOURCE_NAME: {"type": "neural", "checkpoint": str(source), "weights_sha256": tr.sha(source/"detector.safetensors"),
        "metadata_sha256": metadata_fingerprint("neural", source)}, rd.STAT_NAME: {"type": "statistical", "checkpoint": str(stat),
        "weights_sha256": tr.sha(stat), "metadata_sha256": tr.sha(stat)}}
    versions = folder/"source_version_selection.json"; write(versions, {"enabled": True, "synthetic_inherited_only": True})
    protocol.update(candidate_names=list(rd.NAMES), binary_selector=copy.deepcopy(rd.selector.SELECTOR), source_mode="rankings_only",
        source_calibration_sha256=rd.SOURCE_CALIBRATION_SHA256, language_rule_sha256=tr.sha(rd.ROOT/"experiments/english_v1/language_guard.py"))
    protocol["development"] = {"output_dir": str(folder/"development_output"), "cohorts": cohorts,
        "candidate_checkpoints": {name: {"job": job, "epoch": epoch, "path": next(j for j in protocol["training"]["jobs"] if j["name"] == job)["snapshots"][str(epoch)]}
            for name, (job, epoch) in zip(rd.NAMES, rd.CHECKPOINTS)}, "fixed_source_components": sources,
        "inherited_version_selection": spec(versions), "code_sha256": {name: rd.sha(rd.ROOT/name) for name in rd.REQUIRED_CODE}}
    protocol["synthetic_export_failure"] = failure
    # Wrap the existing fake trainer and a separate fake exporter in the exact
    # registered Python entry point. Real subprocesses still receive real argv.
    executable = Path(protocol["training"]["python"])
    trainer = folder/"fake_training_child.py"; trainer.write_bytes(executable.read_bytes())
    exporter = folder/"fake_export_child.py"
    exporter.write_text('''import hashlib,json,pathlib,sys
sys.path.insert(0, PROJECT)
from experiments.english_v1 import runtime
from experiments.english_v1.fingerprints import metadata_fingerprint
import numpy as np
import pyarrow.parquet as pq
P=pathlib.Path(PROTOCOL);v=json.loads(P.read_text());args=sys.argv[1:]
flags={args[i]:args[i+1] for i in range(len(args)-1) if args[i].startswith("--")}
assert "experiments.english_v1.export_component" in args
assert flags["--purpose"]=="development" and flags["--device"]=="mps" and flags["--batch-size"]=="16"
target=pathlib.Path(flags["--output"]);cp=pathlib.Path(flags["--checkpoint"]);source=pathlib.Path(flags["--input"])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
trace=P.parent/"export_order.jsonl"
def event(state):
 with trace.open("a") as f:f.write(json.dumps({"state":state,"output":str(target),"pid":__import__("os").getpid()})+"\\n")
event("start")
if v["synthetic_export_failure"]=="child":sys.exit(13)
rows=pq.read_table(source,columns=["id","group_id","y_ai"]).to_pylist();n=len(rows)
which=next(i for i,name in enumerate(v["candidate_names"]) if pathlib.Path(v["development"]["candidate_checkpoints"][name]["path"]).resolve()==cp.resolve())
y=np.asarray([r["y_ai"] for r in rows],dtype=np.int64);rng=np.random.default_rng(42+which)
margin=(2*y-1)*(1.0+which*.2)+rng.normal(size=n)
lengths=np.full(n,200,dtype=np.int64);lengths[-32:]=64
support=np.ones(n,dtype=bool);support[::19]=False
weights=sha(cp/"detector.safetensors");meta=metadata_fingerprint("neural",cp);guard=v["language_rule_sha256"]
raw={"ids":np.asarray([r["id"] for r in rows]),"group_ids":np.asarray([r["group_id"] for r in rows]),"y_ai":y,
 "lengths":lengths,"scope_word_like_tokens":lengths.copy(),"language_supported":support,"language_probability":np.full(n,.99),
 "ascii_letter_ratio":np.ones(n),"language_rule_sha256":np.asarray(guard),"calibrated":np.asarray(False),
 "export_purpose":np.asarray("development"),"raw_binary":np.column_stack([-margin/2,margin/2]),
 "component_weights_sha256":np.asarray(weights),"component_metadata_sha256":np.asarray(meta)}
target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**raw)
receipt={"purpose":"development","type":"neural","checkpoint":str(cp.resolve()),"weights_sha256":weights,"component_metadata_sha256":meta,
 "input_sha256":sha(source),"original_rows":n,"eligible_export_rows":n,"excluded_ineligible_rows":0,"output_sha256":sha(target),"calibration_fitted":False,
 "frozen_blend_sha256":None,"frozen_calibration_sha256":None,"code_sha256":{name:v["development"]["code_sha256"]["experiments/english_v1/"+name] for name in EXPORT_FILES},
 "deployment_scope":{"rule_sha256":guard,"supported_rows":int(support.sum())}}
if v["synthetic_export_failure"]=="receipt":receipt["input_sha256"]="0"*64
target.with_suffix(".provenance.json").write_text(json.dumps(receipt)+"\\n");event("end")
'''.replace("PROJECT", repr(str(rd.ROOT))).replace("PROTOCOL", repr(str(pp))).replace("EXPORT_FILES", repr(rd.EXPORT_FILES)))
    executable.write_text("#!"+sys.executable+"\nimport runpy,sys\nrunpy.run_path("+repr(str(exporter))+" if 'experiments.english_v1.export_component' in sys.argv else "+repr(str(trainer))+",run_name='__main__')\n")
    executable.chmod(0o700)
    write(pp, protocol)
    tr.prepare(pp, tr.sha(pp))
    tr.run(pp, tr.sha(pp), guard=lambda: [], poll_seconds=.01)
    return pp, protocol, {name: s["weights_sha256"] for name, s in sources.items()}


def rejected(call):
    try:
        call()
    except (ValueError, KeyError, FileNotFoundError):
        return
    raise AssertionError("expected fail-closed development guard")


def run():
    checks = []
    with tempfile.TemporaryDirectory(prefix="synthetic-v2-development-") as temporary:
        root = Path(temporary)
        pp, protocol, fake_sources = fixture(root/"success")
        kwargs = {"guard": lambda: [], "exited": lambda pid: False, "poll_seconds": .01}
        with patch.object(rd, "SOURCE_WEIGHTS", fake_sources):
            rejected(lambda: rd.preflight(pp, "0"*64, exited=lambda pid: False))
            need_out = Path(protocol["development"]["output_dir"])
            assert not need_out.exists()
            evidence = rd.preflight(pp, rd.sha(pp), exited=lambda pid: False)
            assert set(evidence["components"]) == set(rd.NAMES) | set(fake_sources)
            item = protocol["development"]["cohorts"][rd.selector.COHORTS[0]]
            for changed in ({key: value for key, value in item.items() if key != "bytes"}, {**item, "rows": True}):
                rejected(lambda: rd.metadata_rows(changed, allowed_roles=rd.ROLES))
            checks.append("Frozen protocol, committed epochs, actual files/configuration and six metadata cohorts verified without model loading")
            completed = Path(protocol["training"]["work_dir"])/"COMPLETED.json"
            original_complete = completed.read_bytes()
            value = json.loads(original_complete);value["all_children_exited"] = False;write(completed, value)
            rejected(lambda: rd.preflight(pp, rd.sha(pp), exited=lambda pid: False))
            completed.write_bytes(original_complete)
            first = protocol["development"]["cohorts"][rd.selector.COHORTS[0]]
            original_data = Path(first["path"]).read_bytes()
            Path(first["path"]).write_bytes(original_data+b"tamper")
            rejected(lambda: rd.preflight(pp, rd.sha(pp), exited=lambda pid: False))
            Path(first["path"]).write_bytes(original_data)
            checks.append("Unfinished training and changed development input reject before an output or export")
            result = rd.run(pp, rd.sha(pp), **kwargs)
            assert result["all_24_exports_complete"] and result["no_exports_running"]
            out = Path(protocol["development"]["output_dir"])
            trace = [json.loads(line) for line in (pp.parent/"export_order.jsonl").read_text().splitlines()]
            assert len(trace) == 48 and all(trace[i]["state"] == "start" and trace[i+1]["state"] == "end" and
                trace[i]["pid"] == trace[i+1]["pid"] for i in range(0, 48, 2))
            assert all(not rd.pid_exists(row["pid"]) for row in trace)
            selection, blend, freeze = read(out/"binary_selection.json"), read(out/"binary_blend.json"), read(out/"freeze.json")
            assert all(selection[key] == blend[key] for key in selection if key != "components")
            assert set(blend["components"]) == set(rd.NAMES) | set(fake_sources)
            assert freeze["attribution_model"] == rd.SOURCE_NAME and freeze["artifacts"]["version_selection"] == evidence["inherited_version_selection"]
            assert set(freeze["required_components"]) == {name for name, weight in zip(blend["names"], blend["selected"]["weights"]) if weight != 0} | set(fake_sources)
            assert read(out/"COMPLETED.json")["freeze_sha256"] == rd.sha(out/"freeze.json")
            assert not (out/"FAILED.json").exists() and not any("calibration" in p.name or "final" in p.name for p in (out/"raw").rglob("*.npz"))
            # The unchanged release-export preparation accepts this freeze shape.
            from experiments.english_v1.run_release_exports import prepare
            inert = pp.parent/"inert_future_calibration.json";write(inert,{"synthetic_only": True})
            plan_path = pp.parent/"synthetic_future_data_plan.json"
            write(plan_path, {"calibration": {"input": str(inert), "input_sha256": rd.sha(inert), "manifest": str(inert),
                "manifest_sha256": rd.sha(inert), "rows": 2}})
            prepared = prepare(out, plan_path, "calibration", pp.parent/"future_unlaunched_exports")
            assert set(prepared["required_components"]) == set(freeze["required_components"])
            assert not (pp.parent/"future_unlaunched_exports").exists()
            checks.append("Exactly24 FP32-wrapper argv fake exports run serially; unsupported rows retained; all children exit")
            checks.append("Only binary selected; source identity prebound without changing weights/means; immutable freeze compatible with v1 release exporter")
            rejected(lambda: rd.run(pp, rd.sha(pp), **kwargs))
            checks.append("Existing output/selection cannot be overwritten or silently resumed")
        for failure in ("child", "receipt"):
            pp, protocol, fake_sources = fixture(root/("failure_"+failure), failure=failure)
            with patch.object(rd, "SOURCE_WEIGHTS", fake_sources):
                rejected(lambda: rd.run(pp, rd.sha(pp), **kwargs))
            out = Path(protocol["development"]["output_dir"])
            assert (out/"FAILED.json").is_file() and not (out/"freeze.json").exists() and not (out/"binary_selection.json").exists()
            trace = [json.loads(line) for line in (pp.parent/"export_order.jsonl").read_text().splitlines()]
            assert len({row["pid"] for row in trace}) == 1 and all(not rd.pid_exists(row["pid"]) for row in trace)
        checks.append("Failed child or wrong raw receipt stops later exports and selection, marks failure, leaves no child process")
    return {"passed": True, "fake_process_and_synthetic_metadata_scores_only": True,
        "actual_checkpoints_data_or_models_opened": False, "real_calibration_final_or_scores_opened": False, "GPU_used": False,
        "check_count": len(checks), "checks": checks, "runner_sha256": rd.sha(rd.__file__), "checks_sha256": rd.sha(__file__)}


def read(p):
    return json.loads(Path(p).read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tr.write_new(args.output, result)
    print(json.dumps(result, indent=2))
