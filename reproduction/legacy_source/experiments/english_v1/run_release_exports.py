"""Serialize selected-component calibration/final exports after development freeze.

This orchestrates existing guarded inference only. It never selects a model,
fits calibration, changes thresholds or evaluates final performance. Launch it
only after the independent development/MPS queue has finished.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import subprocess
import sys

from experiments.english_v1.export_component import sha
from experiments.english_v1.fingerprints import metadata_fingerprint

PROJECT=Path(__file__).resolve().parents[2]


def resolve(path):
    path=Path(path)
    return path if path.is_absolute() else PROJECT/path


def read_bound(item):
    path=resolve(item["path"])
    if sha(path)!=item["sha256"]:raise ValueError(f"frozen artifact changed: {path}")
    return path,json.loads(path.read_text())


def write_atomic(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+".part")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temporary.replace(path)


def prepare(selection_dir,plan_path,phase,output,calibration=None):
    selection_dir=resolve(selection_dir)
    freeze_path=selection_dir/"freeze.json"; freeze=json.loads(freeze_path.read_text())
    seal=json.loads((selection_dir/"COMPLETED.json").read_text())
    if (selection_dir/"FAILED.json").exists() or seal.get("freeze_sha256")!=sha(freeze_path) or freeze.get("status")!="development_selection_frozen":
        raise ValueError("a completed, matching development-selection seal is required")
    if freeze.get("calibration_opened") is not False or freeze.get("final_test_opened") is not False:
        raise ValueError("development selection must precede any calibration or final scores")
    artifacts=freeze["artifacts"]
    bound={name:read_bound(item) for name,item in artifacts.items()}
    blend_path,blend=bound["binary_blend"]
    _,versions=bound["version_selection"]
    attribution=freeze["attribution_model"]
    needed={name for name,w in zip(blend["names"],blend["selected"]["weights"]) if w!=0}|{attribution}
    if versions.get("enabled"):needed.add("statistical_v2")
    if set(freeze["required_components"])!=needed:raise ValueError("release component set differs from frozen selection")
    components={}
    for name in sorted(needed):
        if not name.replace("_","").isalnum():raise ValueError("simple component name required")
        spec=freeze["components"][name];kind=spec["type"];checkpoint=resolve(spec["checkpoint"])
        if kind not in {"neural","statistical"}:raise ValueError("unsupported component kind")
        weight=checkpoint/"detector.safetensors" if kind=="neural" else checkpoint
        identity={"weights_sha256":sha(weight),"metadata_sha256":metadata_fingerprint(kind,checkpoint)}
        if any(spec[k]!=v for k,v in identity.items()):raise ValueError("selected component changed: "+name)
        if name in blend["components"] and blend["components"][name]!=identity:raise ValueError("blend component identity mismatch: "+name)
        components[name]={"type":kind,"checkpoint":str(checkpoint),**identity}
    plan_path=resolve(plan_path);plan=json.loads(plan_path.read_text());cal_path=None
    if phase=="final":
        if calibration is None:raise ValueError("final exports require frozen calibration")
        cal_path=resolve(calibration);cal=json.loads(cal_path.read_text())
        if cal["provenance"]["blend_sha256"]!=sha(blend_path) or cal["provenance"]["attribution_model"]!=attribution:
            raise ValueError("calibration belongs to another frozen selection")
        if cal["provenance"].get("language_rule_sha256")!=sha(Path(__file__).with_name("language_guard.py")):
            raise ValueError("calibration language rule differs from current deployment")
        if cal["provenance"].get("deployment_decision_protocol_sha256")!=sha(PROJECT/"research/english_v1_data/deployment_decision_amendment_07.json"):
            raise ValueError("calibration decision protocol differs from current deployment")
        if any(not math.isfinite(float(cal["thresholds"][key]["threshold"])) or float(cal["thresholds"][key]["threshold"])<=.5 for key in ("AI","human")):
            raise ValueError("frozen directional thresholds must require strict majority")
        for name,spec in components.items():
            if any(cal["provenance"]["component_"+key].get(name)!=spec[key] for key in ("weights_sha256","metadata_sha256")):
                raise ValueError("calibration component identity mismatch: "+name)
        datasets=plan["final_datasets"]
    else:
        if calibration is not None:raise ValueError("do not supply calibration to raw calibration export")
        datasets=[{"name":"combined_calibration",**plan["calibration"]}]
    tasks=[]
    for item in datasets:
        source=resolve(item["input"]);manifest=resolve(item["manifest"])
        if sha(manifest)!=item["manifest_sha256"] or sha(source)!=item["input_sha256"]:
            raise ValueError("fixed data/manifest identity changed: "+item["name"])
        for name,spec in components.items():
            filename="calibration_combined.npz" if phase=="calibration" else name+".npz"
            target=resolve(output)/(name if phase=="calibration" else item["name"])/filename
            tasks.append({"name":name,"dataset":item["name"],"input":str(source),"input_sha256":item["input_sha256"],
                "expected_rows":item.get("expected_export_rows",item.get("rows")),"output":str(target),**spec})
    return {"schema_version":1,"phase":phase,"selection_freeze_sha256":sha(freeze_path),
        "data_plan_sha256":sha(plan_path),"blend":str(blend_path),"blend_sha256":sha(blend_path),
        "calibration":str(cal_path) if cal_path else None,"calibration_sha256":sha(cal_path) if cal_path else None,
        "attribution_model":attribution,"required_components":sorted(needed),"tasks":tasks,
        "scope":"Serial raw inference export only; no model fitting, calibration fitting or final metric evaluation."}


def complete(task,plan):
    output=Path(task["output"]);side=output.with_suffix(".provenance.json")
    if not output.exists() and not side.exists():return False
    if not output.is_file() or not side.is_file():raise ValueError("incomplete prior export requires review: "+str(output))
    value=json.loads(side.read_text())
    checks={"purpose":plan["phase"],"weights_sha256":task["weights_sha256"],
        "component_metadata_sha256":task["metadata_sha256"],"input_sha256":task["input_sha256"],
        "eligible_export_rows":task["expected_rows"],"output_sha256":sha(output),"calibration_fitted":False,
        "frozen_blend_sha256":plan["blend_sha256"] if plan["phase"]=="final" else None,
        "frozen_calibration_sha256":plan["calibration_sha256"]}
    current_code={name:sha(Path(__file__).with_name(name)) for name in ("export_component.py","neural.py","statistical.py","catalog.py","language_guard.py")}
    checks["code_sha256"]=current_code
    if (value.get("deployment_scope") or {}).get("rule_sha256")!=current_code["language_guard.py"]:
        raise ValueError("prior export used a different deployment language rule")
    if any(value.get(key)!=expected for key,expected in checks.items()):
        raise ValueError("prior export differs from this frozen job: "+str(output))
    return True


def run(args):
    plan=prepare(args.selection_dir,args.data_plan,args.phase,args.output,args.calibration)
    directory=resolve(args.output);directory.mkdir(parents=True,exist_ok=True)
    pp=directory/"export_plan.json"
    if pp.exists() and json.loads(pp.read_text())!=plan:raise ValueError("output directory belongs to another export plan")
    if not pp.exists():
        with pp.open("x") as f:f.write(json.dumps(plan,indent=2)+"\n")
    if args.preflight_only:return {"prepared":True,"tasks":len(plan["tasks"]),"exports_started":False}
    progress={"started_at_utc":datetime.now(timezone.utc).isoformat(),"plan_sha256":sha(pp),"completed":[]}
    for index,task in enumerate(plan["tasks"]):
        target=Path(task["output"]);target.parent.mkdir(parents=True,exist_ok=True)
        reused=complete(task,plan)
        if not reused:
            argv=[sys.executable,"-B","-m","experiments.english_v1.export_component","--type",task["type"],
                "--checkpoint",task["checkpoint"],"--input",task["input"],"--output",task["output"],
                "--purpose",args.phase,"--device",args.device,"--batch-size",str(args.batch_size)]
            if args.phase=="final":argv.extend(["--frozen-blend",plan["blend"],"--frozen-calibration",plan["calibration"]])
            with target.with_suffix(".log").open("wb") as log:subprocess.run(argv,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if not complete(task,plan):raise AssertionError("export did not produce a verified complete result")
        progress["completed"].append({"dataset":task["dataset"],"component":task["name"],"reused":reused,
            "output_sha256":sha(target),"sidecar_sha256":sha(target.with_suffix(".provenance.json"))})
        write_atomic(directory/"export_progress.json",progress)
        print(json.dumps({"completed":index+1,"total":len(plan["tasks"]),"dataset":task["dataset"],"component":task["name"]}),flush=True)
    progress["completed_at_utc"]=datetime.now(timezone.utc).isoformat();write_atomic(directory/"export_progress.json",progress)
    return {"completed":True,"tasks":len(plan["tasks"])}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("calibration","final"),required=True)
    p.add_argument("--selection-dir",default="results/english_v1/selection_frozen_v1")
    p.add_argument("--data-plan",default="research/english_v1_finalization_plan.json")
    p.add_argument("--output",required=True);p.add_argument("--calibration")
    p.add_argument("--device",choices=("cpu","mps"),default="mps");p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--preflight-only",action="store_true")
    args=p.parse_args()
    if args.batch_size<1:p.error("positive batch size required")
    print(json.dumps(run(args),indent=2))
