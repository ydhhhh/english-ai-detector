"""Export raw component scores with input/checkpoint/code fingerprints.

Inference export does not fit weights or thresholds. Opening final input labels
requires already frozen blend and calibration artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from experiments.english_v1 import runtime  # noqa: F401
from experiments.english_v1.fingerprints import metadata_fingerprint


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--type",choices=("neural","statistical"),required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--purpose",choices=("development","calibration","final"),required=True)
    p.add_argument("--frozen-blend",type=Path);p.add_argument("--frozen-calibration",type=Path)
    p.add_argument("--device",default="cpu");p.add_argument("--batch-size",type=int,default=32)
    a=p.parse_args();stem=a.input.stem.casefold()
    if stem=="historical_regression":p.error("full historical corpus now includes training data; use the explicitly separated historical_dev view")
    finalish="test" in stem or "ood" in stem or ("challenge" in stem and not stem.startswith("dev_")) or "final_" in stem or "sealed" in stem
    if finalish and a.purpose!="final":p.error("final/challenge input requires --purpose final")
    if a.purpose=="development" and "calibration" in stem:p.error("calibration input is not development")
    if a.purpose=="final":
        if not a.frozen_blend or not a.frozen_calibration:p.error("final export requires frozen blend and calibration")
        cal=json.loads(a.frozen_calibration.read_text())
        if cal["provenance"]["blend_sha256"]!=sha(a.frozen_blend):raise ValueError("frozen blend/calibration mismatch")
        if cal["provenance"].get("language_rule_sha256")!=sha(Path(__file__).with_name("language_guard.py")):
            raise ValueError("frozen calibration differs from the current deployment language rule")
        if any(not math.isfinite(float(cal["thresholds"][key]["threshold"])) or float(cal["thresholds"][key]["threshold"])<=.5 for key in ("AI","human")):
            raise ValueError("final export requires the frozen strict-majority decision policy")
        decision_protocol=Path(__file__).resolve().parents[2]/"research/english_v1_data/deployment_decision_amendment_07.json"
        if cal["provenance"].get("deployment_decision_protocol_sha256")!=sha(decision_protocol):
            raise ValueError("frozen calibration differs from the current decision protocol")
        weight=a.checkpoint/"detector.safetensors" if a.type=="neural" else a.checkpoint
        blend=json.loads(a.frozen_blend.read_text())
        needed={name for name,w in zip(blend["names"],blend["selected"]["weights"]) if w!=0}
        needed.add(cal["provenance"]["attribution_model"])
        if cal.get("version_refinement"):needed.add(cal["version_refinement"]["statistical_model_name"])
        weights=cal["provenance"].get("component_weights_sha256",{})
        configs=cal["provenance"].get("component_metadata_sha256",{})
        observed=(sha(weight),metadata_fingerprint(a.type,a.checkpoint))
        if observed not in {(weights.get(name),configs.get(name)) for name in needed}:
            raise ValueError("checkpoint weight/configuration pair differs from a selected frozen calibration component")
    import pyarrow.parquet as pq
    import numpy as np
    source=pq.ParquetFile(a.input)
    if "split" not in source.schema.names:raise ValueError("an explicitly role-labeled prepared corpus is required")
    roles=set(source.read(columns=["split"]).column(0).to_pylist())
    if a.purpose=="calibration" and roles!={"calibration"}:
        raise ValueError("calibration export requires the independently frozen calibration role")
    if a.purpose=="development" and (not roles or not roles<={"validation","dev","dev_challenge","dev_challenge_source_domain","dev_backup","dev_backup_allowed"}):
        raise ValueError("development export refuses training/calibration/final corpus roles")
    if a.purpose=="final" and (not roles or any(not isinstance(r,str) or not (r=="test" or r.startswith("challenge_") or r.startswith("final_sealed")) for r in roles)):
        raise ValueError("final export requires a frozen test/challenge/final role")
    rows=source.read().to_pylist();original_n=len(rows)
    # Original OOD diagnostics preserve ineligible rows for auditing only.
    rows=[r for r in rows if r.get("eligible",True)]
    if not rows:raise ValueError("no eligible input rows")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.type=="neural":
        import torch
        from experiments.english_v1.neural import NeuralDetector,TokenCache,tokenizer_hash,export_predictions
        torch.set_num_threads(6)
        model=NeuralDetector.load_local(a.checkpoint).to(a.device).eval()
        cache=TokenCache.build(rows,model.tokenizer,Path(".cache/english_tokens"),
            data_sha256=sha(a.input),tokenizer_sha256=tokenizer_hash(a.checkpoint/"tokenizer"))
        result=export_predictions(model,rows,cache,a.output,batch_size=a.batch_size,device=a.device,precision="fp32",
            progress=lambda item:print(json.dumps(item),flush=True))
        weight=a.checkpoint/"detector.safetensors"
    else:
        from experiments.english_v1.statistical import EnglishStatisticalDetector
        from experiments.english_v1.stat_official_refine import export_raw_predictions
        model=EnglishStatisticalDetector.load(a.checkpoint)
        result,_=export_raw_predictions(model,rows,a.output)
        weight=a.checkpoint
    result["component_weights_sha256"]=np.asarray(sha(weight))
    result["component_metadata_sha256"]=np.asarray(metadata_fingerprint(a.type,a.checkpoint))
    result["export_purpose"]=np.asarray(a.purpose)
    from experiments.english_v1.language_guard import scope_arrays
    scope=scope_arrays(rows,sha(a.input))
    if not np.array_equal(result["lengths"],scope["scope_word_like_tokens"]):
        raise ValueError("model-export and deployment-scope word counts differ")
    result.update(scope)
    np.savez_compressed(a.output,**result)
    value={"purpose":a.purpose,"type":a.type,"checkpoint":str(a.checkpoint.resolve()),
        "weights_sha256":sha(weight),"input":str(a.input.resolve()),"input_sha256":sha(a.input),
        "component_metadata_sha256":str(result["component_metadata_sha256"].item()),
        "original_rows":original_n,"eligible_export_rows":len(rows),"excluded_ineligible_rows":original_n-len(rows),
        "output":str(a.output.resolve()),"output_sha256":sha(a.output),"calibration_fitted":False,
        "frozen_blend_sha256":sha(a.frozen_blend) if a.frozen_blend else None,
        "frozen_calibration_sha256":sha(a.frozen_calibration) if a.frozen_calibration else None,
        "code_sha256":{name:sha(Path(__file__).with_name(name)) for name in ("export_component.py","neural.py","statistical.py","catalog.py","language_guard.py")},
        "deployment_scope":{"rule_sha256":str(result["language_rule_sha256"].item()),"supported_rows":int(result["language_supported"].sum())}}
    a.output.with_suffix(".provenance.json").write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps({"exported":str(a.output),"rows":len(rows),"weights_sha256":value["weights_sha256"]}),flush=True)


if __name__=="__main__":main()
