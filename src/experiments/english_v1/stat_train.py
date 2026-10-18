"""Train the CPU statistical baseline on explicitly named development files.

Final test/challenge files are refused by this training entry point. The model
itself accepts text-only inference so root evaluation can run separately.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))

from experiments.english_v1.statistical import (
    EnglishStatisticalDetector, StatisticalConfig, assert_disjoint, export_predictions,
)


def _pyarrow():
    try:
        import pyarrow.parquet as pq
        return pq
    except ImportError:
        vendor=PROJECT/"research/data_audit_20261017/vendor_fast"
        if not (vendor/"pyarrow").exists():
            raise RuntimeError("Parquet input requires the existing audited PyArrow runtime or a normal PyArrow installation") from None
        sys.path.insert(0,str(vendor))
        import pyarrow.parquet as pq
        return pq


def read_development_records(path, purpose):
    path=Path(path)
    if purpose not in ("train","validation","calibration","dev_challenge"):
        raise ValueError("training loader only accepts train/validation/calibration/dev_challenge")
    stem=path.stem.casefold()
    if stem == "test" or stem.startswith("test_") or stem in ("ood","challenge_source_domain"):
        raise ValueError("refusing final test/challenge labels in statistical training")
    if path.suffix.casefold() in (".parquet",".pq"):
        pq=_pyarrow()
        file=pq.ParquetFile(path)
        columns=[x for x in ("text","id","y_ai","family","model","source","domain","group_id","split",
            "language","mask_family","mask_model","normalized_sha256","label_evidence","out_of_model_catalog") if x in file.schema.names]
        rows=file.read(columns=columns).to_pylist()
    elif path.suffix.casefold() in (".jsonl",".ndjson"):
        with path.open(encoding="utf-8-sig") as stream:
            rows=[json.loads(line) for line in stream if line.strip()]
    else:
        raise ValueError("expected a Parquet or JSONL development file")
    allowed={"train":{"train"},"validation":{"validation","dev"},"calibration":{"calibration"},
        "dev_challenge":{"dev_challenge","dev_challenge_source_domain"}}[purpose]
    observed={row.get("split") for row in rows if row.get("split") is not None}
    if observed and not observed <= allowed:
        raise ValueError(f"unexpected {purpose} split values: {sorted(observed)}")
    return rows


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block:=stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def _dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def batch_invariance_check(model, texts):
    import numpy as np
    values=list(texts)[:8]
    if not values:
        raise ValueError("need development texts for invariance check")
    values += [max(texts,key=len)]
    alone=model.predict_proba(values,batch_size=1)
    together=model.predict_proba(values,batch_size=len(values))
    reverse=model.predict_proba(list(reversed(values)),batch_size=3)
    errors={}
    for task in alone:
        if not alone[task].size:
            errors[task]=0.
            continue
        error=max(float(np.max(np.abs(alone[task]-together[task]))),float(np.max(np.abs(alone[task]-reverse[task][::-1]))))
        if error>1e-9:
            raise AssertionError(f"{task} depends on inference batch composition: {error}")
        errors[task]=error
    return {"passed":True,"rows":len(values),"maximum_absolute_probability_difference":errors,
        "tested":"alone vs grouped and reversed/mixed batch sizes","tolerance":1e-9}


def run_experiment(train_path,validation_path,calibration_path,output_dir,dev_challenge_path=None,
                   config=None,pilot=False):
    started=time.monotonic()
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    config=config or StatisticalConfig()
    input_paths={"train":Path(train_path),"validation":Path(validation_path),"calibration":Path(calibration_path)}
    if dev_challenge_path:
        input_paths["dev_challenge"]=Path(dev_challenge_path)
    # Inspect allowed paths before opening any label-bearing file.
    for path in input_paths.values():
        if path.stem.casefold() in ("test","ood","challenge_source_domain") or path.stem.casefold().startswith("test_"):
            raise ValueError("final test/challenge path refused")
    with (output/"progress.jsonl").open("w",encoding="utf-8") as stream:
        def progress(value):
            item={"elapsed_seconds":round(time.monotonic()-started,3),**value}
            stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
            print(json.dumps(item,ensure_ascii=False,allow_nan=False),flush=True)
        records={name:read_development_records(path,name) for name,path in input_paths.items()}
        for name,rows in records.items():
            if name not in ("train","validation"):
                assert_disjoint(records["train"],rows,"train",name)
                assert_disjoint(records["validation"],rows,"validation",name)
        input_manifest={name:{"path":str(path.resolve()),"bytes":path.stat().st_size,
            "sha256":file_sha256(path),"rows":len(records[name])} for name,path in input_paths.items()}
        progress({"phase":"inputs_ready","inputs":input_manifest,"pilot_throughput_only":bool(pilot)})
        model=EnglishStatisticalDetector(config).fit(records["train"],records["validation"],records.get("dev_challenge"),progress=progress)
        model.fit_info_["pilot_throughput_only"]=bool(pilot)
        model.calibrate(records["calibration"])
        progress({"phase":"calibration_ready","calibration":model.calibration_})
        invariance=batch_invariance_check(model,[row["text"] for row in records["validation"]])
        model.save(output/"model.joblib")
        snapshots=output/"code"
        snapshots.mkdir(exist_ok=True)
        for source in (Path(__file__),Path(__file__).with_name("statistical.py"),PROJECT/"experiments/ntl_v1/baseline.py"):
            shutil.copyfile(source,snapshots/source.name)
        exports={}
        for name in ("validation","calibration","dev_challenge"):
            if name in records:
                exports[name]=export_predictions(model,records[name],output/(name+"_predictions.npz"),output/(name+"_predictions.jsonl"))
        dependencies={}
        for name in ("numpy","scipy","scikit-learn","joblib","threadpoolctl"):
            dependencies[name]=importlib.metadata.version(name)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes=int(peak if sys.platform=="darwin" else peak*1024)
        metadata={**model.metadata(),"inputs":input_manifest,"exports":exports,"batch_invariance":invariance,
            "pilot_throughput_only":bool(pilot),"not_a_formal_quality_result":bool(pilot),
            "dependencies":dependencies,"peak_process_rss_bytes":peak_bytes,
            "total_seconds":time.monotonic()-started,"final_evaluation_not_performed":True,
            "code_sha256":{str(path.relative_to(PROJECT)):file_sha256(path) for path in (
                Path(__file__),Path(__file__).with_name("statistical.py"),PROJECT/"experiments/ntl_v1/baseline.py")}}
        _dump(output/"metadata.json",metadata)
        _dump(output/"labels.json",{"binary":["human","AI"],"family":model.classes_.get("family",[]),"model":model.classes_.get("model",[])})
        _dump(output/"batch_invariance.json",invariance)
        progress({"phase":"complete","output":str(output.resolve()),"total_seconds":metadata["total_seconds"],
            "peak_process_rss_gib":peak_bytes/1024**3,"pilot_throughput_only":bool(pilot)})
    return model,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",required=True,type=Path)
    parser.add_argument("--validation",required=True,type=Path)
    parser.add_argument("--calibration",required=True,type=Path)
    parser.add_argument("--dev-challenge",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--feature-scale",type=float,default=1.)
    parser.add_argument("--max-binary-train-rows",type=int,default=120000)
    parser.add_argument("--cpu-threads",type=int,default=4)
    args=parser.parse_args()
    config=StatisticalConfig(feature_scale=args.feature_scale,max_binary_train_rows=args.max_binary_train_rows,cpu_threads=args.cpu_threads)
    run_experiment(args.train,args.validation,args.calibration,args.output,args.dev_challenge,config,args.pilot)


if __name__=="__main__":
    main()
