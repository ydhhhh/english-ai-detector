"""Frozen-vocabulary nominal18 augmentation; development-only, raw-only output.

The protocol is recorded before official-generic metrics are inspected. This
entry point accepts only its exact hash-bound training/development inputs and
never opens calibration or final data. Binary/family estimators remain frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from experiments.english_v1.catalog import NOMINAL_MODELS
from experiments.english_v1.statistical import (
    EnglishStatisticalDetector, WORD, _aggregate_window_scores, _array, _head_mask,
    _score_matrix, _texts, _window_plan, assert_disjoint,
)
from experiments.english_v1.stat_train import PROJECT, _dump, file_sha256, read_development_records


def training_weights(records,classes,official_ids):
    """Inverse-sqrt source counts, official4x, exactly equal class mass."""
    y=np.asarray([r["model"] for r in records]);sources=[str(r.get("source","")) for r in records]
    counts=Counter(zip(y,sources));n=len(records)
    values=np.asarray([(4. if str(r["id"]) in official_ids else 1.)/np.sqrt(counts[(c,s)])
        for r,c,s in zip(records,y,sources)])
    for label in classes:
        mask=y==label
        if not mask.any(): raise ValueError("every nominal18 training class is required")
        values[mask]*=(n/len(classes))/values[mask].sum()
    return values,{"formula":"(official4x else1)/sqrt(n_class_source), normalized to N/18 within each class",
        "total_weight":float(values.sum()),"minimum":float(values.min()),"maximum":float(values.max()),
        "class_weight_sums":{str(c):float(values[y==c].sum()) for c in classes},
        "source_class":[{"class":c,"source":s,"rows":count,"weight_sum":float(values[(y==c) & (np.asarray(sources)==s)].sum())}
            for (c,s),count in sorted(counts.items())]}


def export_raw_predictions(model,records,npz_path):
    """Standard aligned raw NPZ; intentionally contains no stale probabilities."""
    records=list(records)
    raw,coverage=model.predict_raw_scores(_texts(records),return_coverage=True)
    arrays={"ids":_array(records,"id"),"group_ids":_array(records,"group_id"),
        "normalized_sha256":_array(records,"normalized_sha256"),
        "y_ai":np.asarray([int(r["y_ai"]) for r in records],dtype=np.int64),
        "sources":_array(records,"source"),"domains":_array(records,"domain"),
        "lengths":np.asarray([len(WORD.findall(r["text"])) for r in records],dtype=np.int32),
        "family_true":_array(records,"family"),"model_true":_array(records,"model"),
        "raw_binary":raw["binary"],"raw_family":raw["family"],"raw_model":raw["model"],
        "raw_binary_margin":raw["binary"][:,1]-raw["binary"][:,0],
        "family_labels":np.asarray(model.classes_["family"]),"model_labels":np.asarray(model.classes_["model"]),
        "binary_labels":np.asarray(["human","AI"]),"length_unit":np.asarray("word_like_tokens_regex"),
        "calibrated":np.asarray(False),
        "window_counts":coverage["window_counts"],"input_chars":coverage["input_chars"],"covered_chars":coverage["covered_chars"]}
    for task in ("family","model"):
        arrays["mask_"+task]=_head_mask(records,task)&np.isin(arrays[task+"_true"],model.classes_[task])
    path=Path(npz_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)
    return arrays,coverage["summary"]


def _metrics(records,scores,classes):
    truth=np.asarray([r["model"] for r in records]);prediction=np.asarray(classes)[scores.argmax(1)]
    if set(truth)!=set(classes):raise ValueError("fixed18 development labels must all be supported")
    return {"n":len(records),"nominal18_macro_f1":float(f1_score(truth,prediction,labels=classes,average="macro",zero_division=0)),
        "accuracy":float(accuracy_score(truth,prediction)),"labels":list(classes),"catalogue":"closed nominal18; no unknown gate"}


def _identity(model):
    # sklearn refreshes _stop_words_id to id(stop_words) during transform. That
    # process-local cache address is not a fitted vocabulary/IDF parameter.
    feature_state={key:({name:{k:v for k,v in vars(vectorizer).items() if k!="_stop_words_id"}
        for name,vectorizer in value.items()} if key=="vectorizers_" else value)
        for key,value in vars(model.features_).items()}
    return {"features":joblib.hash(feature_state),"binary":joblib.hash(model.models_["binary"]),
        "family":joblib.hash(model.models_["family"])}


def run(protocol_path,output,complete_from_candidates=False,baseline_raw_dir=None):
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol_path=Path(protocol_path);protocol=json.loads(protocol_path.read_text())
    baseline_raw_dir=(PROJECT/"results/english_v1/development/statistical" if baseline_raw_dir is None else Path(baseline_raw_dir))
    source=PROJECT/protocol["source_model"]
    if file_sha256(source)!=protocol["source_model_sha256"]:raise ValueError("source model drifted from registered protocol")
    # Every permitted path and payload is bound before any label-bearing read.
    for name,entry in protocol["inputs"].items():
        if name not in ("train","official_train","mainval","official_generic"):raise ValueError("unexpected input role")
        if entry["purpose"] not in ("train","validation","dev_challenge"):raise ValueError("calibration/final role refused")
        if file_sha256(PROJECT/entry["path"])!=entry["sha256"]:raise ValueError("input differs from preregistered payload: "+name)
    records={name:read_development_records(PROJECT/e["path"],e["purpose"]) for name,e in protocol["inputs"].items()}
    assert_disjoint(records["train"],records["official_train"],"train","official_train")
    for training in ("train","official_train"):
        for development in ("mainval","official_generic"):assert_disjoint(records[training],records[development],training,development)
    assert_disjoint(records["mainval"],records["official_generic"],"mainval","official_generic")
    model=EnglishStatisticalDetector.load(source);model.config.cpu_threads=4
    classes=np.asarray(model.classes_["model"])
    if set(classes)!=set(NOMINAL_MODELS):raise ValueError("exact nominal18 catalogue required")
    identity=_identity(model)
    old_estimator=model.models_["model"]
    train=[r for r,m in zip(records["train"]+records["official_train"],_head_mask(records["train"]+records["official_train"],"model")) if m and r["model"] in classes]
    official_ids={str(r["id"]) for r in records["official_train"]}
    if len(train)!=36818+2159 or len(official_ids)!=2159:raise ValueError("registered trusted training counts changed")
    y=np.asarray([r["model"] for r in train]);weights,weight_info=training_weights(train,classes,official_ids)
    dev={name:[r for r,m in zip(records[name],_head_mask(records[name],"model")) if m and r["model"] in classes] for name in ("mainval","official_generic")}
    def log(item):
        value={"elapsed_seconds":round(time.monotonic()-started,3),**item}
        with (output/"progress.jsonl").open("a") as f:f.write(json.dumps(value,allow_nan=False)+"\n")
        print(json.dumps(value,allow_nan=False),flush=True)
    log({"phase":"inputs_verified","fit_rows":len(train),"official_rows":len(official_ids),"development_rows":{k:len(v) for k,v in dev.items()}})
    history=[];saved={};best=None;best_score=-np.inf
    prior_progress=[json.loads(line) for line in (output/"progress.jsonl").read_text().splitlines() if line.strip()] if complete_from_candidates else []
    with threadpool_limits(limits=4):
        if complete_from_candidates:
            expected=["unchanged_old_SVC","official_LinearSVC_C0.3","official_LinearSVC_C1.0"]
            history=[{k:v for k,v in entry.items() if k not in ("phase","elapsed_seconds")} for entry in prior_progress if entry.get("phase")=="candidate_ready"]
            if [h["candidate"] for h in history]!=expected:raise ValueError("completion requires exactly the three registered completed candidates")
            for entry in history:
                score=sum(protocol["selection"]["cohort_weights"][cohort]*m["nominal18_macro_f1"] for cohort,m in entry["cohort_metrics"].items())
                if abs(score-entry["selection_score"])>1e-14:raise ValueError("cached candidate selection score changed")
            selected=max(history,key=lambda e:e["selection_score"])
            best_name=selected["candidate"];best_score=selected["selection_score"]
            best=old_estimator if best_name==expected[0] else joblib.load(output/(best_name+".joblib"))
            if not np.array_equal(best.classes_,classes):raise ValueError("cached estimator class order changed")
            estimate=next(x["estimated_solver_peak_gib"] for x in prior_progress if x.get("phase")=="features_ready")
            log({"phase":"complete_saved_candidates","selected":best_name,"retraining_performed":False})
        else:
            fit=model.features_.transform(_texts(train))
            plans={name:_window_plan(_texts(rows),model.config) for name,rows in dev.items()}
            matrices={name:model.features_.transform(plan["texts"]) for name,plan in plans.items()}
            estimate=(fit.data.nbytes+fit.indices.nbytes+fit.indptr.nbytes+24*fit.nnz+16*len(classes)*fit.shape[1]+sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for x in matrices.values()))/1024**3+2.
            log({"phase":"features_ready","features":fit.shape[1],"estimated_solver_peak_gib":estimate})
            if estimate>18.:raise MemoryError("registered estimated solver limit exceeded")
            for name,c in (("unchanged_old_SVC",None),("official_LinearSVC_C0.3",.3),("official_LinearSVC_C1.0",1.)):
                t0=time.monotonic();messages=[]
                if c is None:estimator=old_estimator
                else:
                    estimator=LinearSVC(C=c,dual="auto",class_weight=None,max_iter=model.config.svc_max_iter,random_state=model.config.seed)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always",ConvergenceWarning);estimator.fit(fit,y,sample_weight=weights)
                    messages=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
                if not np.array_equal(estimator.classes_,classes):raise AssertionError("nominal18 class order changed")
                metrics={};scores={}
                for cohort,matrix in matrices.items():
                    scores[cohort]=_aggregate_window_scores(_score_matrix(estimator,matrix),plans[cohort])
                    metrics[cohort]=_metrics(dev[cohort],scores[cohort],classes)
                score=sum(protocol["selection"]["cohort_weights"][cohort]*metrics[cohort]["nominal18_macro_f1"] for cohort in metrics)
                entry={"candidate":name,"C":c,"cohort_metrics":metrics,"selection_score":score,"seconds":time.monotonic()-t0,"convergence_warnings":messages}
                history.append(entry);saved[name]=scores
                if score>best_score:best_score,best,best_name=score,estimator,name
                if c is not None:joblib.dump(estimator,output/(name+".joblib"),compress=3)
                log({"phase":"candidate_ready",**entry})
            del fit,matrices
            gc.collect()
        # Before changing the head, obtain complete development binary/family
        # raw scores. No calibration probabilities are evaluated or exported.
        old_raw={};baseline_raw_inputs={}
        for name,filename in (("mainval","validation.npz"),("official_generic","historical_dev.npz")):
            cached=baseline_raw_dir/filename
            if cached.exists():
                side=json.loads(cached.with_suffix(".provenance.json").read_text())
                if side["output_sha256"]!=file_sha256(cached) or side["input_sha256"]!=protocol["inputs"][name]["sha256"] or side["weights_sha256"]!=protocol["source_model_sha256"]:
                    raise ValueError("cached baseline development raw is not bound to the registered model/input")
                with np.load(cached,allow_pickle=False) as a:
                    if list(map(str,a["ids"]))!=[str(r["id"]) for r in records[name]] or not np.array_equal(a["family_labels"],model.classes_["family"]) or not np.array_equal(a["model_labels"],model.classes_["model"]):
                        raise ValueError("cached baseline row/class layout mismatch")
                    old_raw[name]={task:a["raw_"+task] for task in ("binary","family","model")}
                baseline_raw_inputs[name]={"path":str(cached.resolve()),"sha256":file_sha256(cached),"source":"existing fingerprinted raw export; no repeated forward"}
            else:
                old_raw[name]=model.predict_raw_scores(_texts(records[name]))
                baseline_raw_inputs[name]={"source":"computed from unchanged source checkpoint"}
        model.models_["model"]=best
        model.calibrators_={};model.calibration_={};model.calibrated_=False
        model.calibration_window_coverage_=None
        for field,attr in (("id","training_ids_"),("group_id","training_groups_"),("normalized_sha256","training_hashes_")):
            setattr(model,attr,frozenset(getattr(model,attr)) | frozenset(str(r[field]) for r in records["official_train"] if r.get(field) not in (None,"")))
        for field,attr in (("id","validation_ids_"),("group_id","validation_groups_"),("normalized_sha256","validation_hashes_")):
            setattr(model,attr,frozenset(getattr(model,attr)) | frozenset(str(r[field]) for r in records["official_generic"] if r.get(field) not in (None,"")))
        model.fit_info_["official_nominal_refinement"]={"actual_model_head_training_rows":len(train) if best_name!="unchanged_old_SVC" else 36818,
            "official_training_rows_used_in_selected_head":2159 if best_name!="unchanged_old_SVC" else 0,
            "only_model_head_retrained":best_name!="unchanged_old_SVC","binary_family_and_vocabulary_frozen":True,
            "new_calibration_required":True,"protocol_sha256":file_sha256(protocol_path)}
        model.selection_["model"]={**model.selection_["model"],"selected":best_name,"official_generic_candidates":history,
            "selection_rule":protocol["selection"],"source_calibrators_removed":True}
        if _identity(model)!=identity:raise AssertionError("binary/family or feature parameters changed")
        model.save(output/"model.joblib")
        model_sha=file_sha256(output/"model.joblib");exports={};verification={"unchanged_parameter_hashes":identity,"complete_cohort_raw_equality":{}}
        for name in dev:
            path=output/(name+"_predictions.npz")
            arrays,coverage=export_raw_predictions(model,records[name],path)
            verification["complete_cohort_raw_equality"][name]={task:bool(np.array_equal(arrays["raw_"+task],old_raw[name][task])) for task in ("binary","family")}
            if not all(verification["complete_cohort_raw_equality"][name].values()):raise AssertionError("frozen raw head changed")
            arrays["component_weights_sha256"]=np.asarray(model_sha)
            arrays["component_metadata_sha256"]=np.asarray(model_sha)
            np.savez_compressed(path,**arrays)
            exports[name]={"path":str(path.resolve()),"sha256":file_sha256(path),"rows":len(records[name]),"coverage":coverage}
            _dump(path.with_suffix(".provenance.json"),{"purpose":"development","type":"statistical","checkpoint":str((output/"model.joblib").resolve()),
                "weights_sha256":model_sha,"component_metadata_sha256":model_sha,"output_sha256":file_sha256(path),
                "input_sha256":protocol["inputs"][name]["sha256"],"calibration_fitted":False,"raw_only":True})
        probe=_texts(records["mainval"][:6])+[max(_texts(records["official_generic"]),key=len)]
        x=model.predict_raw_scores(probe,batch_size=1);z=model.predict_raw_scores(list(reversed(probe)),batch_size=7)
        verification["single_batch_order_invariant"]=all(np.allclose(x[k],z[k][::-1],atol=1e-12,rtol=0) for k in x)
        if not verification["single_batch_order_invariant"]:raise AssertionError("raw batch invariance failed")
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes=int(peak if sys.platform=="darwin" else peak*1024)
    info={**model.metadata(),"raw_only":True,"calibrated":False,"calibration_status":"all inherited calibrators removed; root must fit final calibration after selection",
        "inputs":protocol["inputs"],"source_model":str(source.resolve()),"source_weights_sha256":protocol["source_model_sha256"],
        "component_weights_sha256":model_sha,"component_metadata_sha256":model_sha,
        "selected":best_name,"candidates":history,"training_sample_weights":weight_info,"exports":exports,
        "protocol_sha256":file_sha256(protocol_path),"verification":verification,
        "baseline_raw_inputs":baseline_raw_inputs,
        "baseline_raw_cache_directory":str(baseline_raw_dir.resolve()),
        "peak_process_rss_bytes":peak_bytes,"total_seconds":time.monotonic()-started,
        "resource_scope":"completion process only; initial fit peak RSS was not captured after the pre-save exception" if complete_from_candidates else "complete training process",
        "estimated_solver_peak_gib":estimate,
        "completed_from_saved_candidates":bool(complete_from_candidates),
        "fit_code_sha256":file_sha256(output/"code/stat_official_refine_fit.py") if (output/"code/stat_official_refine_fit.py").exists() else file_sha256(__file__),
        "calibration_inputs_opened":False,"final_inputs_opened":False,
        "code_sha256":file_sha256(__file__)}
    # Override inherited generic probability prose for the deliberately raw-only
    # candidate; its joblib refuses probability inference until recalibrated.
    info["probability_semantics"]={"available":False,"reason":"raw-only candidate; no probability calibration is valid"}
    _dump(output/"metadata.json",info);_dump(output/"verification.json",verification)
    _dump(output/"labels.json",model.classes_)
    code=output/"code";code.mkdir(exist_ok=True)
    for p in (Path(__file__),Path(__file__).with_name("statistical.py")):shutil.copyfile(p,code/p.name)
    log({"phase":"complete","selected":best_name,"total_seconds":info["total_seconds"],"peak_process_rss_gib":peak_bytes/1024**3})
    return info


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--protocol",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--complete-from-candidates",action="store_true",help="resume only serialization/verification after all registered candidates completed")
    p.add_argument("--baseline-raw-dir",type=Path,help="Optional fingerprinted source raw cache; use a new empty directory for a cold replay so historical checkpoints cannot be reused")
    a=p.parse_args();run(a.protocol,a.output,a.complete_from_candidates,a.baseline_raw_dir)
