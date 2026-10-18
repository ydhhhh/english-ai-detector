"""Choose a raw-margin blend using development partitions only, never calibration.

The model-selection halves are deterministic hashes of connected group IDs.
These are development results, not an unbiased estimate of final performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def _load(paths):
    arrays = [dict(np.load(p, allow_pickle=False)) for p in paths]
    reference = arrays[0]
    for a in arrays:
        for key in ("ids", "group_ids", "y_ai", "lengths"):
            if not np.array_equal(a[key], reference[key]):
                raise ValueError(f"aligned {key} required for every candidate")
    raw = []
    for a in arrays:
        x = np.asarray(a["raw_binary"], dtype=np.float64)
        if x.shape != (len(reference["ids"]), 2) or not np.all(np.isfinite(x)):
            raise ValueError("finite N by 2 raw_binary required")
        raw.append(x[:,1]-x[:,0])
    return reference, np.stack(raw, axis=1)


def _metrics(y, margin):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("both labels are required in each development selection cohort")
    # A development-only empirical operating point; this is NOT the final threshold.
    h = np.sort(margin[y == 0]); allowed = int(np.floor(.01*len(h)))
    threshold = np.nextafter(h[len(h)-allowed-1], np.inf)
    z = np.clip(margin, -700, 700)
    return {"n": len(y), "human_n": int((y==0).sum()), "AI_n": int((y==1).sum()),
            "AUROC": float(roc_auc_score(y, margin)),
            "AI_recall_at_empirical_human_FPR_1pct": float(np.mean(margin[y==1] >= threshold)),
            "nll": float(np.mean(np.logaddexp(0, z)-y*z))}


def fit_selection(names, validation_paths, challenge_paths, output, *, backup_paths=None):
    if not names or len(names) != len(set(names)) or len(names) != len(validation_paths) or len(names) != len(challenge_paths):
        raise ValueError("one unique name and two development prediction files per candidate")
    if backup_paths is not None and len(backup_paths)!=len(names):
        raise ValueError("one backup development prediction file per candidate required")
    cohort_paths={"validation":validation_paths,"dev_challenge":challenge_paths}
    if backup_paths is not None:cohort_paths["dev_finance"]=backup_paths
    cohort_weights={"validation":.3,"dev_challenge":.7} if backup_paths is None else {"validation":.3,"dev_challenge":.35,"dev_finance":.35}
    for p in [p for paths in cohort_paths.values() for p in paths]:
        stem = Path(p).stem.casefold()
        if any(t in stem for t in ("calibration", "test", "challenge_modern", "sealed", "ood", "final_")) or ("challenge_source_domain" in stem and not stem.startswith("dev_")):
            raise ValueError("selection refuses calibration and final test files")
    cohorts = {}
    components = {}
    for label, paths in cohort_paths.items():
        a, x = _load(paths)
        for name,path in zip(names,paths):
            with np.load(path,allow_pickle=False) as component:
                identity={}
                for source,key in (("component_weights_sha256","weights_sha256"),
                                   ("component_metadata_sha256","metadata_sha256")):
                    if source not in component or not re.fullmatch(r"[0-9a-f]{64}",str(component[source].item())):
                        raise ValueError(f"development selection requires fingerprinted raw predictions: {name}/{source}")
                    identity[key]=str(component[source].item())
            if name in components and components[name]!=identity:
                raise ValueError(f"development cohorts used different weights or inference configuration: {name}")
            components[name]=identity
        core=np.asarray(a["lengths"])>=128
        fit = np.array([int(hashlib.sha256(("blend-development-v1:"+str(g)).encode()).hexdigest()[:8], 16)%2 == 0 for g in a["group_ids"]])
        cohorts[label] = (a, x, fit & core, (~fit) & core, ~fit)
        if (fit & core).sum() < 50 or ((~fit) & core).sum() < 50:
            raise ValueError("insufficient independent development fit/selection rows")
    all_x = np.concatenate([x[m] for _,x,m,_,_ in cohorts.values()])
    all_y = np.concatenate([a["y_ai"][m] for a,_,m,_,_ in cohorts.values()])
    mean, scale = all_x.mean(0), np.maximum(all_x.std(0), 1e-6)
    xx = (all_x-mean)/scale
    # Give IID and source/domain development cohorts equal fitting mass.
    ww = np.concatenate([np.full(int(m.sum()), 1/len(cohorts)/int(m.sum())) for _,_,m,_,_ in cohorts.values()])
    candidates = []
    k = len(names)
    for i, name in enumerate(names):
        w = np.zeros(k); w[i] = 1.
        candidates.append((name, w, 0.))
    for i in range(k):
        for j in range(i+1,k):
            for alpha in (.25,.5,.75):
                w = np.zeros(k); w[i],w[j] = alpha,1-alpha
                candidates.append((f"{names[i]}:{alpha}+{names[j]}:{1-alpha}", w, 0.))
    if k > 2:
        candidates.append(("equal_all",np.full(k,1/k),0.))
    def objective(theta):
        w,b = theta[:-1],theta[-1]
        z = xx@w+b
        err = (expit(z)-all_y)*ww
        loss = np.dot(ww, np.logaddexp(0,z)-all_y*z)+.001*np.sum(w*w)
        return float(loss),np.r_[xx.T@err+.002*w,err.sum()]
    result = minimize(objective, np.r_[np.full(k,1/k),0.], jac=True, method="L-BFGS-B",
                      bounds=[(0.,20.)]*k+[(-20.,20.)])
    if result.success:
        candidates.append(("development_nonnegative_logistic",result.x[:-1],float(result.x[-1])))
    records = []
    for name,w,b in candidates:
        scores = {}
        all_length_scores={}
        for cohort,(a,x,_,m,all_m) in cohorts.items():
            scores[cohort] = _metrics(a["y_ai"][m], ((x[m]-mean)/scale)@w+b)
            all_length_scores[cohort]=_metrics(a["y_ai"][all_m],((x[all_m]-mean)/scale)@w+b)
        def quality(v):
            return .7*v["AUROC"]+.3*v["AI_recall_at_empirical_human_FPR_1pct"]
        selection = sum(cohort_weights[c]*quality(scores[c]) for c in cohorts)
        nll = sum(cohort_weights[c]*scores[c]["nll"] for c in cohorts)
        records.append({"name":name,"weights":w.tolist(),"intercept":b,
                        "selection_score":float(selection),"weighted_nll_tiebreak":float(nll),"metrics":scores,
                        "all_lengths_diagnostic_only":all_length_scores})
    records.sort(key=lambda r:(r["selection_score"],-r["weighted_nll_tiebreak"]),reverse=True)
    selected = records[0]
    artifact = {"schema_version":1,"fit_split":"development_only","names":names,
                "components":components,
                "mean":mean.tolist(),"scale":scale.tolist(),"selected":selected,"candidates":records,
                "criterion":"Core >=128 word-like tokens, fixed cohort weights; each 0.70 AUROC + 0.30 recall at empirical 1% human FPR; NLL tiebreak",
                "cohort_weights":cohort_weights,"minimum_word_like_tokens":128,
                "fit_selection_groups_disjoint":True,"final_test_opened":False,"calibration_opened":False,
                "source_files":{label:[{"path":str(Path(p).resolve()),"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths]
                                for label,paths in cohort_paths.items()}}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(artifact,indent=2,allow_nan=False)+"\n")
    return artifact


def blend_margin(artifact, raw_binary_by_name):
    values = []
    for name in artifact["names"]:
        a = np.asarray(raw_binary_by_name[name], dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 2 or not np.all(np.isfinite(a)):
            raise ValueError("finite raw binary matrices required")
        values.append(a[:,1]-a[:,0])
    x = np.stack(values,axis=1)
    z = (x-np.asarray(artifact["mean"]))/np.asarray(artifact["scale"])
    return z@np.asarray(artifact["selected"]["weights"])+artifact["selected"]["intercept"]


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names",nargs="+",required=True)
    p.add_argument("--validation",nargs="+",required=True)
    p.add_argument("--dev-challenge",nargs="+",required=True)
    p.add_argument("--dev-backup",nargs="+")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    value=fit_selection(a.names,a.validation,a.dev_challenge,a.output,backup_paths=a.dev_backup)
    print(json.dumps(value["selected"],indent=2))
