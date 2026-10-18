"""Optional known-version ranking refinement that preserves neural unknown mass.

Select with validation connected groups only; calibrate known temperatures on the
independent calibration split only. A caller attaches the returned calibration
dictionary as artifact['calibration']. Deployment must verify both component
weight hashes and the exact stored column orders before calling ``apply``.
Apply to calibrated leaves before fitting the final model assertion cutoff.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import f1_score

from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS, MODEL_FAMILIES, NOMINAL_MODELS

SEED = 20261017
WEIGHTS = (0., .25, .5, .75, 1.)
_ROW_KEYS = frozenset(("ids", "group_ids", "normalized_sha256", "normalized_hashes",
    "y_ai", "family_true", "model_true", "raw_model_true", "mask_family", "mask_model",
    "p_family", "p_model", "raw_version", "raw_model", "development_cohort",
    "component_weights_sha256", "component_metadata_sha256"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block:=stream.read(1024*1024): h.update(block)
    return h.hexdigest()


def _selection_hash(artifact):
    return _digest({k:v for k,v in artifact.items() if k!="calibration"})


def _component_identity(arrays,key):
    """Accept an export scalar or an explicitly preserved homogeneous row vector.

    Concatenation code must verify scalar input headers before collapsing them.
    Retaining a per-row identity vector lets this module verify that condition.
    """
    values=np.asarray(arrays.get(key,""))
    if values.ndim>1 or values.size not in (1,len(arrays.get("ids",[]))):
        raise ValueError(f"scalar or aligned row-vector {key} required")
    unique=set(map(str,values.reshape(-1)))
    if len(unique)!=1: raise ValueError(f"development cohorts contain different {key}")
    value=next(iter(unique))
    return value,bool(re.fullmatch(r"[0-9a-f]{64}",value)),("per_row_homogeneous" if values.ndim==1 and values.size>1 else "scalar_declared")


def _source(value, purpose):
    if isinstance(value, (str, Path)):
        path=Path(value)
        tokens=set(re.split(r"[^a-z0-9]+",path.stem.casefold()))
        finalish=bool(tokens & {"test","ood","final","sealed"}) or ("challenge" in tokens and ("dev" not in tokens or "modern" in tokens))
        if finalish or (purpose=="development" and tokens & {"calibration","cal"}):
            raise ValueError("version refinement refuses final inputs and calibration during selection")
        if purpose=="calibration" and tokens & {"validation","development","dev"}:
            raise ValueError("known-temperature fitting requires independent calibration inputs")
        provenance=path.with_suffix(".provenance.json")
        side=json.loads(provenance.read_text()) if provenance.exists() else {}
        if side.get("purpose") and side["purpose"]!=purpose:
            raise ValueError("prediction provenance purpose disagrees with requested operation")
        with np.load(path,allow_pickle=False) as data:
            arrays={k:data[k] for k in data.files}
        file_hash=_file_sha(path)
        fingerprint={"path":str(path.resolve()),"prediction_sha256":file_hash,
            "provenance_sha256":_file_sha(provenance) if provenance.exists() else None}
    else:
        arrays={k:np.asarray(v) for k,v in dict(value).items()}
        side={}
        fingerprint={"path":None,"prediction_sha256":None,
            "provenance_sha256":None,"in_memory_payload":True}
    declared,valid,weights_scope=_component_identity(arrays,"component_weights_sha256")
    if valid and side.get("weights_sha256") and side["weights_sha256"]!=declared:
        raise ValueError("NPZ and source-sidecar model fingerprints disagree")
    if not valid and side.get("output_sha256")==fingerprint.get("prediction_sha256"):
        candidate=side.get("weights_sha256","")
        if re.fullmatch(r"[0-9a-f]{64}",candidate):
            declared,valid=candidate,True
            fingerprint["weights_fingerprint_source"]="hash-bound source sidecar"
    fingerprint.update({"component_weights_sha256":declared if valid else None,
        "weights_status":"declared_in_export" if valid else "not_verified",
        "weights_identity_scope":weights_scope,
        "weights_fingerprint_source":fingerprint.get("weights_fingerprint_source",
            "NPZ.component_weights_sha256" if valid else "not provided")})
    metadata,metadata_valid,metadata_scope=_component_identity(arrays,"component_metadata_sha256")
    if metadata_valid and side.get("component_metadata_sha256") and side["component_metadata_sha256"]!=metadata:
        raise ValueError("NPZ and source-sidecar inference metadata fingerprints disagree")
    metadata_source="NPZ.component_metadata_sha256" if metadata_valid else "not provided"
    if not metadata_valid and side.get("output_sha256")==fingerprint.get("prediction_sha256"):
        candidate=side.get("component_metadata_sha256","")
        if re.fullmatch(r"[0-9a-f]{64}",candidate):
            metadata,metadata_valid,metadata_source=candidate,True,"hash-bound source sidecar"
    fingerprint.update({"component_metadata_sha256":metadata if metadata_valid else None,
        "metadata_status":"declared_in_export" if metadata_valid else "not_verified",
        "metadata_identity_scope":metadata_scope,
        "metadata_fingerprint_source":metadata_source})
    return arrays,fingerprint


def _labels(a,key):
    value=np.asarray(a[key])
    if value.ndim!=1: raise ValueError(f"one-dimensional {key} required")
    result=[str(x) for x in value]
    if len(set(result))!=len(result): raise ValueError(f"duplicate {key}")
    return result


def _matrix(value,n,k,name):
    x=np.asarray(value,dtype=np.float64)
    if x.shape!=(n,k) or not np.isfinite(x).all():
        raise ValueError(f"finite {n} by {k} {name} required")
    return x


def _bools(a,key,n):
    x=np.asarray(a[key])
    if x.shape!=(n,) or not np.isin(x,[0,1,False,True]).all():
        raise ValueError(f"aligned boolean {key} required")
    return x.astype(bool)


def _take_rows(arrays,order):
    n=len(arrays["ids"])
    return {key:(value[order] if key in _ROW_KEYS and value.ndim>0 and value.shape[0]==n else value) for key,value in arrays.items()}


def _pair(reference,statistical,purpose):
    ref,rf=_source(reference,purpose);stat,sf=_source(statistical,purpose)
    ids=_labels(ref,"ids");other=_labels(stat,"ids")
    if not ids or set(ids)!=set(other): raise ValueError("identical nonempty unique row IDs required")
    index={v:i for i,v in enumerate(other)};order=np.asarray([index[v] for v in ids])
    n=len(ids)
    stat=_take_rows(stat,order)
    for a in (ref,stat):
        if np.asarray(a["y_ai"]).shape!=(n,) or not np.isin(a["y_ai"],[0,1]).all():
            raise ValueError("aligned human=0/AI=1 labels required")
    if not np.array_equal(ref["y_ai"],stat["y_ai"]): raise ValueError("root labels disagree")
    families=_labels(ref,"family_labels");leaves=_labels(ref,"model_labels");models=_labels(stat,"model_labels")
    if set(families)!=set(FAMILIES) or set(leaves)!=set(MODEL_LABELS) or set(models)!=set(NOMINAL_MODELS):
        raise ValueError("frozen 7-family/25-leaf neural and nominal-18 statistical catalogues required")
    lf=dict(zip(MODEL_LABELS,MODEL_FAMILIES))
    if "model_families" in ref and list(map(str,ref["model_families"]))!=[lf[x] for x in leaves]:
        raise ValueError("neural leaf-family layout mismatch")
    layout={"family_labels":families,"neural_leaf_labels":leaves,"stat_model_labels":models,
        "leaf_families":[lf[x] for x in leaves],"nominal_models":list(NOMINAL_MODELS),"mapping":{}}
    for family in families:
        known=[i for i,x in enumerate(leaves) if x in NOMINAL_MODELS and lf[x]==family]
        unknown=[i for i,x in enumerate(leaves) if x not in NOMINAL_MODELS and lf[x]==family]
        layout["mapping"][family]={"family_column":families.index(family),
            "known_leaf_columns":known,"stat_model_columns":[models.index(leaves[i]) for i in known],
            "unknown_leaf_columns":unknown}
    raw_n=_matrix(ref["raw_version"],n,len(leaves),"neural raw_version")
    raw_s=_matrix(stat["raw_model"],n,len(models),"statistical raw_model")
    q=_matrix(ref["p_family"],n,len(families),"p_family")
    base=_matrix(ref["p_model"],n,len(leaves),"base_p_leaf")
    _probability_checks(layout,q,base)
    known_mask=_trusted_known(ref,stat,layout,n)
    return {"ref":ref,"stat":stat,"fingerprints":{"reference":rf,"statistical":sf},
        "layout":layout,"raw_n":raw_n,"raw_s":raw_s,"q":q,"base":base,"known_mask":known_mask}


def _trusted_known(ref,stat,layout,n):
    rm=_bools(ref,"mask_model",n);sm=_bools(stat,"mask_model",n)
    rf=_bools(ref,"mask_family",n);sf=_bools(stat,"mask_family",n)
    if np.any(rm & ~rf) or np.any(sm & ~sf): raise ValueError("model mask requires trusted family mask")
    rt=np.asarray(ref.get("raw_model_true",ref["model_true"]),dtype=str)
    st=np.asarray(stat["model_true"],dtype=str)
    if rt.shape!=(n,) or st.shape!=(n,): raise ValueError("aligned fine truths required")
    if np.any(sm & ~np.isin(st,NOMINAL_MODELS)): raise ValueError("statistical trusted labels outside nominal18")
    both=rm & sm
    if np.any(both & (rt!=st)): raise ValueError("trusted nominal label disagreement")
    mask=both & (np.asarray(ref["y_ai"])==1) & np.isin(rt,NOMINAL_MODELS)
    lf=dict(zip(layout["neural_leaf_labels"],layout["leaf_families"]))
    for a in (ref,stat):
        fam=np.asarray(a["family_true"],dtype=str)
        if fam.shape!=(n,) or any(fam[i]!=lf[rt[i]] for i in np.flatnonzero(mask)):
            raise ValueError("trusted nominal model contradicts declared family")
    if np.any((rm|sm) & (np.asarray(ref["y_ai"])==0)):
        raise ValueError("human rows cannot carry trusted version labels")
    return mask


def _probability_checks(layout,q,base):
    if np.any(q<0) or np.any(base<0) or np.any(q>1) or np.any(base>1):
        raise ValueError("probability entries must lie in [0,1]")
    if not np.allclose(q.sum(1),1.,atol=2e-6,rtol=0) or not np.allclose(base.sum(1),1.,atol=2e-6,rtol=0):
        raise ValueError("family and leaf probabilities must sum to one")
    for family,m in layout["mapping"].items():
        columns=m["known_leaf_columns"]+m["unknown_leaf_columns"]
        if not np.allclose(base[:,columns].sum(1),q[:,m["family_column"]],atol=2e-6,rtol=0):
            raise ValueError(f"base leaf mass disagrees with family probability: {family}")


def _identity_tokens(pair):
    n=len(pair["ref"]["ids"])
    result={"ids":set(),"groups":set(),"content":set()}
    for value in pair["ref"]["ids"]: result["ids"].add(_digest(["id",str(value)]))
    for a in (pair["ref"],pair["stat"]):
        if "group_ids" not in a or np.asarray(a["group_ids"]).shape!=(n,):
            raise ValueError("connected-group selection requires group_ids in both exports")
        for value in a["group_ids"]:
            if str(value): result["groups"].add(_digest(["group",str(value)]))
        for key in ("normalized_sha256","normalized_hashes"):
            if key in a:
                if np.asarray(a[key]).shape!=(n,): raise ValueError("content identity row mismatch")
                for value in a[key]:
                    if str(value): result["content"].add(_digest(["content",str(value)]))
    return {key:sorted(values) for key,values in result.items()}


def _connected_split(pair):
    n=len(pair["ref"]["ids"]);parent=np.arange(n)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[max(ra,rb)]=min(ra,rb)
    first={};available=[];missing=[]
    for name in ("ref","stat"):
        a=pair[name]
        if "group_ids" not in a: raise ValueError("group_ids required in both sources")
        for key,namespace in (("group_ids","group"),("normalized_sha256","content"),("normalized_hashes","content")):
            if key not in a: continue
            values=np.asarray(a[key]);available.append(name+"."+key)
            if values.shape!=(n,): raise ValueError("connected identity row mismatch")
            if namespace=="group": missing.append(values.astype(str)=="")
            for i,value in enumerate(values):
                if not str(value): continue
                token=(namespace,str(value))
                if token in first: union(i,first[token])
                else: first[token]=i
    if not first or np.any(np.logical_and.reduce(missing) & pair["known_mask"]):
        raise ValueError("every trusted nominal row needs a supplied group identifier")
    members={}
    for i in range(n): members.setdefault(find(i),[]).append(str(pair["ref"]["ids"][i]))
    keys={root:_digest(sorted(ids)) for root,ids in members.items()}
    components=np.asarray([keys[find(i)] for i in range(n)])
    fit=np.asarray([int(hashlib.sha256(f"version-refinement-v1:{SEED}:{key}".encode()).hexdigest()[:8],16)%2==0 for key in components])
    return fit,components,{"method":"connected components of both exports' group IDs plus available normalized content hashes; deterministic component SHA-256 parity",
        "seed":SEED,"available_identity_edges":available,"connected_components":len(members),
        "fit_components":len(set(components[fit])),"selection_components":len(set(components[~fit])),
        "fit_selection_components_disjoint":not bool(set(components[fit]) & set(components[~fit])),
        "normalized_content_edges_available":any("normalized" in key for key in available)}


def _center(x): return x-x.mean(axis=1,keepdims=True)


def _mixed(artifact,raw_n,raw_s,family):
    m=artifact["mapping"][family];sc=artifact["normalization"][family]
    nn=_center(raw_n[:,m["known_leaf_columns"]])/sc["neural_scale"]
    ss=_center(raw_s[:,m["stat_model_columns"]])/sc["statistical_scale"]
    w=float(artifact["w_stat"])
    return (1-w)*nn+w*ss


def apply(artifact,p_family,base_p_leaf,raw_neural_version,raw_stat_model):
    """Preserve every unknown/Other entry exactly and each family mass to roundoff."""
    original=np.asarray(base_p_leaf)
    if original.ndim!=2 or original.shape[1]!=len(artifact["neural_leaf_labels"]) or not np.isfinite(original).all():
        raise ValueError("finite N by declared-leaf base probabilities required")
    if not artifact["enabled"]:
        return original.copy()  # A true identity baseline, without any rescaling.
    n=len(original)
    q=_matrix(p_family,n,len(artifact["family_labels"]),"p_family")
    base=_matrix(original,n,len(artifact["neural_leaf_labels"]),"base_p_leaf")
    rn=_matrix(raw_neural_version,n,len(artifact["neural_leaf_labels"]),"raw neural version")
    rs=_matrix(raw_stat_model,n,len(artifact["stat_model_labels"]),"raw statistical model")
    _probability_checks(artifact,q,base)
    cal=artifact.get("calibration")
    if cal and cal.get("selection_artifact_sha256")!=_selection_hash(artifact):
        raise ValueError("known calibration does not belong to this frozen selection artifact")
    temperatures=cal.get("temperatures",{}) if cal else {}
    result=base.copy()
    for family,m in artifact["mapping"].items():
        known=m["known_leaf_columns"]
        if not known: continue  # Other-family leaf remains exactly untouched.
        unknown_mass=base[:,m["unknown_leaf_columns"]].sum(1)
        known_mass=q[:,m["family_column"]]-unknown_mass
        if np.any(known_mass < -2e-6): raise ValueError("unknown mass exceeds family mass")
        known_mass=np.maximum(known_mass,0.)
        if len(known)==1:
            result[:,known[0]]=known_mass
            continue
        t=float(temperatures.get(family,{}).get("temperature",1.))
        if not np.isfinite(t) or t<=0: raise ValueError("positive finite known temperature required")
        relative=softmax(_mixed(artifact,rn,rs,family)/t,axis=1)
        allocated=known_mass[:,None]*relative
        # Put roundoff on the largest entry, which cannot become negative when
        # an extremely unlikely last class underflows to zero.
        largest=relative.argmax(1)
        allocated[np.arange(n),largest]+=known_mass-allocated.sum(1)
        result[:,known]=allocated
    frozen=[i for i,x in enumerate(artifact["neural_leaf_labels"]) if x not in NOMINAL_MODELS]
    if not np.array_equal(result[:,frozen],base[:,frozen]): raise AssertionError("unknown probability mass changed")
    _probability_checks(artifact,q,result)
    return result


def _metrics(pair,probabilities,mask):
    labels=pair["layout"]["neural_leaf_labels"]
    truth=np.asarray(pair["ref"].get("raw_model_true",pair["ref"]["model_true"]),dtype=str)[mask]
    if not len(truth): raise ValueError("no trusted nominal18 selection rows")
    p=probabilities[mask];predicted=np.asarray(labels)[p.argmax(1)]
    target=np.asarray([labels.index(x) for x in truth])
    return {"n":len(truth),"nominal18_macro_f1":float(f1_score(truth,predicted,labels=NOMINAL_MODELS,average="macro",zero_division=0)),
        "full25_leaf_accuracy_on_nominal18":float(np.mean(predicted==truth)),
        "joint_leaf_nll":float(-np.log(np.maximum(p[np.arange(len(p)),target],np.finfo(float).tiny)).mean()),
        "prediction_catalogue":"all 25 leaves; unknown predictions count as errors on trusted nominal18 rows"}


def _cohorts(pair,development_cohort,cohort_weights):
    n=len(pair["ref"]["ids"])
    embedded=pair["ref"].get("development_cohort")
    values=np.asarray(development_cohort if development_cohort is not None else
        (embedded if embedded is not None else np.full(n,"validation")),dtype=str)
    if values.shape!=(n,) or np.any(values==""): raise ValueError("aligned nonempty development_cohort vector required")
    for source in (pair["ref"],pair["stat"]):
        if "development_cohort" in source and not np.array_equal(values,np.asarray(source["development_cohort"],dtype=str)):
            raise ValueError("development cohort identities disagree between inputs")
    names=sorted(set(values))
    if cohort_weights is None: weights={name:1/len(names) for name in names}
    else:
        weights={str(name):float(weight) for name,weight in dict(cohort_weights).items()}
        if set(weights)!=set(names) or any(not np.isfinite(w) or w<=0 for w in weights.values()) or not np.isclose(sum(weights.values()),1.,atol=1e-12,rtol=0):
            raise ValueError("positive fixed cohort weights summing to one must exactly match cohort names")
    if len(names)>1:
        for source in pair["fingerprints"].values():
            if source["weights_status"]=="not_verified" or source["metadata_status"]=="not_verified":
                raise ValueError("multiple development cohorts require verified component weight and inference metadata fingerprints")
    return values,weights


def _cohort_metrics(pair,probabilities,selection_mask,cohorts,weights):
    detail={name:_metrics(pair,probabilities,selection_mask & (cohorts==name)) for name in weights}
    result={key:float(sum(weights[name]*detail[name][key] for name in weights)) for key in
        ("nominal18_macro_f1","joint_leaf_nll","full25_leaf_accuracy_on_nominal18")}
    result.update({"n":sum(x["n"] for x in detail.values()),"cohorts":detail,
        "aggregation":"fixed cohort-weighted means; each cohort uses fixed 18-label macro-F1 and full25 argmax"})
    return result


def select_development(ref_npz,stat_npz,development_cohort=None,cohort_weights=None):
    """Select on connected-group halves; optional cohorts receive fixed mass.

    An embedded ``development_cohort`` vector is used when the explicit argument
    is omitted. Concatenation must preserve homogeneous component identities (a
    per-row hash vector is accepted) and must never include blind final rows.
    """
    pair=_pair(ref_npz,stat_npz,"development")
    cohorts,weights=_cohorts(pair,development_cohort,cohort_weights)
    fit,components,split=_connected_split(pair)
    trusted=pair["known_mask"];fit_mask=fit & trusted;selection_mask=(~fit) & trusted
    truth=np.asarray(pair["ref"].get("raw_model_true",pair["ref"]["model_true"]),dtype=str)
    support=lambda mask:{label:int(np.sum(mask & (truth==label))) for label in NOMINAL_MODELS}
    artifact={"schema_version":2,**pair["layout"],"enabled":False,"w_stat":0.,"normalization":{},
        "fingerprints":pair["fingerprints"],"development_split":{**split,"fit_trusted_known_rows":int(fit_mask.sum()),
            "selection_trusted_known_rows":int(selection_mask.sum()),"fit_support":support(fit_mask),"selection_support":support(selection_mask),
            "fit_component_sha256":_digest(sorted(set(components[fit]))),"selection_component_sha256":_digest(sorted(set(components[~fit])))},
        "cohort_weights":weights,"development_cohorts":{name:{"total_rows":int(np.sum(cohorts==name)),
            "fit_support":support(fit_mask & (cohorts==name)),"selection_support":support(selection_mask & (cohorts==name))} for name in weights},
        "development_identity_hashes":_identity_tokens(pair),"calibration_opened":False,"final_inputs_opened":False,
        "fine_source_ood_selection":False,"selection_scope":"nonblind declared development cohorts only; no claim of independent source-OOD fine-label evaluation",
        "criterion":"fixed cohort-weighted trusted nominal18 macro-F1 using full25 argmax; same-weight joint leaf NLL tiebreak; exact identity baseline first on exact ties",
        "normalization_method":"each head's known logits row-centered, then divided by true-family trusted-fit RMS with fixed cohort mass (not row mass); fixed per-family scales at inference",
        "unknown_semantics":"all neural unknown-version and Other-family leaf probabilities are copied; 1-max is not an unknown probability",
        "code_sha256":_file_sha(__file__)}
    if any(v<1 for record in artifact["development_cohorts"].values() for side in ("fit_support","selection_support") for v in record[side].values()):
        artifact.update({"selected":{"name":"unrefined_identity","enabled":False},"candidates":[],
            "disabled_reason":"every nominal class must occur in both independent development halves of every cohort"})
        return artifact
    for family,m in artifact["mapping"].items():
        rows=fit_mask & (np.asarray(pair["ref"]["family_true"],dtype=str)==family)
        k=len(m["known_leaf_columns"])
        record={"fit_rows":int(rows.sum()),"known_classes":k}
        for name,raw,cols in (("neural",pair["raw_n"],m["known_leaf_columns"]),("statistical",pair["raw_s"],m["stat_model_columns"])):
            moments={cohort:float(np.mean(_center(raw[rows & (cohorts==cohort)][:,cols])**2))
                if k>1 and np.any(rows & (cohorts==cohort)) else 0. for cohort in weights}
            scale=float(np.sqrt(sum(weights[cohort]*moments[cohort] for cohort in weights)))
            record[name+"_scale"]=scale if scale>1e-8 else 1.
            record[name+"_degenerate_or_single_class"]=scale<=1e-8
            record[name+"_cohort_mean_square"]=moments
        record["fit_rows_by_cohort"]={cohort:int(np.sum(rows & (cohorts==cohort))) for cohort in weights}
        artifact["normalization"][family]=record
    candidates=[{"name":"unrefined_identity","enabled":False,"w_stat":None,
        "metrics":_cohort_metrics(pair,pair["base"],selection_mask,cohorts,weights)}]
    for weight in WEIGHTS:
        candidate={**artifact,"enabled":True,"w_stat":weight}
        probability=apply(candidate,pair["q"],pair["base"],pair["raw_n"],pair["raw_s"])
        candidates.append({"name":f"known_blend_stat_{weight:g}","enabled":True,"w_stat":weight,
            "metrics":_cohort_metrics(pair,probability,selection_mask,cohorts,weights)})
    # Python max returns the first candidate on exact ties: the identity baseline.
    selected=max(candidates,key=lambda c:(c["metrics"]["nominal18_macro_f1"],-c["metrics"]["joint_leaf_nll"]))
    artifact.update({"enabled":selected["enabled"],"w_stat":selected["w_stat"] if selected["enabled"] else 0.,
        "selected":selected,"candidates":candidates})
    return artifact


def calibrate(artifact,ref_cal_npz,stat_cal_npz):
    """Return known-only temperatures; caller attaches result under 'calibration'."""
    if not artifact["enabled"]:
        return {"enabled":False,"reason":"identity selected; no refinement calibration required",
            "temperatures":{},"calibration_opened":False,"selection_artifact_sha256":_selection_hash(artifact)}
    pair=_pair(ref_cal_npz,stat_cal_npz,"calibration")
    for key in ("family_labels","neural_leaf_labels","stat_model_labels","leaf_families","mapping"):
        if pair["layout"][key]!=artifact[key]: raise ValueError("calibration class layout differs from frozen selection")
    for name in ("reference","statistical"):
        before=artifact["fingerprints"][name];now=pair["fingerprints"][name]
        if before["weights_status"]=="not_verified" or now["weights_status"]=="not_verified":
            raise ValueError("final known calibration requires actual component weight fingerprints in both exports")
        if before["component_weights_sha256"]!=now["component_weights_sha256"]:
            raise ValueError("calibration component model differs from selected development model")
        if before.get("metadata_status")=="not_verified" or now.get("metadata_status")=="not_verified" or not before.get("component_metadata_sha256") or not now.get("component_metadata_sha256"):
            raise ValueError("final known calibration requires actual inference metadata fingerprints in both exports")
        if before["component_metadata_sha256"]!=now["component_metadata_sha256"]:
            raise ValueError("calibration tokenizer, window or inference metadata differs from selected development model")
    identities=_identity_tokens(pair)
    for key,values in identities.items():
        if set(values) & set(artifact["development_identity_hashes"][key]):
            raise ValueError("calibration overlaps development rows, connected groups or content")
    truth=np.asarray(pair["ref"].get("raw_model_true",pair["ref"]["model_true"]),dtype=str)
    result={"enabled":True,"selection_artifact_sha256":_selection_hash(artifact),
        "fit_split":"calibration_trusted_nominal18_only","fingerprints":pair["fingerprints"],
        "fitted_known_rows":int(pair["known_mask"].sum()),"human_or_weak_rows_used":0,
        "development_groups_disjoint":True,"temperatures":{},"metrics_are_in_sample_calibration":True}
    for family,m in artifact["mapping"].items():
        names=[artifact["neural_leaf_labels"][i] for i in m["known_leaf_columns"]]
        mask=pair["known_mask"] & np.isin(truth,names)
        info={"temperature":1.,"rows":int(mask.sum()),"support":{name:int(np.sum(mask & (truth==name))) for name in names}}
        if len(names)<=1:
            info["status"]="single_or_no_known_class_identity"
        elif mask.sum()<50 or len(set(truth[mask]))<2:
            info["status"]="insufficient_trusted_known_calibration_identity"
        else:
            scores=_mixed(artifact,pair["raw_n"],pair["raw_s"],family)[mask]
            yi=np.asarray([names.index(x) for x in truth[mask]])
            def objective(log_t):
                z=scores/np.exp(log_t)
                return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(z)),yi]))
            opt=minimize_scalar(objective,bounds=(np.log(.05),np.log(20.)),method="bounded",options={"xatol":1e-7})
            if not opt.success or not np.isfinite(opt.fun): raise RuntimeError("known temperature fitting failed")
            info.update({"temperature":float(np.exp(opt.x)),"status":"fitted",
                "fit_conditional_known_nll_before":objective(0.),"fit_conditional_known_nll_after":float(opt.fun)})
        result["temperatures"][family]=info
    return result


def _save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def run_checks():
    """Constructed-data checks, including an independent geometric-mean oracle."""
    lf=dict(zip(MODEL_LABELS,MODEL_FAMILIES))
    def fixture(prefix,repeats):
        truth=[name for name in NOMINAL_MODELS for _ in range(repeats)]+["human"]*10+["unverified"]*10
        n=len(truth);rn=np.zeros((n,len(MODEL_LABELS)));stat_labels=sorted(NOMINAL_MODELS,reverse=True)
        rs=np.zeros((n,len(stat_labels)));q=np.full((n,len(FAMILIES)),1/len(FAMILIES));mask=np.asarray([x in NOMINAL_MODELS for x in truth])
        family=np.asarray([lf.get(x,"human" if x=="human" else "unknown") for x in truth])
        y=np.asarray([0 if x=="human" else 1 for x in truth])
        ids=np.asarray([f"{prefix}:row:{i}" for i in range(n)])
        groups=np.asarray([f"{prefix}:prompt:{i//2}" for i in range(n)])
        for i,name in enumerate(truth):
            if mask[i]:
                rn[i,MODEL_LABELS.index(name)]=1.;rs[i,stat_labels.index(name)]=2.
                q[i]=.01;q[i,FAMILIES.index(lf[name])]=.94
        base=np.zeros_like(rn)
        for f in FAMILIES:
            known=[i for i,x in enumerate(MODEL_LABELS) if x in NOMINAL_MODELS and lf[x]==f]
            unknown=[i for i,x in enumerate(MODEL_LABELS) if x not in NOMINAL_MODELS and lf[x]==f]
            column=FAMILIES.index(f)
            if known:
                base[:,known]=q[:,column,None]*.999*softmax(rn[:,known]*10,axis=1)
                base[:,unknown[0]]=q[:,column]*.001
            else: base[:,unknown[0]]=q[:,column]
        common={"ids":ids,"group_ids":groups,"y_ai":y,"family_true":family,
            "model_true":np.asarray(truth),"mask_family":mask,"mask_model":mask}
        ref={**common,"family_labels":np.asarray(FAMILIES),"model_labels":np.asarray(MODEL_LABELS),
            "model_families":np.asarray(MODEL_FAMILIES),"raw_version":rn,"p_family":q,"p_model":base,
            "component_weights_sha256":np.asarray("1"*64),"component_metadata_sha256":np.asarray("4"*64)}
        stat={**common,"family_labels":np.asarray(FAMILIES),"model_labels":np.asarray(stat_labels),
            "raw_model":rs,"component_weights_sha256":np.asarray("2"*64),"component_metadata_sha256":np.asarray("2"*64)}
        return ref,stat
    checks={};ref,stat=fixture("validation",64)
    selected=select_development(ref,stat)
    checks["truly_unrefined_baseline_competes_and_wins_nll_tie"]=not selected["enabled"] and selected["selected"]["name"]=="unrefined_identity" and len(selected["candidates"])==6
    checks["baseline_is_bitwise_identity_and_needs_no_raw_scores"]=np.array_equal(apply(selected,None,ref["p_model"].astype(np.float32),None,None),ref["p_model"].astype(np.float32))
    checks["disabled_baseline_does_not_open_calibration"]=calibrate(selected,None,None)["calibration_opened"] is False
    fit=selected["development_split"]
    checks["all18_labels_in_disjoint_connected_halves"]=fit["fit_selection_components_disjoint"] and min(fit["fit_support"].values())>0 and min(fit["selection_support"].values())>0
    checks["alphabetic_stat18_columns_map_to_the_right_neural_leaves"]=next(c for c in selected["candidates"] if c["w_stat"]==1.)["metrics"]["nominal18_macro_f1"]==1.
    shuffled=np.random.default_rng(SEED).permutation(len(ref["ids"]))
    moved=_take_rows(stat,shuffled)
    again=select_development(ref,moved)
    checks["ID_alignment_handles_statistical_row_reordering"]=again["candidates"]==selected["candidates"] and again["normalization"]==selected["normalization"]
    graph={"ref":{"ids":np.asarray(["0","1","2","3"]),"group_ids":np.asarray(["a","a","b","c"])},
        "stat":{"group_ids":np.asarray(["x","z","z","y"])},"known_mask":np.ones(4,dtype=bool)}
    fit_graph,comp,info=_connected_split(graph)
    checks["cross_export_group_bridges_are_connected"]=comp[0]==comp[1]==comp[2] and info["connected_components"]==2 and fit_graph[0]==fit_graph[1]==fit_graph[2]
    oracle=copy.deepcopy(selected);oracle.update({"enabled":True,"w_stat":.5,"selected":{"name":"constructed_oracle","enabled":True,"w_stat":.5}})
    for record in oracle["normalization"].values(): record.update({"neural_scale":1.,"statistical_scale":1.})
    q=np.tile(np.asarray([.20,.25,.15,.10,.10,.10,.10]),(2,1));base=np.zeros((2,len(MODEL_LABELS)))
    rn=np.zeros_like(base);rs=np.zeros((2,len(oracle["stat_model_labels"])))
    for f,m in oracle["mapping"].items():
        mass=q[:,m["family_column"]]
        if m["known_leaf_columns"]:
            base[:,m["unknown_leaf_columns"][0]]=.4*mass
            base[:,m["known_leaf_columns"]]=(.6*mass/len(m["known_leaf_columns"]))[:,None]
        else: base[:,m["unknown_leaf_columns"][0]]=mass
    g=oracle["mapping"]["GPT"];a=np.asarray([2.,3.,5.,7.,11.]);b=a[::-1]
    rn[:,g["known_leaf_columns"]]=np.log(a);rs[:,g["stat_model_columns"]]=np.log(b)
    frozen=[i for i,x in enumerate(MODEL_LABELS) if x not in NOMINAL_MODELS]
    rn[:,frozen]=1e6  # Never allowed to alter the already computed unknown gate.
    out=apply(oracle,q,base,rn,rs);expected=.15*np.sqrt(a*b)/np.sqrt(a*b).sum()
    checks["independent_geometric_mean_probability_oracle"]=np.allclose(out[:,g["known_leaf_columns"]],expected,atol=2e-16,rtol=0)
    checks["unknown_and_other_entries_exactly_unchanged"]=np.array_equal(out[:,frozen],base[:,frozen])
    checks["every_family_and_total_mass_conserved"]=all(np.allclose(out[:,m["known_leaf_columns"]+m["unknown_leaf_columns"]].sum(1),q[:,m["family_column"]],atol=2e-16,rtol=0) for m in oracle["mapping"].values()) and np.allclose(out.sum(1),1.,atol=3e-16,rtol=0)
    checks["single_known_Qwen_Kimi_are_exact_residuals"]=all(np.allclose(out[:,oracle["mapping"][f]["known_leaf_columns"][0]],.6*q[:,FAMILIES.index(f)],atol=1e-16,rtol=0) for f in ("Qwen","Kimi"))
    endpoints=[]
    for weight,values in ((0.,a),(1.,b)):
        endpoint={**oracle,"w_stat":weight};p=apply(endpoint,q,base,rn,rs)
        endpoints.append(np.allclose(p[:,g["known_leaf_columns"]],.15*values/values.sum(),atol=2e-16,rtol=0))
    checks["raw_head_weight_endpoints_match_hand_oracle"]=all(endpoints)
    varied_n=rn.copy();varied_s=rs.copy()
    varied_n[1,g["known_leaf_columns"]]+=[.7,-.2,2.,-.9,.4]
    varied_s[1,g["stat_model_columns"]]+=[-.3,1.,-.2,.4,-.8]
    together=apply(oracle,q,base,varied_n,varied_s)
    alone=np.concatenate([apply(oracle,q[i:i+1],base[i:i+1],varied_n[i:i+1],varied_s[i:i+1]) for i in range(2)])
    reverse=apply(oracle,q[::-1],base[::-1],varied_n[::-1],varied_s[::-1])[::-1]
    checks["fixed_scales_make_refinement_batch_invariant"]=np.array_equal(together,alone) and np.array_equal(together,reverse)
    extreme=rn.copy();extreme[:,g["known_leaf_columns"]]=[-1e6,-1e6,-1e6,-1e6,1e6]
    checks["underflow_does_not_create_negative_mass"]=bool((apply({**oracle,"w_stat":0.},q,base,extreme,rs)>=0).all())
    try:
        bad=q.copy();bad[:,0]+=.01;bad[:,1]-=.01;apply(oracle,bad,base,rn,rs)
        checks["incoherent_family_leaf_inputs_rejected"]=False
    except ValueError: checks["incoherent_family_leaf_inputs_rejected"]=True
    cr,cs=fixture("calibration",32);before=copy.deepcopy(oracle);cal=calibrate(oracle,cr,cs)
    checks["calibration_does_not_mutate_selected_weights_or_scales"]=oracle==before
    checks["temperatures_use_trusted_known_rows_only"]=cal["fitted_known_rows"]==18*32 and cal["human_or_weak_rows_used"]==0
    checks["single_known_temperatures_remain_one"]=all(cal["temperatures"][f]["temperature"]==1 for f in ("Qwen","Kimi"))
    changed_ref=copy.deepcopy(cr);changed_stat=copy.deepcopy(cs)
    ignored=~cr["mask_model"];changed_ref["raw_version"][ignored]=4321.;changed_stat["raw_model"][ignored]=-4321.
    cal2=calibrate(oracle,changed_ref,changed_stat)
    checks["human_and_weak_logits_cannot_influence_known_temperatures"]=cal["temperatures"]==cal2["temperatures"]
    attached={**oracle,"calibration":cal};calout=apply(attached,q,base,rn,rs)
    checks["calibrated_refinement_preserves_unknown_and_family_mass"]=np.array_equal(calout[:,frozen],base[:,frozen]) and all(np.allclose(calout[:,m["known_leaf_columns"]+m["unknown_leaf_columns"]].sum(1),q[:,m["family_column"]],atol=2e-16,rtol=0) for m in oracle["mapping"].values())
    for name,fn in (
        ("calibration_development_overlap_rejected",lambda:calibrate(oracle,ref,stat)),
        ("mismatched_component_fingerprint_rejected",lambda:calibrate(oracle,cr,{**cs,"component_weights_sha256":np.asarray("3"*64)})),
        ("same_weights_changed_neural_metadata_rejected",lambda:calibrate(oracle,{**cr,"component_metadata_sha256":np.asarray("6"*64)},cs)),
        ("same_weights_changed_statistical_metadata_rejected",lambda:calibrate(oracle,cr,{**cs,"component_metadata_sha256":np.asarray("6"*64)})),
        ("unverified_component_fingerprint_rejected",lambda:calibrate({**oracle,"fingerprints":{**oracle["fingerprints"],"reference":{**oracle["fingerprints"]["reference"],"weights_status":"not_verified"}}},cr,cs)),
        ("temperature_selection_hash_mismatch_rejected",lambda:apply({**attached,"w_stat":.75},q,base,rn,rs)),
        ("duplicate_row_ids_rejected",lambda:select_development(ref,{**stat,"ids":np.repeat("duplicate",len(stat["ids"]))})),
        ("final_selection_path_rejected_before_open",lambda:select_development("/nonexistent/test_predictions.npz","/nonexistent/test_stat.npz")),
        ("calibration_selection_path_rejected_before_open",lambda:select_development("/nonexistent/calibration_predictions.npz","/nonexistent/calibration_stat.npz")),
        ("sealed_development_name_refused_before_open",lambda:select_development("/nonexistent/dev_sealed_predictions.npz","/nonexistent/dev_stat.npz")),
        ("modern_challenge_development_name_refused_before_open",lambda:select_development("/nonexistent/dev_challenge_modern_predictions.npz","/nonexistent/dev_stat.npz")),
    ):
        try: fn();checks[name]=False
        except ValueError: checks[name]=True
    import tempfile
    with tempfile.TemporaryDirectory(prefix="version-refinement-oracle-") as directory:
        rp=Path(directory)/"validation_reference.npz";sp=Path(directory)/"validation_statistical.npz"
        no_identity={"component_weights_sha256","component_metadata_sha256"}
        np.savez_compressed(rp,**{k:v for k,v in ref.items() if k not in no_identity})
        np.savez_compressed(sp,**{k:v for k,v in stat.items() if k not in no_identity})
        unverified=select_development(rp,sp)
        checks["legacy_exports_explicitly_mark_model_fingerprint_unverified"]=all(v["weights_status"]=="not_verified" and v["metadata_status"]=="not_verified" for v in unverified["fingerprints"].values())
        for path,weight,metadata in ((rp,"1"*64,"4"*64),(sp,"2"*64,"2"*64)):
            _save(path.with_suffix(".provenance.json"),{"purpose":"development","output_sha256":_file_sha(path),"weights_sha256":weight,"component_metadata_sha256":metadata})
        bound=select_development(rp,sp)
        checks["hash_bound_source_sidecar_supplies_model_fingerprint"]=all(v["weights_status"]!="not_verified" and v["weights_fingerprint_source"]=="hash-bound source sidecar" and v["metadata_status"]!="not_verified" and v["metadata_fingerprint_source"]=="hash-bound source sidecar" for v in bound["fingerprints"].values())
        _save(rp.with_suffix(".provenance.json"),{"purpose":"calibration","output_sha256":_file_sha(rp),"weights_sha256":"1"*64})
        try: select_development(rp,sp);checks["calibration_sidecar_refused_before_prediction_read"]=False
        except ValueError: checks["calibration_sidecar_refused_before_prediction_read"]=True
    smoke=_integration_smoke(fixture)
    checks.update(smoke["checks"])
    checks.update(_multi_cohort_checks(fixture))
    if not all(checks.values()): raise AssertionError({k:v for k,v in checks.items() if not v})
    return {"passed":True,"checks":{k:bool(v) for k,v in checks.items()},"constructed_data_only":True,
        "final_inputs_opened":False,"real_calibration_inputs_opened":False,
        "code_sha256":_file_sha(__file__),"constructed_end_to_end":smoke["summary"],
        "oracle":"GPT known mass .15 allocated proportional to sqrt([2,3,5,7,11]*[11,7,5,3,2]); unknown .10 and Other .10 fixed"}


def _multi_cohort_checks(fixture):
    """Unequal cohort sizes, global group bridges and config drift oracles."""
    a,b=fixture("mainval",256),fixture("official_generic",32)
    def combine(part):
        result={}
        for key,value in a[part].items():
            other=b[part][key]
            if key.startswith("component_"):
                result[key]=np.r_[np.repeat(value.item(),len(a[part]["ids"])),np.repeat(other.item(),len(b[part]["ids"]))]
            elif key in _ROW_KEYS: result[key]=np.concatenate((value,other))
            else:
                if not np.array_equal(value,other): raise AssertionError("fixture header mismatch")
                result[key]=value.copy()
        result["development_cohort"]=np.r_[np.full(len(a[part]["ids"]),"mainval"),np.full(len(b[part]["ids"]),"official_generic")]
        return result
    ref,stat=combine(0),combine(1)
    cohort=ref["development_cohort"];official=cohort=="official_generic"
    # A large scale shift in the much smaller cohort must receive half mass.
    ref["raw_version"][official]*=9.;stat["raw_model"][official]*=3.
    # A cross-cohort prompt bridge is one connected component globally.
    for x in (ref,stat): x["group_ids"][[0,len(a[0]["ids"])]]="shared-cross-cohort-prompt"
    weights={"mainval":.5,"official_generic":.5}
    artifact=select_development(ref,stat,cohort_weights=weights)
    pair=_pair(ref,stat,"development");fit,components,_=_connected_split(pair)
    checks={}
    checks["two_cohorts_have_fixed_equal_mass_despite_8x_size_difference"]=artifact["cohort_weights"]==weights and artifact["development_cohorts"]["mainval"]["total_rows"]>7*artifact["development_cohorts"]["official_generic"]["total_rows"]
    checks["cross_cohort_connected_groups_never_cross_fit_selection"]=components[0]==components[len(a[0]["ids"])] and fit[0]==fit[len(a[0]["ids"]) ]
    g=artifact["mapping"]["GPT"];rows=fit & pair["known_mask"] & (ref["family_true"]=="GPT")
    moments=[np.mean(_center(ref["raw_version"][rows & (cohort==name)][:,g["known_leaf_columns"]])**2) for name in weights]
    expected=float(np.sqrt(np.mean(moments)))
    row_rms=float(np.sqrt(np.mean(_center(ref["raw_version"][rows][:,g["known_leaf_columns"]])**2)))
    checks["normalization_uses_equal_cohort_second_moments_not_pooled_rows"]=np.isclose(artifact["normalization"]["GPT"]["neural_scale"],expected,atol=1e-14) and not np.isclose(expected,row_rms)
    checks["each_candidate_reports_cohort_metrics_and_fixed_weighted_quality"]=all(
        set(c["metrics"]["cohorts"])==set(weights) and all(np.isclose(c["metrics"][key],sum(weights[name]*c["metrics"]["cohorts"][name][key] for name in weights),atol=1e-15) for key in ("nominal18_macro_f1","joint_leaf_nll")) for c in artifact["candidates"])
    checks["multi_cohort_retains_true_identity_candidate"]=artifact["candidates"][0]["name"]=="unrefined_identity" and np.array_equal(apply({**artifact,"enabled":False},None,ref["p_model"],None,None),ref["p_model"])
    changed=copy.deepcopy(ref);changed["raw_version"][~fit]*=19.
    alternative=select_development(changed,stat,cohort_weights=weights)
    checks["selection_cohort_logits_do_not_change_fitted_RMS"]=alternative["normalization"]==artifact["normalization"]
    checks["homogeneous_per_row_weights_and_metadata_are_verified"]=all(x["weights_identity_scope"]=="per_row_homogeneous" and x["metadata_identity_scope"]=="per_row_homogeneous" for x in artifact["fingerprints"].values())
    for key in ("component_weights_sha256","component_metadata_sha256"):
        different=copy.deepcopy(stat);different[key][official]="7"*64
        try: select_development(ref,different,cohort_weights=weights);checks["cross_cohort_"+key+"_drift_rejected"]=False
        except ValueError: checks["cross_cohort_"+key+"_drift_rejected"]=True
    for name,kwargs in (("explicit_cohort_mismatch",{"development_cohort":np.full(len(cohort),"wrong"),"cohort_weights":weights}),
            ("missing_cohort_weight",{"cohort_weights":{"mainval":1.}}),
            ("cohort_weight_sum_not_one",{"cohort_weights":{"mainval":.5,"official_generic":.6}})):
        try: select_development(ref,stat,**kwargs);checks[name+"_rejected"]=False
        except ValueError: checks[name+"_rejected"]=True
    return checks


def _integration_smoke(fixture):
    """Exercise root finalization on generated arrays, without loading any model."""
    import tempfile
    from unittest.mock import patch
    from experiments.english_v1 import finalize
    from experiments.english_v1.calibration import HierarchicalCalibration

    def raw_fixture(prefix,repeats):
        r,s=fixture(prefix,repeats)
        n_known=18*repeats
        # Six hundred constructed humans support the 1% FPR calibration rule.
        take=np.r_[np.arange(n_known),np.repeat(n_known,600)]
        r=_take_rows(r,take)
        s=_take_rows(s,take)
        n=len(take)
        ids=np.asarray([f"{prefix}:integration:{i}" for i in range(n)])
        groups=np.asarray([f"{prefix}:integration_group:{i//2}" for i in range(n)])
        for a in (r,s):
            a["ids"],a["group_ids"]=ids,groups
            a["lengths"]=np.full(n,200)
            a["language_supported"]=np.ones(n,dtype=bool)
            a["scope_word_like_tokens"]=a["lengths"].copy()
            a["language_rule_sha256"]=np.asarray(_file_sha(Path(__file__).with_name("language_guard.py")))
            a["sources"]=np.full(n,"constructed_only")
            a["domains"]=np.full(n,"constructed_only")
        rn=np.zeros((n,len(MODEL_LABELS)));rf=np.zeros((n,len(FAMILIES)));rs=np.zeros_like(s["raw_model"])
        for i,name in enumerate(r["model_true"][:n_known]):
            family=MODEL_FAMILIES[MODEL_LABELS.index(name)]
            siblings=[x for x,f in zip(MODEL_LABELS,MODEL_FAMILIES) if f==family and x in NOMINAL_MODELS]
            chosen=name if i%2==0 or len(siblings)==1 else siblings[(siblings.index(name)+1)%len(siblings)]
            rn[i,MODEL_LABELS.index(chosen)]=2.
            rs[i,list(s["model_labels"]).index(name)]=4.
            rf[i]=-9.;rf[i,FAMILIES.index(family)]=9.
        qb=softmax(rf,axis=1);leaves=np.zeros_like(rn)
        for family in FAMILIES:
            columns=[i for i,f in enumerate(MODEL_FAMILIES) if f==family]
            leaves[:,columns]=qb[:,FAMILIES.index(family),None]*softmax(rn[:,columns],axis=1)
        margin=np.where(r["y_ai"]==1,4.,-4.)
        binary=np.column_stack((-margin/2,margin/2))
        r.update({"raw_version":rn,"raw_family":rf,"raw_binary":binary,"p_family":qb,"p_model":leaves})
        s.update({"raw_model":rs,"raw_binary":binary})
        return r,s

    vr,vs=raw_fixture("validation",64);cr,cs=raw_fixture("calibration",32)
    artifact=select_development(vr,vs)
    if not artifact["enabled"]: raise AssertionError("constructed complementary statistical head should enable refinement")
    blend={"schema_version":1,"fit_split":"development_only","names":["neural","unused_zero"],"mean":[0.,0.],"scale":[1.,1.],
        "selected":{"weights":[1.,0.],"intercept":0.},"calibration_opened":False,"final_test_opened":False,
        "components":{"neural":{"weights_sha256":"1"*64,"metadata_sha256":"4"*64}}}
    captured=[];original_cutoff=finalize.selective_cutoff
    def record_cutoff(confidence,correct,**kwargs):
        value=original_cutoff(confidence,correct,**kwargs)
        captured.append({"confidence":np.asarray(confidence).copy(),"correct":np.asarray(correct).copy(),"result":value})
        return value
    with tempfile.TemporaryDirectory(prefix="version-refinement-e2e-") as directory:
        root=Path(directory);bp=root/"blend.json";ap=root/"version_selection.json";cp=root/"calibration.json"
        rp=root/"calibration_reference.npz";sp=root/"calibration_statistical.npz"
        _save(bp,blend);_save(ap,artifact);np.savez_compressed(rp,**cr);np.savez_compressed(sp,**cs)
        with patch.object(finalize,"selective_cutoff",side_effect=record_cutoff):
            calibrated=finalize.fit_calibration(bp,{"neural":rp,"statistics":sp},"neural",cp,
                version_selection_path=ap,version_statistics_name="statistics")
        frozen=calibrated["version_refinement"]["artifact"]
        hierarchy=HierarchicalCalibration.from_dict(calibrated["hierarchy"])
        base=hierarchy.predict(cr["raw_family"],finalize._versions(cr))
        refined=apply(frozen,base["family"],base["leaf"],cr["raw_version"],cs["raw_model"])
        result=finalize.apply_calibration(blend,calibrated,{"neural":cr,"statistics":cs},"neural")
        mask=cr["mask_model"].astype(bool)
        predicted=np.asarray(MODEL_LABELS)[refined.argmax(1)]
        frozen_columns=[i for i,x in enumerate(MODEL_LABELS) if x not in NOMINAL_MODELS]
        check={
            "E2E_final_calibration_binds_exact_selection_artifact_hash":frozen["calibration"]["selection_artifact_sha256"]==_selection_hash(frozen),
            "E2E_model_cutoff_fitted_from_refined_confidence":len(captured)==2 and np.array_equal(captured[1]["confidence"],refined.max(1)[mask]) and not np.allclose(captured[1]["confidence"],base["leaf"].max(1)[mask]),
            "E2E_model_cutoff_fitted_from_refined_correctness":np.array_equal(captured[1]["correct"],(predicted==cr["model_true"])[mask]) and calibrated["thresholds"]["model"]["accepted_n"]==int(mask.sum()),
            "E2E_apply_uses_refined_leaves_and_model_cutoff":np.array_equal(result["p_model"],refined) and result["model_accept"].sum()==int(mask.sum()),
            "E2E_calibrated_unknown_and_Other_mass_unchanged":np.array_equal(result["p_model"][:,frozen_columns],base["leaf"][:,frozen_columns]),
            "E2E_calibrated_version_conditionals_are_not_stale":all(np.allclose(result["p_version_given_family"][:,[i for i,f in enumerate(MODEL_FAMILIES) if f==family]]*result["p_family"][:,j:j+1],result["p_model"][:,[i for i,f in enumerate(MODEL_FAMILIES) if f==family]],atol=2e-15,rtol=0) for j,family in enumerate(FAMILIES)),
            "E2E_all_family_and_total_mass_conserved":np.allclose(result["p_model"].sum(1),1.,atol=2e-15,rtol=0) and all(np.allclose(result["p_model"][:,[i for i,f in enumerate(MODEL_FAMILIES) if f==family]].sum(1),result["p_family"][:,j],atol=2e-15,rtol=0) for j,family in enumerate(FAMILIES)),
            "E2E_zero_weight_binary_component_needs_no_prediction_arrays":True,
        }
        for name in ("neural","statistics"):
            wrong={"neural":cr,"statistics":cs}
            wrong[name]={**wrong[name],"component_weights_sha256":np.asarray("3"*64)}
            try:
                finalize.apply_calibration(blend,calibrated,wrong,"neural")
                check[f"E2E_same_shape_wrong_{name}_weight_fingerprint_rejected"]=False
            except ValueError:
                check[f"E2E_same_shape_wrong_{name}_weight_fingerprint_rejected"]=True
            wrong[name]={**wrong[name],"component_weights_sha256":cr["component_weights_sha256"] if name=="neural" else cs["component_weights_sha256"],"component_metadata_sha256":np.asarray("6"*64)}
            try:
                finalize.apply_calibration(blend,calibrated,wrong,"neural")
                check[f"E2E_same_weights_wrong_{name}_metadata_fingerprint_rejected"]=False
            except ValueError:
                check[f"E2E_same_weights_wrong_{name}_metadata_fingerprint_rejected"]=True
        summary={"constructed_only":True,"real_models_loaded":False,"validation_rows":len(vr["ids"]),
            "calibration_rows":len(cr["ids"]),"trusted_calibration_known_rows":int(mask.sum()),
            "selected_w_stat":artifact["w_stat"],"model_cutoff":calibrated["thresholds"]["model"]["threshold"],
            "model_assertions_on_constructed_calibration":int(result["model_accept"].sum()),
            "model_cutoff_uses_refined_not_original_neural_probabilities":True}
    return {"checks":check,"summary":summary}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    for name in ("select","calibrate"):
        p=sub.add_parser(name);p.add_argument("--reference",required=True);p.add_argument("--statistical",required=True);p.add_argument("--output",required=True)
        if name=="calibrate": p.add_argument("--artifact",required=True)
        else: p.add_argument("--cohort-weights",help='fixed JSON object, e.g. {"mainval":0.5,"official_generic":0.5}; cohort vector comes from NPZ')
    p=sub.add_parser("check");p.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="select": value=select_development(args.reference,args.statistical,cohort_weights=json.loads(args.cohort_weights) if args.cohort_weights else None)
    elif args.command=="calibrate": value=calibrate(json.loads(Path(args.artifact).read_text()),args.reference,args.statistical)
    else: value=run_checks()
    _save(args.output,value)
    print(json.dumps({"command":args.command,"output":str(Path(args.output).resolve()),"enabled":value.get("enabled"),"passed":value.get("passed")}))


if __name__=="__main__": main()
