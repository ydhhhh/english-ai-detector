"""Fit frozen-candidate calibration and apply it to aligned raw predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from sklearn.metrics import f1_score
from experiments.english_v1.calibration import (
    BinaryPlatt, HierarchicalCalibration, prediction_fingerprint, selective_cutoff, save_calibration,
)
from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS, MODEL_FAMILIES, NOMINAL_MODELS, VERSION_INDICES
from experiments.english_v1.evaluate import fit_ai_threshold, evaluate_npz_payload, _binary_metrics, deployed_decision_masks
from experiments.english_v1.select_binary import blend_margin

PROJECT = Path(__file__).resolve().parents[2]
DECISION_PROTOCOL = PROJECT / "research/english_v1_data/deployment_decision_amendment_07.json"
DECISION_PROTOCOL_SHA256 = "0e335238cb5b66c77ef633cc9c09494357e23c4cb986651f10887df2bfdaa0fb"


def _decision_protocol_sha256():
    value=hashlib.sha256(DECISION_PROTOCOL.read_bytes()).hexdigest()
    if value!=DECISION_PROTOCOL_SHA256:raise ValueError("frozen strict-majority decision protocol changed")
    return value


def _majority_floor(artifact,y,p):
    """Preserve the pure empirical oracle and record the stricter deploy cut."""
    value=dict(artifact)
    value["original_empirical_threshold"]=float(artifact["threshold"])
    value["original_empirical_calibration_metrics"]=dict(artifact["calibration_metrics"])
    value["nominal_probability_floor"]=.5
    value["actual_probability_floor"]=float(np.nextafter(.5,np.inf))
    value["threshold"]=max(value["original_empirical_threshold"],value["actual_probability_floor"])
    value["tie_policy"]="p_AI==0.5 always inconclusive; strict posterior majority required"
    value["threshold_adjustment"]="max(original empirical cutoff,nextafter(0.5,+infinity))"
    value["calibration_metrics"]=_binary_metrics(np.asarray(y),np.asarray(p),value["threshold"],np.ones(len(y)))
    value["effective_calibration_metrics"]=dict(value["calibration_metrics"])
    return value


def load_aligned(named_paths, reference_name):
    arrays={name:dict(np.load(path,allow_pickle=False)) for name,path in named_paths.items()}
    ref=arrays[reference_name]
    for name,a in arrays.items():
        for key in ("ids","group_ids","y_ai","lengths"):
            if not np.array_equal(a[key],ref[key]): raise ValueError(f"unaligned {key}: {name}")
        if not np.array_equal(_language_support(a),_language_support(ref)):
            raise ValueError(f"unaligned language support: {name}")
        if "scope_word_like_tokens" in a and not np.array_equal(a["scope_word_like_tokens"],a["lengths"]):
            raise ValueError(f"language-guard and inference word-like lengths disagree: {name}")
    if list(ref["family_labels"])!=FAMILIES or list(ref["model_labels"])!=MODEL_LABELS:
        raise ValueError("hierarchical attribution catalog mismatch")
    return arrays,ref


def _language_support(payload):
    n=len(payload["ids"])
    if "language_supported" not in payload:return np.ones(n,dtype=bool)
    value=np.asarray(payload["language_supported"])
    if value.shape!=(n,) or not np.isin(value,[0,1,False,True]).all():
        raise ValueError("aligned boolean language_supported required")
    return value.astype(bool)


def _language_identity(arrays,needed,*,required=False,expected=None):
    """Legacy constructed arrays may omit scope; final artifacts may not."""
    if set(needed)-set(arrays):raise ValueError("missing required component language-scope evidence")
    explicit=all("language_supported" in arrays[name] and "scope_word_like_tokens" in arrays[name] and "language_rule_sha256" in arrays[name] for name in needed)
    hashes=[];support=None
    for name in needed:
        a=arrays[name]
        current=_language_support(a)
        if support is not None and not np.array_equal(support,current):raise ValueError("components have different language support for aligned rows")
        support=current
        if "scope_word_like_tokens" in a and not np.array_equal(a["scope_word_like_tokens"],a["lengths"]):raise ValueError("language-guard and inference word-like lengths disagree")
        if "language_rule_sha256" in a:
            value=np.asarray(a["language_rule_sha256"])
            if value.ndim!=0 or not re.fullmatch(r"[0-9a-f]{64}",str(value.item())):raise ValueError("scalar language guard SHA256 required")
            hashes.append(str(value.item()))
    if len(set(hashes))>1:raise ValueError("components used different language-guard rules")
    if required and (not explicit or not hashes):raise ValueError("formal calibration/evaluation requires explicit fingerprinted language scope")
    if expected is not None and (not explicit or hashes[0]!=expected):raise ValueError("evaluation language rule differs from fitted calibration")
    return {"status":"explicit_verified" if explicit else "not_verified",
        "language_rule_sha256":hashes[0] if explicit else None,
        "legacy_missing_scope_assumption":"all supported for constructed compatibility only" if not explicit else None}


def hierarchy_from_reference(ref):
    return HierarchicalCalibration(FAMILIES,{f:[MODEL_LABELS[j] for j in VERSION_INDICES[f]] for f in FAMILIES})


def _versions(ref):
    return {f:ref["raw_version"][:,VERSION_INDICES[f]] for f in FAMILIES}


def _required_components(blend,attribution_name,refinement=None):
    needed={n for n,w in zip(blend["names"],blend["selected"]["weights"]) if w!=0}
    needed.add(attribution_name)
    if refinement:needed.add(refinement["statistical_model_name"])
    return needed


def _binary_inputs(blend,arrays,n):
    output={}
    for name,w in zip(blend["names"],blend["selected"]["weights"]):
        if name in arrays:output[name]=arrays[name]["raw_binary"]
        elif w==0:output[name]=np.zeros((n,2))
        else:raise ValueError(f"missing nonzero binary component: {name}")
    return output


def fit_calibration(blend_path,named_paths,attribution_name,output,*,version_selection_path=None,version_statistics_name=None):
    decision_protocol_sha=_decision_protocol_sha256()
    for path in named_paths.values():
        if "calibration" not in Path(path).stem.casefold():
            raise ValueError("fitting requires explicitly named calibration prediction files")
    blend=json.loads(Path(blend_path).read_text())
    if blend.get("fit_split")!="development_only" or blend.get("calibration_opened") or blend.get("final_test_opened"):
        raise ValueError("blend must have been selected without calibration or final test")
    arrays,ref=load_aligned(named_paths,attribution_name)
    needed=_required_components(blend,attribution_name)
    if version_selection_path and json.loads(Path(version_selection_path).read_text()).get("enabled"):
        if not version_statistics_name:raise ValueError("version statistical model name required")
        needed.add(version_statistics_name)
    for name in needed:
        for field in ("component_weights_sha256","component_metadata_sha256"):
            if name not in arrays or field not in arrays[name] or not re.fullmatch(r"[0-9a-f]{64}",str(arrays[name][field].item())):
                raise ValueError(f"calibration requires fingerprinted raw predictions: {name}/{field}")
    if attribution_name not in blend.get("components",{}):
        raise ValueError("attribution model must have a frozen development component fingerprint")
    bound={n for n,w in zip(blend["names"],blend["selected"]["weights"]) if w!=0}
    bound.add(attribution_name)
    for name in bound:
        identity=blend.get("components",{}).get(name)
        if not identity:raise ValueError(f"blend must bind its selected development component fingerprints: {name}")
        for key,field in (("weights_sha256","component_weights_sha256"),("metadata_sha256","component_metadata_sha256")):
            if field not in arrays[name] or str(arrays[name][field].item())!=identity.get(key):
                raise ValueError(f"calibration differs from development-selected component: {name}/{key}")
    language_identity=_language_identity(arrays,needed)
    margin=blend_margin(blend,_binary_inputs(blend,arrays,len(ref["ids"])))
    binary=BinaryPlatt().fit(margin,ref["y_ai"])
    hierarchy=hierarchy_from_reference(ref).fit(ref["raw_family"],_versions(ref),
        ref["family_true"],ref["model_true"],ref["mask_family"],ref["mask_model"])
    p_ai=binary.predict(margin); p=hierarchy.predict(ref["raw_family"],_versions(ref))
    refinement=None
    if version_selection_path:
        from experiments.english_v1 import version_refinement
        selected=json.loads(Path(version_selection_path).read_text())
        if selected.get("enabled"):
            if not version_statistics_name or version_statistics_name not in arrays:
                raise ValueError("version refinement requires its statistical raw calibration predictions")
            selected["calibration"]=version_refinement.calibrate(selected,ref,arrays[version_statistics_name])
            p["leaf"]=version_refinement.apply(selected,p["family"],p["leaf"],ref["raw_version"],arrays[version_statistics_name]["raw_model"])
            refinement={"statistical_model_name":version_statistics_name,"artifact":selected}
    core=ref["lengths"]>=128
    if not core.any(): raise ValueError("no core-length calibration documents")
    cal_id=prediction_fingerprint(ref["ids"].tolist())
    ai=_majority_floor(fit_ai_threshold(ref["y_ai"][core],p_ai[core],target_fpr=.01,calibration_id=cal_id),ref["y_ai"][core],p_ai[core])
    human=_majority_floor(fit_ai_threshold(1-ref["y_ai"][core],1-p_ai[core],target_fpr=.01,calibration_id=cal_id),1-ref["y_ai"][core],1-p_ai[core])
    human["task"]="human_detection"; human["negative_class"]="AI"
    # Independent one-sided empirical cutoffs retain their own error budgets.
    # The registered strict-majority floor removes the broad overlap possible
    # when both empirical cutoffs are tiny, and rejects the exact posterior tie.
    human["tightened_for_disjoint_AI_region"]=False
    human["decision_regions_exclusive_by_boolean_gate"]=True
    for art in (ai,human): art["minimum_word_like_tokens"]=128
    decisions=deployed_decision_masks(p_ai,ref["lengths"],_language_support(ref),ai,human)
    ai_accepted=decisions["AI"]
    fp=p["family"].argmax(1); mp=p["leaf"].argmax(1)
    family_pred=np.asarray(FAMILIES)[fp]; model_pred=np.asarray(MODEL_LABELS)[mp]
    mf=ref["mask_family"].astype(bool) & core
    family=selective_cutoff(p["family"].max(1)[mf],(family_pred==ref["family_true"])[mf],
        target_accuracy=.85,eligible=(ai_accepted & (family_pred!="Other"))[mf],minimum_n=100)
    family["cohort"]="core-length trusted AI calibration including true Other-family negatives; assertions require supported English and exclusive AI region"
    family["calibration_id"]=cal_id
    fa=ai_accepted & (family_pred!="Other") & (p["family"].max(1)>=family["threshold"])
    mm=ref["mask_model"].astype(bool) & core
    consistent=np.asarray(MODEL_FAMILIES)[mp]==family_pred
    model=selective_cutoff(p["leaf"].max(1)[mm],(model_pred==ref["model_true"])[mm],target_accuracy=.75,
        eligible=(fa & np.isin(model_pred,NOMINAL_MODELS) & consistent)[mm],minimum_n=100)
    model["cohort"]="core-length trusted 25-leaf AI calibration including true Other-version negatives; assertions require supported English and exclusive AI region"
    model["calibration_id"]=cal_id
    provenance={"calibration_id":cal_id,"rows":len(ref["ids"]),"core_rows":int(core.sum()),
        "attribution_model":attribution_name,
        "component_weights_sha256":{name:str(a["component_weights_sha256"].item()) for name,a in arrays.items() if "component_weights_sha256" in a},
        "component_metadata_sha256":{name:str(a["component_metadata_sha256"].item()) for name,a in arrays.items() if "component_metadata_sha256" in a},
        "blend_sha256":hashlib.sha256(Path(blend_path).read_bytes()).hexdigest(),
        "prediction_files":{name:{"path":str(Path(path).resolve()),"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()} for name,path in named_paths.items()},
        "final_test_opened":False,"selection_after_calibration":False,
        "selection_cutoff_caveat":"Wilson bound describes calibration only; threshold selection and domain shift preclude a population accuracy guarantee"}
    provenance["language_scope"]=language_identity
    provenance["language_rule_sha256"]=language_identity["language_rule_sha256"]
    provenance["deployment_decision_protocol_sha256"]=decision_protocol_sha
    provenance["deployment_decision_protocol_path"]=str(DECISION_PROTOCOL.relative_to(PROJECT))
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    value=save_calibration(output,binary=binary,hierarchy=hierarchy,ai_threshold=ai,human_threshold=human,
        family_threshold=family,model_threshold=model,provenance=provenance)
    if refinement:
        value["version_refinement"]=refinement
        value["provenance"]["version_selection_sha256"]=hashlib.sha256(Path(version_selection_path).read_bytes()).hexdigest()
        Path(output).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    return value


def apply_calibration(blend,calibration,arrays,attribution_name):
    ref=arrays[attribution_name]
    if calibration["provenance"].get("deployment_decision_protocol_sha256"):
        if calibration["provenance"]["deployment_decision_protocol_sha256"]!=_decision_protocol_sha256():raise ValueError("calibration decision protocol differs from frozen amendment07")
        if any(not np.isfinite(calibration["thresholds"][name]["threshold"]) or calibration["thresholds"][name]["threshold"]<=.5 for name in ("AI","human")):raise ValueError("registered deploy thresholds must require strict probability majority")
    if calibration["provenance"]["attribution_model"]!=attribution_name:raise ValueError("attribution calibration mismatch")
    if calibration["provenance"].get("language_rule_sha256"):
        _language_identity(arrays,_required_components(blend,attribution_name,calibration.get("version_refinement")),required=True,expected=calibration["provenance"]["language_rule_sha256"])
    for name in _required_components(blend,attribution_name,calibration.get("version_refinement")):
        expected=calibration["provenance"].get("component_weights_sha256",{}).get(name)
        observed=arrays[name].get("component_weights_sha256") if name in arrays else None
        if not expected or observed is None or str(observed.item())!=expected:
            raise ValueError(f"raw prediction component differs from fitted calibration: {name}")
        config_expected=calibration["provenance"].get("component_metadata_sha256",{}).get(name)
        if not config_expected or "component_metadata_sha256" not in arrays[name] or str(arrays[name]["component_metadata_sha256"].item())!=config_expected:
            raise ValueError(f"raw inference configuration differs from fitted calibration: {name}")
    binary=BinaryPlatt(**calibration["binary"])
    hierarchy=HierarchicalCalibration.from_dict(calibration["hierarchy"])
    margin=blend_margin(blend,_binary_inputs(blend,arrays,len(ref["ids"])))
    result=dict(ref)
    result["p_ai"]=binary.predict(margin)
    result["p_binary"]=np.stack([1-result["p_ai"],result["p_ai"]],axis=1)
    p=hierarchy.predict(ref["raw_family"],_versions(ref)); result["p_family"]=p["family"];result["p_model"]=p["leaf"]
    if calibration.get("version_refinement"):
        from experiments.english_v1 import version_refinement
        vr=calibration["version_refinement"]
        result["p_model"]=version_refinement.apply(vr["artifact"],p["family"],p["leaf"],ref["raw_version"],arrays[vr["statistical_model_name"]]["raw_model"])
        p["leaf"]=result["p_model"]
    within=np.empty_like(p["leaf"])
    for j,f in enumerate(FAMILIES):
        indices=VERSION_INDICES[f]
        within[:,indices]=np.divide(p["leaf"][:,indices],p["family"][:,j:j+1],
            out=p["within_family"][f].copy(),where=p["family"][:,j:j+1]>0)
    result["p_version_given_family"]=within
    fp=p["family"].argmax(1);mp=p["leaf"].argmax(1)
    fa=np.asarray(FAMILIES)[fp]; ma=np.asarray(MODEL_LABELS)[mp]
    tt=calibration["thresholds"]
    language=_language_support(ref)
    result["language_supported"]=language
    decisions=deployed_decision_masks(result["p_ai"],ref["lengths"],language,tt["AI"],tt["human"])
    ai=decisions["AI"]
    for key in ("AI","human","inconclusive"):result["decision_"+key]=decisions[key]
    result["family_accept"]=ai & (fa!="Other") & (p["family"].max(1)>=tt["family"]["threshold"])
    result["model_accept"]=result["family_accept"] & np.isin(ma,NOMINAL_MODELS) & (np.asarray(MODEL_FAMILIES)[mp]==fa) & (p["leaf"].max(1)>=tt["model"]["threshold"])
    result["calibrated"]=np.asarray(True)
    return result


def apply_source_release(result,source_state):
    """Disable assertions only; preserve candidate masks and every probability."""
    value=dict(result)
    family=bool(source_state.get("valid") is True and source_state.get("family_enabled") is True)
    model=bool(family and source_state.get("model_enabled") is True)
    for key,enabled in (("family_accept",family),("model_accept",model)):
        value["candidate_"+key]=np.asarray(result[key],dtype=bool).copy()
        value[key]=value["candidate_"+key] & enabled
    value["source_release_family_enabled"]=np.asarray(family)
    value["source_release_model_enabled"]=np.asarray(model)
    return value


def evaluate_frozen(blend_path,calibration_path,named_paths,attribution_name,output,n_boot=500,*,source_gate_path=None):
    blend=json.loads(Path(blend_path).read_text());cal=json.loads(Path(calibration_path).read_text())
    if cal["provenance"].get("deployment_decision_protocol_sha256")!=_decision_protocol_sha256():raise ValueError("formal evaluation requires the frozen amendment07 decision protocol")
    if cal["provenance"]["blend_sha256"]!=hashlib.sha256(Path(blend_path).read_bytes()).hexdigest():
        raise ValueError("evaluation blend differs from frozen probability calibration")
    language_sha=cal["provenance"].get("language_rule_sha256")
    current_language_sha=hashlib.sha256(Path(__file__).with_name("language_guard.py").read_bytes()).hexdigest()
    if not language_sha or language_sha!=current_language_sha:
        raise ValueError("formal evaluation requires calibration bound to the current language-guard rule")
    source_state=None
    if source_gate_path:
        from experiments.english_v1.source_release_gate import load_gate,PROTOCOL_SHA256
        needed=_required_components(blend,attribution_name,cal.get("version_refinement"))
        expected={"blend_sha256":hashlib.sha256(Path(blend_path).read_bytes()).hexdigest(),
            "calibration_sha256":hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),
            "language_rule_sha256":language_sha,
            "deployment_decision_protocol_sha256":cal["provenance"]["deployment_decision_protocol_sha256"],
            "source_assertion_release_protocol_sha256":PROTOCOL_SHA256,
            "components":{name:{"weights_sha256":cal["provenance"]["component_weights_sha256"][name],
                "metadata_sha256":cal["provenance"]["component_metadata_sha256"][name]} for name in needed}}
        source_state=load_gate(source_gate_path,expected=expected)
        if not source_state["valid"]:raise ValueError("release evaluation requires a valid matching source gate: "+"; ".join(source_state["errors"]))
    arrays,ref=load_aligned(named_paths,attribution_name)
    _language_identity(arrays,_required_components(blend,attribution_name,cal.get("version_refinement")),required=True,expected=language_sha)
    result=apply_calibration(blend,cal,arrays,attribution_name)
    candidate_result=result
    if source_state is not None:result=apply_source_release(result,source_state)
    report=evaluate_npz_payload(result,threshold=cal["thresholds"]["AI"],human_threshold=cal["thresholds"]["human"],n_boot=n_boot)
    report["binary_at_probability_half"]={
        "threshold":.5,"threshold_source":"constant default probability cutoff, never fitted",
        "all_lengths":_binary_metrics(result["y_ai"],result["p_ai"],.5,np.ones(len(result["y_ai"]))),
        "core_ge_128_word_like_tokens":_binary_metrics(result["y_ai"],result["p_ai"],.5,(result["lengths"]>=128).astype(float))}
    # Specific-version metrics must not be inflated by easy Other-version labels.
    nominal=dict(result);nominal["mask_model"]=result["mask_model"].astype(bool)&np.isin(result["model_true"],NOMINAL_MODELS)
    nominal["mask_family"]=result["mask_family"].astype(bool)&np.isin(result["family_true"],FAMILIES[:-1])
    report["supported_catalog_attribution"]=evaluate_npz_payload(nominal,threshold=cal["thresholds"]["AI"],human_threshold=cal["thresholds"]["human"],n_boot=0)
    if source_state is not None:
        report["source_assertion_release_gate_applied"]=True
        before=dict(nominal)
        for key in ("family_accept","model_accept"):before[key]=candidate_result[key]
        candidate=evaluate_npz_payload(before,threshold=cal["thresholds"]["AI"],human_threshold=cal["thresholds"]["human"],n_boot=0)
        report["candidate_source_assertions_before_release"]={key:candidate[key] for key in ("all_lengths","core_ge_128_tokens")}
        report["source_release"]={"gate_sha256":source_state["gate_sha256"],
            **{key:source_state[key] for key in ("valid","family_enabled","model_enabled")},
            "final_evidence_used_for_release_switch":True,
            "probabilities_binary_decisions_weights_thresholds_changed":False,
            "interpretation":"Final evidence participates in a preregistered disable-only publication switch; original candidate reports remain separate."}
    catalog_scores={}
    for name,truth,prob,mask,labels,order in (
        ("nominal18",result["model_true"],result["p_model"],nominal["mask_model"],NOMINAL_MODELS,MODEL_LABELS),
        ("supported6families",result["family_true"],result["p_family"],nominal["mask_family"],FAMILIES[:-1],FAMILIES)):
        predicted=np.asarray(order)[prob.argmax(1)]
        for subset,m in (("all_lengths",mask),("core_ge_128_word_like_tokens",mask & (result["lengths"]>=128))):
            catalog_scores[name+"_"+subset]={"n":int(m.sum()),
                "macro_f1_on_named_catalog":float(f1_score(truth[m],predicted[m],labels=labels,average="macro",zero_division=0)) if m.any() else None,
                "accuracy":float(np.mean(truth[m]==predicted[m])) if m.any() else None,
                "prediction_space":list(order),"macro_f1_label_set":list(labels),
                "note":"Other predictions remain errors; probability columns are not dropped or renormalized."}
    report["catalog_specific_scores"]=catalog_scores
    report["provenance"]={"blend_sha256":hashlib.sha256(Path(blend_path).read_bytes()).hexdigest(),
        "calibration_sha256":hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),
        "deployment_decision_protocol_sha256":cal["provenance"]["deployment_decision_protocol_sha256"],
        "language_rule_sha256":language_sha,
        "inputs":{name:{"path":str(Path(p).resolve()),"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest()} for name,p in named_paths.items()},
        "thresholds_refitted_on_evaluation":False}
    if source_state is not None:report["provenance"]["source_release_gate_sha256"]=source_state["gate_sha256"]
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    np.savez_compressed(out.with_suffix(".npz"),**result)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["calibrate","evaluate"])
    p.add_argument("--blend",required=True);p.add_argument("--calibration")
    p.add_argument("--predictions",nargs="+",required=True,help="name=path pairs")
    p.add_argument("--attribution-model",required=True);p.add_argument("--output",required=True)
    p.add_argument("--bootstrap",type=int,default=500)
    p.add_argument("--version-selection");p.add_argument("--version-statistics-model")
    p.add_argument("--source-release-gate",help="Optional frozen disable-only gate; use separate output files to preserve original candidate evaluation")
    a=p.parse_args();paths=dict(x.split("=",1) for x in a.predictions)
    if a.command=="calibrate":
        value=fit_calibration(a.blend,paths,a.attribution_model,a.output,
            version_selection_path=a.version_selection,version_statistics_name=a.version_statistics_model)
        print(json.dumps(value["thresholds"],indent=2))
    else:
        if not a.calibration:p.error("evaluation requires --calibration")
        value=evaluate_frozen(a.blend,a.calibration,paths,a.attribution_model,a.output,a.bootstrap,source_gate_path=a.source_release_gate)
        print(json.dumps({"evaluation_saved":a.output,"rows":len(np.load(Path(a.output).with_suffix('.npz'))['ids'])}))
