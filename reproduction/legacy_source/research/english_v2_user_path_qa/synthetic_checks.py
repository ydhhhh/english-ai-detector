"""Metadata/math-only checks of the future v2 bundle QA wrapper.

No actual bundle, corpus, score file, checkpoint or GPU is opened. Scalar
probabilities and synthetic IDs below are constructed in memory only.
"""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.english_v1 import runtime  # noqa: F401
import numpy as np
from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS, MODEL_FAMILIES, VERSION_INDICES

P = ROOT/"research/english_v2_user_path_qa/offline_live_bundle_checks.py"
spec = importlib.util.spec_from_file_location("synthetic_v2_offline_live_qa", P)
qa = importlib.util.module_from_spec(spec); spec.loader.exec_module(qa)


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected fail-closed metadata readiness")


def run():
    checks = []
    require = qa.require
    metadata = [{"id": f"synthetic:{i}", "y_ai": i%2, "split": "final_sealed", "eligible": True} for i in range(12)]
    lengths = {r["id"]: 64 if i < 4 else 200+i for i, r in enumerate(metadata)}
    roles, absent = qa.choose_ids(metadata, lengths)
    expected = {}
    for role, label, core in (("core_human",0,True),("core_AI",1,True),("short_human",0,False),("short_AI",1,False)):
        pool = [r["id"] for r in metadata if r["y_ai"] == label and (lengths[r["id"]] >= 128) == core]
        selected = sorted(pool, key=lambda value: (qa.legacy.ordering(value)))[0]
        expected.setdefault(selected, []).append(role)
    longest = max(lengths, key=lengths.get); expected.setdefault(longest, []).append("longest")
    require(roles == expected and absent == [] and qa.choose_ids(metadata[::-1], lengths) == (roles, absent), "pure deterministic fixed hash/length oracle")
    core_only = {key: max(value, 128) for key, value in lengths.items()}
    core_roles, absent = qa.choose_ids(metadata, core_only)
    require(absent == ["short_human", "short_AI"] and len(core_roles) <= 3, "empty short strata must not be fabricated")
    require(qa.POLICY["seed"] == 20261017 and qa.legacy.SEED == 20261017 and len(roles) <= 5, "seed and at-most5 fixed in advance")
    rejected(lambda: qa.choose_ids([{**r, "split": "calibration"} for r in metadata], lengths))
    rejected(lambda: qa.choose_ids([*metadata, metadata[0]], lengths))
    checks.append("Predeclared20261017 hash/length policy, order invariance, <=5 dedup, exact final_sealed roles and absent-short disclosure")
    p_family = np.asarray([[.30,.20,.15,.10,.10,.10,.05]]*2)
    p_model = np.zeros((2,25))
    for i, f in enumerate(FAMILIES):
        p_model[:,VERSION_INDICES[f]] = p_family[:,i,None]/len(VERSION_INDICES[f])
    reference = {"p_ai": np.asarray([.8,.2]), "p_family": p_family, "p_model": p_model,
        "decision_AI":np.asarray([True,False]), "decision_human":np.asarray([False,True]), "decision_inconclusive":np.zeros(2,dtype=bool),
        "family_accept":np.zeros(2,dtype=bool), "model_accept":np.zeros(2,dtype=bool), "lengths":np.full(2,200),
        "encoder_tokens":np.full(2,260), "encoder_coverage":np.ones(2), "window_counts":np.ones(2,dtype=int), "language_supported":np.ones(2,dtype=bool)}
    results=[]
    for i in range(2):
        result = {"probabilities":{"AI":float(reference["p_ai"][i]),"human":float(1-reference["p_ai"][i])},
            "family_probabilities":sorted([{"label":label,"probability":float(p_family[i,j])} for j,label in enumerate(FAMILIES)],key=lambda r:-r["probability"]),
            "model_probabilities":sorted([{"label":label,"family":MODEL_FAMILIES[j],"probability":float(p_model[i,j])} for j,label in enumerate(MODEL_LABELS)],key=lambda r:-r["probability"]),
            "status":"AI-generated" if i==0 else "likely human","accepted_family":None,"accepted_model":None,
            "quality":{"word_like_tokens":200,"core_length_supported":True,"encoder_tokens":260,"encoder_coverage":1.,"window_counts":1,"language":"en"}}
        results.append(result)
    selected = [{"id_sha256":qa.legacy.hash_id("synthetic-selected:"+str(i)),"selection_roles":["constructed"]} for i in range(2)]
    cal = {"thresholds":{key:{"threshold":.7} for key in ("AI","human","family","model")}}
    compare, vectors = qa.legacy.check_live(results, reference, selected, cal, "cpu", 1e-5)
    require(vectors.shape == (2,33) and all(r["probabilities_passed"] and r["decisions_match"] and r["quality_match"] and r["full_token_coverage"] for r in compare), "original pure33 probability/decision/quality comparison")
    drift = copy.deepcopy(results);drift[0]["probabilities"].update(AI=.80006,human=.19994)
    cpu,_ = qa.legacy.check_live(drift,reference,selected,cal,"cpu",1e-5)
    mps,_ = qa.legacy.check_live(drift,reference,selected,cal,"mps",1e-4)
    require(not cpu[0]["probabilities_passed"] and mps[0]["probabilities_passed"], "unchanged device tolerances")
    drift=copy.deepcopy(results);drift[0]["status"]="inconclusive"
    compare,_=qa.legacy.check_live(drift,reference,selected,cal,"cpu",1e-5)
    require(compare[0]["probabilities_passed"] and not compare[0]["decisions_match"] and "decision_difference_disclosure" in compare[0], "decision boundary differences disclosed even with exact probabilities")
    drift=copy.deepcopy(results);drift[0]["quality"]["window_counts"]=2
    compare,_=qa.legacy.check_live(drift,reference,selected,cal,"cpu",1e-5)
    require(not compare[0]["quality_match"], "full-window quality drift fails")
    checks.append("Unchanged original33-dimension comparison, CPU1e-5/MPS1e-4, hierarchy mass, decision-boundary disclosure and full-window quality")
    with patch.object(qa, "prepare", side_effect=AssertionError("unapproved action reached artifact preparation")):
        rejected(lambda: qa.run(SimpleNamespace(root_approved=False)))
    with patch.object(qa.legacy,"idle",side_effect=ValueError("synthetic producer still running")), patch.object(qa,"bound",side_effect=AssertionError("busy GPU must reject before binding artifacts")):
        rejected(lambda:qa.prepare("not-opened", "0"*64, 1))
    with tempfile.TemporaryDirectory(prefix="synthetic-v2-qa-bindings-") as temp:
        p=Path(temp)/"fake_binding.json";p.write_text("{}")
        with patch.object(qa.legacy,"idle",return_value={"recorded_pid_exited":True}):
            rejected(lambda:qa.prepare(p,"0"*64,1))
            rejected(lambda:qa.prepare(p,qa.sha(p),1))
    checks.append("No root approval, busy producer, wrong binding-plan SHA, or incomplete ready plan rejects before model/corpus access")
    require(qa.sha(qa.LEGACY_PATH)==qa.LEGACY_SHA and qa.sha(ROOT/"experiments/english_v1/bundle_checks.py")==qa.BUNDLE_CHECKS_SHA,"legacy helpers unchanged")
    return {"schema_version":1,"passed":True,"synthetic_metadata_and_math_only":True,"real_data_scores_bundle_or_weights_opened":False,
        "model_forward_or_GPU_used":False,"checks":checks,"check_count":len(checks),"qa_code_sha256":qa.sha(P),
        "checks_code_sha256":qa.sha(__file__),"legacy_comparison_helper_sha256":qa.LEGACY_SHA,"original_bundle_checks_sha256":qa.BUNDLE_CHECKS_SHA}


if __name__=="__main__":
    result=run()
    out=Path(__file__).with_name("synthetic_checks_report.json")
    qa.legacy.new_json(out,result)
    print(json.dumps(result,indent=2))
