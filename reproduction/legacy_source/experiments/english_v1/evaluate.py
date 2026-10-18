"""Fixed-calibration evaluation of binary detection and conditional attribution.

Class columns are explicit strings and contain AI sources only, with no human
column. This module never chooses a threshold from evaluation labels. Cluster
bootstrap intervals condition on the fitted models and calibration artifacts.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

EPS = 1e-15


def _vector(value: Any, name: str, n: int | None = None) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != 1 or not a.size or (n is not None and len(a) != n):
        raise ValueError(f"{name} must be a nonempty vector of length {n}")
    return a


def _binary_truth(value: Any) -> np.ndarray:
    a = _vector(value, "y_ai")
    if not np.all(np.isin(a, [0, 1])):
        raise ValueError("y_ai uses human=0 and AI=1 only")
    return a.astype(np.int64)


def _probability(value: Any, n: int) -> np.ndarray:
    a = _vector(value, "p_ai", n).astype(np.float64)
    if not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1)):
        raise ValueError("p_ai must contain finite probabilities in [0, 1]")
    return a


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def _mask(value: Any, name: str, n: int) -> np.ndarray:
    a = _vector(value, name, n)
    if not np.all(np.isin(a, [False, True, 0, 1])):
        raise ValueError(f"{name} must contain booleans, not truthy strings")
    return a.astype(bool)


def _ece(confidence: np.ndarray, target: np.ndarray, w: np.ndarray,
         bins: int = 15) -> float | None:
    if not w.sum():
        return None
    index = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    mass = np.bincount(index, weights=w, minlength=bins)
    delta = np.bincount(index, weights=w * (confidence - target), minlength=bins)
    return float(np.abs(delta[mass > 0]).sum() / w.sum())


def _binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float,
                    w: np.ndarray) -> dict[str, Any]:
    pred = p >= threshold
    positive, negative = y == 1, y == 0
    tp, fp = float(w[positive & pred].sum()), float(w[negative & pred].sum())
    fn, tn = float(w[positive & ~pred].sum()), float(w[negative & ~pred].sum())
    n = float(w.sum())
    tpr, fpr = _ratio(tp, tp + fn), _ratio(fp, fp + tn)
    both = (tp + fn) > 0 and (fp + tn) > 0
    use = w > 0
    return {
        "n": int(n), "human_n": int(tn + fp), "AI_n": int(tp + fn),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": _ratio(tp + tn, n), "tpr": tpr, "fpr": fpr,
        "precision": _ratio(tp, tp + fp), "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (tpr + 1.0 - fpr) / 2.0 if both else None,
        "auroc": float(roc_auc_score(y[use], p[use], sample_weight=w[use])) if both else None,
        "auprc": float(average_precision_score(y[use], p[use], sample_weight=w[use])) if (tp + fn) else None,
        "nll": float(np.dot(w, -(y * np.log(np.clip(p, EPS, 1)) + (1-y) * np.log(np.clip(1-p, EPS, 1)))) / n) if n else None,
        "brier": float(np.dot(w, (p-y)**2) / n) if n else None,
        "ece": _ece(p, y, w),
    }


def fit_ai_threshold(y_ai_cal: Any, p_ai_cal: Any, *, target_fpr: float = 0.01,
                     calibration_id: str | None = None) -> dict[str, Any]:
    """Fit the most permissive cutoff satisfying empirical calibration FPR.

    Decisions use ``p_ai >= threshold``. All scores tied at the disallowed human
    order statistic are excluded together. The cutoff may be nextafter(1,+inf),
    meaning no probability is accepted. This is an empirical operating point,
    not a population false-positive guarantee; human_n determines its resolution.
    """
    y = _binary_truth(y_ai_cal); p = _probability(p_ai_cal, len(y))
    if not np.isfinite(target_fpr) or not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must lie in [0, 1)")
    negatives = np.sort(p[y == 0])
    if not len(negatives):
        raise ValueError("calibration requires human negatives to fit FPR")
    allowed = int(np.floor(target_fpr * len(negatives) + 1e-12))
    boundary = float(negatives[len(negatives) - allowed - 1])
    threshold = float(np.nextafter(boundary, np.inf))
    fingerprint = hashlib.sha256(y.tobytes() + p.tobytes()).hexdigest()
    metrics = _binary_metrics(y, p, threshold, np.ones(len(y)))
    return {
        "task": "AI_detection", "fit_split": "calibration",
        "threshold": threshold, "comparison": ">=", "target_fpr": float(target_fpr),
        "calibration_id": calibration_id, "calibration_prediction_sha256": fingerprint,
        "calibration_human_n": int(len(negatives)), "allowed_false_positives": allowed,
        "empirical_fpr_resolution": 1.0 / len(negatives),
        "calibration_metrics": metrics,
        "population_FPR_guarantee": False,
    }


def fit_acceptance_threshold(confidence_cal: Any, *, target_coverage: float = .90,
                             calibration_id: str | None = None) -> dict[str, Any]:
    """Fit source acceptance on labeled, known AI calibration examples only.

    This controls known-source empirical coverage, not an unknown probability.
    Pass the resulting per-row acceptance mask to evaluate_predictions. Unknown
    rejection quality still requires real generator classes absent from fitting.
    """
    a = _vector(confidence_cal, "confidence_cal").astype(float)
    if not np.all(np.isfinite(a)) or not 0 < target_coverage <= 1:
        raise ValueError("finite scores and target_coverage in (0,1] required")
    index = len(a) - int(np.ceil(target_coverage * len(a)))
    cutoff = float(np.sort(a)[index])
    return {"task": "known_source_acceptance", "fit_split": "calibration",
            "threshold": cutoff, "comparison": ">=", "calibration_id": calibration_id,
            "target_coverage": float(target_coverage),
            "empirical_coverage": float(np.mean(a >= cutoff)),
            "score_is_unknown_probability": False}


def _fixed_threshold(artifact: Mapping[str, Any]) -> float:
    if not isinstance(artifact, Mapping) or artifact.get("fit_split") != "calibration":
        raise ValueError("threshold must be a calibration-fitted artifact")
    if artifact.get("comparison") != ">=":
        raise ValueError("the fixed threshold comparison must be >=")
    t = float(artifact["threshold"])
    if not np.isfinite(t):
        raise ValueError("threshold must be finite")
    return t


def deployed_decision_masks(p_ai: Any, lengths: Any, language_supported: Any,
                            ai_threshold: Mapping[str, Any],
                            human_threshold: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Exact online gate: English + core + mutually exclusive AI/human regions."""
    values=_vector(p_ai,"p_ai");p=_probability(values,len(values));n=len(p)
    ll=_vector(lengths,"lengths",n).astype(float)
    if not np.isfinite(ll).all() or np.any(ll<0) or np.any(ll!=np.floor(ll)):
        raise ValueError("lengths must be nonnegative finite token counts")
    language=np.ones(n,dtype=bool) if language_supported is None else _mask(language_supported,"language_supported",n)
    a=p>=_fixed_threshold(ai_threshold);h=(1-p)>=_fixed_threshold(human_threshold)
    eligible=(ll>=128)&language
    ai=eligible&a&~h;human=eligible&h&~a
    return {"AI":ai,"human":human,"inconclusive":~(ai|human),
        "eligible":eligible,"language_supported":language,"core":ll>=128,
        "overlapping_raw_regions":eligible&a&h}


def _deployed_decision_audit(y,decisions,w):
    if decisions is None:
        return {"status":"not_evaluated","reason":"both a fixed human threshold and original-text lengths are required to reproduce online decisions"}
    n=float(w.sum());human=y==0;ai=y==1
    nh=float(w[human].sum());na=float(w[ai].sum())
    pa=float(w[decisions["AI"]].sum());ph=float(w[decisions["human"]].sum());inc=float(w[decisions["inconclusive"]].sum())
    h_as_a=float(w[human&decisions["AI"]].sum());a_as_h=float(w[ai&decisions["human"]].sum())
    ca=float(w[ai&decisions["AI"]].sum());ch=float(w[human&decisions["human"]].sum())
    return {"status":"evaluated","scope":"Exact online English/core/exclusive AI-human gate; rejected rows remain in every subgroup denominator. Conditional decided accuracy is always accompanied by coverage.",
        "n":int(n),"human_n":int(nh),"AI_n":int(na),
        "language_supported_n":int(w[decisions["language_supported"]].sum()),
        "language_unsupported_n":int(w[~decisions["language_supported"]].sum()),
        "core_ge128_n":int(w[decisions["core"]].sum()),"eligible_English_core_n":int(w[decisions["eligible"]].sum()),
        "predicted_AI_n":int(pa),"predicted_human_n":int(ph),"inconclusive_n":int(inc),
        "AI_decision_coverage":_ratio(pa,n),"human_decision_coverage":_ratio(ph,n),"inconclusive_coverage":_ratio(inc,n),
        "decided_coverage":_ratio(pa+ph,n),"conditional_accuracy_on_decided":_ratio(ca+ch,pa+ph),
        "human_misclassified_as_AI_n":int(h_as_a),"human_misclassified_as_AI_rate":_ratio(h_as_a,nh),
        "AI_misclassified_as_human_n":int(a_as_h),"AI_misclassified_as_human_rate":_ratio(a_as_h,na),
        "correct_AI_n":int(ca),"correct_AI_recall":_ratio(ca,na),"correct_human_n":int(ch),"correct_human_recall":_ratio(ch,nh),
        "inconclusive_human_n":int(w[human&decisions["inconclusive"]].sum()),
        "inconclusive_AI_n":int(w[ai&decisions["inconclusive"]].sum()),
        "overlapping_raw_regions_rejected_n":int(w[decisions["overlapping_raw_regions"]].sum())}


def _head(y: Any, probabilities: Any, classes: Sequence[str] | None, mask: Any,
          accepted: Any, ai: np.ndarray, name: str) -> dict[str, Any] | None:
    if all(x is None for x in (y, probabilities, classes, mask, accepted)):
        return None
    if y is None or probabilities is None or classes is None:
        raise ValueError(f"{name}_y, probabilities and classes are required together")
    names = list(classes)
    if not names or any(not isinstance(s, (str, np.str_)) or not s or s.casefold() == "human" for s in names) or len(set(names)) != len(names):
        raise ValueError(f"{name}_classes must be unique, nonempty strings")
    labels = _vector(y, name+"_y", len(ai))
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(ai), len(names)) or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError(f"{name}_probs must be finite N by K probabilities")
    if not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError(f"{name}_probs rows must sum to one; no silent normalization")
    m = ai.astype(bool) & np.isin(labels, names) if mask is None else _mask(mask, name+"_mask", len(ai))
    if np.any(m & ~ai.astype(bool)) or np.any(m & ~np.isin(labels, names)):
        raise ValueError(f"{name}_mask may only include AI rows with supported labels")
    accept = np.ones(len(ai), dtype=bool) if accepted is None else _mask(accepted, name+"_accept", len(ai))
    lookup = {s: i for i, s in enumerate(names)}
    ids = np.array([lookup.get(v, -1) for v in labels], dtype=np.int64)
    return {"classes": names, "ids": ids, "probs": p, "mask": m, "accept": accept}


def _multiclass(y: np.ndarray, p: np.ndarray, classes: Sequence[str],
                w: np.ndarray) -> dict[str, Any]:
    n, k = float(w.sum()), len(classes)
    if not n:
        return {"n": 0, **{s: None for s in ("accuracy", "balanced_accuracy", "macro_f1", "nll", "brier", "ece", "top1", "top3", "top5")}, "classes": list(classes)}
    use = w > 0; yy = y[use]; pp = p[use]; ww = w[use]
    pred = pp.argmax(axis=1)
    supports = np.bincount(yy, weights=ww, minlength=k)
    corrects = np.bincount(yy[pred == yy], weights=ww[pred == yy], minlength=k)
    present = supports > 0
    rank = np.argsort(-pp, axis=1, kind="stable")
    brier = (pp**2).sum(axis=1) - 2 * pp[np.arange(len(yy)), yy] + 1
    result = {
        "n": int(n), "classes": list(classes),
        "accuracy": float(np.dot(ww, pred == yy) / n),
        "balanced_accuracy": float(np.mean(corrects[present] / supports[present])),
        "macro_f1": float(f1_score(yy, pred, labels=np.arange(k), average="macro", sample_weight=ww, zero_division=0)),
        "nll": float(np.dot(ww, -np.log(np.clip(pp[np.arange(len(yy)), yy], EPS, 1))) / n),
        "brier": float(np.dot(ww, brier) / n),
        "ece": _ece(pp.max(axis=1), (pred == yy).astype(float), ww),
        "class_support": {name: int(supports[i]) for i, name in enumerate(classes)},
        "present_classes": int(present.sum()),
    }
    for top in (1, 3, 5):
        result[f"top{top}"] = float(np.dot(ww, np.any(rank[:, :min(top, k)] == yy[:, None], axis=1)) / n)
    result["effective_topk"] = {str(top): min(top, k) for top in (1, 3, 5)}
    return result


def _source_metrics(head: dict[str, Any], ai: np.ndarray, p_ai: np.ndarray,
                    cutoff: float, w: np.ndarray) -> dict[str, Any]:
    m, accept, p, names = head["mask"], head["accept"], head["probs"], head["classes"]
    weights = w * m
    ai_n, supported_n = float(w[ai == 1].sum()), float(weights.sum())
    pred_ai = p_ai >= cutoff
    accepted = m & accept
    eligible = (ai == 0) | m
    joint_y = np.where(ai == 0, 0, head["ids"] + 1)
    joint_p = np.concatenate([(1-p_ai)[:, None], p_ai[:, None] * p], axis=1)
    joint_w = w * eligible
    deployed = np.where(pred_ai, np.where(accept, p.argmax(axis=1) + 1, len(names) + 1), 0)
    n_joint = float(joint_w.sum())
    selected = joint_w > 0
    deployed_f1 = float(f1_score(joint_y[selected], deployed[selected], labels=np.arange(len(names)+1), average="macro", sample_weight=joint_w[selected], zero_division=0)) if n_joint else None
    claims = accept & pred_ai
    human_claims = float(w[claims & (ai == 0)].sum())
    trusted_claims = float(w[claims & m].sum())
    correct_claims = float(w[claims & m & (p.argmax(axis=1) == head["ids"])].sum())
    return {
        "conditional": _multiclass(head["ids"], p, names, weights),
        "accepted_conditional": _multiclass(head["ids"], p, names, w * accepted),
        "supported_AI_n": int(supported_n), "all_AI_n": int(ai_n),
        "supported_AI_coverage": _ratio(supported_n, ai_n),
        "accepted_coverage_on_supported_AI": _ratio(float(w[accepted].sum()), supported_n),
        "deployed_attribution_coverage_on_supported_AI": _ratio(float(w[accepted & pred_ai].sum()), supported_n),
        "assertion_audit": {
            "scope": "Assertions on humans are errors; trusted-source AI assertions are scored; AI without a trusted source label is counted separately and excluded from assertion-accuracy denominator.",
            "human_n": int(w[ai == 0].sum()),
            "asserted_on_humans_n": int(human_claims),
            "assertion_rate_on_humans": _ratio(human_claims, float(w[ai == 0].sum())),
            "asserted_on_trusted_AI_n": int(trusted_claims),
            "correct_asserted_on_trusted_AI_n": int(correct_claims),
            "wrong_asserted_on_trusted_AI_n": int(trusted_claims - correct_claims),
            "scored_assertions_including_humans_n": int(human_claims + trusted_claims),
            "accuracy_including_human_false_assertions": _ratio(correct_claims, human_claims + trusted_claims),
            "asserted_on_AI_without_trusted_source_label_n": int(w[claims & (ai == 1) & ~m].sum()),
            "conditional_accuracy_scope": "accepted_conditional scores trusted AI source labels only; it does not include human false assertions.",
        },
        "joint_probability": _multiclass(joint_y, joint_p, ["human"] + list(names), joint_w),
        "end_to_end": {
            "n": int(n_joint),
            "eligible_fraction_of_all_test": _ratio(n_joint, float(w.sum())),
            "accuracy": _ratio(float(joint_w[deployed == joint_y].sum()), n_joint),
            "macro_f1": deployed_f1,
            "correctly_attributed_AI_recall": _ratio(float(w[m & (deployed == joint_y)].sum()), supported_n),
            "description": "fixed-threshold source research metric, not full online AI/human/inconclusive decisions; source rejection counts as attribution error; unsupported AI source labels excluded, coverage shown",
        },
    }


def _unknown_metrics(evidence: Mapping[str, Any] | None, ai: np.ndarray,
                     w: np.ndarray) -> dict[str, Any]:
    if evidence is None:
        return {"status": "not_evaluated", "reason": "real held-out model classes and fitting-class provenance are required"}
    train_values = _vector(evidence.get("train_model_ids", []), "train_model_ids")
    heldout_values = _vector(evidence.get("heldout_model_ids", []), "heldout_model_ids")
    if any(not isinstance(v, (str, np.str_)) or not v for v in list(train_values)+list(heldout_values)):
        raise ValueError("fitting and heldout model identities must be nonempty strings")
    train = set(train_values); heldout = set(heldout_values)
    if not train or not heldout or train & heldout:
        raise ValueError("unknown evaluation needs nonempty, disjoint train/heldout model-ID sets")
    ids = _vector(evidence["test_model_ids"], "test_model_ids", len(ai))
    scores = _vector(evidence["rejection_scores"], "rejection_scores", len(ai)).astype(float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("unknown rejection scores must be finite; they need not be probabilities")
    mask = (ai == 1) & np.isin(ids, list(train | heldout))
    truth = np.isin(ids, list(heldout))
    ww = w * mask
    unknown_n, known_n = float(ww[truth].sum()), float(ww[~truth].sum())
    point = {"status": "evaluated" if unknown_n and known_n else "insufficient_heldout_classes",
             "known_AI_n": int(known_n), "truly_heldout_AI_n": int(unknown_n),
             "AI_evaluation_coverage": _ratio(float(ww.sum()), float(w[ai == 1].sum())),
             "score_name": evidence.get("score_name", "rejection_statistic"),
             "score_is_unknown_probability": False,
             "train_model_ids": sorted(train), "heldout_model_ids": sorted(heldout),
             "auroc": None, "auprc": None, "unknown_tpr": None, "known_false_rejection_rate": None}
    if unknown_n and known_n:
        selected = ww > 0
        point["auroc"] = float(roc_auc_score(truth[selected], scores[selected], sample_weight=ww[selected]))
        point["auprc"] = float(average_precision_score(truth[selected], scores[selected], sample_weight=ww[selected]))
    if evidence.get("threshold") is not None:
        cutoff = _fixed_threshold(evidence["threshold"])
        rejected = scores >= cutoff
        point["unknown_tpr"] = _ratio(float(ww[truth & rejected].sum()), unknown_n)
        point["known_false_rejection_rate"] = _ratio(float(ww[~truth & rejected].sum()), known_n)
    return point


def _scalar_summary(result: Mapping[str, Any]) -> dict[str, float | None]:
    out = {"binary."+k: result["binary"][k] for k in ("accuracy", "fpr", "tpr", "balanced_accuracy", "f1", "auroc", "auprc", "nll", "brier", "ece")}
    if result.get("deployed_decision_audit",{}).get("status")=="evaluated":
        for key in ("AI_decision_coverage","human_decision_coverage","inconclusive_coverage","decided_coverage",
                "conditional_accuracy_on_decided","human_misclassified_as_AI_rate","AI_misclassified_as_human_rate","correct_AI_recall","correct_human_recall"):
            out["deployed_decision_audit."+key]=result["deployed_decision_audit"][key]
    for h in ("family", "model"):
        if result.get(h):
            for k in ("accuracy", "macro_f1", "top1", "top3", "top5", "nll", "brier", "ece"):
                out[h+".conditional."+k] = result[h]["conditional"][k]
                out[h+".accepted_conditional."+k] = result[h]["accepted_conditional"][k]
            for k in ("assertion_rate_on_humans", "accuracy_including_human_false_assertions"):
                out[h+".assertion_audit."+k] = result[h]["assertion_audit"][k]
            for k in ("accuracy", "macro_f1", "correctly_attributed_AI_recall"):
                out[h+".end_to_end."+k] = result[h]["end_to_end"][k]
            for k in ("supported_AI_coverage", "accepted_coverage_on_supported_AI", "deployed_attribution_coverage_on_supported_AI"):
                out[h+"."+k] = result[h][k]
    if result["unknown_model"]["status"] != "not_evaluated":
        for k in ("auroc", "auprc", "unknown_tpr", "known_false_rejection_rate"):
            out["unknown_model."+k] = result["unknown_model"][k]
    return out


def evaluate_predictions(
    y_ai: Any, p_ai: Any, *, threshold: Mapping[str, Any], groups: Any,
    sources: Any = None, domains: Any = None, lengths: Any = None,
    length_unit: str = "word_like_tokens", family_y: Any = None,
    family_probs: Any = None, family_classes: Sequence[str] | None = None,
    family_mask: Any = None, model_y: Any = None, model_probs: Any = None,
    model_classes: Sequence[str] | None = None, model_mask: Any = None,
    family_accept: Any = None, model_accept: Any = None,
    language_supported: Any = None, human_threshold: Mapping[str, Any] | None = None,
    unknown_model_evidence: Mapping[str, Any] | None = None,
    n_boot: int = 500, seed: int = 42,
) -> dict[str, Any]:
    """Evaluate one locked prediction set without using its labels for tuning.

    ``family/model_y`` are raw source strings (None allowed where masked out).
    ``*_probs`` are N by K probabilities conditional on AI, in the exact order
    of ``*_classes``. Masks denote trusted, supported AI labels. Source acceptance
    masks must come from previously calibrated rules; omitted means all accepted.
    Group IDs must join original prompts/texts and their transformations.
    """
    y = _binary_truth(y_ai); p = _probability(p_ai, len(y)); n = len(y)
    if not isinstance(threshold, Mapping) or threshold.get("task") != "AI_detection":
        raise ValueError("binary evaluation requires an AI_detection threshold artifact")
    cutoff = _fixed_threshold(threshold)
    g = _vector(groups, "groups", n)
    if any(v is None or str(v) in ("", "nan") for v in g):
        raise ValueError("every evaluation row needs a stable nonmissing group ID")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
        raise ValueError("n_boot must be a nonnegative integer")
    family = _head(family_y, family_probs, family_classes, family_mask, family_accept, y, "family")
    model = _head(model_y, model_probs, model_classes, model_mask, model_accept, y, "model")
    decisions=deployed_decision_masks(p,lengths,language_supported,threshold,human_threshold) if lengths is not None and human_threshold is not None else None
    if language_supported is not None:_mask(language_supported,"language_supported",n)
    for head in (family,model):
        if head is not None:
            if language_supported is not None:head["accept"] &= _mask(language_supported,"language_supported",n)
            if decisions is not None:head["accept"] &= decisions["AI"]
    def point(w):
        return {"binary": _binary_metrics(y, p, cutoff, w),
                "family": _source_metrics(family, y, p, cutoff, w) if family else None,
                "model": _source_metrics(model, y, p, cutoff, w) if model else None,
                "unknown_model": _unknown_metrics(unknown_model_evidence, y, w),
                "deployed_decision_audit":_deployed_decision_audit(y,decisions,w)}
    result = point(np.ones(n))
    result["all_lengths"] = dict(result)
    result["threshold"] = dict(threshold)
    result["metric_definitions"] = {
        "binary_brier": "mean((p_AI-y_AI)^2)",
        "multiclass_brier": "mean(sum_k((p_k-1[y=k])^2))",
        "binary_ece": "15 equal-width bins of p_AI versus observed AI frequency",
        "multiclass_ece": "15 equal-width bins of top-label confidence versus top-label correctness",
        "macro_f1": "full declared supported class set, zero_division=0; support is reported",
        "binary_balanced_accuracy": "undefined unless both human and AI are present",
        "multiclass_balanced_accuracy": "mean recall of classes present in evaluation subset",
        "auprc": "average precision, positive class AI (or truly held-out AI for unknown benchmark)",
        "joint_vs_deployed": "joint probability argmax may differ from the fixed-threshold deployment gate; both are reported separately",
        "binary_scope":"ROC, probability metrics and fixed-AI-threshold binary counts include unsupported-language and short rows in their stated subgroup; they are not online AI/human/inconclusive decisions.",
        "deployed_decision_scope":"all_lengths/core/strata.source/strata.domain/core_strata contain explicit online coverage and both directions of false assertions; rejected rows are not removed from denominators",
    }
    result["strata"] = {"source": {}, "domain": {}, "length": {}}
    for name, values in (("source", sources), ("domain", domains)):
        if values is not None:
            vv = _vector(values, name, n).astype(str)
            for value in np.unique(vv):
                result["strata"][name][value] = point((vv == value).astype(float))
    core_mask = None
    if lengths is not None:
        ll = _vector(lengths, "lengths", n).astype(float)
        if not np.all(np.isfinite(ll)) or np.any(ll < 0) or np.any(ll != np.floor(ll)):
            raise ValueError("lengths must be nonnegative finite token counts")
        bins = (0, 64, 128, 256, 512, 1024, np.inf)
        for low, high in zip(bins[:-1], bins[1:]):
            name = f"{low}-{int(high)-1}" if np.isfinite(high) else f"{low}+"
            result["strata"]["length"][name] = point(((ll >= low) & (ll < high)).astype(float))
        result["core_ge_128_tokens"] = point((ll >= 128).astype(float))
        result["short_lt_128_tokens"] = point((ll < 128).astype(float))
        core_mask = (ll >= 128).astype(float)
        result["core_strata"]={"source":{},"domain":{}}
        for name,values in (("source",sources),("domain",domains)):
            if values is not None:
                vv=_vector(values,name,n).astype(str)
                for value in np.unique(vv):
                    result["core_strata"][name][value]=point((vv==value).astype(float)*core_mask)
    result["length_unit"] = length_unit
    unique, index = np.unique(g.astype(str), return_inverse=True)
    result["group_bootstrap"] = {
        "n_groups": int(len(unique)), "n_boot": int(n_boot), "seed": int(seed),
        "method": "percentile whole-group bootstrap; same resampling weights across tasks; fixed thresholds",
        "scope": "conditional on fitted predictions/calibration and sampled evaluation groups; excludes retraining/calibration/model-selection uncertainty; zero observed FP may produce a degenerate interval",
        "intervals95": {},
        "core_groups": int(len(np.unique(g[core_mask.astype(bool)]))) if core_mask is not None else None,
    }
    if n_boot and len(unique) >= 2:
        rng = np.random.default_rng(seed); samples: dict[str, list[float]] = {}
        for _ in range(n_boot):
            counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
            flat = _scalar_summary(point(counts[index].astype(float)))
            if core_mask is not None:
                flat.update({"core_ge_128_tokens."+k: v for k, v in _scalar_summary(point(counts[index].astype(float) * core_mask)).items()})
            for key, value in flat.items():
                if value is not None and np.isfinite(value):
                    samples.setdefault(key, []).append(float(value))
        points = _scalar_summary(result)
        if core_mask is not None:
            points.update({"core_ge_128_tokens."+k: v for k,v in _scalar_summary(result["core_ge_128_tokens"]).items()})
        for key, values in samples.items():
            lo, hi = np.quantile(values, [.025, .975])
            result["group_bootstrap"]["intervals95"][key] = {
                "point": points[key], "low": float(lo), "high": float(hi),
                "valid_replicates": len(values), "requested_replicates": int(n_boot),
            }
    return result


def evaluate_npz_payload(payload: Mapping[str, Any], *, threshold: Mapping[str, Any],
                         n_boot: int = 500, seed: int = 42,
                         unknown_model_evidence: Mapping[str, Any] | None = None,
                         length_unit: str = "word_like_tokens",
                         human_threshold: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Adapter for the shared training NPZ schema; no file/model side effects."""
    mapping = {"group_ids": "groups", "sources": "sources", "domains": "domains", "lengths": "lengths",
               "family_true": "family_y", "p_family": "family_probs", "family_labels": "family_classes", "mask_family": "family_mask",
               "model_true": "model_y", "p_model": "model_probs", "model_labels": "model_classes", "mask_model": "model_mask",
               "family_accept": "family_accept", "model_accept": "model_accept","language_supported":"language_supported"}
    kwargs = {out: payload[key] for key, out in mapping.items() if key in payload}
    return evaluate_predictions(payload["y_ai"], payload["p_ai"], threshold=threshold,
                                n_boot=n_boot, seed=seed,
                                unknown_model_evidence=unknown_model_evidence,
                                length_unit=length_unit,human_threshold=human_threshold, **kwargs)
