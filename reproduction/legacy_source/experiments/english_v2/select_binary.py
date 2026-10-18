"""Development-only, maximin selection for the preregistered v2 binary run.

No calibration or final scores enter this module. The source encoders and
their calibration are deliberately outside this selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from experiments.english_v1 import runtime  # noqa: F401
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

COHORTS = ("general", "source_domain", "modern", "paraphrase", "legacy_main", "legacy_creative")
SELECTOR = {
    "algorithm": "v2_maximin_binary_v1",
    "cohorts": list(COHORTS),
    "legacy_retention_disclosure": "legacy_main and legacy_creative are observed v1 development, never fresh final evidence",
    "core_min_words": 128,
    "fit_selection_group_salt": "english-v2-binary-development:",
    "empirical_human_FPR": 0.02,
    "quality": "0.5*AUROC+0.5*AI_recall_at_empirical_2pct_human_FPR",
    "ordering": ["minimum_cohort_quality", "mean_cohort_quality", "negative_mean_cohort_NLL", "negative_nonzero_components"],
    "fit_weighting": "equal cohort mass; equal human/AI mass within each cohort",
    "candidates": "each single; each pair at 0.25/0.5/0.75; equal-all; nonnegative logistic if optimizer converges",
    "standardization": "all core fitting halves only; unweighted mean/std; scale floor 1e-6",
    "logistic": {"l2": 0.001, "weight_bounds": [0, 20], "intercept_bounds": [-20, 20], "solver": "L-BFGS-B"},
    "minimum_core_each_class_each_half": 50,
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_bound(item):
    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise ValueError("bound file changed: " + str(path))
    return path


def metrics(y, margin):
    y = np.asarray(y, dtype=np.int64)
    human = np.sort(margin[y == 0])
    allowed = int(np.floor(SELECTOR["empirical_human_FPR"] * len(human)))
    threshold = np.nextafter(human[len(human) - allowed - 1], np.inf)
    recall = float(np.mean(margin[y == 1] >= threshold))
    auc = float(roc_auc_score(y, margin))
    return {"n": len(y), "human_n": int((y == 0).sum()), "AI_n": int((y == 1).sum()),
        "AUROC": auc, "AI_recall_at_empirical_human_FPR_2pct": recall,
        "nll": float(np.mean(np.logaddexp(0, margin) - y * margin)),
        "quality": 0.5 * auc + 0.5 * recall}


def fit_selection(plan_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("never overwrite a development selection")
    plan = json.loads(Path(plan_path).read_text())
    protocol_path = load_bound(plan["protocol"])
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_V2_TRAINING_AND_SCORES" or protocol.get("binary_selector") != SELECTOR:
        raise ValueError("a pre-training frozen protocol adopting this exact selector is required")
    if plan.get("calibration_opened") is not False or plan.get("final_test_opened") is not False:
        raise ValueError("development selection must precede calibration and final scores")
    names = list(plan["names"])
    if names != protocol["candidate_names"] or len(names) < 1 or len(names) != len(set(names)):
        raise ValueError("candidate names differ from the preregistered budget")
    if set(plan["cohorts"]) != set(COHORTS):
        raise ValueError("all six registered development cohorts are required")
    records, components, source_files = {}, {}, {}
    seen_groups, seen_ids = set(), set()
    for cohort in COHORTS:
        spec = plan["cohorts"][cohort]
        if set(spec) != set(names):
            raise ValueError("every candidate requires every cohort")
        raw, reference = [], None
        source_files[cohort] = {}
        for name in names:
            path = load_bound(spec[name])
            with np.load(path, allow_pickle=False) as value:
                if str(value["export_purpose"].item()) != "development":
                    raise ValueError("only explicitly development raw exports are accepted")
                if "calibrated" not in value or value["calibrated"].shape != () or value["calibrated"].dtype.kind != "b" or bool(value["calibrated"].item()):
                    raise ValueError("explicitly uncalibrated raw exports are required")
                if any("calibration" in key.casefold() for key in value.files):
                    raise ValueError("calibration-bound arrays cannot enter development selection")
                data = {key: value[key].copy() for key in ("ids", "group_ids", "y_ai", "lengths", "raw_binary")}
                identity = {"weights_sha256": str(value["component_weights_sha256"].item()),
                    "metadata_sha256": str(value["component_metadata_sha256"].item())}
            if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in identity.values()):
                raise ValueError("complete SHA256 component fingerprints required")
            if identity != plan["components"][name]:
                raise ValueError("raw export identity differs from its frozen checkpoint registry")
            if name in components and components[name] != identity:
                raise ValueError("a candidate changed between development cohorts")
            components[name] = identity
            if reference is None:
                reference = data
            elif any(not np.array_equal(reference[k], data[k]) for k in ("ids", "group_ids", "y_ai", "lengths")):
                raise ValueError("raw exports must be aligned")
            logits = np.asarray(data["raw_binary"], dtype=np.float64)
            if logits.shape != (len(data["ids"]), 2) or not np.isfinite(logits).all():
                raise ValueError("finite N by 2 raw binary logits required")
            raw.append(logits[:, 1] - logits[:, 0])
            source_files[cohort][name] = {"path": str(path.resolve()), "sha256": spec[name]["sha256"]}
        n = len(reference["ids"])
        for key in ("ids", "group_ids"):
            value = reference[key]
            if value.shape != (n,) or value.dtype.kind != "U" or not all(str(item).strip() for item in value):
                raise ValueError("aligned nonempty string IDs and parent groups required")
        groups = set(str(g) for g in reference["group_ids"])
        if groups & seen_groups:
            raise ValueError("development cohort parent components must be disjoint")
        seen_groups.update(groups)
        if len(set(reference["ids"].tolist())) != len(reference["ids"]):
            raise ValueError("duplicate document IDs")
        ids = set(reference["ids"].tolist())
        if ids & seen_ids:
            raise ValueError("document IDs must be disjoint across development cohorts")
        seen_ids.update(ids)
        y = reference["y_ai"]
        if y.shape != (n,) or y.dtype.kind not in "biuf" or not np.all(np.isin(y, [0, 1])):
            raise ValueError("binary truth required")
        y = y.astype(np.int64)
        lengths = reference["lengths"]
        if (lengths.shape != (n,) or lengths.dtype.kind not in "iuf" or
                not np.isfinite(lengths).all() or np.any(lengths < 0) or np.any(lengths != np.floor(lengths))):
            raise ValueError("aligned finite nonnegative integer word counts required")
        core = reference["lengths"] >= SELECTOR["core_min_words"]
        fit = np.asarray([int(hashlib.sha256((SELECTOR["fit_selection_group_salt"] + str(g)).encode()).hexdigest()[:8], 16) % 2 == 0
            for g in reference["group_ids"]])
        for mask in (core & fit, core & ~fit):
            if min(int((mask & (y == label)).sum()) for label in (0, 1)) < SELECTOR["minimum_core_each_class_each_half"]:
                raise ValueError("insufficient core human/AI rows in a development half")
        records[cohort] = (reference, np.stack(raw, axis=1), core & fit, core & ~fit, ~fit)
    all_x = np.concatenate([x[m] for _, x, m, _, _ in records.values()])
    all_y = np.concatenate([a["y_ai"][m] for a, _, m, _, _ in records.values()])
    mean, scale = all_x.mean(0), np.maximum(all_x.std(0), 1e-6)
    xx = (all_x - mean) / scale
    fit_weights = []
    for a, _, mask, _, _ in records.values():
        y = a["y_ai"][mask]
        fit_weights.append(np.asarray([1 / len(COHORTS) / 2 / int((y == label).sum()) for label in y]))
    ww = np.concatenate(fit_weights)
    count = len(names)
    candidates = []
    for i, name in enumerate(names):
        weights = np.zeros(count); weights[i] = 1
        candidates.append((name, weights, 0.0))
    for i in range(count):
        for j in range(i + 1, count):
            for alpha in (0.25, 0.5, 0.75):
                weights = np.zeros(count); weights[i], weights[j] = alpha, 1 - alpha
                candidates.append((f"{names[i]}:{alpha}+{names[j]}:{1-alpha}", weights, 0.0))
    if count > 1:
        candidates.append(("equal_all", np.full(count, 1 / count), 0.0))

    def objective(theta):
        weights, intercept = theta[:-1], theta[-1]
        margin = xx @ weights + intercept
        err = (expit(margin) - all_y) * ww
        loss = np.dot(ww, np.logaddexp(0, margin) - all_y * margin) + 0.001 * np.sum(weights ** 2)
        return float(loss), np.r_[xx.T @ err + 0.002 * weights, err.sum()]

    optimized = minimize(objective, np.r_[np.full(count, 1 / count), 0.0], jac=True,
        method="L-BFGS-B", bounds=[(0.0, 20.0)] * count + [(-20.0, 20.0)])
    if optimized.success and np.isfinite(optimized.x).all() and np.any(optimized.x[:-1] > 0):
        candidates.append(("development_nonnegative_logistic", optimized.x[:-1], float(optimized.x[-1])))
    scored = []
    for name, weights, intercept in candidates:
        scores, all_length_scores = {}, {}
        for cohort, (a, x, _, mask, all_mask) in records.items():
            margin = ((x - mean) / scale) @ weights + intercept
            scores[cohort] = metrics(a["y_ai"][mask], margin[mask])
            all_length_scores[cohort] = metrics(a["y_ai"][all_mask], margin[all_mask])
        scored.append({"name": name, "weights": weights.tolist(), "intercept": intercept,
            "minimum_cohort_quality": min(v["quality"] for v in scores.values()),
            "mean_cohort_quality": float(np.mean([v["quality"] for v in scores.values()])),
            "mean_cohort_nll": float(np.mean([v["nll"] for v in scores.values()])),
            "nonzero_components": int(np.count_nonzero(weights)), "metrics": scores,
            "all_lengths_diagnostic_only": all_length_scores})
    scored.sort(key=lambda r: (r["minimum_cohort_quality"], r["mean_cohort_quality"], -r["mean_cohort_nll"], -r["nonzero_components"]), reverse=True)
    artifact = {"schema_version": 1, "fit_split": "development_only", "names": names,
        "components": components, "mean": mean.tolist(), "scale": scale.tolist(), "selected": scored[0],
        "candidates": scored, "selector": SELECTOR, "minimum_word_like_tokens": SELECTOR["core_min_words"],
        "fit_selection_groups_disjoint": True, "development_cohorts_groups_disjoint": True,
        "calibration_opened": False, "final_test_opened": False,
        "protocol": plan["protocol"], "plan_sha256": sha(plan_path), "code_sha256": sha(__file__),
        "source_files": source_files, "optimizer": {"converged": bool(optimized.success), "message": str(optimized.message)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(fit_selection(args.plan, args.output)["selected"], indent=2, allow_nan=False))
