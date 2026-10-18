"""Post-selection calibration and selective attribution with explicit semantics.

Fit only on the independent calibration partition. Development-time blend
selection uses separate development labels and is deliberately not implemented
here. A confidence cutoff is not a probability of an unseen generator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp, softmax


def _matrix(value: Any, columns: int | None = None) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("nonempty finite N by K scores required")
    if columns is not None and x.shape[1] != columns:
        raise ValueError("score class order/shape mismatch")
    return x


def _labels(value: Any, n: int, k: int) -> np.ndarray:
    y = np.asarray(value)
    if y.shape != (n,) or not np.all(np.isin(y, np.arange(k))):
        raise ValueError("labels must be integer class indices in the declared order")
    return y.astype(np.int64)


def prediction_fingerprint(ids: Sequence[str]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@dataclass
class BinaryPlatt:
    slope: float = 1.0
    intercept: float = 0.0
    fitted_n: int = 0
    fit_split: str = "unfitted"

    def fit(self, raw_margin: Any, labels: Any, *, split: str = "calibration"):
        if split != "calibration":
            raise ValueError("final probability calibration requires the calibration split")
        x = np.asarray(raw_margin, dtype=np.float64)
        if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
            raise ValueError("finite raw AI-oriented margins required")
        y = _labels(labels, len(x), 2)
        if len(np.unique(y)) != 2:
            raise ValueError("calibration requires both human and AI examples")

        def objective(theta):
            a, b = theta
            z = a*x+b
            loss = np.mean(np.logaddexp(0, z)-y*z) + 1e-6*(a*a+b*b)
            error = expit(z)-y
            gradient = np.array([np.mean(error*x), np.mean(error)])+2e-6*theta
            return float(loss), gradient

        result = minimize(objective, np.array([1., 0.]), jac=True,
                          bounds=[(0., 100.), (-50., 50.)], method="L-BFGS-B")
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"binary calibration failed: {result.message}")
        self.slope, self.intercept = map(float, result.x)
        self.fitted_n, self.fit_split = len(y), split
        return self

    def predict(self, raw_margin: Any) -> np.ndarray:
        x = np.asarray(raw_margin, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("nonfinite prediction scores")
        return expit(self.slope*x+self.intercept)


@dataclass
class Temperature:
    value: float = 1.
    fitted_n: int = 0
    fit_split: str = "unfitted"
    status: str = "identity_unfitted"

    def fit(self, logits: Any, labels: Any, *, minimum_n: int = 50,
            split: str = "calibration"):
        if split != "calibration":
            raise ValueError("final probability calibration requires calibration")
        x = _matrix(logits)
        y = _labels(labels, len(x), x.shape[1])
        self.fitted_n, self.fit_split = len(y), split
        if len(y) < minimum_n or len(np.unique(y)) < 2:
            self.status = "insufficient_calibration_labels_identity"
            return self

        def objective(log_temperature):
            scaled = x/np.exp(log_temperature)
            return float(np.mean(logsumexp(scaled, axis=1)-scaled[np.arange(len(y)), y]))

        result = minimize_scalar(objective, bounds=(np.log(.05), np.log(20.)),
                                 method="bounded", options={"xatol": 1e-7})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("temperature fit failed")
        self.value, self.status = float(np.exp(result.x)), "fitted"
        return self

    def predict(self, logits: Any) -> np.ndarray:
        return softmax(_matrix(logits)/self.value, axis=1)


def selective_cutoff(confidence: Any, correct: Any, *, target_accuracy: float,
                     eligible: Any | None = None, minimum_n: int = 50,
                     wilson_z: float = 1.6448536269514722) -> dict[str, Any]:
    """Maximize calibration coverage subject to a Wilson lower accuracy bound.

    Ties are accepted together. The Wilson number describes the calibration
    cohort only; threshold selection and domain shift prevent a population
    coverage/accuracy guarantee. An unattainable target accepts no assertions.
    """
    p = np.asarray(confidence, dtype=np.float64)
    c = np.asarray(correct)
    if p.ndim != 1 or c.shape != p.shape or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("finite confidence in [0,1] and aligned correctness required")
    if not np.all(np.isin(c, [0, 1, False, True])) or not 0 < target_accuracy <= 1:
        raise ValueError("boolean correctness and target in (0,1] required")
    m = np.ones(len(p), dtype=bool) if eligible is None else np.asarray(eligible)
    if m.shape != p.shape or not np.all(np.isin(m, [0, 1, False, True])):
        raise ValueError("aligned boolean eligibility required")
    m = m.astype(bool)
    pp, cc = p[m], c[m].astype(int)
    order = np.argsort(-pp, kind="stable")
    pp, cc = pp[order], cc[order]
    n = np.arange(1, len(pp)+1)
    total = np.cumsum(cc)
    rate = total/np.maximum(n, 1)
    z2 = wilson_z**2
    lower = (rate+z2/(2*np.maximum(n, 1))-wilson_z*np.sqrt(rate*(1-rate)/np.maximum(n,1)+z2/(4*np.maximum(n,1)**2)))/(1+z2/np.maximum(n,1))
    end_tie = np.r_[pp[:-1] != pp[1:], True] if len(pp) else np.array([], dtype=bool)
    valid = np.flatnonzero((n >= minimum_n) & (lower >= target_accuracy) & end_tie)
    if len(valid):
        j = int(valid[-1]); threshold = float(pp[j]); accepted = int(n[j])
        accuracy, lower_bound = float(rate[j]), float(lower[j])
    else:
        threshold, accepted, accuracy, lower_bound = float(np.nextafter(1., np.inf)), 0, None, None
    return {"fit_split": "calibration", "comparison": ">=", "threshold": threshold,
            "target_accuracy": target_accuracy, "minimum_n": minimum_n,
            "eligible_n": int(m.sum()), "cohort_n": len(m), "accepted_n": accepted,
            "eligible_coverage": accepted/int(m.sum()) if m.sum() else 0.,
            "cohort_coverage": accepted/len(m) if len(m) else 0.,
            "empirical_accuracy": accuracy, "wilson_lower": lower_bound,
            "wilson_z": wilson_z, "population_guarantee": False,
            "score_is_unknown_probability": False}


class HierarchicalCalibration:
    """Calibrate family and within-family logits, preserving exact hierarchy."""
    def __init__(self, family_classes: Sequence[str], children: dict[str, Sequence[str]]):
        self.family_classes = list(family_classes)
        self.children = {f: list(children[f]) for f in self.family_classes}
        self.leaf_classes = [c for f in self.family_classes for c in self.children[f]]
        if len(set(self.family_classes)) != len(self.family_classes) or len(set(self.leaf_classes)) != len(self.leaf_classes):
            raise ValueError("unique declared class names required")
        self.family = Temperature()
        self.versions = {f: Temperature() for f in self.family_classes}

    def fit(self, family_logits, version_logits: dict[str, Any], family_labels,
            leaf_labels, family_mask, leaf_mask):
        x = _matrix(family_logits, len(self.family_classes))
        f = np.asarray(family_labels); l = np.asarray(leaf_labels)
        fm, lm = np.asarray(family_mask), np.asarray(leaf_mask)
        if any(a.shape != (len(x),) for a in (f, l, fm, lm)):
            raise ValueError("aligned hierarchy labels/masks required")
        if not np.all(np.isin(fm, [0,1,True,False])) or not np.all(np.isin(lm, [0,1,True,False])):
            raise ValueError("boolean hierarchy masks required")
        fm, lm = fm.astype(bool), lm.astype(bool)
        if np.any(fm & ~np.isin(f, self.family_classes)) or np.any(lm & ~np.isin(l, self.leaf_classes)) or np.any(lm & ~fm):
            raise ValueError("masked hierarchy labels are absent from catalog")
        if fm.any():
            y = np.array([self.family_classes.index(v) for v in f[fm]])
            self.family.fit(x[fm], y)
        for name in self.family_classes:
            vv = _matrix(version_logits[name], len(self.children[name]))
            if len(vv) != len(x):
                raise ValueError("version logits row mismatch")
            m = lm & (f == name)
            if np.any(m & ~np.isin(l, self.children[name])):
                raise ValueError("leaf label contradicts family label")
            if m.any():
                y = np.array([self.children[name].index(v) for v in l[m]])
                self.versions[name].fit(vv[m], y)
        return self

    def predict(self, family_logits, version_logits):
        q = self.family.predict(family_logits)
        conditionals = {f: self.versions[f].predict(version_logits[f]) for f in self.family_classes}
        leaves = np.concatenate([q[:, i:i+1]*conditionals[f] for i,f in enumerate(self.family_classes)], axis=1)
        if not np.allclose(leaves.sum(1), 1., atol=1e-10, rtol=0):
            raise AssertionError("hierarchy probabilities must sum to one")
        return {"family": q, "leaf": leaves, "within_family": conditionals}

    def to_dict(self):
        return {"family_classes": self.family_classes, "children": self.children,
                "leaf_classes": self.leaf_classes, "family": asdict(self.family),
                "versions": {f: asdict(t) for f,t in self.versions.items()},
                "semantics": "family and leaf probabilities are conditional on AI; leaf sums equal family; unknown buckets refer only to trained catalog complement"}

    @classmethod
    def from_dict(cls, value):
        obj = cls(value["family_classes"], value["children"])
        if value.get("leaf_classes") != obj.leaf_classes:
            raise ValueError("saved hierarchy class order mismatch")
        obj.family = Temperature(**value["family"])
        obj.versions = {f: Temperature(**value["versions"][f]) for f in obj.family_classes}
        for t in [obj.family, *obj.versions.values()]:
            if not np.isfinite(t.value) or t.value <= 0:
                raise ValueError("saved temperature must be positive and finite")
            if t.fit_split not in {"calibration", "unfitted"}:
                raise ValueError("saved temperature has an invalid fitting partition")
        return obj


def save_calibration(path, *, binary: BinaryPlatt, hierarchy: HierarchicalCalibration,
                     ai_threshold: dict, family_threshold: dict, model_threshold: dict,
                     provenance: dict, human_threshold: dict | None = None):
    value = {"schema_version": 1, "binary": asdict(binary), "hierarchy": hierarchy.to_dict(),
             "thresholds": {"AI": ai_threshold, "family": family_threshold, "model": model_threshold},
             "provenance": provenance,
             "probability_scope": "Calibration-cohort probabilities; deployment priors/domain shifts can change reliability. No authenticity proof or fraction-of-AI-words interpretation."}
    if human_threshold is not None:
        value["thresholds"]["human"] = human_threshold
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return value
