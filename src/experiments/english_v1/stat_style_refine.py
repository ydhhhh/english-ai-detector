"""Finite binary style-channel ablation; choose with development raw scores only.

Run through the canonical module import so joblib stores an importable wrapper:
python -c 'from experiments.english_v1.stat_style_refine import main; main()' ...
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from experiments.english_v1.statistical import (
    EnglishStatisticalDetector, _aggregate_window_scores, _binary_sample,
    _score_matrix, _selection_metrics, _style_rows, _texts, _window_plan,
    assert_disjoint, export_predictions,
)
from experiments.english_v1.stat_train import (
    PROJECT, _dump, batch_invariance_check, file_sha256, read_development_records,
)


class StyleOnlyDecisionHead:
    """Binary estimator operating on a fixed subset of the saved feature layout."""

    def __init__(self, estimator, columns):
        self.estimator = estimator
        self.columns = np.asarray(columns, dtype=np.int64)
        self.classes_ = estimator.classes_

    def decision_function(self, matrix):
        return self.estimator.decision_function(matrix[:, self.columns])


def style_columns(features):
    offset = 0
    columns = []
    for name in features.vectorizers_:
        count = features.feature_counts_[name]
        if name in ("shape", "function"):
            columns.extend(range(offset, offset + count))
        offset += count
    columns.extend(range(offset, offset + 28))
    return np.asarray(columns, dtype=np.int64)


def style_matrix(features, texts):
    """Equivalent to full transform[:, style_columns], without lexical blocks."""
    values = features._prepare(texts)
    blocks = [vectorizer.transform(features._channel(values, name)) * features.channel_weights_[name]
        for name, vectorizer in features.vectorizers_.items() if name in ("shape", "function")]
    scaled = np.clip(features.style_scaler_.transform(_style_rows(values)), -8., 8.)
    blocks.append(normalize(csr_matrix(scaled), norm="l2", copy=False) * .35)
    return hstack(blocks, dtype=np.float32, format="csr")


def run(source, train, validation, calibration, dev_challenge, output):
    started = time.monotonic()
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = EnglishStatisticalDetector.load(source / "model.joblib")
    metadata = json.loads((source / "metadata.json").read_text())
    paths = {"train": Path(train), "validation": Path(validation),
        "calibration": Path(calibration), "dev_challenge": Path(dev_challenge)}
    # Inputs are the exact development cohorts already used in the base model.
    for key, path in paths.items():
        if path.stem.casefold() in ("test", "ood", "challenge_source_domain") or path.stem.casefold().startswith("test_"):
            raise ValueError("final test/challenge path refused")
        if file_sha256(path) != metadata["inputs"][key]["sha256"]:
            raise ValueError(f"{key} differs from the frozen base input")
    records = {key: read_development_records(path, key) for key, path in paths.items()}
    for key in ("validation", "dev_challenge", "calibration"):
        assert_disjoint(records["train"], records[key], "train", key)
    indices = _binary_sample(records["train"], model.config.max_binary_train_rows,
        model.config.seed, model.config.max_binary_train_chars, model.config.max_chars_per_text)
    training = [records["train"][i] for i in indices]
    y = np.asarray([r["y_ai"] for r in training], dtype=np.int64)
    columns = style_columns(model.features_)
    plans = {key: _window_plan(_texts(records[key]), model.config)
        for key in ("validation", "dev_challenge")}
    history = []
    with threadpool_limits(limits=model.config.cpu_threads):
        check_texts = _texts(training[:12])
        direct = style_matrix(model.features_, check_texts)
        sliced = model.features_.transform(check_texts)[:, columns]
        delta = direct - sliced
        if delta.nnz and float(np.max(np.abs(delta.data))) > 1e-7:
            raise AssertionError("style transform differs from saved full-feature column layout")
        fit = style_matrix(model.features_, _texts(training))
        dev = {key: style_matrix(model.features_, plan["texts"]) for key, plan in plans.items()}
        print(json.dumps({"phase":"style_features_ready","rows":len(training),
            "class_support":dict(Counter(str(v) for v in y)),"features":fit.shape[1],
            "csr_bytes":fit.data.nbytes+fit.indices.nbytes+fit.indptr.nbytes}), flush=True)
        base = metadata["selection"]["binary"]
        base_entry = next(entry for entry in base["candidates"] if entry["candidate"] == base["selected"])
        best_score, best, best_name = base_entry["selection_score"], None, base["selected"]
        candidates = [
            ("style_linear_svc_C0.3", LinearSVC(C=.3,dual="auto",class_weight="balanced",max_iter=model.config.svc_max_iter,random_state=model.config.seed)),
            ("style_linear_svc_C1.0", LinearSVC(C=1.,dual="auto",class_weight="balanced",max_iter=model.config.svc_max_iter,random_state=model.config.seed)),
            ("style_sgd_logloss_a1e-5", SGDClassifier(loss="log_loss",alpha=1e-5,class_weight="balanced",average=True,max_iter=model.config.sgd_max_iter,tol=1e-4,random_state=model.config.seed)),
        ]
        for name, estimator in candidates:
            t0 = time.monotonic()
            estimator.fit(fit, y)
            values = {}
            for key, matrix in dev.items():
                scores = _aggregate_window_scores(_score_matrix(estimator,matrix), plans[key])
                truth = np.asarray([r["y_ai"] for r in records[key]])
                values[key] = _selection_metrics(truth, scores, np.asarray([0,1]), "binary")
            score = model.config.external_validation_weight * values["dev_challenge"]["auroc"] + (1-model.config.external_validation_weight) * values["validation"]["auroc"]
            entry = {"candidate":name,"validation":values["validation"],"dev_challenge":values["dev_challenge"],"selection_score":score,"fit_seconds":time.monotonic()-t0}
            history.append(entry)
            print(json.dumps({"phase":"style_candidate_ready",**entry}),flush=True)
            if score > best_score:
                best_score, best, best_name = score, estimator, name
        del fit, dev
        gc.collect()
    summary = {"source":str(source.resolve()),"selected":best_name,
        "style_selected":best is not None,"base_selection_score":base_entry["selection_score"],
        "best_selection_score":best_score,"selection_used_calibration_labels":False,
        "selection_used_final_test_or_challenge_labels":False,
        "method":"Fixed train-fitted shape/function/numeric columns; same deterministic class-balanced training subset, three predeclared classifiers, raw full-window development AUROC selection with 0.7 external weight.",
        "style_column_equivalence_check":True,"style_features":len(columns),"candidates":history}
    if best is not None:
        model.models_["binary"] = StyleOnlyDecisionHead(best, columns)
        model.selection_["binary"] = {**base,"selected":best_name,"candidates":base["candidates"]+history,
            "selected_feature_channels":["shape","function","numeric_style"],"raw_score_layout_unchanged":True}
        # The ablation winner is frozen before any new calibration fit.
        model.calibrate(records["calibration"])
        invariance = batch_invariance_check(model, _texts(records["validation"]))
        model.save(output / "model.joblib")
        exports = {key:export_predictions(model,records[key],output/(key+"_predictions.npz"),output/(key+"_predictions.jsonl"))
            for key in ("validation","calibration","dev_challenge")}
        snapshots=output/"code";snapshots.mkdir(exist_ok=True)
        for path in (Path(__file__),Path(__file__).with_name("statistical.py"),Path(__file__).with_name("stat_train.py"),PROJECT/"experiments/ntl_v1/baseline.py"):
            shutil.copyfile(path,snapshots/path.name)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes=int(peak if sys.platform=="darwin" else peak*1024)
        result={**metadata,**model.metadata(),"exports":exports,"batch_invariance":invariance,
            "source_base_artifact":str(source.resolve()),"style_ablation":summary,
            "peak_process_rss_bytes":peak_bytes,"total_seconds":time.monotonic()-started,
            "code_sha256":{str(p.relative_to(PROJECT)):file_sha256(p) for p in (Path(__file__),Path(__file__).with_name("statistical.py"),Path(__file__).with_name("stat_train.py"),PROJECT/"experiments/ntl_v1/baseline.py")}}
        _dump(output/"metadata.json",result)
        _dump(output/"labels.json",{"binary":["human","AI"],"family":model.classes_["family"],"model":model.classes_["model"]})
        _dump(output/"batch_invariance.json",invariance)
    summary["total_seconds"]=time.monotonic()-started
    _dump(output/"summary.json",summary)
    print(json.dumps({"phase":"style_ablation_complete","selected":best_name,"style_selected":best is not None,"output":str(output.resolve())}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","train","validation","calibration","dev-challenge","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    run(args.source,args.train,args.validation,args.calibration,args.dev_challenge,args.output)


if __name__ == "__main__":
    raise SystemExit("Invoke main through the canonical module import shown in the docstring.")
