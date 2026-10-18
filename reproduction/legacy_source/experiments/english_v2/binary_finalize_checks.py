"""Synthetic-only checks for v2 binary finalization; no model/data inference.

The original v1 scalar calibration artifact supplies the frozen source schema
and temperatures. Every raw score, label, ID and Parquet row is constructed in
memory or a TemporaryDirectory. No real v2calibration/final input is opened.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from experiments.english_v1 import runtime  # noqa: F401
import numpy as np
from experiments.english_v1.calibration import BinaryPlatt, HierarchicalCalibration
from experiments.english_v1.catalog import FAMILIES, MODEL_LABELS, VERSION_INDICES
from experiments.english_v1.evaluate import deployed_decision_masks
from experiments.english_v1.inference import EnglishDetector
from experiments.english_v1 import version_refinement
from experiments.english_v2 import binary_finalize as bf


def run():
    source_path = bf.PROJECT / "results/english_v1/release_v1/calibration.json"
    source = bf.load_json_bound(source_path, bf.SOURCE_CALIBRATION_SHA256)
    before = bf.digest(source)
    rng = np.random.default_rng(20261017)
    n = 256
    y = np.arange(n) % 2
    margin = (2*y-1)*1.8 + rng.normal(0, .8, n)
    lengths = np.full(n, 200, dtype=np.int64)
    lengths[:2] = 64
    supported = np.ones(n, dtype=bool)
    supported[2:4] = False
    fake_identity = {"weights_sha256": bf.digest("synthetic_binary_weights"), "metadata_sha256": bf.digest("synthetic_binary_metadata")}
    common = {"ids": np.asarray([f"synthetic:{i}" for i in range(n)]), "group_ids": np.asarray([f"synthetic_parent:{i//2}" for i in range(n)]),
        "calibrated": np.asarray(False),
        "y_ai": y, "lengths": lengths, "scope_word_like_tokens": lengths.copy(), "language_supported": supported,
        "language_rule_sha256": np.asarray(bf.language_sha()), "raw_binary": np.column_stack([-margin/2, margin/2]),
        "family_true": np.full(n, "unknown"), "model_true": np.full(n, "unknown"),
        "mask_family": np.zeros(n, dtype=bool), "mask_model": np.zeros(n, dtype=bool),
        "family_labels": np.asarray(FAMILIES), "model_labels": np.asarray(MODEL_LABELS),
        "encoder_coverage": np.ones(n), "encoder_tokens": lengths+2, "window_counts": np.ones(n, dtype=np.int64)}
    binary = {**common, "component_weights_sha256": np.asarray(fake_identity["weights_sha256"]),
        "component_metadata_sha256": np.asarray(fake_identity["metadata_sha256"])}
    components = {"synthetic_binary": fake_identity, **{name: {"weights_sha256": source["provenance"]["component_weights_sha256"][name],
        "metadata_sha256": source["provenance"]["component_metadata_sha256"][name]} for name in bf.SOURCE_WEIGHTS}}
    blend = {"names": ["synthetic_binary"], "selected": {"weights": [1.], "intercept": 0.}, "mean": [0.], "scale": [1.],
        "components": components, "fit_split": "development_only", "calibration_opened": False, "final_test_opened": False}
    protocol = {"schema_version": 1, "status": "FROZEN_BEFORE_V2_TRAINING_AND_SCORES", "source_mode": "rankings_only", "source_calibration_sha256": bf.SOURCE_CALIBRATION_SHA256,
        "language_rule_sha256": bf.language_sha(), "deployment_decision_protocol_sha256": bf._decision_protocol_sha256(),
        "calibration_input_sha256": bf.digest("synthetic_input"), "calibration_mixture": {"synthetic_fixture_only": True},
        "calibration_role_history": {"synthetic_fixture_only": True, "fresh": {"synthetic_rows": 128},
            "historical_CAL_replay": {"source_sha256": bf.digest("synthetic_historical_CAL"), "selected_rows": 128,
                "previous_role": "calibration", "current_role": "calibration", "observed_history_disclosed": True}},
        "binary_calibration": {"probability_fit_subset": "supported_all_lengths", "threshold_subset": "supported_core",
            "minimum_word_like_tokens": 128, "AI_target_fpr": .01, "human_target_fpr": .01}, "final_input_sha256s": []}
    kwargs = {"blend_sha256": bf.digest(blend), "source_calibration_sha256": bf.SOURCE_CALIBRATION_SHA256,
              "protocol": protocol, "protocol_sha256": bf.digest(protocol), "calibration_input_sha256": protocol["calibration_input_sha256"]}
    original_fit = HierarchicalCalibration.fit
    with patch.object(HierarchicalCalibration, "fit", side_effect=AssertionError("source fit forbidden")), \
         patch.object(version_refinement, "calibrate", side_effect=AssertionError("source version fit forbidden")):
        calibrated = bf.fit_binary_only(blend, {"synthetic_binary": binary}, source, **kwargs)
    assert HierarchicalCalibration.fit is original_fit
    checks = ["No hierarchy/refinement fit function called"]
    assert bf.digest(source) == before and bf.digest(bf.source_objects(calibrated)) == bf.digest(bf.source_objects(source))
    assert calibrated["hierarchy"] is not source["hierarchy"]
    assert calibrated["provenance"]["calibration_input_sha256"] == protocol["calibration_input_sha256"]
    assert "fresh_calibration_sha256" not in calibrated["provenance"]
    assert calibrated["provenance"]["only_binary_fitted_on_registered_calibration"] is True
    assert calibrated["provenance"]["binary_calibration"]["calibration_role_history"] == protocol["calibration_role_history"]
    checks.append("Parent source artifact untouched and all inherited objects exactly preserved")
    assert calibrated["binary"]["fitted_n"] == int(supported.sum())
    assert calibrated["provenance"]["binary_calibration"]["threshold_rows"] == int((supported & (lengths>=128)).sum())
    for key in ("AI", "human"):
        t = calibrated["thresholds"][key]
        assert t["threshold"] == max(t["original_empirical_threshold"], np.nextafter(.5, np.inf)) and t["threshold"] > .5
    tied = deployed_decision_masks(np.asarray([.5]), np.asarray([128]), np.asarray([True]),
        calibrated["thresholds"]["AI"], calibrated["thresholds"]["human"])
    assert tied["inconclusive"].tolist() == [True] and not tied["AI"].any() and not tied["human"].any()
    checks.append("Explicit fit/threshold subsets and unchanged strict-majority exact-tie rejection")
    fine_changed = dict(binary)
    fine_changed.update(mask_family=y.astype(bool), mask_model=y.astype(bool), family_true=np.full(n, "Claude"), model_true=np.full(n, "claude-opus-4-8"))
    again = bf.fit_binary_only(blend, {"synthetic_binary": fine_changed}, source, **kwargs)
    assert bf.digest(again) == bf.digest(calibrated)
    checks.append("Fine labels/trust masks cannot affect binary fitting or source calibration")
    for bad in (True, 0, None, np.asarray([False])):
        invalid = {**binary, "calibrated": bad}
        try:
            bf.fit_binary_only(blend, {"synthetic_binary": invalid}, source, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("calibrated must be a scalar bool False")
    for changed in ({"group_ids": np.full(n, " ")}, {"lengths": lengths.astype(float)},
                    {"scope_word_like_tokens": lengths.astype(np.int32)}):
        try:
            bf.fit_binary_only(blend, {"synthetic_binary": {**binary, **changed}}, source, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid group/count/scope metadata must fail")
    unfrozen = copy.deepcopy(protocol)
    unfrozen.pop("status")
    try:
        bf.fit_binary_only(blend, {"synthetic_binary": binary}, source, **{**kwargs, "protocol": unfrozen})
    except ValueError:
        pass
    else:
        raise AssertionError("unfrozen protocol must fail")
    no_history = copy.deepcopy(protocol)
    no_history["calibration_role_history"] = {}
    try:
        bf.fit_binary_only(blend, {"synthetic_binary": binary}, source, **{**kwargs, "protocol": no_history})
    except ValueError:
        pass
    else:
        raise AssertionError("registered CAL replay role history must be explicit and nonempty")
    checks.append("Unfrozen policy, calibrated/nonboolean flags, malformed groups and converted scope counts fail closed")
    arrays = {"synthetic_binary": binary}
    for name in bf.SOURCE_WEIGHTS:
        arrays[name] = {**common, "component_weights_sha256": np.asarray(components[name]["weights_sha256"]),
                        "component_metadata_sha256": np.asarray(components[name]["metadata_sha256"])}
    reference = arrays[bf.SOURCE_NAME]
    reference["raw_family"] = rng.normal(size=(n, 7))
    reference["raw_version"] = rng.normal(size=(n, 25))
    stat_labels = source["version_refinement"]["artifact"]["stat_model_labels"]
    arrays[bf.STAT_NAME]["raw_model"] = rng.normal(size=(n, len(stat_labels)))
    old_h = HierarchicalCalibration.from_dict(source["hierarchy"])
    expected = old_h.predict(reference["raw_family"], {f: reference["raw_version"][:, VERSION_INDICES[f]] for f in FAMILIES})
    expected_leaf = version_refinement.apply(source["version_refinement"]["artifact"], expected["family"], expected["leaf"],
        reference["raw_version"], arrays[bf.STAT_NAME]["raw_model"])
    result = bf.apply_rankings_only(blend, calibrated, arrays)
    assert result["p_family"].dtype == expected["family"].dtype and result["p_family"].tobytes() == expected["family"].tobytes()
    assert result["p_model"].dtype == expected_leaf.dtype and result["p_model"].tobytes() == expected_leaf.tobytes()
    for j, family in enumerate(FAMILIES):
        assert np.allclose(result["p_model"][:, VERSION_INDICES[family]].sum(1), result["p_family"][:, j], atol=1e-12, rtol=0)
    assert not result["family_accept"].any() and not result["model_accept"].any()
    assert result["source_release_family_enabled"].item() is False and result["source_release_model_enabled"].item() is False
    checks.append("Old conditional7/25probabilities bitwise unchanged; family/leaf mass conserved; offline flags all off")
    tampered = copy.deepcopy(calibrated)
    tampered["hierarchy"]["family"]["value"] *= 1.1
    try:
        bf.apply_rankings_only(blend, tampered, arrays)
    except ValueError:
        pass
    else:
        raise AssertionError("changed inherited source parameters must fail")
    checks.append("Changed inherited source objects fail closed")
    misaligned = {**arrays, "synthetic_binary": {**binary, "group_ids": binary["group_ids"][::-1]}}
    try:
        bf.apply_rankings_only(blend, calibrated, misaligned)
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned component groups must fail")
    checks.append("Misaligned raw component groups fail before applying probabilities")
    # Real online arithmetic and decision code, but fake raw-score models and a
    # synthetic guard: no detector checkpoint or transformer forward is loaded.
    selected = [0, 5]
    class Guard:
        def check(self, text):
            return {"supported": True, "word_like_tokens": len(text.split())}
    class Fake:
        def __init__(self, raw, classes=None):
            self.raw, self.classes_ = raw, classes
        def predict_raw_scores(self, texts, **kwargs):
            assert len(texts) == len(selected)
            return self.raw
    live = object.__new__(EnglishDetector)
    live.blend, live.binary, live.hierarchy = blend, BinaryPlatt(**calibrated["binary"]), HierarchicalCalibration.from_dict(calibrated["hierarchy"])
    live.version_refinement, live.thresholds = calibrated["version_refinement"], calibrated["thresholds"]
    live.source_assertions, live.language_guard = dict(bf.OFF), Guard()
    live.attribution_name, live.model_id, live.batch_size, live.device = bf.SOURCE_NAME, "synthetic-only", 4, "cpu"
    live.metadata = {"models": {name: {"type": "statistical" if name == bf.STAT_NAME else "neural"} for name in arrays}}
    live.models = {}
    for name, a in arrays.items():
        raw = {"binary": a["raw_binary"][selected]}
        if name == bf.SOURCE_NAME:
            raw.update(family=a["raw_family"][selected], version=a["raw_version"][selected],
                encoder_tokens=a["encoder_tokens"][selected], encoder_coverage=a["encoder_coverage"][selected], window_counts=a["window_counts"][selected])
        if name == bf.STAT_NAME:
            raw["model"] = a["raw_model"][selected]
        live.models[name] = Fake(raw, {"model": stat_labels} if name == bf.STAT_NAME else None)
    outputs = live.predict_many(["word "*int(lengths[i]) for i in selected])
    for index, output in zip(selected, outputs):
        assert output["accepted_family"] is None and output["accepted_model"] is None
        assert output["probabilities"]["AI"] == result["p_ai"][index]
        assert output["status"] == ("AI-generated" if result["decision_AI"][index] else "likely human" if result["decision_human"][index] else "inconclusive")
        assert np.array_equal([next(v["probability"] for v in output["family_probabilities"] if v["label"]==f) for f in FAMILIES], result["p_family"][index])
        assert np.array_equal([next(v["probability"] for v in output["model_probabilities"] if v["label"]==m) for m in MODEL_LABELS], result["p_model"][index])
    checks.append("Unmodified EnglishDetector online arithmetic matches offline probabilities/status and no-gate source abstention")
    # A small file/receipt/CLI round-trip uses only constructed metadata rows.
    import pyarrow as pa
    import pyarrow.parquet as pq
    with tempfile.TemporaryDirectory(prefix="synthetic-v2-binary-finalize-") as temp:
        folder = Path(temp)
        input_path = folder / "synthetic_calibration.parquet"
        pq.write_table(pa.Table.from_pylist([{"id": str(common["ids"][i]), "y_ai": int(y[i]), "split": "calibration", "eligible": True} for i in range(n)]), input_path)
        real_protocol = copy.deepcopy(protocol)
        real_protocol["calibration_input_sha256"] = bf.sha(input_path)
        protocol_path, blend_path = folder / "protocol.json", folder / "blend.json"
        bf.write_new(protocol_path, real_protocol)
        bf.write_new(blend_path, blend)
        a = {**binary, "export_purpose": np.asarray("calibration")}
        raw_path = folder / "synthetic_raw.npz"
        np.savez_compressed(raw_path, **a)
        bf.write_new(raw_path.with_suffix(".provenance.json"), {"purpose": "calibration", "input_sha256": bf.sha(input_path),
            "output_sha256": bf.sha(raw_path), "eligible_export_rows": n, "calibration_fitted": False,
            "weights_sha256": fake_identity["weights_sha256"], "component_metadata_sha256": fake_identity["metadata_sha256"],
            "deployment_scope": {"rule_sha256": bf.language_sha()}, "synthetic_fixture_only": True})
        argv = ["binary_finalize", "fit", "--blend", str(blend_path), "--source-calibration", str(source_path),
            "--protocol", str(protocol_path), "--protocol-sha256", bf.sha(protocol_path), "--input", str(input_path),
            "--input-sha256", bf.sha(input_path), "--predictions", f"synthetic_binary={raw_path}", "--output-dir", str(folder/"new_cal")]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            bf.main()
        saved = json.loads((folder / "new_cal/calibration.json").read_text())
        receipt = json.loads((folder / "new_cal/fit_manifest.json").read_text())
        assert receipt["source_objects_unchanged"] is True and receipt["calibration_sha256"] == bf.sha(folder/"new_cal/calibration.json")
        assert bf.digest(bf.source_objects(saved)) == bf.digest(bf.source_objects(source))
        try:
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                bf.main()
        except ValueError:
            pass
        else:
            raise AssertionError("existing output directory must not be overwritten")
        dev_input = folder / "synthetic_development.parquet"
        pq.write_table(pa.Table.from_pylist([{"id": str(common["ids"][i]), "y_ai": int(y[i]), "split": "validation", "eligible": True} for i in range(n)]), dev_input)
        named = []
        for name, raw in arrays.items():
            path = folder / (name + "_dev.npz")
            np.savez_compressed(path, **{**raw, "export_purpose": np.asarray("development")})
            who = bf.identity(raw)
            bf.write_new(path.with_suffix(".provenance.json"), {"purpose": "development", "input_sha256": bf.sha(dev_input),
                "output_sha256": bf.sha(path), "eligible_export_rows": n, "calibration_fitted": False,
                "weights_sha256": who["weights_sha256"], "component_metadata_sha256": who["metadata_sha256"],
                "deployment_scope": {"rule_sha256": bf.language_sha()}, "synthetic_fixture_only": True})
            named.append(f"{name}={path}")
        eval_argv = ["binary_finalize", "evaluate", "--blend", str(blend_path), "--calibration", str(folder/"new_cal/calibration.json"),
            "--protocol", str(protocol_path), "--protocol-sha256", bf.sha(protocol_path), "--input", str(dev_input),
            "--input-sha256", bf.sha(dev_input), "--purpose", "development", "--predictions", *named,
            "--output-dir", str(folder/"new_eval"), "--n-bootstrap", "0"]
        with patch.object(sys, "argv", eval_argv), redirect_stdout(io.StringIO()):
            bf.main()
        dev_report = json.loads((folder/"new_eval/report.json").read_text())
        assert dev_report["binary_at_probability_half"]["threshold"] == .5
        assert "balanced_accuracy" in dev_report["binary_at_probability_half"]["all_lengths"]
        assert "balanced_accuracy" in dev_report["binary_at_probability_half"]["core_ge_128_word_like_tokens"]
        with np.load(folder/"new_eval/shipping.npz", allow_pickle=False) as output:
            assert not output["family_accept"].any() and not output["model_accept"].any()
    checks.append("Synthetic Parquet/raw receipt/CLI artifact manifest round-trip, no overwrite")
    checks.append("Synthetic evaluation CLI preserves rankings-only and reports fixed0.5 all/core balanced accuracy without fitting")
    assert bf.sha(source_path) == bf.SOURCE_CALIBRATION_SHA256 and bf.digest(source) == before
    return {"passed": True, "synthetic_scores_labels_metadata_only": True, "real_v2_calibration_or_final_opened": False,
        "real_model_forward_or_training": False, "old_scalar_source_calibration_artifact_read_only": True,
        "source_calibration_sha256": bf.SOURCE_CALIBRATION_SHA256, "source_family_leaf_bitwise_identity": True,
        "checks": checks, "check_count": len(checks), "code_sha256": bf.sha(Path(bf.__file__)), "checks_code_sha256": bf.sha(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bf.write_new(args.output, result)
    print(json.dumps(result, indent=2))
