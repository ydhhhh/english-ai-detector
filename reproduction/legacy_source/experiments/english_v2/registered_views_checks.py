"""Synthetic metadata/aggregate oracles for the preregistered rewrite views.

No model, real corpus, calibration/final score, or GPU is opened. The only
Parquet rows here are constructed IDs, binary truth, and boolean membership.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from experiments.english_v1 import runtime  # noqa: F401
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from experiments.english_v2 import acceptance as a, binary_finalize as bf
from experiments.english_v2.acceptance_checks_independent import fixture, write


def run():
    checks = []
    with tempfile.TemporaryDirectory(prefix="synthetic-registered-rewrite-views-") as temp:
        root = Path(temp)
        n = 96
        y = np.asarray([0]*32+[1]*64)
        lengths = np.full(n, 180, dtype=np.int64)
        groups = np.asarray([f"synthetic-parent:{i%32}" for i in range(n)])
        payload = {"ids": np.asarray([f"synthetic-row:{i}" for i in range(n)]), "y_ai": y,
            "p_ai": np.where(y == 1, .97, .03), "lengths": lengths, "group_ids": groups,
            "language_supported": np.ones(n, dtype=bool)}
        payload["language_supported"][[0, 32, 64]] = False
        # These rejected rows remain in both true-class denominators.
        calibration = {"thresholds": {"AI": {"task": "AI_detection", "fit_split": "calibration", "threshold": .8, "comparison": ">="},
            "human": {"task": "human_detection", "fit_split": "calibration", "threshold": .8, "comparison": ">="}}}
        path = root/"synthetic_metadata.parquet"
        rows = [{"id": str(payload["ids"][i]), "y_ai": int(y[i]), "eligible": True,
            "view_human_origin_rewrite": i < 64, "view_AI_origin_rewrite": i < 32 or i >= 64} for i in range(n)]
        pq.write_table(pa.Table.from_pylist(rows), path)
        counts = {"n": 64, "human_n": 32, "AI_n": 32, "groups": 32}
        specs = {name: {"column": "view_"+name, "rows": 64, "groups": 32, "core_counts": counts}
                 for name in ("human_origin_rewrite", "AI_origin_rewrite")}
        protocol = {"necessary_final_suites": {"synthetic": {"input_sha256": bf.sha(path), "required_views": specs}}}
        original = {key: value.tobytes() for key, value in payload.items()}
        # Reverse raw order to prove membership is matched by ID, never row offset.
        reversed_payload = {key: value[::-1] for key, value in payload.items()}
        reports = bf.registered_view_reports(protocol, path, bf.sha(path), reversed_payload, calibration, n_boot=500)
        for name, report in reports.items():
            assert report["registered_view"]["column"] == specs[name]["column"]
            assert report["registered_view"]["core_counts"] == counts
            assert report["group_bootstrap"]["n_boot"] == 500 and report["group_bootstrap"]["n_groups"] == 32
            online = report["core_ge_128_tokens"]["deployed_decision_audit"]
            assert online["AI_n"] == online["human_n"] == 32
            assert online["correct_AI_recall"] == online["correct_human_recall"] == 31/32
            assert online["inconclusive_n"] == 2
            assert report["binary_at_probability_half"]["core_ge_128_word_like_tokens"]["balanced_accuracy"] == 1.
        assert all(payload[key].tobytes() == value for key, value in original.items())
        checks.append("ID-aligned shared human controls; same frozen scores and thresholds; 500 whole-group draws; abstentions retained")
        checks.append("View fixed0.5 metrics and input payload bytes unchanged")
        with patch.object(pq, "ParquetFile", side_effect=AssertionError("undeclared metadata must not be read")):
            assert bf.registered_view_reports({}, path, bf.sha(path), payload, calibration, n_boot=0) == {}
        checks.append("No required_views declaration performs no extra Parquet projection")
        for which in ("count", "null", "nonbool", "bootstrap"):
            altered = copy.deepcopy(protocol)
            bad_path = path
            if which == "count":
                altered["necessary_final_suites"]["synthetic"]["required_views"]["human_origin_rewrite"]["rows"] = 63
            elif which in {"null", "nonbool"}:
                changed_rows = copy.deepcopy(rows)
                if which == "null":
                    changed_rows[0]["view_human_origin_rewrite"] = None
                else:
                    for row in changed_rows:
                        row["view_human_origin_rewrite"] = int(row["view_human_origin_rewrite"])
                bad_path = root/(which+".parquet")
                pq.write_table(pa.Table.from_pylist(changed_rows), bad_path)
                altered["necessary_final_suites"]["synthetic"]["input_sha256"] = bf.sha(bad_path)
            try:
                bf.registered_view_reports(altered, bad_path, bf.sha(bad_path), payload, calibration, n_boot=0 if which == "bootstrap" else 500)
            except ValueError:
                pass
            else:
                raise AssertionError("registered view guard did not reject "+which)
        checks.append("Wrong counts, null/integer membership, and non500 view bootstrap fail closed")

        # Reuse independent, inert JSON/receipt fixture; add required views before
        # hashing the protocol and mechanically update all dependent references.
        folder = root/"aggregates"
        protocol_path, blend_path, cal_path, report_paths = fixture(folder)
        p, cal = json.loads(protocol_path.read_text()), json.loads(cal_path.read_text())
        suite_name = "fresh_paraphrase_with_human_controls"
        spec = p["necessary_final_suites"][suite_name]
        spec["required_views"] = {name: {"column": "view_"+name, "rows": spec["rows"], "groups": spec["groups"],
            "core_counts": copy.deepcopy(spec["core_counts"]), "targets": copy.deepcopy(spec["targets"])} for name in specs}
        write(protocol_path, p)
        cal["provenance"]["v2_protocol_sha256"] = a._sha(protocol_path)
        write(cal_path, cal)
        for name, report_path in report_paths.items():
            report = json.loads(report_path.read_text())
            report["provenance"].update(v2_protocol_sha256=a._sha(protocol_path), calibration_sha256=a._sha(cal_path))
            for evidence in report["provenance"]["raw_exports"].values():
                side = Path(evidence["path"]).with_suffix(".provenance.json")
                receipt = json.loads(side.read_text())
                receipt["frozen_calibration_sha256"] = a._sha(cal_path)
                write(side, receipt)
                evidence["sidecar_sha256"] = a._sha(side)
            if name == suite_name:
                report["registered_views"] = {}
                for view_name, vspec in spec["required_views"].items():
                    view = copy.deepcopy(report)
                    view.pop("registered_views", None)
                    view.pop("provenance", None)  # Parent receipts bind the complete parent N only.
                    view["registered_view"] = {key: copy.deepcopy(vspec[key]) for key in ("column", "rows", "groups", "core_counts")}
                    view["registered_view"].update(parent_input_sha256=spec["input_sha256"],
                        shared_human_controls_not_independent_or_additive=True, probabilities_or_thresholds_refitted=False)
                    report["registered_views"][view_name] = view
            write(report_path, report)
        def summarize():
            return a.summarize(protocol_path, a._sha(protocol_path), blend_path, cal_path, report_paths)
        assert summarize()["all_necessary_suites_pass"] is True
        checks.append("All four suites plus both registered views pass only with parent complete-N receipts")
        target_path = report_paths[suite_name]
        baseline = json.loads(target_path.read_text())
        alterations = ("missing", "failed_human_origin", "denominator", "missing_CI", "refit", "threshold", "shared_controls", "bootstrap")
        for which in alterations:
            changed = copy.deepcopy(baseline)
            view = changed["registered_views"]["human_origin_rewrite"]
            if which == "missing":
                changed["registered_views"].pop("human_origin_rewrite")
            elif which == "failed_human_origin":
                view["core_ge_128_tokens"]["binary"]["auroc"] = .85
                view["group_bootstrap"]["intervals95"]["core_ge_128_tokens.binary.auroc"]["point"] = .85
            elif which == "denominator":
                view["registered_view"]["rows"] -= 1
            elif which == "missing_CI":
                view["group_bootstrap"]["intervals95"].pop("core_ge_128_tokens.binary.auroc")
            elif which == "refit":
                view["registered_view"]["probabilities_or_thresholds_refitted"] = True
            elif which == "threshold":
                view["threshold"] = {**view["threshold"], "threshold": .9}
            elif which == "shared_controls":
                view["registered_view"]["shared_human_controls_not_independent_or_additive"] = False
            else:
                view["group_bootstrap"]["n_boot"] = 499
            write(target_path, changed)
            result = summarize()
            assert result["all_necessary_suites_pass"] is False and result["suites"][suite_name]["pass"] is False, which
        write(target_path, baseline)
        checks.append("Missing/failed rewrite view cannot be masked by passing parent/other view; denominator/CI/threshold/refit/nonindependence/500 guards")
    return {"passed": True, "synthetic_metadata_aggregates_only": True, "real_data_or_scores_read": False,
        "model_forward_or_training": False, "checks": checks, "check_count": len(checks),
        "binary_finalize_sha256": bf.sha(bf.__file__), "acceptance_sha256": bf.sha(a.__file__), "checks_sha256": bf.sha(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bf.write_new(args.output, result)
    print(json.dumps(result, indent=2))
