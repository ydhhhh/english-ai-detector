"""Validation-only compatibility for the frozen stat2 int32 count storage.

Never edits raw NPZ, calibration, model or frozen producer files. Execution
requires an exact-SHA reviewed amendment. Only the four registered final inputs
are allowed; fit/development are unavailable. Probability arithmetic is the
unchanged binary_finalize/post_final_breakdown implementation.

  python -B -m experiments.english_v2.count_storage_compat evaluate \
    --amendment FROZEN_JSON --amendment-sha256 SHA -- <original evaluate flags>
  python -B -m experiments.english_v2.count_storage_compat post-diagnostics \
    --amendment FROZEN_JSON --amendment-sha256 SHA -- <original diagnostic flags>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

from experiments.english_v1 import runtime  # noqa: F401
import numpy as np
from experiments.english_v2 import binary_finalize as bf

ROOT = Path(__file__).resolve().parents[2]
FROZEN_STATUS = "FROZEN_MECHANICAL_COUNT_STORAGE_COMPATIBILITY_BEFORE_FINAL_METRICS"
STAT_SHA = "32a35bcfa2fc09a49d841bf2f2ad67f3d0d5fa926590c282f4dd14363c41fc45"
SCOPE_ERROR = "language-scope word counts must match original counts without conversion"
FIXED_SHA = {
    "protocol": "276c4757774c0f1867470a040c0a303209a6a2444e8481d4402e4693ce221da4",
    "blend": "df0a837b1ce7016d56373d06f822fa48d36a9e969bd525051fd1ae83ceef4628",
    "calibration": "31330e39f471587ad50156c041717b326b1cbfcd6e0230ef6ff32887c653d50c",
    "original_finalizer": "182f82e85c71f59de997b13da89bda7193c179b748c5f6185f312bbaad67e83c",
    "post_diagnostics_source": "dbb6884f04f71193231f4f649bcb53a60bfaca27af2e080b8b257e023d368ede",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def path(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT/p).resolve()


def bound(spec):
    p = path(spec["path"])
    require(p.is_file() and sha(p) == spec["sha256"], "amendment-bound file SHA mismatch")
    return p


def storage_validator(original):
    """Return an exact stat-only exception; original payload is untouched.

    Original validation must first fail at its known representation-only
    check. All its remaining checks run again on a shallow validation proxy.
    The proxy aliases scope to lengths only after their exact integer values
    agree. No actual payload, probabilities, labels or counts are converted.
    """
    def validate(payload):
        try:
            return original(payload)
        except ValueError as error:
            if str(error) != SCOPE_ERROR:
                raise
            lengths = np.asarray(payload["lengths"])
            scope = np.asarray(payload["scope_word_like_tokens"])
            weight = np.asarray(payload.get("component_weights_sha256", ""))
            metadata = np.asarray(payload.get("component_metadata_sha256", ""))
            require(weight.shape == metadata.shape == () and str(weight.item()) == STAT_SHA and
                str(metadata.item()) == STAT_SHA, "only the pinned original statistical_v2 payload permits count storage compatibility")
            require(lengths.dtype == np.dtype("int32") and scope.dtype == np.dtype("int64") and
                lengths.ndim == scope.ndim == 1 and lengths.shape == scope.shape and
                (lengths >= 0).all() and (scope >= 0).all() and np.array_equal(lengths, scope),
                "stat compatibility requires native int32/int64, nonnegative same-shape exactly equal integer counts")
            proxy = dict(payload)
            proxy["scope_word_like_tokens"] = payload["lengths"]
            result = original(proxy)
            validate.accepted_storage_exceptions += 1
            return result
    validate.accepted_storage_exceptions = 0
    return validate


def arguments(operation, argv):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--protocol", required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--blend", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output-dir", required=True)
    if operation == "evaluate":
        p.add_argument("--purpose", required=True, choices=("final",))
        p.add_argument("--input", required=True)
        p.add_argument("--input-sha256", required=True)
        p.add_argument("--predictions", required=True, nargs="+")
        p.add_argument("--n-bootstrap", type=int, default=500)
    else:
        p.add_argument("--blend-sha256", required=True)
        p.add_argument("--calibration-sha256", required=True)
        p.add_argument("--reports", required=True, nargs="+")
    return p.parse_args(argv)


def prepare(amendment_path, amendment_sha, operation, argv):
    require(operation in {"evaluate", "post-diagnostics"}, "only fixed final evaluation and post-diagnostics are allowed; fit is forbidden")
    amendment_path = path(amendment_path)
    require(bf.is_sha(amendment_sha) and sha(amendment_path) == amendment_sha, "explicit amendment SHA mismatch")
    amendment = json.loads(amendment_path.read_text())
    require(amendment.get("schema_version") == 1 and amendment.get("status") == FROZEN_STATUS and
        amendment.get("validation_only") is True and amendment.get("probabilities_or_thresholds_changed") is False,
        "reviewed frozen mechanical amendment required; DRAFT cannot execute")
    artifacts = amendment["artifacts"]
    require(all(artifacts[name]["sha256"] == expected for name,expected in FIXED_SHA.items()),
        "compatibility cannot substitute the original protocol, frozen blend/calibration or original producers")
    files = {name: bound(spec) for name, spec in artifacts.items()}
    require(files["compatibility_source"] == Path(__file__).resolve() and
        files["original_finalizer"] == Path(bf.__file__).resolve() and
        files["post_diagnostics_source"] == ROOT/"experiments/english_v2/post_final_breakdown.py",
        "amendment bound unexpected execution sources")
    args = arguments(operation, argv)
    for field in ("protocol", "blend", "calibration"):
        require(path(getattr(args, field)) == files[field], "CLI artifact path differs from frozen amendment")
    require(args.protocol_sha256 == artifacts["protocol"]["sha256"], "CLI protocol SHA differs")
    protocol = json.loads(files["protocol"].read_text())
    require(protocol.get("status") == "FROZEN_BEFORE_V2_TRAINING_AND_SCORES" and
        set(amendment["registered_inputs"]) == set(protocol["necessary_final_suites"]), "exact four original registered suites required")
    cal = json.loads(files["calibration"].read_text())
    require(cal["provenance"]["v2_protocol_sha256"] == artifacts["protocol"]["sha256"] and
        cal["provenance"]["blend_sha256"] == artifacts["blend"]["sha256"], "frozen calibration/protocol/blend differ")
    audit = json.loads(files["count_dtype_audit"].read_text())
    require(audit.get("counts_only_no_probability_values_loaded") is True and len(audit["raw_files"]) == 16,
        "complete counts-only sixteen-raw audit required")
    by_suite = {}
    for row in audit["raw_files"]:
        p = bound({"path": row["path"], "sha256": row["npz_sha256"]})
        bound({"path": row["sidecar_path"], "sha256": row["sidecar_sha256"]})
        require(row["component"] not in by_suite.setdefault(row["suite"], {}), "duplicate raw audit component")
        by_suite[row["suite"]][row["component"]] = p
    require(set(by_suite) == set(amendment["registered_inputs"]), "raw audit suite directory differs")
    for suite, spec in amendment["registered_inputs"].items():
        require(spec["sha256"] == protocol["necessary_final_suites"][suite]["input_sha256"] and
            spec["rows"] == protocol["necessary_final_suites"][suite]["rows"], "amendment cannot change a registered input/denominator")
        require(set(by_suite[suite]) == set(cal["provenance"]["component_weights_sha256"]), "raw audit must retain every original selected component")
        bound(spec)
    if operation == "evaluate":
        selected = [s for s,v in amendment["registered_inputs"].items() if path(v["path"]) == path(args.input) and v["sha256"] == args.input_sha256]
        require(len(selected) == 1 and args.n_bootstrap == 500, "only an unchanged registered final input with fixed500 bootstrap is allowed")
        named = {n: path(p) for n,p in bf.parse_named(args.predictions).items()}
        require(named == by_suite[selected[0]], "evaluation must use the exact sixteen-audit original raw cohort components")
    else:
        require(args.blend_sha256 == artifacts["blend"]["sha256"] and args.calibration_sha256 == artifacts["calibration"]["sha256"],
            "diagnostic CLI artifact SHA differs")
        reports = bf.parse_named(args.reports)
        require(set(reports) == set(amendment["registered_inputs"]), "all four original necessary report paths required")
        for report in reports.values():
            value = json.loads(path(report).read_text())
            stamp = value.get("provenance", {}).get("count_storage_compatibility", {})
            require(stamp.get("amendment_sha256") == amendment_sha and
                stamp.get("compatibility_source_sha256") == artifacts["compatibility_source"]["sha256"],
                "post-diagnostics requires explicitly compatibility-stamped completed reports")
    require(not path(args.output_dir).exists(), "compatibility execution requires a new output directory")
    return amendment, files, args


def execute(amendment_path, amendment_sha, operation, argv):
    amendment, files, args = prepare(amendment_path, amendment_sha, operation, argv)
    original = bf.validate_raw_rows
    replacement = storage_validator(original)
    old_argv = sys.argv
    bf.validate_raw_rows = replacement
    try:
        if operation == "evaluate":
            sys.argv = [str(files["original_finalizer"]), "evaluate", *argv]
            bf.main()
            result_path = path(args.output_dir)/"report.json"
        else:
            sys.argv = [str(files["post_diagnostics_source"]), *argv]
            runpy.run_module("experiments.english_v2.post_final_breakdown", run_name="__main__")
            result_path = path(args.output_dir)/"breakdown.json"
        require(all(sha(p) == amendment["artifacts"][name]["sha256"] for name,p in files.items()),
            "an amendment-bound artifact/source changed during execution")
        for row in json.loads(files["count_dtype_audit"].read_text())["raw_files"]:
            bound({"path": row["path"], "sha256": row["npz_sha256"]})
            bound({"path": row["sidecar_path"], "sha256": row["sidecar_sha256"]})
        for spec in amendment["registered_inputs"].values():
            bound(spec)
        result = json.loads(result_path.read_text())
        provenance = result["provenance"]
        require("count_storage_compatibility" not in provenance, "compatibility provenance already exists")
        provenance["count_storage_compatibility"] = {
            "amendment_sha256": amendment_sha, "compatibility_source_sha256": sha(__file__),
            "original_finalizer_sha256": amendment["artifacts"]["original_finalizer"]["sha256"],
            "post_diagnostics_source_sha256": amendment["artifacts"]["post_diagnostics_source"]["sha256"],
            "count_dtype_audit_sha256": amendment["artifacts"]["count_dtype_audit"]["sha256"],
            "validation_only": True, "original_raw_arrays_modified": False,
            "probabilities_labels_thresholds_or_algorithms_changed": False,
            "storage_exception": "pinned statistical_v2 only: exact equal native int32 lengths versus int64 scope counts; shallow validator proxy only",
            "accepted_storage_exception_calls": replacement.accepted_storage_exceptions,
            "original_code_sha256_field_identifies_algorithm_file_not_complete_execution_chain": True}
        temporary = result_path.with_suffix(".compatibility.part")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
        temporary.replace(result_path)
    finally:
        bf.validate_raw_rows = original
        sys.argv = old_argv
    print(json.dumps({"completed": True, "operation": operation, "compatibility_provenance_recorded": True,
        "amendment_sha256": amendment_sha, "output": str(result_path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("operation", choices=("evaluate", "post-diagnostics"))
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--amendment-sha256", required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    execute(args.amendment, args.amendment_sha256, args.operation, forwarded)
