"""JSON-only release gate for nominal family/version assertions.

Candidate reports are immutable inputs. This module never opens predictions,
corpus files, checkpoints or probability calibrators, and never refits anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import ntpath
from pathlib import Path
import posixpath

from experiments.english_v1.release_acceptance import _aims, _count, _hash, _prob

PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT / "research/english_v1_data/source_assertion_release_protocol_v1.json"
PROTOCOL_SHA256 = "4e305da0f94122383f210ba68711801025cb3b065c457ef16449a54ea1132715"
IDENTITIES = ("blend_sha256", "calibration_sha256", "language_rule_sha256", "deployment_decision_protocol_sha256")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _parse(raw):
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _read(path):
    path = Path(path)
    if path.suffix.casefold() != ".json":
        raise ValueError("source-release gate reads JSON only")
    raw = path.read_bytes()
    return _parse(raw), _sha(raw), raw.decode("utf-8")


def _resolve(path, root):
    path = Path(path)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def _historical_path(path, root):
    """Normalize recorded paths lexically, including foreign-OS audit paths."""
    path, root = str(path), str(root)
    module = ntpath if ntpath.splitdrive(root)[0] or root.startswith("\\\\") else posixpath
    value = path if module.isabs(path) else module.join(root, path)
    return module.normcase(module.normpath(value))


def _protocol(path):
    value, digest, _ = _read(path)
    if digest != PROTOCOL_SHA256:
        raise ValueError("source assertion release protocol differs from the preregistered bytes")
    return value


def _snapshot(report):
    keys = ("constructed_engineering_only", "length_unit", "provenance", "threshold", "all_lengths", "core_ge_128_tokens",
            "supported_catalog_attribution", "group_bootstrap")
    return {key: report.get(key) for key in keys}


def _identity(role, evidence, protocol, generation_root):
    """Recheck embedded report/sidecar evidence without reopening their files."""
    errors, components = [], {}
    snap = evidence.get("candidate_snapshot")
    if not isinstance(snap, dict) or evidence.get("candidate_snapshot_sha256") != _sha(_canonical(snap)):
        raise ValueError("candidate snapshot missing or changed")
    if not _hash(evidence.get("report_sha256")):
        errors.append("original candidate report hash missing")
    p = snap.get("provenance")
    if not isinstance(p, dict):
        raise ValueError("candidate report provenance missing")
    for key in IDENTITIES:
        if not _hash(p.get(key)):
            errors.append("invalid " + key)
    if p.get("language_rule_sha256") != protocol["language_rule"]["sha256"]:
        errors.append("report language guard differs from protocol")
    if p.get("deployment_decision_protocol_sha256") != protocol["deployment_decision_protocol"]["sha256"]:
        errors.append("report decision policy differs from amendment07")
    if p.get("thresholds_refitted_on_evaluation") is not False:
        errors.append("candidate must declare no final threshold refitting")
    if snap.get("length_unit") not in {"word_like_tokens", "word_like_tokens_regex"}:
        errors.append("core is not original-text word-like units")
    expected = protocol["cohorts"][role]
    all_binary = (snap.get("all_lengths") or {}).get("binary") or {}
    core = snap.get("core_ge_128_tokens") or {}
    binary = core.get("binary") or {}
    if not isinstance(all_binary, dict) or all_binary.get("n") != expected["rows"]:
        errors.append("all-length report omits fixed input rows")
    if not isinstance(binary, dict) or not all(_count(binary.get(k)) for k in ("n", "human_n", "AI_n")) or binary.get("n", 0) <= 0 or binary.get("AI_n", 0) <= 0:
        errors.append("core true-class denominators missing")
    elif binary["n"] != binary["human_n"] + binary["AI_n"] or binary["n"] > expected["rows"]:
        errors.append("core true-class denominators inconsistent")
    threshold = snap.get("threshold") or {}
    if not isinstance(threshold, dict) or threshold.get("fit_split") != "calibration" or threshold.get("task") != "AI_detection" or threshold.get("comparison") != ">=":
        errors.append("candidate AI threshold is not frozen calibration evidence")
    else:
        cut = threshold.get("threshold")
        if not isinstance(cut, (int, float)) or isinstance(cut, bool) or not math.isfinite(cut) or cut <= .5:
            errors.append("candidate AI threshold does not require strict majority")
    inputs = p.get("inputs")
    sides = evidence.get("input_sidecars")
    if not isinstance(inputs, dict) or not inputs or not isinstance(sides, dict) or set(inputs) != set(sides):
        raise ValueError("complete final raw-export sidecars are required")
    for name, info in inputs.items():
        try:
            if not isinstance(info, dict) or not _hash(info.get("sha256")):
                raise ValueError("invalid evaluated prediction identity")
            wrapper = sides[name]
            raw = wrapper["raw_json"].encode("utf-8")
            if _sha(raw) != wrapper.get("sha256"):
                raise ValueError("embedded sidecar bytes changed")
            side = _parse(raw)
            output_path = _historical_path(info["path"], generation_root)
            if _historical_path(wrapper["path"], generation_root) != ntpath.splitext(output_path)[0] + ".provenance.json":
                raise ValueError("sidecar path differs from evaluated component")
            if side.get("purpose") != "final" or side.get("calibration_fitted") is not False:
                raise ValueError("raw export is not final-only")
            if side.get("output_sha256") != info["sha256"] or _historical_path(side["output"], generation_root) != output_path:
                raise ValueError("sidecar output differs from evaluator input")
            if side.get("input_sha256") != expected["sha256"] or _historical_path(side["input"], generation_root) != _historical_path(expected["path"], generation_root):
                raise ValueError("wrong frozen main/modernAI source corpus")
            if side.get("original_rows") != expected["rows"] or side.get("eligible_export_rows") != expected["rows"] or side.get("excluded_ineligible_rows") != 0:
                raise ValueError("candidate export does not cover the complete fixed corpus")
            if any(side.get("frozen_" + k) != p.get(k) for k in ("blend_sha256", "calibration_sha256")):
                raise ValueError("sidecar frozen blend/calibration differs")
            if (side.get("deployment_scope") or {}).get("rule_sha256") != p.get("language_rule_sha256"):
                raise ValueError("sidecar language guard differs")
            if not _hash(side.get("weights_sha256")) or not _hash(side.get("component_metadata_sha256")):
                raise ValueError("component weight/configuration identity missing")
            components[name] = {"weights_sha256": side["weights_sha256"], "metadata_sha256": side["component_metadata_sha256"]}
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(str(name) + ":" + str(exc))
    return errors, {key: p.get(key) for key in IDENTITIES}, components, threshold


def _metrics(evidence, admitted):
    snap = evidence["candidate_snapshot"]
    aims = _aims(snap)
    binary = (snap.get("core_ge_128_tokens") or {}).get("binary") or {}
    out = {}
    for head in ("family", "model"):
        aim = aims[head]
        checks = list(aim["checks"])
        accepted = checks[0]["cohort"]["accepted_trusted_AI_n"]
        supported = checks[0]["cohort"]["trusted_AI_n"]
        errors = list(aim["cohort_errors"])
        audit = aim["catalog_assertion_audit"]
        point=((snap.get("supported_catalog_attribution") or {}).get("core_ge_128_tokens") or {}).get(head) or {}
        supports=(point.get("conditional") or {}).get("class_support") or {}
        accepted_supports=(point.get("accepted_conditional") or {}).get("class_support") or {}
        caps=admitted.get("masked_family_counts" if head=="family" else "masked_model_counts") or {}
        if any(not _count(n) or n>caps.get(label,0) for label,n in supports.items()):
            errors.append("core trusted label support exceeds the frozen input trusted-mask counts")
        if any(not _count(n) or n>supports.get(label,0) for label,n in accepted_supports.items()):
            errors.append("accepted label support exceeds its trusted core support")
        if _count(supported) and _count(binary.get("AI_n")) and supported > binary["AI_n"]:
            errors.append("trusted named-catalog support exceeds all core AI")
        if not _count(audit.get("human_n")) or audit.get("human_n") != binary.get("human_n"):
            errors.append("human assertion audit denominator differs from the core")
        counts = ("correct_asserted_on_trusted_AI_n", "wrong_asserted_on_trusted_AI_n", "asserted_on_humans_n", "asserted_on_AI_without_trusted_source_label_n")
        if not all(_count(audit.get(k)) for k in counts):
            errors.append("candidate source assertion counts missing")
        elif (not _count(accepted) or audit[counts[0]] + audit[counts[1]] != accepted or
              audit[counts[2]] > binary.get("human_n", -1) or
              not _count(supported) or audit[counts[3]] > binary.get("AI_n", -1) - supported):
            errors.append("candidate source assertion counts inconsistent")
        minimum = {"name": "accepted_n", "actual": accepted, "target": 100, "comparison": ">=",
                   "pass": _count(accepted) and accepted >= 100,
                   "status": "missing" if accepted is None else "pass" if _count(accepted) and accepted >= 100 else "fail"}
        checks.append(minimum)
        out[head] = {"pass": not errors and all(c["pass"] is True for c in checks), "checks": checks,
                     "errors": errors, "candidate_aim": aim}
    family_counts=out["family"]["checks"][0]["cohort"]
    model_counts=out["model"]["checks"][0]["cohort"]
    for key in ("trusted_AI_n","accepted_trusted_AI_n"):
        f,m=family_counts.get(key),model_counts.get(key)
        if _count(f) and _count(m) and m>f:
            out["model"]["errors"].append("model "+key+" exceeds its prerequisite family cohort")
            out["model"]["pass"]=False
    return out


def _derive(evidence, protocol, generation_root):
    errors, identities, components, thresholds, metrics = [], {}, {}, {}, {}
    try:
        wrapper=evidence["data_manifest"]
        raw=wrapper["raw_json"].encode("utf-8")
        if _sha(raw)!=wrapper.get("sha256") or _sha(raw)!=protocol["data_manifest"]["sha256"]:
            raise ValueError("embedded data-manifest bytes differ from preregistration")
        manifest=_parse(raw)
        if manifest.get("frozen") is not True:raise ValueError("embedded data manifest is not frozen")
    except (ValueError,KeyError,TypeError,AttributeError) as exc:
        errors.append("data_manifest:"+str(exc))
        manifest={}
    for role in ("main", "modernAI"):
        try:
            item = evidence[role]
            if item.get("errors"):
                raise ValueError("; ".join(item["errors"]))
            ee, identities[role], components[role], thresholds[role] = _identity(role, item, protocol, generation_root)
            errors.extend(role + ":" + text for text in ee)
            admitted=manifest["output_files"][protocol["cohorts"][role]["manifest_role"]]
            if any(admitted.get(k)!=protocol["cohorts"][role][k] for k in ("path","sha256","rows")):
                raise ValueError("embedded fixed source role differs from preregistration")
            metrics[role] = _metrics(item,admitted)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(role + ":" + str(exc))
            metrics[role] = {h: {"pass": False, "errors": [str(exc)]} for h in ("family", "model")}
    if set(identities) == {"main", "modernAI"}:
        if identities["main"] != identities["modernAI"]:
            errors.append("main/modernAI blend/calibration/guard/decision identities differ")
        if components["main"] != components["modernAI"]:
            errors.append("main/modernAI component names, weights or configurations differ")
        if thresholds["main"] != thresholds["modernAI"]:
            errors.append("main/modernAI frozen AI thresholds differ")
    family = not errors and all(metrics[r]["family"]["pass"] for r in metrics)
    model_candidate = not errors and all(metrics[r]["model"]["pass"] for r in metrics)
    return {"valid": not errors, "family_enabled": bool(family), "model_enabled": bool(family and model_candidate),
            "model_candidate_gate_pass": bool(model_candidate), "errors": errors, "cohort_head_checks": metrics,
            "common_identity": identities.get("main", {}), "components": components.get("main", {})}


def _dependencies():
    return {name: _sha(Path(__file__).with_name(name).read_bytes()) for name in ("source_release_gate.py", "release_acceptance.py", "catalog.py")}


def build_gate(main_report=None, modern_report=None, *, protocol_path=PROTOCOL_PATH, data_manifest_path=None, project_root=PROJECT):
    """Build a disabled-or-enabled release artifact from immutable JSON reports."""
    global_errors, evidence, protocol = [], {}, None
    try:
        protocol = _protocol(protocol_path)
        manifest_path = data_manifest_path or _resolve(protocol["data_manifest"]["path"], project_root)
        manifest, digest, raw_manifest = _read(manifest_path)
        if digest != protocol["data_manifest"]["sha256"] or manifest.get("frozen") is not True:
            raise ValueError("data manifest differs from the preregistered frozen identity")
        for role, spec in protocol["cohorts"].items():
            entry = manifest["output_files"][spec["manifest_role"]]
            if any(entry.get(k) != spec[k] for k in ("path", "sha256", "rows")):
                raise ValueError("frozen cohort differs from preregistration:" + role)
        evidence["data_manifest"] = {"path": str(Path(manifest_path).resolve()), "sha256": digest,"raw_json":raw_manifest}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        global_errors.append(str(exc))
    for role, path in (("main", main_report), ("modernAI", modern_report)):
        item = {"report_path": str(Path(path).resolve()) if path is not None else None, "errors": []}
        evidence[role] = item
        try:
            if path is None:
                raise ValueError("required candidate final report missing")
            report, digest, _ = _read(path)
            if report.get("source_assertion_release_gate_applied") or report.get("published_source_assertion_gate_applied"):
                raise ValueError("release gate requires original pre-global-gate candidate metrics")
            snap = _snapshot(report)
            item.update(report_sha256=digest, candidate_snapshot=snap, candidate_snapshot_sha256=_sha(_canonical(snap)))
            sides = {}
            for name, info in (snap.get("provenance") or {}).get("inputs", {}).items():
                sidepath = _resolve(info["path"], project_root).with_suffix(".provenance.json")
                _, side_sha, raw_text = _read(sidepath)
                sides[name] = {"path": str(sidepath), "sha256": side_sha, "raw_json": raw_text}
            item["input_sidecars"] = sides
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            item["errors"].append(str(exc))
    derived = _derive(evidence, protocol, str(Path(project_root).resolve())) if protocol is not None else {
        "valid": False, "family_enabled": False, "model_enabled": False, "model_candidate_gate_pass": False,
        "errors": [], "cohort_head_checks": {}, "common_identity": {}, "components": {}}
    global_errors += derived["errors"]
    enabled_f = not global_errors and derived["family_enabled"]
    enabled_m = not global_errors and derived["model_enabled"]
    provenance = {**{key: derived["common_identity"].get(key) for key in IDENTITIES},
                  "source_assertion_release_protocol_sha256": PROTOCOL_SHA256,
                  "main_report_sha256": evidence["main"].get("report_sha256"),
                  "modern_report_sha256": evidence["modernAI"].get("report_sha256")}
    return {"schema_version": 1, "kind": "source_assertion_release_gate", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "constructed_engineering_only":any(bool((evidence.get(role,{}).get("candidate_snapshot") or {}).get("constructed_engineering_only")) for role in ("main","modernAI")),
            "valid": not global_errors, "family_enabled": bool(enabled_f), "model_enabled": bool(enabled_m),
            "status": "closed_invalid_or_missing_evidence" if global_errors else "assertions_enabled" if enabled_m else "family_only" if enabled_f else "rankings_only",
            "provenance": provenance, "generation_project_root": str(Path(project_root).resolve()),
            "protocol_path": str(Path(protocol_path).resolve()), "producer_code_sha256": _dependencies(),
            "evidence": evidence, "components": derived["components"], "cohort_head_checks": derived["cohort_head_checks"],
            "model_candidate_gate_pass": derived["model_candidate_gate_pass"], "global_errors": global_errors,
            "published_effect": {"disabled_accepted_family": None, "disabled_accepted_model": None,
                "family_assertion_coverage_if_disabled": 0., "model_assertion_coverage_if_disabled": 0.,
                "assertion_accuracy_if_disabled": None, "model_requires_family_enabled": True,
                "candidate_acceptance_statistics_preserved_in_snapshots": True,
                "probabilities_rankings_unknown_mass_binary_weights_calibrators_thresholds_changed": False},
            "final_selection_disclosure": "Final main and modernAI evidence selects only these release switches under a fixed preregistered rule; original candidate results remain available. This is not an untouched final estimate of an independently chosen assertion policy.",
            "prediction_or_corpus_arrays_opened": False, "refits_or_threshold_changes": False}


def load_gate(path, *, expected, protocol_path=PROTOCOL_PATH):
    """Fail closed against caller-owned selected-artifact fingerprints.

    expected requires IDENTITIES, source_assertion_release_protocol_sha256,
    plus components={name:{weights_sha256,
    metadata_sha256}}. Optional gate_sha256 binds a packaged file manifest.
    No original report, sidecar, data manifest, NPZ or corpus is reopened.
    """
    result = {"valid": False, "family_enabled": False, "model_enabled": False, "errors": []}
    try:
        protocol = _protocol(protocol_path)
        gate, digest, _ = _read(path)
        result["gate_sha256"] = digest
        if expected.get("gate_sha256") is not None and digest != expected["gate_sha256"]:
            raise ValueError("gate file differs from packaged manifest")
        if gate.get("schema_version") != 1 or gate.get("kind") != "source_assertion_release_gate":
            raise ValueError("unknown source-gate schema")
        if gate.get("valid") is not True:
            raise ValueError("gate was not built from valid complete candidate evidence")
        if type(gate.get("family_enabled")) is not bool or type(gate.get("model_enabled")) is not bool:
            raise ValueError("strict boolean source enablement required")
        if gate.get("producer_code_sha256") != _dependencies():
            raise ValueError("gate producer/dependency code differs from deployment")
        p = gate.get("provenance") or {}
        if p.get("source_assertion_release_protocol_sha256") != PROTOCOL_SHA256 or expected.get("source_assertion_release_protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError("wrong source-release protocol")
        for key in IDENTITIES:
            if not _hash(expected.get(key)) or p.get(key) != expected[key]:
                raise ValueError("gate differs from selected " + key)
        if not isinstance(expected.get("components"), dict) or not expected["components"] or gate.get("components") != expected["components"]:
            raise ValueError("gate differs from selected component weights/configuration")
        evidence = gate.get("evidence") or {}
        if (evidence.get("data_manifest") or {}).get("sha256") != protocol["data_manifest"]["sha256"]:
            raise ValueError("gate was not built against the registered data manifest")
        for role, key in (("main", "main_report_sha256"), ("modernAI", "modern_report_sha256")):
            if not _hash(p.get(key)) or (evidence.get(role) or {}).get("report_sha256") != p[key]:
                raise ValueError("gate report fingerprint missing or inconsistent")
        derived = _derive(evidence, protocol, gate["generation_project_root"])
        if derived["errors"] or gate.get("global_errors"):
            raise ValueError("invalid candidate final evidence:" + "; ".join(derived["errors"] + gate.get("global_errors", [])))
        if derived["common_identity"] != {k: p[k] for k in IDENTITIES} or derived["components"] != expected["components"]:
            raise ValueError("embedded candidate evidence differs from selected release identities")
        if (gate["family_enabled"], gate["model_enabled"]) != (derived["family_enabled"], derived["model_enabled"]):
            raise ValueError("stored enablement does not follow preregistered candidate metrics")
        result.update(valid=True, family_enabled=derived["family_enabled"], model_enabled=derived["model_enabled"],
                      provenance=p, status=gate.get("status"), cohort_head_checks=derived["cohort_head_checks"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result["errors"].append(str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path)
    parser.add_argument("--modern", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix.casefold() != ".json":
        parser.error("use a new JSON output path; candidate evidence is never overwritten")
    value = build_gate(args.main, args.modern, protocol_path=args.protocol, data_manifest_path=args.data_manifest, project_root=args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: value[key] for key in ("valid", "status", "family_enabled", "model_enabled")}))


if __name__ == "__main__":
    main()
