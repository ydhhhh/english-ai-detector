"""Bind independently audited prepared roles to the pre-score v2 protocol.

Only prepared metadata and immutable source/receipt bytes are read. This does
not initialize a model, execute a forward pass, fit a parameter or start a job.
An explicit key map avoids guessing dataset names from directory contents.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.english_v1 import runtime  # noqa: E402,F401
import pyarrow as pa
import pyarrow.parquet as pq

SKELETON = ROOT / "research/english_v2_protocol_skeleton.json"
POLICY = ROOT / "research/english_v2_data/acceptance_targets_v1.json"
POLICY_SHA = "ebb755a65df48344ba9d26a7855e8fdf45ec2df841ad46e6bc7205e81e53b4bd"
COHORTS = ("general", "source_domain", "modern", "paraphrase", "legacy_main", "legacy_creative")
FINALS = ("fresh_main", "fresh_source_domain", "fresh_modern_with_human_controls", "fresh_paraphrase_with_human_controls")
BUCKETS = ("general", "source_domain", "modern", "paraphrase", "historical_WP_CAL_replay")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def sha(value):
    with path(value).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(value):
    return json.loads(path(value).read_text())


def relative(value):
    return str(path(value).resolve().relative_to(ROOT))


def bound(value, *, rows=None):
    p = path(value)
    result = {"path": relative(p), "bytes": p.stat().st_size, "sha256": sha(p)}
    if rows is not None:
        result["rows"] = rows
    return result


def write_new(value, obj):
    p = path(value)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as stream:
        stream.write(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def metadata(item, *, view=None, roles=None):
    p = path(item["path"])
    pf = pq.ParquetFile(p)
    columns = ["id", "group_id", "y_ai", "wordlike_tokens", "split", "eligible"]
    if view:
        field = pf.schema_arrow.field(view)
        need(pa.types.is_boolean(field.type), "registered membership must have Arrow bool type")
        columns.append(view)
    table = pf.read(columns=columns)
    need(table.num_rows == item["rows"], "declared dataset row count differs")
    if view:
        need(table[view].null_count == 0, "registered membership has nulls")
        table = table.filter(table[view])
    records = table.to_pylist()
    if roles:
        need(all(r["split"] in roles for r in records), "prepared rows have the wrong data role")
    need(records and all(isinstance(r["id"], str) and r["id"].strip() and
        isinstance(r["group_id"], str) and r["group_id"].strip() and
        r["y_ai"] in (0, 1) and r["eligible"] is True and
        isinstance(r["wordlike_tokens"], int) and not isinstance(r["wordlike_tokens"], bool) and
        r["wordlike_tokens"] >= 0 for r in records), "prepared metadata schema/eligibility invalid")
    need(len({r["id"] for r in records}) == len(records), "duplicate dataset IDs")
    core = [r for r in records if r["wordlike_tokens"] >= 128]
    def counts(rows):
        return {"n": len(rows), "human_n": sum(r["y_ai"] == 0 for r in rows),
            "AI_n": sum(r["y_ai"] == 1 for r in rows), "groups": len({r["group_id"] for r in rows})}
    return {"all": counts(records), "core": counts(core)}, records


def freeze(args):
    need(sha(POLICY) == POLICY_SHA, "the pre-score numeric policy changed")
    policy, protocol, manifest, mapping, qa, allocation = (read(p) for p in
        (POLICY, SKELETON, args.manifest, args.bindings, args.qa, args.allocation))
    need(protocol["status"] == "DRAFT_NOT_FROZEN_NOT_EXECUTABLE", "wrong skeleton")
    need(qa.get("passed") is True, "actual independent metadata QA must pass")
    need(qa.get("prepared_manifest_sha256") == sha(args.manifest) and
        qa.get("allocation_protocol_sha256") == sha(args.allocation), "QA binds another manifest/allocation")
    need(allocation.get("status", "").startswith("FROZEN"), "allocation must have been frozen before role construction")
    need(set(mapping["development"]) == set(COHORTS) and set(mapping["final"]) == set(FINALS) and
        set(mapping["calibration_buckets"]) == set(BUCKETS), "explicit complete dataset mapping required")
    entries, stats = {}, {}
    roles_by_key = {}
    def register(key, roles):
        if key in roles_by_key:
            need(roles_by_key[key] == roles, "one mapped file has contradictory roles")
        roles_by_key[key] = roles
    for name, key in mapping["training"].items():
        need(name in {"train", "validation", "dev_challenge"}, "unrecognized training role")
        register(key, {"train"} if name == "train" else {"validation", "dev"} if name == "validation" else {"dev_challenge", "dev_challenge_source_domain"})
    for name, key in mapping["development"].items():
        register(key, {"validation", "dev"} if name == "general" else {"dev", "dev_challenge", "dev_challenge_source_domain"})
    for key in [mapping["calibration"], *mapping["calibration_buckets"].values()]:
        register(key, {"calibration"})
    for key in mapping["final"].values():
        register(key, {"final_sealed"})
    def get(key):
        if key not in entries:
            declared = manifest["output_files"][key]
            need(declared["role"] in roles_by_key[key], "prepared manifest declares wrong role")
            item = bound(declared["path"], rows=declared["rows"])
            need(item["sha256"] == declared["sha256"] and
                ("bytes" not in declared or item["bytes"] == declared["bytes"]), "prepared manifest file identity mismatch")
            need(qa.get("audited_output_sha256", {}).get(key) == item["sha256"], "mapped output was not covered by independent QA")
            entries[key] = item
            stats[key] = metadata(item, roles=roles_by_key[key])[0]
        return entries[key]
    for key in [*mapping["training"].values(), *mapping["development"].values(), mapping["calibration"],
        *mapping["calibration_buckets"].values(), *mapping["final"].values()]:
        get(key)
    protocol = {k: v for k, v in protocol.items() if not k.startswith("_")}
    protocol["status"] = "FROZEN_BEFORE_V2_TRAINING_AND_SCORES"
    protocol["binary_calibration"] = copy.deepcopy(policy["threshold_policy"])
    protocol["acceptance"] = {"criterion": policy["criterion"], "bootstrap": copy.deepcopy(policy["bootstrap"])}
    protocol["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["data_allocation_protocol"] = bound(args.allocation)
    protocol["prepared_manifest"] = bound(args.manifest)
    protocol["dataset_key_mapping"] = bound(args.bindings)
    protocol["independent_data_qa"] = bound(args.qa)
    protocol["protocol_constructor"] = bound(__file__)
    protocol["training"]["data"] = {k: get(v) for k, v in mapping["training"].items()}
    need(set(protocol["training"]["data"]) == {"train", "validation", "dev_challenge"}, "three trainer inputs required")
    expected = {k: v["sha256"] for k, v in protocol["training"]["data"].items()}
    need(qa.get("training_inputs_sha256") == expected, "independent seal binds another training input set")
    protocol["training"]["data_qa"] = bound(args.qa)
    protocol["development"]["cohorts"] = {k: get(v) for k, v in mapping["development"].items()}
    need(protocol["training"]["data"]["validation"] == protocol["development"]["cohorts"]["general"], "general monitor must be the exact selector general view")
    cal = get(mapping["calibration"])
    protocol["calibration_input"], protocol["calibration_input_sha256"] = cal, cal["sha256"]
    totals = stats[mapping["calibration"]]["all"]
    mixture = protocol["calibration_mixture"]
    mixture.update(input_rows=totals["n"], human_rows=totals["human_n"], AI_rows=totals["AI_n"])
    for bucket in mixture["buckets"]:
        key = mapping["calibration_buckets"][bucket["name"]]
        n = stats[key]["all"]
        bucket.update(input=get(key), selected_rows=n["n"], selected_human_rows=n["human_n"], selected_AI_rows=n["AI_n"])
    need(sum(b["selected_rows"] for b in mixture["buckets"]) == totals["n"], "calibration bucket counts differ")
    history = protocol["calibration_role_history"]
    new_buckets = [b for b in mixture["buckets"] if b["name"] != "historical_WP_CAL_replay"]
    history["fresh_source_CAL"].update(inputs=[b["input"] for b in new_buckets], selection_manifest_sha256=sha(args.manifest),
        selected_rows=sum(b["selected_rows"] for b in new_buckets))
    old = history["historical_WP_CAL_replay"]
    selected = next(b for b in mixture["buckets"] if b["name"] == "historical_WP_CAL_replay")
    old.update(source_sha256=sha(old["source_path"]), selected_view=selected["input"], selected_rows=selected["selected_rows"],
        selected_human_rows=selected["selected_human_rows"], selected_AI_rows=selected["selected_AI_rows"], selection_manifest_sha256=sha(args.manifest))
    protocol["final_input_sha256s"] = []
    for name, key in mapping["final"].items():
        item, n = get(key), stats[key]
        spec = protocol["necessary_final_suites"][name]
        expected_views = set(policy["required_rewrite_views"]) if name == "fresh_paraphrase_with_human_controls" else set()
        need(set(spec.get("required_views", {})) == expected_views, "required rewrite views differ from frozen policy")
        spec.update(input_sha256=item["sha256"], rows=n["all"]["n"], groups=n["all"]["groups"], core_counts=n["core"])
        need(spec["targets"] == policy["targets"][name], "numeric suite targets differ from frozen policy")
        for view, view_spec in spec.get("required_views", {}).items():
            need(view_spec["column"] == view, "required rewrite membership column changed")
            vs, _ = metadata(item, view=view_spec["column"], roles={"final_sealed"})
            view_spec.update(rows=vs["all"]["n"], groups=vs["all"]["groups"], core_counts=vs["core"])
            need(view_spec["targets"] == policy["targets"][name], "rewrite targets differ from frozen policy")
        protocol["final_input_sha256s"].append(item["sha256"])
    plan = {"schema_version": 1, "status": "FROZEN_BEFORE_V2_TRAINING_AND_SCORES", "calibration": {}, "final_datasets": []}
    def plan_item(item):
        return {"input": item["path"], "input_sha256": item["sha256"], "manifest": relative(args.manifest),
            "manifest_sha256": sha(args.manifest), "rows": item["rows"], "expected_export_rows": item["rows"]}
    plan["calibration"] = plan_item(cal)
    plan["final_datasets"] = [{"name": name, **plan_item(get(key))} for name, key in mapping["final"].items()]
    need(not path(args.data_plan).exists() and not path(args.output).exists(), "immutable output already exists")
    # Verify frozen execution files and inherited parent weights before emitting either file.
    for registry in (protocol["training"]["code_sha256"], protocol["development"]["code_sha256"], protocol["code_registry"]):
        for filename, expected_sha in registry.items():
            need(sha(filename) == expected_sha, "registered code changed: " + filename)
    for job in protocol["training"]["jobs"]:
        parent = job["parent"]
        need(sha(path(parent["path"])/"detector.safetensors") == parent["weights_sha256"] and
            sha(path(parent["path"])/"detector.json") == parent["detector_metadata_sha256"], "warm-start parent changed")
    plan_bytes = (json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    protocol["release_data_plan"] = {"path": relative(args.data_plan), "bytes": len(plan_bytes),
        "sha256": hashlib.sha256(plan_bytes).hexdigest()}
    def no_null(value):
        if isinstance(value, dict):
            return all(no_null(v) for v in value.values())
        if isinstance(value, list):
            return all(no_null(v) for v in value)
        return value is not None
    need(no_null(protocol), "unfilled draft field remains; outputs must not launch training")
    write_new(args.data_plan, plan)
    need(bound(args.data_plan) == protocol["release_data_plan"], "serialized data-plan differs from its binding")
    write_new(args.output, protocol)
    result = {"created_at_utc": protocol["created_at_utc"], "protocol": bound(args.output),
        "data_plan": bound(args.data_plan), "prepared_inputs": entries, "metadata_counts": stats,
        "real_training_started": False, "development_scores_opened": False, "calibration_fitted": False, "final_predictions_opened": False}
    write_new(str(args.output) + ".receipt.json", result)
    return {"protocol": result["protocol"], "data_plan": result["data_plan"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "bindings", "qa", "allocation", "output", "data-plan"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(freeze(parser.parse_args()), indent=2))
