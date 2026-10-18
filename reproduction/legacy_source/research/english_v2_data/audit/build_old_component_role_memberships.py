"""Additional immutable old-component role ledger; metadata only, no new roles.

This does not modify historical indexes. Complete experimental training-input
supersets are conservative ancestry seeds; all old pilot roles are included.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/data_audit_20261017/vendor_fast"))
import pyarrow as pa
import pyarrow.parquet as pq

OUT = ROOT / "research/english_v2_data/audit"
ROLE_NAMES = ("train_fit", "development", "calibration", "old_final")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    target = OUT / "old_component_role_memberships.parquet"
    receipt_path = target.with_suffix(".receipt.json")
    assert not target.exists() and not receipt_path.exists(), "Do not replace an existing role ledger"
    projections = []
    def read(relative, columns):
        assert not set(columns) & {"text", "prompt", "generation", "title"}
        path = ROOT / relative
        before = sha(path)
        table = pq.read_table(path, columns=columns)
        assert sha(path) == before
        projections.append({"path": relative, "columns": columns, "rows": table.num_rows,
                            "bytes": path.stat().st_size, "sha256": before})
        return table.to_pylist()
    graph = read("data/english_calibration_v2/graph_index.parquet", ["id", "component_id", "reserved_role"])
    id_comp = {}
    for row in graph:
        assert row["id"] not in id_comp or id_comp[row["id"]] == row["component_id"]
        id_comp[row["id"]] = row["component_id"]
    components = sorted({r["component_id"] for r in graph})
    seeds = {role: defaultdict(set) for role in ROLE_NAMES}
    path_counts = defaultdict(Counter)
    def add(role, identifier, source):
        assert identifier in id_comp, "Old role seed missing from the full saved graph"
        component = id_comp[identifier]
        seeds[role][component].add(identifier)
        path_counts[component][role + "|" + source] += 1
    for r in read("data/english_v1/full_index.parquet", ["id", "split"]):
        s = r["split"]
        role = ("train_fit" if s == "train" else "calibration" if s == "calibration" else
                "development" if s == "validation" or s.startswith("dev_") else
                "old_final" if s == "test" or s.startswith("challenge_") else None)
        if role:
            add(role, r["id"], "primary_assignment/" + s)
    files = {
        "train_fit": ["data/english_augmentation_v2/train_mixed100k_with_official.parquet",
                      "data/english_adaptation_v3/training_index.parquet", "data/english_v1/pilot/train.parquet"],
        "development": ["data/english_external_v1/dev_backup_allowed.parquet",
                        "data/english_augmentation_v2/historical_dev.parquet", "data/english_v1/pilot/validation.parquet"],
        "calibration": ["data/english_v1/pilot/calibration.parquet"], "old_final": []}
    plan_path = ROOT / "research/english_v1_finalization_plan.json"
    for spec in json.loads(plan_path.read_text())["final_datasets"]:
        assert sha(ROOT / spec["input"]) == spec["input_sha256"]
        files["old_final"].append(spec["input"])
    for role, paths in files.items():
        for relative in paths:
            for row in read(relative, ["id"]):
                add(role, row["id"], relative)
    reservation_seen = set()
    for row in graph:
        role = {"wp_calibration": "calibration", "test_creative_fresh": "old_final"}.get(row["reserved_role"])
        key = (row["id"], row["reserved_role"])
        if role and key not in reservation_seen:
            add(role, row["id"], "reserved05/" + row["reserved_role"] + "_whole_component")
            reservation_seen.add(key)
    previous = read("research/english_v2_data/audit/existing_component_roles.parquet",
                    ["component_id", "observed_manual_review", "forbidden_wikihow", "reserved05"])
    previous = {r["component_id"]: r for r in previous}
    assert set(previous) == set(components)
    rows = []
    for component in components:
        roles = [role for role in ROLE_NAMES if component in seeds[role]]
        rows.append({"component_id": component, **{role: role in roles for role in ROLE_NAMES},
            "roles": roles, "role_count": len(roles), "single_role": roles[0] if len(roles) == 1 else None,
            **{role + "_seed_unique_ID_count": len(seeds[role].get(component, ())) for role in ROLE_NAMES},
            "observed_manual_review": previous[component]["observed_manual_review"],
            "forbidden_wikihow": previous[component]["forbidden_wikihow"],
            "reserved05": previous[component]["reserved05"],
            "seed_path_counts_json": json.dumps(dict(sorted(path_counts[component].items())), sort_keys=True)})
    schema = pa.schema([("component_id", pa.string()), *[(r, pa.bool_()) for r in ROLE_NAMES],
        ("roles", pa.list_(pa.string())), ("role_count", pa.int32()), ("single_role", pa.string()),
        *[(r + "_seed_unique_ID_count", pa.int32()) for r in ROLE_NAMES],
        ("observed_manual_review", pa.bool_()), ("forbidden_wikihow", pa.bool_()), ("reserved05", pa.bool_()),
        ("seed_path_counts_json", pa.string())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), target, compression="zstd")
    assert all(sha(ROOT / r["path"]) == r["sha256"] for r in projections)
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "metadata_only": True, "text_predictions_models_read": False,
        "new_roles_assigned": False, "all_original_old_components_retained": len(rows),
        "role_counts": {role: sum(row[role] for row in rows) for role in ROLE_NAMES},
        "role_combination_counts": dict(Counter("+".join(row["roles"]) or "none" for row in rows)),
        "role_seed_ID_union_counts": {role: len(set().union(*seeds[role].values())) for role in ROLE_NAMES},
        "no_role_ID_overlap": all(not set().union(*seeds[a].values()) & set().union(*seeds[b].values())
            for i, a in enumerate(ROLE_NAMES) for b in ROLE_NAMES[i + 1:]),
        "role_history_notes": ["All actual training input supersets including pilot vocabulary/fit are conservative ancestry protection seeds.",
            "Retired historical_regression is represented by actual official training and historical_dev roles, not as an independent final.",
            "All original final input IDs, including ineligible rows, and full05 calibration/final reservation siblings retain protection.",
            "Old component memberships must be OR-propagated through the complete new joint graph before choosing legacy development/calibration retention.",
            "Legacy development retention requires development only; historical calibration replay requires calibration only; all actual-fit kin and old-final/multi-role kin are ineligible."],
        "input_metadata_projections": projections, "finalization_plan": {"path": str(plan_path.relative_to(ROOT)), "sha256": sha(plan_path)},
        "output": {"path": str(target.relative_to(ROOT)), "sha256": sha(target), "bytes": target.stat().st_size,
                   "columns": schema.names}, "audit_code_sha256": sha(__file__)}
    with receipt_path.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "components": len(rows), "role_counts": result["role_counts"],
                      "multi_role_components": sum(r["role_count"] > 1 for r in rows), "output": result["output"],
                      "receipt_sha256": sha(receipt_path)}))


if __name__ == "__main__":
    main()
