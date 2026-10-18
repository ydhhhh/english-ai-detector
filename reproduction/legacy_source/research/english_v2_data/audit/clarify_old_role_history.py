"""Explain old-role metadata overlaps without text, predictions or new roles."""
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/data_audit_20261017/vendor_fast"))
import pyarrow.parquet as pq


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    projections = []
    def rows(path, columns):
        assert not set(columns) & {"text", "prompt", "generation"}
        p = ROOT / path
        projections.append({"path": path, "columns": columns, "sha256": sha(p)})
        return pq.read_table(p, columns=columns).to_pylist()
    graph = rows("data/english_calibration_v2/graph_index.parquet", ["id", "component_id", "reserved_role"])
    comp = {r["id"]: r["component_id"] for r in graph}
    roles = {k: set() for k in ("train", "development", "calibration", "final")}
    locations = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    descriptors = {}
    def add(role, identifier, origin):
        assert identifier in comp
        roles[role].add(identifier)
        locations[comp[identifier]][role][origin].add(identifier)
    for r in rows("data/english_v1/full_index.parquet", ["id", "split", "source", "domain", "eligible"]):
        descriptors[r["id"]] = {k: r[k] for k in ("source", "domain", "eligible")}
        s = r["split"]
        role = ("train" if s == "train" else "development" if s == "validation" or s.startswith("dev_")
                else "calibration" if s == "calibration" else "final" if s == "test" or s.startswith("challenge_") else None)
        if role:
            add(role, r["id"], "primary_assignment/" + s)
    files = {
        "train": ["data/english_augmentation_v2/train_mixed100k_with_official.parquet", "data/english_adaptation_v3/training_index.parquet"],
        "development": ["data/english_external_v1/dev_backup_allowed.parquet", "data/english_augmentation_v2/historical_dev.parquet"],
        "calibration": [], "final": []}
    plan = ROOT / "research/english_v1_finalization_plan.json"
    for spec in json.loads(plan.read_text())["final_datasets"]:
        assert sha(ROOT / spec["input"]) == spec["input_sha256"]
        files["final"].append(spec["input"])
    for role, paths in files.items():
        for path in paths:
            for r in rows(path, ["id"]):
                add(role, r["id"], path)
    for r in graph:
        if r["reserved_role"] == "wp_calibration":
            add("calibration", r["id"], "reserved05/wp_calibration_whole_component")
        elif r["reserved_role"] == "test_creative_fresh":
            add("final", r["id"], "reserved05/test_creative_fresh_whole_component")
    def summary():
        masks = defaultdict(set)
        for role, ids in roles.items():
            for identifier in ids:
                masks[comp[identifier]].add(role)
        return masks, dict(Counter("+".join(sorted(v)) for v in masks.values() if len(v) > 1))
    masks, without_pilot = summary()
    details = []
    for component, role_set in masks.items():
        if len(role_set) < 2:
            continue
        entry = {"old_component_id": component, "roles": sorted(role_set), "by_role": {}}
        for role in sorted(role_set):
            origins = locations[component][role]
            identifiers = set().union(*origins.values())
            descriptor_counts = Counter((descriptors.get(i, {}).get("source"), descriptors.get(i, {}).get("domain"),
                                         descriptors.get(i, {}).get("eligible")) for i in identifiers)
            entry["by_role"][role] = {"unique_ID_count": len(identifiers),
                "metadata_locations": {path: len(ids) for path, ids in origins.items()},
                "primary_descriptors": [{"source": k[0], "domain": k[1], "eligible": k[2], "count": n}
                                        for k, n in sorted(descriptor_counts.items(), key=lambda v: str(v[0]))]}
        details.append(entry)
    nonpilot_ids = {role: set(ids) for role, ids in roles.items()}
    for role, split in (("train", "train"), ("development", "validation"), ("calibration", "calibration")):
        path = f"data/english_v1/pilot/{split}.parquet"
        for r in rows(path, ["id"]):
            add(role, r["id"], path)
    _, with_pilot = summary()
    output = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_only": True, "models_predictions_or_bodies_read": False,
        "full_near_qa_sha256": sha(ROOT / "research/english_v2_data/audit/independent_full_near_qa.json"),
        "old_original_graph_sha256": sha(ROOT / "data/english_calibration_v2/graph_index.parquet"),
        "preexisting_multirole_components_excluding_pilot": without_pilot,
        "preexisting_multirole_components_including_pilot": with_pilot,
        "direct_ID_overlap_excluding_pilot": {a + "__" + b: len(nonpilot_ids[a] & nonpilot_ids[b])
                                             for a, b in itertools.combinations(roles, 2)},
        "nonpilot_component_metadata_details": sorted(details, key=lambda x: x["old_component_id"]),
        "interpretation": ["These are preexisting saved-graph component relationships, not newly discovered by the v2 lexical edges.",
            "All experimental training inputs are conservative ancestry supersets; component membership does not by itself identify exact duplicate rows or prove selected model contamination.",
            "Pilot roles are early throughput/interface experiments, separately disclosed rather than silently equated with the final frozen v1 roles.",
            "Ineligible original diagnostic input rows and unselected reserved-component siblings remain protected metadata; their graph presence need not mean they were evaluated.",
            "The old archived predictions and scores are untouched. Every such component remains protected against new fresh v2 roles."],
        "projections": projections, "audit_code_sha256": sha(__file__)}
    p = ROOT / "research/english_v2_data/audit/old_role_history_clarification.json"
    with p.open("x") as f:
        f.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"without_pilot": without_pilot, "with_pilot": with_pilot,
                      "details": details, "output_sha256": sha(p)}))


if __name__ == "__main__":
    main()
