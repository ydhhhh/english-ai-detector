"""Independent metadata-only audit of the completed v2 lexical graph.

No body column, text SQLite value, prediction, or model is opened. Binary file
hashing is allowed. Jaccard values are checked for consistency, not recomputed
from bodies: actual lexical verification remains the bound producer's work.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/data_audit_20261017/vendor_fast"))
import numpy as np
import pyarrow.parquet as pq

AUDIT = ROOT / "research/english_v2_data/audit"
ROLE_NAMES = ("train", "development", "calibration", "final")
ROLE_BITS = {role: 1 << i for i, role in enumerate(ROLE_NAMES)}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.z = [1] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.z[a] < self.z[b]:
            a, b = b, a
        self.p[b] = a
        self.z[a] += self.z[b]


def run():
    reads, file_checks, checks = [], {}, []

    def require(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def bound(relative, expected=None):
        path = ROOT / relative
        actual = sha(path)
        if expected is not None:
            require(actual == expected, "file_sha:" + str(relative))
        file_checks[str(relative)] = {"sha256": actual, "bytes": path.stat().st_size}
        return path

    def read(relative, columns):
        require(not set(columns) & {"text", "generation", "prompt", "title", "native_title", "canonical_title"},
                "metadata_projection:" + str(relative))
        path = bound(relative)
        table = pq.read_table(path, columns=columns)
        reads.append({"path": str(relative), "columns": columns, "rows": table.num_rows})
        return table

    completion_path = bound("research/english_v2_data/audit/full_lexical_near_completion.json")
    complete = json.loads(completion_path.read_text())
    require(complete.get("complete") is True and complete["verified_edges"] == 12928,
            "completed_expected_verified_edge_count")
    for spec in complete["inputs"] + complete["outputs"]:
        bound(spec["path"], spec["sha256"])
    cache = json.loads((ROOT / ".cache/english_v2_lexical_v1/complete.json").read_text())
    require(cache.get("cache_complete") is True and cache["signature_count"] == complete["unique_normalized_bodies"],
            "cache_and_graph_identity")
    header = cache["header"]
    require(header == json.loads((ROOT / ".cache/english_v2_lexical_v1/header.json").read_text()), "cache_header_identical")
    require(header["helper_sha256"] == sha(AUDIT / "lexical_signature.py") and
            header["legacy_algorithm_sha256"] == sha(ROOT / "experiments/english_v1/data_grouping.py"),
            "cache_and_verifier_use_same_frozen_lexical_rules")
    for relative, spec in cache["sources"].items():
        bound(relative, spec["sha256"])
    for spec in cache["outputs"]:
        bound(spec["path"], spec["sha256"])

    table = read("research/english_v2_data/audit/full_lexical_signature_index.parquet",
                 ["uid", "normalized_sha256", "signature", "shingle_count"])
    uids = table["uid"].to_numpy()
    norms = table["normalized_sha256"].to_pylist()
    counts = table["shingle_count"].to_numpy()
    signatures = np.frombuffer(b"".join(table["signature"].to_pylist()), dtype=np.uint32).reshape(len(uids), 32)
    require(np.array_equal(uids, np.arange(len(uids))) and len(set(norms)) == len(norms), "unique_contiguous_signature_ids")
    require(len(uids) == 991022 and int((counts >= 20).sum()) == 961983, "signature_population_counts")
    require((counts >= 0).all() and (counts <= 2996).all(), "first3000_unique_word5gram_count_bounds")
    eligible = np.flatnonzero(counts >= 20)
    candidate_keys, oversized = set(), 0
    for band in range(8):
        bands = np.ascontiguousarray(signatures[eligible, band * 4:band * 4 + 4]).view("V16").reshape(-1)
        order = np.argsort(bands, kind="stable")
        sorted_bands = bands[order]
        starts = np.r_[0, np.flatnonzero(sorted_bands[1:] != sorted_bands[:-1]) + 1, len(order)]
        histogram = Counter()
        occurrences = 0
        for start, stop in zip(starts[:-1], starts[1:]):
            size = int(stop - start)
            if size < 2:
                continue
            histogram[size] += 1
            if size >= 100:
                oversized += 1
                continue
            members = sorted(eligible[order[start:stop]].tolist())
            occurrences += size * (size - 1) // 2
            candidate_keys.update((int(a) << 32) | int(b) for a, b in itertools.combinations(members, 2))
        recorded = complete["bands"][band]
        require(histogram == Counter({int(k): v for k, v in recorded["bucket_size_histogram"].items()}) and
                occurrences == recorded["exact_candidate_occurrences"], f"independently_retrieved_band_{band}")
    require(len(candidate_keys) == 222486 == complete["normal_bucket_unique_retrieved_pairs"], "all_normal_bucket_candidates_retrieved")
    possible = {key for key in candidate_keys if min(counts[key >> 32], counts[key & 0xffffffff]) /
                max(counts[key >> 32], counts[key & 0xffffffff]) >= .85}
    saved_pairs = np.load(ROOT / ".cache/english_v2_lexical_v1/exact_candidate_pairs.npy", allow_pickle=False)
    require(possible == set(saved_pairs.tolist()) and len(possible) == 93617 and
            len(candidate_keys) - len(possible) == 128869, "complete_candidate_and_safe_ratio_prune_coverage")
    edges = read("research/english_v2_data/audit/full_lexical_near_edges.parquet",
                 ["left_uid", "right_uid", "jaccard", "kind", "band", "bucket_key"]).to_pylist()
    large = read("research/english_v2_data/audit/full_lexical_oversized_bucket_members.parquet",
                 ["band", "bucket_key", "bucket_size", "uid"])
    require(oversized == large.num_rows == complete["conservative_edges_including_crossband_duplicates"] == 0,
            "zero_actual_oversized_buckets_or_conservative_edges")
    encoded = set()
    for edge in edges:
        a, b, j = edge["left_uid"], edge["right_uid"], edge["jaccard"]
        require(0 <= a < b < len(uids) and np.isfinite(j) and .85 <= j <= 1 and
                edge["kind"] == "verified_jaccard_ge_0.85" and edge["band"] is None and edge["bucket_key"] is None,
                "valid_typed_verified_edge")
        key = (a << 32) | b
        require(key in possible and key not in encoded, "verified_edge_is_unique_retrieved_candidate")
        encoded.add(key)
        intersection = j * (int(counts[a]) + int(counts[b])) / (1 + j)
        require(abs(intersection - round(intersection)) < 1e-6 and
                0 <= round(intersection) <= min(counts[a], counts[b]), "jaccard_matches_integral_set_count_bounds")
    require(len(edges) == 12928, "actual_verified_edge_count")

    graph = read("data/english_calibration_v2/graph_index.parquet",
                 ["id", "component_id", "normalized_sha256", "reserved_role"]).to_pylist()
    require(len(graph) == 645773 and {r["normalized_sha256"] for r in graph} <= set(norms), "all_old_graph_nodes_have_signature_alias_identity")
    old_components = sorted({r["component_id"] for r in graph})
    component_pos = {value: i for i, value in enumerate(old_components)}
    id_comp, norm_comps = {}, defaultdict(set)
    for row in graph:
        cid = component_pos[row["component_id"]]
        require(row["id"] not in id_comp or id_comp[row["id"]] == cid, "old_alias_id_component_consistent")
        id_comp[row["id"]] = cid
        norm_comps[row["normalized_sha256"]].add(cid)
    role_ids = {name: set() for name in ROLE_NAMES}
    primary = read("data/english_v1/full_index.parquet", ["id", "split"]).to_pylist()
    for row in primary:
        split = row["split"]
        if split == "train":
            role_ids["train"].add(row["id"])
        elif split == "calibration":
            role_ids["calibration"].add(row["id"])
        elif split == "validation" or split.startswith("dev_"):
            role_ids["development"].add(row["id"])
        elif split == "test" or split.startswith("challenge_"):
            role_ids["final"].add(row["id"])
    additional = {
        "train": ["data/english_augmentation_v2/train_mixed100k_with_official.parquet",
                  "data/english_adaptation_v3/training_index.parquet", "data/english_v1/pilot/train.parquet"],
        "development": ["data/english_external_v1/dev_backup_allowed.parquet",
                        "data/english_augmentation_v2/historical_dev.parquet", "data/english_v1/pilot/validation.parquet"],
        "calibration": ["data/english_v1/pilot/calibration.parquet"], "final": []}
    # Historical_regression was explicitly retired and divided into official
    # training and observed development before v1 finalization; it is not final.
    plan_path = bound("research/english_v1_finalization_plan.json")
    plan = json.loads(plan_path.read_text())
    for spec in plan["final_datasets"]:
        bound(spec["input"], spec["input_sha256"])
        additional["final"].append(spec["input"])
    for role, paths in additional.items():
        for relative in paths:
            role_ids[role].update(read(relative, ["id"])["id"].to_pylist())
    role_masks = [0] * len(old_components)
    for role, ids in role_ids.items():
        require(ids <= id_comp.keys(), "all_old_role_seed_ids_present:" + role)
        for identifier in ids:
            role_masks[id_comp[identifier]] |= ROLE_BITS[role]
    for row in graph:
        if row["reserved_role"] == "wp_calibration":
            role_masks[component_pos[row["component_id"]]] |= ROLE_BITS["calibration"]
        elif row["reserved_role"] == "test_creative_fresh":
            role_masks[component_pos[row["component_id"]]] |= ROLE_BITS["final"]
    pairs = list(itertools.combinations(ROLE_NAMES, 2))
    direct_counts, direct_unique = Counter(), defaultdict(set)
    node_for_uid = {}
    uf = UF(len(old_components) + 2 * len(edges))
    next_node = len(old_components)
    def node(uid):
        nonlocal next_node
        if uid in node_for_uid:
            return node_for_uid[uid]
        comps = sorted(norm_comps.get(norms[uid], ()))
        if comps:
            value = comps[0]
            for other in comps[1:]:
                uf.union(value, other)
        else:
            value = next_node
            next_node += 1
        node_for_uid[uid] = value
        return value
    def closure_summary():
        grouped = defaultdict(lambda: defaultdict(set))
        for cid, mask in enumerate(role_masks):
            if not mask:
                continue
            for role in ROLE_NAMES:
                if mask & ROLE_BITS[role]:
                    grouped[uf.find(cid)][role].add(cid)
        return {a + "__" + b: {
            "connected_groups": sum(bool(rr.get(a) and rr.get(b) and len(rr[a] | rr[b]) > 1) for rr in grouped.values()),
            "old_components_in_these_groups": sum(len(rr[a] | rr[b]) for rr in grouped.values()
                if rr.get(a) and rr.get(b) and len(rr[a] | rr[b]) > 1)} for a, b in pairs}
    baseline = closure_summary()
    for edge in edges:
        left, right = edge["left_uid"], edge["right_uid"]
        for a in norm_comps.get(norms[left], ()):
            for b in norm_comps.get(norms[right], ()):
                if a == b:
                    continue
                for first, second in pairs:
                    x, y = ROLE_BITS[first], ROLE_BITS[second]
                    if (role_masks[a] & x and role_masks[b] & y) or (role_masks[a] & y and role_masks[b] & x):
                        key = first + "__" + second
                        direct_counts[key] += 1
                        direct_unique[key].add(tuple(sorted((a, b))))
        uf.union(node(left), node(right))
    after = closure_summary()
    require(all(sha(ROOT / relative) == spec["sha256"] for relative, spec in file_checks.items()), "all_audited_files_still_unchanged")
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True, "checks": len(checks), "full_near_completion_sha256": sha(completion_path),
        "observed": {"unique_normalized_bodies": len(uids), "all_old_graph_rows": len(graph),
            "retrieved_unique_candidates": len(candidate_keys), "count_ratio_pruned": len(candidate_keys) - len(possible),
            "actual_candidate_verifications": len(possible), "verified_edges": len(edges),
            "conservative_edges": 0, "oversized_buckets": 0},
        "old_role_definition": {"retired_historical_regression": "actual official_train / observed historical_dev roles, never counted as independent final",
            "all_ten_final_input_rows_protected_including_ineligible": True,
            "assigned_primary_challenges_and_whole_reserved05_components_protected": True},
        "old_role_seed_counts": {role: {"unique_ids": len(ids), "original_components": len({id_comp[x] for x in ids})}
                                 for role, ids in role_ids.items()},
        "new_direct_verified_edges_between_different_old_components": {
            a + "__" + b: {"edge_occurrences": direct_counts[a + "__" + b],
                           "distinct_old_component_pairs": len(direct_unique[a + "__" + b])} for a, b in pairs},
        "old_component_role_connections_before_lexical_edges": baseline,
        "old_component_role_connections_after_verified_edges": after,
        "preexisting_old_components_with_multiple_roles": sum(mask.bit_count() > 1 for mask in role_masks),
        "metadata_only": True, "body_columns_or_sqlite_texts_read": False,
        "models_predictions_or_scores_read": False, "roles_assigned_here": False,
        "read_projections": reads, "bound_files": file_checks,
        "limitations": ["Recorded Jaccard values were checked for arithmetic consistency; this independent pass did not reread bodies to recompute them.",
            "Closure here uses saved old components and the new lexical edges only. The later joint graph adds all new native-parent/title/hash bridges and must be separately audited.",
            "Approximate first3000-token lexical LSH does not prove exhaustive semantic, hidden-parent or pretraining independence.",
            "New old-role connections are evidence limitations of the archived v1 split, not a reason to alter its frozen predictions, metrics or thresholds."]}


if __name__ == "__main__":
    output = AUDIT / "independent_full_near_qa.json"
    result = run()
    result["audit_code_sha256"] = sha(__file__)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "checks", "observed",
        "new_direct_verified_edges_between_different_old_components", "old_component_role_connections_after_verified_edges")}))
