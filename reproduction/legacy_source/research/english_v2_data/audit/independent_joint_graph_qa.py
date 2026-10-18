"""Reconstruct the joint graph from metadata, without text or predictions."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/data_audit_20261017/vendor_fast"))
import pyarrow.parquet as pq

AUDIT = ROOT / "research/english_v2_data/audit"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sh(value):
    return hashlib.sha256(value.encode()).hexdigest()


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


def audit(expected_sha):
    checks, inputs, projections = 0, {}, []
    def need(condition, message):
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1
    def bind(relative, expected=None):
        path = ROOT / relative
        value = sha(path)
        if expected:
            need(value == expected, "file SHA mismatch: " + str(relative))
        inputs[str(relative)] = value
        return path
    def read(relative, columns):
        need(not set(columns) & {"text", "generation", "prompt", "title", "native_title", "canonical_title",
             "human_output", "model_output", "human_edits", "gpt-4o_edits", "llama-3.1-70b_edits"}, "body projection refused")
        p = bind(relative)
        table = pq.read_table(p, columns=columns)
        projections.append({"path": str(relative), "columns": columns, "rows": table.num_rows})
        return table
    manifest_path = bind("research/english_v2_data/audit/joint_full_near_capacity.json", expected_sha)
    manifest = json.loads(manifest_path.read_text())
    for spec in manifest["inputs"] + manifest["outputs"]:
        bind(spec["path"], spec["sha256"])
    base_qa = json.loads(bind("research/english_v2_data/audit/independent_full_near_qa.json").read_text())
    need(base_qa["passed"] is True and base_qa["full_near_completion_sha256"] == sha(AUDIT / "full_lexical_near_completion.json"), "base lexical QA not applicable")
    observed_guard = json.loads(bind("research/english_v2_data/audit/raid_observed_reference_guard.json").read_text())
    for spec in observed_guard["inputs"] + observed_guard["outputs"]:
        bind(spec["path"], spec["sha256"])
    need(observed_guard["first_native_rows"] == observed_guard["native_parent_components"] == 100 and
         observed_guard["verified_edges"] == 4 and observed_guard["conservative_edges"] == 0, "observed-reference increment contract changed")
    nodes = read("research/english_v2_data/audit/joint_component_node_index.parquet",
                 ["node_index", "kind", "identity", "norm_uid", "pre_near_component_id", "component_id",
                  "seed_barrier_flags", "pre_near_barrier_flags", "barrier_flags"])
    index = nodes["node_index"].to_pylist()
    kinds, identities = nodes["kind"].to_pylist(), nodes["identity"].to_pylist()
    pre, final = nodes["pre_near_component_id"].to_pylist(), nodes["component_id"].to_pylist()
    seeds = nodes["seed_barrier_flags"].to_pylist()
    preflags, finalflags = nodes["pre_near_barrier_flags"].to_pylist(), nodes["barrier_flags"].to_pylist()
    need(index == list(range(len(index))) and len(index) == manifest["joint_graph_nodes"], "complete contiguous graph node registry")
    lookup = {kind: {} for kind in ("old_component", "native_parent", "normalized_body")}
    for i, (kind, identity) in enumerate(zip(kinds, identities)):
        need(identity not in lookup[kind], "duplicate graph node identity")
        lookup[kind][identity] = i
    need(len(lookup["old_component"]) == 399660 and len(lookup["normalized_body"]) == 991122 and
         len(lookup["native_parent"]) == manifest["new_native_parent_nodes"], "old/native/normal registry coverage")
    uid_nodes = {int(uid): i for i, uid in enumerate(nodes["norm_uid"].to_pylist()) if uid is not None}
    need(set(uid_nodes) == set(range(991122)), "all base and incremental normalized UIDs represented")
    old_flags = read("research/english_v2_data/audit/existing_component_roles.parquet",
        ["component_id", "forbid_fresh", "touches_nontraining_assigned", "reserved05", "touches_actual_fit",
         "forbidden_wikihow", "observed_manual_review"]).to_pylist()
    for row in old_flags:
        expected = set()
        if row["forbid_fresh"]:
            expected.add("old_used_or_reserved_forbid_fresh")
        if row["touches_nontraining_assigned"] or row["reserved05"]:
            expected.add("old_nontrain_or_reserved_forbid_new_training")
        if row["touches_actual_fit"]:
            expected.add("old_actual_fit_kin")
        if row["forbidden_wikihow"]:
            expected.add("wikiHow_forbidden")
        if row["observed_manual_review"]:
            expected.add("prior_manual_review_observed")
        need(set(seeds[lookup["old_component"][row["component_id"]]]) == expected, "all old protection flags copied exactly")
    uf = UF(len(index))
    owners, parent_keys = {}, {}
    parents = read("research/english_v2_data/audit/joint_native_parent_keys.parquet",
                   ["parent", "node_index", "observable_key_hashes", "seed_barrier_flags", "pre_near_component_id", "component_id"]).to_pylist()
    for row in parents:
        i = lookup["native_parent"][row["parent"]]
        need(i == row["node_index"] and pre[i] == row["pre_near_component_id"] and final[i] == row["component_id"], "parent registry node alignment")
        keys = set(row["observable_key_hashes"])
        parent_keys[row["parent"]] = keys
        for key in keys:
            if key in owners:
                uf.union(i, owners[key])
            else:
                owners[key] = i
    newcols = ["row_record_id", "id", "parent", "collector", "domain", "canonical_domain", "y_ai", "normalized_sha256",
        "wordlike_tokens", "structural_core", "structural_quality_reason", "publisher_English", "source_rights_allowed",
        "nominal_generator", "variant", "norm_uid", "pre_near_component_id", "component_id", "barrier_flags",
        "binary_conflicting_normalized_text", "capacity_qualified", "candidate_fresh", "candidate_train",
        "candidate_train_under_draft_holdouts", "runtime_language_supported_diagnostic"]
    new = read("research/english_v2_data/audit/joint_new_candidate_index.parquet", newcols).to_pylist()
    need(len({r["row_record_id"] for r in new}) == len(new) == 435879, "all new variants including rejected rows retained")
    for row in new:
        i, j = lookup["native_parent"][row["parent"]], lookup["normalized_body"][row["normalized_sha256"]]
        uf.union(i, j)
        need(j == uid_nodes[row["norm_uid"]], "candidate hash to UID alignment")
        need(row["structural_core"] == (row["structural_quality_reason"] is None and row["y_ai"] in (0, 1)), "structural eligibility independent of runtime language flag")
        if row["collector"] == "MAGA":
            if row["canonical_domain"] == "naturalquestions":
                need("draft_NQ_source_holdout" in seeds[i], "all NQ native parents marked holdout")
            if row["nominal_generator"] == "Gemini-2.0-flash":
                need("draft_Gemini_nominal_holdout" in seeds[i], "all Gemini native parents marked holdout")
            if row["canonical_domain"] == "wikihow":
                need("wikiHow_forbidden" in seeds[i], "all WikiHow native parents forbidden")
    observed = read("research/english_v2_data/audit/raid_observed_reference_metadata.parquet",
                    ["id", "source_id", "normalized_sha256", "norm_uid", "observable_key_hashes"]).to_pylist()
    for row in observed:
        parent, norm = lookup["native_parent"]["RAID:" + row["source_id"]], lookup["normalized_body"][row["normalized_sha256"]]
        need(pre[parent] == pre[norm], "observed reference parent and body MUST already be connected before near search")
        need("upstream_documentation_example_observed" in seeds[parent] and "upstream_documentation_example_observed" in seeds[norm], "both observed reference identities protected")
        uf.union(parent, norm)
    old_graph = read("data/english_calibration_v2/graph_index.parquet",
                     ["id", "component_id", "normalized_sha256", "observable_key_hashes"])
    old_id_norm, hits = {}, 0
    for row in old_graph.to_pylist():
        i = lookup["old_component"][row["component_id"]]
        j = lookup["normalized_body"][row["normalized_sha256"]]
        old_id_norm[row["id"]] = row["normalized_sha256"]
        uf.union(i, j)
        for key in row["observable_key_hashes"]:
            if key in owners:
                uf.union(i, owners[key])
                hits += 1
    need(old_graph.num_rows == 645773 and hits == manifest["old_new_observable_key_matches"], "all old nodes and observable key bridges preserved")
    completion = json.loads(bind("research/english_v2_data/raw/raid_paired_range_v1/COMPLETED.json").read_text())
    screened_ids = set()
    for file in completion["files"]:
        relative = str(Path(file["payload"]["path"]).parent / "screened_lineage_metadata.parquet")
        for row in read(relative, ["id", "source_id", "adv_source_id"]).to_pylist():
            parent = "RAID:" + row["source_id"]
            need(parent in parent_keys, "every screened RAID source retained even without eligible body")
            for key in ("id", "source_id", "adv_source_id"):
                value = row[key]
                if isinstance(value, str) and value.strip() and value.casefold() not in {"none", "null", "nan", "na", "n/a", "<na>"}:
                    need(sh("RAID_identifier:" + value) in parent_keys[parent], "all RAID native/adv/source identifiers preserved")
            screened_ids.add(row["id"])
    need(len(screened_ids) == 268000, "complete screened lineage coverage")
    title_rows = read("research/english_v2_data/audit/article_title_bridge_metadata.parquet",
                      ["parent", "observable_key_hashes", "ambiguous_title_quarantine"]).to_pylist()
    for row in title_rows:
        need(set(row["observable_key_hashes"]) <= parent_keys[row["parent"]], "Wikipedia article-title bridge keys preserved")
        if row["ambiguous_title_quarantine"]:
            need("ambiguous_Wikipedia_article_title_quarantine" in seeds[lookup["native_parent"][row["parent"]]], "generic/ambiguous title quarantine propagated")
    def partition_and_flags(stored, stored_flags, stage):
        root_to_stored, stored_to_root, expected_flags, actual_flags = {}, {}, defaultdict(set), {}
        for i in index:
            root, value = uf.find(i), stored[i]
            need(root not in root_to_stored or root_to_stored[root] == value, stage + " missing required edge")
            need(value not in stored_to_root or stored_to_root[value] == root, stage + " unexplained additional merge")
            root_to_stored[root], stored_to_root[value] = value, root
            expected_flags[root].update(seeds[i])
            flags = frozenset(stored_flags[i])
            need(root not in actual_flags or actual_flags[root] == flags, stage + " barrier flags differ inside a component")
            actual_flags[root] = flags
        need(all(actual_flags[root] == flags for root, flags in expected_flags.items()), stage + " exact whole-component seed flag union")
        return len(root_to_stored)
    pre_count = partition_and_flags(pre, preflags, "pre_near")
    annotated = read("research/english_v2_data/audit/joint_near_annotated_edges.parquet",
        ["left_uid", "right_uid", "jaccard", "kind", "left_pre_component", "right_pre_component",
         "left_pre_barrier_flags", "right_pre_barrier_flags", "different_pre_components"]).to_pylist()
    source_edges = read("research/english_v2_data/audit/full_lexical_near_edges.parquet", ["left_uid", "right_uid", "jaccard", "kind"]).to_pylist()
    inc = read("research/english_v2_data/audit/raid_observed_reference_near_edges.parquet",
               ["left_normalized_sha256", "right_normalized_sha256", "jaccard", "kind"]).to_pylist()
    expected_edges = {(r["left_uid"], r["right_uid"], r["kind"], r["jaccard"]) for r in source_edges}
    reverse_uid = {node: uid for uid, node in uid_nodes.items()}
    expected_edges.update((reverse_uid[lookup["normalized_body"][r["left_normalized_sha256"]]],
                           reverse_uid[lookup["normalized_body"][r["right_normalized_sha256"]]], r["kind"], r["jaccard"]) for r in inc)
    need(expected_edges == {(r["left_uid"], r["right_uid"], r["kind"], r["jaccard"]) for r in annotated} and
         len(annotated) == 12932, "all 12928 base and four observed-reference verified edges preserved exactly")
    for row in annotated:
        i, j = uid_nodes[row["left_uid"]], uid_nodes[row["right_uid"]]
        need(row["left_pre_component"] == pre[i] and row["right_pre_component"] == pre[j] and
             row["left_pre_barrier_flags"] == preflags[i] and row["right_pre_barrier_flags"] == preflags[j] and
             row["different_pre_components"] == (pre[i] != pre[j]), "near annotations align to pre-near identities")
        uf.union(i, j)
    final_count = partition_and_flags(final, finalflags, "after_near")
    need(final_count == manifest["joint_components"], "exact final whole-graph component count")
    labels = defaultdict(set)
    for row in new:
        if row["y_ai"] in (0, 1):
            labels[row["normalized_sha256"]].add(row["y_ai"])
    for relative in ("data/english_v1/full_index.parquet", "data/english_augmentation_v2/full_index.parquet",
                     "data/english_external_v1/full_index.parquet", "data/raid_external_v1/full_index.parquet", "data/raid_external_v2/full_index.parquet"):
        for row in read(relative, ["normalized_sha256", "y_ai"]).to_pylist():
            if row["y_ai"] in (0, 1):
                labels[row["normalized_sha256"]].add(row["y_ai"])
    for split in ("train", "validation", "calibration"):
        for row in read(f"data/english_v1/pilot/{split}.parquet", ["id", "y_ai"]).to_pylist():
            labels[old_id_norm[row["id"]]].add(row["y_ai"])
    conflicts = {norm for norm, yy in labels.items() if len(yy) > 1}
    need(len(conflicts) == manifest["new_binary_conflict_normalized_hashes"] == 7, "all-old and new exact binary-label conflicts audited")
    forbidden = {"wikiHow_forbidden", "missing_parent_ID", "ambiguous_Wikipedia_article_title_quarantine"}
    fresh_barrier = {"old_used_or_reserved_forbid_fresh", "Beemo_documentation_or_viewer_observed", "upstream_documentation_example_observed", "prior_manual_review_observed"}
    for row in new:
        i = lookup["native_parent"][row["parent"]]
        flags = set(finalflags[i])
        need(pre[i] == row["pre_near_component_id"] and final[i] == row["component_id"] and flags == set(row["barrier_flags"]), "all new candidate components/flags aligned")
        conflict = row["normalized_sha256"] in conflicts
        qualified = row["structural_core"] and row["publisher_English"] and row["source_rights_allowed"] and not conflict and not forbidden & flags
        fresh = qualified and not fresh_barrier & flags
        train = qualified and "old_nontrain_or_reserved_forbid_new_training" not in flags
        need(row["binary_conflicting_normalized_text"] == conflict and row["capacity_qualified"] == qualified and
             row["candidate_fresh"] == fresh and row["candidate_train"] == train and
             row["candidate_train_under_draft_holdouts"] == (train and not {"draft_NQ_source_holdout", "draft_Gemini_nominal_holdout"} & flags),
             "new capacity predicates recomputed without runtime-language filtering")
    ledger = read("research/english_v2_data/audit/old_component_role_memberships.parquet", ["component_id", "roles"]).to_pylist()
    for row in ledger:
        flags = set(seeds[lookup["old_component"][row["component_id"]]])
        if "train_fit" in row["roles"]:
            need("old_actual_fit_kin" in flags, "independently audited training ancestry protected")
        if set(row["roles"]) & {"development", "calibration", "old_final"}:
            need("old_used_or_reserved_forbid_fresh" in flags and "old_nontrain_or_reserved_forbid_new_training" in flags,
                 "independently audited nontraining roles protected")
    existing = read("research/english_v2_data/audit/joint_existing_candidate_index.parquet",
        ["id", "old_component_id", "pre_near_component_id", "component_id", "normalized_sha256", "y_ai",
         "wordlike_tokens", "capacity_gate", "structural_quality_reason", "barrier_flags",
         "binary_conflicting_normalized_text", "capacity_qualified", "candidate_fresh"]).to_pylist()
    original_existing_ids = read("research/english_v2_data/audit/existing_graph_candidate_index.parquet", ["id"])["id"].to_pylist()
    need(len(existing) == len(original_existing_ids) and {r["id"] for r in existing} == set(original_existing_ids), "all old unused candidate metadata retained")
    for row in existing:
        i = lookup["old_component"][row["old_component_id"]]
        flags, conflict = set(finalflags[i]), row["normalized_sha256"] in conflicts
        qualified = row["structural_quality_reason"] is None and row["y_ai"] in (0, 1) and not conflict and not forbidden & flags
        need(row["pre_near_component_id"] == pre[i] and row["component_id"] == final[i] and
             set(row["barrier_flags"]) == flags and row["binary_conflicting_normalized_text"] == conflict and
             row["capacity_qualified"] == qualified and row["candidate_fresh"] == (qualified and not fresh_barrier & flags),
             "old candidate ancestry/structural/conflict/fresh predicates independently recomputed")
        if qualified:
            need(row["capacity_gate"] == "prior_admitted" and row["wordlike_tokens"] >= 128, "no old admission or core scope silently relaxed")
    role_pairs = list(itertools.combinations(("train_fit", "development", "calibration", "old_final"), 2))
    def old_connections(column):
        grouped = defaultdict(lambda: defaultdict(set))
        for row in ledger:
            group = column[lookup["old_component"][row["component_id"]]]
            for role in row["roles"]:
                grouped[group][role].add(row["component_id"])
        return {a + "__" + b: sum(bool(rr.get(a) and rr.get(b) and len(rr[a] | rr[b]) > 1) for rr in grouped.values()) for a, b in role_pairs}
    before_roles, after_roles = old_connections(pre), old_connections(final)
    need(all(sha(ROOT / relative) == value for relative, value in inputs.items()), "all graph evidence remained immutable during independent audit")
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "graph_manifest_sha256": expected_sha, "checks": checks,
        "graph_reconstructed_from_metadata": True, "graph_nodes": len(index), "pre_near_components": pre_count,
        "final_components": final_count, "verified_edges": len(annotated), "conservative_edges": 0,
        "observed_reference_native_body_links_checked": len(observed), "binary_label_conflict_hashes": len(conflicts),
        "cross_old_role_connections_between_distinct_old_components_before_near": before_roles,
        "cross_old_role_connections_between_distinct_old_components_after_verified_near": after_roles,
        "fresh_runtime_unsupported_rows_retained": sum(r["candidate_fresh"] and not r["runtime_language_supported_diagnostic"] for r in new),
        "metadata_only": True, "body_columns_predictions_models_read": False, "roles_allocated_here": False,
        "input_sha256": inputs, "metadata_projections": projections,
        "limitations": ["Jaccard is trusted only as bound producer evidence; independent audit rereads no bodies.",
            "Approximate lexical plus available native/title/hash relationships do not authenticate hidden parents, semantic or pretrained-model independence.",
            "Old-component role ledger includes conservative actual-fit supersets and early pilot roles; preexisting archived v1 multi-role memberships remain separately disclosed."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.expected_manifest_sha)
    result["audit_code_sha256"] = sha(__file__)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "graph_manifest_sha256", "checks", "graph_nodes", "final_components", "verified_edges", "fresh_runtime_unsupported_rows_retained")}))
