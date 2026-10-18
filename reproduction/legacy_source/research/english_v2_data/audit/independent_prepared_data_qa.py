"""Independent prepared-role QA using metadata projections and binary hashes.

Never opens text columns, model weights, scores, raw NPZs, or text SQLite rows.
The bound materializer owns text parsing/structural checks; this pass verifies
identity, role lineage, reserved connected components, labels and all outputs.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/data_audit_20261017/vendor_fast"))
import pyarrow.parquet as pq

META = ("id", "eligible", "y_ai", "family", "model", "mask_family", "mask_model", "out_of_model_catalog",
        "canonical_model_id", "source", "domain", "collector", "group_id", "native_parent_id", "split",
        "dataset_role", "language", "language_evidence", "runtime_language_supported_diagnostic", "wordlike_tokens",
        "raw_sha256", "normalized_sha256", "variant", "config", "new_vs_replay", "rewrite_kind", "original_generator",
        "last_editor", "view_human_origin_rewrite", "view_AI_origin_rewrite", "source_revision")
DEV = ("validation_general", "dev_source_domain", "dev_modern", "dev_paraphrase", "dev_legacy_main", "dev_legacy_creative")
CAL = ("cal_general", "cal_source_domain", "cal_modern", "cal_paraphrase", "cal_historical_WP")
FINAL = ("final_main", "final_source_domain", "final_modern", "final_paraphrase")
PRIMITIVES = ("train_registered", *DEV, *CAL, *FINAL)
FRESH = set(PRIMITIVES) - {"train_registered", "dev_legacy_main", "dev_legacy_creative", "cal_historical_WP"}
FORBIDDEN = {"wikiHow_forbidden", "ambiguous_Wikipedia_article_title_quarantine", "missing_parent_ID"}
FRESH_FLAGS = {"old_used_or_reserved_forbid_fresh", "Beemo_documentation_or_viewer_observed",
               "upstream_documentation_example_observed", "prior_manual_review_observed"}
HOLDOUT = {"draft_NQ_source_holdout", "draft_Gemini_nominal_holdout"}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def audit(manifest_path, allocation_path, expected_allocation_sha):
    checks, bound, projections = 0, {}, []
    def need(condition, message):
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1
    def bind(path, expected=None):
        path = Path(path)
        path = path if path.is_absolute() else ROOT / path
        value = sha(path)
        if expected is not None:
            need(value == expected, "bound SHA mismatch: " + str(path.relative_to(ROOT)))
        bound[str(path.relative_to(ROOT))] = value
        return path
    def read(path, columns):
        need(not set(columns) & {"text", "generation", "prompt", "title", "canonical_title", "native_title"}, "body projection forbidden")
        p = bind(path)
        table = pq.read_table(p, columns=list(columns))
        projections.append({"path": str(p.relative_to(ROOT)), "columns": list(columns), "rows": table.num_rows})
        return table.to_pylist()
    manifest_path = bind(manifest_path)
    allocation_path = bind(allocation_path, expected_allocation_sha)
    m, allocation = json.loads(manifest_path.read_text()), json.loads(allocation_path.read_text())
    need(allocation["status"] == "FROZEN_BEFORE_V2_TRAINING_AND_SCORES" and
         m["status"] == "FROZEN_DATA_BEFORE_V2_MODEL_PREDICTIONS" and m["frozen"] is True, "correct frozen pre-score stages required")
    need(m["protocol"]["sha256"] == expected_allocation_sha and ROOT / m["protocol"]["path"] == allocation_path and
         m["allocation_contract"] == allocation["data_allocation_contract"], "prepared roles match exact allocation contract")
    need(datetime.fromisoformat(allocation["created_at_utc"]) <= datetime.fromisoformat(m["frozen_at_utc"]), "allocation frozen before materialization")
    need(sha(ROOT / "experiments/english_v2/prepare_registered_data.py") == m["role_builder_sha256"] == allocation["role_builder_sha256"], "actual frozen builder source unchanged")
    graph_path = bind("research/english_v2_data/audit/joint_full_near_capacity.json", m["graph_manifest_sha256"])
    need(m["graph_manifest_sha256"] == allocation["joint_graph_manifest_sha256"], "allocation and prepared data use identical graph")
    graph_qa_path = bind(m["graph_metadata_QA"]["path"], m["graph_metadata_QA"]["sha256"])
    graph_qa = json.loads(graph_qa_path.read_text())
    need(graph_qa["passed"] is True and graph_qa["graph_manifest_sha256"] == sha(graph_path), "independent whole-graph QA applicable")
    for path, spec in m["input_files"].items():
        bind(path, spec["sha256"])
    files = m["output_files"]
    need(set(files) == set(PRIMITIVES) | {"calibration", "dev_challenge_stress"}, "all eighteen declared output files must be audited")
    outputs, audited = {}, {}
    for name, spec in files.items():
        p = bind(spec["path"], spec["sha256"])
        need(p.stat().st_size == spec["bytes"], "output byte length: " + name)
        rows = read(p, META)
        outputs[name], audited[name] = rows, spec["sha256"]
        need(len(rows) == spec["rows"] and len({r["id"] for r in rows}) == len(rows), "complete unique output IDs: " + name)
        need(all(isinstance(r["id"], str) and r["id"].strip() and isinstance(r["group_id"], str) and r["group_id"].strip() and
             isinstance(r["y_ai"], int) and not isinstance(r["y_ai"], bool) and r["y_ai"] in (0, 1) and r["eligible"] is True and
             isinstance(r["wordlike_tokens"], int) and not isinstance(r["wordlike_tokens"], bool) and r["wordlike_tokens"] >= 128 and
             re.fullmatch(r"[0-9a-f]{64}", r["raw_sha256"] or "") and re.fullmatch(r"[0-9a-f]{64}", r["normalized_sha256"] or "") and
             r["language"] == "en" and r["runtime_language_supported_diagnostic"] in (True, False, None) for r in rows), "explicit eligible core English metadata: " + name)
        need(all(r["split"] == spec["role"] for r in rows), "output split role: " + name)
        if "class_counts" in spec:
            need(dict(Counter(str(r["y_ai"]) for r in rows)) == spec["class_counts"] and
                 len({r["group_id"] for r in rows}) == spec["groups"] and
                 spec["core_counts"] == {"n": len(rows), "human_n": sum(r["y_ai"] == 0 for r in rows),
                     "AI_n": sum(r["y_ai"] == 1 for r in rows), "groups": len({r["group_id"] for r in rows})}, "declared class/core/group denominators: " + name)
            need(spec["runtime_unsupported_diagnostic_rows"] == sum(r["runtime_language_supported_diagnostic"] is False for r in rows) and
                 spec["runtime_support_unknown_diagnostic_rows"] == sum(r["runtime_language_supported_diagnostic"] is None for r in rows), "False and unknown language diagnostics distinct: " + name)
    expected_counts = {"train_registered": (30000, 30000), "validation_general": (1000, 1000), "dev_source_domain": (400, 400),
        "dev_modern": (800, 800), "dev_paraphrase": (300, 600), "dev_legacy_main": (1000, 1000), "dev_legacy_creative": (1000, 1000),
        "cal_general": (1000, 1000), "cal_source_domain": (400, 400), "cal_modern": (1000, 1000), "cal_paraphrase": (400, 800),
        "cal_historical_WP": (1000, 1000), "calibration": (3800, 4200), "final_main": (1500, 1500), "final_source_domain": (400, 400),
        "final_modern": (1000, 1000), "final_paraphrase": (600, 1200), "dev_challenge_stress": (1500, 1800)}
    for name, counts in expected_counts.items():
        need(tuple(sum(r["y_ai"] == y for r in outputs[name]) for y in (0, 1)) == counts, "fixed exact class quotas: " + name)
    def idmap(rows):
        return {r["id"]: r for r in rows}
    def equivalent(a, b):
        return all(a[k] == b[k] for k in META if k != "dataset_role")
    def alias(target, sources):
        rows = [r for source in sources for r in outputs[source]]
        need(len(rows) == len({r["id"] for r in rows}), "alias source IDs disjoint")
        expected, actual = idmap(rows), idmap(outputs[target])
        need(set(expected) == set(actual) and all(equivalent(expected[i], actual[i]) for i in expected), "alias preserves all metadata: " + target)
    alias("calibration", CAL)
    alias("dev_challenge_stress", ("dev_source_domain", "dev_modern", "dev_paraphrase"))
    primitive_ids, primitive_groups = {}, {}
    for name in PRIMITIVES:
        primitive_ids[name] = set(idmap(outputs[name]))
        primitive_groups[name] = {r["group_id"] for r in outputs[name]}
    for a, b in itertools.combinations(PRIMITIVES, 2):
        need(not primitive_ids[a] & primitive_ids[b] and not primitive_groups[a] & primitive_groups[b], "atomic whole-component role isolation: " + a + "/" + b)
    nodes = read("research/english_v2_data/audit/joint_component_node_index.parquet", ["kind", "identity", "component_id", "barrier_flags"])
    bykind = defaultdict(dict)
    for r in nodes:
        bykind[r["kind"]][r["identity"]] = r
    del nodes
    ledger = read("research/english_v2_data/audit/old_component_role_memberships.parquet", ["component_id", "roles"])
    global_roles = defaultdict(set)
    for r in ledger:
        global_roles[bykind["old_component"][r["component_id"]]["component_id"]].update(r["roles"])
    graph_ids = {r["id"]: r for r in read("data/english_calibration_v2/graph_index.parquet", ["id", "component_id", "raw_sha256", "normalized_sha256"])}
    actualfit = set()
    for p in ("data/english_v1/train.parquet", "data/english_augmentation_v2/train_mixed100k_with_official.parquet",
              "data/english_adaptation_v3/train_replay50k_wp.parquet", "data/english_v1/pilot/train.parquet"):
        actualfit.update(r["id"] for r in read(p, ["id"]))
    need(len(actualfit) == m["actual_fit_ID_superset"] == 174019, "complete old training ancestry superset")
    inherited = {"dev_legacy_main": ("data/english_v1/validation.parquet", "development"),
                 "dev_legacy_creative": ("data/english_v1/dev_challenge_source_domain.parquet", "development"),
                 "cal_historical_WP": ("data/english_calibration_v2/wp_calibration.parquet", "calibration")}
    for role, (p, expected_role) in inherited.items():
        ids = {r["id"] for r in read(p, ["id"])}
        need(primitive_ids[role] <= ids and all(global_roles[g] == {expected_role} for g in primitive_groups[role]), "exact-source single-role historical retention: " + role)
    selection_spec = m["selection_index"]
    selection_path = bind(selection_spec["path"], selection_spec["sha256"])
    selection = read(selection_path, ["row_record_id", "id", "parent", "component_id", "raw_sha256", "normalized_sha256", "y_ai",
        "source", "domain", "collector", "variant", "config", "wordlike_tokens", "runtime_language_supported_diagnostic", "new_vs_replay", "nominal_generator", "reserved_role"])
    selected_by_id = {r["id"]: r for r in selection}
    need(len(selected_by_id) == len(selection) == sum(len(outputs[n]) for n in PRIMITIVES) == 84900, "complete selection ledger without duplicate IDs")
    reservation_spec = m["component_reservations"]
    reservation_path = bind(reservation_spec["path"], reservation_spec["sha256"])
    reservations = read(reservation_path, ["component_id", "reserved_role"])
    reserve = {r["component_id"]: r["reserved_role"] for r in reservations}
    need(len(reserve) == len(reservations) and set(reserve) == set().union(*primitive_groups.values()), "every selected full component reserved once")
    newpool = {r["row_record_id"]: r for r in read("research/english_v2_data/audit/joint_new_candidate_index.parquet",
        ["row_record_id", "parent", "y_ai", "raw_sha256", "normalized_sha256", "candidate_fresh", "capacity_qualified", "nominal_generator"])}
    oldpool = {r["id"]: r for r in read("research/english_v2_data/audit/joint_existing_candidate_index.parquet",
        ["id", "y_ai", "raw_sha256", "normalized_sha256", "candidate_fresh", "capacity_qualified"])}
    conflicts = {r["normalized_sha256"] for r in read("research/english_v2_data/audit/joint_binary_conflict_hashes.parquet", ["normalized_sha256"])}
    for role in PRIMITIVES:
        reserved_role = "dev_general" if role == "validation_general" else role
        for row in outputs[role]:
            seed = selected_by_id[row["id"]]
            need(seed["reserved_role"] == reserved_role == reserve[row["group_id"]] and
                 seed["component_id"] == row["group_id"] and seed["parent"] == row["native_parent_id"] and
                 all(seed[key] == row[key] for key in ("y_ai", "raw_sha256", "normalized_sha256", "source", "domain", "collector", "variant", "config", "wordlike_tokens", "runtime_language_supported_diagnostic")), "selected-row metadata and reservation aligned")
            node = bykind["normalized_body"][row["normalized_sha256"]]
            flags = set(node["barrier_flags"])
            need(node["component_id"] == row["group_id"] and not FORBIDDEN & flags and row["normalized_sha256"] not in conflicts, "saved full graph and exclusion flags respected")
            if row["native_parent_id"].startswith("old:"):
                old = graph_ids[row["id"]]
                need(old["raw_sha256"] == row["raw_sha256"] and old["normalized_sha256"] == row["normalized_sha256"] and
                     bykind["old_component"][old["component_id"]]["component_id"] == row["group_id"], "old metadata identity retained")
            else:
                need(bykind["native_parent"][row["native_parent_id"]]["component_id"] == row["group_id"], "native parent and all variants share full graph role")
                original = newpool[seed["row_record_id"]]
                need(original["parent"] == row["native_parent_id"] and all(original[k] == row[k] for k in ("y_ai", "raw_sha256", "normalized_sha256")) and
                     original["capacity_qualified"] and not row["mask_family"] and not row["mask_model"] and row["model"] is None, "new-source binary labels and no stock-version supervision")
            if role in FRESH or role == "train_registered" and row["new_vs_replay"] == "new":
                need(not FRESH_FLAGS & flags, "no old used/observed/reserved kin admitted to new fresh roles")
                original = newpool[seed["row_record_id"]] if row["collector"] in {"MAGA", "RAID", "Beemo"} else oldpool[row["id"]]
                need(original["candidate_fresh"], "freshness comes from complete graph eligibility")
            if HOLDOUT & flags:
                need(role in {"final_source_domain", "final_modern"}, "all heldout-model/source relatives final-only or unused")
            if role == "train_registered" and row["new_vs_replay"] == "replay":
                need(row["id"] in actualfit and global_roles[row["group_id"]] == {"train_fit"}, "replay exclusively actual-training ancestry, without nontraining kin")
            if role == "cal_historical_WP":
                need(row["new_vs_replay"] == "historical_calibration_replay", "historical calibration never mislabeled fresh")
    train = outputs["train_registered"]
    new_counts = Counter(r["y_ai"] for r in train if r["new_vs_replay"] == "new")
    replay_counts = Counter(r["y_ai"] for r in train if r["new_vs_replay"] == "replay")
    need(new_counts[0] == new_counts[1] == m["new_training_each_class"] and 6000 <= new_counts[0] <= 10000 and
         replay_counts[0] == replay_counts[1] == m["old_replay_each_class"] == 30000 - new_counts[0], "fixed new/replay training composition")
    wp = Counter(r["y_ai"] for r in train if str(r["domain"]).casefold() == "wp")
    need(wp[0] == wp[1] == m["balanced_WP_replay_each_class"] <= 1061, "real balanced WP replay capacity")
    pair_roles = {"dev_source_domain": ("S2ORC", 400), "dev_modern": ("Wikipedia", 800), "cal_source_domain": ("S2ORC", 400),
                  "cal_modern": ("Wikipedia", 1000), "final_source_domain": ("Natural Questions", 400)}
    for role, (domain, count) in pair_roles.items():
        rr = outputs[role]
        grouped = defaultdict(list)
        for r in rr:
            need(r["collector"] == "MAGA" and r["domain"] == domain, "registered modern/source domain")
            grouped[r["native_parent_id"]].append(r)
        need(len(grouped) == count and all(Counter(r["y_ai"] for r in rows) == {0: 1, 1: 1} and
             len({r["group_id"] for r in rows}) == 1 for rows in grouped.values()) and len(primitive_groups[role]) == count, "one matched H/AI native parent per independent component")
    modern = outputs["final_modern"]
    need(Counter((r["domain"], r["y_ai"]) for r in modern) == {("Wikipedia", 0): 700, ("Wikipedia", 1): 700, ("S2ORC", 0): 300, ("S2ORC", 1): 300}, "fixed final modern domain quotas")
    modern_parents = defaultdict(list)
    for r in modern:
        modern_parents[r["native_parent_id"]].append(r)
    need(len(modern_parents) == len(primitive_groups["final_modern"]) == 1000 and all(Counter(x["y_ai"] for x in rr) == {0: 1, 1: 1} for rr in modern_parents.values()), "final modern matched controls")
    gemini = Counter(r["domain"] for r in modern if r["y_ai"] == 1 and r["original_generator"] == "Gemini-2.0-flash")
    need(sum(gemini.values()) == 300 and dict(gemini) == m["final_modern_Gemini_domain_quotas"], "registered heldout Gemini minimum and exact first-stage quotas")
    rewrite_views = {}
    for role, each_domain in (("dev_paraphrase", 150), ("cal_paraphrase", 200), ("final_paraphrase", 300)):
        rr = outputs[role]
        grouped = defaultdict(list)
        for r in rr:
            need(r["collector"] == "RAID" and r["domain"] in {"books", "news"}, "RAID rewrite source")
            grouped[r["native_parent_id"]].append(r)
            human, human_edit = r["variant"] == "human", r["variant"] == "human_origin_DIPPER"
            need(r["y_ai"] == (0 if human else 1) and r["view_human_origin_rewrite"] == (human or human_edit) and
                 r["view_AI_origin_rewrite"] == (human or not human_edit), "human original and both AI-edit origins labeled separately")
        need(len(grouped) == len(primitive_groups[role]) == 2 * each_domain and
             Counter(rows[0]["domain"] for rows in grouped.values()) == {"books": each_domain, "news": each_domain} and
             all(Counter(r["variant"] for r in rows) == {"human": 1, "human_origin_DIPPER": 1, "AI_origin_DIPPER": 1} and
                 len({r["group_id"] for r in rows}) == 1 for rows in grouped.values()), "exact three-way native rewrite pairing and domain quotas")
        rewrite_views[role] = {view: {"n": sum(r[view] for r in rr), "human": sum(r[view] and r["y_ai"] == 0 for r in rr),
            "AI": sum(r[view] and r["y_ai"] == 1 for r in rr)} for view in ("view_human_origin_rewrite", "view_AI_origin_rewrite")}
    need(all(sha(ROOT / p) == value for p, value in bound.items()), "all allocation/graph/input/output evidence unchanged through audit")
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "prepared_manifest_sha256": sha(manifest_path), "allocation_protocol_sha256": expected_allocation_sha,
        "audited_output_sha256": audited, "training_inputs_sha256": {"train": audited["train_registered"],
            "validation": audited["validation_general"], "dev_challenge": audited["dev_challenge_stress"]},
        "checks": checks, "metadata_only": True, "body_columns_models_or_scores_read": False,
        "all_atomic_roles_whole_component_disjoint": True, "calibration_five_buckets_disjoint_exact_union": True,
        "stress_exact_intentional_three_fresh_dev_alias": True, "validation_exact_general_dev_alias": True,
        "whole_component_reservations_cover_unselected_siblings_by_graph_identity": True,
        "training_composition": {"new_each_class": new_counts[0], "replay_each_class": replay_counts[0], "WP_each_class": wp[0]},
        "per_output_counts": {name: {"n": len(rr), "human": sum(r["y_ai"] == 0 for r in rr), "AI": sum(r["y_ai"] == 1 for r in rr),
            "groups": len({r["group_id"] for r in rr}), "language_false": sum(r["runtime_language_supported_diagnostic"] is False for r in rr),
            "language_unknown": sum(r["runtime_language_supported_diagnostic"] is None for r in rr)} for name, rr in outputs.items()},
        "rewrite_required_views_shared_human_controls_not_additive": rewrite_views,
        "input_sha256": bound, "read_projections": projections,
        "limitations": ["No body reparse or text-to-hash recomputation in this independent metadata-only pass; exact output bytes and bound materializer checks provide that layer.",
            "Full graph covers observed native/title/hash/approximate lexical relations, not hidden parent, semantic or pretrained-model independence.",
            "Training ancestry is a conservative input superset; old CAL replay and observed dev retention are not newly independent data.",
            "Runtime language false and unknown are preserved separately; later formal export recomputes the same guard and retains all class denominators.",
            "This verifies the registered deterministic allocation's identities, reservations, quotas and predicates; it does not independently recompute old-fit body-derived structural eligibility or rerun model selection."]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--allocation", required=True)
    p.add_argument("--expected-allocation-sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.manifest, args.allocation, args.expected_allocation_sha)
    result["audit_code_sha256"] = sha(__file__)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "prepared_manifest_sha256", "allocation_protocol_sha256", "checks", "training_inputs_sha256", "training_composition")}))
