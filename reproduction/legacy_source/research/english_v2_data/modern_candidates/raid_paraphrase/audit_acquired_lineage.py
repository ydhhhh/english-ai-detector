"""Metadata-only QA of completed RAID range acquisition; no text or scores read."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
RAW=ROOT/"research/english_v2_data/raw/raid_paired_range_v1"
sys.path.insert(0,str(ROOT/"research/data_audit_20261017/vendor_fast"))
import pyarrow.parquet as pq


def sha(path):
    with Path(path).open("rb")as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    complete=json.loads((RAW/"COMPLETED.json").read_text())
    assert complete["all_8_files_complete"] is True
    rows=[];originals=[];inputs=[]
    for item in complete["files"]:
        name=Path(item["source"]).stem
        folder=RAW/name
        receipt=json.loads((folder/"receipt.json").read_text())
        assert sha(folder/"receipt.json")==item["receipt_sha256"]
        for key in ["payload","lineage_index"]:
            path=ROOT/receipt[key]["path"]
            assert sha(path)==receipt[key]["sha256"]
        for record in receipt["ranges"]:
            path=ROOT/record["path"]
            assert path.stat().st_size==record["bytes"] and sha(path)==record["sha256"]
        if receipt["selected_rows"]:
            source=pq.read_table(ROOT/receipt["lineage_index"]["path"]).to_pylist()
            assert len(source)==receipt["selected_rows"]
            rows.extend(source)
        bridge=folder/"screened_lineage_metadata.parquet"
        originals.extend(pq.read_table(bridge).to_pylist())
        inputs.append({"file":name,"receipt_sha256":sha(folder/"receipt.json"),
            "lineage_index_sha256":receipt["lineage_index"]["sha256"],"screened_lineage_sha256":sha(bridge)})
    assert len(rows)==complete["selected_rows"]
    assert len({r["id"]for r in rows})==len(rows)
    for r in rows:
        assert isinstance(r["source_id"],str)and r["source_id"]
        assert isinstance(r["adv_source_id"],str)and r["adv_source_id"]
        assert r["mask_family"]is False and r["mask_model"]is False
        assert r["y_ai"]==(1 if r["attack"]=="paraphrase"else 0)
        assert r["attack"]=="paraphrase"or(r["attack"]=="none"and r["model"]=="human")
    original_map={r["id"]:r for r in originals if r["attack"]=="none"and r["domain"]in {"books","news"}}
    human_by_source={r["source_id"]:r for r in rows if r["y_ai"]==0}
    assert len(human_by_source)==sum(r["y_ai"]==0 for r in rows)
    issues=collections.Counter();domain_stats=[];parent_index=[]
    for domain in ["books","news"]:
        subset=[r for r in rows if r["domain"]==domain]
        human=[r for r in subset if r["y_ai"]==0]
        para=[r for r in subset if r["y_ai"]==1]
        pairs=collections.defaultdict(list)
        for r in para:
            pairs[r["source_id"]].append(r)
            base=original_map.get(r["adv_source_id"])
            if base is None:issues["paraphrase_missing_original_adv_source_metadata"]+=1
            elif any(base[key]!=r[key]for key in ["source_id","model","domain","decoding","repetition_penalty"]):
                issues["paraphrase_original_metadata_mismatch"]+=1
            if r["source_id"]not in human_by_source:issues["paraphrase_missing_clean_human_parent"]+=1
        exact=normalized=0
        for r in para:
            if r["model"]=="human"and r["source_id"]in human_by_source:
                h=human_by_source[r["source_id"]]
                exact+=r["raw_sha256"]==h["raw_sha256"]
                normalized+=r["normalized_sha256"]==h["normalized_sha256"]
        core_parents=[]
        for h in human:
            variants=pairs.get(h["source_id"],[])
            core_variants=[r for r in variants if r["word_like_tokens"]>=128]
            good=h["word_like_tokens"]>=128 and bool(core_variants)
            if good:core_parents.append(h["source_id"])
            parent_index.append({"domain":domain,"source_id":h["source_id"],"human_id":h["id"],
                "human_core_ge128":h["word_like_tokens"]>=128,"paraphrase_n":len(variants),
                "core_paraphrase_n":len(core_variants),"has_core_pair":good,
                "human_parent_hash":h["normalized_sha256"]})
        domain_stats.append({"domain":domain,"clean_human_rows":len(human),
            "clean_human_core_rows":sum(r["word_like_tokens"]>=128 for r in human),
            "paraphrase_rows":len(para),"paraphrase_core_rows":sum(r["word_like_tokens"]>=128 for r in para),
            "human_origin_paraphrase_rows":sum(r["model"]=="human"for r in para),
            "human_origin_paraphrase_core_rows":sum(r["model"]=="human"and r["word_like_tokens"]>=128 for r in para),
            "distinct_native_source_ids":len({r["source_id"]for r in subset}),"core_paired_native_sources":len(core_parents),
            "unchanged_human_paraphrase_raw":exact,"unchanged_human_paraphrase_normalized":normalized,
            "variants_per_parent":dict(sorted(collections.Counter(len(pairs[h["source_id"]])for h in human).items())),
            "core_AI_by_original_model":dict(sorted(collections.Counter(r["model"]for r in para if r["word_like_tokens"]>=128).items()))})
    index=RAW/"native_parent_index.parquet"
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(parent_index),index,compression="zstd")
    report={"schema_version":1,"completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "passed":not issues,"code_sha256":sha(__file__),"acquisition_completion_sha256":sha(RAW/"COMPLETED.json"),
        "body_columns_read":[],"detector_predictions_or_manual_preview":False,"range_bytes":complete["range_bytes"],
        "scope":"official converted partial-train books/news none-human plus all paraphrase; not all RAID",
        "domains":domain_stats,"structural_issues":dict(issues),"inputs":inputs,
        "native_parent_index":{"path":str(index.relative_to(ROOT)),"sha256":sha(index)},
        "critical_label_rule":"human+paraphrase is AI-edited positive, clean human+none is negative; ambiguous version masks false",
        "next_barriers":["supported-English guard","old actually-used graph protection","all new-source exact/near links","unchanged-human rewrite label conflict quarantine","source rights","whole-parent role freeze"],
        "eligible_fresh_capacity_after_all_barriers":None}
    out=RAW/"lineage_qa.json"
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":report["passed"],"domains":domain_stats,"issues":dict(issues),"range_bytes":report["range_bytes"],"report_sha256":sha(out)}))
    if issues:raise RuntimeError("structural lineage audit failed; do not allocate roles")


if __name__=="__main__":main()
