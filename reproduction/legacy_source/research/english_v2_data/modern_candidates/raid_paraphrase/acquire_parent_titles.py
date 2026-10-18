"""Only native clean-human title metadata; no prompt/generation is read.

Bind title values to the completed parent lineage by exact native document ID.
Canonicalization is intentionally literal: NFKC, casefold, whitespace. It does
not infer Wikipedia article aliases or strip year/disambiguation suffixes.
"""
import collections
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import unicodedata
import urllib.request

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
RAW=ROOT/"research/english_v2_data/raw/raid_paired_range_v1"
OUT=ROOT/"research/english_v2_data/raw/raid_paired_range_v1_titles"
sys.path.insert(0,str(ROOT/"research/data_audit_20261017/vendor_fast"))
import pyarrow as pa
import pyarrow.parquet as pq
from audit_parquet_footers import FooterFile


def sha(path):
    with Path(path).open("rb")as f:return hashlib.file_digest(f,"sha256").hexdigest()


def normalized_title(value):
    return " ".join(unicodedata.normalize("NFKC",value).casefold().split()) if isinstance(value,str) else None


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not(OUT/"COMPLETED.json").exists(),"never overwrite completed title evidence"
    qa=json.loads((RAW/"lineage_qa.json").read_text())
    assert qa["passed"]is True
    completed=json.loads((RAW/"COMPLETED.json").read_text())
    assert qa["acquisition_completion_sha256"]==sha(RAW/"COMPLETED.json")
    qa_inputs={item["file"]:item for item in qa["inputs"]}
    result=[];evidence=[]
    for source_item in completed["files"]:
        name=Path(source_item["source"]).stem
        folder=RAW/name
        receipt=json.loads((folder/"receipt.json").read_text())
        assert sha(folder/"receipt.json")==source_item["receipt_sha256"]
        if not receipt["selected_rows"]:continue
        index=ROOT/receipt["lineage_index"]["path"]
        assert sha(index)==receipt["lineage_index"]["sha256"]
        humans=[r for r in pq.read_table(index,columns=["id","source_id","domain","model","attack"]).to_pylist() if r["model"]=="human"and r["attack"]=="none"]
        if not humans:continue
        target={r["id"]:r for r in humans}
        spec_file=BASE/f"{name}.parquet.footer_metadata.json"
        spec=json.loads(spec_file.read_text());size=spec["size"]
        assert sha(spec_file)==receipt["footer_spec_sha256"]
        bridge=folder/"screened_lineage_metadata.parquet"
        assert sha(bridge)==qa_inputs[name]["screened_lineage_sha256"]
        metadata=pq.read_table(bridge,columns=["id","source_id","domain","model","attack"]).to_pylist()
        footer_record=next(r for r in receipt["ranges"] if r["purpose"]=="Parquet footer")
        footer_path=ROOT/footer_record["path"]
        assert sha(footer_path)==footer_record["sha256"]
        footer=footer_path.read_bytes();start=spec["footer_start"]
        parsed=pq.ParquetFile(FooterFile(size,start,footer))
        selected=[];positions=[];requests=[];cursor=0
        for gi in receipt["screened_metadata_rowgroups"]:
            n=parsed.metadata.row_group(gi).num_rows
            rows=metadata[cursor:cursor+n];cursor+=n
            if any(r["id"]in target for r in rows):
                selected.append(gi);positions.extend(rows)
                rg=parsed.metadata.row_group(gi)
                col=next(rg.column(ci)for ci in range(rg.num_columns)if rg.column(ci).path_in_schema=="title")
                offset=min(col.data_page_offset,col.dictionary_page_offset)if col.has_dictionary_page else col.data_page_offset
                requests.append((offset,offset+col.total_compressed_size,gi))
        assert cursor==len(metadata)
        output_folder=OUT/name;output_folder.mkdir(exist_ok=True)
        def fetch(item):
            a,b,gi=item
            path=output_folder/f"{a:012d}_{b:012d}.title.range"
            req=urllib.request.Request(spec["url"]+f"?native_title_range={a}-{b-1}",headers={"Range":f"bytes={a}-{b-1}","User-Agent":"metadata-only-parent-title-audit/1.0"})
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req,timeout=45)as response:
                        if response.status!=206 or response.headers.get("Content-Range")!=f"bytes {a}-{b-1}/{size}":raise RuntimeError("exact title range refused; response body unread")
                        raw=response.read(b-a+1)
                    if len(raw)!=b-a:raise RuntimeError("incomplete title range")
                    break
                except Exception:
                    if attempt==3:raise
            temp=path.with_name(path.name+".part");temp.write_bytes(raw);os.replace(temp,path)
            return a,raw,{"row_group":gi,"url":spec["url"],"source_lfs_sha256":spec["lfs_sha256"],"start":a,"stop_exclusive":b,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"path":str(path.relative_to(ROOT))}
        fetched=list(concurrent.futures.ThreadPoolExecutor(max_workers=8).map(fetch,requests))
        class TitleFile(FooterFile):
            def read(self,n=-1):
                position=self.pos;data=bytearray(super().read(n));stop=position+len(data)
                for offset,raw,_ in fetched:
                    a,b=max(position,offset),min(stop,offset+len(raw))
                    if a<b:data[a-position:b-position]=raw[a-offset:b-offset]
                return bytes(data)
        titles=pq.ParquetFile(TitleFile(size,start,footer)).read_row_groups(selected,columns=["title"],use_threads=False).column("title").to_pylist()
        assert len(titles)==len(positions)
        added=[]
        for r,title in zip(positions,titles):
            if r["id"]not in target:continue
            assert r==target[r["id"]]
            if title is not None and not isinstance(title,str):raise RuntimeError("unexpected native title type; do not infer")
            added.append({"native_human_id":r["id"],"source_id":r["source_id"],"domain":r["domain"],"native_title":title,
                "canonical_title":normalized_title(title),"canonicalization":"NFKC_casefold_unicode_whitespace_only",
                "native_title_missing":not isinstance(title,str)or not title.strip(),"converted_revision":"5008b2d167f74bfd4ceff6b3b0512b7097d7cbb1","converted_file":spec["path"]})
        assert {r["native_human_id"]for r in added}==set(target)
        result.extend(added)
        evidence.append({"source_file":spec["path"],"human_rows":len(added),"row_groups":selected,"new_title_ranges":[r[2]for r in fetched],
            "lineage_index_sha256":sha(index),"screened_metadata_sha256":sha(bridge),"original_acquisition_receipt_sha256":sha(folder/"receipt.json"),"footer_spec_sha256":sha(spec_file)})
    assert len(result)==3561 and len({r["source_id"]for r in result})==3561
    destination=OUT/"parent_titles.parquet"
    pq.write_table(pa.Table.from_pylist(result),destination,compression="zstd")
    report={"schema_version":1,"completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"passed":True,
        "purpose":"observable title bridge for whole-parent role isolation, not a role allocation","no_generation_or_prompt_columns_read":True,"no_detector_scores_or_manual_preview":True,
        "converted_revision":"5008b2d167f74bfd4ceff6b3b0512b7097d7cbb1","original_lineage_QA_sha256":sha(RAW/"lineage_qa.json"),
        "rows":len(result),"missing_title_rows":sum(r["native_title_missing"]for r in result),
        "domain_counts":dict(collections.Counter(r["domain"]for r in result)),
        "canonical_unique_by_domain":{d:len({r["canonical_title"]for r in result if r["domain"]==d and not r["native_title_missing"]})for d in {"books","news"}},
        "canonicalization":"NFKC_casefold_unicode_whitespace_only; no aliases/disambiguator deletion inferred",
        "NQ_question_is_not_wikipedia_title":True,"AI_variants_inherit_title_by":"source_id join to exact clean-human parent; no generated title substituted",
        "new_title_bytes":sum(r["bytes"]for e in evidence for r in e["new_title_ranges"]),"evidence":evidence,
        "output":{"path":str(destination.relative_to(ROOT)),"sha256":sha(destination)},"code_sha256":sha(__file__)}
    (OUT/"COMPLETED.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k]for k in ["passed","rows","missing_title_rows","domain_counts","canonical_unique_by_domain","new_title_bytes","output"]}))


if __name__=="__main__":main()
