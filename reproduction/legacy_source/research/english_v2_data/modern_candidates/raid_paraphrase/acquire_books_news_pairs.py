"""Acquire only native RAID clean-human / DIPPER rows from pinned row groups.

Authorized v2 data preparation, without detector scores or corpus previews.
All HTTP ranges bind an immutable converted commit, byte interval and SHA256.
Other boundary rows retain lineage metadata only. This is not all-RAID coverage.
"""
from __future__ import annotations
import collections
import concurrent.futures
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
import urllib.request

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
RAW = ROOT / "research/english_v2_data/raw/raid_paired_range_v1"
sys.path.insert(0, str(ROOT / "research/data_audit_20261017/vendor_fast"))
import pyarrow as pa
import pyarrow.parquet as pq
from audit_parquet_footers import FooterFile

REV = "5008b2d167f74bfd4ceff6b3b0512b7097d7cbb1"
ORIGINAL_REV = "865cac74188466cb0c3b7574a10204007b57a459"
META = ["id", "adv_source_id", "source_id", "model", "decoding", "repetition_penalty", "attack", "domain"]
DOMAINS = {"books", "news"}
ATTACKS = {"none", "paraphrase"}
WORD = re.compile(r"\b\w+\b", re.UNICODE)
CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
POOL = concurrent.futures.ThreadPoolExecutor(max_workers=16)


def digest(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_name(path.name + f".part-{os.getpid()}")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    os.replace(temp,path)


class ProjectedFile(FooterFile):
    def __init__(self, size, start, footer, chunks):
        super().__init__(size,start,footer)
        self.chunks = chunks
    def read(self,n=-1):
        start=self.pos
        raw=bytearray(super().read(n))
        stop=start+len(raw)
        for offset,data in self.chunks:
            a,b=max(offset,start),min(offset+len(data),stop)
            if a<b:raw[a-start:b-start]=data[a-offset:b-offset]
        return bytes(raw)


def acquire(name):
    spec_path = BASE / f"{name}.parquet.footer_metadata.json"
    spec = json.loads(spec_path.read_text())
    source_url, size = spec["url"], spec["size"]
    assert f"/{REV}/" in source_url
    folder = RAW / name
    folder.mkdir(parents=True,exist_ok=True)
    output, index, receipt = folder/"payload.parquet", folder/"lineage_index.parquet", folder/"receipt.json"
    if receipt.exists():
        old=json.loads(receipt.read_text())
        assert old["code_sha256"]==CODE_SHA and old["footer_spec_sha256"]==digest(spec_path)
        assert old["payload"]["sha256"]==digest(output) and old["lineage_index"]["sha256"]==digest(index)
        print(json.dumps({"file":name,"resumed_complete":True,"rows":old["selected_rows"]}),flush=True)
        return old
    ledger={}
    def fetch(start,stop,purpose):
        target=folder/f"{start:012d}_{stop:012d}.range"
        sidecar=target.with_suffix(".range.json")
        if target.exists() and sidecar.exists():
            record=json.loads(sidecar.read_text())
            assert record["url"]==source_url and record["start"]==start and record["stop_exclusive"]==stop
            assert target.stat().st_size==stop-start and record["sha256"]==digest(target)
            data=target.read_bytes()
        else:
            request=urllib.request.Request(source_url+f"?paired_acquisition_range={start}-{stop-1}",headers={"Range":f"bytes={start}-{stop-1}","User-Agent":"v2-paired-data-acquisition/1.0"})
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(request,timeout=55) as r:
                        if r.status!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{stop-1}/{size}":
                            raise RuntimeError("server refused exact range; body not read")
                        data=r.read(stop-start+1)
                    if len(data)!=stop-start:raise RuntimeError("partial range size mismatch")
                    break
                except Exception:
                    if attempt==3:raise
            temp=target.with_name(target.name+f".part-{os.getpid()}")
            temp.write_bytes(data);os.replace(temp,target)
            record={"url":source_url,"source_file_size":size,"source_lfs_sha256":spec["lfs_sha256"],
                "start":start,"stop_exclusive":stop,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),
                "purpose":purpose,"retrieved_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
            write_json(sidecar,record)
        ledger[(start,stop)]={**record,"path":str(target.relative_to(ROOT))}
        return start,data
    footer_start=spec["footer_start"]
    _,footer=fetch(footer_start,size,"Parquet footer")
    parsed=pq.ParquetFile(FooterFile(size,footer_start,footer))
    groups=[];meta_ranges=[]
    for g in spec["row_groups"]:
        cols=g["metadata_columns"];dom,atk=cols["domain"],cols["attack"]
        if not any(dom.get("minimum","")<=d<=dom.get("maximum","\U0010ffff") for d in DOMAINS):continue
        if not any(atk.get("minimum","")<=a<=atk.get("maximum","\U0010ffff") for a in ATTACKS):continue
        start=min(cols[k]["start"]for k in META);stop=max(cols[k]["stop_exclusive"]for k in META)
        rg=parsed.metadata.row_group(g["row_group"])
        for ci in range(rg.num_columns):
            c=rg.column(ci)
            if c.path_in_schema not in META:
                offset=min(c.data_page_offset,c.dictionary_page_offset)if c.has_dictionary_page else c.data_page_offset
                if max(start,offset)<min(stop,offset+c.total_compressed_size):raise RuntimeError("metadata overlaps a body column")
        groups.append(g["row_group"]);meta_ranges.append((start,stop,f"lineage metadata rowgroup {g['row_group']}"))
    chunks=list(POOL.map(lambda x:fetch(*x),meta_ranges))
    metadata=pq.ParquetFile(ProjectedFile(size,footer_start,footer,chunks)).read_row_groups(groups,columns=META,use_threads=False)
    meta_rows=metadata.to_pylist();offset=0;target_groups=[];target_counts={}
    for gi in groups:
        n=parsed.metadata.row_group(gi).num_rows
        rows=meta_rows[offset:offset+n];offset+=n
        selected=[r for r in rows if r["domain"]in DOMAINS and (r["attack"]=="paraphrase" or(r["attack"]=="none"and r["model"]=="human"))]
        if selected:target_groups.append(gi);target_counts[gi]=len(selected)
    assert offset==len(meta_rows)
    body_ranges=[]
    for gi in target_groups:
        rg=parsed.metadata.row_group(gi)
        column=next(rg.column(ci)for ci in range(rg.num_columns)if rg.column(ci).path_in_schema=="generation")
        start=min(column.data_page_offset,column.dictionary_page_offset)if column.has_dictionary_page else column.data_page_offset
        body_ranges.append((start,start+column.total_compressed_size,f"generation rowgroup {gi}, filtered after decode"))
    bodies=list(POOL.map(lambda x:fetch(*x),body_ranges))
    table=pq.ParquetFile(ProjectedFile(size,footer_start,footer,chunks+bodies)).read_row_groups(target_groups,columns=META+["generation"],use_threads=False)
    selected=[];derived=[]
    for r in table.to_pylist():
        if r["domain"]not in DOMAINS:continue
        if not(r["attack"]=="paraphrase"or(r["attack"]=="none"and r["model"]=="human")):continue
        text=r["generation"]
        if not isinstance(text,str):raise RuntimeError("unexpected non-string generation; not silently dropped")
        y=1 if r["attack"]=="paraphrase"else 0
        details={"y_ai":y,"original_author":"human"if r["model"]=="human"else"AI",
            "original_generator":None if r["model"]=="human"else r["model"],
            "last_editor":"kalpeshk2011/dipper-paraphraser-xxl"if r["attack"]=="paraphrase"else None,
            "editor_evidence":"published RAID attack implementation default; not per-row snapshot receipt"if y else"original dataset human label",
            "mask_family":False,"mask_model":False,"word_like_tokens":len(WORD.findall(text)),
            "raw_sha256":hashlib.sha256(text.encode()).hexdigest(),
            "normalized_sha256":hashlib.sha256(" ".join(unicodedata.normalize("NFKC",text).casefold().split()).encode()).hexdigest(),
            "source_revision":ORIGINAL_REV,"converted_revision":REV,"converted_file":spec["path"]}
        selected.append({**r,**details});derived.append({**{k:r[k]for k in META},**details})
    if len(selected)!=sum(target_counts.values()):raise RuntimeError("decoded target rows differ from metadata planning")
    pq.write_table(pa.Table.from_pylist(selected),output,compression="zstd")
    pq.write_table(pa.Table.from_pylist(derived),index,compression="zstd")
    # Full screened boundary lineage is retained, with no non-target body text.
    pq.write_table(metadata,folder/"screened_lineage_metadata.parquet",compression="zstd")
    cells=collections.Counter((r["domain"],r["model"],r["attack"],r["y_ai"],r["word_like_tokens"]>=128)for r in derived)
    report={"schema_version":1,"completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope":"official converted partial-train books/news rowgroup blocks; not all original RAID train or all variants",
        "original_revision":ORIGINAL_REV,"converted_revision":REV,"source_file":spec["path"],"source_url":source_url,
        "source_lfs_sha256":spec["lfs_sha256"],"original_full_file_sha256_recomputed":False,
        "code_sha256":CODE_SHA,"footer_spec_sha256":digest(spec_path),"detector_predictions_or_manual_preview":False,
        "selected_domains":sorted(DOMAINS),"rule":"(attack=none AND model=human)->human; all attack=paraphrase including model=human->AI",
        "fine_masks_all_false":True,"screened_metadata_rowgroups":groups,"generation_rowgroups":target_groups,
        "selected_rows":len(derived),"core_ge128_rows":sum(r["word_like_tokens"]>=128 for r in derived),
        "source_ids":len({r["source_id"]for r in derived}),"range_bytes":sum(r["bytes"]for r in ledger.values()),
        "ranges":list(ledger.values()),"payload":{"path":str(output.relative_to(ROOT)),"sha256":digest(output)},
        "lineage_index":{"path":str(index.relative_to(ROOT)),"sha256":digest(index)},
        "cells":[{"domain":d,"original_model":m,"attack":a,"y_ai":y,"core_ge128":c,"rows":n}for(d,m,a,y,c),n in sorted(cells.items())],
        "unperformed_checks":["supported English guard","old-role graph exclusion","full candidate near-duplicate replay","whole-parent allocation","per-domain original content rights"]}
    write_json(receipt,report)
    print(json.dumps({"file":name,"selected_rows":len(derived),"core_ge128_rows":report["core_ge128_rows"],"range_MB":round(report["range_bytes"]/1e6,2),"receipt_sha256":digest(receipt)}),flush=True)
    return report


def main():
    RAW.mkdir(parents=True,exist_ok=True)
    started=RAW/"STARTED.json"
    if started.exists():assert json.loads(started.read_text())["code_sha256"]==CODE_SHA
    else:write_json(started,{"started_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"code_sha256":CODE_SHA,"converted_revision":REV,"generation_scope":"books/news clean human and all paraphrase only"})
    names=[f"{i:04d}"for i in range(2,10)]
    results=list(concurrent.futures.ThreadPoolExecutor(max_workers=3).map(acquire,names))
    completion={"schema_version":1,"completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"code_sha256":CODE_SHA,
        "all_8_files_complete":True,"manual_preview_or_detector_predictions":False,"range_bytes":sum(r["range_bytes"]for r in results),
        "selected_rows":sum(r["selected_rows"]for r in results),"core_ge128_rows":sum(r["core_ge128_rows"]for r in results),
        "files":[{"source":r["source_file"],"payload":r["payload"],"lineage_index":r["lineage_index"],"receipt_sha256":digest(RAW/Path(r["source_file"]).stem/"receipt.json")}for r in results]}
    write_json(RAW/"COMPLETED.json",completion)
    print(json.dumps({k:completion[k]for k in ["all_8_files_complete","range_bytes","selected_rows","core_ge128_rows"]}),flush=True)


if __name__=="__main__":main()
