"""Project only RAID native lineage/attack columns from candidate row groups.

No title, prompt, or generation bytes are requested. Row groups are screened by
the official Parquet footer's attack statistics, then validated on actual values.
This is a partial upstream metadata index, not a role allocation or core audit.
"""
import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT / "research/data_audit_20261017/vendor_fast"))
import pyarrow as pa
import pyarrow.parquet as pq
from audit_parquet_footers import FooterFile

COLUMNS = ["id", "adv_source_id", "source_id", "model", "decoding", "repetition_penalty", "attack", "domain"]


def project(name, attacks):
    meta_path = BASE / (name + ".parquet.footer_metadata.json")
    spec = json.loads(meta_path.read_text())
    size, url, ranges = spec["size"], spec["url"], []
    def fetch(start, stop):
        request = urllib.request.Request(url + f"?metadata_projection_range={start}-{stop-1}",
            headers={"Range": f"bytes={start}-{stop-1}", "User-Agent": "metadata-only-audit/1.0"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop-1}/{size}":
                        raise RuntimeError("server did not honor exact Range; body unread")
                    data = response.read(stop-start+1)
                if len(data) != stop-start: raise RuntimeError("range size mismatch")
                return start, data
            except Exception:
                if attempt == 2: raise
    footer_start = spec["footer_start"]
    _, footer = fetch(footer_start, size)
    parsed = pq.ParquetFile(FooterFile(size, footer_start, footer))
    selected = []
    for group in spec["row_groups"]:
        a = group["metadata_columns"]["attack"]
        if any(a.get("minimum", "") <= attack <= a.get("maximum", "\U0010ffff") for attack in attacks):
            selected.append(group["row_group"])
            permitted = [group["metadata_columns"][key] for key in COLUMNS]
            start, stop = min(x["start"] for x in permitted), max(x["stop_exclusive"] for x in permitted)
            # Verify the combined request cannot include even a slice of any body column.
            rg = parsed.metadata.row_group(group["row_group"])
            for c in range(rg.num_columns):
                col = rg.column(c)
                if col.path_in_schema not in COLUMNS:
                    offset = min(col.data_page_offset, col.dictionary_page_offset) if col.has_dictionary_page else col.data_page_offset
                    if max(start, offset) < min(stop, offset + col.total_compressed_size):
                        raise RuntimeError("metadata span overlaps a body column")
            ranges.append((start, stop))
    chunks = list(concurrent.futures.ThreadPoolExecutor(max_workers=12).map(lambda r: fetch(*r), ranges))
    class ProjectionFile(FooterFile):
        def read(self, n=-1):
            start = self.pos
            data = bytearray(super().read(n))
            stop = start + len(data)
            for offset, raw in chunks:
                a, b = max(offset, start), min(offset + len(raw), stop)
                if a < b: data[a-start:b-start] = raw[a-offset:b-offset]
            return bytes(data)
    table = pq.ParquetFile(ProjectionFile(size, footer_start, footer)).read_row_groups(selected, columns=COLUMNS, use_threads=False)
    # Keep boundary rows as metadata bridges; they are never body-inspected here.
    destination = BASE / (name + ".lineage_metadata.parquet")
    pq.write_table(table, destination)
    rows = table.to_pylist()
    cells = collections.Counter((r["domain"], r["attack"], r["model"]) for r in rows)
    para = [r for r in rows if r["attack"] == "paraphrase"]
    humans = [r for r in rows if r["model"] == "human" and r["attack"] == "none"]
    summary = {"source": spec["path"], "source_lfs_sha256": spec["lfs_sha256"], "converted_revision": "5008b2d167f74bfd4ceff6b3b0512b7097d7cbb1",
        "metadata_only": True, "body_columns_requested": [], "permitted_columns": COLUMNS,
        "selected_row_groups": selected, "row_groups_selected_by": sorted(attacks),
        "rows": len(rows), "paraphrase_rows": len(para), "paraphrase_model_human_rows": sum(r["model"] == "human" for r in para),
        "paraphrase_distinct_native_source_ids": len({r["source_id"] for r in para}),
        "clean_human_rows": len(humans), "clean_human_distinct_source_ids": len({r["source_id"] for r in humans}),
        "cells": [{"domain": d, "attack": a, "model": m, "rows": n} for (d,a,m),n in sorted(cells.items())],
        "ranges": [{"start": start, "stop_exclusive": start+len(raw), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for start,raw in chunks],
        "footer_bytes": len(footer), "metadata_column_bytes": sum(len(raw) for _,raw in chunks),
        "output": {"path": str(destination.relative_to(ROOT)), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()},
        "footer_spec_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "core_length_or_body_hashes_computed": False}
    (BASE / (name + ".lineage_metadata_summary.json")).write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ["source","rows","paraphrase_rows","paraphrase_model_human_rows","paraphrase_distinct_native_source_ids","clean_human_rows","clean_human_distinct_source_ids","metadata_column_bytes","output"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--attacks", nargs="+", default=["paraphrase","none"])
    args=parser.parse_args()
    for name in args.files: project(name, set(args.attacks))
