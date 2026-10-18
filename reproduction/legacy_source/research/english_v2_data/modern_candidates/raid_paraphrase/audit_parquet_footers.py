"""Read only the pinned official RAID train Parquet footers, never body columns."""
import concurrent.futures
import datetime
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import urllib.request

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT / "research/data_audit_20261017/vendor_fast"))
import pyarrow.parquet as pq


class FooterFile(io.RawIOBase):
    def __init__(self, size, footer_start, footer):
        self.size, self.start, self.footer, self.pos = size, footer_start, footer, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos
    def read(self, n=-1):
        stop = self.size if n < 0 else min(self.size, self.pos + n)
        data = bytearray(stop - self.pos)
        for start, raw in [(0, b"PAR1"), (self.start, self.footer)]:
            a, b = max(start, self.pos), min(start + len(raw), stop)
            if a < b: data[a-self.pos:b-self.pos] = raw[a-start:b-start]
        self.pos = stop
        return bytes(data)


def audit(file):
    path, size = file["rfilename"], file["size"]
    url = f"https://huggingface.co/datasets/liamdugan/raid/resolve/5008b2d167f74bfd4ceff6b3b0512b7097d7cbb1/{path}"
    requests = []
    def fetch(start, stop):
        q = urllib.request.Request(url + f"?footer_metadata_range={start}-{stop-1}",
            headers={"Range": f"bytes={start}-{stop-1}", "User-Agent": "metadata-only-audit/1.0"})
        with urllib.request.urlopen(q, timeout=60) as r:
            if r.status != 206 or r.headers.get("Content-Range") != f"bytes {start}-{stop-1}/{size}":
                raise RuntimeError("exact range unavailable; response body not read")
            raw = r.read(stop-start+1)
        if len(raw) != stop-start: raise RuntimeError("partial response mismatch")
        requests.append({"start": start, "stop_exclusive": stop, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        return raw
    tail = fetch(size-8, size)
    assert tail[4:] == b"PAR1"
    length = struct.unpack("<I", tail[:4])[0]
    assert 0 < length < 8_000_000
    start = size - length - 8
    footer = fetch(start, size-8) + tail
    meta = pq.ParquetFile(FooterFile(size, start, footer)).metadata
    groups = []
    for i in range(meta.num_row_groups):
        rg = meta.row_group(i)
        columns = {}
        for j in range(rg.num_columns):
            col = rg.column(j)
            if col.path_in_schema not in {"id", "source_id", "adv_source_id", "model", "attack", "domain", "decoding", "repetition_penalty"}:
                continue
            offset = min(col.data_page_offset, col.dictionary_page_offset) if col.has_dictionary_page else col.data_page_offset
            record = {"start": offset, "stop_exclusive": offset + col.total_compressed_size,
                      "compressed_size": col.total_compressed_size}
            if col.path_in_schema in {"model", "attack", "domain"} and col.statistics and col.statistics.has_min_max:
                record["minimum"], record["maximum"] = col.statistics.min, col.statistics.max
            columns[col.path_in_schema] = record
        groups.append({"row_group": i, "rows": rg.num_rows, "metadata_columns": columns})
    result = {"path": path, "size": size, "url": url, "lfs_sha256": file.get("lfs", {}).get("sha256"),
              "footer_requests": requests, "footer_start": start, "rows": meta.num_rows,
              "row_groups": groups, "body_columns_read": []}
    out = BASE / (Path(path).name + ".footer_metadata.json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"path": path, "rows": meta.num_rows, "row_groups": meta.num_row_groups,
        "domain_ranges": sorted({(g["metadata_columns"]["domain"].get("minimum"), g["metadata_columns"]["domain"].get("maximum")) for g in groups}),
        "attack_ranges": sorted({(g["metadata_columns"]["attack"].get("minimum"), g["metadata_columns"]["attack"].get("maximum")) for g in groups})}), flush=True)
    return {"path": str(out.relative_to(ROOT)), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "source": path, "rows": meta.num_rows, "footer_bytes_read": len(footer), "size": size}


def main():
    api = json.loads((BASE / "convert_revision_api.json").read_text())
    assert api["sha"] == "5008b2d167f74bfd4ceff6b3b0512b7097d7cbb1"
    files = [f for f in api["siblings"] if f["rfilename"].startswith("raid/partial-train/") and f["rfilename"].endswith(".parquet")]
    results = list(concurrent.futures.ThreadPoolExecutor(max_workers=4).map(audit, files))
    out = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "partial_dataset": True,
           "revision": api["sha"], "file_count": len(results), "body_columns_read": [],
           "footer_bytes_read": sum(x["footer_bytes_read"] for x in results), "files": results,
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (BASE / "footer_audit_summary.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__": main()
