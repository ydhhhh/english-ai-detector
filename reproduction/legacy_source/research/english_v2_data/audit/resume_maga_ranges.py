"""Resume the four pinned public files with four strict HTTP range workers.

Partial prefixes are immutable evidence. No row is parsed. Only a complete
file matching its predeclared official LFS SHA is renamed to the target.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
RUN = AUDIT / "maga_range_transport_v1"
BLOCK = 4 * 1024 * 1024


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    part = path.with_suffix(path.suffix + ".tmp")
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    part.replace(path)


def main():
    plan_path, handoff_path = AUDIT / "maga_acquisition_plan.json", AUDIT / "maga_transport_handoff.json"
    plan, handoff = json.loads(plan_path.read_text()), json.loads(handoff_path.read_text())
    if not handoff["writers_released"] or plan["revision"] != "34c51fc6e71561bf1203945d90d253269f9f2ab8":
        raise ValueError("exact released transport ownership and revision required")
    RUN.mkdir(exist_ok=False)
    prefixes = {}
    for item in handoff["prefixes"]:
        path = ROOT / item["path"]
        if path.stat().st_size != item["contiguous_prefix_bytes"] or sha(path) != item["prefix_sha256"]:
            raise ValueError("a released contiguous prefix changed")
        key = item["path"].removesuffix(".part").removesuffix(".retry")
        prefixes[key] = item
    chunks, files = [], []
    for spec in plan["files"]:
        if (ROOT / spec["local_path"]).exists():
            raise FileExistsError("unexpected complete target; inspect before changing transport")
        prefix = prefixes.get(spec["local_path"])
        start = prefix["contiguous_prefix_bytes"] if prefix else 0
        if not 0 <= start < spec["bytes"]:
            raise ValueError("invalid contiguous prefix size")
        directory = RUN / spec["native_path"].replace("/", "_")
        directory.mkdir()
        segments = []
        for begin in range(start, spec["bytes"], BLOCK):
            end = min(begin + BLOCK, spec["bytes"]) - 1
            segment = {"spec": spec, "begin": begin, "end": end,
                       "target": str(directory / f"{begin}_{end}.bin")}
            segments.append(segment)
        files.append({"spec": spec, "prefix": prefix, "segments": segments})
    # Interleave files so no two long streams block either validation file.
    for i in range(max(len(item["segments"]) for item in files)):
        chunks.extend(item["segments"][i] for item in files if i < len(item["segments"]))
    policy = {"started_at_utc": utc(), "plan_sha256": sha(plan_path), "handoff_sha256": sha(handoff_path),
        "workers": 4, "range_bytes": BLOCK, "segments": len(chunks), "full_files": plan["files"],
        "prefixes": handoff["prefixes"], "partial_rows_parsed": False,
        "admission": "exact official complete byte length and LFS SHA256 before rename", "source_sha256": sha(__file__)}
    write(RUN / "policy.json", policy)
    state = {"started_at_utc": utc(), "completed_segments": [], "failed_segments": [], "completed_files": [], "all_complete": False}
    started = time.monotonic()
    lock = threading.Lock()

    def fetch(segment):
        spec, begin, end = segment["spec"], segment["begin"], segment["end"]
        target = Path(segment["target"])
        wanted = end - begin + 1
        errors = []
        for attempt in range(3):
            part = target.with_suffix(f".attempt{attempt}.part")
            t = time.monotonic()
            try:
                request = urllib.request.Request(spec["url"] + f"?download=true&v2_range={begin}-{end}",
                    headers={"Range": f"bytes={begin}-{end}", "Accept-Encoding": "identity",
                             "User-Agent": "english-v2-pinned-data-transport/1.0"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected_range = f"bytes {begin}-{end}/{spec['bytes']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError("server did not honor the exact pinned byte range")
                    if response.headers.get("Content-Length") not in (None, str(wanted)):
                        raise ValueError("range Content-Length differs")
                    with part.open("xb") as output:
                        received = 0
                        while block := response.read(min(256 * 1024, wanted + 1 - received)):
                            output.write(block)
                            received += len(block)
                            if received > wanted:
                                raise ValueError("range payload exceeds requested byte count")
                if received != wanted:
                    raise ValueError("incomplete byte range")
                part.replace(target)
                result = {"file": spec["native_path"], "begin": begin, "end": end, "bytes": wanted,
                    "sha256": sha(target), "attempt": attempt + 1, "seconds": time.monotonic() - t, "previous_error_types": errors}
                with lock:
                    state["completed_segments"].append(result)
                    write(RUN / "progress.json", state)
                    print(json.dumps({"ranges_complete": len(state["completed_segments"]), "ranges_total": len(chunks),
                        "new_bytes_complete": sum(item["bytes"] for item in state["completed_segments"]),
                        "wall_seconds": round(time.monotonic() - started, 2)}), flush=True)
                return result
            except Exception as exc:
                errors.append(type(exc).__name__)
                if attempt < 2:
                    time.sleep(1 + attempt)
        raise RuntimeError(json.dumps({"file": spec["native_path"], "begin": begin, "end": end, "error_types": errors}))

    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(fetch, item): item for item in chunks}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                with lock:
                    state["failed_segments"].append(str(exc))
                    write(RUN / "progress.json", state)
    if state["failed_segments"]:
        raise RuntimeError("range acquisition incomplete; preserve all prefixes and received segments for review")
    for item in files:
        spec, prefix = item["spec"], item["prefix"]
        target = ROOT / spec["local_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        assembled = target.with_suffix(target.suffix + ".assembled.part")
        with assembled.open("xb") as output:
            if prefix:
                source = ROOT / prefix["path"]
                if source.stat().st_size != prefix["contiguous_prefix_bytes"] or sha(source) != prefix["prefix_sha256"]:
                    raise ValueError("prefix changed during transfer")
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
            for segment in item["segments"]:
                with Path(segment["target"]).open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        if assembled.stat().st_size != spec["bytes"] or sha(assembled) != spec["sha256"]:
            raise ValueError("whole pinned file SHA verification failed; do not admit")
        assembled.replace(target)
        result = {**spec, "complete": True, "observed_bytes": target.stat().st_size,
            "observed_sha256": sha(target), "completed_at_utc": utc(), "transport": "strict206ranges_and_verified_contiguous_prefix",
            "partial_rows_parsed": False}
        write(target.with_suffix(target.suffix + ".receipt.json"), result)
        state["completed_files"].append(result)
        write(RUN / "progress.json", state)
    state.update(all_complete=True, completed_at_utc=utc(), wall_seconds=time.monotonic() - started)
    write(RUN / "progress.json", state)
    write(AUDIT / "maga_range_acquisition_receipts.json", state)
    print(json.dumps({"all_complete": True, "files": len(files), "bytes": sum(item["spec"]["bytes"] for item in files), "wall_seconds": state["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
