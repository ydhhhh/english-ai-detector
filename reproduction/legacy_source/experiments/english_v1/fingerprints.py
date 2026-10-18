"""Portable inference-configuration identities, separate from tensor hashes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()


def metadata_fingerprint(kind,path):
    path=Path(path)
    if kind=="statistical":return sha(path)
    if kind!="neural":raise ValueError("unsupported component fingerprint type")
    files=[path/"detector.json"]
    for directory in ("encoder_config","tokenizer"):
        files.extend(p for p in (path/directory).rglob("*") if p.is_file())
    if not all(p.is_file() for p in files) or len(files)<3:raise ValueError("incomplete inference configuration")
    manifest={str(p.relative_to(path)):sha(p) for p in sorted(files)}
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
