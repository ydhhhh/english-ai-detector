"""Hash-bound, bounded extraction of the complete public detector bundle.

Only Python's standard library is used. No model code is imported here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

MAX_ARCHIVE_MEMBERS = 64
MAX_ARCHIVE_BYTES = 900_000_000
MAX_MEMBER_BYTES = 650_000_000
MAX_MANIFEST_BYTES = 1_000_000


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_member(info):
    name = info.filename
    _require(isinstance(name, str) and info.orig_filename == name and "\\" not in name and ":" not in name and
        not any(ord(c) < 32 for c in name), "unsafe archive path")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and path.as_posix() == name and
        all(p not in {".", ".."} for p in path.parts) and
        len(path.parts) >= 2 and path.parts[0] == "bundle", "unsafe archive root/path")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    _require(not info.is_dir() and kind in {0, stat.S_IFREG}, "archive member must be a regular file")
    _require(not info.flag_bits & 1 and info.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED},
        "encrypted or unsupported archive compression")
    _require(0 <= info.file_size <= MAX_MEMBER_BYTES, "archive member size exceeds limit")
    return name


def _verify_tree(root, expected, manifest_sha):
    _require(root.is_dir() and not root.is_symlink(), "invalid extracted bundle cache")
    observed = set()
    for p in root.rglob("*"):
        _require(not p.is_symlink(), "symlink in extracted bundle cache")
        if p.is_file():
            observed.add(p.relative_to(root).as_posix())
    _require(observed == set(expected), "extracted bundle file set differs")
    _require(_sha(root / "bundle.json") == manifest_sha, "extracted manifest changed")
    _require((root / "bundle.sha256").read_text().split() == [manifest_sha, "bundle.json"],
        "extracted checksum declaration differs")
    for name, row in expected.items():
        p = root / name
        _require(p.is_file() and p.stat().st_size == row["bytes"] and _sha(p) == row["sha256"],
            "extracted bundle member changed")


def extract_bundle_archive(archive, *, expected_archive_sha256, expected_manifest_sha256,
        member_names=None, cache_directory=None):
    """Verify ZIP first, extract the exact inventory, then return its bundle/ root.

    Optional ``member_names(manifest)`` adds the caller's frozen schema validator.
    The cache is content-addressed and published only after full verification.
    """
    for value in (expected_archive_sha256, expected_manifest_sha256):
        _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "approved archive and manifest hashes must be configured")
    archive = Path(archive)
    _require(archive.is_file() and _sha(archive) == expected_archive_sha256,
        "bundle archive SHA256 differs from the approved artifact")
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        _require(0 < len(infos) <= MAX_ARCHIVE_MEMBERS, "archive member count exceeds limit")
        names = [_safe_member(info) for info in infos]
        _require(len({name.casefold() for name in names}) == len(names), "duplicate archive member")
        _require(sum(info.file_size for info in infos) <= MAX_ARCHIVE_BYTES, "archive size exceeds limit")
        by_name = dict(zip(names, infos))
        manifest_info = by_name.get("bundle/bundle.json")
        _require(manifest_info is not None and manifest_info.file_size <= MAX_MANIFEST_BYTES,
            "missing or oversized archive manifest")
        data = zf.read(manifest_info)
        _require(hashlib.sha256(data).hexdigest() == expected_manifest_sha256,
            "archive manifest is not the approved bundle")
        manifest = json.loads(data.decode("utf-8"))
        inventory = manifest.get("integrity")
        _require(isinstance(inventory, list), "missing manifest inventory")
        expected_names = ["bundle.json", "bundle.sha256"]
        for row in inventory:
            name = row["path"]
            _safe_member(zipfile.ZipInfo("bundle/"+name))
            _require(isinstance(row["bytes"], int) and not isinstance(row["bytes"], bool) and
                0 <= row["bytes"] <= MAX_MEMBER_BYTES and isinstance(row["sha256"], str) and
                re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "invalid manifest inventory member")
            expected_names.append(name)
        _require(len(set(name.casefold() for name in expected_names)) == len(expected_names),
            "duplicate manifest inventory member")
        if member_names is not None:
            _require(set(member_names(manifest)) == set(expected_names), "caller manifest schema differs")
        _require(set(names) == {"bundle/"+name for name in expected_names},
            "archive must contain exactly the complete approved bundle")
        expected = {row["path"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
            for row in manifest["integrity"]}
        expected["bundle.json"] = {"bytes": len(data), "sha256": expected_manifest_sha256}
        _require(by_name["bundle/bundle.sha256"].file_size <= 256, "oversized checksum declaration")
        checksum_data = zf.read(by_name["bundle/bundle.sha256"])
        _require(checksum_data.decode("utf-8").split() == [expected_manifest_sha256, "bundle.json"],
            "archive checksum declaration differs")
        expected["bundle.sha256"] = {"bytes": len(checksum_data),
            "sha256": hashlib.sha256(checksum_data).hexdigest()}
        for name, row in expected.items():
            _require(by_name["bundle/"+name].file_size == row["bytes"], "archive inventory size differs")

        base = Path(cache_directory) if cache_directory is not None else Path(tempfile.gettempdir()) / "english_detector_bundles"
        base.mkdir(parents=True, exist_ok=True)
        target = base / expected_archive_sha256
        _require(not target.is_symlink(), "invalid bundle cache link")
        if target.exists():
            _verify_tree(target / "bundle", expected, expected_manifest_sha256)
            return target / "bundle"
        staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=base))
        try:
            for name, row in expected.items():
                output = staging / "bundle" / name
                output.parent.mkdir(parents=True, exist_ok=True)
                digest, count = hashlib.sha256(), 0
                with zf.open(by_name["bundle/"+name]) as source, output.open("xb") as destination:
                    while block := source.read(1024*1024):
                        count += len(block)
                        _require(count <= row["bytes"], "archive member expanded beyond declared size")
                        destination.write(block); digest.update(block)
                _require(count == row["bytes"] and digest.hexdigest() == row["sha256"],
                    "archive member bytes differ from frozen inventory")
            _verify_tree(staging / "bundle", expected, expected_manifest_sha256)
            # This app serializes startup; publish a fully verified tree atomically.
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return target / "bundle"
