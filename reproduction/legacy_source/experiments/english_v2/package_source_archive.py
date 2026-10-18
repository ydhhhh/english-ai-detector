"""Deterministic additive source archive over the immutable curated v1 ZIP.

No directory scan or model/data build occurs. Only an exact-SHA reviewed plan
can add/explicitly replace UTF8 source/docs/metadata. Original103 data entries,
their allowlist and the legacy source manifest remain byte-identical.

CLI: python -B -m experiments.english_v2.package_source_archive \
  --plan REVIEWED_JSON --sha256 EXACT_PLAN_SHA --output NEW_SOURCE_ZIP

Plan schema1: status=FROZEN_REVIEWED_SOURCE_ARCHIVE_INCREMENT_PLAN,
reviewed_no_corpus_payloads=true, base_archive:{path,bytes,sha256},
base_data_allowlist_sha256, entries:[{operation:add|replace, kind, source_path,
archive_path, bytes, sha256, contains_corpus_text:false, replaces_sha256?}].
archive_path is project-relative under the fixed english-ai-detector ZIP root.
Replace requires an existing unprotected base member and exact old SHA.
Source/archive paths are canonical project-relative paths, never symlinks.
The trusted base ZIP alone may be read from .qa. A DRAFT cannot execute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import unicodedata
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
BASE_SHA = "d5bc38eb31171c7f6ce85f1cb8e3f6a948c34c481e576ef6ede8ade160f4b524"
ALLOW_SHA = "fc4bf4ccbb031a6c185b75286879ad2e77cc547b8ac4f0f5bc8864a63512d068"
ALLOW_COUNT = 103
PREFIX = "english-ai-detector/"
ALLOW_PATH = "research/english_v1_data/source_release_allowlist.json"
MANIFEST = "SOURCE_MANIFEST_V2.json"
PLAN_MEMBER = "SOURCE_INCREMENT_PLAN.json"
STATUS = "FROZEN_REVIEWED_SOURCE_ARCHIVE_INCREMENT_PLAN"
FORBIDDEN_DIRS = {"raw", "files", "vendor", "vendor_fast", "runs", ".venv", ".runtime", ".cache", "cache", ".qa",
                  "__pycache__", ".git", ".aws", ".ssh", ".codex"}
KINDS = {".py": {"source"}, ".html": {"source"}, ".js": {"source"}, ".css": {"source"}, ".md": {"document"},
         ".json": {"metadata"}, ".txt": {"license_notice", "dependency_specification"}}
BODY_KEYS = {"text", "full_text", "article_text", "prompt_text", "generated_text", "human_answers", "chatgpt_answers",
             "prompt", "question", "answer", "completion", "response_text"}
MAX_MEMBER = 32*1024*1024
MAX_TOTAL = 256*1024*1024


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def canonical(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value and
        not any(ord(c) < 32 for c in value) and unicodedata.normalize("NFC", value) == value,
        "canonical UTF8 relative path required")
    p = PurePosixPath(value)
    require(not p.is_absolute() and p.as_posix() == value and all(part not in {".", ".."} for part in p.parts),
        "path traversal or noncanonical path")
    return p


def project_path(value, root, *, base_exception=False):
    root = Path(root).resolve()
    p = Path(value)
    if p.is_absolute():
        try:
            value = p.relative_to(root).as_posix()
        except ValueError:
            raise ValueError("source/output must remain inside the project") from None
    else:
        value = str(value)
    relative = canonical(value)
    if not base_exception:
        require(not any(part.casefold() in FORBIDDEN_DIRS for part in relative.parts), "forbidden artifact directory")
    current = root
    for part in relative.parts:
        current /= part
        require(not current.is_symlink(), "symlink source/output escape is forbidden")
    require(current.resolve().is_relative_to(root), "source/output escaped the project")
    return current


def archive_path(value):
    p = canonical(value)
    require(p.parts[0].casefold() != PREFIX.rstrip("/").casefold(), "archive paths must omit the fixed ZIP root prefix")
    require(not any(part.casefold() in FORBIDDEN_DIRS for part in p.parts), "forbidden archive directory")
    return p.as_posix()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def json_value(data):
    text = data.decode("utf-8", errors="strict")
    return json.loads(text, object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")


def reject_record_bodies(value):
    """Structural JSON-record check, never a keyword scan of source/doc prose.

    Human review of the exact plan still declares content eligibility; this is
    a further guard against common JSON corpus/preview payloads misfiled as
    metadata. Source .py strings/variable names are never inspected here.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in BODY_KEYS:
                require(not isinstance(item, str) or not item.strip(), "JSON corpus body/answer/prompt is not release metadata")
                require(not isinstance(item, list) or not any(isinstance(v, str) and v.strip() for v in item),
                    "JSON answer/text array is not release metadata")
            reject_record_bodies(item)
    elif isinstance(value, list):
        for item in value:
            reject_record_bodies(item)


def read_base(path):
    members = {}
    folded = set()
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, "base ZIP CRC failure")
        for info in archive.infolist():
            require(not info.is_dir() and info.filename.startswith(PREFIX) and not info.flag_bits & 1,
                "only unencrypted regular files under the fixed archive root are accepted")
            mode = stat.S_IFMT(info.external_attr >> 16)
            require(mode in (0, stat.S_IFREG), "base ZIP symlink/special member is forbidden")
            name = archive_path(info.filename[len(PREFIX):])
            require(name not in members and name.casefold() not in folded and info.file_size <= MAX_MEMBER,
                "duplicate/colliding or oversized base member")
            data = archive.read(info)
            require(len(data) == info.file_size, "base member size mismatch")
            data.decode("utf-8", errors="strict")
            members[name] = data; folded.add(name.casefold())
            require(sum(map(len, members.values())) <= MAX_TOTAL, "source archive is oversized")
    require(ALLOW_PATH in members and sha_bytes(members[ALLOW_PATH]) == ALLOW_SHA, "original103 allowlist differs")
    allow = json_value(members[ALLOW_PATH])
    require(len(allow["files"]) == ALLOW_COUNT and len({r["path"] for r in allow["files"]}) == ALLOW_COUNT,
        "original data allowlist count/identity differs")
    protected = {ALLOW_PATH, "SOURCE_MANIFEST.json"}
    for row in allow["files"]:
        name = archive_path(row["path"])
        require(name in members and sha_bytes(members[name]) == row["sha256"], "original data entry differs inside the trusted base")
        protected.add(name)
    require("SOURCE_MANIFEST.json" in members and MANIFEST not in members and PLAN_MEMBER not in members,
        "base must be the original source ZIP, not a previous incremental archive")
    return members, protected


def zip_write(stream, members):
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(PREFIX+name, date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.create_system = 3
            archive.writestr(info, data, compresslevel=9)


def build(plan_path, expected_plan_sha, output, *, root=PROJECT):
    root = Path(root).resolve()
    plan_path = project_path(plan_path, root)
    require(is_sha(expected_plan_sha) and sha(plan_path) == expected_plan_sha, "explicit reviewed-plan SHA mismatch")
    plan_data = plan_path.read_bytes()
    require(sha_bytes(plan_data) == expected_plan_sha, "plan changed during read")
    plan = json_value(plan_data)
    require(isinstance(plan, dict) and plan.get("schema_version") == 1 and plan.get("status") == STATUS and
        plan.get("reviewed_no_corpus_payloads") is True and plan.get("base_data_allowlist_sha256") == ALLOW_SHA,
        "a frozen reviewed additive plan is required; DRAFT cannot execute")
    reject_record_bodies(plan)
    base_spec = plan["base_archive"]
    require(base_spec["sha256"] == BASE_SHA and is_sha(BASE_SHA), "only the pinned original v1 source ZIP may be used as base")
    base = project_path(base_spec["path"], root, base_exception=True)
    require(base.is_file() and base.stat().st_size == base_spec["bytes"] and sha(base) == BASE_SHA, "base ZIP size/SHA mismatch")
    output = project_path(output, root)
    require(output.suffix == ".zip", "new output ZIP required")
    receipt_path = output.with_suffix(".receipt.json")
    require(not output.exists() and not receipt_path.exists() and output not in {base, plan_path} and receipt_path not in {base, plan_path},
        "ZIP/receipt outputs must be new, distinct and exclusive")
    members, protected = read_base(base)
    original_hashes = {name: sha_bytes(data) for name, data in members.items()}
    require(isinstance(plan.get("entries"), list) and plan["entries"], "explicit nonempty increment entries required")
    seen, seen_sources, folded = set(), set(), {name.casefold(): name for name in members}
    inputs = {str(plan_path): expected_plan_sha, str(base): BASE_SHA}
    code_sha = sha(__file__)
    operations = []
    for entry in plan["entries"]:
        name = archive_path(entry["archive_path"])
        require(name not in seen and name.casefold() not in {MANIFEST.casefold(), PLAN_MEMBER.casefold()} and name not in protected,
            "duplicate/reserved/original103 archive target cannot be changed")
        require(name.casefold() not in folded or folded[name.casefold()] == name, "case-fold archive collision")
        seen.add(name); folded[name.casefold()] = name
        kind = entry.get("kind")
        suffix = PurePosixPath(name).suffix.casefold()
        require(suffix in KINDS and kind in KINDS[suffix] and entry.get("contains_corpus_text") is False,
            "only explicitly reviewed UTF8 source/docs/metadata/notices are allowed")
        source = project_path(entry["source_path"], root)
        require(source.suffix.casefold() == suffix and source.is_file() and source not in {output, receipt_path} and source not in seen_sources,
            "source/archive types differ, source duplicated, or source is not a file")
        seen_sources.add(source)
        require(isinstance(entry["bytes"], int) and not isinstance(entry["bytes"], bool) and 0 <= entry["bytes"] <= MAX_MEMBER and
            source.stat().st_size == entry["bytes"] and is_sha(entry["sha256"]) and sha(source) == entry["sha256"], "increment source size/SHA mismatch")
        data = source.read_bytes(); data.decode("utf-8", errors="strict")
        require(len(data) == entry["bytes"] and sha_bytes(data) == entry["sha256"], "increment source changed during read")
        if kind == "metadata":
            value = json_value(data); require(isinstance(value, dict), "metadata JSON must be an object, not a corpus list")
            reject_record_bodies(value)
        operation = entry.get("operation")
        if operation == "add":
            require(name not in members and "replaces_sha256" not in entry, "add cannot cover an existing member")
        elif operation == "replace":
            require(name in original_hashes and entry.get("replaces_sha256") == original_hashes[name], "replacement requires the exact existing base-member SHA")
        else:
            raise ValueError("explicit add or intentional replace operation required")
        members[name] = data; inputs[str(source)] = entry["sha256"]
        operations.append({"operation": operation, "source_path": source.relative_to(root).as_posix(), "archive_path": name,
            "kind": kind, "bytes": len(data), "sha256": entry["sha256"], **({"replaces_sha256": entry["replaces_sha256"]} if operation == "replace" else {})})
    require(all(sha(p) == digest for p, digest in inputs.items()), "an input changed before archive creation")
    members[PLAN_MEMBER] = plan_data
    member_rows = [{"path": PREFIX+name, "bytes": len(data), "sha256": sha_bytes(data),
        "origin": "reviewed_increment" if name in seen else "exact_plan" if name == PLAN_MEMBER else "trusted_original_v1_base"}
        for name, data in sorted(members.items())]
    manifest = {"schema_version": 1, "scope": "Curated UTF8 project source/docs/reviewed metadata; no corpus, models or caches",
        "base_archive_sha256": BASE_SHA, "base_data_allowlist_sha256": ALLOW_SHA, "original103_entries_preserved": True,
        "increment_plan_sha256": expected_plan_sha, "increment_plan_member": PREFIX+PLAN_MEMBER,
        "legacy_manifest": "SOURCE_MANIFEST.json remains the byte-identical historical base manifest; this v2 manifest describes the current archive",
        "self_hash_note": "This manifest cannot include its own hash; its exact bytes and the final archive SHA are in the external receipt.",
        "operations": operations, "files_excluding_this_manifest": member_rows, "training_texts_included": False, "model_weights_included": False}
    members[MANIFEST] = encode(manifest)
    require(len({name.casefold() for name in members}) == len(members), "final archive has a case-fold member collision")
    require(sum(map(len, members.values())) <= MAX_TOTAL, "source archive is oversized")
    created_zip = created_receipt = False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            created_zip = True
            zip_write(stream, members)
        with zipfile.ZipFile(output) as archive:
            require(archive.testzip() is None and archive.namelist() == [PREFIX+n for n in sorted(members)], "deterministic sorted ZIP member/CRC mismatch")
            require(all(sha_bytes(archive.read(PREFIX+n)) == sha_bytes(data) for n, data in members.items()), "written member hash differs")
        require(all(sha(p) == digest for p, digest in inputs.items()) and sha(__file__) == code_sha, "an input/execution source changed while archiving")
        require(all(sha_bytes(members[name]) == original_hashes[name] for name in protected), "protected original data entry changed")
        receipt = {"schema_version": 1, "archive": output.relative_to(root).as_posix(), "archive_sha256": sha(output), "archive_bytes": output.stat().st_size,
            "member_count": len(members), "manifest_member": PREFIX+MANIFEST, "manifest_sha256": sha_bytes(members[MANIFEST]),
            "reviewed_plan_sha256": expected_plan_sha, "base_archive_sha256": BASE_SHA, "base_data_allowlist_sha256": ALLOW_SHA,
            "inputs_unchanged_before_after": True, "original103_entries_preserved": True,
            "source_inputs_sha256": {str(Path(p).relative_to(root)): digest for p, digest in inputs.items()},
            "archive_code_sha256": code_sha, "model_weights_included": False, "training_texts_included": False}
        with receipt_path.open("x", encoding="utf-8") as stream:
            created_receipt = True
            stream.write(encode(receipt).decode("utf-8"))
        return receipt
    except BaseException:
        # Only our exclusively created outputs are removed. No base, source,
        # original allowlist or existing output is ever overwritten/deleted.
        if created_receipt:
            receipt_path.unlink(missing_ok=True)
        if created_zip:
            output.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = build(args.plan, args.sha256, args.output)
    print(json.dumps({"archive": receipt["archive"], "archive_sha256": receipt["archive_sha256"], "member_count": receipt["member_count"],
        "inputs_unchanged_before_after": receipt["inputs_unchanged_before_after"]}))
