"""Synthetic-only archive safety/reproducibility checks; no real base/data read."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from experiments.english_v2 import package_source_archive as pack


def run():
    results = []
    old = (pack.BASE_SHA, pack.ALLOW_SHA, pack.ALLOW_COUNT, pack.zip_write)
    with tempfile.TemporaryDirectory(prefix="source-archive-synthetic-") as directory:
        root = Path(directory).resolve()
        def put(name, data):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
            return path
        fixed = {"research/reference_metadata.json": b'{"rows": 12}\n', "NOTICE.txt": b'Synthetic legal notice.\n'}
        allow = pack.encode({"files": [{"path": name, "sha256": pack.sha_bytes(data)} for name, data in sorted(fixed.items())]})
        members = {**fixed, pack.ALLOW_PATH: allow, "SOURCE_MANIFEST.json": b'{"synthetic": true}\n',
            "README.md": b'Old synthetic readme.\n', "experiments/old.py": b'x = 1\n'}
        base = root/".qa/base.zip"
        base.parent.mkdir()
        with base.open("xb") as stream:
            pack.zip_write(stream, members)
        pack.BASE_SHA, pack.ALLOW_SHA, pack.ALLOW_COUNT = pack.sha(base), pack.sha_bytes(allow), len(fixed)
        body_code = put("experiments/new.py", 'text = "synthetic source string"\n')
        doc = put("README.md", "New synthetic readme.\n")
        metadata = put("research/metadata.json", '{"counts": {"human": 4}, "path": "not a body"}\n')
        def entry(source, target, operation="add", **extra):
            path = root/source
            suffix = path.suffix
            return {"operation": operation, "kind": {".py":"source", ".md":"document", ".json":"metadata", ".txt":"license_notice"}[suffix],
                "source_path": source, "archive_path": target, "bytes": path.stat().st_size,
                "sha256": pack.sha(path), "contains_corpus_text": False, **extra}
        template = {"schema_version": 1, "status": pack.STATUS, "reviewed_no_corpus_payloads": True,
            "base_data_allowlist_sha256": pack.ALLOW_SHA,
            "base_archive": {"path": ".qa/base.zip", "bytes": base.stat().st_size, "sha256": pack.BASE_SHA},
            "entries": [entry("experiments/new.py", "experiments/new.py"),
                entry("README.md", "README.md", "replace", replaces_sha256=pack.sha_bytes(members["README.md"])),
                entry("research/metadata.json", "research/reference_manifests/metadata.json")]}
        counter = 0
        def invoke(plan, *, output=None, digest=None):
            nonlocal counter
            counter += 1
            path = put(f"plans/plan{counter}.json", pack.encode(plan))
            destination = output or f"outputs/archive{counter}.zip"
            receipt = pack.build(path, digest or pack.sha(path), destination, root=root)
            return receipt, root/destination
        def good(name, check):
            check(); results.append({"name": name, "passed": True})
        def rejected(name, plan=None, setup=None, **kwargs):
            if setup: setup()
            value = deepcopy(template) if plan is None else plan
            before = set((root/"outputs").glob("*")) if (root/"outputs").exists() else set()
            try:
                invoke(value, **kwargs)
            except (ValueError, KeyError, OSError, UnicodeError):
                after = set((root/"outputs").glob("*")) if (root/"outputs").exists() else set()
                assert after == before, f"{name}: failed build left outputs"
                results.append({"name": name, "passed": True})
            else:
                raise AssertionError(name+": unexpectedly accepted")
        try:
            r1, first = invoke(template)
            r2, second = invoke(template)
            # Plan bytes differ only if their content changes; filenames do not
            # affect the ZIP. Both generated plans here are byte-identical.
            good("deterministic_archive_bytes", lambda: _assert(first.read_bytes() == second.read_bytes()))
            def preservation():
                import zipfile
                with zipfile.ZipFile(first) as archive:
                    assert archive.read(pack.PREFIX+"README.md") == doc.read_bytes()
                    assert archive.read(pack.PREFIX+"experiments/new.py") == body_code.read_bytes()
                    assert archive.read(pack.PREFIX+"research/reference_manifests/metadata.json") == metadata.read_bytes()
                    for name in [*fixed, pack.ALLOW_PATH, "SOURCE_MANIFEST.json"]:
                        assert archive.read(pack.PREFIX+name) == members[name]
                    manifest = json.loads(archive.read(pack.PREFIX+pack.MANIFEST))
                    for row in manifest["files_excluding_this_manifest"]:
                        assert pack.sha_bytes(archive.read(row["path"])) == row["sha256"]
                    assert archive.namelist() == sorted(archive.namelist())
                assert r1["inputs_unchanged_before_after"] and r1["original103_entries_preserved"]
            good("add_explicit_replace_remap_and_all_protected_bytes", preservation)
            rejected("exclusive_output_does_not_overwrite", output=str(first))
            p=deepcopy(template); p["status"]="DRAFT"; rejected("draft_plan_refused", p)
            rejected("incorrect_independent_plan_sha", digest="0"*64)
            p=deepcopy(template); p["base_archive"]["sha256"]="0"*64; rejected("base_identity_pinned", p)
            p=deepcopy(template); p["entries"][0]["archive_path"]="../escape.py"; rejected("archive_traversal", p)
            p=deepcopy(template); p["entries"][0]["archive_path"]=pack.PREFIX+"experiments/new.py"; rejected("second_zip_root_prefix_refused", p)
            p=deepcopy(template); p["entries"][0]["source_path"]="../escape.py"; rejected("source_traversal", p)
            p=deepcopy(template); p["entries"][0]["source_path"]="/tmp/escape.py"; rejected("outside_project", p)
            p=deepcopy(template); p["entries"][0]["archive_path"]="raw/new.py"; rejected("raw_archive_path", p)
            put("raw/new.py", b'x=1\n'); p=deepcopy(template); p["entries"][0]=entry("raw/new.py", "experiments/raw_new.py"); rejected("raw_source_path", p)
            p=deepcopy(template); p["entries"].append(deepcopy(p["entries"][0])); rejected("duplicate_archive_target", p)
            p=deepcopy(template); repeated=deepcopy(p["entries"][0]); repeated["archive_path"]="experiments/copy.py"; p["entries"].append(repeated); rejected("duplicate_source_path", p)
            p=deepcopy(template); p["entries"][0]["archive_path"]="readme.md"; rejected("casefold_collision", p)
            p=deepcopy(template); p["entries"][1]["operation"]="add"; p["entries"][1].pop("replaces_sha256"); rejected("unintentional_cover", p)
            p=deepcopy(template); p["entries"][1]["replaces_sha256"]="0"*64; rejected("replace_old_sha_required", p)
            p=deepcopy(template); p["entries"][2]["archive_path"]="research/reference_metadata.json"; p["entries"][2]["operation"]="replace"; p["entries"][2]["replaces_sha256"]=pack.sha_bytes(fixed["research/reference_metadata.json"]); rejected("original_allowlisted_metadata_immutable", p)
            p=deepcopy(template); p["entries"][2]["archive_path"]=pack.ALLOW_PATH; rejected("original_allowlist_immutable", p)
            p=deepcopy(template); p["entries"][2]["archive_path"]="SOURCE_MANIFEST.json"; rejected("original_manifest_immutable", p)
            p=deepcopy(template); p["entries"][2]["archive_path"]=pack.MANIFEST; rejected("generated_manifest_reserved", p)
            p=deepcopy(template); p["entries"][2]["archive_path"]=pack.MANIFEST.lower(); rejected("generated_manifest_casefold_reserved", p)
            p=deepcopy(template); p["entries"][2]["archive_path"]=pack.PLAN_MEMBER.lower(); rejected("embedded_plan_casefold_reserved", p)
            p=deepcopy(template); p["entries"][0]["archive_path"]="model.safetensors"; rejected("model_binary_type", p)
            put("research/nonutf8.json", b'\xff'); p=deepcopy(template); p["entries"][2]=entry("research/nonutf8.json", "research/nonutf8.json"); rejected("non_utf8_payload", p)
            put("research/body.json", '{"rows": [{"text": "Synthetic corpus record"}]}'); p=deepcopy(template); p["entries"][2]=entry("research/body.json", "research/body.json"); rejected("corpus_json_structure", p)
            put("research/list.json", '[{"rows": 1}]'); p=deepcopy(template); p["entries"][2]=entry("research/list.json", "research/list.json"); rejected("top_level_record_list", p)
            put("research/duplicate.json", '{"rows": 1, "rows": 2}'); p=deepcopy(template); p["entries"][2]=entry("research/duplicate.json", "research/duplicate.json"); rejected("duplicate_json_keys", p)
            (root/"link").symlink_to(root/"experiments", target_is_directory=True)
            p=deepcopy(template); p["entries"][0]["source_path"]="link/new.py"; rejected("symlink_source_even_inside_project", p)
            rejected("symlink_output_parent", output="link/output.zip")
            p=deepcopy(template); p["entries"][0]["sha256"]="0"*64; rejected("source_sha_required", p)
            p=deepcopy(template); p["entries"][0]["bytes"]+=1; rejected("source_size_required", p)
            def mutate_after_write(stream, content):
                old[3](stream, content)
                body_code.write_bytes(b'x = 999\n')
            pack.zip_write=mutate_after_write
            rejected("concurrent_input_change_removes_own_outputs")
            pack.zip_write=old[3]; body_code.write_bytes(b'text = "synthetic source string"\n')
            def fail_write(stream, content):
                stream.write(b'partial')
                raise OSError("synthetic write failure")
            pack.zip_write=fail_write
            rejected("partial_write_failure_removes_own_outputs")
            pack.zip_write=old[3]
            assert pack.sha(base) == pack.BASE_SHA and pack.sha_bytes(allow) == pack.ALLOW_SHA
        finally:
            pack.BASE_SHA, pack.ALLOW_SHA, pack.ALLOW_COUNT, pack.zip_write = old
    return {"schema_version":1, "status":"PASS", "synthetic_only":True, "real_base_or_corpus_read":False,
        "model_or_GPU_used":False, "checks":results, "passed":len(results), "failed":0,
        "package_source_archive_sha256":pack.sha(pack.__file__), "checks_source_sha256":pack.sha(__file__)}


def _assert(value):
    assert value


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    result=run()
    output=Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False); stream.write("\n")
    print(json.dumps({"status":result["status"], "passed":result["passed"], "synthetic_only":True}))
