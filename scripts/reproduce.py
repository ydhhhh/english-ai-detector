#!/usr/bin/env python3
"""Check reproduction source or explicitly dispatch a prepared external project.

This standard-library launcher never downloads data, loads model weights, or
trains a model during check/show/help. Historical receipts are not new run seals.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
INVENTORY = REPOSITORY / "reproduction/source_manifest.json"
BUNDLED_SOURCE = REPOSITORY / "reproduction/legacy_source"
STAGES = {
    "prepare-data": "experiments.english_v2.prepare_registered_data",
    "prepare-init": "experiments.english_v2.prepare_binary_init",
    "train": "experiments.english_v2.train_registered",
    "export": "experiments.english_v1.export_component",
    "export-release": "experiments.english_v1.run_release_exports",
    "develop": "experiments.english_v2.run_development",
    "select": "experiments.english_v2.select_binary",
    "finalize": "experiments.english_v2.binary_finalize",
    "evaluate-compatible": "experiments.english_v2.count_storage_compat",
    "acceptance": "experiments.english_v2.acceptance",
}


def check(project_root):
    """Read code, hashes and paths only; never import or execute legacy modules."""
    manifest = json.loads(INVENTORY.read_text(encoding="utf-8"))
    problems = []
    count = 0
    for item in manifest["files"] + manifest["generated_namespace_files"]:
        path = project_root / item["path"]
        if not path.is_file() or path.is_symlink():
            problems.append({"path": item["path"], "error": "missing regular source file"})
            continue
        payload = path.read_bytes()
        expected = item.get("shipped_sha256", item.get("sha256"))
        if hashlib.sha256(payload).hexdigest() != expected:
            problems.append({"path": item["path"], "error": "source SHA mismatch"})
        text = payload.decode("utf-8")
        ast.parse(text, filename=item["path"])
        if any("\u4e00" <= character <= "\u9fff" for character in text):
            problems.append({"path": item["path"], "error": "non-English CJK code found"})
        count += 1
    dynamic = project_root / "research/english_v2_data/audit/index_maga_metadata.py"
    if not dynamic.is_file():
        problems.append({"path": str(dynamic.relative_to(project_root)), "error": "required dynamic structural helper missing"})
    # These are prerequisites, not an attempt to read or authorize their content.
    prerequisites = {name: (project_root / name).exists() for name in ("data", "models", "research/english_v2_data/audit", "research/english_v2_training_registered_v1")}
    return {"source_inventory_matches": not problems, "source_files_checked": count,
            "source_code_problems": problems, "external_prerequisite_paths_present": prerequisites,
            "dataset_or_checkpoint_content_verified": False, "fresh_QA_or_protocol_approved": False,
            "data_downloaded": False, "weights_loaded": False, "training_started": False,
            "notice": "Existence and historical hashes do not approve data rights, roles, parent checkpoints or a new training run."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "show", "run"))
    parser.add_argument("--project-root", type=Path, help="External legacy project root containing experiments/, research/, prepared data/ and parent models/")
    parser.add_argument("--stage", choices=tuple(STAGES))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--acknowledge-external-inputs", action="store_true", help="Affirm that this explicit run has independently prepared data, permissions, new QA/code registry and an approved exact protocol")
    values = list(sys.argv[1:] if argv is None else argv)
    separator = values.index("--") if "--" in values else len(values)
    forwarded = values[separator+1:] if separator < len(values) else []
    args = parser.parse_args(values[:separator])
    project = (args.project_root or BUNDLED_SOURCE).resolve()
    result = check(project)
    if args.action == "check":
        print(json.dumps(result, indent=2))
        return 0 if result["source_inventory_matches"] else 2
    if not args.stage:
        parser.error("--stage is required for show/run")
    if not result["source_inventory_matches"]:
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 2
    extra = forwarded
    command = [str(args.python.absolute()), "-B", "-m", STAGES[args.stage], *extra]
    if args.action == "show":
        print(json.dumps({"command": command, "cwd": str(project), "executed": False,
                          "requires_external_data_parent_models_and_new_exact_protocol": True}, indent=2))
        return 0
    if args.project_root is None:
        parser.error("run requires an explicit --project-root; check/show alone are read-only")
    help_only = extra in (["--help"], ["-h"])
    if not help_only:
        if project.is_relative_to(REPOSITORY):
            parser.error("real execution requires a separately prepared external project root, not the repository snapshot")
        if not args.acknowledge_external_inputs:
            parser.error("real execution requires --acknowledge-external-inputs; historical QA/configs are reference-only")
    reference = REPOSITORY / "reproduction/reference_metadata"
    for value in extra:
        token = value.split("=", 1)[1] if value.startswith("--") and "=" in value else value
        if token.startswith("--"):
            continue
        candidate = Path(token)
        if candidate.exists() and candidate.resolve().is_relative_to(reference):
            parser.error("sanitized historical reference metadata cannot be dispatched as a current protocol")
    env = dict(os.environ, PYTHONPATH=str(project), PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    return subprocess.run(command, cwd=project, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
