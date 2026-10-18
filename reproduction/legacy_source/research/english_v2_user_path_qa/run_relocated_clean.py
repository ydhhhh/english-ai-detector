"""Prepare or run relocated source+bundle CPU engineering QA, never quality QA.

Default invocation prints policy without opening any archive/bundle. Future run:
  python -B research/english_v2_user_path_qa/run_relocated_clean.py --execute \
    --source-zip SOURCE.zip --source-sha256 SHA --bundle BUNDLE_DIR \
    --bundle-manifest-sha256 SHA --workdir NEW_DIR --output NEW_REPORT_DIR

Reuses the installed .qa/english-clean-venv without installing dependencies.
Only newly authored synthetic inputs reach the real CPU model/localhost server.
The external report necessarily postdates the ZIP; it is not inside that same
ZIP. Any later documentation/source archive has a new SHA and is a new artifact,
not recursive evidence that the original ZIP contained its own future QA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import site
import stat
import subprocess
import sys
import tempfile
import time
import unicodedata
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
VENV = PROJECT / ".qa/english-clean-venv"
PREFIX = "english-ai-detector/"
MAX_MEMBER = 32 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024
MODULES = (
    ("CLI_DOCX", "experiments.english_v1.inference", "c13a75461671d7e30dcef76f00f78e96a7d1eeef9109f7430af8df99afc4debf"),
    ("bundle_checks", "experiments.english_v1.bundle_checks", "04861add6a61718a75efd559c4fbecb67b49174243c9a564cdb31acb73012d5f"),
    ("HTTP_checks", "experiments.english_v1.http_bundle_checks", "51352a71206597e416a92ab2779df48246c288f9e46255b6461690c3d66c3fbe"),
)
LEGACY_DRIVER_SHA = "4a79cd4f9952e2e66e542d1e286fb67db0b3a3412ec9608c2b8f2e79832914ba"
POLICY = {
    "schema_version": 1, "scope": "Relocated packaged-source engineering QA, not model quality or acceptance",
    "default_opens_no_artifacts": True, "CPU_only": True, "dependency_installation": False,
    "reuse": "Original v1 pure child CLI DOCX logic and unchanged shipped bundle_checks/http_bundle_checks",
    "cases": ["real CLI synthetic DOCX", "CPU single/batch/order/reload/long coverage checks",
        "real localhost server 14 cases: TXT UTF8/BOM/UTF16, DOCX table/body, genuine DOC, error recovery"],
    "source_assertions": "rankings only; accepted_family/model always null",
    "archive_root": PREFIX, "no_source_injection": True,
    "source_report_cycle": "Report is external and postdates this exact source ZIP. A later source ZIP has a new SHA; no self-containing QA claim.",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def relative_name(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value and
        not any(ord(c) < 32 for c in value) and unicodedata.normalize("NFC", value) == value, "unsafe relative path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and p.as_posix() == value and all(q not in {".", ".."} for q in p.parts), "noncanonical relative path")
    return value


def no_symlinks(path):
    p = Path(os.path.abspath(path))
    for parent in (*reversed(p.parents), p):
        require(not parent.is_symlink(), "symlink input/output path forbidden")
    return p


def inside(path, parent):
    return path == parent or parent in path.parents


def archive_index(path):
    """Inspect only an explicitly SHA-bound source ZIP during --execute."""
    rows, folded, total = [], set(), 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            require(not info.is_dir() and info.filename.startswith(PREFIX) and not info.flag_bits & 1, "wrong archive prefix/type")
            require(stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG), "archive symlink/special member")
            name = relative_name(info.filename[len(PREFIX):])
            require(not name.startswith(PREFIX) and name.casefold() not in folded, "repeated root/colliding archive member")
            require(not any(p.casefold() in {"data", "models", ".venv", ".runtime", "vendor", "vendor_fast", ".git", ".cache", "__pycache__"}
                        for p in PurePosixPath(name).parts), "source contains forbidden corpus/model/runtime/vendor tree")
            require(0 <= info.file_size <= MAX_MEMBER, "oversized source member")
            total += info.file_size
            require(total <= MAX_TOTAL, "oversized source archive")
            folded.add(name.casefold()); rows.append((name, info))
        require(rows and z.testzip() is None, "empty or corrupt source ZIP")
    return rows


def extract_source(path, expected_sha, destination):
    require(is_sha(expected_sha) and sha(path) == expected_sha and not destination.exists(), "source SHA/new extraction path required")
    rows = archive_index(path)
    destination.mkdir(parents=True, exist_ok=False)
    entries = {}
    with zipfile.ZipFile(path) as z:
        for name, info in rows:
            data = z.read(info)
            data.decode("utf-8", errors="strict")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(data)
            entries[name] = {"bytes": len(data), "sha256": sha(target)}
    require(sha(path) == expected_sha, "source changed while extracting")
    return entries


def inspect_bundle(bundle, expected_sha):
    bundle = no_symlinks(bundle)
    manifest_path = bundle / "bundle.json"
    require(bundle.is_dir() and is_sha(expected_sha) and sha(manifest_path) == expected_sha, "bundle manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require((bundle / "bundle.sha256").read_text().split() == [expected_sha, "bundle.json"], "bundle checksum declaration differs")
    require(manifest.get("source_release_gate") is None and manifest.get("source_release_protocol") is None and
        manifest.get("source_assertions") == {"valid": False, "family_enabled": False, "model_enabled": False}, "v2 must remain rankings-only")
    rows = {"bundle.json": {"sha256": expected_sha, "bytes": manifest_path.stat().st_size},
            "bundle.sha256": {"sha256": sha(bundle / "bundle.sha256"), "bytes": (bundle / "bundle.sha256").stat().st_size}}
    folded = {p.casefold() for p in rows}
    for entry in manifest["integrity"]:
        name = relative_name(entry["path"])
        require(name.casefold() not in folded and is_sha(entry["sha256"]) and isinstance(entry["bytes"], int) and not isinstance(entry["bytes"], bool), "duplicate/invalid bundle integrity entry")
        p = no_symlinks(bundle / name)
        require(p.is_file() and p.stat().st_size == entry["bytes"] and sha(p) == entry["sha256"], "bundle member SHA/size differs")
        folded.add(name.casefold()); rows[name] = {"sha256": entry["sha256"], "bytes": entry["bytes"]}
    actual = set()
    for p in bundle.rglob("*"):
        require(not p.is_symlink(), "symlink in bundle tree")
        if p.is_file(): actual.add(p.relative_to(bundle).as_posix())
        else: require(p.is_dir(), "special bundle member")
    require(actual == set(rows), "bundle contains unlisted/missing files")
    for name in (manifest["binary_blend"], manifest["calibration"], manifest["decision_protocol"]):
        require(relative_name(name) in rows, "bundle references unlisted policy artifact")
    cal = json.loads((bundle / manifest["calibration"]).read_text())
    require(cal["source_policy"]["mode"] == "rankings_only" and cal["source_policy"]["family_enabled"] is False and
        cal["source_policy"]["model_enabled"] is False, "calibration enables source assertions")
    require(sha(manifest_path) == expected_sha, "bundle manifest changed while inspecting")
    return rows


def relocate_bundle(bundle, expected_sha, destination):
    rows = inspect_bundle(bundle, expected_sha)
    require(not destination.exists(), "relocated bundle path must be new")
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(rows):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream, (bundle / name).open("rb") as source:
            shutil.copyfileobj(source, stream, 1024 * 1024)
    require(inspect_bundle(destination, expected_sha) == rows and inspect_bundle(bundle, expected_sha) == rows, "bundle changed while relocating")
    return rows


def clean_environment(work, threads):
    env = {k: v for k, v in os.environ.items() if k in {"HOME", "TMPDIR", "LANG", "LC_CTYPE", "TZ", "USER", "LOGNAME", "SYSTEMROOT"}}
    env.update(PATH=str(VENV / "bin") + ":/usr/bin:/bin:/usr/sbin:/sbin", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(work / "empty-hf-cache"), TORCH_HOME=str(work / "empty-torch-cache"), CUDA_VISIBLE_DEVICES="")
    env.update({k: str(threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    return env


def imported_origins(source, venv):
    origins = {"project": {}, "third_party": {}, "stdlib": {}}
    stdlib = Path(sys.base_prefix).resolve()
    for name, imported in sorted(sys.modules.items()):
        raw = getattr(imported, "__file__", None)
        if not raw or raw.startswith("<"): continue
        if name in {"torch.classes", "torch.ops"}:
            implementation = {"torch.classes": "torch._classes", "torch.ops": "torch._ops"}[name]
            require(type(imported).__module__ == implementation, "unexpected dynamic torch origin")
            p = Path(sys.modules[implementation].__file__).resolve()
            require(inside(p, venv), "torch dynamic implementation outside clean venv")
            origins["third_party"][name] = {"implementation": implementation, "path": str(p.relative_to(venv))}
            continue
        p = Path(raw).resolve()
        if name in {"__main__", "__mp_main__"}:
            require(p == Path(__file__).resolve(), "unexpected QA launcher origin")
        elif name == "experiments" or name.startswith("experiments."):
            require(inside(p, source), "project import escaped relocated source")
            origins["project"][name] = {"path": str(p.relative_to(source)), "sha256": sha(p)}
        elif inside(p, venv):
            origins["third_party"][name] = str(p.relative_to(venv))
        elif inside(p, stdlib) and "site-packages" not in p.parts:
            origins["stdlib"][name] = str(p.relative_to(stdlib))
        else:
            raise ValueError("unexpected imported module origin: " + name)
    for value in sys.path:
        if value:
            p = Path(value).resolve()
            require(not inside(p, PROJECT) or inside(p, source) or inside(p, venv), "original workspace/vendor path leaked into isolated child")
    return origins


def child(module, output, source, bundle, venv):
    """Original child cases, parameterized paths; no extra shipped QA injection."""
    source, bundle, venv = source.resolve(), bundle.resolve(), venv.resolve()
    require(Path(sys.prefix).resolve() == venv and sys.flags.isolated and not site.ENABLE_USER_SITE, "clean isolated interpreter required")
    require(Path.cwd().resolve() == source and "PYTHONPATH" not in os.environ, "relocated cwd/clean environment required")
    require(all(not (source / p).exists() for p in ("data", "models", ".runtime", ".venv", "research/data_audit_20261017/vendor_fast")), "unshipped dependencies present")
    require(module in {m for _, m, _ in MODULES}, "only original three child modules allowed")
    sys.path.insert(0, str(source))
    entry = importlib.util.find_spec(module)
    require(entry is not None and inside(Path(entry.origin).resolve(), source), "entry not from shipped source")
    started = time.monotonic()
    sys.argv = [module, "--bundle", str(bundle), "--output", str(output)]
    if module.endswith("bundle_checks") and not module.endswith("http_bundle_checks"):
        sys.argv += ["--device", "cpu"]
    code = 0
    if module.endswith("inference"):
        from experiments.english_v1.http_bundle_checks import synthetic_body, docx_fixture
        from experiments.english_v1.bundle_checks import validate_result
        paragraphs, text = synthetic_body()
        with tempfile.TemporaryDirectory(prefix="relocated-CLI-") as folder:
            fixture, raw = Path(folder) / "synthetic.docx", Path(folder) / "prediction.json"
            fixture.write_bytes(docx_fixture(paragraphs))
            sys.argv = [module, "--bundle", str(bundle), "--file", str(fixture), "--device", "cpu", "--output", str(raw)]
            runpy.run_module(module, run_name="__main__")
            value = json.loads(raw.read_text()); validate_result(value)
            require(value["accepted_family"] is None and value["accepted_model"] is None, "source assertions unexpectedly enabled")
            write_new(output, {"passed": True, "real_bundle_loaded": True, "synthetic_DOCX_CLI": True,
                "model_id": value["model_id"], "input_characters": len(text), "word_like_tokens": value["quality"]["word_like_tokens"],
                "probabilities_finite_and_hierarchical": True, "source_rankings_only": True,
                "prediction_and_input_text_stored": False, "CPU_only": True})
    else:
        try: runpy.run_module(module, run_name="__main__")
        except SystemExit as exc: code = exc.code or 0
    require(code == 0, "shipped child CLI failed")
    report = json.loads(output.read_text()); require(report["passed"] is True, "child engineering check failed")
    import torch
    require(torch.get_default_device().type == "cpu", "default device is not CPU")
    runtime = sys.modules.get("experiments.english_v1.runtime")
    require(runtime is not None and runtime.ROOT.resolve() == source, "runtime ROOT not relocated source")
    if module.endswith("http_bundle_checks"):
        require(report["real_bundle_loaded"] and report["legacy_DOC_upload_verified"] and report["checks_failed"] == report["checks_skipped"] == 0
            and report["checks_passed"] == 14 and report["server_closed"], "all original14 HTTP cases including DOC/recovery required")
    elif module.endswith("bundle_checks"):
        require(report["device"] == "cpu" and report["compare_device"] is None and report["long_input_windows"] > 1, "CPU original bundle checks required")
    distributions = sorted([{"name": d.metadata["Name"], "version": d.version,
        "relative_location": str(Path(d.locate_file("")).resolve().relative_to(venv))}
        for d in importlib.metadata.distributions()], key=lambda d: d["name"].casefold())
    origins = imported_origins(source, venv)
    write_new(output.with_suffix(".imports.json"), {"passed": True, "module": module, "entry_sha256": sha(entry.origin),
        "clean_interpreter": str(venv / "bin/python"), "python": sys.version, "isolated_mode": True,
        "user_site_disabled": True, "PYTHONPATH_present": False, "runtime_ROOT_verified": True,
        "original_workspace_vendor_runtime_data_imports_used": False, "no_extra_shipped_QA_code_injection": True,
        "actual_bundle_only_synthetic_inputs": True, "no_real_calibration_final_corpus_or_predictions_read": True,
        "CPU_only": True, "elapsed_seconds": time.monotonic() - started, "output_sha256": sha(output),
        "module_origins": origins, "installed_distributions": distributions, "sys_path": sys.path})


def execute(args):
    source_zip, bundle = no_symlinks(args.source_zip), no_symlinks(args.bundle)
    work, output = no_symlinks(args.workdir), no_symlinks(args.output)
    require(not work.exists() and not output.exists() and not inside(work, output) and not inside(output, work), "new disjoint work/report directories required")
    require(not any(inside(work, p) or inside(output, p) or inside(p, work) or inside(p, output) for p in (bundle, source_zip)), "outputs cannot overlap artifact inputs")
    require(is_sha(args.source_sha256) and sha(source_zip) == args.source_sha256, "explicit source archive SHA required")
    require((VENV / "bin/python").is_file() and 1 <= args.threads <= 8, "existing clean venv/CPU thread range required")
    # No writes until both artifact inventories and every input SHA pass.
    archive_index(source_zip)
    bundle_rows = inspect_bundle(bundle, args.bundle_manifest_sha256)
    work.mkdir(parents=True, exist_ok=False); output.mkdir(parents=True, exist_ok=False)
    source = work / "source/english-ai-detector"
    relocated = work / "relocated/english_detector_v2"
    source_rows = extract_source(source_zip, args.source_sha256, source)
    relocate_bundle(bundle, args.bundle_manifest_sha256, relocated)
    for _, module, expected in MODULES:
        require(sha(source / (module.replace(".", "/") + ".py")) == expected, "original child module source changed")
    env = clean_environment(work, args.threads)
    driver_sha = sha(__file__)
    state = {"schema_version": 1, "policy": POLICY, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive_sha256": args.source_sha256, "bundle_manifest_sha256": args.bundle_manifest_sha256,
        "launcher_sha256": driver_sha, "legacy_child_logic_reference_sha256": LEGACY_DRIVER_SHA,
        "source_file_count": len(source_rows), "relocated_bundle_file_count": len(bundle_rows), "results": [],
        "no_dependency_reinstall": True, "no_injected_shipped_QA_code": True, "CPU_only": True,
        "report_external_to_source_archive": True, "source_archive_contains_this_future_QA_result": False,
        "no_quality_acceptance_claim": True}
    try:
        for name, command in [("pip_check", [str(VENV / "bin/python"), "-I", "-B", "-m", "pip", "check"])] + [
            (name, [str(VENV / "bin/python"), "-I", "-B", str(Path(__file__).resolve()), "--child-module", module,
                "--child-source", str(source), "--child-bundle", str(relocated), "--child-venv", str(VENV), "--output", str(output / (name + ".json"))])
            for name, module, _ in MODULES]:
            start = time.monotonic()
            with (output / (name + ".log")).open("xb") as log:
                proc = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
            state["results"].append({"name": name, "returncode": proc.returncode, "elapsed_seconds": time.monotonic() - start,
                "log_sha256": sha(output / (name + ".log"))})
            require(proc.returncode == 0, "isolated child failed: " + name)
        require(sha(__file__) == driver_sha and sha(source_zip) == args.source_sha256, "launcher/archive changed during QA")
        require(inspect_bundle(bundle, args.bundle_manifest_sha256) == bundle_rows and inspect_bundle(relocated, args.bundle_manifest_sha256) == bundle_rows, "bundle changed during QA")
        require(all(sha(source / n) == row["sha256"] for n, row in source_rows.items()), "extracted source changed during QA")
        state.update(passed=True, completed_at_utc=datetime.now(timezone.utc).isoformat(), all_input_and_relocated_hashes_unchanged=True,
            reports={p.name: sha(p) for p in sorted(output.glob("*.json"))})
    except BaseException as exc:
        state.update(passed=False, failed_at_utc=datetime.now(timezone.utc).isoformat(), error_type=type(exc).__name__)
        write_new(output / "verification.json", state)
        raise
    write_new(output / "verification.json", state)
    print(json.dumps({"passed": True, "report": str(output / "verification.json"), "CPU_only": True}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--source-zip", type=Path); p.add_argument("--source-sha256")
    p.add_argument("--bundle", type=Path); p.add_argument("--bundle-manifest-sha256")
    p.add_argument("--workdir", type=Path); p.add_argument("--output", type=Path)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--child-module", choices=[m for _, m, _ in MODULES])
    p.add_argument("--child-source", type=Path); p.add_argument("--child-bundle", type=Path); p.add_argument("--child-venv", type=Path)
    args = p.parse_args()
    if args.child_module:
        require(all((args.child_source, args.child_bundle, args.child_venv, args.output)), "internal child paths required")
        child(args.child_module, args.output, args.child_source, args.child_bundle, args.child_venv)
    elif args.execute:
        require(all((args.source_zip, args.source_sha256, args.bundle, args.bundle_manifest_sha256, args.workdir, args.output)), "explicit artifact bindings and new directories required")
        execute(args)
    else:
        print(json.dumps(POLICY, indent=2))


if __name__ == "__main__":
    main()
