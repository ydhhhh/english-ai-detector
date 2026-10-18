"""Pure small-file/path checks. Does not open any real archive/bundle/model."""
from __future__ import annotations
import argparse
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("relocated_clean_fixture_target", HERE / "run_relocated_clean.py")
qa = importlib.util.module_from_spec(spec); spec.loader.exec_module(qa)


def run():
    checks = []
    def check(name, function):
        function(); checks.append(name)
    def rejects(function):
        try: function()
        except (ValueError, FileNotFoundError, FileExistsError): return
        raise AssertionError("constructed invalid input was not rejected")
    with tempfile.TemporaryDirectory(prefix="relocated-clean-synthetic-") as folder:
        root = Path(folder).resolve()
        def makezip(name, entries):
            path = root / name
            with zipfile.ZipFile(path, "w") as z:
                for member, payload, mode in entries:
                    info = zipfile.ZipInfo(member)
                    info.external_attr = mode << 16
                    z.writestr(info, payload)
            return path
        regular = stat.S_IFREG | 0o644
        clean = makezip("source.zip", [(qa.PREFIX + "experiments/alpha.py", b"x = 1\n", regular),
            (qa.PREFIX + "README.md", b"Constructed fixture only\n", regular)])
        check("fixed-prefix regular UTF8 source extraction", lambda: qa.extract_source(clean, qa.sha(clean), root / "source"))
        check("source exact bytes preserved", lambda: qa.require((root / "source/experiments/alpha.py").read_bytes() == b"x = 1\n", "not exact"))
        check("source SHA drift rejected", lambda: rejects(lambda: qa.extract_source(clean, "0"*64, root / "bad-sha")))
        check("existing extraction directory never overwritten", lambda: rejects(lambda: qa.extract_source(clean, qa.sha(clean), root / "source")))
        for number, (name, member, mode) in enumerate([
            ("absolute archive path", "/tmp/escape.py", regular),
            ("traversal archive member", qa.PREFIX + "../escape.py", regular),
            ("repeated prefix", qa.PREFIX + qa.PREFIX + "x.py", regular),
            ("backslash member", qa.PREFIX + "dir\\escape.py", regular),
            ("symlink archive member", qa.PREFIX + "x.py", stat.S_IFLNK | 0o777),
            ("model tree in source", qa.PREFIX + "models/weight.py", regular),
            ("corpus tree in source", qa.PREFIX + "data/text.py", regular),
            ("local vendor runtime in source", qa.PREFIX + "research/a/vendor_fast/pkg.py", regular),
        ]):
            path = makezip(str(number)+".zip", [(member, b"constructed", mode)])
            check(name + " rejected", lambda path=path: rejects(lambda: qa.archive_index(path)))
        collision = makezip("collision.zip", [(qa.PREFIX + "A.py", b"a", regular), (qa.PREFIX + "a.py", b"b", regular)])
        check("casefold archive collision rejected", lambda: rejects(lambda: qa.archive_index(collision)))
        badtext = makezip("binary.zip", [(qa.PREFIX + "x.py", b"\xff\xfe", regular)])
        try: qa.extract_source(badtext, qa.sha(badtext), root / "badtext")
        except UnicodeDecodeError: checks.append("non-UTF8 source rejected")
        else: raise AssertionError("binary source accepted")
        link = root / "source_link"; link.symlink_to(root / "source", target_is_directory=True)
        check("symlink path input rejected", lambda: rejects(lambda: qa.no_symlinks(link / "experiments/alpha.py")))
        bundle = root / "fake_bundle"; bundle.mkdir()
        for name, value in {"calibration.json": {"source_policy": {"mode": "rankings_only", "family_enabled": False, "model_enabled": False}},
                            "binary_blend.json": {"constructed": True}, "decision_protocol.json": {"constructed": True}}.items():
            (bundle / name).write_text(json.dumps(value))
        (bundle / "components").mkdir(); (bundle / "components/not_a_model.bin").write_bytes(b"dummy file; never loaded")
        files = sorted(p for p in bundle.rglob("*") if p.is_file())
        manifest = {"schema_version": 1, "source_release_gate": None, "source_release_protocol": None,
            "source_assertions": {"valid": False, "family_enabled": False, "model_enabled": False},
            "binary_blend": "binary_blend.json", "calibration": "calibration.json", "decision_protocol": "decision_protocol.json",
            "integrity": [{"path": p.relative_to(bundle).as_posix(), "bytes": p.stat().st_size, "sha256": qa.sha(p)} for p in files]}
        (bundle / "bundle.json").write_text(json.dumps(manifest))
        manifest_sha = qa.sha(bundle / "bundle.json")
        (bundle / "bundle.sha256").write_text(manifest_sha+"  bundle.json\n")
        rows = qa.inspect_bundle(bundle, manifest_sha)
        check("complete dummy bundle member inventory accepted", lambda: qa.require(len(rows) == 6, "wrong count"))
        check("dummy bundle relocated with exact all-member SHA", lambda: qa.relocate_bundle(bundle, manifest_sha, root / "relocated"))
        check("existing relocated bundle never overwritten", lambda: rejects(lambda: qa.relocate_bundle(bundle, manifest_sha, root / "relocated")))
        check("bundle manifest SHA drift rejected", lambda: rejects(lambda: qa.inspect_bundle(bundle, "0"*64)))
        (bundle / "components/not_a_model.bin").write_bytes(b"tamper")
        check("bundle member byte drift rejected", lambda: rejects(lambda: qa.inspect_bundle(bundle, manifest_sha)))
        (bundle / "components/not_a_model.bin").write_bytes(b"dummy file; never loaded")
        (bundle / "unlisted.txt").write_text("not in inventory")
        check("unlisted bundle file rejected", lambda: rejects(lambda: qa.inspect_bundle(bundle, manifest_sha)))
        (bundle / "unlisted.txt").unlink()
        (bundle / "components/alias").symlink_to(bundle / "components/not_a_model.bin")
        check("bundle symlink rejected", lambda: rejects(lambda: qa.inspect_bundle(bundle, manifest_sha)))
        with mock.patch.dict(qa.os.environ, {"PYTHONPATH": "bad", "PYTHONHOME": "bad", "VIRTUAL_ENV": "bad", "DYLD_INSERT_LIBRARIES": "bad", "HF_HOME": "bad", "PATH": "bad"}, clear=False):
            env = qa.clean_environment(root / "work", 4)
        check("Python/injection/local cache environment stripped", lambda: qa.require(not any(k in env for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "DYLD_INSERT_LIBRARIES")) and env["HF_HOME"] != "bad", "dirty env"))
        check("CPU offline env uses existing interpreter and no install", lambda: qa.require(env["CUDA_VISIBLE_DEVICES"] == "" and env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1" and env["PATH"].startswith(str(qa.VENV / "bin")), "not isolated"))
        with mock.patch.object(sys, "argv", [str(HERE / "run_relocated_clean.py")]), mock.patch.object(qa.Path, "open", side_effect=AssertionError("default opened a file")), mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            qa.main(); policy = json.loads(stdout.getvalue())
        check("default invocation opens no artifact or subprocess", lambda: qa.require(policy == qa.POLICY and policy["dependency_installation"] is False, "wrong policy"))
        check("source ZIP report timing is explicit and not recursive", lambda: qa.require("postdates" in qa.POLICY["source_report_cycle"] and "new SHA" in qa.POLICY["source_report_cycle"], "cycle not explicit"))
    return {"passed": True, "constructed_only": True, "checks": len(checks), "check_names": checks,
        "real_bundle_archive_models_predictions_or_corpus_opened": False, "GPU_or_actual_service_or_CLI_executed": False,
        "network_or_dependency_installation": False, "launcher_sha256": qa.sha(HERE / "run_relocated_clean.py"), "checks_sha256": qa.sha(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); report = run(); qa.write_new(args.output, report)
    print(json.dumps({"passed": report["passed"], "constructed_only": True, "checks": report["checks"]}))
