"""Constructed-only adapter checks. No model, network, GPU or server is used."""
from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import warnings
import zipfile
from unittest.mock import patch

import space_core as core
import bundle_archive


def fake_result(p_ai=.9, *, core_length=True, english=True):
    families = ["Claude", "GPT", "Gemini", "Qwen", "GLM", "Kimi", "Other"]
    counts = [6, 6, 5, 2, 3, 2, 1]
    leaves = [{"label": f"{f}:{i}", "family": f, "probability": 1 / (7*n)}
              for f, n in zip(families, counts) for i in range(n)]
    supported = core_length and english
    status = "AI-generated" if supported and p_ai >= core.AI_CUTOFF else "likely human" if supported and 1-p_ai >= core.HUMAN_CUTOFF else "inconclusive"
    return {"model_id": core.MODEL_ID, "status": status,
        "probabilities": {"human": 1-p_ai, "AI": p_ai},
        "family_probabilities": [{"label": f, "probability": 1/7} for f in families],
        "model_probabilities": leaves, "accepted_family": None, "accepted_model": None,
        "source_assertions": {"valid": False, "family_enabled": False, "model_enabled": False},
        "quality": {"word_like_tokens": 200 if core_length else 12,
            "core_length_supported": core_length, "encoder_tokens": 250,
            "encoder_coverage": 1., "window_counts": 1,
            "language": "en" if english else "unsupported"},
        "warnings": ["Constructed fixture, not a quality result."], "conclusion": "Constructed explanation."}


def fake_bundle(root):
    root.mkdir(parents=True)
    (root / "weights.txt").write_text("constructed bytes; not weights", encoding="utf-8")
    (root / "blend.json").write_text("{}", encoding="utf-8")
    (root / "decision.json").write_text("{}", encoding="utf-8")
    cal = {"thresholds": {"AI": {"threshold": core.AI_CUTOFF}, "human": {"threshold": core.HUMAN_CUTOFF}},
           "provenance": {"blend_sha256": core.sha(root / "blend.json")}}
    (root / "calibration.json").write_text(json.dumps(cal), encoding="utf-8")
    manifest = {"schema_version": 1, "model_id": core.MODEL_ID,
        "source_release_gate": None, "source_release_protocol": None,
        "source_assertions": {"valid": False, "family_enabled": False, "model_enabled": False},
        "binary_blend": "blend.json", "calibration": "calibration.json", "decision_protocol": "decision.json",
        "integrity": [{"path": p.name, "bytes": p.stat().st_size, "sha256": core.sha(p)} for p in sorted(root.iterdir())]}
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    digest = core.sha(root / "bundle.json")
    (root / "bundle.sha256").write_text(digest + "  bundle.json\n", encoding="utf-8")
    return manifest, digest


def fake_archive(root, destination, *, changes=None, extras=()):
    changes = changes or {}
    with warnings.catch_warnings(), zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        warnings.simplefilter("ignore", UserWarning)
        for p in sorted(root.iterdir()):
            name = "bundle/" + p.name
            zf.writestr(name, changes.get(name, p.read_bytes()))
        for name, data in extras:
            zf.writestr(name, data)
    return destination


class CoreChecks(unittest.TestCase):
    def test_config_is_immutable_and_explicit_zerogpu(self):
        c = core.Config.from_environment({})
        self.assertEqual(c.revision, "4a98aa247f3cf227c81eec54596b9c0f8d7c358e")
        self.assertEqual((c.repo, c.subdirectory, c.backend, c.library_package),
            ("ydh0822/english-ai-detector", "bundle", "zerogpu", "english_ai_detector"))
        requirement = "english-ai-detector @ git+https://github.com/ydhhhh/english-ai-detector.git@0835c57e49e9a44fb3414064ed0f2e93ec864434"
        self.assertIn(requirement, Path(__file__).with_name("requirements.txt").read_text().splitlines())
        self.assertEqual(core.Config.from_environment({"MODEL_REVISION": "a"*40, "INFERENCE_BACKEND": "cpu"}).backend, "cpu")
        for change in ({"MODEL_REVISION": ""}, {"MODEL_REVISION": "main"}, {"MODEL_REVISION": "a"*40, "INFERENCE_BACKEND": "paid"},
            {"MODEL_REVISION": "a"*40, "MODEL_REPO_ID": "bad"},
            {"MODEL_REVISION": "a"*40, "DETECTOR_LIBRARY_PACKAGE": "x; bad"},
            {"MODEL_REVISION": "a"*40, "BUNDLE_SUBDIRECTORY": "../x"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                core.Config.from_environment(change)

    def test_safe_relative_paths(self):
        for value in ("/x", "../x", "a/../x", "a//x", "a\\x", "C:x", "./x", "a\nx", ""):
            with self.subTest(path=value), self.assertRaises(ValueError): core.relative_name(value)
        self.assertEqual(core.relative_name("nested/file.json"), "nested/file.json")

    def test_complete_download_pinned_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "bundle"
            manifest, digest = fake_bundle(root)
            archive = fake_archive(root, Path(td) / "bundle.zip")
            calls = []
            def hub(**kw):
                calls.append(kw); return archive
            c = core.Config("owner/model", "a"*40)
            extract = functools.partial(bundle_archive.extract_bundle_archive, cache_directory=Path(td)/"cache")
            with patch.object(core, "BUNDLE_MANIFEST_SHA256", digest), patch.object(core, "BUNDLE_ARCHIVE_SHA256", core.sha(archive)), patch.object(core, "extract_bundle_archive", extract):
                extracted = core.download_bundle(c, hub)
                self.assertEqual({p.name for p in extracted.iterdir()}, set(core.member_names(manifest)))
                self.assertEqual(calls[0]["filename"], core.BUNDLE_ARCHIVE_FILENAME)
                self.assertTrue(all(x["revision"] == "a"*40 and x["token"] is False for x in calls))
                self.assertEqual(core.download_bundle(c, hub), extracted)
                (extracted / "weights.txt").write_text("tampered", encoding="utf-8")
                with self.assertRaises(ValueError): core.download_bundle(c, hub)
            with self.assertRaises(ValueError): core.download_bundle(c, hub)

    def test_inventory_source_and_collision_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            m, _ = fake_bundle(Path(td) / "bundle")
            for mutate in (lambda x: x.update(source_release_gate="gate.json"),
                lambda x: x["source_assertions"].update(family_enabled=True),
                lambda x: x["integrity"].append({"path": "BUNDLE.JSON", "bytes": 0, "sha256": "a"*64}),
                lambda x: x["integrity"][0].update(path="../escape"),
                lambda x: x["integrity"][0].update(bytes=True)):
                bad = copy.deepcopy(m); mutate(bad)
                with self.assertRaises(ValueError): core.member_names(bad)

    def test_text_preserved_and_never_truncated(self):
        self.assertEqual(core.pasted_text("  hello\n "), "  hello\n ")
        for text in (None, " ", "x"*(core.MAX_CHARACTERS+1)):
            with self.assertRaises(core.InputError): core.pasted_text(text)

    def test_uploaded_document_boundary_and_errors(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "uploads"; root.mkdir()
            path = root / "a.txt"; path.write_text("constructed English")
            def parser(data, name, **kw):
                self.assertEqual(kw, {"max_bytes": core.MAX_UPLOAD_BYTES, "max_characters": core.MAX_CHARACTERS})
                return data.decode()
            self.assertEqual(core.document_text(path, parser, root), "constructed English")
            bad = Path(td) / "outside.txt"; bad.write_text("outside")
            link = root / "link.txt"; link.symlink_to(path)
            pdf = root / "a.pdf"; pdf.write_text("pdf")
            empty = root / "empty.txt"; empty.write_bytes(b"")
            large = root / "large.txt"; large.write_bytes(b"x"*(core.MAX_UPLOAD_BYTES+1))
            for p in (bad, link, pdf, empty, large, None):
                with self.subTest(p=str(p)), self.assertRaises(core.InputError): core.document_text(p, parser, root)
            def fail(*args, **kwargs): raise ValueError("secret input/path")
            with self.assertRaises(core.InputError) as exc: core.document_text(path, fail, root)
            self.assertNotIn("secret", str(exc.exception))

    def test_display_preserves_all_probabilities_and_policy(self):
        for p_ai, core_length, english in ((.9, True, True), (.01, True, True), (.5, True, True), (.99, False, True), (.99, True, False)):
            value = fake_result(p_ai, core_length=core_length, english=english)
            before = copy.deepcopy(value)
            outputs = core.formatted_result(value)
            self.assertEqual(value, before)
            self.assertEqual([len(outputs[i]) for i in (2, 3, 4)], [2, 7, 25])
            self.assertEqual(outputs[1], value["conclusion"])
            if not core_length or not english: self.assertEqual(outputs[0], "Inconclusive")

    def test_display_rejects_invalid_or_enabled_assertions(self):
        for mutate in (lambda x: x.update(model_id="other"), lambda x: x.update(accepted_family="GPT"),
            lambda x: x["source_assertions"].update(model_enabled=True),
            lambda x: x.update(status="likely human"),
            lambda x: x["probabilities"].update(AI=float("nan")),
            lambda x: x["model_probabilities"][0].update(probability=.4),
            lambda x: x["quality"].update(encoder_coverage=.9)):
            value = fake_result(); mutate(value)
            with self.assertRaises(ValueError): core.formatted_result(value)


class Component:
    events = []
    all = []
    def __init__(self, *args, **kwargs): self.args=args; self.kwargs=kwargs; self.all.append(self)
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def click(self, fn, **kwargs): self.events.append((fn, kwargs)); return self
    def queue(self, **kwargs): self.queue_kwargs=kwargs; return self
    def launch(self, **kwargs): raise AssertionError("Checks must never start a service")


def mock_app():
    gr = types.ModuleType("gradio")
    for name in ("Blocks", "Markdown", "Tabs", "Tab", "Textbox", "Button", "File", "Dataframe", "ClearButton"):
        setattr(gr, name, type(name, (Component,), {}))
    gr.Error = type("Error", (Exception,), {})
    spaces = types.ModuleType("spaces"); spaces.calls=[]
    def gpu(**kwargs):
        spaces.calls.append(kwargs)
        def decorate(fn):
            def invoke(text): return fn(text)
            return invoke
        return decorate
    spaces.GPU=gpu
    hub = types.ModuleType("huggingface_hub")
    def forbidden(**kwargs): raise AssertionError("No network is allowed in constructed checks")
    hub.hf_hub_download=forbidden; hub.snapshot_download=forbidden
    spec = importlib.util.spec_from_file_location("constructed_space_app", Path(__file__).with_name("app.py"))
    app = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"gradio": gr, "spaces": spaces, "huggingface_hub": hub}):
        spec.loader.exec_module(app)
    return app, gr, spaces


class WiringChecks(unittest.TestCase):
    def setUp(self):
        Component.events=[]; Component.all=[]
        self.app, self.gr, self.spaces = mock_app()

    def test_both_inputs_share_one_queue_and_no_service(self):
        config=core.Config("owner/repo", "a"*40)
        demo=self.app.build_app(config, lambda text: fake_result(), lambda *a, **k: "constructed")
        self.assertEqual(demo.queue_kwargs, {"default_concurrency_limit": 1, "max_size": 16, "api_open": False})
        self.assertEqual(len(Component.events), 2)
        for fn, kwargs in Component.events:
            self.assertEqual((kwargs["concurrency_id"], kwargs["concurrency_limit"], kwargs["api_visibility"]), ("detector", 1, "private"))
        self.assertEqual(Component.events[0][0]("hello")[0], "Likely AI-generated")

    def test_uninitialized_and_quota_errors_never_substitute(self):
        self.app.build_app(initialization_error="Constructed")
        with self.assertRaises(self.gr.Error): Component.events[0][0]("hello")
        Component.events=[]
        def unavailable(text): raise RuntimeError("secret path or input")
        self.app.build_app(core.Config("owner/repo", "a"*40), unavailable, lambda *a: "hello")
        with self.assertRaises(self.gr.Error) as exc: Component.events[0][0]("hello")
        self.assertIn("ZeroGPU", str(exc.exception)); self.assertIn("no hardware upgrade", str(exc.exception))
        self.assertNotIn("secret", str(exc.exception))

    def test_public_api_cpu_and_zerogpu_startup_placement(self):
        loads=[]; predictions=[]; threads=[]
        detector=types.SimpleNamespace(model_id=core.MODEL_ID,
            source_assertions={"valid":False,"family_enabled":False,"model_enabled":False,"errors":[]},
            thresholds={"AI":{"threshold":core.AI_CUTOFF},"human":{"threshold":core.HUMAN_CUTOFF}},
            predict=lambda text: predictions.append(text) or fake_result())
        def load(path, **kwargs): loads.append(kwargs); return detector
        library=types.ModuleType("constructed_library")
        library.EnglishDetector=types.SimpleNamespace(load=load); library.extract_document=lambda *a, **k: "hello"
        torch=types.ModuleType("torch"); torch.set_num_threads=threads.append
        with patch.dict(sys.modules, {"constructed_library":library,"torch":torch}), patch.object(self.app, "download_bundle", return_value=Path("/constructed/not-real")):
            for backend in ("cpu", "zerogpu"):
                runner, parser, bundle=self.app.load_backend(core.Config("owner/repo", "a"*40, backend=backend, library_package="constructed_library"))
                self.assertEqual(predictions, [] if backend == "cpu" else ["hello"])
                runner("hello")
        self.assertEqual(loads, [{"device":"cpu","batch_size":4},{"device":"cuda","batch_size":4}])
        self.assertEqual(self.spaces.calls, [{"duration":30}])
        self.assertEqual(threads, [2, 2])

    def test_bad_archive_rejected_before_import_or_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"bundle"; _, digest=fake_bundle(root)
            archive=fake_archive(root, Path(td)/"bad.zip", changes={"bundle/weights.txt":b"x"*len((root/"weights.txt").read_bytes())})
            with patch.object(core, "BUNDLE_MANIFEST_SHA256", digest), patch.object(core, "BUNDLE_ARCHIVE_SHA256", core.sha(archive)), patch.object(self.app, "hf_hub_download", return_value=archive), patch.object(self.app.importlib, "import_module", side_effect=AssertionError("Model import must not occur")):
                with self.assertRaises(ValueError): self.app.load_backend(core.Config("owner/repo", "a"*40))


class ArchiveChecks(unittest.TestCase):
    def test_shared_helper_standalone_complete_and_missing_member(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"bundle"; manifest, digest=fake_bundle(root)
            archive=fake_archive(root, Path(td)/"good.zip")
            result=bundle_archive.extract_bundle_archive(archive, expected_archive_sha256=core.sha(archive),
                expected_manifest_sha256=digest, cache_directory=Path(td)/"cache")
            self.assertEqual(core.sha(result/"bundle.json"), digest)
            (root/"weights.txt").unlink()
            bad=fake_archive(root, Path(td)/"missing.zip")
            with self.assertRaises(ValueError): bundle_archive.extract_bundle_archive(bad,
                expected_archive_sha256=core.sha(bad), expected_manifest_sha256=digest, cache_directory=Path(td)/"cache")

    def test_archive_sha_is_checked_before_zip_open(self):
        with tempfile.TemporaryDirectory() as td:
            archive=Path(td)/"not-zip"; archive.write_bytes(b"constructed")
            with patch.object(bundle_archive.zipfile, "ZipFile", side_effect=AssertionError("ZIP must not open")):
                with self.assertRaises(ValueError): bundle_archive.extract_bundle_archive(archive,
                    expected_archive_sha256="0"*64, expected_manifest_sha256="a"*64)

    def test_traversal_duplicates_symlink_and_wrong_root_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"bundle"; _, digest=fake_bundle(root)
            names=["../escape", "/absolute", "bundle/../escape", "bundle\\evil", "other/file", "bundle/weights.txt", "bundle/WEIGHTS.TXT"]
            link=zipfile.ZipInfo("bundle/link"); link.create_system=3; link.external_attr=(0o120777 << 16)
            for i, name in enumerate([*names, link]):
                archive=fake_archive(root, Path(td)/f"bad-{i}.zip", extras=[(name,b"bad")])
                with self.subTest(name=str(name)), self.assertRaises(ValueError):
                    bundle_archive.extract_bundle_archive(archive, expected_archive_sha256=core.sha(archive),
                        expected_manifest_sha256=digest, cache_directory=Path(td)/"cache")

    def test_size_count_inventory_and_checksum_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"bundle"; _, digest=fake_bundle(root)
            good=fake_archive(root, Path(td)/"good.zip")
            for limit in ("MAX_ARCHIVE_MEMBERS", "MAX_ARCHIVE_BYTES", "MAX_MEMBER_BYTES", "MAX_MANIFEST_BYTES"):
                with self.subTest(limit=limit), patch.object(bundle_archive, limit, 1), self.assertRaises(ValueError):
                    bundle_archive.extract_bundle_archive(good, expected_archive_sha256=core.sha(good),
                        expected_manifest_sha256=digest, cache_directory=Path(td)/"cache")
            for i, change in enumerate(({"bundle/bundle.sha256":b"x"*257}, {"bundle/weights.txt":b"wrong-size"}, {"bundle/bundle.json":b"{}"})):
                bad=fake_archive(root, Path(td)/f"changed-{i}.zip", changes=change)
                with self.assertRaises(ValueError): bundle_archive.extract_bundle_archive(bad,
                    expected_archive_sha256=core.sha(bad), expected_manifest_sha256=digest, cache_directory=Path(td)/"cache")


if __name__ == "__main__":
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    root=Path(__file__).resolve().parent
    report={"schema_version":1,"constructed_only":True,"passed":result.wasSuccessful(),
        "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "actual_model_loaded":False,"network_used":False,"login_attempted":False,
        "gpu_used":False,"server_started":False,
        "publication_pins":{"model_revision":core.MODEL_REVISION,
            "bundle_archive_sha256":core.BUNDLE_ARCHIVE_SHA256,
            "bundle_manifest_sha256":core.BUNDLE_MANIFEST_SHA256,
            "runtime_github_commit":"0835c57e49e9a44fb3414064ed0f2e93ec864434"},
        "code_sha256":{name:core.sha(root/name) for name in ("checks.py","space_core.py","app.py","bundle_archive.py")},
        "scope":"Adapter integrity, transport, fixed policy display and mocked UI/runtime wiring only; not hosted inference or CUDA parity."}
    (root/"constructed_checks.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
