"""CPU-only, offline checks for packaging, parsing and operating semantics."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.resources
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import english_ai_detector
from english_ai_detector import EnglishDetector, DocumentError, extract_text
from english_ai_detector.cli import main
from experiments.english_v1.language_guard import LanguageGuard
from tests.fixtures import PASSAGE, detector

class OfflineChecks(unittest.TestCase):
    def test_exact_numeric_compatibility_files(self):
        package = importlib.resources.files("english_ai_detector")
        manifest = json.loads(package.joinpath("compatibility_manifest.json").read_text())
        source = Path(english_ai_detector.__file__).resolve().parent.parent
        for item in manifest["exact_files"]:
            path = source / item["public_path"].removeprefix("src/")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
        from experiments.english_v1 import runtime
        self.assertTrue(Path(runtime.__file__).resolve().is_relative_to(source))
        self.assertNotIn("sys.path.insert", Path(runtime.__file__).read_text())

    def test_installed_english_html_resource(self):
        page = importlib.resources.files("english_ai_detector").joinpath("web/index.html").read_text()
        self.assertIn('<html lang="en">', page)
        self.assertNotIn("innerHTML", page)
        self.assertIn("rankings only", page)

    def test_real_language_scope(self):
        guard = LanguageGuard()
        self.assertTrue(guard.check(PASSAGE)["supported"])
        self.assertFalse(guard.check("La biblioteca del barrio abre cada semana para todos los vecinos. " * 20)["supported"])

    def test_complete_window_coverage(self):
        from experiments.english_v1.neural import windows
        for n in (0, 1, 510, 511, 10000, 50001):
            w = windows(n, 510, 448)
            if n:
                self.assertEqual(w[-1][1], n)
                self.assertEqual(sum(novel for _, _, novel in w), n)
                self.assertTrue(all(w[i][0] <= w[i-1][1] for i in range(1, len(w))))
        with self.assertRaises(ValueError):
            windows(10, 5, 6)

    def test_constructed_full_decision_and_hierarchy(self):
        model = detector()
        result = model.predict(PASSAGE)
        self.assertEqual(result["status"], "AI-generated")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.)
        self.assertIsNone(result["accepted_family"])
        self.assertIsNone(result["accepted_model"])
        for family in result["family_probabilities"]:
            total = sum(x["probability"] for x in result["model_probabilities"] if x["family"] == family["label"])
            self.assertAlmostEqual(total, family["probability"])
        self.assertEqual(model.predict("A short constructed passage.")["status"], "inconclusive")
        self.assertEqual(model.predict("Balanced fixture. " + PASSAGE)["status"], "inconclusive")
        self.assertEqual(model.predict("La biblioteca del barrio abre cada semana para todos los vecinos. " * 20)["status"], "inconclusive")
        with self.assertRaises(ValueError):
            model.predict_many([])
        with self.assertRaises(ValueError):
            model.predict_many(PASSAGE)

    def test_cli_and_api_probability_consistency_with_constructed_component(self):
        model = detector()
        with patch.object(EnglishDetector, "load", return_value=model), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["predict", "--bundle", "synthetic", "--text", PASSAGE]), 0)
        self.assertEqual(json.loads(output.getvalue())["probabilities"], model.predict(PASSAGE)["probabilities"])
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(main(["predict", "--bundle", "missing", "--file", "/synthetic-nonexistent-passage.txt"]), 2)
        self.assertIn("Error:", error.getvalue())

    def test_bundle_integrity_fails_closed_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "manifest checksum"):
                EnglishDetector.load(d, device="cpu")

    def test_full_synthetic_document_suite(self):
        from tests.parser_checks import run
        result = run()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["planned_cases"], 43)
        self.assertEqual(sum(result["case_counts"].values()), 43)

if __name__ == "__main__":
    unittest.main()
