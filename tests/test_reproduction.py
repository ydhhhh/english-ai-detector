"""Read-only reproduction preflight and accidental-execution guards."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reproduce.py"
spec = importlib.util.spec_from_file_location("public_reproduction_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

class ReproductionChecks(unittest.TestCase):
    def test_check_and_show_do_not_launch_processes(self):
        with patch.object(launcher.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launcher.main(["check"]), 0)
        child.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertFalse(result["training_started"])
        self.assertFalse(result["fresh_QA_or_protocol_approved"])
        with patch.object(launcher.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launcher.main(["show", "--stage", "train"]), 0)
        child.assert_not_called()
        self.assertFalse(json.loads(output.getvalue())["executed"])

    def test_source_drift_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            shutil.copytree(launcher.BUNDLED_SOURCE, root)
            path = root / "experiments/english_v2/train_registered.py"
            path.write_text(path.read_text() + "\n# Deliberate synthetic drift.\n")
            result = launcher.check(root)
            self.assertFalse(result["source_inventory_matches"])
            self.assertTrue(any(x["error"] == "source SHA mismatch" for x in result["source_code_problems"]))

    def test_repository_snapshot_cannot_start_training(self):
        with patch.object(launcher.subprocess, "run") as child, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                launcher.main(["run", "--project-root", str(launcher.BUNDLED_SOURCE), "--stage", "train", "--acknowledge-external-inputs", "--", "run"])
        self.assertEqual(error.exception.code, 2)
        child.assert_not_called()

    def test_reference_wrapper_cannot_be_used_as_current_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            shutil.copytree(launcher.BUNDLED_SOURCE, root)
            reference = next((launcher.REPOSITORY / "reproduction/reference_metadata").glob("*.json"))
            for binding in (["--protocol", str(reference)], ["--protocol=" + str(reference)]):
                with patch.object(launcher.subprocess, "run") as child, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        launcher.main(["run", "--project-root", str(root), "--stage", "train", "--acknowledge-external-inputs", "--", "run", *binding])
                self.assertEqual(error.exception.code, 2)
                child.assert_not_called()

if __name__ == "__main__":
    unittest.main()
