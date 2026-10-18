"""Project-local optional dependencies; the historical v1 environment stays intact."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in [ROOT / "research/data_audit_20261017/vendor_fast", ROOT / ".runtime/english"]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/english_hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
