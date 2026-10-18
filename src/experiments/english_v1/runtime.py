"""Installed-package runtime defaults; never import project-local vendors."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
