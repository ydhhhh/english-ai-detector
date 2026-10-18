"""Small transport/display adapter; the installed frozen library owns inference."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from bundle_archive import extract_bundle_archive

BUNDLE_MANIFEST_SHA256 = "f588dc69355cc96105ea627e0aa20bdcd9bed78709038884466abf008a4271cc"
BUNDLE_ARCHIVE_SHA256 = "da04e4314bdc4571e1dfff8414d203865ac03ad6d14aeff2a182ac070b5be164"
BUNDLE_ARCHIVE_FILENAME = "english_ai_detector_v2_bundle.zip"
MODEL_REVISION = "4a98aa247f3cf227c81eec54596b9c0f8d7c358e"
MODEL_ID = "english-ai-detector-v2-practical"
AI_CUTOFF = 0.7233173320180257
HUMAN_CUTOFF = 0.9049688379295407
MAX_CHARACTERS = 200_000
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


class ConfigurationError(ValueError):
    pass


class InputError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_name(value, *, allow_empty=False):
    if allow_empty and value == "":
        return ""
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value and
            not any(ord(c) < 32 for c in value), "unsafe bundle-relative path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and p.as_posix() == value and all(q not in {".", ".."} for q in p.parts), "unsafe bundle-relative path")
    return value


@dataclass(frozen=True)
class Config:
    repo: str
    revision: str
    subdirectory: str = "bundle"
    backend: str = "zerogpu"
    library_package: str = "english_ai_detector"

    @classmethod
    def from_environment(cls, env=None):
        env = os.environ if env is None else env
        repo = env.get("MODEL_REPO_ID", "ydh0822/english-ai-detector").strip()
        revision = env.get("MODEL_REVISION", MODEL_REVISION).strip()
        backend = env.get("INFERENCE_BACKEND", "zerogpu").strip().lower()
        package = env.get("DETECTOR_LIBRARY_PACKAGE", "english_ai_detector").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo):
            raise ConfigurationError("Set MODEL_REPO_ID to the approved public Model Hub repository.")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ConfigurationError("Set MODEL_REVISION to its exact 40-character commit, not a moving branch.")
        if backend not in {"cpu", "zerogpu"}:
            raise ConfigurationError("INFERENCE_BACKEND must be cpu or zerogpu; no paid-hardware fallback exists.")
        if not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", package):
            raise ConfigurationError("Invalid installed detector library package.")
        subdirectory = env.get("BUNDLE_SUBDIRECTORY", "bundle").strip()
        if subdirectory != "bundle":
            raise ConfigurationError("The approved archive has one fixed internal bundle/ root.")
        return cls(repo, revision, subdirectory, backend, package)


def member_names(manifest):
    require(manifest.get("model_id") == MODEL_ID and manifest.get("schema_version") == 1, "unexpected model identity")
    require(manifest.get("source_release_gate") is None and manifest.get("source_release_protocol") is None and
            manifest.get("source_assertions") == {"valid": False, "family_enabled": False, "model_enabled": False}, "source assertions must stay disabled")
    names = ["bundle.json", "bundle.sha256"]
    folded = {x.casefold() for x in names}
    for row in manifest["integrity"]:
        name = relative_name(row["path"])
        require(name.casefold() not in folded and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) and
                isinstance(row["bytes"], int) and not isinstance(row["bytes"], bool) and row["bytes"] >= 0, "invalid bundle inventory")
        names.append(name); folded.add(name.casefold())
    for field in ("binary_blend", "calibration", "decision_protocol"):
        require(manifest[field] in names, "missing frozen policy member")
    return names


def download_bundle(config, hub_download):
    """Download one complete approved archive at an immutable Hub commit."""
    require(isinstance(BUNDLE_ARCHIVE_SHA256, str) and re.fullmatch(r"[0-9a-f]{64}", BUNDLE_ARCHIVE_SHA256),
        "the approved bundle archive hash is not configured")
    archive = Path(hub_download(repo_id=config.repo, filename=BUNDLE_ARCHIVE_FILENAME,
        revision=config.revision, repo_type="model", token=False))
    bundle = extract_bundle_archive(archive, expected_archive_sha256=BUNDLE_ARCHIVE_SHA256,
        expected_manifest_sha256=BUNDLE_MANIFEST_SHA256, member_names=member_names)
    require(sha(bundle / "bundle.json") == BUNDLE_MANIFEST_SHA256, "bundle manifest changed during download")
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    require((bundle / "bundle.sha256").read_text().split() == [BUNDLE_MANIFEST_SHA256, "bundle.json"], "bundle checksum declaration differs")
    for row in manifest["integrity"]:
        p = bundle / row["path"]
        require(p.is_file() and p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], "incomplete or changed bundle member")
    cal = json.loads((bundle / manifest["calibration"]).read_text())
    require(cal["thresholds"]["AI"]["threshold"] == AI_CUTOFF and cal["thresholds"]["human"]["threshold"] == HUMAN_CUTOFF,
            "frozen decision cutoffs differ")
    require(cal["provenance"]["blend_sha256"] == sha(bundle / manifest["binary_blend"]), "blend/calibration identity differs")
    return bundle


def pasted_text(value):
    if not isinstance(value, str) or not value.strip():
        raise InputError("Enter a nonempty English passage.")
    if len(value) > MAX_CHARACTERS:
        raise InputError("This demo accepts at most 200,000 characters per request. Nothing was truncated.")
    return value  # Preserve the original text, including its whitespace.


def document_text(filename, parser, upload_root):
    if not isinstance(filename, (str, os.PathLike)):
        raise InputError("Upload one TXT, DOCX or DOC file.")
    p = Path(filename)
    resolved, root = p.resolve(), Path(upload_root).resolve()
    if root not in resolved.parents or p.is_symlink() or not p.is_file() or p.suffix.lower() not in {".txt", ".docx", ".doc"}:
        raise InputError("Use a file uploaded through this page: TXT, DOCX or DOC.")
    if not 0 < p.stat().st_size <= MAX_UPLOAD_BYTES:
        raise InputError("The file must be nonempty and no larger than 8 MiB.")
    payload = p.read_bytes()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise InputError("The upload exceeded 8 MiB. Nothing was truncated.")
    try:
        value = parser(payload, p.name, max_bytes=MAX_UPLOAD_BYTES, max_characters=MAX_CHARACTERS)
    except ValueError:
        raise InputError("The document could not be read within the limits. Remove passwords or save as DOCX or UTF-8/UTF-16 TXT. Scanned images need OCR.") from None
    return pasted_text(value)


def formatted_result(result):
    """Display the original calibrated output, never renormalize or infer a source."""
    require(result.get("model_id") == MODEL_ID, "wrong model returned a result")
    require(result.get("accepted_family") is None and result.get("accepted_model") is None and
            result.get("source_assertions") == {"valid": False, "family_enabled": False, "model_enabled": False}, "source assertions unexpectedly enabled")
    binary = result["probabilities"]
    families, leaves = result["family_probabilities"], result["model_probabilities"]
    require(set(binary) == {"human", "AI"} and len(families) == 7 and len(leaves) == 25 and
            len({r["label"] for r in families}) == 7 and len({r["label"] for r in leaves}) == 25, "probability catalog differs")
    values = list(binary.values()) + [r["probability"] for r in families + leaves]
    require(all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in values), "nonfinite probability result")
    require(abs(sum(binary.values()) - 1) < 1e-8 and abs(sum(r["probability"] for r in families) - 1) < 1e-8 and
            abs(sum(r["probability"] for r in leaves) - 1) < 1e-8, "probability mass differs")
    for family in families:
        require(abs(sum(r["probability"] for r in leaves if r["family"] == family["label"]) - family["probability"]) < 1e-8, "hierarchical probability mass differs")
    quality = result["quality"]
    require(quality["encoder_coverage"] == 1, "incomplete encoder coverage")
    supported = quality["core_length_supported"] and quality["language"] == "en"
    expected = "AI-generated" if supported and binary["AI"] >= AI_CUTOFF else "likely human" if supported and binary["human"] >= HUMAN_CUTOFF else "inconclusive"
    require(result["status"] == expected, "decision differs from frozen language/length/cutoff policy")
    status = {"AI-generated": "Likely AI-generated", "likely human": "Likely human-written", "inconclusive": "Inconclusive"}[result["status"]]
    binary_rows = [["Human-written", f"{100*binary['human']:.2f}%"], ["AI-generated / substantively AI-edited", f"{100*binary['AI']:.2f}%"]]
    family_rows = [[r["label"], f"{100*r['probability']:.2f}%"] for r in sorted(families, key=lambda r: -r["probability"])]
    leaf_rows = [[r["label"], r["family"], f"{100*r['probability']:.2f}%"] for r in sorted(leaves, key=lambda r: -r["probability"])]
    notes = "\n".join(str(w) for w in result.get("warnings", []))
    quality_text = f"{quality['word_like_tokens']} word-like units · {quality['encoder_tokens']} encoder tokens · {quality['window_counts']} windows · full text covered · language: {quality['language']}"
    return status, result["conclusion"], binary_rows, family_rows, leaf_rows, notes, quality_text
