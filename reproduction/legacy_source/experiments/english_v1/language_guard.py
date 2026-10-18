"""Shared deployment language/length scope, without changing model scores.

Finite language sampling is an abstention heuristic, not proof that every span
is English. Both offline operating-decision audits and live inference use it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

RULE_VERSION = "english_scope_v1"
METHOD = "langid1.1.6:full<=6000chars,otherwise head/middle/tail2000chars+ASCII-letter guard; finite sampling cannot certify every span"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LanguageGuard:
    def __init__(self, identifier=None):
        self.identifier = identifier

    def check(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("nonempty text required")
        if self.identifier is None:
            vendor = Path(__file__).resolve().parents[2] / "research/english_v1_data/vendor"
            if vendor.is_dir() and str(vendor) not in sys.path:
                sys.path.insert(0, str(vendor))
            from langid.langid import LanguageIdentifier, model
            self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        if len(text) <= 6000:
            segments = [text]
        else:
            center = len(text) // 2
            segments = [text[:2000], text[center-1000:center+1000], text[-2000:]]
        identified = [(str(lang), float(prob)) for lang, prob in
                      (self.identifier.classify(s) for s in segments)]
        all_english = all(lang == "en" and prob >= .8 for lang, prob in identified)
        language = "en" if all_english else next(
            (lang for lang, prob in identified if lang != "en" or prob < .8), "und")
        letter_count = ascii_letters = 0
        for character in text:
            if character.isalpha():
                letter_count += 1
                ascii_letters += int(character.isascii())
        latin = ascii_letters / max(letter_count, 1)
        return {"supported": bool(all_english and latin >= .9),
                "word_like_tokens": len(re.findall(r"\b\w+\b", text)),
                "identified": language,
                "probability": min(prob for _, prob in identified),
                "segments": [{"language": lang, "probability": prob} for lang, prob in identified],
                "ascii_letter_ratio": latin, "method": METHOD, "rule_version": RULE_VERSION}


def scope_arrays(rows, input_sha256, cache_dir=Path(".cache/english_scope")):
    """Cache only row identifiers and scope evidence, never input text/labels.

    The caller must establish the corpus role/frozen-final gate before reading
    rows. Cache identity includes the immutable input and this exact rule code.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise ValueError("input SHA256 required")
    code_sha = _sha(__file__)
    key = hashlib.sha256((input_sha256 + code_sha).encode()).hexdigest()
    path = Path(cache_dir) / (key + ".npz")
    ids = np.asarray([str(row["id"]) for row in rows])
    expected_keys = {"ids", "language_supported", "scope_word_like_tokens", "language_probability", "ascii_letter_ratio", "language_rule_sha256"}
    if path.exists():
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta.get("input_sha256") != input_sha256 or meta.get("output_sha256") != _sha(path):
            raise ValueError("language cache integrity mismatch")
        with np.load(path, allow_pickle=False) as value:
            if set(value.files) != expected_keys or not np.array_equal(value["ids"], ids):
                raise ValueError("language cache row identity mismatch")
            result = {k: value[k] for k in value.files if k != "ids"}
        if str(result["language_rule_sha256"].item()) != code_sha:
            raise ValueError("language cache rule identity mismatch")
        if any(result[k].shape != (len(rows),) for k in expected_keys - {"ids", "language_rule_sha256"}):
            raise ValueError("language cache evidence shape mismatch")
        return result
    guard = LanguageGuard()
    evidence = [guard.check(row["text"]) for row in rows]
    result = {"language_supported": np.asarray([x["supported"] for x in evidence], dtype=bool),
              "scope_word_like_tokens": np.asarray([x["word_like_tokens"] for x in evidence], dtype=np.int64),
              "language_probability": np.asarray([x["probability"] for x in evidence], dtype=np.float64),
              "ascii_letter_ratio": np.asarray([x["ascii_letter_ratio"] for x in evidence], dtype=np.float64),
              "language_rule_sha256": np.asarray(code_sha)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, ids=ids, **result)
    temporary.replace(path)
    path.with_suffix(".json").write_text(json.dumps({"rule_version": RULE_VERSION,
        "method": METHOD, "rule_sha256": code_sha, "input_sha256": input_sha256,
        "output_sha256": _sha(path),
        "rows": len(rows), "supported_rows": int(result["language_supported"].sum()),
        "contains_text_or_labels": False}, indent=2) + "\n")
    return result
