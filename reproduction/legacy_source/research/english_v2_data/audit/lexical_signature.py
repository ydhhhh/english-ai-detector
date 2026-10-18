"""Pure lexical primitives matching english_v1.data_grouping exactly.

This module performs no I/O, candidate retrieval, grouping, or role assignment.
The caller must retain the original eligibility condition ``shingle_count >= 20``.
The 32 uint32 values are intended for eight consecutive four-value LSH bands.
Only the first 3,000 tokens are represented; this is lexical, not semantic dedup.
CRC32 collisions are retained because the original implementation uses CRC32.
An empty shingle set has an all-uint32-max sentinel and exact Jaccard zero.
"""
from __future__ import annotations

import re
import unicodedata
import zlib
from functools import lru_cache

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
TOKEN_WINDOW = 3000
N_PERMUTATIONS = 32
N_BANDS = 8
ROWS_PER_BAND = 4
MIN_ELIGIBLE_SHINGLES = 20
DEFAULT_SEED = 20261017


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingle_set(text: str) -> set[str]:
    words = TOKEN.findall(normalized(text))[:TOKEN_WINDOW]
    return {" ".join(words[k:k + 5]) for k in range(max(0, len(words) - 4))}


@lru_cache(maxsize=16)
def _coefficients(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    aa = rng.integers(1, 2**32 - 1, size=N_PERMUTATIONS, dtype=np.uint64)
    bb = rng.integers(0, 2**32 - 1, size=N_PERMUTATIONS, dtype=np.uint64)
    aa.flags.writeable = False
    bb.flags.writeable = False
    return aa, bb


def signature(text: str, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, int]:
    """Return the original 32-affine-CRC32 MinHash and unique shingle count.

    Counts below 20 are mathematically representable but must not enter the
    caller's original near-duplicate retrieval. The empty-set sentinel is an
    explicit extension for such callers; original near_group skips those rows.
    """
    shingles = shingle_set(text)
    count = len(shingles)
    if not count:
        return np.full(N_PERMUTATIONS, np.uint32(0xffffffff), dtype=np.uint32), 0
    aa, bb = _coefficients(seed)
    values = np.fromiter((zlib.crc32(x.encode()) for x in shingles), dtype=np.uint64)
    sig = (((values[:, None] * aa + bb) & np.uint64(0xffffffff))
           .min(axis=0).astype(np.uint32))
    return sig, count


def exact_jaccard(text_a: str, text_b: str) -> float:
    """Verify the same prefix word-five-gram Jaccard used by near_group."""
    aa, bb = shingle_set(text_a), shingle_set(text_b)
    intersection = len(aa & bb)
    return intersection / max(1, len(aa) + len(bb) - intersection)


def synthetic_selfcheck() -> dict:
    """Check original function locals on synthetic inputs, without corpus I/O.

    A temporary Python trace captures the *actual* original near_group's sig
    local after its computation. This avoids a second copied formula serving
    as its own oracle. The previous tracing function is restored afterwards.
    """
    import importlib
    import sys
    from pathlib import Path

    project = Path(__file__).resolve().parents[3]
    inserted = str(project) not in sys.path
    if inserted:
        sys.path.insert(0, str(project))
    try:
        original = importlib.import_module("experiments.english_v1.data_grouping")
    finally:
        if inserted:
            sys.path.remove(str(project))

    checks: list[str] = []

    def require(condition, label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    def capture_original(text, seed):
        captured = []
        code = original.near_group.__code__
        previous_trace = sys.gettrace()

        def trace(frame, event, arg):
            if frame.f_code is code and "sig" in frame.f_locals and not captured:
                captured.append(frame.f_locals["sig"].copy())
            return trace

        try:
            sys.settrace(trace)
            result = original.near_group(
                [{"id": "synthetic", "eligible": True}],
                original.UnionFind(1), lambda _: text, seed=seed)
        finally:
            sys.settrace(previous_trace)
        return captured, result

    long_text = " ".join(f"word{i}" for i in range(3070))
    samples = [
        " ".join(f"token{i}" for i in range(50)),
        "ＦＵＬＬＷＩＤＴＨ\t CASE  can't\n" + " ".join(f"x{i}" for i in range(60)),
        long_text,
    ]
    for i, text in enumerate(samples):
        require(normalized(text) == original.normalized(text), f"normalization_{i}")
        require(shingle_set(text) == original.shingle_set(text), f"shingles_{i}")
        for seed in (DEFAULT_SEED, 7):
            sig, count = signature(text, seed)
            captured, report = capture_original(text, seed)
            require(len(captured) == 1 and np.array_equal(sig, captured[0]),
                    f"original_signature_{i}_seed{seed}")
            require(sig.dtype == np.uint32 and sig.shape == (32,),
                    f"signature_layout_{i}_seed{seed}")
            require(count == len(original.shingle_set(text)) and report["documents"] == 1,
                    f"original_count_{i}_seed{seed}")

    for i, text in enumerate(("", "\u77ed\u6587\u672c\u6ca1\u6709\u7a7a\u683c", "one two three four",
                              "one two three four five", " ".join(f"s{x}" for x in range(23)))):
        sig, count = signature(text)
        captured, report = capture_original(text, DEFAULT_SEED)
        require(count == len(original.shingle_set(text)), f"short_count_{i}")
        require(count < MIN_ELIGIBLE_SHINGLES and not captured and report["documents"] == 0,
                f"original_short_skipped_{i}")
        if count == 0:
            require(np.all(sig == np.uint32(0xffffffff)), f"empty_sentinel_{i}")

    a = " ".join(f"j{i}" for i in range(80))
    b = a.replace("j35", "different35")
    aset, bset = original.shingle_set(a), original.shingle_set(b)
    inter = len(aset & bset)
    require(exact_jaccard(a, b) == inter / max(1, len(aset) + len(bset) - inter),
            "jaccard_original_formula")
    require(exact_jaccard(a, a) == 1.0, "jaccard_identity_nonempty")
    require(exact_jaccard("", "") == 0.0, "jaccard_both_empty_original_semantics")
    prefix = " ".join(f"word{i}" for i in range(3000))
    require(shingle_set(long_text) == shingle_set(prefix), "first3000_shingles")
    require(np.array_equal(signature(long_text)[0], signature(prefix)[0]), "first3000_signature")
    changed_inside = " ".join([*(f"word{i}" for i in range(2999)), "replacement2999"])
    require(shingle_set(changed_inside) != shingle_set(prefix), "token2999_included")
    return {"passed": True, "checks": len(checks), "check_names": checks,
            "corpus_read": False, "reference": "experiments.english_v1.data_grouping.near_group",
            "seed": DEFAULT_SEED, "permutations": N_PERMUTATIONS,
            "bands": N_BANDS, "rows_per_band": ROWS_PER_BAND,
            "minimum_eligible_shingles": MIN_ELIGIBLE_SHINGLES, "token_window": TOKEN_WINDOW}


if __name__ == "__main__":
    import json
    print(json.dumps(synthetic_selfcheck(), sort_keys=True))
