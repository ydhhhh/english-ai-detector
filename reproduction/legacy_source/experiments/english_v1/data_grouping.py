"""Deterministic leakage grouping and bounded lexical near-duplicate search."""
from __future__ import annotations

import hashlib
import re
import unicodedata
import zlib
from collections import defaultdict

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        i, j = self.find(i), self.find(j)
        if i == j:
            return
        if self.size[i] < self.size[j]:
            i, j = j, i
        self.parent[j] = i
        self.size[i] += self.size[j]


def shingle_set(text):
    words = TOKEN.findall(normalized(text))[:3000]
    return {" ".join(words[k:k+5]) for k in range(max(0, len(words)-4))}


def near_group(records, uf, get_text, seed=20261017):
    """32 permutations, 8 x 4 LSH; exact lexical Jaccard >= .85 verifies hits.

    Very large collision buckets are conservatively unioned rather than silently
    truncating candidate retrieval. This can over-group; it cannot authenticate
    semantic paraphrases. Texts >3000 tokens use the documented prefix window.
    """
    rng = np.random.default_rng(seed)
    aa = rng.integers(1, 2**32-1, size=32, dtype=np.uint64)
    bb = rng.integers(0, 2**32-1, size=32, dtype=np.uint64)
    buckets = defaultdict(list)
    cache = {}
    examined = found = oversized = eligible = 0
    examples = []
    for i, rec in enumerate(records):
        if not rec["eligible"]:
            continue
        text = get_text(i)
        shingles = shingle_set(text)
        if len(shingles) < 20:
            continue
        eligible += 1
        values = np.fromiter((zlib.crc32(x.encode()) for x in shingles), dtype=np.uint64)
        sig = (((values[:, None] * aa + bb) & np.uint64(0xffffffff)).min(axis=0).astype(np.uint32))
        candidates = set()
        for band in range(8):
            key = (band, sig[band*4:band*4+4].tobytes())
            bucket = buckets[key]
            if len(bucket) >= 100:
                # Conservative grouping is preferable to an unrecorded search cap.
                uf.union(i, bucket[0])
                oversized += 1
            else:
                candidates.update(bucket)
            bucket.append(i)
        for j in candidates:
            if uf.find(i) == uf.find(j):
                continue
            other = cache.get(j)
            if other is None:
                other = shingle_set(get_text(j))
            examined += 1
            inter = len(shingles & other)
            similarity = inter / max(1, len(shingles) + len(other) - inter)
            if similarity >= .85:
                uf.union(i, j)
                found += 1
                if len(examples) < 100:
                    examples.append({"a": rec["id"], "b": records[j]["id"], "jaccard": similarity})
        if len(cache) >= 4000:
            cache.clear()
        cache[i] = shingles
        if eligible % 20000 == 0:
            print(f"near grouping: {eligible:,} texts, {found:,} verified edges", flush=True)
    return {"method": "32-minhash,8-bands,4-rows; verified word5gram Jaccard>=0.85",
            "seed": seed, "documents": eligible, "candidate_pairs_verified": examined,
            "near_edges": found, "conservative_large_bucket_unions": oversized,
            "examples": examples, "exhaustive_semantic_deduplication": False,
            "token_window": 3000}
