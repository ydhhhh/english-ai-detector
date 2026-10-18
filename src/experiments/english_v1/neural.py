"""Local-only hierarchical English detector, full-document windows and training.

Training opens development splits only. Final evaluation is a separate predict
command. Raw logits are deliberately uncalibrated; calibration belongs to the
independent calibration split and downstream evaluation code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import signal
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from experiments.english_v1 import runtime  # noqa: F401: local dependencies first
from experiments.english_v1.catalog import (
    FAMILIES, NOMINAL_MODELS, MODEL_LABELS, MODEL_FAMILIES, VERSION_INDICES,
    canonical_family, catalog_dict,
)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class NeuralConfig:
    head_hidden: int = 256
    dropout: float = .1
    max_length: int = 512
    stride: int = 448
    family_loss_weight: float = .3
    version_loss_weight: float = .3
    attention_implementation: str = "sdpa"


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_hash(path):
    # Semantic tokenization hash survives save_pretrained JSON formatting changes.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if not getattr(tok, "is_fast", False):
        raise ValueError("reviewed token caches require a fast local tokenizer")
    backend = json.loads(tok.backend_tokenizer.to_str())
    # The project always supplies padding=False/truncation=False for the
    # complete stream. Fast tokenizers mutate these runtime settings when
    # called, so save_pretrained may serialize different inert defaults.
    backend["padding"] = None
    backend["truncation"] = None
    identity = {"backend": backend,
                "special_tokens": {k: str(v) for k,v in tok.special_tokens_map.items()},
                "padding_side": "right", "add_special_tokens": False}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def special_token_template(tokenizer):
    """Read reviewed single-sequence postprocessing without re-encoding IDs."""
    processor = json.loads(tokenizer.backend_tokenizer.to_str()).get("post_processor")
    if not isinstance(processor, dict):
        raise ValueError("local tokenizer has no reviewed single-sequence postprocessor")
    kind = processor.get("type")
    if kind == "TemplateProcessing":
        prefix, suffix = [], []; seen = False
        for operation in processor.get("single", []):
            if set(operation) == {"Sequence"}:
                if seen or operation["Sequence"].get("id") != "A":
                    raise ValueError("expected one single sequence A in tokenizer template")
                seen = True
            elif set(operation) == {"SpecialToken"}:
                key = operation["SpecialToken"]["id"]
                ids = processor["special_tokens"][key]["ids"]
                (suffix if seen else prefix).extend(ids)
            else:
                raise ValueError("unsupported tokenizer postprocessing operation")
        if not seen:
            raise ValueError("tokenizer single template lacks sequence A")
    elif kind in {"BertProcessing", "RobertaProcessing"}:
        prefix, suffix = [processor["cls"][1]], [processor["sep"][1]]
    else:
        raise ValueError(f"unreviewed tokenizer postprocessor: {kind}")
    if any(not isinstance(v, int) or v < 0 for v in prefix+suffix):
        raise ValueError("invalid tokenizer special-token IDs")
    if tokenizer.pad_token_id is None:
        raise ValueError("local tokenizer must declare its padding token ID")
    return tuple(prefix), tuple(suffix)


def windows(n_tokens: int, payload: int, stride: int):
    """Contiguous windows and weights for *new* token coverage, including tail."""
    if n_tokens < 0 or payload < 1 or not 1 <= stride <= payload:
        raise ValueError("require n_tokens>=0 and 1<=stride<=payload")
    if not n_tokens:
        return [(0, 0, 1)]
    out = []; covered = 0
    for start in range(0, n_tokens, stride):
        end = min(start + payload, n_tokens)
        novel = end - covered
        if novel > 0:
            out.append((start, end, novel)); covered = end
        if end == n_tokens:
            break
    return out


def _bool(row, key):
    v = row.get(key, False)
    if v is None:
        return False
    if v not in (True, False, 0, 1) or isinstance(v, str):
        raise ValueError(f"{key} must be an explicit boolean")
    return bool(v)


def targets(rows):
    """Fine supervision requires trusted labels; weak labels never become Other."""
    y = np.asarray([r["y_ai"] for r in rows])
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_ai must be human=0 / AI=1")
    y = y.astype(np.int64)
    family = np.full(len(rows), -100, np.int64)
    version = np.full(len(rows), -100, np.int64)
    model_true = np.asarray([str(r.get("model") or "") for r in rows])
    family_true = np.asarray([canonical_family(r.get("family")) for r in rows])
    for i, r in enumerate(rows):
        mf, mm = _bool(r, "mask_family"), _bool(r, "mask_model")
        if not y[i]:
            if mf or mm:
                raise ValueError("human rows must not supervise family or version")
            continue
        if mf:
            f = family_true[i]
            if f not in FAMILIES:
                raise ValueError(f"trusted family label outside explicit catalog: {f}")
            family[i] = FAMILIES.index(f)
        if mm:
            m = model_true[i]
            if not mf or m not in NOMINAL_MODELS or MODEL_FAMILIES[MODEL_LABELS.index(m)] != family_true[i]:
                raise ValueError("trusted nominal version requires matching trusted family")
            version[i] = MODEL_LABELS.index(m)
        elif mf and family_true[i] == "Other":
            version[i] = MODEL_LABELS.index("other_family")
        elif mf and _bool(r, "out_of_model_catalog"):
            version[i] = MODEL_LABELS.index(f"{family_true[i]}:other_version")
    mapped = np.asarray([MODEL_LABELS[v] if v >= 0 else model_true[i] for i, v in enumerate(version)])
    return {"binary": y, "family": family, "version": version,
            "family_true": family_true, "model_true": mapped,
            "raw_model_true": model_true, "mask_family": family >= 0, "mask_model": version >= 0}


def training_weights(rows, *, mode="legacy_v1"):
    """Train-only moderate y/source weighting; no replacement or tiny-source blowup."""
    keys = [(int(r["y_ai"]), str(r.get("source") or "")) for r in rows]
    counts = Counter(keys)
    # Two stages avoid balancing the artificial Cartesian product of source and y.
    y_count = {y: sum(k[0] == y for k in keys) for y in (0, 1)}
    within = {}
    for y in (0, 1):
        active = [count for (yy, _), count in counts.items() if yy == y]
        within[y] = float(np.median(active)) if active else 1.
    weights = np.asarray([math.sqrt(len(rows) / max(2 * y_count[k[0]], 1)) *
                          math.sqrt(within[k[0]] / counts[k]) for k in keys])
    weights = np.clip(weights, .25, 4.)
    weights = weights / weights.mean()
    if mode in {"class_normalized_v2", "class_domain_v3"}:
        labels = np.asarray([k[0] for k in keys])
        for y in (0, 1):
            m = labels == y
            if not m.any():
                raise ValueError("class_normalized_v2 requires both human and AI training rows")
            weights[m] *= (len(rows)/2) / weights[m].sum()
        if mode == "class_domain_v3":
            writing = np.asarray([str(r.get("domain") or "").casefold() == "wp" for r in rows])
            for y in (0, 1):
                for domain_mask, fraction in ((writing, .30), (~writing, .70)):
                    selected = (labels == y) & domain_mask
                    if not selected.any():
                        raise ValueError("class_domain_v3 requires WP and replay rows in each binary class")
                    weights[selected] *= (len(rows)/2 * fraction) / weights[selected].sum()
    elif mode != "legacy_v1":
        raise ValueError("unknown training weight mode")
    return weights


def training_weight_summary(rows, weights, *, mode):
    """Only train labels/source metadata; record effective binary class masses."""
    summary = {"mode":mode,"rows":len(rows),"total_weight":float(np.sum(weights)),
               "by_y":{},"by_source_y":[],"fit_split":"train"}
    labels = np.asarray([int(r["y_ai"]) for r in rows])
    sources = np.asarray([str(r.get("source") or "") for r in rows])
    for y in (0,1):
        m = labels == y
        summary["by_y"][str(y)] = {"rows":int(m.sum()),"weight_mass":float(weights[m].sum()),
            "mean_weight":float(weights[m].mean()) if m.any() else None,
            "fraction_of_weight":float(weights[m].sum()/weights.sum())}
    for source,y in sorted({(str(r.get("source") or ""),int(r["y_ai"])) for r in rows}):
        m = (sources == source) & (labels == y)
        summary["by_source_y"].append({"source":source,"y_ai":y,"rows":int(m.sum()),
            "weight_mass":float(weights[m].sum()),"mean_weight":float(weights[m].mean())})
    if mode == "class_domain_v3":
        writing=np.asarray([str(r.get("domain") or "").casefold()=="wp" for r in rows])
        summary["domain_mass_rule"]="Within each class: WP30%, non-WP replay70%; relative source weighting retained inside each cell. Batch-normalized loss does not imply exact optimization prior."
        summary["by_domain_class"]=[{"domain":"wp" if is_wp else "replay_non_wp","y_ai":y,
            "rows":int(((labels==y)&(writing==is_wp)).sum()),
            "weight_mass":float(weights[(labels==y)&(writing==is_wp)].sum())}
            for y in (0,1) for is_wp in (False,True)]
    return summary


class TokenCache:
    """Immutable full token stream, keyed by data content, row IDs and tokenizer."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        self.offsets = np.load(self.directory / "offsets.npy", mmap_mode="r")
        count = int(self.offsets[-1])
        self.flat = np.memmap(self.directory / "tokens.int32", dtype="int32", mode="r", shape=(count,)) if count else np.empty(0, np.int32)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        return self.flat[int(self.offsets[i]):int(self.offsets[i+1])]

    @classmethod
    def build(cls, rows, tokenizer, directory, *, data_sha256, tokenizer_sha256, progress=None):
        ids_digest = hashlib.sha256("\n".join(str(r["id"]) for r in rows).encode()).hexdigest()
        text_digest = hashlib.sha256()
        for row in rows:
            b = row["text"].encode(); text_digest.update(len(b).to_bytes(8, "little")); text_digest.update(b)
        identity = {"schema": 1, "data_sha256": data_sha256, "tokenizer_sha256": tokenizer_sha256,
                    "ids_sha256": ids_digest, "texts_sha256": text_digest.hexdigest(), "rows": len(rows),
                    "add_special_tokens": False, "truncation": False}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        destination = Path(directory) / key
        if (destination / "metadata.json").exists():
            cached = cls(destination)
            if any(cached.metadata.get(k) != v for k, v in identity.items()):
                raise ValueError("token cache identity mismatch")
            if (destination / "tokens.int32").stat().st_size != int(cached.offsets[-1]) * 4:
                raise ValueError("token cache byte count mismatch")
            return cached
        destination.mkdir(parents=True, exist_ok=True)
        offsets = [0]; started = time.monotonic(); last = started
        with (destination / "tokens.int32.part").open("wb") as stream:
            for start in range(0, len(rows), 128):
                batch = tokenizer([r["text"] for r in rows[start:start+128]], add_special_tokens=False,
                                  truncation=False, padding=False, verbose=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
                for ids in batch:
                    a = np.asarray(ids, dtype=np.int32)
                    stream.write(a.tobytes()); offsets.append(offsets[-1] + len(a))
                now = time.monotonic()
                if progress and (now-last >= 30 or start+128 >= len(rows)):
                    progress({"phase": "tokenize", "rows": min(start+128, len(rows)), "total_rows": len(rows),
                              "tokens": offsets[-1], "seconds": now-started}); last = now
        (destination / "tokens.int32.part").replace(destination / "tokens.int32")
        np.save(destination / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        dump_json(destination / "metadata.json", {**identity, "total_tokens": offsets[-1], "complete_stream": True})
        return cls(destination)


class NeuralDetector(nn.Module):
    def __init__(self, encoder, config: NeuralConfig | dict | None = None, tokenizer=None):
        super().__init__()
        self.config = NeuralConfig(**config) if isinstance(config, dict) else (config or NeuralConfig())
        if self.config.max_length < 4 or self.config.stride < 1 or self.config.head_hidden < 1 or not 0 <= self.config.dropout < 1:
            raise ValueError("invalid maximum length or dropout")
        self.encoder = encoder
        self.tokenizer = tokenizer
        self._special_template = special_token_template(tokenizer) if tokenizer is not None else None
        hidden = int(encoder.config.hidden_size)
        self.features = nn.Sequential(nn.LayerNorm(hidden*2), nn.Dropout(self.config.dropout),
                                      nn.Linear(hidden*2, self.config.head_hidden), nn.GELU(), nn.Dropout(self.config.dropout))
        self.binary_head = nn.Linear(self.config.head_hidden, 2)
        self.family_head = nn.Linear(self.config.head_hidden, len(FAMILIES))
        self.version_heads = nn.ModuleDict({f: nn.Linear(self.config.head_hidden, len(ix)) for f, ix in VERSION_INDICES.items()})

    @classmethod
    def from_pretrained_local(cls, path, config=None, **architecture):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        path = Path(path).resolve()
        if not (path / "model.safetensors").is_file():
            raise ValueError("expected one local model.safetensors; remote/custom code is disabled")
        ec = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        if ec.model_type not in {"bert", "modernbert", "roberta", "deberta-v2"}:
            raise ValueError(f"encoder architecture is outside the reviewed allowlist: {ec.model_type}")
        cfg = NeuralConfig(**{**(asdict(config) if isinstance(config, NeuralConfig) else (config or {})), **architecture})
        if ec.model_type == "modernbert":
            # Older Transformers builds compile reference kernels by default.
            ec.reference_compile = False
        context = int(getattr(ec, "max_position_embeddings", cfg.max_length))
        if cfg.max_length > context:
            raise ValueError("requested maximum length exceeds the encoder configuration")
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        tok.padding_side = "right"
        encoder = AutoModel.from_pretrained(str(path), config=ec, local_files_only=True,
                                            trust_remote_code=False, use_safetensors=True,
                                            attn_implementation=cfg.attention_implementation)
        return cls(encoder, cfg, tok)

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        weight = attention_mask.unsqueeze(-1).to(encoded.dtype)
        mean = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)
        h = self.features(torch.cat([encoded[:, 0], mean], dim=-1))
        binary = self.binary_head(h); family = self.family_head(h)
        version = torch.empty((len(h), len(MODEL_LABELS)), dtype=h.dtype, device=h.device)
        for f, indices in VERSION_INDICES.items():
            version[:, indices] = self.version_heads[f](h)
        return {"binary_logits": binary, "family_logits": family, "version_logits": version}

    @staticmethod
    def probabilities(logits):
        pb = logits["binary_logits"].float().softmax(-1)
        pf = logits["family_logits"].float().softmax(-1)
        pv = torch.zeros_like(logits["version_logits"], dtype=torch.float32)
        pm = torch.zeros_like(pv)
        for i, f in enumerate(FAMILIES):
            indices = VERSION_INDICES[f]
            pv[:, indices] = logits["version_logits"][:, indices].float().softmax(-1)
            pm[:, indices] = pf[:, i, None] * pv[:, indices]
        return {"p_binary": pb, "p_ai": pb[:, 1], "p_family": pf,
                "p_version_given_family": pv, "p_model": pm}

    def make_batch(self, sequences, device):
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for text windows")
        prefix, suffix = self._special_template
        prepared = [list(prefix) + list(map(int, ids)) + list(suffix) for ids in sequences]
        if not prepared or any(len(ids) > self.config.max_length for ids in prepared):
            raise ValueError("empty batch or a token window exceeds max_length; no truncation is performed")
        width = max(len(ids) for ids in prepared)
        input_ids = torch.full((len(prepared), width), int(self.tokenizer.pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, ids in enumerate(prepared):
            input_ids[i, :len(ids)] = torch.as_tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1
        return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}

    @property
    def payload(self):
        specials = sum(map(len, self._special_template))
        capacity = self.config.max_length - specials
        if capacity < 1:
            raise ValueError("max_length cannot accommodate tokenizer special tokens")
        return capacity

    @torch.inference_mode()
    def predict_cache(self, cache, *, batch_size=16, device="cpu", precision="fp32", progress=None):
        self.eval(); device = torch.device(device)
        payload = self.payload; stride = min(self.config.stride, payload)
        totals = {"binary_logits": np.zeros((len(cache), 2), np.float64),
                  "family_logits": np.zeros((len(cache), len(FAMILIES)), np.float64),
                  "version_logits": np.zeros((len(cache), len(MODEL_LABELS)), np.float64)}
        counts = np.zeros(len(cache), np.int64); covered = np.zeros(len(cache), np.int64)
        token_counts = np.diff(np.asarray(cache.offsets)).astype(np.int64)
        sequence_batch = []; routing = []; last = time.monotonic()
        def flush():
            nonlocal sequence_batch, routing
            with autocast(device, precision):
                out = self(**self.make_batch(sequence_batch, device))
            arrays = {k: v.float().cpu().numpy() for k,v in out.items()}
            for j, (doc, weight) in enumerate(routing):
                for k in totals:
                    totals[k][doc] += arrays[k][j] * weight
                counts[doc] += 1
            sequence_batch = []; routing = []
        for i in range(len(cache)):
            stream = cache[i]
            for start, end, novel in windows(len(stream), payload, stride):
                sequence_batch.append(stream[start:end]); routing.append((i, novel))
                if len(sequence_batch) == batch_size:
                    flush()
            covered[i] = len(stream)
            if progress and time.monotonic()-last >= 30:
                progress({"phase": "predict_windows", "documents": i+1, "total_documents": len(cache)}); last = time.monotonic()
        if sequence_batch:
            flush()
        denom = np.maximum(token_counts, 1)
        result = {k: (v / denom[:, None]).astype(np.float32) for k,v in totals.items()}
        result.update({"encoder_tokens": token_counts, "covered_encoder_tokens": covered,
                       "window_counts": counts, "encoder_coverage": np.ones(len(cache), np.float32)})
        probability = self.probabilities({k: torch.from_numpy(result[k]) for k in totals})
        result.update({k: v.numpy() for k,v in probability.items()})
        return result

    def predict_raw_scores(self, texts, *, batch_size=16, device=None, precision="fp32"):
        """Uncalibrated document logits; version columns normalize per family.

        Every token is covered. This small-input API keeps token streams in RAM;
        large file jobs should use TokenCache and predict_cache instead.
        """
        if isinstance(texts, str):
            raise ValueError("texts must be a sequence of documents, not one bare string")
        values = list(texts)
        if not values or any(not isinstance(s, str) for s in values):
            raise ValueError("texts must contain at least one string")
        encoded = self.tokenizer(values, add_special_tokens=False, truncation=False,
                                 padding=False, verbose=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        class MemoryCache:
            def __len__(self): return len(encoded)
            def __getitem__(self, i): return np.asarray(encoded[i], dtype=np.int32)
        cache = MemoryCache()
        cache.offsets = np.cumsum([0] + [len(v) for v in encoded], dtype=np.int64)
        result = self.predict_cache(cache, batch_size=batch_size,
                    device=device or next(self.parameters()).device, precision=precision)
        return {"binary": result["binary_logits"], "family": result["family_logits"],
                "version": result["version_logits"], **result}

    predict_raw = predict_raw_scores

    def save_local(self, destination, *, training_config=None, provenance=None):
        from safetensors.torch import save_file
        destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / "detector.safetensors.part"
        # Explicit copies avoid shared-storage ambiguities and keep parameters safe.
        save_file({k: v.detach().cpu().contiguous().clone() for k,v in self.state_dict().items()}, str(temporary))
        temporary.replace(destination / "detector.safetensors")
        encoder_dir = destination / "encoder_config"; encoder_dir.mkdir(exist_ok=True)
        self.encoder.config.save_pretrained(encoder_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(destination / "tokenizer")
        dump_json(destination / "detector.json", {"architecture": asdict(self.config),
            "family_labels": FAMILIES, "model_labels": MODEL_LABELS, "model_families": MODEL_FAMILIES,
            **catalog_dict(),
            "probability_scope": "P(AI), P(family|AI), P(version|family,AI); model=P(family|AI)*P(version|family,AI)",
            "weights_sha256": sha256_file(destination / "detector.safetensors"),
            "training_config": training_config or {}, "provenance": provenance or {}})

    @classmethod
    def load_local(cls, path):
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        path = Path(path); metadata = json.loads((path / "detector.json").read_text())
        if metadata["family_labels"] != FAMILIES or metadata["model_labels"] != MODEL_LABELS:
            raise ValueError("checkpoint class order differs from current schema")
        if sha256_file(path / "detector.safetensors") != metadata["weights_sha256"]:
            raise ValueError("checkpoint weights SHA-256 mismatch")
        ec = AutoConfig.from_pretrained(path / "encoder_config", local_files_only=True, trust_remote_code=False)
        if ec.model_type not in {"bert", "modernbert", "roberta", "deberta-v2"}:
            raise ValueError("unreviewed checkpoint architecture")
        cfg = NeuralConfig(**metadata["architecture"])
        if ec.model_type == "modernbert":
            ec.reference_compile = False
        encoder = AutoModel.from_config(ec, trust_remote_code=False, attn_implementation=cfg.attention_implementation)
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "right"
        model = cls(encoder, cfg, tokenizer)
        model.load_state_dict(load_file(str(path / "detector.safetensors")), strict=True)
        return model


def autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError("precision must be fp32 or bf16")
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)


def loss_components(logits, y, family, version, sample_weights=None, config=None):
    cfg = config or NeuralConfig()
    weight = torch.ones(len(y), device=y.device) if sample_weights is None else sample_weights
    binary_each = F.cross_entropy(logits["binary_logits"].float(), y, reduction="none")
    binary = (binary_each * weight).sum() / weight.sum().clamp_min(1e-12)
    family_mask = family >= 0
    family_loss = logits["family_logits"].sum() * 0
    if family_mask.any():
        each = F.cross_entropy(logits["family_logits"][family_mask].float(), family[family_mask], reduction="none")
        family_loss = (each * weight[family_mask]).sum() / weight[family_mask].sum()
    version_sum = logits["version_logits"].sum() * 0; version_weight = weight.new_zeros(())
    for fi, f in enumerate(FAMILIES):
        m = (family == fi) & (version >= 0)
        if m.any():
            indices = VERSION_INDICES[f]
            lookup = torch.full((len(MODEL_LABELS),), -100, dtype=torch.long, device=y.device)
            lookup[indices] = torch.arange(len(indices), device=y.device)
            local_target = lookup[version[m]]
            if (local_target < 0).any():
                raise ValueError("version label does not belong to its family")
            each = F.cross_entropy(logits["version_logits"][m][:, indices].float(), local_target, reduction="none")
            version_sum = version_sum + (each * weight[m]).sum(); version_weight = version_weight + weight[m].sum()
    version_loss = version_sum / version_weight.clamp_min(1e-12)
    total = binary + cfg.family_loss_weight * family_loss + cfg.version_loss_weight * version_loss
    return total, {"binary": binary, "family": family_loss, "version": version_loss}


def read_records(path, purpose, maximum=None, seed=20261017):
    path = Path(path)
    stem = path.stem.casefold()
    if purpose != "predict" and ("test" in stem or "ood" in stem or ("challenge" in stem and not stem.startswith("dev_"))):
        raise ValueError("training/benchmark refuses final test or challenge files")
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    if purpose != "predict":
        allowed = {"train": {"train"}, "validation": {"validation", "dev"},
                   "dev_challenge": {"dev_challenge", "dev_challenge_source_domain"}}[purpose]
        observed = {r.get("split") for r in rows if r.get("split") is not None}
        if observed and not observed <= allowed:
            raise ValueError(f"unexpected {purpose} split values: {observed}")
    if any(not isinstance(r.get("text"), str) or not r.get("id") or r.get("y_ai") not in (0, 1) for r in rows):
        raise ValueError("each row requires text, id and binary truth")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate row IDs")
    if maximum and len(rows) > maximum:
        # Stable order within labels and near-preserved prevalence without prefix bias.
        rows = sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).digest())[:maximum]
    if not rows:
        raise ValueError("empty input split")
    return rows


def assert_disjoint(a, b):
    for key in ("id", "group_id", "normalized_sha256"):
        aa = {r[key] for r in a if r.get(key)}; bb = {r[key] for r in b if r.get(key)}
        if aa & bb:
            raise ValueError(f"development splits overlap on {key}")
    normalized = lambda r: hashlib.sha256(" ".join(r["text"].casefold().split()).encode()).digest()
    if {normalized(r) for r in a} & {normalized(r) for r in b}:
        raise ValueError("development splits contain identical normalized texts")


def export_predictions(model, rows, cache, destination, **predict_kwargs):
    result = model.predict_cache(cache, **predict_kwargs)
    result.update({"raw_binary": result["binary_logits"], "raw_family": result["family_logits"],
                   "raw_version": result["version_logits"]})
    labels = targets(rows)
    result.update({"ids": np.asarray([str(r["id"]) for r in rows]),
        "group_ids": np.asarray([str(r.get("group_id") or r["id"]) for r in rows]),
        "sources": np.asarray([str(r.get("source") or "") for r in rows]),
        "domains": np.asarray([str(r.get("domain") or "") for r in rows]),
        "lengths": np.asarray([len(re.findall(r"\b\w+\b", r["text"])) for r in rows], dtype=np.int64),
        "y_ai": labels["binary"], "family_true": labels["family_true"], "model_true": labels["model_true"],
        "raw_model_true": labels["raw_model_true"], "mask_family": labels["mask_family"], "mask_model": labels["mask_model"],
        "family_labels": np.asarray(FAMILIES), "model_labels": np.asarray(MODEL_LABELS),
        "model_families": np.asarray(MODEL_FAMILIES), "length_unit": np.asarray("word_like_tokens"),
        "calibrated": np.asarray(False)})
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **result)
    return result


def _device(name):
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this execution environment; run with authorized MPS access")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    return device


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _progress_factory(output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def progress(value):
        value = {"elapsed_seconds": round(time.monotonic()-started, 3), **value}
        line = json.dumps(value, ensure_ascii=False, allow_nan=False)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)
    return progress


def make_training_batch(model, cache, indices, rng, device):
    sequences = []
    for index in indices:
        stream = cache[int(index)]
        start = int(rng.integers(max(len(stream)-model.payload, 0)+1))
        sequences.append(stream[start:start+model.payload])
    return model.make_batch(sequences, device)


def _development_diagnostics(arrays, truth, mask=None):
    """Development-only ranking/attribution monitors, never a selection loss."""
    from sklearn.metrics import f1_score, roc_auc_score
    m = np.ones(len(truth["binary"]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    y = truth["binary"][m]; p = np.asarray(arrays["p_ai"],dtype=np.float64)[m]
    human = np.sort(p[y == 0]); ai = p[y == 1]
    threshold = None; fpr = None; recall = None
    if len(human):
        allowed = int(np.floor(.01*len(human)))
        threshold = float(np.nextafter(float(human[len(human)-allowed-1]), np.inf))
        fpr = float(np.mean(human >= threshold))
        recall = float(np.mean(ai >= threshold)) if len(ai) else None
    family_true = truth["family"][m]
    family_pred = arrays["p_family"][m].argmax(1)
    family_mask = family_true >= 0
    version_true = truth["version"][m]
    # Keep the entire 25-leaf argmax: predicting Other must remain an error for
    # an observed nominal class. Only the F1 averaging labels are restricted.
    nominal_indices = [MODEL_LABELS.index(name) for name in NOMINAL_MODELS]
    nominal_mask = np.isin(version_true, nominal_indices)
    version_pred = arrays["p_model"][m].argmax(1)
    return {
        "binary_AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "development_empirical_recall_at_human_FPR_1pct": recall,
        "development_empirical_human_FPR": fpr,
        "development_empirical_AI_score_threshold": threshold,
        "development_human_n": int(len(human)), "development_AI_n": int(len(ai)),
        "family_accuracy": float(np.mean(family_pred[family_mask] == family_true[family_mask])) if family_mask.any() else None,
        "family_supervised_n": int(family_mask.sum()),
        "nominal18_top1": float(np.mean(version_pred[nominal_mask] == version_true[nominal_mask])) if nominal_mask.any() else None,
        "nominal18_macro_f1": float(f1_score(version_true[nominal_mask], version_pred[nominal_mask],
            labels=nominal_indices, average="macro", zero_division=0)) if nominal_mask.any() else None,
        "nominal18_supervised_n": int(nominal_mask.sum()),
        "nominal18_present_classes": int(len(np.unique(version_true[nominal_mask]))),
        "documents": int(m.sum()),
    }


def _development_metrics(model, rows, cache, *, batch_size, device, precision, progress):
    arrays = model.predict_cache(cache, batch_size=batch_size, device=device, precision=precision, progress=progress)
    truth = targets(rows)
    logits = {k: torch.from_numpy(arrays[k]) for k in ("binary_logits", "family_logits", "version_logits")}
    loss, parts = loss_components(logits, torch.from_numpy(truth["binary"]), torch.from_numpy(truth["family"]),
                                  torch.from_numpy(truth["version"]), config=model.config)
    p = arrays["p_ai"]; y = truth["binary"]
    recall = [float(np.mean((p[y == c] >= .5) == bool(c))) for c in (0, 1) if np.any(y == c)]
    core = np.asarray([len(re.findall(r"\b\w+\b", row["text"])) >= 128 for row in rows])
    return {"selection_loss": float(loss), "loss": {k: float(v) for k,v in parts.items()},
            **_development_diagnostics(arrays, truth),
            "binary_accuracy_at_half": float(np.mean((p >= .5) == y)),
            "binary_balanced_accuracy_at_half": float(np.mean(recall)) if len(recall) == 2 else None,
            "core_ge_128_word_like_tokens": _development_diagnostics(arrays, truth, core),
            "monitoring_scope": "development diagnostics only; empirical FPR cutoff is fitted on this development cohort, not a calibrated deployment threshold or final performance; additional monitors do not change checkpoint selection",
            "nominal18_metric_scope": "trusted true nominal18 labels; predictions retain all25 leaves including Other errors; macroF1 averages the fixed18 nominal classes with zero_division=0",
            "length_unit": "word_like_tokens",
            "documents": len(rows), "windows": int(arrays["window_counts"].sum()),
            "encoder_coverage_min": float(arrays["encoder_coverage"].min()),
            "mean_encoder_tokens": float(arrays["encoder_tokens"].mean())}


def batch_invariance_check(model, texts, *, device="cpu", precision="fp32"):
    values = list(texts)
    if not values:
        raise ValueError("invariance check requires development text")
    selected = values[:3] + [min(values, key=len), max(values, key=len)]
    # Add an empty and a short document so padding differs even in uniform data.
    selected += ["", "A short example."]
    alone = model.predict_raw(selected, batch_size=1, device=device, precision=precision)
    together = model.predict_raw(selected, batch_size=len(selected), device=device, precision=precision)
    reverse = model.predict_raw(list(reversed(selected)), batch_size=3, device=device, precision=precision)
    tolerance = 5e-3 if precision == "bf16" else 1e-4
    differences = {k: max(float(np.max(np.abs(alone[k]-together[k]))),
                          float(np.max(np.abs(alone[k]-reverse[k][::-1]))))
                   for k in ("p_binary", "p_family", "p_model")}
    if any(error > tolerance for error in differences.values()):
        raise AssertionError(f"predictions depend on inference padding/batch beyond {tolerance}: {differences}")
    return {"passed": True, "probability_atol": tolerance, "maximum_absolute_probability_difference": differences,
            "documents": len(selected), "encoder_coverage_min": float(alone["encoder_coverage"].min()),
            "tested": "alone, mixed padded batches, reversed ordering and full document windows"}


def benchmark(args):
    device = _device(args.device); seed_everything(args.seed); torch.set_num_threads(args.cpu_threads)
    output = Path(args.output); progress = _progress_factory(output)
    rows = read_records(args.train, "train", args.max_train or 512, args.seed)
    cfg = NeuralConfig(max_length=args.max_length, stride=args.stride, attention_implementation=args.attention)
    model = NeuralDetector.from_pretrained_local(args.encoder, cfg).to(device)
    cache = TokenCache.build(rows, model.tokenizer, args.cache_dir, data_sha256=sha256_file(args.train),
                             tokenizer_sha256=tokenizer_hash(args.encoder), progress=progress)
    target = targets(rows); rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train(); measurements = []; finite = True
    for step in range(args.benchmark_steps):
        indices = rng.choice(len(rows), size=min(args.batch_size, len(rows)), replace=False)
        batch = make_training_batch(model, cache, indices, rng, device)
        synchronize(device); start = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.precision):
            logits = model(**batch)
            loss, _ = loss_components(logits, *(torch.as_tensor(target[k][indices], device=device) for k in ("binary", "family", "version")), config=cfg)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        finite = finite and bool(torch.isfinite(loss)) and bool(torch.isfinite(grad_norm))
        if not finite:
            raise FloatingPointError("non-finite benchmark loss or gradients")
        optimizer.step(); synchronize(device)
        duration = time.monotonic()-start
        measurements.append({"step": step+1, "seconds": duration, "documents": len(indices),
                             "padded_tokens": batch["input_ids"].numel(), "loss": float(loss.detach()), "gradient_norm": float(grad_norm)})
        progress({"phase": "benchmark", **measurements[-1]})
    measured = measurements[min(3, max(0,len(measurements)-1)):]
    report = {"precision": args.precision, "device": str(device), "steps": args.benchmark_steps,
              "finite_loss_and_gradients": finite, "batch_size": args.batch_size,
              "max_length": args.max_length, "warmup_steps_excluded": len(measurements)-len(measured),
              "documents_per_second": sum(m["documents"] for m in measured)/sum(m["seconds"] for m in measured),
              "padded_tokens_per_second": sum(m["padded_tokens"] for m in measured)/sum(m["seconds"] for m in measured),
              "parameter_count": sum(p.numel() for p in model.parameters()), "measurements": measurements,
              "quality_evaluation": False,
              "timing_scope": "synchronized forward, loss, backward, clipping and optimizer; excludes tokenization and batch preparation"}
    if device.type == "mps":
        report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    report["batch_invariance"] = batch_invariance_check(model, [r["text"] for r in rows], device=device, precision=args.precision)
    dump_json(output / "benchmark.json", report); progress({"phase": "benchmark_complete", **{k:v for k,v in report.items() if k != "measurements"}})
    return report


def _save_training(model, optimizer, scheduler, directory, state, rng, config, provenance, device):
    model.save_local(directory, training_config=config, provenance=provenance)
    saved = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()}
    if device.type == "mps":
        saved["mps_rng"] = torch.mps.get_rng_state().cpu()
    if device.type == "cuda":
        saved["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = Path(directory) / "training_state.pt.part"; torch.save(saved, temporary)
    temporary.replace(Path(directory) / "training_state.pt")
    dump_json(Path(directory) / "resume.json", {**state, "numpy_generator": rng.bit_generator.state,
        "training_state_sha256": sha256_file(Path(directory) / "training_state.pt")})


def _initial_checkpoint_provenance(path):
    path = Path(path).resolve()
    return {"path":str(path),"detector_weights_sha256":sha256_file(path/"detector.safetensors"),
            "detector_metadata_sha256":sha256_file(path/"detector.json")}


def train(args):
    initial_path = getattr(args,"init_checkpoint",None)
    if args.resume and initial_path:
        raise ValueError("init_checkpoint and resume are mutually exclusive")
    device = _device(args.device); seed_everything(args.seed); torch.set_num_threads(args.cpu_threads)
    output = Path(args.output); progress = _progress_factory(output)
    paths = {"train": Path(args.train), "validation": Path(args.validation)}
    if args.dev_challenge:
        paths["dev_challenge"] = Path(args.dev_challenge)
    rows = {name: read_records(path, name, args.max_train if name == "train" else args.max_validation, args.seed) for name,path in paths.items()}
    names = list(rows)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            assert_disjoint(rows[name], rows[other])
    cfg = NeuralConfig(max_length=args.max_length, stride=args.stride, attention_implementation=args.attention,
                       family_loss_weight=args.family_loss_weight, version_loss_weight=args.version_loss_weight)
    origin = args.resume or initial_path
    resume_metadata = json.loads((Path(args.resume)/"detector.json").read_text()) if args.resume else None
    initial_provenance = _initial_checkpoint_provenance(initial_path) if initial_path else None
    if resume_metadata and resume_metadata["provenance"].get("initial_checkpoint"):
        recorded = resume_metadata["provenance"]["initial_checkpoint"]
        initial_provenance = _initial_checkpoint_provenance(recorded["path"])
        if initial_provenance != recorded:
            raise ValueError("resume initial checkpoint fingerprint changed")
    model = NeuralDetector.load_local(origin) if origin else NeuralDetector.from_pretrained_local(args.encoder, cfg)
    if origin and asdict(model.config) != asdict(cfg):
        raise ValueError("checkpoint architecture/length/loss configuration changed")
    if initial_path:
        # Warm start deliberately ignores every optimizer, scheduler, progress
        # and RNG file in the initialization directory.
        seed_everything(args.seed)
    model.to(device)
    tok_path = Path(origin) / "tokenizer" if origin else args.encoder
    thash = tokenizer_hash(tok_path)
    provenance = {name: {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(rows[name])} for name,path in paths.items()}
    provenance["tokenizer_sha256"] = thash
    if initial_provenance:
        provenance["initial_checkpoint"] = initial_provenance
    caches = {name: TokenCache.build(rr, model.tokenizer, args.cache_dir,
               data_sha256=provenance[name]["sha256"], tokenizer_sha256=thash, progress=progress) for name,rr in rows.items()}
    weight_mode = getattr(args,"weight_mode","legacy_v1")
    target = targets(rows["train"]); weights = training_weights(rows["train"],mode=weight_mode)
    weight_summary = training_weight_summary(rows["train"],weights,mode=weight_mode)
    dump_json(output / "training_weight_summary.json",weight_summary)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(rows["train"])/args.batch_size)
    total_steps = steps_per_epoch*args.epochs; warmup_steps = int(total_steps*args.warmup_ratio)
    def lr_factor(step):
        return min((step+1)/max(warmup_steps, 1), max((total_steps-step)/max(total_steps-warmup_steps, 1), 0.))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    rng = np.random.default_rng(args.seed)
    state = {"epoch": 0, "cursor": 0, "order": [], "global_step": 0, "best_selection_loss": None}
    train_config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k not in {"resume"}}
    train_config["weight_mode"] = weight_mode
    train_config["init_checkpoint"] = initial_provenance["path"] if initial_provenance else None
    if args.resume:
        metadata = resume_metadata
        if metadata["provenance"] != provenance:
            raise ValueError("resume data/tokenizer provenance changed")
        prior_config = metadata["training_config"]
        for key in ("batch_size", "epochs", "lr", "seed", "precision", "weight_decay", "warmup_ratio"):
            if prior_config.get(key) != train_config.get(key):
                raise ValueError(f"resume training option changed: {key}")
        if prior_config.get("weight_mode","legacy_v1") != weight_mode:
            raise ValueError("resume training option changed: weight_mode")
        state = json.loads((Path(args.resume) / "resume.json").read_text())
        state_path = Path(args.resume) / "training_state.pt"
        if sha256_file(state_path) != state.pop("training_state_sha256"):
            raise ValueError("resume optimizer/RNG checksum mismatch")
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"]); random.setstate(saved["python_rng"])
        rng.bit_generator.state = state.pop("numpy_generator")
        if device.type == "mps" and "mps_rng" in saved:
            torch.mps.set_rng_state(saved["mps_rng"])
        if device.type == "cuda" and "cuda_rng" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        for item in optimizer.state.values():
            for key,value in item.items():
                if torch.is_tensor(value) and key != "step":
                    item[key] = value.to(device)
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    last_log = time.monotonic(); running = []; last_save = state["global_step"]
    progress({"phase": "train_ready", "inputs": provenance, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
              "precision": args.precision, "device": str(device), "weight_min": float(weights.min()), "weight_max": float(weights.max()),
              "training_weight_summary":weight_summary,
              "initialization_mode":"resume" if args.resume else "init_checkpoint_new_training" if initial_path else "pretrained_encoder_new_training",
              "global_step":state["global_step"],"optimizer_state_entries":len(optimizer.state),
              "family_supervision": int(target["mask_family"].sum()), "version_supervision": int(target["mask_model"].sum())})
    try:
        for epoch in range(state["epoch"], args.epochs):
            model.train()
            if not state["order"]:
                state["order"] = rng.permutation(len(rows["train"])).tolist(); state["cursor"] = 0
            while state["cursor"] < len(state["order"]):
                indices = np.asarray(state["order"][state["cursor"]:state["cursor"]+args.batch_size], np.int64)
                batch = make_training_batch(model, caches["train"], indices, rng, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, args.precision):
                    logits = model(**batch)
                    loss, parts = loss_components(logits, *(torch.as_tensor(target[k][indices], device=device) for k in ("binary", "family", "version")),
                        sample_weights=torch.as_tensor(weights[indices], dtype=torch.float32, device=device), config=cfg)
                loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad)):
                    raise FloatingPointError("non-finite training loss or gradients; last checkpoint remains available")
                optimizer.step(); scheduler.step(); state["cursor"] += len(indices); state["global_step"] += 1
                running.append(float(loss.detach()))
                if state["global_step"] % 50 == 0 or time.monotonic()-last_log >= 45:
                    progress({"phase": "train", "epoch": epoch+1, "step": state["global_step"], "documents_seen_epoch": state["cursor"],
                        "loss": float(np.mean(running)), "last_losses": {k: float(v.detach()) for k,v in parts.items()},
                        "learning_rate": scheduler.get_last_lr()[0], "gradient_norm": float(grad)})
                    last_log = time.monotonic(); running = []
                if stop[0] or (args.save_every and state["global_step"]-last_save >= args.save_every):
                    _save_training(model, optimizer, scheduler, output / "last", state, rng, train_config, provenance, device)
                    last_save = state["global_step"]
                if stop[0]:
                    progress({"phase": "paused", "resume": str((output / "last").resolve()), "step": state["global_step"]})
                    return model
            metrics = {name: _development_metrics(model, rows[name], caches[name], batch_size=args.batch_size,
                       device=device, precision=args.precision, progress=progress) for name in names if name != "train"}
            selection = float(np.mean([v["selection_loss"] for v in metrics.values()]))
            improved = state["best_selection_loss"] is None or selection < state["best_selection_loss"]
            if improved:
                state["best_selection_loss"] = selection
                model.save_local(output / "best", training_config=train_config, provenance=provenance)
                dump_json(output / "best" / "selection.json", {"epoch": epoch+1, "step": state["global_step"],
                    "selection_loss": selection, "criterion": "mean development binary NLL + weighted conditional fine NLL; validation and dev challenge equally weighted", "metrics": metrics})
            state.update({"epoch": epoch+1, "cursor": 0, "order": []})
            _save_training(model, optimizer, scheduler, output / "last", state, rng, train_config, provenance, device)
            last_save = state["global_step"]
            progress({"phase": "epoch_complete", "epoch": epoch+1, "selection_loss": selection, "improved": improved, "development": metrics})
            if stop[0]:
                return model
        progress({"phase": "training_complete", "best": str((output / "best").resolve()), "step": state["global_step"], "final_evaluation_performed": False})
        return model
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def predict(args):
    device = _device(args.device); torch.set_num_threads(args.cpu_threads)
    progress = _progress_factory(Path(args.output).parent)
    model = NeuralDetector.load_local(args.checkpoint).to(device)
    rows = read_records(args.input, "predict")
    cache = TokenCache.build(rows, model.tokenizer, args.cache_dir, data_sha256=sha256_file(args.input),
                            tokenizer_sha256=tokenizer_hash(Path(args.checkpoint) / "tokenizer"), progress=progress)
    arrays = export_predictions(model, rows, cache, args.output, batch_size=args.batch_size,
                                device=device, precision=args.precision, progress=progress)
    progress({"phase": "prediction_complete", "documents": len(rows), "windows": int(arrays["window_counts"].sum()),
              "encoder_coverage_min": float(arrays["encoder_coverage"].min()), "output": str(Path(args.output).resolve()),
              "calibrated": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("benchmark", "train", "predict"))
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--dev-challenge", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint",type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT / ".cache/english_tokens")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=448)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--weight-mode",choices=("legacy_v1","class_normalized_v2","class_domain_v3"),default="legacy_v1")
    parser.add_argument("--warmup-ratio", type=float, default=.06)
    parser.add_argument("--clip-grad", type=float, default=1.)
    parser.add_argument("--family-loss-weight", type=float, default=.3)
    parser.add_argument("--version-loss-weight", type=float, default=.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-validation", type=int)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.benchmark_steps < 1 or args.stride < 1:
        parser.error("batch size, epochs, steps and stride must be positive")
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if args.command in {"benchmark", "train"} and (not args.train or (not args.encoder and not args.resume and not args.init_checkpoint)):
        parser.error("benchmark/train requires --train and --encoder (or train --resume/--init-checkpoint)")
    if args.command == "benchmark" and not args.encoder:
        parser.error("benchmark requires --encoder")
    if args.command == "train" and not args.validation:
        parser.error("train requires --validation")
    if args.command == "predict" and (not args.input or not args.checkpoint):
        parser.error("predict requires --input and --checkpoint")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1 or args.clip_grad <= 0:
        parser.error("invalid learning rate, weight decay, warmup ratio or gradient clipping")
    if min(args.family_loss_weight, args.version_loss_weight) < 0 or args.save_every < 0:
        parser.error("loss weights and save interval must be nonnegative")
    if any(value is not None and value < 1 for value in (args.max_train, args.max_validation)):
        parser.error("maximum row counts must be positive")
    {"benchmark": benchmark, "train": train, "predict": predict}[args.command](args)


if __name__ == "__main__":
    main()
