"""One synthetic-only Modern2 MPS batch8/batch16 resource probe.

Prepare the policy/code/hash receipt first (no torch/model loading):
  .venv/bin/python -B research/english_v2_resources/probe_modern_batch.py --output NEW_DIR
Then execute exactly once, with MPS access:
  .venv/bin/python -B research/english_v2_resources/probe_modern_batch.py --output NEW_DIR --execute

No dataset is accepted. The only model input is random valid token IDs forming
512-token full windows and alternating balanced fake binary labels. Updated
parameters exist only in memory and are discarded. No model save API is called.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
PARENT = PROJECT / "models/candidates/modernbert_epoch2"
WEIGHTS = "8c884a1c0172e6fe218292190f5792353fa0b4ebd27d0dba90b99928f1d4b4c8"
METADATA = "c3a7d3d67590038633e64d6db4e168021d2415c914dbd06296112588ad47764f"
POLICY = {
    "purpose": "Pure synthetic hardware throughput/memory preflight, not model selection or detection accuracy.",
    "batch_sizes_in_order": [8, 16], "warmup_steps_per_batch": 2, "timed_steps_per_batch": 10,
    "sequence_length": 512, "seed": 20261017, "device": "mps", "precision": "bf16",
    "inputs": "Uniform valid non-special token IDs with the checkpoint's special-token template; full512 attention mask; no natural-language or corpus text.",
    "targets": "Alternating0/1 fake binary labels; exactly balanced in every batch; fine targets absent.",
    "model": "Independent fresh load of the same read-only Modern2 checkpoint per batch size; model.train dropout; no shared optimizer state.",
    "loss": "Frozen loss_components with effective1/0/0 objective, all sample weights1; no accuracy metric.",
    "optimizer": {"name": "AdamW", "lr": 0.000005, "weight_decay": 0.01, "gradient_clip_norm": 1.0, "scheduler": "none in short hardware probe"},
    "cpu_threads": 6,
    "timer": "Synchronized forward+loss+backward+gradient check/clip+AdamW step; excludes model loading and CPU batch preparation, matches frozen benchmark timing scope.",
    "memory": "Sample MPS current/driver after forward and after optimizer; current process RSS after each synchronized step; process maximum RSS separately labeled.",
    "oom": "Record error and cleanup; skip batch16 if batch8 fails; no retry, batch32, memory-watermark override, or alternate real-data probe.",
    "resource_choice_only": "Suggest the faster finite batch only if no OOM and measured driver memory stays below80% of MPS recommended maximum; root freezes actual v2 recipe later.",
    "forbidden": ["dataset/text reading", "accuracy/AUROC/calibration", "realv2 training", "initializer creation", "saving updated weights", "modifying parent or frozen v1 source"],
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def new_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_inventory():
    return {str(path.relative_to(PARENT)): sha(path) for path in sorted(PARENT.rglob("*")) if path.is_file()}


def prepare(output):
    before = source_inventory()
    if before.get("detector.safetensors") != WEIGHTS or before.get("detector.json") != METADATA:
        raise ValueError("the frozen Modern2 parent identity changed")
    output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).read_bytes()
    (output / "probe_source.py").write_bytes(code)
    record = {"prepared_at_utc": utc(), "policy": POLICY, "probe_code_sha256": sha(__file__),
        "probe_source_snapshot_sha256": sha(output / "probe_source.py"), "parent_path": str(PARENT),
        "parent_all_files_sha256": before,
        "frozen_training_code_sha256": sha(PROJECT / "experiments/english_v1/neural.py"),
        "no_model_loaded_or_forward_performed": True, "no_dataset_opened": True}
    new_json(output / "policy_and_source_audit.json", record)
    return record


def rss_bytes():
    import os
    result = subprocess.run(["ps", "-p", str(os.getpid()), "-o", "rss="], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(result.stdout.strip()) * 1024


def memory(torch):
    return {"mps_current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory())}


def probe(batch_size, torch, np, NeuralDetector, autocast, loss_components):
    result = {"batch_size": batch_size, "warmup_steps": 2, "timed_steps": 10,
              "passed": False, "OOM": False, "measurements": []}
    model = optimizer = batch = target = missing = logits = loss = parts = grad = None
    try:
        torch.manual_seed(POLICY["seed"])
        torch.mps.manual_seed(POLICY["seed"])
        torch.set_num_threads(6)
        model = NeuralDetector.load_local(PARENT).to("mps").train()
        if model.encoder.config.model_type != "modernbert" or model.config.max_length != 512:
            raise ValueError("expected the frozen512-token ModernBERT architecture")
        config = replace(model.config, family_loss_weight=0.0, version_loss_weight=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.000005, weight_decay=0.01)
        prefix, suffix = model._special_template
        payload = 512 - len(prefix) - len(suffix)
        allowed = np.setdiff1d(np.arange(int(model.encoder.config.vocab_size), dtype=np.int64),
                               np.asarray(model.tokenizer.all_special_ids, dtype=np.int64))
        rng = np.random.default_rng(POLICY["seed"])
        target = torch.as_tensor(np.arange(batch_size) % 2, dtype=torch.long, device="mps")
        missing = torch.full_like(target, -100)
        torch.mps.synchronize()
        result["loaded_memory"] = {**memory(torch), "rss_bytes": rss_bytes()}
        for step in range(12):
            sequences = [rng.choice(allowed, size=payload, replace=True) for _ in range(batch_size)]
            batch = model.make_batch(sequences, "mps")
            if tuple(batch["input_ids"].shape) != (batch_size, 512) or not bool((batch["attention_mask"] == 1).all()):
                raise ValueError("synthetic probe must have fully populated512-token windows")
            torch.mps.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast("mps", "bf16"):
                logits = model(**batch)
                loss, parts = loss_components(logits, target, missing, missing, config=config)
            forward_memory = memory(torch)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad)):
                raise FloatingPointError("nonfinite synthetic loss or gradient")
            if float(parts["family"].detach()) != 0 or float(parts["version"].detach()) != 0:
                raise ValueError("the hardware probe objective must be binary only")
            optimizer.step()
            torch.mps.synchronize()
            seconds = time.perf_counter() - started
            after_memory = memory(torch)
            result["measurements"].append({"step": step + 1, "warmup": step < 2, "seconds": seconds,
                "documents": batch_size, "tokens": batch_size * 512, "finite_loss_and_gradient": True,
                "forward_memory": forward_memory, "after_optimizer_memory": after_memory, "rss_bytes": rss_bytes()})
            print(json.dumps({"batch_size": batch_size, "step": step + 1, "warmup": step < 2,
                              "seconds": seconds, **after_memory}), flush=True)
        measured = result["measurements"][2:]
        seconds = sum(m["seconds"] for m in measured)
        result.update({"passed": True, "documents_per_second": batch_size * 10 / seconds,
            "tokens_per_second": batch_size * 512 * 10 / seconds,
            "median_step_seconds": statistics.median(m["seconds"] for m in measured),
            "min_step_seconds": min(m["seconds"] for m in measured),
            "max_step_seconds": max(m["seconds"] for m in measured)})
        result["maximum_sampled_mps_current_bytes"] = max(m[point]["mps_current_allocated_bytes"] for m in result["measurements"] for point in ("forward_memory", "after_optimizer_memory"))
        result["maximum_sampled_mps_driver_bytes"] = max(m[point]["mps_driver_allocated_bytes"] for m in result["measurements"] for point in ("forward_memory", "after_optimizer_memory"))
        result["maximum_sampled_rss_bytes"] = max(m["rss_bytes"] for m in result["measurements"])
    except BaseException as error:
        result.update({"error_type": type(error).__name__, "error": str(error),
                       "OOM": "out of memory" in str(error).casefold()})
    finally:
        model = optimizer = batch = target = missing = logits = loss = parts = grad = None
        gc.collect()
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
            result["after_cleanup_memory"] = {**memory(torch), "rss_bytes": rss_bytes()}
        except BaseException as error:
            result["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def execute(output):
    record_path = output / "policy_and_source_audit.json"
    record = json.loads(record_path.read_text())
    if (record["policy"] != POLICY or record["probe_code_sha256"] != sha(__file__) or
            record["probe_source_snapshot_sha256"] != sha(output / "probe_source.py") or
            record["parent_all_files_sha256"] != source_inventory() or
            record["frozen_training_code_sha256"] != sha(PROJECT / "experiments/english_v1/neural.py")):
        raise ValueError("prepared source/policy/checkpoint identity changed")
    new_json(output / "STARTED.json", {"started_at_utc": utc(), "policy_receipt_sha256": sha(record_path)})
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from experiments.english_v1 import runtime  # noqa: F401
    import numpy as np
    import resource
    import torch
    from experiments.english_v1.neural import NeuralDetector, autocast, loss_components
    report = {"started_at_utc": utc(), "pure_synthetic_only": True, "quality_metrics_computed": False,
              "trained_weights_saved": False, "policy_receipt_sha256": sha(record_path), "batches": []}
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        listing = subprocess.run(["ps", "-axo", "command="], check=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True).stdout
        if any(re.search(r"--device(?:=|\s+)mps(?:\s|$)", line) for line in listing.splitlines()):
            raise RuntimeError("another MPS CLI process is active")
        limit_fn = getattr(torch.mps, "recommended_max_memory", None)
        limit = int(limit_fn()) if limit_fn else None
        report["mps_recommended_max_memory_bytes"] = limit
        for size in (8, 16):
            item = probe(size, torch, np, NeuralDetector, autocast, loss_components)
            report["batches"].append(item)
            new_json(output / f"batch{size}.json", item)
            if not item["passed"] or "cleanup_error" in item:
                break
        successful = [v for v in report["batches"] if v["passed"] and "cleanup_error" not in v]
        safe = [v for v in successful if limit is not None and v["maximum_sampled_mps_driver_bytes"] < 0.8 * limit]
        report["resource_only_suggested_batch"] = max(safe, key=lambda v: v["documents_per_second"])["batch_size"] if safe else None
        if len(successful) == 2:
            report["batch16_vs_batch8_throughput_ratio"] = successful[1]["documents_per_second"] / successful[0]["documents_per_second"]
        report["passed"] = len(successful) == 2
    except BaseException as error:
        report.update({"passed": False, "error_type": type(error).__name__, "error": str(error)})
    finally:
        report["parent_all_files_unchanged"] = source_inventory() == record["parent_all_files_sha256"]
        report["parent_weights_sha256_after"] = sha(PARENT / "detector.safetensors")
        report["frozen_training_code_unchanged"] = record["frozen_training_code_sha256"] == sha(PROJECT / "experiments/english_v1/neural.py")
        report["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
        report["completed_at_utc"] = utc()
        report["passed"] = bool(report.get("passed") and report["parent_all_files_unchanged"] and report["frozen_training_code_unchanged"])
        new_json(output / "report.json", report)
    print(json.dumps({key: report.get(key) for key in ("passed", "resource_only_suggested_batch", "batch16_vs_batch8_throughput_ratio",
        "parent_all_files_unchanged", "trained_weights_saved", "error_type")}), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.execute:
        return execute(output)
    prepare(output)
    print(json.dumps({"prepared": True, "no_model_loaded": True, "policy": str(output / "policy_and_source_audit.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
