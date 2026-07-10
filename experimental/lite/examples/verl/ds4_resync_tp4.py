# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate DS4 mixed-checkpoint MLite BF16 export against a pure-FP8 rollout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


_DAPO_P99_RATIO_LIMIT = 0.01
_DAPO_MAX_RATIO_LIMIT = 0.05
_DAPO_P99_KL_LIMIT = 1e-4
_DAPO_CLIP_LOW = 0.8
_DAPO_CLIP_HIGH = 1.2
_DEFAULT_MAX_SHARD_BYTES = 5 * 1024**3
_FINGERPRINT_MISMATCH_EXAMPLES = 100


def math_prompts() -> list[str]:
    cases = (
        "17 * 19",
        "144 / 12",
        "2^10",
        "gcd(84, 126)",
        "lcm(12, 18)",
        "the next prime after 97",
        "the sum of the first 20 integers",
        "15% of 240",
        "3/4 + 5/8",
        "7/9 - 2/3",
        "sqrt(2025)",
        "the area of a circle of radius 3 in terms of pi",
        "the perimeter of a 5 by 8 rectangle",
        "x if 3x + 7 = 31",
        "x if x^2 = 169 and x is positive",
        "the mean of 4, 7, 9, 10",
        "the median of 2, 3, 8, 11, 15",
        "the remainder of 1234 divided by 9",
        "the number of diagonals in a hexagon",
        "6 choose 2",
        "the derivative of x^3 at x=2",
        "the integral of 2x from 0 to 3",
        "log base 2 of 128",
        "the slope through (1,2) and (5,10)",
        "the hypotenuse of a right triangle with legs 5 and 12",
        "the missing angle of a triangle with angles 35 and 65 degrees",
        "the sum of interior angles of an octagon",
        "0.125 as a fraction",
        "the reciprocal of 7/11",
        "the solution count of x^2+1=0 over reals",
        "the smallest positive multiple of both 8 and 14",
        "the value of 5!",
        "the sum 1+3+5+7+9",
        "the cube root of 1728",
        "the probability of heads on one fair coin toss",
        "the distance between -7 and 5 on the number line",
    )
    return [f"Solve briefly: what is {case}?" for case in cases]


def percentile(values: torch.Tensor, quantile: float) -> float:
    if values.ndim != 1 or not values.numel():
        raise ValueError("percentile requires a non-empty flat tensor")
    rank = max(1, min(values.numel(), int(quantile * values.numel() + 0.999999)))
    return float(torch.kthvalue(values, rank).values)


def pure_block_fp8_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a checkpoint config whose quantized matrices all use block FP8."""
    result = copy.deepcopy(config)
    quantization = result.setdefault("quantization_config", {})
    if not quantization:
        raise ValueError("DeepSeek-V4 pure FP8 conversion requires quantization_config")
    result["expert_dtype"] = "fp8"
    quantization["expert_dtype"] = "fp8"
    quantization["scale_fmt"] = "float32"
    return result


def _fingerprint_layer(name: str) -> str:
    match = re.search(r"(?:^|\.)(layers|mtp)\.(\d+)(?:\.|$)", name)
    return f"{match.group(1)}.{match.group(2)}" if match else "global"


def _fingerprint_set_digest(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: (item["name"], item["kind"])):
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def compare_engine_weight_fingerprints(
    cold_workers: list[list[dict[str, Any]]],
    online_workers: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compare vLLM TP shards before and after the native reload lifecycle."""
    if len(cold_workers) != len(online_workers) or not cold_workers:
        raise ValueError(
            "cold and online engine worker counts must match and be nonzero"
        )

    mismatch_count = 0
    mismatch_examples = []
    tensor_count = 0
    byte_count = 0
    workers = []
    layers: dict[str, dict[str, Any]] = {}
    layer_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for worker_index, (cold_records, online_records) in enumerate(
        zip(cold_workers, online_workers, strict=True)
    ):
        cold = {record["name"]: record for record in cold_records}
        online = {record["name"]: record for record in online_records}
        if len(cold) != len(cold_records) or len(online) != len(online_records):
            raise ValueError("engine fingerprint records contain duplicate names")
        worker_mismatches = 0
        for name in sorted(cold.keys() | online.keys()):
            cold_record = cold.get(name)
            online_record = online.get(name)
            source = cold_record or online_record
            assert source is not None
            layer_name = _fingerprint_layer(name)
            layer = layers.setdefault(
                layer_name,
                {"tensor_count": 0, "byte_count": 0, "mismatch_count": 0},
            )
            per_layer = layer_records.setdefault(layer_name, {"cold": [], "online": []})
            if cold_record is not None:
                tensor_count += 1
                byte_count += int(cold_record["nbytes"])
                layer["tensor_count"] += 1
                layer["byte_count"] += int(cold_record["nbytes"])
                per_layer["cold"].append({"worker": worker_index, **cold_record})
            if online_record is not None:
                per_layer["online"].append({"worker": worker_index, **online_record})
            differing_fields = []
            if cold_record is None or online_record is None:
                differing_fields = ["presence"]
            else:
                differing_fields = [
                    field
                    for field in ("kind", "dtype", "shape", "nbytes", "sha256")
                    if cold_record[field] != online_record[field]
                ]
            if differing_fields:
                mismatch_count += 1
                worker_mismatches += 1
                layer["mismatch_count"] += 1
                if len(mismatch_examples) < _FINGERPRINT_MISMATCH_EXAMPLES:
                    mismatch_examples.append(
                        {
                            "worker": worker_index,
                            "name": name,
                            "differing_fields": differing_fields,
                        }
                    )
        workers.append(
            {
                "worker": worker_index,
                "tensor_count": len(cold_records),
                "mismatch_count": worker_mismatches,
                "cold_sha256": _fingerprint_set_digest(cold_records),
                "online_sha256": _fingerprint_set_digest(online_records),
            }
        )

    for layer_name, layer in layers.items():
        exact = layer["mismatch_count"] == 0
        layer["exact_match"] = exact
        layer["cold_sha256"] = _fingerprint_set_digest(
            layer_records[layer_name]["cold"]
        )
        layer["online_sha256"] = _fingerprint_set_digest(
            layer_records[layer_name]["online"]
        )
        layer["implied_dequantized_max_abs"] = 0.0 if exact else None
        layer["implied_dequantized_relative_l2"] = 0.0 if exact else None

    return {
        "schema_version": 1,
        "worker_count": len(cold_workers),
        "tensor_count": tensor_count,
        "byte_count": byte_count,
        "exact_match": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "mismatch_examples": mismatch_examples,
        "layers": dict(sorted(layers.items())),
        "workers": workers,
        "metric_basis": (
            "Bitwise-identical engine weights and scales imply zero deterministic "
            "dequantized diff; non-identical states are not assigned a numeric diff."
        ),
    }


def payload_row(
    token_ids: list[int] | torch.Tensor, logprobs: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Build one token-aligned payload row without reducing FP32 precision."""
    ids = torch.as_tensor(token_ids, dtype=torch.int32).cpu()
    values = logprobs.detach().to(dtype=torch.float32, device="cpu")
    if values.ndim != 2 or values.shape[0] != ids.numel():
        raise ValueError("token ids and logprob rows are not aligned")
    return {"token_ids": ids, "logprobs": values}


def mlite_payload_row(
    token_ids: list[int] | torch.Tensor, logits: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Align MLite position i logits with the prompt token at position i + 1."""
    ids = torch.as_tensor(token_ids, dtype=torch.int64)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != ids.numel():
        raise ValueError("MLite logits are not aligned with the prompt tokens")
    logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    return payload_row(ids[1:], logprobs)


def _bf16_rounded_logprobs(values: torch.Tensor) -> torch.Tensor:
    rounded = values.to(torch.bfloat16).float()
    return rounded - torch.logsumexp(rounded, dim=-1, keepdim=True)


def _distribution_metrics(
    reference: list[dict[str, torch.Tensor]],
    candidate: list[dict[str, torch.Tensor]],
    *,
    bf16_rounded: bool,
) -> dict[str, float | int]:
    deltas, kls, selected_deltas = [], [], []
    token_count = 0
    for left, right in zip(reference, candidate, strict=True):
        stored_ref = left["logprobs"]
        stored_cand = right["logprobs"]
        if stored_ref.dtype != torch.float32 or stored_cand.dtype != torch.float32:
            raise ValueError("stored logprob artifacts must be FP32")
        ref = stored_ref.float()
        cand = stored_cand.float()
        ids = left["token_ids"].long()
        if ref.shape != cand.shape or ids.shape != ref.shape[:-1]:
            raise ValueError("token-aligned distribution shapes differ")
        if not torch.equal(ids, right["token_ids"].long()):
            raise ValueError("tokenized prompts differ between arms")
        if bf16_rounded:
            ref = _bf16_rounded_logprobs(ref)
            cand = _bf16_rounded_logprobs(cand)
        delta = cand - ref
        deltas.append(delta.abs().flatten())
        kls.append((ref.exp() * (ref - cand)).sum(dim=-1).flatten())
        selected_deltas.append(delta.gather(-1, ids.unsqueeze(-1)).flatten())
        token_count += ids.numel()
    all_delta = torch.cat(deltas)
    all_kl = torch.cat(kls)
    all_selected_delta = torch.cat(selected_deltas)
    selected_abs = all_selected_delta.abs()
    ratio_deviation = (all_selected_delta.clamp(-80, 80).exp() - 1.0).abs()
    clip_crossings = int(
        (
            (all_selected_delta < torch.log(torch.tensor(_DAPO_CLIP_LOW)))
            | (all_selected_delta > torch.log(torch.tensor(_DAPO_CLIP_HIGH)))
        )
        .sum()
        .item()
    )
    return {
        "max_abs": float(all_delta.max()),
        "p99_abs": percentile(all_delta, 0.99),
        "max_kl": float(all_kl.max()),
        "p99_kl": percentile(all_kl, 0.99),
        "max_selected_token_logprob_delta": float(selected_abs.max()),
        "p99_selected_token_logprob_delta": percentile(selected_abs, 0.99),
        "max_ratio_deviation": float(ratio_deviation.max()),
        "p99_ratio_deviation": percentile(ratio_deviation, 0.99),
        "clipping_boundary_crossings": clip_crossings,
        "token_count": token_count,
    }


def compare_distributions(
    reference: list[dict[str, torch.Tensor]],
    candidate: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("prompt result counts differ")
    if not reference:
        raise ValueError("at least one prompt result is required")
    fp32 = _distribution_metrics(reference, candidate, bf16_rounded=False)
    bf16 = _distribution_metrics(reference, candidate, bf16_rounded=True)
    return {
        "prompt_count": len(reference),
        "token_count": fp32["token_count"],
        "fp32": fp32,
        "bf16_rounded": bf16,
    }


def compare_three_arms(
    vllm_fp8_cold: list[dict[str, torch.Tensor]],
    vllm_fp8_online: list[dict[str, torch.Tensor]],
    mlite_bf16: list[dict[str, torch.Tensor]],
    *,
    minimum_prompts: int = 32,
) -> dict[str, Any]:
    if len(vllm_fp8_cold) < minimum_prompts:
        raise ValueError(
            f"three-arm parity requires at least {minimum_prompts} prompts, "
            f"got {len(vllm_fp8_cold)}"
        )
    pairs = {
        "vllm_fp8_cold__vllm_fp8_online": compare_distributions(
            vllm_fp8_cold, vllm_fp8_online
        ),
        "mlite_bf16__vllm_fp8_cold": compare_distributions(mlite_bf16, vllm_fp8_cold),
        "mlite_bf16__vllm_fp8_online": compare_distributions(
            mlite_bf16, vllm_fp8_online
        ),
    }
    failures = []
    for name, pair in pairs.items():
        metrics = pair["fp32"]
        if metrics["p99_ratio_deviation"] > _DAPO_P99_RATIO_LIMIT:
            failures.append(f"{name}: p99 ratio deviation")
        if metrics["max_ratio_deviation"] > _DAPO_MAX_RATIO_LIMIT:
            failures.append(f"{name}: max ratio deviation")
        if metrics["p99_kl"] > _DAPO_P99_KL_LIMIT:
            failures.append(f"{name}: p99 KL")
        if metrics["clipping_boundary_crossings"]:
            failures.append(f"{name}: clipping boundary crossings")
    return {
        "schema_version": 3,
        "arm_semantics": {
            "mlite_bf16": "official mixed checkpoint loaded into the MLite BF16 master",
            "vllm_fp8_cold": "pure block-FP8 artifact exported from that BF16 master",
            "vllm_fp8_online": "the same pure block-FP8 export reloaded online",
        },
        "pairs": pairs,
        "gate": {
            "acceptable": not failures,
            "failures": failures,
            "thresholds": {
                "p99_ratio_deviation": _DAPO_P99_RATIO_LIMIT,
                "max_ratio_deviation": _DAPO_MAX_RATIO_LIMIT,
                "p99_kl": _DAPO_P99_KL_LIMIT,
                "clip_interval": [_DAPO_CLIP_LOW, _DAPO_CLIP_HIGH],
                "clipping_boundary_crossings": 0,
            },
        },
    }


def copy_checkpoint_metadata(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if path.is_dir():
            shutil.copytree(path, output / path.name, dirs_exist_ok=True)
        elif (
            path.suffix != ".safetensors"
            and path.name != "model.safetensors.index.json"
        ):
            shutil.copy2(path, output / path.name)


def checkpoint_tensor_names(source: Path) -> set[str]:
    """Read the authoritative tensor-name set from an indexed checkpoint."""
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"checkpoint coverage requires an index file: {index_path}"
        )
    weight_map = json.loads(index_path.read_text()).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"checkpoint index has no non-empty weight_map: {index_path}")
    return set(weight_map)


def write_exported_checkpoint(
    weights: Iterable[tuple[str, torch.Tensor]],
    source: Path,
    output: Path,
    *,
    max_shard_bytes: int = _DEFAULT_MAX_SHARD_BYTES,
) -> dict[str, Any]:
    """Write a bounded-memory runtime export as indexed safetensor shards."""
    from safetensors.torch import save_file

    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    output.mkdir(parents=True, exist_ok=False)
    copy_checkpoint_metadata(source, output)
    source_config = json.loads((source / "config.json").read_text())
    (output / "config.json").write_text(
        json.dumps(pure_block_fp8_config(source_config), indent=2, sort_keys=True)
        + "\n"
    )

    pending: list[tuple[Path, list[str], int]] = []
    bucket: dict[str, torch.Tensor] = {}
    bucket_bytes = 0
    seen: set[str] = set()

    def flush() -> None:
        nonlocal bucket, bucket_bytes
        if not bucket:
            return
        path = output / f"model-{len(pending) + 1:05d}.safetensors"
        save_file(bucket, path)
        pending.append((path, list(bucket), bucket_bytes))
        bucket = {}
        bucket_bytes = 0

    for name, tensor in weights:
        if name in seen:
            raise ValueError(f"duplicate exported checkpoint tensor: {name}")
        seen.add(name)
        tensor = tensor.detach().to(device="cpu").contiguous()
        tensor_bytes = tensor.numel() * tensor.element_size()
        if bucket and bucket_bytes + tensor_bytes > max_shard_bytes:
            flush()
        bucket[name] = tensor
        bucket_bytes += tensor_bytes
    flush()
    if not pending:
        raise ValueError("MLite runtime export produced no checkpoint tensors")

    source_names = checkpoint_tensor_names(source)
    missing = sorted(source_names - seen)
    unexpected = sorted(seen - source_names)
    if missing or unexpected:
        raise ValueError(
            "MLite export tensor coverage differs from the source checkpoint: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    coverage = {
        "exact_match": True,
        "source_tensor_count": len(source_names),
        "exported_tensor_count": len(seen),
    }

    weight_map: dict[str, str] = {}
    total_size = 0
    shard_count = len(pending)
    for index, (temporary, names, shard_bytes) in enumerate(pending, 1):
        filename = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        temporary.replace(output / filename)
        weight_map.update(dict.fromkeys(names, filename))
        total_size += shard_bytes
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    return coverage


def model_vocab_size(config: Any, tokenizer: Any) -> int:
    size = int(config.vocab_size)
    if size < int(tokenizer.vocab_size):
        raise ValueError("model vocabulary is smaller than tokenizer vocabulary")
    return size


def _dense_logprobs(entries: list[Any], vocab_size: int) -> torch.Tensor:
    rows = []
    for entry in entries:
        row = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
        for token_id, value in entry.items():
            row[int(token_id)] = float(value.logprob)
        if not torch.isfinite(row).all():
            raise ValueError("vLLM did not return the complete prompt distribution")
        rows.append(row)
    return torch.stack(rows)


def _checkpoint_expert_dtype(model: Path) -> str:
    config = json.loads((model / "config.json").read_text())
    quantization = config.get("quantization_config") or {}
    return str(config.get("expert_dtype") or quantization.get("expert_dtype", "fp4"))


def require_pure_block_fp8_checkpoint(model: Path) -> None:
    expert_dtype = _checkpoint_expert_dtype(model)
    if expert_dtype != "fp8":
        raise ValueError(
            f"formal DS4 parity requires expert_dtype='fp8', got {expert_dtype!r}"
        )


def require_mixed_flash_checkpoint(model: Path) -> None:
    expert_dtype = _checkpoint_expert_dtype(model)
    if expert_dtype != "fp4":
        raise ValueError(
            f"formal DS4 MLite source requires expert_dtype='fp4', got {expert_dtype!r}"
        )


def _collect_vllm_rows(llm: Any, tokenizer: Any, vocab_size: int) -> list[dict]:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=-1)
    rows = []
    for prompt in math_prompts():
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        result = llm.generate(
            [{"prompt_token_ids": token_ids}], params, use_tqdm=False
        )[0]
        entries = result.prompt_logprobs
        if entries is None or len(entries) != len(token_ids):
            raise ValueError("vLLM prompt logprobs are not token-aligned")
        rows.append(
            payload_row(token_ids[1:], _dense_logprobs(entries[1:], vocab_size))
        )
    return rows


def reload_resync_checkpoint(llm: Any, model: Path) -> None:
    """Inject a checkpoint through vLLM's native layerwise reload lifecycle."""
    llm.collective_rpc("reload_checkpoint_from_path", args=(str(model),), timeout=None)


def collect_engine_weight_fingerprints(llm: Any) -> list[list[dict[str, Any]]]:
    """Collect one deterministic state manifest from every vLLM TP worker."""
    records = llm.collective_rpc("checkpoint_state_fingerprints", timeout=None)
    if not isinstance(records, list) or not records:
        raise ValueError("vLLM returned no engine weight fingerprints")
    return records


def collect(
    model: Path,
    output: Path,
    *,
    resync_model: Path | None = None,
    resync_output: Path | None = None,
    weight_output: Path | None = None,
) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    require_pure_block_fp8_checkpoint(model)
    if (resync_model is None) != (resync_output is None):
        raise ValueError("resync_model and resync_output must be provided together")
    if (resync_model is None) != (weight_output is None):
        raise ValueError("weight_output is required exactly when resync_model is set")
    if resync_model is not None:
        require_pure_block_fp8_checkpoint(resync_model)
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    vocab_size = model_vocab_size(config, tokenizer)
    llm = LLM(
        model=str(model),
        tensor_parallel_size=4,
        trust_remote_code=True,
        kv_cache_dtype="fp8",
        max_model_len=512,
        max_num_seqs=1,
        max_logprobs=vocab_size,
        gpu_memory_utilization=0.90,
        worker_extension_cls=(
            "verl_mlite.rollout.vllm_worker.VllmCheckpointPathWorkerExtension"
        ),
    )
    cold_weights = (
        collect_engine_weight_fingerprints(llm) if resync_model is not None else None
    )
    rows = _collect_vllm_rows(llm, tokenizer, vocab_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, output)
    print(f"DS4_TP4_FP8_COLD_COLLECT_COMPLETE={output}", flush=True)
    if resync_model is not None and resync_output is not None:
        reload_resync_checkpoint(llm, resync_model)
        online_weights = collect_engine_weight_fingerprints(llm)
        assert cold_weights is not None and weight_output is not None
        weight_report = compare_engine_weight_fingerprints(cold_weights, online_weights)
        weight_output.write_text(
            json.dumps(weight_report, indent=2, sort_keys=True) + "\n"
        )
        print(f"DS4_TP4_ENGINE_WEIGHT_REPORT={weight_output}", flush=True)
        if not weight_report["exact_match"]:
            raise AssertionError("vLLM cold-load and online-reload weights differ")
        resync_rows = _collect_vllm_rows(llm, tokenizer, vocab_size)
        torch.save(resync_rows, resync_output)
        print(f"DS4_TP4_FP8_ONLINE_RELOAD_COMPLETE={resync_output}", flush=True)


def _build_mlite_runtime(model: Path):
    from megatron.lite.runtime import RuntimeConfig, create_runtime
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.contracts import ParallelConfig

    require_mixed_flash_checkpoint(model)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if world_size != 4:
        raise ValueError(
            f"formal MLite parity requires EP4, got world_size={world_size}"
        )

    backend_config = MegatronLiteConfig(
        model_name="deepseek_v4",
        impl="lite",
        hf_path=str(model),
        parallel=ParallelConfig(tp=1, etp=1, ep=4, pp=1, vpp=1, cp=1),
        attention_backend_override="fused",
        load_hf_weights=True,
        impl_cfg={
            "optimizer": None,
            "mtp_enable": False,
            "mtp_enable_train": False,
            "use_deepep": False,
        },
    )
    runtime = create_runtime(
        RuntimeConfig(backend="mlite", hf_path=str(model), backend_cfg=backend_config)
    )
    return runtime, runtime.build_model()


def load_mlite(model: Path) -> None:
    """Build and load the official mixed checkpoint, then exit without forward/export."""
    started = time.monotonic()
    _, handle = _build_mlite_runtime(model)
    if dist.get_rank() == 0:
        print(
            "DS4_MLITE_LOAD_ONLY_COMPLETE "
            f"elapsed_seconds={time.monotonic() - started:.3f} "
            f"peak_allocated_bytes={torch.cuda.max_memory_allocated()}",
            flush=True,
        )
    dist.barrier()
    del handle


def collect_mlite(
    model: Path, output: Path, fp8_output: Path, coverage_output: Path
) -> None:
    """Load official mixed weights into MLite, run BF16 forward, and export FP8."""
    from transformers import AutoConfig, AutoTokenizer

    from megatron.lite.runtime.contracts import PackedBatch

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    vocab_size = model_vocab_size(config, tokenizer)
    runtime, handle = _build_mlite_runtime(model)
    rows = []
    with torch.inference_mode(), runtime.eval_mode(handle):
        for prompt in math_prompts():
            token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            tokens = torch.tensor(token_ids, dtype=torch.long, device="cuda")
            batch = PackedBatch(
                input_ids=tokens,
                labels=None,
                seq_lens=torch.tensor(
                    [len(token_ids)], dtype=torch.int32, device="cuda"
                ),
            )
            result = runtime.forward_backward(
                handle,
                iter([batch]),
                loss_fn=None,
                num_microbatches=1,
                forward_only=True,
            )
            logits = result.model_output.vocab_parallel_logits
            if logits is None or logits.shape[-1] != vocab_size:
                raise ValueError(
                    "MLite full-model forward did not return full-vocabulary logits"
                )
            if dist.get_rank() == 0:
                rows.append(mlite_payload_row(token_ids, logits))

    if dist.get_rank() == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rows, output)
        print(f"DS4_MLITE_BF16_FORWARD_COMPLETE={output}", flush=True)

    weights = runtime.export_weights(
        handle,
        target="vllm_checkpoint",
        export_dtype="bfloat16",
        resync_config={"expert_dtype": "fp8"},
        rank0_only=True,
    )
    if dist.get_rank() == 0:
        coverage = write_exported_checkpoint(weights, model, fp8_output)
        coverage_output.write_text(
            json.dumps(coverage, indent=2, sort_keys=True) + "\n"
        )
        print(f"DS4_MLITE_PURE_FP8_EXPORT_COMPLETE={fp8_output}", flush=True)
        print(f"DS4_MLITE_EXPORT_COVERAGE={coverage_output}", flush=True)
    else:
        for _ in weights:
            raise AssertionError("rank0_only MLite export yielded on a nonzero rank")
    dist.barrier()


def compare(
    cold: Path,
    online: Path,
    mlite: Path,
    weights: Path,
    coverage: Path,
    output: Path,
) -> None:
    weight_report = json.loads(weights.read_text())
    if not weight_report.get("exact_match"):
        raise ValueError("engine weight report is missing cold-vs-online exact parity")
    coverage_report = json.loads(coverage.read_text())
    if not coverage_report.get("exact_match"):
        raise ValueError("MLite export coverage is not exact")
    report = compare_three_arms(
        torch.load(cold, weights_only=True),
        torch.load(online, weights_only=True),
        torch.load(mlite, weights_only=True),
    )
    report["engine_weights"] = weight_report
    report["export_coverage"] = coverage_report
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    print("DS4_TP4_PARITY_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--model", type=Path, required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--model", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--resync-model", type=Path)
    collect_parser.add_argument("--resync-output", type=Path)
    collect_parser.add_argument("--weight-output", type=Path)
    load_mlite_parser = subparsers.add_parser("load-mlite")
    load_mlite_parser.add_argument("--model", type=Path, required=True)
    mlite_parser = subparsers.add_parser("collect-mlite")
    mlite_parser.add_argument("--model", type=Path, required=True)
    mlite_parser.add_argument("--output", type=Path, required=True)
    mlite_parser.add_argument("--fp8-output", type=Path, required=True)
    mlite_parser.add_argument("--coverage-output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--cold", type=Path, required=True)
    compare_parser.add_argument("--online", type=Path, required=True)
    compare_parser.add_argument("--mlite", type=Path, required=True)
    compare_parser.add_argument("--weights", type=Path, required=True)
    compare_parser.add_argument("--coverage", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        require_pure_block_fp8_checkpoint(args.model)
        print(f"DS4_PURE_BLOCK_FP8_CHECKPOINT={args.model}", flush=True)
    elif args.command == "collect":
        collect(
            args.model,
            args.output,
            resync_model=args.resync_model,
            resync_output=args.resync_output,
            weight_output=args.weight_output,
        )
    elif args.command == "load-mlite":
        load_mlite(args.model)
    elif args.command == "collect-mlite":
        collect_mlite(
            args.model, args.output, args.fp8_output, args.coverage_output
        )
    else:
        compare(
            args.cold,
            args.online,
            args.mlite,
            args.weights,
            args.coverage,
            args.output,
        )


if __name__ == "__main__":
    main()
