# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Audit an official DeepSeek-V4 Flash checkpoint through MLite resync quantization."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from megatron.lite.model.deepseek_v4.lite.resync import (
    is_release_unquantized_weight,
    is_routed_expert,
)
from megatron.lite.primitive.quantization.block_fp8 import (
    dequantize_block_fp8,
    quantize_block_fp8,
)
from megatron.lite.primitive.quantization.mxfp4 import (
    dequantize_mxfp4,
    quantize_mxfp4,
)


def _scale_name(weight_name: str) -> str:
    return f"{weight_name[:-7]}.scale"


def _layer_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] in {"layers", "mtp"}:
        return ".".join(parts[:2])
    return "global"


def _matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def _checkpoint_index(path: Path) -> tuple[dict[str, str], list[str]]:
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return weight_map, sorted(set(weight_map.values()))

    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors checkpoint found in {path}")
    weight_map: dict[str, str] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in weight_map:
                    raise ValueError(f"duplicate checkpoint tensor: {name}")
                weight_map[name] = shard.name
    return weight_map, [shard.name for shard in shards]


def _special_family(name: str) -> str | None:
    if ".attn.indexer." in name:
        return "indexer"
    if "hc_attn" in name:
        return "mhc"
    if "o_lora" in name:
        return "o_lora"
    if ".ffn.gate." in name or name.endswith(".ffn.gate.weight"):
        return "router"
    return None


def _audit_checkpoint(config: dict[str, Any], names: set[str]) -> dict[str, Any]:
    quantization = config.get("quantization_config") or {}
    configured = tuple(
        quantization.get("ignored_layers")
        or quantization.get("modules_to_not_convert")
        or ()
    )
    scaled_weights = {
        name
        for name in names
        if name.endswith(".weight") and _scale_name(name) in names
    }
    unscaled_weights = sorted(
        name
        for name in names
        if name.endswith(".weight") and _scale_name(name) not in names
    )
    matched = {
        prefix: sorted(name for name in names if _matches_prefix(name, prefix))
        for prefix in configured
    }
    missing = sorted(prefix for prefix, values in matched.items() if not values)
    if missing:
        raise ValueError(
            f"configured ignored layers have no checkpoint tensors: {missing}"
        )

    configured_unscaled = {
        name
        for prefix in configured
        for name in matched[prefix]
        if name.endswith(".weight")
    }
    violations = sorted(configured_unscaled & scaled_weights)
    unknown = sorted(
        name
        for name in unscaled_weights
        if not is_release_unquantized_weight(name)
        and not any(_matches_prefix(name, prefix) for prefix in configured)
    )
    if unknown:
        raise ValueError(f"unrecognized unscaled checkpoint weights: {unknown[:20]}")

    special = {
        family: {
            "direct_tensors": [],
            "scaled_weights": [],
            "unscaled_weights": [],
        }
        for family in ("indexer", "mhc", "o_lora", "router")
    }
    for name in sorted(names):
        family = _special_family(name)
        if family is None:
            continue
        if name in scaled_weights:
            special[family]["scaled_weights"].append(name)
        elif name in unscaled_weights:
            special[family]["unscaled_weights"].append(name)
        elif not name.endswith(".scale"):
            special[family]["direct_tensors"].append(name)

    return {
        "configured_ignored_layers": list(configured),
        "effective_ignored_weights": unscaled_weights,
        "matched": matched,
        "source": "config" if configured else "inferred_from_scale_pairs",
        "special_layers": special,
        "violations": violations,
    }


def _byte_mismatch(left: torch.Tensor, right: torch.Tensor) -> tuple[int, int]:
    left_bytes = left.contiguous().view(torch.uint8)
    right_bytes = right.contiguous().view(torch.uint8)
    if left_bytes.shape != right_bytes.shape:
        raise ValueError(
            f"serialized shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    return int((left_bytes != right_bytes).sum().item()), left_bytes.numel()


def _measure_pair(
    name: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    expert_dtype: str,
    block_shape: tuple[int, int],
    device: torch.device,
) -> dict[str, Any]:
    weight = weight.to(device)
    scale = scale.to(device)
    if expert_dtype == "fp4" and is_routed_expert(name):
        if weight.dtype != torch.int8:
            raise ValueError(
                f"MXFP4 tensor {name} has unsupported dtype {weight.dtype}"
            )
        kind = "mxfp4"
        original = dequantize_mxfp4(weight, scale)
        requantized, requantized_scale = quantize_mxfp4(original.to(torch.bfloat16))
        restored = dequantize_mxfp4(requantized, requantized_scale)
    else:
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"block-FP8 tensor {name} has unsupported dtype {weight.dtype}"
            )
        kind = "block_fp8"
        original = dequantize_block_fp8(weight, scale, block_shape)
        requantized, requantized_scale = quantize_block_fp8(
            original.to(torch.bfloat16),
            block_shape,
            scale_format="float32" if expert_dtype == "fp8" else "e8m0",
        )
        restored = dequantize_block_fp8(requantized, requantized_scale, block_shape)

    difference = restored - original
    source_l2_sq = float(original.square().sum().item())
    diff_l2_sq = float(difference.square().sum().item())
    relative_l2 = math.sqrt(diff_l2_sq / source_l2_sq) if source_l2_sq else 0.0
    scale_mismatched, scale_total = _byte_mismatch(scale, requantized_scale)
    weight_mismatched, weight_total = _byte_mismatch(weight, requantized)
    return {
        "diff_l2_sq": diff_l2_sq,
        "kind": kind,
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": relative_l2,
        "scale_byte_mismatch": {
            "mismatched": scale_mismatched,
            "total": scale_total,
        },
        "source_l2_sq": source_l2_sq,
        "weight_byte_mismatch": {
            "mismatched": weight_mismatched,
            "total": weight_total,
        },
    }


def _distribution(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "max": float(tensor.max().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p99": float(torch.quantile(tensor, 0.99).item()),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "max_abs": {},
            "relative_l2": {},
            "scale_byte_mismatch": {"mismatched": 0, "rate": 0.0, "total": 0},
            "tensor_count": 0,
            "weight_byte_mismatch": {"mismatched": 0, "rate": 0.0, "total": 0},
        }
    output: dict[str, Any] = {
        "max_abs": _distribution([record["max_abs"] for record in records]),
        "relative_l2": _distribution([record["relative_l2"] for record in records]),
        "tensor_count": len(records),
    }
    for field in ("scale_byte_mismatch", "weight_byte_mismatch"):
        mismatched = sum(record[field]["mismatched"] for record in records)
        total = sum(record[field]["total"] for record in records)
        output[field] = {
            "mismatched": mismatched,
            "rate": mismatched / total if total else 0.0,
            "total": total,
        }
    return output


def run_roundtrip(checkpoint: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    expert_dtype = config.get("expert_dtype") or (
        config.get("quantization_config") or {}
    ).get("expert_dtype", "fp4")
    if expert_dtype not in {"fp4", "fp8"}:
        raise ValueError(f"unsupported expert_dtype={expert_dtype!r}")
    raw_block_shape = (config.get("quantization_config") or {}).get(
        "weight_block_size", [128, 128]
    )
    block_shape = tuple(int(value) for value in raw_block_shape)
    weight_map, shards = _checkpoint_index(checkpoint)
    names = set(weight_map)
    audit = _audit_checkpoint(config, names)
    if audit["violations"]:
        raise ValueError(
            f"ignored checkpoint weights unexpectedly have scales: {audit['violations']}"
        )

    pair_names = sorted(
        name
        for name in names
        if name.endswith(".weight") and _scale_name(name) in names
    )
    for name in pair_names:
        scale_name = _scale_name(name)
        if weight_map[name] != weight_map[scale_name]:
            raise ValueError(f"weight and scale are split across shards: {name}")

    records: list[dict[str, Any]] = []
    target_device = torch.device(device)
    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for name in pair_names:
        names_by_shard[weight_map[name]].append(name)
    for shard in shards:
        if shard not in names_by_shard:
            continue
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            for name in names_by_shard[shard]:
                record = _measure_pair(
                    name,
                    handle.get_tensor(name),
                    handle.get_tensor(_scale_name(name)),
                    expert_dtype=expert_dtype,
                    block_shape=block_shape,
                    device=target_device,
                )
                record["layer"] = _layer_name(name)
                record["name"] = name
                records.append(record)

    by_kind = {
        kind: [record for record in records if record["kind"] == kind]
        for kind in ("block_fp8", "mxfp4")
    }
    layers: dict[str, Any] = {}
    for layer in sorted({record["layer"] for record in records}):
        layers[layer] = {
            kind: _aggregate(
                [record for record in by_kind[kind] if record["layer"] == layer]
            )
            for kind in ("block_fp8", "mxfp4")
        }
    return {
        "audit": audit,
        "checkpoint": str(checkpoint.resolve()),
        "expert_dtype": expert_dtype,
        "layers": layers,
        "shard_count": len(shards),
        "summary": {kind: _aggregate(by_kind[kind]) for kind in by_kind},
        "tensor_count": len(names),
        "tensors": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_roundtrip(args.checkpoint, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))
    print("DS4_OFFICIAL_CHECKPOINT_ROUNDTRIP_COMPLETE")


if __name__ == "__main__":
    main()
