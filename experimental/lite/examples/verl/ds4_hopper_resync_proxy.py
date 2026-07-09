# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate DS4 BF16 training to pure block-FP8 resync on an H100 proxy.

The full DeepSeek-V4 checkpoint is too large to train on a small Hopper
allocation.  This proxy therefore loads one scaled dense matrix and one routed
expert matrix from the official mixed checkpoint, dequantizes both into BF16,
runs a deterministic short training loop, and exports the updated matrices
through the production DS4 resync adapter with ``expert_dtype=fp8``.  The same
token inputs are then evaluated with the BF16 weights and the dequantized FP8
export.  This validates the resync mechanism, not full-model quality parity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

from examples.verl.ds4_checkpoint_roundtrip import _checkpoint_index
from examples.verl.ds4_resync_tp4 import (
    _DAPO_MAX_RATIO_LIMIT,
    _DAPO_P99_KL_LIMIT,
    _DAPO_P99_RATIO_LIMIT,
    compare_distributions,
    payload_row,
)
from megatron.lite.model.deepseek_v4.lite.resync import (
    export_resync_weights,
    is_release_unquantized_weight,
    is_routed_expert,
)
from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4


_PROXY_SHAPE = (128, 128)


def _scale_name(weight_name: str) -> str:
    return f"{weight_name[:-7]}.scale"


def select_proxy_weights(weight_map: Mapping[str, str]) -> dict[str, str]:
    """Select deterministic dense and expert scale pairs from a DS4 index."""
    candidates = [
        name
        for name in sorted(weight_map)
        if name.endswith(".weight")
        and _scale_name(name) in weight_map
        and weight_map[name] == weight_map[_scale_name(name)]
        and not is_release_unquantized_weight(name)
    ]
    dense = next((name for name in candidates if not is_routed_expert(name)), None)
    expert = next((name for name in candidates if is_routed_expert(name)), None)
    if dense is None or expert is None:
        raise ValueError(
            "official DS4 proxy requires one dense and one routed-expert scale pair"
        )
    return {"dense": dense, "routed_expert": expert}


def crop_source_pair(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    source_kind: str,
    block_shape: tuple[int, int],
    proxy_shape: tuple[int, int] = _PROXY_SHAPE,
) -> torch.Tensor:
    """Dequantize a small block-aligned crop without materializing a full matrix."""
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("DS4 proxy only supports two-dimensional checkpoint matrices")
    rows, columns = proxy_shape
    block_rows, block_columns = block_shape
    if rows <= 0 or columns <= 0:
        raise ValueError(f"proxy shape must be positive, got {proxy_shape}")
    if rows % block_rows or columns % block_columns:
        raise ValueError(
            f"proxy shape {proxy_shape} must align to block shape {block_shape}"
        )

    if source_kind == "block_fp8":
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"block-FP8 source must use float8_e4m3fn, got {weight.dtype}"
            )
        if weight.shape[0] < rows or weight.shape[1] < columns:
            raise ValueError(
                "block-FP8 source is smaller than the requested proxy crop"
            )
        cropped_weight = weight[:rows, :columns].contiguous()
        cropped_scale = scale[
            : rows // block_rows, : columns // block_columns
        ].contiguous()
        restored = dequantize_block_fp8(
            cropped_weight, cropped_scale, block_shape=block_shape
        )
    elif source_kind == "mxfp4":
        if weight.dtype != torch.int8:
            raise TypeError(f"MXFP4 source must use int8 packing, got {weight.dtype}")
        if columns % 32:
            raise ValueError("MXFP4 proxy columns must be divisible by 32")
        if weight.shape[0] < rows or weight.shape[1] * 2 < columns:
            raise ValueError("MXFP4 source is smaller than the requested proxy crop")
        cropped_weight = weight[:rows, : columns // 2].contiguous()
        cropped_scale = scale[:rows, : columns // 32].contiguous()
        restored = dequantize_mxfp4(cropped_weight, cropped_scale)
    else:
        raise ValueError(f"unsupported DS4 proxy source kind: {source_kind!r}")

    if tuple(restored.shape) != proxy_shape:
        raise ValueError(
            f"dequantized proxy shape {tuple(restored.shape)} != {proxy_shape}"
        )
    return restored.to(torch.bfloat16)


def _load_official_proxy(
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, torch.Tensor], dict[str, str]]:
    config = json.loads((checkpoint / "config.json").read_text())
    quantization = config.get("quantization_config") or {}
    source_expert_dtype = config.get("expert_dtype") or quantization.get(
        "expert_dtype", "fp4"
    )
    if source_expert_dtype != "fp4":
        raise ValueError(
            "Hopper proxy expects the official mixed checkpoint with FP4 experts; "
            f"got expert_dtype={source_expert_dtype!r}"
        )
    raw_block_shape = quantization.get("weight_block_size", [128, 128])
    block_shape = tuple(int(value) for value in raw_block_shape)
    if len(block_shape) != 2:
        raise ValueError(f"invalid checkpoint block shape: {raw_block_shape}")

    weight_map, _ = _checkpoint_index(checkpoint)
    selected = select_proxy_weights(weight_map)
    values: dict[str, torch.Tensor] = {}
    source_kinds: dict[str, str] = {}
    for role, name in selected.items():
        source_kind = "mxfp4" if role == "routed_expert" else "block_fp8"
        shard = checkpoint / weight_map[name]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            values[name] = crop_source_pair(
                handle.get_tensor(name),
                handle.get_tensor(_scale_name(name)),
                source_kind=source_kind,
                block_shape=block_shape,
            )
        source_kinds[name] = source_kind
    return config, selected, values, source_kinds


def fixed_token_sequences(vocab_size: int) -> list[list[int]]:
    if vocab_size < 32:
        raise ValueError("DS4 Hopper proxy vocabulary must contain at least 32 entries")
    return [
        [int((row * 31 + column * 17 + 3) % vocab_size) for column in range(12)]
        for row in range(8)
    ]


def _proxy_logits(
    dense: torch.Tensor, expert: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    inputs = F.one_hot(token_ids, num_classes=dense.shape[1]).to(dense.dtype)
    hidden = F.silu(F.linear(inputs, dense) / math.sqrt(dense.shape[1]))
    return F.linear(hidden, expert) / math.sqrt(expert.shape[1])


def _proxy_loss(
    dense: torch.Tensor,
    expert: torch.Tensor,
    sequences: list[list[int]],
) -> torch.Tensor:
    losses = []
    for sequence in sequences:
        ids = torch.tensor(sequence, dtype=torch.long, device=dense.device)
        logits = _proxy_logits(dense, expert, ids[:-1])
        losses.append(F.cross_entropy(logits.float(), ids[1:]))
    return torch.stack(losses).mean()


def _proxy_payload(
    dense: torch.Tensor,
    expert: torch.Tensor,
    sequences: list[list[int]],
) -> list[dict[str, torch.Tensor]]:
    rows = []
    with torch.inference_mode():
        for sequence in sequences:
            ids = torch.tensor(sequence, dtype=torch.long, device=dense.device)
            logits = _proxy_logits(dense, expert, ids[:-1])
            rows.append(payload_row(ids[1:], torch.log_softmax(logits.float(), dim=-1)))
    return rows


def _train_proxy(
    selected: Mapping[str, str],
    source: Mapping[str, torch.Tensor],
    sequences: list[list[int]],
    *,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if steps < 1:
        raise ValueError("DS4 Hopper proxy requires at least one BF16 training step")
    dense = torch.nn.Parameter(source[selected["dense"]].to(device))
    expert = torch.nn.Parameter(source[selected["routed_expert"]].to(device))
    initial = {
        selected["dense"]: dense.detach().float().clone(),
        selected["routed_expert"]: expert.detach().float().clone(),
    }
    optimizer = torch.optim.AdamW(
        [dense, expert], lr=learning_rate, weight_decay=0.0, foreach=False
    )
    loss_trace: list[float] = []
    grad_norm_trace: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _proxy_loss(dense, expert, sequences)
        if not torch.isfinite(loss):
            raise ValueError("DS4 Hopper proxy produced a non-finite BF16 loss")
        loss_trace.append(float(loss.detach()))
        loss.backward()
        grad_norm = torch.linalg.vector_norm(
            torch.stack((dense.grad.float().norm(), expert.grad.float().norm()))
        )
        if not torch.isfinite(grad_norm) or grad_norm.item() <= 0:
            raise ValueError("DS4 Hopper proxy produced an invalid gradient norm")
        grad_norm_trace.append(float(grad_norm))
        optimizer.step()

    final_loss = _proxy_loss(dense, expert, sequences)
    loss_trace.append(float(final_loss.detach()))
    trained = {
        selected["dense"]: dense.detach(),
        selected["routed_expert"]: expert.detach(),
    }
    param_delta = sum(
        float((trained[name].float() - initial[name]).abs().sum()) for name in trained
    )
    if not math.isfinite(param_delta) or param_delta <= 0:
        raise ValueError("DS4 Hopper proxy optimizer did not change parameters")
    if loss_trace[-1] > loss_trace[0]:
        raise ValueError(
            f"DS4 Hopper proxy loss increased from {loss_trace[0]} to {loss_trace[-1]}"
        )
    return trained, {
        "steps": steps,
        "learning_rate": learning_rate,
        "loss_trace": loss_trace,
        "grad_norm_trace": grad_norm_trace,
        "param_delta_sum": param_delta,
    }


def export_trained_proxy(
    weights: Mapping[str, torch.Tensor], config: Any
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Export both proxy matrices through the production pure-FP8 adapter."""
    exported = dict(
        export_resync_weights(
            weights.items(), config, resync_config={"expert_dtype": "fp8"}
        )
    )
    block_shape = tuple(
        int(value) for value in config.quantization_config["weight_block_size"]
    )
    restored: dict[str, torch.Tensor] = {}
    for name, source in weights.items():
        scale_name = _scale_name(name)
        quantized = exported.get(name)
        scale = exported.get(scale_name)
        if quantized is None or scale is None:
            raise ValueError(f"pure-FP8 export omitted proxy tensor {name}")
        if quantized.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
            raise TypeError(
                f"pure-FP8 export has wrong dtype for {name}: "
                f"weight={quantized.dtype}, scale={scale.dtype}"
            )
        restored[name] = dequantize_block_fp8(
            quantized, scale, block_shape=block_shape
        ).to(device=source.device, dtype=torch.bfloat16)
    return exported, restored


def _weight_differences(
    trained: Mapping[str, torch.Tensor], restored: Mapping[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    output = {}
    for name, reference in trained.items():
        delta = restored[name].float() - reference.float()
        denominator = reference.float().norm().clamp_min(1e-30)
        output[name] = {
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / denominator),
        }
    return output


def evaluate_proxy_gate(comparison: Mapping[str, Any]) -> dict[str, Any]:
    metrics = comparison["fp32"]
    failures = []
    if metrics["p99_ratio_deviation"] > _DAPO_P99_RATIO_LIMIT:
        failures.append("p99 ratio deviation")
    if metrics["max_ratio_deviation"] > _DAPO_MAX_RATIO_LIMIT:
        failures.append("max ratio deviation")
    if metrics["p99_kl"] > _DAPO_P99_KL_LIMIT:
        failures.append("p99 KL")
    if metrics["clipping_boundary_crossings"]:
        failures.append("clipping boundary crossings")
    return {
        "acceptable": not failures,
        "failures": failures,
        "thresholds": {
            "p99_ratio_deviation": _DAPO_P99_RATIO_LIMIT,
            "max_ratio_deviation": _DAPO_MAX_RATIO_LIMIT,
            "p99_kl": _DAPO_P99_KL_LIMIT,
            "clipping_boundary_crossings": 0,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(
    checkpoint: Path,
    output_dir: Path,
    *,
    steps: int = 4,
    learning_rate: float = 1e-2,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("DS4 Hopper resync proxy requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise RuntimeError(
            f"DS4 Hopper proxy requires SM90, got SM{capability[0]}{capability[1]}"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    torch.manual_seed(20260709)
    config_dict, selected, source, source_kinds = _load_official_proxy(checkpoint)
    sequences = fixed_token_sequences(_PROXY_SHAPE[1])
    torch.cuda.synchronize()
    started = time.perf_counter()
    trained, training = _train_proxy(
        selected,
        source,
        sequences,
        steps=steps,
        learning_rate=learning_rate,
        device=device,
    )
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started

    proxy_config = SimpleNamespace(
        expert_dtype=config_dict.get("expert_dtype", "fp4"),
        quantization_config={"weight_block_size": list(_PROXY_SHAPE)},
    )
    started = time.perf_counter()
    exported, restored = export_trained_proxy(trained, proxy_config)
    torch.cuda.synchronize()
    resync_seconds = time.perf_counter() - started

    bf16_rows = _proxy_payload(
        trained[selected["dense"]], trained[selected["routed_expert"]], sequences
    )
    fp8_rows = _proxy_payload(
        restored[selected["dense"]], restored[selected["routed_expert"]], sequences
    )
    comparison = compare_distributions(bf16_rows, fp8_rows)
    gate = evaluate_proxy_gate(comparison)
    torch.save(bf16_rows, output_dir / "bf16-trained.pt")
    torch.save(fp8_rows, output_dir / "fp8-export.pt")

    report = {
        "schema_version": 1,
        "scope": (
            "official-checkpoint tensor proxy; does not replace full-model "
            "HSG/GB200 validation"
        ),
        "checkpoint": str(checkpoint.resolve()),
        "environment": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
            "torch": torch.__version__,
        },
        "source": {
            "expert_dtype": config_dict.get("expert_dtype", "fp4"),
            "selected": selected,
            "kinds": source_kinds,
        },
        "training": {**training, "seconds": training_seconds},
        "export": {
            "target_expert_dtype": "fp8",
            "weight_dtype": str(exported[selected["dense"]].dtype),
            "expert_weight_dtype": str(exported[selected["routed_expert"]].dtype),
            "scale_dtype": str(exported[_scale_name(selected["dense"])].dtype),
            "seconds": resync_seconds,
        },
        "weight_diff": _weight_differences(trained, restored),
        "logprobs": comparison,
        "gate": gate,
        "arm_semantics": {
            "arm_a": "official mixed tensors loaded to BF16 and updated by short training",
            "arm_b": "the same updated tensors exported as pure block-FP8 and dequantized",
        },
    }
    _write_json(output_dir / "report.json", report)
    if not gate["acceptable"]:
        raise RuntimeError(
            f"DS4 Hopper resync proxy failed DAPO gate: {gate['failures']}"
        )
    marker = output_dir / "DS4_HOPPER_RESYNC_PROXY_COMPLETE"
    marker.write_text("complete\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    metrics = comparison["fp32"]
    print(
        "DS4_HOPPER_RESYNC_PROXY_COMPLETE "
        f"loss_initial={training['loss_trace'][0]:.6e} "
        f"loss_final={training['loss_trace'][-1]:.6e} "
        f"param_delta_sum={training['param_delta_sum']:.6e} "
        f"p99_kl={metrics['p99_kl']:.6e} "
        f"p99_ratio_deviation={metrics['p99_ratio_deviation']:.6e} "
        f"clipping_crossings={metrics['clipping_boundary_crossings']}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    args = parser.parse_args()
    run(
        args.checkpoint,
        args.output_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
