#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Measure element-wise deltas between two tensor checkpoints.

The analyzer is deliberately CPU-only and reads tensors one at a time.  It
accepts exported safetensors or a trusted PyTorch distributed-checkpoint model
state.  Optional block-FP8 statistics serialize eligible matrices as E4M3
weights plus FP32 block scales, matching the rollout resync wire contract.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import re
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from safetensors import safe_open

try:
    import zstandard
except ImportError:  # pragma: no cover - the other statistics remain useful.
    zstandard = None


DEFAULT_THRESHOLDS = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
DEFAULT_MAGNITUDE_EDGES = (
    0.0,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    math.inf,
)
DEFAULT_FP8_FAMILIES = ("attention", "dense_mlp", "expert")
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|mtp)\.(\d+)(?:\.|$)")


def _number_key(value: float) -> str:
    return format(value, ".12g")


def _target_dtype_name(dtype: torch.dtype) -> str:
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}.get(
        dtype, str(dtype).removeprefix("torch.")
    )


def classify_parameter_family(name: str) -> str:
    """Classify an exported parameter name for reporting, not dispatch."""

    lowered = name.lower()
    if lowered.endswith("lm_head.weight") or "output_layer" in lowered:
        return "head"
    if any(token in lowered for token in ("embed_tokens", "word_embeddings", "wte")):
        return "embedding"
    if lowered.endswith("embed.embedding.weight"):
        return "embedding"
    if (
        ".router." in lowered
        or ".mlp.gate." in lowered
        or "shared_expert_gate" in lowered
        or lowered.endswith("router.weight")
    ):
        return "router"
    if any(token in lowered for token in (".experts.", ".expert.", "shared_expert")):
        return "expert"
    if "norm" in lowered:
        return "norm"
    if any(
        token in lowered
        for token in (
            ".self_attn.",
            ".linear_attn.",
            ".attention.",
            ".attn.",
            ".q_proj.",
            ".k_proj.",
            ".v_proj.",
            ".o_proj.",
        )
    ):
        return "attention"
    if any(token in lowered for token in (".mlp.", "feed_forward", ".ffn.")):
        return "dense_mlp"
    return "other"


def _layer_index(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _layer_depth(index: int | None, layer_count: int) -> str:
    if index is None:
        return "global"
    bucket = min(2, index * 3 // max(layer_count, 1))
    return ("shallow", "middle", "deep")[bucket]


def _is_weight_tensor_name(name: str) -> bool:
    return name.endswith(".weight") or re.search(r"\.weight\d+$", name) is not None


def _quantize_block_fp8(
    tensor: torch.Tensor, block_shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serialize one matrix like the rollout block-FP8 checkpoint contract."""

    if tensor.ndim < 2:
        raise ValueError(
            f"block-FP8 tensor must have at least two dimensions: {tensor.shape}"
        )
    block_m, block_k = block_shape
    if block_m <= 0 or block_k <= 0:
        raise ValueError(f"block shape must be positive, got {block_shape}")
    if tensor.shape[-2] % block_m or tensor.shape[-1] % block_k:
        raise ValueError(
            f"tensor trailing shape {tuple(tensor.shape[-2:])} must be divisible by "
            f"block shape {block_shape}"
        )

    source = tensor.float()
    *leading, rows, columns = source.shape
    blocked = source.reshape(
        *leading,
        rows // block_m,
        block_m,
        columns // block_k,
        block_k,
    )
    amax = blocked.abs().amax(dim=(-3, -1))
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (amax.clamp_min(1e-4) / fp8_max).contiguous()
    expanded = scale.repeat_interleave(block_m, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    weight = (source / expanded).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return weight, scale


def _direct_target_statistics(
    changed_count: int, numel: int, value_bytes: int, *, kind: str, dtype: str
) -> dict[str, object]:
    serialized_bytes = numel * value_bytes
    changed_value_bytes = changed_count * value_bytes
    return {
        "kind": kind,
        "weight_dtype": dtype,
        "scale_dtype": None,
        "weight_numel": numel,
        "weight_changed_count": changed_count,
        "scale_numel": 0,
        "scale_changed_count": 0,
        "serialized_bytes": serialized_bytes,
        "changed_value_bytes": changed_value_bytes,
        "changed_value_fraction": changed_value_bytes / serialized_bytes
        if serialized_bytes
        else 0.0,
        "bitmap_value_bytes": math.ceil(numel / 8) + changed_value_bytes,
    }


def _block_fp8_target_statistics(
    name: str,
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    block_shape: tuple[int, int],
    quantized_families: Sequence[str],
    exact_changed_count: int,
) -> dict[str, object]:
    family = classify_parameter_family(name)
    if (
        family not in set(quantized_families)
        or not _is_weight_tensor_name(name)
        or before.ndim < 2
    ):
        dtype_name = _target_dtype_name(before.dtype)
        return _direct_target_statistics(
            exact_changed_count,
            before.numel(),
            before.element_size(),
            kind=f"passthrough_{dtype_name}",
            dtype=dtype_name,
        )

    before_weight, before_scale = _quantize_block_fp8(before, block_shape)
    after_weight, after_scale = _quantize_block_fp8(after, block_shape)
    weight_changed = int(
        torch.count_nonzero(
            before_weight.contiguous().view(torch.uint8)
            != after_weight.contiguous().view(torch.uint8)
        ).item()
    )
    scale_changed = int(torch.count_nonzero(before_scale != after_scale).item())
    serialized_bytes = before_weight.numel() + before_scale.numel() * 4
    changed_value_bytes = weight_changed + scale_changed * 4
    return {
        "kind": "quantized",
        "weight_dtype": "float8_e4m3fn",
        "scale_dtype": "float32",
        "weight_numel": before_weight.numel(),
        "weight_changed_count": weight_changed,
        "scale_numel": before_scale.numel(),
        "scale_changed_count": scale_changed,
        "serialized_bytes": serialized_bytes,
        "changed_value_bytes": changed_value_bytes,
        "changed_value_fraction": changed_value_bytes / serialized_bytes
        if serialized_bytes
        else 0.0,
        "bitmap_value_bytes": math.ceil(before_weight.numel() / 8)
        + weight_changed
        + math.ceil(before_scale.numel() / 8)
        + scale_changed * 4,
    }


def _validate_bins(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(value < 0 for value in result):
        raise ValueError(f"{name} must contain non-negative values")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be strictly increasing")
    return result


def tensor_delta_statistics(
    name: str,
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    magnitude_edges: Sequence[float] = DEFAULT_MAGNITUDE_EDGES,
    chunk_elements: int = 8 * 1024 * 1024,
    fp8_block_shape: tuple[int, int] | None = None,
    fp8_quantized_families: Sequence[str] = DEFAULT_FP8_FAMILIES,
) -> dict[str, object]:
    """Return bounded-memory delta statistics for a same-shape tensor pair."""

    thresholds = _validate_bins(thresholds, name="thresholds")
    magnitude_edges = _validate_bins(magnitude_edges, name="magnitude_edges")
    if thresholds[0] != 0.0:
        raise ValueError("thresholds must start at zero for lossless density")
    if magnitude_edges[0] != 0.0 or not math.isinf(magnitude_edges[-1]):
        raise ValueError("magnitude_edges must span zero through infinity")
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    if before.shape != after.shape:
        raise ValueError(f"shape mismatch for {name}: {before.shape} != {after.shape}")
    if before.dtype != after.dtype:
        raise ValueError(f"dtype mismatch for {name}: {before.dtype} != {after.dtype}")
    if not before.dtype.is_floating_point:
        raise TypeError(
            f"non-floating checkpoint tensor is unsupported: {name} {before.dtype}"
        )

    threshold_keys = [_number_key(value) for value in thresholds]
    changed_counts = {key: 0 for key in threshold_keys}
    histogram = [0] * (len(magnitude_edges) - 1)
    delta_square_sum = 0.0
    before_square_sum = 0.0
    l_inf = 0.0
    xor_nonzero_bytes = 0
    xor_compressor = (
        zstandard.ZstdCompressor(level=3).compressobj() if zstandard else None
    )
    xor_zstd_bytes = 0 if xor_compressor else None
    flat_before = before.reshape(-1)
    flat_after = after.reshape(-1)

    for start in range(0, flat_before.numel(), chunk_elements):
        stop = min(start + chunk_elements, flat_before.numel())
        before_raw = flat_before[start:stop].contiguous()
        after_raw = flat_after[start:stop].contiguous()
        xor_bytes = torch.bitwise_xor(
            before_raw.view(torch.uint8), after_raw.view(torch.uint8)
        )
        xor_nonzero_bytes += int(torch.count_nonzero(xor_bytes).item())
        if xor_compressor is not None:
            xor_zstd_bytes += len(xor_compressor.compress(xor_bytes.numpy().tobytes()))

        before_chunk = before_raw.to(torch.float32)
        after_chunk = after_raw.to(torch.float32)
        absolute = (after_chunk - before_chunk).abs()
        l_inf = max(l_inf, float(absolute.max().item()) if absolute.numel() else 0.0)
        delta_square_sum += float(
            torch.sum(absolute * absolute, dtype=torch.float64).item()
        )
        before_square_sum += float(
            torch.sum(before_chunk * before_chunk, dtype=torch.float64).item()
        )
        for threshold, key in zip(thresholds, threshold_keys, strict=True):
            changed_counts[key] += int(torch.count_nonzero(absolute > threshold).item())

        nonzero = absolute[absolute > 0]
        if nonzero.numel():
            boundaries = torch.tensor(
                magnitude_edges[1:-1], dtype=nonzero.dtype, device=nonzero.device
            )
            bin_ids = torch.bucketize(nonzero, boundaries, right=False)
            counts = torch.bincount(bin_ids, minlength=len(histogram)).tolist()
            histogram = [
                left + int(right) for left, right in zip(histogram, counts, strict=True)
            ]

    if xor_compressor is not None:
        xor_zstd_bytes += len(xor_compressor.flush())

    numel = flat_before.numel()
    value_bytes = before.element_size()
    fractions = {
        key: count / numel if numel else 0.0 for key, count in changed_counts.items()
    }
    bitmap_value_bytes = {
        key: math.ceil(numel / 8) + count * value_bytes
        for key, count in changed_counts.items()
    }
    coo32_value_bytes = {
        key: count * (4 + value_bytes) for key, count in changed_counts.items()
    }
    l2 = math.sqrt(delta_square_sum)
    before_l2 = math.sqrt(before_square_sum)
    exact_changed_count = changed_counts[threshold_keys[0]]
    target_dtype = _target_dtype_name(before.dtype)
    target_formats = {
        target_dtype: _direct_target_statistics(
            exact_changed_count,
            numel,
            value_bytes,
            kind="direct",
            dtype=target_dtype,
        )
    }
    if fp8_block_shape is not None:
        target_formats["block_fp8"] = _block_fp8_target_statistics(
            name,
            before,
            after,
            block_shape=fp8_block_shape,
            quantized_families=fp8_quantized_families,
            exact_changed_count=exact_changed_count,
        )

    return {
        "name": name,
        "family": classify_parameter_family(name),
        "shape": list(before.shape),
        "dtype": str(before.dtype).removeprefix("torch."),
        "numel": numel,
        "dense_bytes": numel * value_bytes,
        "value_bytes": value_bytes,
        "xor_nonzero_bytes": xor_nonzero_bytes,
        "xor_nonzero_byte_fraction": xor_nonzero_bytes / (numel * value_bytes)
        if numel
        else 0.0,
        "xor_zstd_bytes": xor_zstd_bytes,
        "l_inf": l_inf,
        "l2": l2,
        "reference_l2": before_l2,
        "relative_l2": l2 / before_l2 if before_l2 else (0.0 if l2 == 0.0 else None),
        "changed_counts": changed_counts,
        "changed_fractions": fractions,
        "bitmap_value_bytes": bitmap_value_bytes,
        "coo32_value_bytes": coo32_value_bytes,
        "magnitude_histogram": histogram,
        "target_formats": target_formats,
    }


class _SafetensorCheckpoint:
    format = "safetensors"

    def __init__(self, root: Path):
        self.root = root
        self.index: dict[str, Path] = {}
        for path in sorted(root.rglob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in self.index:
                        raise ValueError(
                            f"duplicate tensor {key!r} in {self.index[key]} and {path}"
                        )
                    self.index[key] = path

    @property
    def names(self) -> set[str]:
        return set(self.index)

    def ordered_names(self) -> list[str]:
        return sorted(self.index, key=lambda name: (str(self.index[name]), name))

    def get_tensor(self, name: str) -> torch.Tensor:
        with safe_open(self.index[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


class _DistributedCheckpoint:
    format = "torch_dcp"

    def __init__(self, root: Path):
        self.root = root
        # DCP metadata is pickle. Only point this analyzer at trusted training
        # artifacts produced by the same organization.
        with (root / ".metadata").open("rb") as stream:
            metadata = pickle.load(stream)  # noqa: S301
        self.index = {
            name: item
            for name, item in metadata.state_dict_metadata.items()
            if isinstance(item, TensorStorageMetadata)
            and item.properties.dtype.is_floating_point
        }
        self.storage = {
            name: sorted(
                (
                    (index, info)
                    for index, info in metadata.storage_data.items()
                    if index.fqn == name
                ),
                key=lambda item: (item[1].relative_path, item[1].offset),
            )
            for name in self.index
        }

    @property
    def names(self) -> set[str]:
        return set(self.index)

    def ordered_names(self) -> list[str]:
        return sorted(
            self.index,
            key=lambda name: (
                self.storage[name][0][1].relative_path,
                self.storage[name][0][1].offset,
            ),
        )

    def get_tensor(self, name: str) -> torch.Tensor:
        metadata = self.index[name]
        tensor = torch.empty(tuple(metadata.size), dtype=metadata.properties.dtype)
        chunk_sizes = {
            tuple(chunk.offsets): tuple(chunk.sizes) for chunk in metadata.chunks
        }
        for index, storage in self.storage[name]:
            with (self.root / storage.relative_path).open("rb") as stream:
                stream.seek(storage.offset)
                payload = io.BytesIO(stream.read(storage.length))
            chunk = torch.load(payload, map_location="cpu", weights_only=True)
            offsets = tuple(index.offset or (0,) * tensor.ndim)
            expected = chunk_sizes[offsets]
            if tuple(chunk.shape) != expected:
                raise ValueError(
                    f"DCP chunk shape mismatch for {name} at {offsets}: "
                    f"{tuple(chunk.shape)} != {expected}"
                )
            slices = tuple(
                slice(offset, offset + size)
                for offset, size in zip(offsets, expected, strict=True)
            )
            tensor[slices].copy_(chunk)
        return tensor


def _checkpoint_reader(root: Path) -> _SafetensorCheckpoint | _DistributedCheckpoint:
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    if (root / ".metadata").is_file():
        reader = _DistributedCheckpoint(root)
    else:
        reader = _SafetensorCheckpoint(root)
    if not reader.names:
        raise FileNotFoundError(
            f"no floating tensor state found in safetensors or DCP below {root}"
        )
    return reader


def _empty_aggregate(
    threshold_keys: Iterable[str], histogram_bins: int
) -> dict[str, object]:
    keys = tuple(threshold_keys)
    return {
        "tensor_count": 0,
        "numel": 0,
        "dense_bytes": 0,
        "l_inf": 0.0,
        "delta_square_sum": 0.0,
        "before_square_sum": 0.0,
        "xor_nonzero_bytes": 0,
        "xor_zstd_bytes": 0,
        "changed_counts": {key: 0 for key in keys},
        "changed_value_bytes": {key: 0 for key in keys},
        "magnitude_histogram": [0] * histogram_bins,
        "target_formats": {},
    }


def _add_tensor(aggregate: dict[str, object], stats: dict[str, object]) -> None:
    aggregate["tensor_count"] += 1
    aggregate["numel"] += stats["numel"]
    aggregate["dense_bytes"] += stats["dense_bytes"]
    aggregate["l_inf"] = max(aggregate["l_inf"], stats["l_inf"])
    aggregate["delta_square_sum"] += stats["l2"] ** 2
    aggregate["before_square_sum"] += stats["reference_l2"] ** 2
    aggregate["xor_nonzero_bytes"] += stats["xor_nonzero_bytes"]
    if aggregate["xor_zstd_bytes"] is not None:
        if stats["xor_zstd_bytes"] is None:
            aggregate["xor_zstd_bytes"] = None
        else:
            aggregate["xor_zstd_bytes"] += stats["xor_zstd_bytes"]
    for key, count in stats["changed_counts"].items():
        aggregate["changed_counts"][key] += count
        aggregate["changed_value_bytes"][key] += count * stats["value_bytes"]
    aggregate["magnitude_histogram"] = [
        left + right
        for left, right in zip(
            aggregate["magnitude_histogram"], stats["magnitude_histogram"], strict=True
        )
    ]
    for name, target_stats in stats["target_formats"].items():
        target = aggregate["target_formats"].setdefault(
            name,
            {
                "tensor_count": 0,
                "quantized_tensor_count": 0,
                "passthrough_tensor_count": 0,
                "weight_numel": 0,
                "weight_changed_count": 0,
                "scale_numel": 0,
                "scale_changed_count": 0,
                "serialized_bytes": 0,
                "changed_value_bytes": 0,
                "bitmap_value_bytes": 0,
            },
        )
        target["tensor_count"] += 1
        if target_stats["kind"] == "quantized":
            target["quantized_tensor_count"] += 1
        else:
            target["passthrough_tensor_count"] += 1
        for field in (
            "weight_numel",
            "weight_changed_count",
            "scale_numel",
            "scale_changed_count",
            "serialized_bytes",
            "changed_value_bytes",
            "bitmap_value_bytes",
        ):
            target[field] += target_stats[field]


def _finish_aggregate(aggregate: dict[str, object]) -> dict[str, object]:
    numel = aggregate.pop("numel")
    delta_square_sum = aggregate.pop("delta_square_sum")
    before_square_sum = aggregate.pop("before_square_sum")
    changed_value_bytes = aggregate.pop("changed_value_bytes")
    changed_counts = aggregate["changed_counts"]
    l2 = math.sqrt(delta_square_sum)
    before_l2 = math.sqrt(before_square_sum)
    for target in aggregate["target_formats"].values():
        target["weight_changed_fraction"] = (
            target["weight_changed_count"] / target["weight_numel"]
            if target["weight_numel"]
            else 0.0
        )
        target["scale_changed_fraction"] = (
            target["scale_changed_count"] / target["scale_numel"]
            if target["scale_numel"]
            else 0.0
        )
        target["changed_value_fraction"] = (
            target["changed_value_bytes"] / target["serialized_bytes"]
            if target["serialized_bytes"]
            else 0.0
        )
        target["bitmap_value_fraction"] = (
            target["bitmap_value_bytes"] / target["serialized_bytes"]
            if target["serialized_bytes"]
            else 0.0
        )
    aggregate.update(
        {
            "numel": numel,
            "l2": l2,
            "relative_l2": l2 / before_l2
            if before_l2
            else (0.0 if l2 == 0.0 else None),
            "xor_nonzero_byte_fraction": aggregate["xor_nonzero_bytes"]
            / aggregate["dense_bytes"]
            if aggregate["dense_bytes"]
            else 0.0,
            "changed_fractions": {
                key: count / numel if numel else 0.0
                for key, count in changed_counts.items()
            },
            "bitmap_value_bytes": {
                key: math.ceil(numel / 8) + changed_value_bytes[key]
                for key in changed_counts
            },
            "coo32_value_bytes": {
                key: count * 4 + changed_value_bytes[key]
                for key, count in changed_counts.items()
            },
        }
    )
    return aggregate


def _layer_concentration(layers: dict[str, dict[str, object]]) -> dict[str, object]:
    transformer_layers = [
        value for key, value in layers.items() if key.startswith("layer.")
    ]
    changes = sorted(
        (int(layer["changed_counts"]["0"]) for layer in transformer_layers),
        reverse=True,
    )
    total = sum(changes)
    layer_count = len(changes)
    top_count = max(1, math.ceil(layer_count * 0.1)) if layer_count else 0
    running = 0
    layers_for_80pct = 0
    if total:
        for layers_for_80pct, count in enumerate(changes, start=1):
            running += count
            if running / total >= 0.8:
                break
    return {
        "layer_count": layer_count,
        "total_changed_values": total,
        "layers_for_80pct_changes": layers_for_80pct,
        "fraction_of_layers_for_80pct_changes": layers_for_80pct / layer_count
        if layer_count
        else 0.0,
        "top_10pct_layer_count": top_count,
        "top_10pct_change_share": sum(changes[:top_count]) / total if total else 0.0,
    }


def analyze_checkpoints(
    before_root: str | Path,
    after_root: str | Path,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    magnitude_edges: Sequence[float] = DEFAULT_MAGNITUDE_EDGES,
    chunk_elements: int = 8 * 1024 * 1024,
    fp8_block_shape: tuple[int, int] | None = None,
    fp8_quantized_families: Sequence[str] = DEFAULT_FP8_FAMILIES,
) -> dict[str, object]:
    """Compare matching tensors below two checkpoint directories."""

    before_root = Path(before_root).resolve()
    after_root = Path(after_root).resolve()
    if before_root == after_root:
        raise ValueError("before and after checkpoints must be different directories")
    thresholds = _validate_bins(thresholds, name="thresholds")
    magnitude_edges = _validate_bins(magnitude_edges, name="magnitude_edges")
    before_reader = _checkpoint_reader(before_root)
    after_reader = _checkpoint_reader(after_root)
    if before_reader.format != after_reader.format:
        raise ValueError(
            f"checkpoint formats differ: {before_reader.format} != {after_reader.format}"
        )
    missing_after = sorted(before_reader.names - after_reader.names)
    missing_before = sorted(after_reader.names - before_reader.names)
    if missing_after or missing_before:
        raise ValueError(
            f"checkpoint tensor sets differ: missing_after={missing_after[:8]}, "
            f"missing_before={missing_before[:8]}"
        )

    threshold_keys = [_number_key(value) for value in thresholds]
    summary = _empty_aggregate(threshold_keys, len(magnitude_edges) - 1)
    families: dict[str, dict[str, object]] = {}
    depths: dict[str, dict[str, object]] = {}
    layers: dict[str, dict[str, object]] = {}
    family_depths: dict[str, dict[str, dict[str, object]]] = {}
    tensors = []
    layer_indices = [
        index
        for name in before_reader.names
        if (index := _layer_index(name)) is not None
    ]
    layer_count = max(layer_indices, default=-1) + 1

    def add_to(
        mapping: dict[str, dict[str, object]], key: str, stats: dict[str, object]
    ) -> None:
        if key not in mapping:
            mapping[key] = _empty_aggregate(threshold_keys, len(magnitude_edges) - 1)
        _add_tensor(mapping[key], stats)

    for name in before_reader.ordered_names():
        before = before_reader.get_tensor(name)
        after = after_reader.get_tensor(name)
        stats = tensor_delta_statistics(
            name,
            before,
            after,
            thresholds=thresholds,
            magnitude_edges=magnitude_edges,
            chunk_elements=chunk_elements,
            fp8_block_shape=fp8_block_shape,
            fp8_quantized_families=fp8_quantized_families,
        )
        index = _layer_index(name)
        layer = f"layer.{index}" if index is not None else "global"
        depth = _layer_depth(index, layer_count)
        stats["layer"] = layer
        stats["depth"] = depth
        tensors.append(stats)
        _add_tensor(summary, stats)
        family = stats["family"]
        add_to(families, family, stats)
        add_to(depths, depth, stats)
        add_to(layers, layer, stats)
        if family not in family_depths:
            family_depths[family] = {}
        add_to(family_depths[family], depth, stats)
        del before, after

    finished_layers = {
        layer: _finish_aggregate(aggregate)
        for layer, aggregate in sorted(layers.items())
    }
    return {
        "schema_version": "mlite.rl_checkpoint_delta.v2",
        "checkpoint_format": before_reader.format,
        "before": str(before_root),
        "after": str(after_root),
        "thresholds": list(thresholds),
        "magnitude_edges": [_number_key(value) for value in magnitude_edges],
        "block_fp8_contract": {
            "block_shape": list(fp8_block_shape),
            "weight_dtype": "float8_e4m3fn",
            "scale_dtype": "float32",
            "quantized_families": sorted(set(fp8_quantized_families)),
        }
        if fp8_block_shape is not None
        else None,
        "summary": _finish_aggregate(summary),
        "families": {
            family: _finish_aggregate(aggregate)
            for family, aggregate in sorted(families.items())
        },
        "depths": {
            depth: _finish_aggregate(aggregate)
            for depth, aggregate in sorted(depths.items())
        },
        "layers": finished_layers,
        "family_depths": {
            family: {
                depth: _finish_aggregate(aggregate)
                for depth, aggregate in sorted(by_depth.items())
            }
            for family, by_depth in sorted(family_depths.items())
        },
        "layer_concentration": _layer_concentration(finished_layers),
        "tensors": tensors,
    }


def _parse_csv_floats(
    value: str, *, append_infinity: bool = False
) -> tuple[float, ...]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if append_infinity and (not values or not math.isinf(values[-1])):
        values.append(math.inf)
    return tuple(values)


def _parse_metadata(items: Sequence[str]) -> dict[str, str]:
    metadata = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"metadata must use KEY=VALUE: {item!r}")
        key, value = item.split("=", 1)
        metadata[key] = value
    return metadata


def _parse_block_shape(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = tuple(int(item) for item in value.split(","))
    if len(parts) != 2 or any(item <= 0 for item in parts):
        raise ValueError(
            "FP8 block shape must be two positive integers, for example 128,128"
        )
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    parser.add_argument(
        "--magnitude-edges", default=",".join(map(str, DEFAULT_MAGNITUDE_EDGES[:-1]))
    )
    parser.add_argument("--chunk-elements", type=int, default=8 * 1024 * 1024)
    parser.add_argument(
        "--fp8-block-shape",
        help="measure E4M3 weights plus FP32 scales using ROWS,COLUMNS blocks",
    )
    parser.add_argument(
        "--fp8-quantized-families",
        default=",".join(DEFAULT_FP8_FAMILIES),
        help="analysis-layer families serialized as block-FP8; others pass through",
    )
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    report = analyze_checkpoints(
        args.before,
        args.after,
        thresholds=_parse_csv_floats(args.thresholds),
        magnitude_edges=_parse_csv_floats(args.magnitude_edges, append_infinity=True),
        chunk_elements=args.chunk_elements,
        fp8_block_shape=_parse_block_shape(args.fp8_block_shape),
        fp8_quantized_families=tuple(
            item for item in args.fp8_quantized_families.split(",") if item
        ),
    )
    report["metadata"] = _parse_metadata(args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tensor_count": report["summary"]["tensor_count"],
                "numel": report["summary"]["numel"],
                "exact_changed_fraction": report["summary"]["changed_fractions"]["0"],
                "block_fp8_changed_value_fraction": report["summary"]["target_formats"]
                .get("block_fp8", {})
                .get("changed_value_fraction"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
