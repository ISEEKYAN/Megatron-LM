#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Measure element-wise deltas between two exported safetensor checkpoints.

The analyzer is deliberately CPU-only and reads tensors one at a time.  It
measures the values that an inference backend would receive, rather than
optimizer master weights, so exact nonzero counts describe the lossless
replacement payload needed by weight synchronization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
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


def _number_key(value: float) -> str:
    return format(value, ".12g")


def classify_parameter_family(name: str) -> str:
    """Classify an exported parameter name for reporting, not dispatch."""

    lowered = name.lower()
    if any(token in lowered for token in ("embed_tokens", "word_embeddings", "wte")):
        return "embedding"
    if lowered.endswith("lm_head.weight") or "output_layer" in lowered:
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
    if "norm" in lowered:
        return "norm"
    if any(token in lowered for token in (".mlp.", "feed_forward", ".ffn.")):
        return "dense_mlp"
    return "other"


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
    }


def _checkpoint_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    result: dict[str, Path] = {}
    files = sorted(root.rglob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensor shards found below {root}")
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise ValueError(
                        f"duplicate tensor {key!r} in {result[key]} and {path}"
                    )
                result[key] = path
    return result


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


def _finish_aggregate(aggregate: dict[str, object]) -> dict[str, object]:
    numel = aggregate.pop("numel")
    delta_square_sum = aggregate.pop("delta_square_sum")
    before_square_sum = aggregate.pop("before_square_sum")
    changed_value_bytes = aggregate.pop("changed_value_bytes")
    changed_counts = aggregate["changed_counts"]
    l2 = math.sqrt(delta_square_sum)
    before_l2 = math.sqrt(before_square_sum)
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


def analyze_checkpoints(
    before_root: str | Path,
    after_root: str | Path,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    magnitude_edges: Sequence[float] = DEFAULT_MAGNITUDE_EDGES,
    chunk_elements: int = 8 * 1024 * 1024,
) -> dict[str, object]:
    """Compare matching exported tensors below two checkpoint directories."""

    before_root = Path(before_root).resolve()
    after_root = Path(after_root).resolve()
    if before_root == after_root:
        raise ValueError("before and after checkpoints must be different directories")
    thresholds = _validate_bins(thresholds, name="thresholds")
    magnitude_edges = _validate_bins(magnitude_edges, name="magnitude_edges")
    before_index = _checkpoint_index(before_root)
    after_index = _checkpoint_index(after_root)
    missing_after = sorted(before_index.keys() - after_index.keys())
    missing_before = sorted(after_index.keys() - before_index.keys())
    if missing_after or missing_before:
        raise ValueError(
            f"checkpoint tensor sets differ: missing_after={missing_after[:8]}, "
            f"missing_before={missing_before[:8]}"
        )

    threshold_keys = [_number_key(value) for value in thresholds]
    summary = _empty_aggregate(threshold_keys, len(magnitude_edges) - 1)
    families: dict[str, dict[str, object]] = {}
    tensors = []
    for name in sorted(before_index):
        with safe_open(
            before_index[name], framework="pt", device="cpu"
        ) as before_handle:
            before = before_handle.get_tensor(name)
        with safe_open(after_index[name], framework="pt", device="cpu") as after_handle:
            after = after_handle.get_tensor(name)
        stats = tensor_delta_statistics(
            name,
            before,
            after,
            thresholds=thresholds,
            magnitude_edges=magnitude_edges,
            chunk_elements=chunk_elements,
        )
        tensors.append(stats)
        _add_tensor(summary, stats)
        family = stats["family"]
        if family not in families:
            families[family] = _empty_aggregate(
                threshold_keys, len(magnitude_edges) - 1
            )
        _add_tensor(families[family], stats)
        del before, after

    return {
        "schema_version": "mlite.rl_checkpoint_delta.v1",
        "before": str(before_root),
        "after": str(after_root),
        "thresholds": list(thresholds),
        "magnitude_edges": [_number_key(value) for value in magnitude_edges],
        "summary": _finish_aggregate(summary),
        "families": {
            family: _finish_aggregate(aggregate)
            for family, aggregate in sorted(families.items())
        },
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
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    report = analyze_checkpoints(
        args.before,
        args.after,
        thresholds=_parse_csv_floats(args.thresholds),
        magnitude_edges=_parse_csv_floats(args.magnitude_edges, append_infinity=True),
        chunk_elements=args.chunk_elements,
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
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
