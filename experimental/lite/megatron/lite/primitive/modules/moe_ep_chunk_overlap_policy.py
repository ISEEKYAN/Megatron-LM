# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Policy helpers for token-wise expert-parallel chunk overlap."""

from __future__ import annotations

import math
import os

ChunkSpec = int


def parse_ep_chunk_spec(value: int) -> ChunkSpec:
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            value = int(normalized)
        except ValueError as exc:
            raise ValueError("chunk count must be 1 or 2") from exc
    if int(value) not in (1, 2):
        raise ValueError("chunk count must be 1 or 2")
    return int(value)


def resolve_ep_chunk_overlap_chunks(
    num_tokens: int, *, ep_size: int, hidden_size: int, spec: ChunkSpec = 1
) -> int:
    del num_tokens, ep_size, hidden_size
    return parse_ep_chunk_spec(spec)


def validate_ep_chunk_overlap_config(
    num_chunks: int, *, use_deepep: bool, ep_size: int
) -> int:
    num_chunks = parse_ep_chunk_spec(num_chunks)
    if num_chunks == 2 and (not use_deepep or ep_size <= 1):
        raise ValueError("ChunkedEP requires DeepEP and EP > 1")
    return num_chunks


def recompute_modules_for_ep_chunk_overlap(
    modules: list[str], *, num_chunks: int
) -> list[str]:
    """Avoid wrapping the native ChunkedEP full recompute in another checkpoint."""
    chunks = parse_ep_chunk_spec(num_chunks)
    if chunks == 1:
        return modules
    return [name for name in modules if name != "moe"]


def ep_chunk_ranges(
    num_tokens: int,
    num_chunks: int,
    *,
    weights_env: str | tuple[str, ...] | None = "MEGATRON_LITE_EP_CHUNK_WEIGHTS",
) -> list[tuple[int, int]]:
    num_chunks = min(max(int(num_chunks), 1), max(num_tokens, 1))
    if num_tokens == 0:
        return [(0, 0)]

    env_names: tuple[str, ...] = ()
    if isinstance(weights_env, str):
        env_names = (weights_env,)
    elif weights_env is not None:
        env_names = weights_env
    raw = source = None
    for name in env_names:
        raw = os.environ.get(name)
        if raw:
            source = name
            break
    if raw:
        try:
            weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError(
                f"{source} must be comma-separated positive numbers"
            ) from exc
        if len(weights) != num_chunks:
            raise ValueError(f"{source} must provide {num_chunks} weights")
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
            raise ValueError(f"{source} must contain finite positive weights")

        remainder = num_tokens - num_chunks
        exact = [remainder * weight / sum(weights) for weight in weights]
        extra = [int(value) for value in exact]
        for idx in sorted(
            range(num_chunks), key=lambda item: (extra[item] - exact[item], item)
        )[: remainder - sum(extra)]:
            extra[idx] += 1

        ranges = []
        start = 0
        for more in extra:
            end = start + 1 + more
            ranges.append((start, end))
            start = end
        return ranges

    base = num_tokens // num_chunks
    remainder = num_tokens % num_chunks
    ranges = []
    start = 0
    for idx in range(num_chunks):
        end = start + base + (1 if idx < remainder else 0)
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges or [(0, num_tokens)]


__all__ = [
    "ChunkSpec",
    "ep_chunk_ranges",
    "parse_ep_chunk_spec",
    "recompute_modules_for_ep_chunk_overlap",
    "resolve_ep_chunk_overlap_chunks",
    "validate_ep_chunk_overlap_config",
]
