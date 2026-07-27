# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Policy helpers for token-wise expert-parallel chunk overlap."""

from __future__ import annotations

import math
import os
from typing import Literal

ChunkSpec = int | Literal["auto"]
ChunkDirection = Literal["forward", "fused_backward"]


def parse_ep_chunk_spec(
    value: ChunkSpec | None, *, default: ChunkSpec = "auto"
) -> ChunkSpec:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        try:
            value = int(normalized)
        except ValueError as exc:
            raise ValueError("chunk spec must be an integer or 'auto'") from exc
    if int(value) < 1:
        raise ValueError("chunk count must be >= 1")
    return int(value)


def resolve_ep_chunk_overlap_chunks(
    num_tokens: int,
    *,
    ep_size: int,
    hidden_size: int,
    spec: ChunkSpec = "auto",
    direction: ChunkDirection = "forward",
) -> int:
    del hidden_size
    spec = parse_ep_chunk_spec(spec)
    if spec != "auto":
        return int(spec)
    if ep_size <= 1 or num_tokens < 16_384:
        return 1
    if direction == "forward":
        return 2 if num_tokens < 32_768 else 3
    if direction == "fused_backward":
        return 2
    raise ValueError("direction must be 'forward' or 'fused_backward'")


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
            range(num_chunks),
            key=lambda item: (extra[item] - exact[item], item),
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
    "ChunkDirection",
    "ChunkSpec",
    "ep_chunk_ranges",
    "parse_ep_chunk_spec",
    "resolve_ep_chunk_overlap_chunks",
]
