# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Derive decoder ownership and layer policies from the release topology."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TopologySpec:
    num_hidden_layers: int = 40
    num_nextn_predict_layers: int = 3
    compress_ratios: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20 + (0,) * 3
    kv_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    candidate_source_layer_id: int = 20
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_num_embeddings: tuple[int, ...] = (384006168, 384016682)


@dataclass(frozen=True)
class LayerPolicy:
    index: int
    compress_ratio: int
    kv_owner: int | None
    index_owner: int | None
    candidate_mode: Literal["none", "build", "reuse"]
    engram_rows: int
    engram_slot: int | None
    is_ced_boundary: bool


def build_topology(spec: TopologySpec) -> tuple[LayerPolicy, ...]:
    """Validate sources once, then assign the last published owner to each layer.

    MTP entries remain archival; no executable policy is generated for them.
    Smaller valid graphs are accepted for numerical fixtures. Shape/config
    validation is separate from this ownership contract.
    """
    n = spec.num_hidden_layers
    if type(n) is not int or n <= 0:
        raise ValueError("num_hidden_layers must be a positive integer")
    mtp = spec.num_nextn_predict_layers
    if type(mtp) is not int or mtp < 0:
        raise ValueError("num_nextn_predict_layers must be a nonnegative integer")
    if len(spec.compress_ratios) != n + mtp:
        raise ValueError("compress_ratios must describe backbone and archival layers")
    if any(type(r) is not int or r not in (0, 1, 2) for r in spec.compress_ratios):
        raise ValueError("compression ratios must be 0, 1, or 2")
    if any(spec.compress_ratios[n:]):
        raise ValueError("archival layers must have zero compression ratios")
    for name in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids"):
        ids = getattr(spec, name)
        if any(type(i) is not int for i in ids):
            raise ValueError(f"{name} must contain integer layer ids")
        if any(i < 0 or i >= n for i in ids):
            raise ValueError(f"{name} has a layer id outside the backbone range")
        if any(a >= b for a, b in zip(ids, ids[1:])):
            raise ValueError(f"{name} must be strictly increasing")
    if len(spec.engram_layer_ids) != len(spec.engram_num_embeddings):
        raise ValueError("engram requires one row count per layer")
    if any(type(r) is not int or r <= 0 for r in spec.engram_num_embeddings):
        raise ValueError("engram row counts must be positive integers")
    for source in (*spec.kv_source_layer_ids, *spec.index_source_layer_ids):
        if not spec.compress_ratios[source]:
            raise ValueError("attention source cannot be uncompressed")
    candidate = spec.candidate_source_layer_id
    if (
        type(candidate) is not int
        or candidate not in spec.kv_source_layer_ids
        or candidate not in spec.index_source_layer_ids
        or spec.compress_ratios[candidate] != 1
    ):
        raise ValueError("candidate source must own KV and index with ratio 1")
    rows = dict(zip(spec.engram_layer_ids, spec.engram_num_embeddings))
    slots = {layer: slot for slot, layer in enumerate(spec.engram_layer_ids)}
    kv_owner = index_owner = None
    policies = []
    for i, ratio in enumerate(spec.compress_ratios[:n]):
        if i in spec.kv_source_layer_ids:
            kv_owner = i
        if i in spec.index_source_layer_ids:
            index_owner = i
        if ratio:
            if kv_owner is None:
                raise ValueError(f"layer {i} has no KV owner")
            if index_owner is None:
                raise ValueError(f"layer {i} has no index owner")
            if any(spec.compress_ratios[o] != ratio for o in (kv_owner, index_owner)):
                raise ValueError(f"layer {i} disagrees with its owner ratio")
        mode = "none" if i < candidate else "build" if i == candidate else "reuse"
        policies.append(
            LayerPolicy(
                i,
                ratio,
                kv_owner if ratio else None,
                index_owner if ratio else None,
                mode,
                rows.get(i, 0),
                slots.get(i),
                i + 1 == candidate,
            )
        )
    return tuple(policies)


V41_TOPOLOGY = build_topology(TopologySpec())
