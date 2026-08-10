from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.ckpt.fused_weights import (
    FusedWeightLayout,
    QuantizedWeight,
    WeightSegment,
)
from megatron.lite.primitive.ckpt.hf_weights import split_gate_up, split_qkv


def _kda_layout(*, q_heads: int = 6) -> FusedWeightLayout:
    return FusedWeightLayout(
        name="in_proj_qkvgfab",
        segments=(
            WeightSegment("q", q_heads, 4),
            WeightSegment("k", 6, 4),
            WeightSegment("v", 6, 4),
            WeightSegment("g", 6, 4),
            WeightSegment("f_a", 1, 4, replicated=True),
            WeightSegment("b", 6, 1),
        ),
    )


def _parts(layout: FusedWeightLayout) -> dict[str, torch.Tensor]:
    offset = 0
    parts = {}
    for segment in layout.segments:
        numel = segment.rows * 3
        parts[segment.name] = torch.arange(offset, offset + numel).reshape(
            segment.rows, 3
        )
        offset += numel
    return parts


def test_layout_fuses_and_splits_against_an_independent_reference() -> None:
    layout = _kda_layout()
    parts = _parts(layout)
    reference = torch.cat(
        [parts[name] for name in ("q", "k", "v", "g", "f_a", "b")], dim=0
    )

    fused = layout.fuse(parts)
    restored = layout.split(fused)

    assert torch.equal(fused, reference)
    assert tuple(restored) == ("q", "k", "v", "g", "f_a", "b")
    assert all(torch.equal(restored[name], parts[name]) for name in restored)


def test_layout_rejects_a_declared_head_count_mismatch() -> None:
    layout = _kda_layout(q_heads=7)
    parts = _parts(_kda_layout())

    with pytest.raises(ValueError, match=r"q.*28 rows.*got 24"):
        layout.fuse(parts)


def test_layout_rejects_components_in_the_wrong_declared_order() -> None:
    layout = _kda_layout()
    parts = _parts(layout)
    wrong_order = [("q", parts["q"]), ("v", parts["v"]), ("k", parts["k"])] + [
        (name, parts[name]) for name in ("g", "f_a", "b")
    ]

    with pytest.raises(ValueError, match=r"segment order mismatch.*k.*v"):
        layout.fuse_ordered(wrong_order)


def test_tp_shard_matches_independent_mixed_shard_and_replica_semantics() -> None:
    layout = _kda_layout()
    parts = _parts(layout)

    actual = layout.tp_shard(layout.fuse(parts), rank=1, world_size=2)
    expected = torch.cat(
        [parts[name].chunk(2, dim=0)[1] for name in ("q", "k", "v", "g")]
        + [parts["f_a"], parts["b"].chunk(2, dim=0)[1]],
        dim=0,
    )

    assert torch.equal(actual, expected)


def test_quantized_fusion_keeps_each_scale_attached_to_its_segment() -> None:
    layout = FusedWeightLayout(
        name="gate_up",
        segments=(WeightSegment("gate", 2, 1), WeightSegment("up", 2, 1)),
    )
    gate = QuantizedWeight(
        packed=torch.full((2, 2), 1, dtype=torch.uint8),
        scale=torch.full((2, 1), 11, dtype=torch.uint8),
    )
    up = QuantizedWeight(
        packed=torch.full((2, 2), 2, dtype=torch.uint8),
        scale=torch.full((2, 1), 22, dtype=torch.uint8),
    )

    fused = layout.fuse_quantized(
        {"gate": gate, "up": up}, materialize=lambda pair: pair.scale.expand(-1, 2)
    )

    assert torch.equal(
        fused,
        torch.tensor([[11, 11], [11, 11], [22, 22], [22, 22]], dtype=torch.uint8),
    )


def test_quantized_split_checks_each_scale_and_round_trips_exactly() -> None:
    layout = _kda_layout()
    parts = _parts(layout)
    scales = {
        segment.name: torch.full((segment.rows, 1), index, dtype=torch.uint8)
        for index, segment in enumerate(layout.segments, start=1)
    }
    packed = layout.fuse(parts)
    fused_scales = layout.fuse(scales)

    restored = layout.split_quantized(packed, fused_scales)

    for segment in layout.segments:
        assert torch.equal(restored[segment.name].packed, parts[segment.name])
        assert torch.equal(restored[segment.name].scale, scales[segment.name])


def test_shared_hf_qkv_and_gate_up_sharding_use_declared_segment_boundaries() -> None:
    q = torch.arange(0, 16).reshape(8, 2)
    k = torch.arange(100, 108).reshape(4, 2)
    v = torch.arange(200, 208).reshape(4, 2)
    qkv = torch.cat((q, k, v), dim=0)
    gate = torch.arange(300, 316).reshape(8, 2)
    up = torch.arange(400, 416).reshape(8, 2)

    assert torch.equal(
        split_qkv(qkv, rank=1, world=2, num_q_heads=4, num_kv_heads=2, head_dim=2),
        torch.cat((q.chunk(2)[1], k.chunk(2)[1], v.chunk(2)[1]), dim=0),
    )
    assert torch.equal(
        split_gate_up(torch.cat((gate, up)), rank=1, world=2),
        torch.cat((gate.chunk(2)[1], up.chunk(2)[1]), dim=0),
    )
