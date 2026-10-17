# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deferred PP state keeps native operands and both autograd branches explicit."""
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.primitive.modules import paired_stream as stream


def state(dtype=torch.float32, length=5):
    shapes = [
        (1, length, 2, 4),
        (1, length, 2),
        (1, length, 4),
        (1, length, 2, 4),
        (1, length, 2),
        (1, length, 2, 2),
    ]
    dtypes = [dtype, torch.float32, dtype, dtype, torch.float32, torch.float32]
    return [
        torch.randn(shape).to(kind).requires_grad_()
        for shape, kind in zip(shapes, dtypes)
    ]


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('length', [0, 1, 5])
def test_pending_state_roundtrip_preserves_every_field_and_dtype(dtype, length):
    hidden, pre, *pending = state(dtype, length)
    carrier = stream.pack_state(hidden, pre, tuple(pending))
    restored = stream.unpack_state(
        carrier, (1, length), 2, 4, dtype, require_pending=True
    )
    for expected, actual in zip([hidden, pre, *pending], [*restored[:2], *restored[2]]):
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)


def test_all_finite_bf16_values_roundtrip_bitwise_through_fp32():
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    actual = values.float().bfloat16()
    assert torch.equal(actual.view(torch.int16), values.view(torch.int16))


@pytest.mark.parametrize('engram_first', [False, True])
def test_unused_carrier_branch_has_zero_gradient_and_active_gradients_match(
    engram_first,
):
    leaves = state()
    hidden, pre, *pending = leaves
    carrier = stream.pack_state(hidden, pre, tuple(pending))
    h, p, operands = stream.unpack_state(
        carrier, (1, 5), 2, 4, torch.float32, require_pending=True
    )

    def loss(h, p, args):
        # Boundary dependency probe, not a native-kernel numerical substitute.
        if engram_first:
            return (h + 2).square().sum() + p.sum()
        return p.sum() + sum(
            (i + 2) * value.square().sum() for i, value in enumerate(args)
        )

    reference = [value.detach().clone().requires_grad_() for value in leaves]
    loss(h, p, operands).backward()
    loss(reference[0], reference[1], reference[2:]).backward()
    for i, (actual, expected) in enumerate(zip(leaves, reference)):
        wanted = torch.zeros_like(expected) if expected.grad is None else expected.grad
        assert torch.equal(actual.grad, wanted), i
    unused = pending if engram_first else [hidden]
    assert all(torch.count_nonzero(value.grad) == 0 for value in unused)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_legacy_pair_abi_bytes_are_unchanged(dtype):
    hidden, pre, *_ = state(dtype)
    before = stream.pack_pair(hidden, pre)
    after = stream.pack_state(hidden, pre, None)
    assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))
    old = stream.unpack_pair(before, (1, 5), 2, 4, dtype, 'legacy error')
    new = stream.unpack_state(after, (1, 5), 2, 4, dtype)
    assert new[2] is None
    assert all(torch.equal(a, b) for a, b in zip(old, new[:2]))
    with pytest.raises(ValueError, match='requires deferred post'):
        stream.unpack_state(after, (1, 5), 2, 4, dtype, require_pending=True)


@pytest.mark.parametrize('fault', ['missing', 'shape', 'dtype', 'carrier_width'])
def test_pending_contract_fails_closed(fault):
    hidden, pre, *pending = state(torch.bfloat16)
    if fault == 'missing':
        pending.pop()
    elif fault == 'shape':
        pending[0] = pending[0][:, :-1]
    elif fault == 'dtype':
        pending[2] = pending[2].bfloat16()
    if fault == 'carrier_width':
        carrier = stream.pack_state(hidden, pre, tuple(pending))[..., :-1]
        with pytest.raises(ValueError):
            stream.unpack_state(carrier, (1, 5), 2, 4, torch.bfloat16)
    else:
        with pytest.raises(ValueError):
            stream.pack_state(hidden, pre, tuple(pending))


def test_packed_slice_carries_its_pending_and_gradients(v41_core_te):
    from megatron.lite.model.deepseek_v41.lite.protocol import packed_pipeline_forward

    hidden, pre, *pending = state()
    seen = []

    def sequence(h, p, *, previous_post, return_pending, geometry, **kwargs):
        # One call over the whole slice; documents are marked by the geometry.
        assert return_pending
        seen.append((tuple(geometry.boundaries), previous_post))
        return h, p, tuple(value * 2 for value in previous_post)

    h, p, result = packed_pipeline_forward(
        sequence,
        hidden,
        pre,
        torch.tensor([0, 2, 5], dtype=torch.int32),
        previous_post=tuple(pending),
        input_ids=torch.arange(5).reshape(1, 5),
    )
    assert torch.equal(h, hidden) and torch.equal(p, pre)
    ((boundaries, received),) = seen
    assert boundaries == (0, 2, 5)
    assert all(got is wanted for got, wanted in zip(received, pending))
    assert all(torch.equal(got, wanted * 2) for got, wanted in zip(result, pending))
    sum(value.sum() for value in result).backward()
    assert all(torch.equal(value.grad, torch.full_like(value, 2)) for value in pending)


@pytest.mark.parametrize('engram_layer', [0, 1])
def test_engram_pending_reset_matches_pp1_before_and_after_cut(
    v41_core_te, monkeypatch, engram_layer
):
    from megatron.lite.model.deepseek_v41.lite import model as assembly

    class CudaFlag(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    ids = torch.tensor([[2, 3, 4]])
    hidden = torch.ones(1, 3, 2, 4).as_subclass(CudaFlag)
    pre = torch.ones(1, 3, 2)
    generated = [
        (torch.ones(1, 3, 4) * (i + 1), hidden, pre, torch.ones(1, 3, 2, 2))
        for i in range(2)
    ]
    seen = []

    class Layer:
        def __init__(self, index):
            self.index = index
            self.engram = (lambda h, *args: h + 1) if index == engram_layer else None

        def forward(self, h, p, attention_state, *, previous_post, **kwargs):
            seen.append((self.index, previous_post))
            return h, p, attention_state, generated[self.index]

    monkeypatch.setattr(
        assembly,
        'sequence_hashes',
        lambda *args: torch.zeros(1, 3, 1, 2, dtype=torch.int64),
    )
    model = SimpleNamespace(
        local_layer_range=(0, 2),
        deployment_math=True,
        routes_image_sentinels=False,
        engram_hash=object(),
        engram_layer_ids=[engram_layer],
        layers=[Layer(0), Layer(1)],
        topology=[SimpleNamespace(engram_slot=0)] * 2,
    )
    with torch.no_grad():
        assembly.DeepseekV41Model._sequence(model, hidden, pre, input_ids=ids)
        pp1_seen = seen[:]
        seen.clear()
        model.local_layer_range = (0, 1)
        h, p, outgoing = assembly.DeepseekV41Model._sequence(
            model, hidden, pre, input_ids=ids, return_pending=True
        )
        assert outgoing is generated[0]  # Last Engram still publishes a fresh FFN post.
        model.local_layer_range = (1, 2)
        assembly.DeepseekV41Model._sequence(
            model, h, p, input_ids=ids, previous_post=outgoing, return_pending=True
        )
    assert [i for i, _ in seen] == [i for i, _ in pp1_seen]
    assert all(a is b for (_, a), (_, b) in zip(seen, pp1_seen))
    assert seen[1][1] is (None if engram_layer == 1 else generated[0])


def test_duplicate_materialized_hidden_does_not_change_stage0_source_gradients():
    from megatron.lite.primitive.modules.deployment_math.mhc import reference_post

    _, pre, *operands = state()
    materialized = reference_post(*operands)
    materialized.retain_grad()
    carrier = stream.pack_state(materialized, pre, tuple(operands))
    h, p, recovered = stream.unpack_state(
        carrier, (1, 5), 2, 4, torch.float32, require_pending=True
    )
    # Match the native fused reference dependency: use pending, not materialized h.
    actual = (reference_post(*recovered) * p.unsqueeze(-1)).square().sum()
    originals = [pre, *operands]
    reference = [value.detach().clone().requires_grad_() for value in originals]
    wanted = (
        (reference_post(*reference[1:]) * reference[0].unsqueeze(-1)).square().sum()
    )
    actual.backward()
    wanted.backward()
    assert torch.count_nonzero(materialized.grad) == 0
    assert all(
        torch.equal(value.grad, expected.grad)
        for value, expected in zip(originals, reference)
    )
