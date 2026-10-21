# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU layout/codec raw-bit controls; native GPU parity is a separate gate."""
import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm import batching, lora
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4


def bits(x):
    return x.contiguous().view(torch.uint8)


@pytest.mark.parametrize('counts', [(0, 0), (3, 0, 2, 0), (0, 2, 1), (129, 0, 3)])
def test_vectorized_layout_preserves_old_positions_padding_and_expert_order(counts):
    positions, ids, ends, capacity = batching.aligned_rows(counts, 128, torch.device('cpu'))
    expected, expected_ids, expected_ends, offset = [], [], [], 0
    for expert, count in enumerate(counts):
        expected += list(range(offset, offset + count))
        expected_ids += [expert] * count
        expected_ends.append(offset + count)
        offset += (count + 127) // 128 * 128
    assert positions.tolist() == expected and ids.tolist() == expected_ids
    assert ends.tolist() == expected_ends and capacity == max(offset, 128)


@pytest.mark.parametrize('count', [0, 3])
def test_grouped_adjoints_empty_experts_match_autograd(count):
    torch.manual_seed(19)
    counts = (count, 0, 2)
    x = torch.randn(sum(counts), 8, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(3, 6, 8, dtype=torch.bfloat16, requires_grad=True)
    out = torch.cat([a.float() @ b.float().T for a, b in zip(x.split(counts), w)])
    dy = torch.randn_like(out).bfloat16()
    dx, dw = torch.autograd.grad(out, (x, w), dy.float())
    assert torch.equal(batching.grouped_mm(dy, w, counts).bfloat16(), dx)
    assert torch.equal(batching.grouped_mm(dy, x, counts, wgrad=True).bfloat16(), dw)


def test_one_native_te_grouped_call_for_all_experts_and_empty_groups(monkeypatch):
    extension = ModuleType('transformer_engine.pytorch.cpp_extensions')
    calls = []

    def native(a, b, outputs, biases, dtype, *, layout, m_splits, single_output, grad):
        assert grad and dtype == torch.float32 and m_splits == (2, 0, 1)
        calls.append(layout)
        values = [
            y.float() @ x.float() if layout == 'NN' else y.float().T @ x.float()
            for x, y in zip(a, b)
        ]
        if single_output:
            outputs[0].copy_(torch.cat(values))
        else:
            for result, value in zip(outputs, values):
                result.copy_(value)

    extension.general_grouped_gemm = native
    monkeypatch.setitem(sys.modules, extension.__name__, extension)
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    x = torch.randn(3, 8).bfloat16()
    dy = torch.randn(3, 6).bfloat16()
    w = torch.randn(3, 6, 8).bfloat16()
    dx = batching.grouped_mm(dy, w, (2, 0, 1))
    dw = batching.grouped_mm(dy, x, (2, 0, 1), wgrad=True)
    assert calls == ['NN', 'NT']
    assert dx.shape == x.shape and dw.shape == w.shape and dw[1].count_nonzero() == 0


def test_dense_adjoint_calls_public_te_gemm(monkeypatch):
    module = ModuleType('transformer_engine.pytorch.cpp_extensions')
    calls = []

    def gemm(a, b, dtype, *, layout, grad):
        assert dtype == torch.float32 and layout == 'NN' and grad
        calls.append(True)
        return (b.float() @ a.float(),)

    module.general_gemm = gemm
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    x, w = torch.randn(3, 8).bfloat16(), torch.randn(8, 4).bfloat16()
    assert torch.equal(batching.dense_mm(x, w), x.float() @ w.float())
    assert calls == [True]


def test_real_te_public_api_accepts_dense_and_grouped_contract():
    te = pytest.importorskip('transformer_engine.pytorch.cpp_extensions')
    dense = inspect.signature(te.general_gemm)
    dense.bind(None, None, torch.float32, layout='NN', grad=True)
    with pytest.raises(TypeError):
        dense.bind(None, None, workspace=None)
    grouped = inspect.signature(te.general_grouped_gemm)
    for layout, single in [('NN', True), ('NT', False)]:
        grouped.bind(
            [],
            [],
            [],
            [],
            torch.float32,
            layout=layout,
            m_splits=(2, 0, 1),
            single_output=single,
            grad=True,
        )


def test_native_metadata_reused_and_groups_never_cross_contaminate(monkeypatch):
    module = ModuleType('vllm.lora.ops.triton_ops.lora_kernel_metadata')
    calls = []

    class Meta:
        @staticmethod
        def make(groups, rows, device):
            calls.append((groups, rows, device))
            return Meta()

        def prepare_tensors(self, mapping):
            self.mapping = mapping.clone()

        def meta_args(self, rows, specialize_active_lora):
            assert specialize_active_lora
            return (self.mapping, rows)

    module.LoRAKernelMeta = Meta
    monkeypatch.setitem(sys.modules, module.__name__, module)
    batching._metadata.cache_clear()
    try:
        x = torch.zeros(24, 8)
        a = batching.metadata(x, 8)
        assert batching.metadata(x, 8) is a
        assert a[0].tolist() == [g for g in range(8) for _ in range(3)]
        b = batching.metadata(x)
        assert b[0].count_nonzero() == 0 and len(calls) == 2
        assert torch.equal(a[0], torch.arange(8).repeat_interleave(3))
    finally:
        batching._metadata.cache_clear()


def test_batched_codec_bits_cache_hit_version_and_trainability(monkeypatch):
    from megatron.lite.primitive.quantization import mxfp4

    torch.manual_seed(41)
    owners = [torch.randn(4, 64).bfloat16() for _ in range(3)]
    owners[0][0, 0] = -0.0
    reference = [quantize_mxfp4(w) for w in owners]
    calls = []
    original = mxfp4.quantize_mxfp4

    def codec(w):
        calls.append(w.shape)
        return original(w)

    monkeypatch.setattr(mxfp4, 'quantize_mxfp4', codec)
    batching._PACKED.clear()
    first = batching.packed_weights(owners)
    assert calls == [(3, 4, 64)]
    for part in (0, 1):
        assert torch.equal(bits(first[part]), bits(torch.stack([v[part] for v in reference])))
    assert batching.packed_weights(owners) is first
    owners[0].add_(0.25)
    changed = batching.packed_weights(owners)
    assert len(calls) == 2 and not torch.equal(bits(first[0]), bits(changed[0]))
    owners[0].requires_grad_(True)
    batching.packed_weights(owners)
    batching.packed_weights(owners)
    assert len(calls) == 4  # Never persist a trainable Core model view.
    batching._PACKED.clear()


@pytest.mark.parametrize('groups', [1, 8])
@pytest.mark.parametrize('rows', [0, 3])
def test_batched_woa_nonzero_forward_bits_match_unbatched_native_contract(
    monkeypatch, groups, rows
):
    module = ModuleType('vllm.lora.ops.triton_ops')
    calls = []

    def shrink(x, weights, out, g, scale):
        a = weights[0][:, 0]
        assert a.shape[0] == g and scale == 1.0
        values = torch.bmm(x.reshape(g, rows, -1).float(), a.float().transpose(1, 2))
        out.copy_(values.reshape_as(out))
        calls.append('shrink')

    def expand(hidden, weights, out, *, add_inputs):
        assert add_inputs
        values = torch.stack([b[0, 0] for b in weights])
        delta = torch.bmm(hidden.bfloat16().float(), values.float().transpose(1, 2)).bfloat16()
        out.add_(delta.transpose(0, 1).reshape_as(out))
        calls.append('expand')

    module.lora_shrink, module.lora_expand = shrink, expand
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(lora, 'metadata', lambda x, groups: (groups,))
    monkeypatch.setattr(lora, '_metadata', lambda x: ())
    torch.manual_seed(80)
    x = torch.randn(rows, groups, 32, dtype=torch.bfloat16, requires_grad=True)
    base = torch.randn(rows, groups, 16, dtype=torch.bfloat16, requires_grad=True)
    a = torch.randn(groups, 4, 32, dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(groups, 16, 4, dtype=torch.bfloat16, requires_grad=True)
    owner = SimpleNamespace(lora_a=a, lora_b=b, scale=0.5)
    actual = lora.grouped_add(x, base, owner)
    expected = base.detach().clone()
    for g in range(groups):
        hidden = (x[:, g].float() @ a[g].float().T).bfloat16()
        delta = (hidden.float() @ (b[g].detach() * 0.5).bfloat16().float().T).bfloat16()
        expected[:, g] += delta
    assert torch.equal(bits(actual), bits(expected))
    assert calls == (['shrink', 'expand'] if rows else [])
    actual.float().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (x, a, b, base))


@pytest.mark.gpus(1, min_architecture='blackwell')
@pytest.mark.skipif(not torch.cuda.is_available(), reason='native parity requires CUDA')
def test_native_group_shrink_matches_eight_single_shrinks_bitwise():
    """Deferred native gate: no CPU reference can certify Triton reduction bits."""
    torch.manual_seed(81)
    x = torch.randn(7, 8, 512, device='cuda', dtype=torch.bfloat16)
    a = torch.randn(8, 32, 512, device='cuda', dtype=torch.bfloat16)
    actual = lora._GroupShrink.apply(x, a)
    expected = torch.stack([lora.shrink(x[:, g].contiguous(), a[g]) for g in range(8)])
    assert torch.equal(bits(actual), bits(expected))


@pytest.mark.parametrize('first,local', [(0, 4), (4, 4), (8, 4)])
def test_stable_owned_slots_match_old_expert_scans(first, local):
    routes = torch.tensor([[6, 1, -1, 4], [4, 1, 6, 0], [5, -1, 7, 5]])
    positions, counts = batching.owned_slots(routes, first, local)
    old = [(routes.flatten() == e).nonzero().flatten() for e in range(first, first + local)]
    assert torch.equal(positions, torch.cat(old))
    assert counts == tuple(p.numel() for p in old)


def test_invalid_layout_and_group_metadata_fail_before_import():
    with pytest.raises(ValueError, match='counts'):
        batching.aligned_rows((1, -1), 128, 'cpu')
    with pytest.raises(ValueError, match='alignment'):
        batching.aligned_rows((1, 0), 0, 'cpu')
    with pytest.raises(ValueError, match='partitions'):
        batching.metadata(torch.empty(7, 8), 3)
