# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reduced single-rank V4.1 semantics and gradient checks."""

from types import SimpleNamespace

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import attention as attn
from megatron.lite.model.deepseek_v41.lite import block as hc
from megatron.lite.model.deepseek_v41.lite import candidates, engram, packing
from megatron.lite.primitive.quantization import ds41_fp8


@pytest.fixture
def moe(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v41.lite import moe

    return moe


@pytest.mark.parametrize('layer', [2, 8, 14, 20])
def test_v41_owner_reuse_reindex_gradients(layer):
    torch.manual_seed(42)
    config = attn.CSA2Config(
        dim=32,
        heads=2,
        head_dim=32,
        rope_dim=4,
        q_rank=32,
        o_rank=4,
        groups=2,
        index_heads=2,
        index_dim=32,
        topk=2,
        window=3,
        candidate_blocks=1,
        block_size=2,
        linear_fp8=False,
        main_qat=False,
        index_qat=False,
        swa_fp8=False,
    )
    owner, reuse = [attn.CSA2Attention(config, n).float() for n in (layer, layer + 1)]
    x = torch.randn(1, 4, 32, requires_grad=True)
    _, state = owner(x, attn.AttentionState())
    y, shared = reuse(x, state)
    assert shared.main_kv is state.main_kv and shared.indices is state.indices
    assert all((not p.requires_grad for p in owner.indexer.parameters()))
    assert torch.count_nonzero(torch.autograd.grad(y.square().sum(), state.main_kv)[0])
    if layer == 20:
        reindex = attn.CSA2Attention(config, 24).float()
        _, refreshed = reindex(x, state)
        assert refreshed.index_owner == 24 and refreshed.main_kv is state.main_kv
    block = hc.DeepseekV41Block(32, 2, owner, torch.nn.Linear(32, 32)).float()

    def sequence(h, p):
        h, p, _ = block.forward_with_state(h, p, attn.AttentionState())
        return (h, p)

    h, p = hc.expand_hc(x, 2)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    actual, mixes = packing.packed_forward(sequence, h, p, cu)
    expected = [sequence(h[:, n : n + 2], p[:, n : n + 2]) for n in (0, 2)]
    torch.testing.assert_close(actual, torch.cat([v[0] for v in expected], 1))
    torch.testing.assert_close(mixes, torch.cat([v[1] for v in expected], 1))
    grad = torch.autograd.grad(actual[:, 2:].square().sum(), x)[0]
    assert not grad[:, :2].any() and grad[:, 2:].any()


@pytest.mark.parametrize('trainable', [False, True])
def test_v41_engram_storage_and_reset(trainable):
    encoded = ds41_fp8.quantize_swa(torch.linspace(-2, 2, 96).reshape(3, 32))
    table = engram.EngramTable(encoded.values, encoded.scale, trainable=trainable)
    ids = torch.tensor([[0, 0, 2]])
    original = table(ids).float()
    table.bfloat16()
    assert table.weight.dtype == torch.float8_e4m3fn
    assert table.scale.dtype == torch.float8_e8m0fnu
    if trainable:
        table(ids).float().sum().backward()
        assert table.master.dtype == torch.float32
        torch.testing.assert_close(
            table.master.grad, torch.tensor([2.0, 0.0, 1.0])[:, None].expand(3, 32)
        )
        with torch.no_grad():
            table.master.mul_(16)
        table.refresh_storage()
        torch.testing.assert_close(table(ids).float(), original * 16)
    else:
        assert not list(table.parameters()) and 'master' not in table.state_dict()
    hasher = engram.NgramHash(
        [0, 1, 2, 3], 0, torch.tensor([[3, 5, 7]]), torch.tensor([[[11], [13]]])
    )
    hashed = hasher(torch.tensor([[1, 2, 3]]), cu_seqlens=torch.tensor([0, 2, 3]))
    assert hashed.tolist() == [[[[3, 14]], [[3, 14]], [[9, 20]]]]


@pytest.mark.parametrize('image', [False, True])
def test_v41_modality_bias_selection_and_vjp(image, moe):
    config = SimpleNamespace(
        hidden_size=3,
        n_routed_experts=3,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        scoring_func='sqrtsoftplus',
    )
    router = moe.ModalityRouter(
        config, SimpleNamespace(tp_size=1), gate_temperature=0.7
    )
    with torch.no_grad():
        router.router.gate.weight.copy_(torch.eye(3))
        (router.bias_vl if image else router.bias)[2] = 4
    x = torch.tensor([[2.0, 1.0, -1.0]], requires_grad=True)
    weights, indices, stats = router(x, torch.tensor([image]))
    assert indices.tolist() == [[0, 2]]
    raw = torch.nn.functional.softplus(x / 0.7).sqrt()[:, [0, 2]]
    expected = raw / raw.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(weights, expected)
    cotangent = torch.tensor([[1.0, -2.0]])
    torch.testing.assert_close(
        torch.autograd.grad((weights * cotangent).sum(), x)[0],
        torch.autograd.grad((expected * cotangent).sum(), x)[0],
    )
    before = torch.stack([router.bias.clone(), router.bias_vl.clone()])
    router.update_bias(stats)
    delta = torch.zeros(2, 3)
    delta[int(image)] = torch.tensor([-0.001, 0.001, -0.001])
    torch.testing.assert_close(
        torch.stack([router.bias, router.bias_vl]), before + delta
    )
    assert not router.router.compute_aux_loss


@pytest.mark.parametrize('visible', [0, 3, 9])
def test_v41_candidate_pool_visibility(visible):
    scores = torch.arange(9, 0, -1).float().reshape(1, 1, 9)
    pool = candidates.candidate_blocks(scores, visible, topk_blocks=2, block_size=2)
    later = scores.clone()
    later[..., 6] = 100
    selected = candidates.select_positions(later, visible, 2, candidates=pool)
    assert not ((selected >= visible) & (selected != -1)).any()
    if visible == 9:
        assert pool[..., -1].all() and (not pool[..., 6].any())
        assert 6 not in selected
    if visible == 0:
        assert (selected == -1).all()


@pytest.mark.parametrize('copies', [2, 4])
def test_v41_mhc_source_destination_orientation(copies):
    torch.manual_seed(9)
    residual = torch.randn(1, 2, copies, 3, requires_grad=True)
    output = torch.randn(1, 2, 3, requires_grad=True)
    post = torch.randn(1, 2, copies, requires_grad=True)
    comb = torch.randn(1, 2, copies, copies, requires_grad=True)
    expected = torch.stack(
        [
            post[..., j, None] * output
            + sum((comb[..., i, j, None] * residual[..., i, :] for i in range(copies)))
            for j in range(copies)
        ],
        -2,
    )
    actual = hc.mix_residual(output, residual, post, comb)
    torch.testing.assert_close(actual, expected)
    args = (output, residual, post, comb)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), args),
        torch.autograd.grad(expected.square().sum(), args),
    ):
        torch.testing.assert_close(a, b)
