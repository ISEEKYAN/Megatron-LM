# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import attention as attn, block as hc, candidates, packing


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

