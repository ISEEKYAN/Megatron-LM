# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import attention as attn
from megatron.lite.model.deepseek_v41.lite import block as hc
from megatron.lite.model.deepseek_v41.lite import candidates


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_ced_chain_and_gradient(official, model_config, dtype):
    torch.manual_seed(1729)
    cfg = model_config.attention_config(
        linear_fp8=False, main_qat=False, index_qat=False, swa_fp8=False
    )
    attention = attn.CSA2Attention(cfg, 20).to(dtype)
    block = hc.DeepseekV41Block(
        cfg.dim, 3, attention, torch.nn.Identity(), norm_eps=cfg.eps
    ).to(dtype)
    modules = [
        block.attn_norm,
        attention.compressor.wkv,
        attention.compressor.norm,
        attention.indexer.wk,
        attention.indexer.k_norm,
    ]
    with torch.no_grad():
        for module in (modules[0], modules[2], modules[4]):
            module.weight.copy_(torch.linspace(0.3, 1.7, module.weight.numel()))
    ref = [
        SimpleNamespace(
            weight=torch.nn.Parameter(
                m.weight.detach().clone(), requires_grad=m.weight.requires_grad
            ),
            eps=cfg.eps,
        )
        for m in modules
    ]
    norm = official('model.py', 'RMSNorm', 'forward')
    collapse = official('model.py', 'Block', 'hc_pre')
    compress = official('model.py', 'Compressor', 'forward')
    h = torch.randn(1, 5, 3, cfg.dim, dtype=dtype, requires_grad=True)
    p = torch.randn(1, 5, 3, requires_grad=True)
    assert not torch.equal(h[:, :, 0], h[:, :, 1])
    x = norm(ref[0], collapse(None, h, p))
    projection = lambda value: torch.nn.functional.linear(value, ref[1].weight)
    normalization = lambda value: norm(ref[2], value)
    latent = compress(
        SimpleNamespace(compress_ratio=1, wkv=projection, norm=normalization), x, 0
    )
    key = norm(ref[4], torch.nn.functional.linear(latent, ref[3].weight))
    seen = {}
    handles = []
    for module, name in (
        (block.attn_norm, 'x'),
        (attention.compressor, 'latent'),
        (attention.indexer.k_norm, 'key'),
    ):
        handles.append(
            module.register_forward_hook(
                lambda m, args, out, name=name: seen.update({name: out})
            )
        )
    try:
        block.forward_with_state(h, p, attn.AttentionState())
    finally:
        for handle in handles:
            handle.remove()
    for name, expected in zip(('x', 'latent', 'key'), (x, latent, key)):
        torch.testing.assert_close(
            seen[name], expected, rtol=0, atol=0, msg='CED:' + name
        )
    probe = torch.randn_like(latent)
    actual_args = (h, p, *(m.weight for m in modules[:3]))
    reference_args = (h, p, *(m.weight for m in ref[:3]))
    for a, b in zip(
        torch.autograd.grad((seen['latent'] * probe).sum(), actual_args),
        torch.autograd.grad((latent * probe).sum(), reference_args),
    ):
        torch.testing.assert_close(
            a, b, rtol=0, atol=0, msg=lambda detail: 'CED gradient: ' + detail
        )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in attention.indexer.parameters()
    )


@pytest.mark.parametrize('layer', [2, 8, 14, 20])
def test_owner_reuse_reindex(model_config, layer):
    config = model_config.attention_config(
        linear_fp8=False, main_qat=False, index_qat=False, swa_fp8=False
    )
    owner, reuse = [attn.CSA2Attention(config, n).float() for n in (layer, layer + 1)]
    x = torch.randn(1, 4, config.dim, requires_grad=True)
    _, state = owner(x, attn.AttentionState())
    y, shared = reuse(x, state)
    assert shared.main_kv is state.main_kv and shared.indices is state.indices
    assert torch.count_nonzero(torch.autograd.grad(y.square().sum(), state.main_kv)[0])
    if layer == 20:
        _, refreshed = attn.CSA2Attention(config, 24).float()(x, state)
        assert refreshed.index_owner == 24 and refreshed.main_kv is state.main_kv


@pytest.mark.parametrize('visible', [0, 3, 9])
def test_candidate_visibility(visible):
    scores = torch.arange(9, 0, -1).float().reshape(1, 1, 9)
    pool = candidates.candidate_blocks(scores, visible, topk_blocks=2, block_size=2)
    scores[..., 6] = 100
    selected = candidates.select_positions(scores, visible, 2, candidates=pool)
    assert not ((selected >= visible) & (selected != -1)).any()
    if visible == 9:
        assert pool[..., -1].all() and not pool[..., 6].any() and 6 not in selected
