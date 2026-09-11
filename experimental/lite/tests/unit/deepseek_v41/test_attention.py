import torch
from megatron.lite.model.deepseek_v41.lite import attention as attn


def config():
    return attn.CSA2Config(
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


def test_ratio_one_rope_and_inverse():
    x = torch.arange(128).reshape(1, 2, 2, 32).float()
    positions = torch.tensor([127, 65536])
    c = config()
    y = attn.rotate(x, positions, c, 1)
    assert not torch.allclose(y, attn.rotate(x, positions, c, 0))
    torch.testing.assert_close(
        attn.rotate(y, positions, c, 1, inverse=True), x, atol=2e-5, rtol=1e-6
    )


def test_owner_reuse_reindex_and_shared_gradient_sum():
    torch.manual_seed(42)
    c = config()
    owner, reuse, reindex = [
        attn.CSA2Attention(c, layer).float() for layer in (20, 21, 24)
    ]
    x = torch.randn(1, 4, 32, requires_grad=True)
    y, state = owner(x, attn.AttentionState())
    assert state.kv_owner == 20 and state.index_owner == 20
    assert state.latent.shape == (1, 4, 32)
    a, reused = reuse(x * 0.3, state)
    b, reindexed = reindex(x * -0.7, state)
    assert reused.main_kv is state.main_kv
    assert reused.indices is state.indices
    assert reindexed.index_k is state.index_k and reindexed.index_owner == 24
    assert reuse.indexer is None and reindex.compressor is None
    ga = torch.autograd.grad(a.square().sum(), state.main_kv, retain_graph=True)[0]
    gb = torch.autograd.grad(b.square().sum(), state.main_kv, retain_graph=True)[0]
    total = torch.autograd.grad(a.square().sum() + b.square().sum(), state.main_kv)[0]
    torch.testing.assert_close(total, ga + gb)
    assert torch.count_nonzero(ga) and torch.count_nonzero(gb)


def test_empty_compressed_prefix_and_ced_pair():
    from megatron.lite.model.deepseek_v41.lite import block

    torch.manual_seed(17)
    c = config()
    layer = attn.CSA2Attention(c, 2).float()
    output, state = layer(torch.randn(1, 1, 32), attn.AttentionState())
    assert torch.isfinite(output).all() and state.main_kv.shape[1] == 0
    assert state.indices.shape[-1] == 0
    layer20 = attn.CSA2Attention(c, 20).float()
    h = torch.randn(1, 4, 2, 32, requires_grad=True)
    p = torch.tensor([0.2, 0.8]).expand(1, 4, 2).clone().requires_grad_()
    norm = block.RMSNorm(32, c.eps)
    x20 = norm(block.contract_hc(h, p))
    _, state = layer20(x20, attn.AttentionState())
    expected = layer20.compressor.norm(layer20.compressor.wkv(x20))
    torch.testing.assert_close(state.latent, expected)
    grad_h, grad_p = torch.autograd.grad(state.latent[..., 0].sum(), [h, p])
    assert torch.count_nonzero(grad_h) and torch.count_nonzero(grad_p)


def test_quantizers_consume_rotated_vectors_and_independent_switches():
    from dataclasses import replace

    from megatron.lite.primitive.quantization.ds41_index import quantize_index
    from megatron.lite.primitive.quantization.ds41_kv import quantize_main_kv

    torch.manual_seed(93)
    base = config()
    c = replace(base, main_qat=True, index_qat=True, swa_fp8=True)
    layer = attn.CSA2Attention(c, 20).float()
    x = torch.randn(1, 4, 32)
    seen = []
    handle = layer.indexer.wk.register_forward_pre_hook(
        lambda module, args: seen.append(args[0])
    )
    output, state = layer(x, attn.AttentionState())
    handle.remove()
    assert torch.isfinite(output).all()
    assert seen[0] is state.latent
    positions = torch.arange(4)
    expected = quantize_main_kv(attn.rotate(state.latent, positions, c, 1)).decoded
    torch.testing.assert_close(state.main_kv, expected, rtol=0, atol=0)
    index_pre_rope = layer.indexer.k_norm(layer.indexer.wk(state.latent))
    expected_index = quantize_index(
        attn.rotate(index_pre_rope, positions, c, 1)
    ).decoded
    torch.testing.assert_close(state.index_k, expected_index, rtol=0, atol=0)
    layer.config = replace(c, main_qat=False)
    _, other = layer(x, attn.AttentionState())
    torch.testing.assert_close(other.index_k, state.index_k, rtol=0, atol=0)
    assert not torch.equal(other.main_kv, state.main_kv)
