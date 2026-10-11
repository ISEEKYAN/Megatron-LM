# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Visible CUDA math is explicit; its reference VJP owns FP32 master gradients."""
import importlib

import pytest
import torch


def test_visible_vjp_does_not_truncate_master_gradient():
    from megatron.lite.primitive.modules.deployment_math import visible_forward

    torch.manual_seed(603)
    x = torch.randn(7, 32).bfloat16().requires_grad_()
    weight = torch.nn.Parameter(torch.randn(16, 32))
    incoming = torch.randn(7, 16).bfloat16()
    output = visible_forward(
        lambda a, b: torch.nn.functional.linear(a, b.bfloat16()),
        lambda a, b: torch.nn.functional.linear(a.float(), b),
        x,
        weight,
    )
    output.backward(incoming)
    expected = incoming.float().T @ x.detach().float()
    assert torch.equal(weight.grad, expected)
    assert weight.grad.dtype == torch.float32
    assert not torch.equal(weight.grad, weight.grad.bfloat16().float())
    assert x.grad.dtype == torch.bfloat16


def test_visible_multi_output_unused_gradient():
    from megatron.lite.primitive.modules.deployment_math import visible_forward

    x = torch.tensor([2.0, 3.0], requires_grad=True)
    first, _ = visible_forward(
        lambda a: (a.square(), a * 3), lambda a: (a.square(), a * 3), x
    )
    first.sum().backward()
    assert torch.equal(x.grad, torch.tensor([4.0, 6.0]))


def test_deployment_math_rejects_fp32_residuals(v41_core_te):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from test_w4a8_fp32 import tiny_config

    with pytest.raises(ValueError, match='Deployment math requires'):
        protocol.build_model(
            tiny_config(),
            impl_cfg=protocol.ImplConfig(
                device='cpu', quantized=False, deployment_math=True, dtype=torch.float32
            ),
        )


def test_norm_decodes_updated_master_and_keeps_fp32_gradient():
    from megatron.lite.primitive.modules.attention.mhc import RMSNorm

    torch.manual_seed(611)
    norm = RMSNorm(32, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(1 + torch.randn(32) * 0.01)
    x = torch.randn(3, 32).bfloat16()
    incoming = torch.randn_like(x)
    norm.deployment_math = True
    y = norm(x)
    decoded = norm.weight.detach().bfloat16().float().requires_grad_()
    expected = torch.nn.functional.rms_norm(x, (32,), decoded, norm.eps)
    assert torch.equal(y, expected)
    y.backward(incoming)
    expected.backward(incoming)
    assert torch.equal(norm.weight.grad, decoded.grad)
    assert norm.weight.grad.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA deployment router')
def test_router_native_slots_weights_and_fp32_reference_gradient():
    from megatron.lite.primitive.modules.deployment_math import router_topk
    from vllm import _custom_ops as ops

    torch.manual_seed(719)
    for experts in (4, 16):
        logits = torch.randn(7, experts, device='cuda', requires_grad=True)
        bias = torch.linspace(-0.03, 0.03, experts, device='cuda')
        weights, indices = router_topk(logits, bias, topk=2, scaling_factor=1.5)
        expected_weights = torch.empty_like(weights)
        expected_indices = torch.empty_like(indices, dtype=torch.int32)
        ops.topk_hash_softplus_sqrt(
            expected_weights,
            expected_indices,
            torch.empty_like(expected_indices),
            logits.detach(),
            renormalize=True,
            routed_scaling_factor=1.5,
            e_score_correction_bias=bias,
        )
        assert torch.equal(indices, expected_indices.long())
        assert torch.equal(weights, expected_weights)
        incoming = torch.randn_like(weights)
        weights.backward(incoming)
        leaf = logits.detach().clone().requires_grad_()
        selected = torch.nn.functional.softplus(leaf).sqrt().gather(1, indices)
        (selected / selected.sum(-1, keepdim=True) * 1.5).backward(incoming)
        assert torch.equal(logits.grad, leaf.grad)
        assert logits.grad.dtype == torch.float32
        assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA deployment QKV RMS')
@pytest.mark.parametrize('width', [128, 512, 1280, 2048])
def test_qkv_norm_native_forward_fp32_vjp_and_process_state(width):
    from megatron.lite.primitive.modules import deployment_math
    from megatron.lite.primitive.modules.attention.mhc import RMSNorm
    from vllm.model_executor.determinism import batch_invariant
    from vllm.models.common.ops import fused_q_kv_rmsnorm

    torch.manual_seed(942)
    norm = RMSNorm(width, eps=1e-6, device='cuda', dtype=torch.float32)
    norm.deployment_math = True
    with torch.no_grad():
        norm.weight.copy_(1 + torch.randn_like(norm.weight) * 0.03)
    x = torch.randn(2, 13, width, device='cuda').bfloat16().requires_grad_()
    incoming = torch.randn_like(x)
    before = norm(x)
    native, _ = fused_q_kv_rmsnorm(
        x.detach().reshape(-1, width),
        x.detach().reshape(-1, width),
        norm.weight.detach().bfloat16(),
        norm.weight.detach().bfloat16(),
        norm.eps,
    )
    assert torch.equal(before.detach(), native.reshape_as(x))
    batch_invariant.init_batch_invariance()
    after = norm(x)
    assert torch.equal(before, after)
    assert torch.equal(after[:1], norm(x[:1]))
    actual = torch.autograd.grad(after, (x, norm.weight), incoming)
    values = x.detach().requires_grad_()
    master = norm.weight.detach().requires_grad_()
    reference = torch.nn.functional.rms_norm(
        values, (width,), deployment_math.decoded_bf16_master(master), norm.eps
    )
    expected = torch.autograd.grad(reference, (values, master), incoming)
    for got, wanted in zip(actual, expected, strict=True):
        assert torch.isfinite(got).all()
        assert torch.equal(got, wanted)
    assert actual[1].dtype == torch.float32
    assert not torch.equal(actual[1], actual[1].bfloat16().float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA deployment router')
@pytest.mark.parametrize('experts', (256, 384))
@pytest.mark.parametrize('tokens', (1, 7, 65))
def test_release_router_matches_native_dispatch_and_owned_vjp(experts, tokens):
    from megatron.lite.primitive.modules.deployment_math import router_topk

    native = importlib.import_module(
        'vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router'
    )

    torch.manual_seed(1429)
    logits = torch.randn(tokens, experts, device='cuda', requires_grad=True)
    bias = torch.linspace(-0.07, 0.04, experts, device='cuda')
    incoming = torch.randn(tokens, 6, device='cuda')
    weights, indices = router_topk(logits, bias, topk=6, scaling_factor=1.5)
    expected_weights, expected_indices = native.fused_topk_bias(
        torch.empty(tokens, 32, device='cuda', dtype=torch.bfloat16),
        logits.detach(),
        'sqrtsoftplus',
        bias,
        6,
        True,
        indices_type=torch.int32,
        routed_scaling_factor=1.5,
    )
    assert torch.equal(weights, expected_weights)
    assert torch.equal(indices, expected_indices.long())
    weights.backward(incoming)
    leaf = logits.detach().clone().requires_grad_()
    selected = torch.nn.functional.softplus(leaf).sqrt().gather(1, indices)
    (selected / selected.sum(-1, keepdim=True) * 1.5).backward(incoming)
    assert logits.grad.dtype == torch.float32
    assert torch.equal(logits.grad, leaf.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA sparse metadata')
@pytest.mark.parametrize('has_main', (False, True))
def test_sparse_metadata_stable_order_batch1_graph_and_one_launch(
    has_main, monkeypatch
):
    from megatron.lite.primitive.kernels import sparse_metadata as module

    torch.manual_seed(1903)
    batch, rows, window, main_size, topk, width = 3, 7, 65, 41, 32, 128
    mask = torch.rand(batch, rows, window + main_size, device='cuda') > 0.5
    main = (
        torch.randint(
            -3, main_size + 4, (batch, rows, topk), device='cuda', dtype=torch.int32
        )
        if has_main
        else None
    )
    calls = []
    native = module._compact

    class Launcher:
        def __getitem__(self, grid):
            launch = native[grid]

            def call(*args, **kwargs):
                calls.append(grid)
                return launch(*args, **kwargs)

            return call

    monkeypatch.setattr(module, '_compact', Launcher())
    kwargs = dict(main_size=main_size, window_size=window, width=width)
    indices, lengths = module.sparse_metadata(mask, main, **kwargs)
    torch.cuda.synchronize()
    assert calls == [(batch * rows,)]
    expected = torch.full_like(indices, -1)
    expected_lengths = torch.zeros_like(lengths)
    cpu_mask = mask.cpu()
    cpu_main = main.cpu() if has_main else None
    for b in range(batch):
        for s in range(rows):
            chosen = (
                [v for v in cpu_main[b, s].tolist() if 0 <= v < main_size]
                if has_main
                else []
            )
            chosen += [
                main_size + v for v in torch.where(cpu_mask[b, s, :window])[0].tolist()
            ]
            expected[b, s, : len(chosen)] = torch.tensor(
                chosen, device='cuda', dtype=torch.int32
            )
            expected_lengths[b, s] = len(chosen)
    assert torch.equal(indices, expected)
    assert torch.equal(lengths, expected_lengths)
    single = module.sparse_metadata(
        mask[1:2], main[1:2] if has_main else None, **kwargs
    )
    assert torch.equal(single[0], indices[1:2]) and torch.equal(single[1], lengths[1:2])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = module.sparse_metadata(mask, main, **kwargs)
    graph.replay()
    assert torch.equal(actual[0], indices) and torch.equal(actual[1], lengths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA FlashMLA')
def test_sparse_attention_one_batched_call_matches_serial_native(monkeypatch):
    from megatron.lite.primitive.kernels.sparse_metadata import sparse_metadata
    from megatron.lite.primitive.modules import deployment_math as dm
    from vllm.v1.attention.ops import flashmla

    torch.manual_seed(1917)
    batch, rows, heads, dim, window, main_size, width = 2, 5, 32, 512, 65, 19, 128
    q = torch.randn(
        batch, rows, heads, dim, device='cuda', dtype=torch.bfloat16
    ).requires_grad_()
    kv = torch.randn(
        batch, window + main_size, dim, device='cuda', dtype=torch.bfloat16
    ).requires_grad_()
    sink = torch.randn(heads, device='cuda', dtype=torch.float32).requires_grad_()
    mask = torch.rand(batch, rows, window + main_size, device='cuda') > 0.5
    main = torch.randint(
        -1, main_size, (batch, rows, 16), device='cuda', dtype=torch.int32
    )
    indices, lengths = sparse_metadata(
        mask, main, main_size=main_size, window_size=window, width=width
    )
    reordered = torch.cat([kv[:, window:], kv[:, :window]], 1)
    native = flashmla.flash_mla_sparse_fwd
    padded_sink = torch.nn.functional.pad(sink.detach(), (0, 64 - heads))
    expected = []
    for b in range(batch):
        query = torch.nn.functional.pad(q[b].detach(), (0, 0, 0, 64 - heads))
        out = torch.empty_like(query)
        native(
            q=query,
            kv=reordered[b].detach().unsqueeze(1).contiguous(),
            indices=indices[b].unsqueeze(1),
            sm_scale=dim**-0.5,
            attn_sink=padded_sink,
            topk_length=lengths[b],
            out=out,
        )
        expected.append(out[:, :heads])
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs['q'].shape[0])
        return native(*args, **kwargs)

    monkeypatch.setattr(flashmla, 'flash_mla_sparse_fwd', counted)
    kwargs = dict(main_size=main_size, window_size=window, width=width, scale=dim**-0.5)
    actual = dm.sparse_attention(q, kv, sink, mask, main, **kwargs)
    assert calls == [batch * rows]
    assert torch.equal(actual.detach(), torch.stack(expected))
    single = dm.sparse_attention(q[:1], kv[:1], sink, mask[:1], main[:1], **kwargs)
    assert torch.equal(actual[:1], single)
    actual.float().sum().backward()
    assert all(
        x.grad is not None and torch.isfinite(x.grad).all() for x in (q, kv, sink)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA grouped FP8')
def test_grouped_fp8_one_launch_serial_batch1_and_vllm_bitwise(monkeypatch):
    from megatron.lite.primitive.kernels import mxfp8_block32
    from megatron.lite.primitive.quantization import mxfp8

    bi_fp8 = importlib.import_module(
        'vllm.model_executor.kernels.linear.mxfp8.batch_invariant'
    )
    bi_ops = importlib.import_module('vllm.model_executor.determinism.batch_invariant')
    bi_ops.init_batch_invariance()
    torch.manual_seed(1931)
    x = torch.randn(
        2, 7, 4, 256, device='cuda', dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(
        4, 64, 256, device='cuda', dtype=torch.float32, requires_grad=True
    )
    expected = torch.stack(
        [mxfp8.dynamic_fp8_linear(x[:, :, g], weight[g]) for g in range(4)], 2
    )
    calls = []
    native = mxfp8_block32._block32_mm

    class Launcher:
        def __getitem__(self, grid):
            launch = native[grid]

            def call(*args, **kwargs):
                calls.append(grid)
                return launch(*args, **kwargs)

            return call

    monkeypatch.setattr(mxfp8_block32, '_block32_mm', Launcher())
    actual = mxfp8.dynamic_fp8_grouped_linear(x, weight)
    assert len(calls) == 1 and calls[0][2] == 4
    assert torch.equal(actual, expected)
    assert torch.equal(actual[:1], mxfp8.dynamic_fp8_grouped_linear(x[:1], weight))
    activation = mxfp8.quantize_linear_activation(x)
    codes, scales = mxfp8.quantize_block_fp8(weight, (32, 32), scale_format='e8m0')
    # Native rollout stores four E8M0 bytes in each INT32 scale word.
    sa = (
        activation.scale.view(torch.uint8)
        .reshape(-1, 4, 8)
        .contiguous()
        .view(torch.int32)
    )
    sw = (
        scales.view(torch.uint8).repeat_interleave(32, 1).contiguous().view(torch.int32)
    )
    deployed = bi_fp8.packed_block32_grouped_mm(
        activation.values.reshape(-1, 4, 256), sa, codes, sw, torch.float32
    )
    assert torch.equal(actual, deployed.reshape_as(actual))
    expected_gradients = torch.autograd.grad(expected.sum(), (x, weight))
    actual.sum().backward()
    assert torch.equal(x.grad, expected_gradients[0])
    assert torch.equal(weight.grad, expected_gradients[1])
    assert x.grad is not None and weight.grad.dtype == torch.float32
    assert torch.isfinite(x.grad).all() and torch.isfinite(weight.grad).all()
