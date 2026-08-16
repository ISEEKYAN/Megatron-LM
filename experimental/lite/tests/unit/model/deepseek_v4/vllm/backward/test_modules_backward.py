"""Module VJP gates for DS4 vLLM-visible training bridges."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.vllm.primitive._contract import (
    check_parameter_versions,
    parameter_versions,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.linear import (
    block_fp8_linear,
    gate_linear,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.norm import (
    fused_qkv_rms_norm,
    rms_norm,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.mhc import (
    _post_graph,
    _pre_graph,
    mhc_head,
    mhc_post,
    mhc_post_pre,
    mhc_pre_broadcast,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.o_proj import (
    _inverse_rope,
    o_projection,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.router import fixed_route_vjp
from megatron.lite.model.deepseek_v4.vllm.primitive.moe import deep_ep_moe
from megatron.lite.model.deepseek_v4.vllm.primitive.attention import (
    _compressed_sequence_graph,
    _rope_and_qnorm,
    attach_indexer_aux_loss,
    attention_core,
)
from megatron.lite.primitive.modules.attention.hca import split_sinkhorn


def test_inference_tensor_versions_use_non_mutating_sentinel() -> None:
    with torch.inference_mode():
        parameter = torch.ones(2)
        versions = parameter_versions((parameter,))
        assert versions == (-1,)
        check_parameter_versions((parameter,), versions)


def _asymmetric(shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(1, 1 + shape.numel(), dtype=torch.float32)
    return values.reshape(shape).div_(shape.numel()).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_block_fp8_linear_keeps_visible_value_and_uses_master_vjp(dtype) -> None:
    torch.manual_seed(20260813)
    value = torch.randn(2, 3, 5, dtype=dtype, requires_grad=True)
    weight = torch.randn(7, 5, dtype=dtype, requires_grad=True)

    def visible(x, w):
        # The offset represents deployment quantization/fusion error. It must
        # remain visible but must not alter the declared BF16-master VJP.
        return F.linear(x, w) + torch.tensor(0.25, dtype=dtype)

    output = block_fp8_linear(visible, value, weight)
    expected_visible = visible(value.detach(), weight.detach())
    assert torch.equal(output, expected_visible)

    grad_output = _asymmetric(output.shape, dtype)
    output.backward(grad_output)

    ref_value = value.detach().float().requires_grad_(True)
    ref_weight = weight.detach().float().requires_grad_(True)
    F.linear(ref_value, ref_weight).backward(grad_output.float())
    torch.testing.assert_close(value.grad.float(), ref_value.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(weight.grad.float(), ref_weight.grad, rtol=5e-3, atol=5e-3)


def test_gate_linear_uses_bound_master_weight_for_backward() -> None:
    torch.manual_seed(1)
    value = torch.randn(4, 6, requires_grad=True)
    weight = torch.randn(9, 6, requires_grad=True)

    output = gate_linear(lambda x: (F.linear(x, weight), {"aux": 1}), value, weight)
    grad_output = _asymmetric(output.shape, output.dtype)
    output.backward(grad_output)

    expected_dx = grad_output @ weight.detach()
    expected_dw = grad_output.T @ value.detach()
    torch.testing.assert_close(value.grad, expected_dx)
    torch.testing.assert_close(weight.grad, expected_dw)


def test_linear_backward_rejects_master_mutation_after_forward() -> None:
    value = torch.randn(2, 4, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)
    output = block_fp8_linear(F.linear, value, weight)
    with torch.no_grad():
        weight.add_(1)
    with pytest.raises(RuntimeError, match="changed between forward and backward"):
        output.sum().backward()


@pytest.mark.parametrize("shape", [(3, 8), (2, 3, 8)])
def test_rms_norm_analytic_vjp_matches_pytorch(shape) -> None:
    torch.manual_seed(17)
    eps = 1e-6
    value = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(shape[-1], dtype=torch.float64, requires_grad=True)

    def visible(x, w, epsilon):
        return F.rms_norm(x, (x.shape[-1],), w, epsilon)

    output = rms_norm(visible, value, weight, eps)
    grad_output = _asymmetric(output.shape, output.dtype)
    output.backward(grad_output)

    ref_value = value.detach().requires_grad_(True)
    ref_weight = weight.detach().requires_grad_(True)
    visible(ref_value, ref_weight, eps).backward(grad_output)
    torch.testing.assert_close(value.grad, ref_value.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=1e-6, atol=1e-6)


def test_fused_qkv_rms_norm_vjp_matches_two_reference_norms() -> None:
    torch.manual_seed(97)
    eps = 1e-6
    q = torch.randn(4, 6, requires_grad=True)
    kv = torch.randn(4, 8, requires_grad=True)
    qw = torch.randn(6, requires_grad=True)
    kvw = torch.randn(8, requires_grad=True)

    def visible(q_value, kv_value, q_weight, kv_weight, epsilon):
        return (
            F.rms_norm(q_value, (6,), q_weight, epsilon),
            F.rms_norm(kv_value, (8,), kv_weight, epsilon),
        )

    q_out, kv_out = fused_qkv_rms_norm(visible, q, kv, qw, kvw, eps)
    dq_out = _asymmetric(q_out.shape, q_out.dtype)
    dkv_out = -_asymmetric(kv_out.shape, kv_out.dtype)
    torch.autograd.backward((q_out, kv_out), (dq_out, dkv_out))

    refs = [tensor.detach().requires_grad_(True) for tensor in (q, kv, qw, kvw)]
    ref_q, ref_kv = visible(*refs, eps)
    torch.autograd.backward((ref_q, ref_kv), (dq_out, dkv_out))
    for candidate, reference in zip((q, kv, qw, kvw), refs, strict=True):
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=1e-5, atol=1e-6)


def test_mhc_post_analytic_graph_matches_reference() -> None:
    torch.manual_seed(1)
    tensors = (
        torch.randn(3, 4, requires_grad=True),
        torch.randn(3, 2, 4, requires_grad=True),
        torch.randn(3, 2, 1, requires_grad=True),
        torch.randn(3, 2, 2, requires_grad=True),
    )
    output = mhc_post(lambda *args: _post_graph(*args) + 0.125, *tensors)
    assert torch.equal(output, _post_graph(*tensors).detach() + 0.125)
    grad = _asymmetric(output.shape, output.dtype)
    output.backward(grad)

    refs = tuple(value.detach().requires_grad_(True) for value in tensors)
    _post_graph(*refs).backward(grad)
    for candidate, reference in zip(tensors, refs, strict=True):
        torch.testing.assert_close(candidate.grad, reference.grad)


def test_mhc_pre_broadcast_functional_vjp_covers_all_parameters() -> None:
    torch.manual_seed(17)
    mult, hidden, eps, norm_eps = 2, 4, 1e-6, 1e-5
    mix = (2 + mult) * mult
    inputs = (
        torch.randn(3, hidden, requires_grad=True),
        torch.randn(mix, mult * hidden, requires_grad=True),
        torch.randn(3, requires_grad=True),
        torch.randn(mix, requires_grad=True),
        torch.randn(hidden, requires_grad=True),
    )

    def functional(x, fn, scale, base, norm_weight):
        residual, post, comb, value = _pre_graph(
            x, fn, scale, base, mult=mult, iters=3, eps=eps
        )
        value = F.rms_norm(value, (hidden,), norm_weight, norm_eps)
        return residual, post, comb, value

    output = mhc_pre_broadcast(
        functional,
        *inputs,
        mult=mult,
        iters=3,
        eps=eps,
        norm_eps=norm_eps,
    )
    grad_outputs = tuple(_asymmetric(value.shape, value.dtype) for value in output)
    torch.autograd.backward(output, grad_outputs)

    refs = tuple(value.detach().requires_grad_(True) for value in inputs)
    torch.autograd.backward(functional(*refs), grad_outputs)
    for candidate, reference in zip(inputs, refs, strict=True):
        assert candidate.grad is not None
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=1e-5, atol=1e-6)


def test_mhc_post_pre_functional_vjp_covers_transition_parameters() -> None:
    torch.manual_seed(19)
    mult, hidden, eps, norm_eps = 2, 4, 1e-6, 1e-5
    mix = (2 + mult) * mult
    inputs = (
        torch.randn(3, hidden, requires_grad=True),
        torch.randn(3, mult, hidden, requires_grad=True),
        torch.randn(3, mult, 1, requires_grad=True),
        torch.randn(3, mult, mult, requires_grad=True),
        torch.randn(mix, mult * hidden, requires_grad=True),
        torch.randn(3, requires_grad=True),
        torch.randn(mix, requires_grad=True),
        torch.randn(hidden, requires_grad=True),
    )

    def functional(x, residual, post, comb, fn, scale, base, norm_weight):
        streams = _post_graph(x, residual, post, comb)
        new_residual, new_post, new_comb, value = _pre_graph(
            streams, fn, scale, base, mult=mult, iters=3, eps=eps
        )
        value = F.rms_norm(value, (hidden,), norm_weight, norm_eps)
        return new_residual, new_post, new_comb, value

    output = mhc_post_pre(
        functional,
        *inputs,
        mult=mult,
        iters=3,
        eps=eps,
        norm_eps=norm_eps,
    )
    grad_outputs = tuple(_asymmetric(value.shape, value.dtype) for value in output)
    torch.autograd.backward(output, grad_outputs)

    refs = tuple(value.detach().requires_grad_(True) for value in inputs)
    torch.autograd.backward(functional(*refs), grad_outputs)
    for candidate, reference in zip(inputs, refs, strict=True):
        assert candidate.grad is not None
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=1e-5, atol=1e-6)


def test_mhc_head_functional_vjp_covers_all_parameters() -> None:
    torch.manual_seed(23)
    tokens, mult, hidden, eps = 3, 2, 4, 1e-6
    inputs = (
        torch.randn(tokens, mult, hidden, requires_grad=True),
        torch.randn(mult, mult * hidden, requires_grad=True),
        torch.randn(mult, requires_grad=True),
        torch.randn(mult, requires_grad=True),
    )

    def functional(x, fn, scale, base):
        flat = x.flatten(-2).float()
        rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
        mixes = F.linear(flat, fn.float()) * rstd
        pre = torch.sigmoid(mixes * scale.float() + base.float()) + eps
        return torch.sum(pre.unsqueeze(-1) * x.float(), dim=-2).to(x.dtype)

    output = mhc_head(functional, *inputs, eps=eps)
    grad = _asymmetric(output.shape, output.dtype)
    output.backward(grad)

    refs = tuple(value.detach().requires_grad_(True) for value in inputs)
    functional(*refs).backward(grad)
    for candidate, reference in zip(inputs, refs, strict=True):
        assert candidate.grad is not None
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=1e-5, atol=1e-6)


def test_o_projection_functional_vjp_replays_positions() -> None:
    torch.manual_seed(97)
    tokens, groups, heads, nope, rope, rank, output_size = 3, 2, 2, 4, 4, 3, 5
    o = torch.randn(tokens, groups * heads, nope + rope, requires_grad=True)
    wa = torch.randn(groups * rank, heads * (nope + rope), requires_grad=True)
    wb = torch.randn(output_size, groups * rank, requires_grad=True)
    positions = torch.tensor([2, 0, 3])
    cache = torch.randn(5, rope)

    def functional(o_, wa_, wb_):
        inverse = _inverse_rope(o_, positions, cache, nope, rope)
        z = torch.einsum(
            "tgd,grd->tgr",
            inverse.reshape(tokens, groups, -1),
            wa_.reshape(groups, rank, -1),
        )
        return F.linear(z.flatten(1), wb_)

    output = o_projection(
        lambda *args: functional(*args) + 0.25,
        o,
        wa,
        wb,
        positions=positions,
        cos_sin_cache=cache,
        n_groups=groups,
        heads_per_group=heads,
        nope_dim=nope,
        rope_dim=rope,
        o_lora_rank=rank,
    )
    assert torch.equal(output, functional(o, wa, wb).detach() + 0.25)
    grad = _asymmetric(output.shape, output.dtype)
    output.backward(grad)

    refs = tuple(value.detach().requires_grad_(True) for value in (o, wa, wb))
    functional(*refs).backward(grad)
    for candidate, reference in zip((o, wa, wb), refs, strict=True):
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=1e-5, atol=1e-6)


def test_router_backward_uses_visible_fixed_ids() -> None:
    torch.manual_seed(20260813)
    logits = torch.randn(3, 7, requires_grad=True)
    ids = torch.tensor([[4, 1], [0, 6], [3, 2]], dtype=torch.int32)
    scale = 1.5

    def visible(value):
        scores = torch.sqrt(F.softplus(value)).gather(-1, ids.long())
        return scores / scores.sum(-1, keepdim=True) * scale, ids

    weights, visible_ids = fixed_route_vjp(
        visible, logits, renormalize=True, route_scale=scale
    )
    assert torch.equal(visible_ids, ids)
    grad = _asymmetric(weights.shape, weights.dtype)
    weights.backward(grad)

    reference = logits.detach().requires_grad_(True)
    visible(reference)[0].backward(grad)
    torch.testing.assert_close(logits.grad, reference.grad)


def test_router_backward_keeps_ids_snapshot_when_consumer_mutates_output() -> None:
    torch.manual_seed(31)
    logits = torch.randn(2, 5, requires_grad=True)
    original_ids = torch.tensor([[4, 1], [0, 3]], dtype=torch.int32)

    def visible(value):
        scores = torch.sqrt(F.softplus(value)).gather(-1, original_ids.long())
        return scores / scores.sum(-1, keepdim=True), original_ids.clone()

    weights, returned_ids = fixed_route_vjp(
        visible, logits, renormalize=True, route_scale=1.0
    )
    returned_ids.fill_(99)
    grad = _asymmetric(weights.shape, weights.dtype)
    weights.backward(grad)

    reference = logits.detach().requires_grad_(True)
    visible(reference)[0].backward(grad)
    torch.testing.assert_close(logits.grad, reference.grad)


def test_grouped_moe_backward_covers_hidden_probability_and_weights() -> None:
    torch.manual_seed(17)
    tokens, hidden_size, intermediate, experts, topk = 4, 6, 5, 2, 2
    hidden = torch.randn(tokens, hidden_size, requires_grad=True)
    probs = torch.rand(tokens, topk, requires_grad=True)
    ids = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.int64)
    w13 = tuple(
        torch.randn(2 * intermediate, hidden_size, requires_grad=True)
        for _ in range(experts)
    )
    w2 = tuple(
        torch.randn(hidden_size, intermediate, requires_grad=True)
        for _ in range(experts)
    )

    def functional(hidden_, probs_, ids_, *weights):
        first, second = weights[:experts], weights[experts:]
        output = torch.zeros_like(hidden_)
        for token in range(tokens):
            for slot in range(topk):
                expert = int(ids_[token, slot])
                gate, up = F.linear(hidden_[token], first[expert]).chunk(2)
                expert_out = F.linear(F.silu(gate) * up, second[expert])
                output[token] = output[token] + probs_[token, slot] * expert_out
        return output

    output = deep_ep_moe(
        lambda *args: functional(*args) + 0.125,
        hidden,
        probs,
        ids,
        w13,
        w2,
        global_expert_start=0,
    )
    assert torch.equal(output, functional(hidden, probs, ids, *w13, *w2).detach() + 0.125)
    grad = _asymmetric(output.shape, output.dtype)
    output.backward(grad)

    refs = tuple(
        value.detach().requires_grad_(True) for value in (hidden, probs, *w13, *w2)
    )
    functional(refs[0], refs[1], ids, *refs[2:]).backward(grad)
    for candidate, reference in zip((hidden, probs, *w13, *w2), refs, strict=True):
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=1e-5, atol=1e-6)


def test_attention_core_replays_slots_and_precache_rope_vjp() -> None:
    torch.manual_seed(97)
    tokens, heads, dim, rope_dim = 3, 2, 6, 4
    q = torch.randn(tokens, heads, dim, requires_grad=True)
    kv = torch.randn(tokens, dim, requires_grad=True)
    workspace = torch.randn(5, 1, dim)
    indices = torch.zeros(tokens, 1, 2, dtype=torch.int32)
    lengths = torch.full((tokens,), 2, dtype=torch.int32)
    sink = torch.zeros(heads)
    slots = torch.tensor([4, 1, 3], dtype=torch.int64)
    positions = torch.tensor([2, 0, 1], dtype=torch.int64)
    cache = torch.randn(4, rope_dim)
    q_visible = _rope_and_qnorm(q.detach(), positions, cache, rope_dim, 1e-6, normalize=True)
    visible_out = q_visible + 0.25
    lse = torch.randn(tokens, heads)
    dq_visible = torch.randn_like(q_visible)
    dworkspace = torch.randn_like(workspace)

    def visible(q_, kv_):
        del q_, kv_
        return visible_out, lse, q_visible

    def backward(*args):
        del args
        return dq_visible, dworkspace

    output = attention_core(
        visible,
        q,
        kv,
        workspace,
        indices,
        lengths,
        sink,
        slots,
        positions,
        cache,
        softmax_scale=0.5,
        eps=1e-6,
        rope_dim=rope_dim,
        backward_op=backward,
    )
    assert torch.equal(output, visible_out)
    output.backward(torch.ones_like(output))

    rq = q.detach().requires_grad_(True)
    rkv = kv.detach().requires_grad_(True)
    fq = _rope_and_qnorm(rq, positions, cache, rope_dim, 1e-6, normalize=True)
    fkv = _rope_and_qnorm(rkv, positions, cache, rope_dim, 1e-6, normalize=False)
    expected_dkv = dworkspace.reshape(-1, dim).index_select(0, slots)
    torch.autograd.backward((fq, fkv), (dq_visible, expected_dkv))
    torch.testing.assert_close(q.grad, rq.grad)
    torch.testing.assert_close(kv.grad, rkv.grad)


def test_attention_core_packed_batch_uses_workspace_coordinates() -> None:
    """Paged-cache slots must never address the padded FlashMLA workspace."""
    torch.manual_seed(20260815)
    counts = (3, 2)
    tokens, heads, dim, rope_dim = sum(counts), 2, 6, 4
    workspace_width = max(counts)
    q = torch.randn(tokens, heads, dim, requires_grad=True)
    kv = torch.randn(tokens, dim, requires_grad=True)
    workspace = torch.randn(len(counts) * workspace_width, 1, dim)
    dworkspace = torch.randn_like(workspace)
    positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)
    # The second request owns a distant physical cache page.  Using these as
    # workspace rows reproduces the RL device-side gather assertion.
    physical_slots = torch.tensor([0, 1, 2, 64, 65], dtype=torch.int64)
    workspace_slots = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    query_start = torch.tensor([0, 3, 5], dtype=torch.int32)
    cache = torch.randn(max(physical_slots).item() + 1, rope_dim)
    q_visible = _rope_and_qnorm(
        q.detach(), positions, cache, rope_dim, 1e-6, normalize=True
    )

    output = attention_core(
        lambda *_: (q_visible, torch.zeros(tokens, heads), q_visible),
        q,
        kv,
        workspace,
        torch.zeros(tokens, 1, 1, dtype=torch.int32),
        torch.ones(tokens, dtype=torch.int32),
        torch.zeros(heads),
        workspace_slots,
        positions,
        cache,
        softmax_scale=0.5,
        eps=1e-6,
        rope_dim=rope_dim,
        backward_op=lambda *_: (torch.zeros_like(q_visible), dworkspace),
        query_start_loc=query_start,
    )
    output.sum().backward()

    reference = kv.detach().requires_grad_(True)
    functional = _rope_and_qnorm(
        reference, positions, cache, rope_dim, 1e-6, normalize=False
    )
    functional.backward(
        dworkspace.reshape(-1, dim).index_select(0, workspace_slots)
    )
    torch.testing.assert_close(kv.grad, reference.grad)


def test_attention_core_saves_metadata_finalized_inside_visible_call() -> None:
    tokens, heads, dim = 3, 2, 6
    q = torch.randn(tokens, heads, dim, requires_grad=True)
    kv = torch.randn(tokens, dim, requires_grad=True)
    stale_workspace = torch.zeros(tokens, 1, dim)
    stale_indices = torch.full((tokens, 1, 4), -1, dtype=torch.int32)
    stale_lengths = torch.empty(tokens, dtype=torch.int32)
    actual_workspace = torch.randn_like(stale_workspace)
    actual_indices = torch.zeros_like(stale_indices)
    actual_lengths = torch.ones(tokens, dtype=torch.int32)
    positions = slots = torch.arange(tokens)
    cache = torch.randn(tokens, 2)
    seen = {}

    def visible(*_):
        return (
            q.detach(),
            torch.zeros(tokens, heads),
            q.detach(),
            actual_workspace,
            actual_indices,
            actual_lengths,
        )

    def backward(q_, workspace_, out_, grad_, lse_, sink_, indices_, scale_, lengths_):
        del q_, out_, grad_, lse_, sink_, scale_
        seen["workspace"] = workspace_
        seen["indices"] = indices_
        seen["lengths"] = lengths_
        return torch.zeros_like(q), torch.zeros_like(actual_workspace)

    output = attention_core(
        visible,
        q,
        kv,
        stale_workspace,
        stale_indices,
        stale_lengths,
        torch.zeros(heads),
        slots,
        positions,
        cache,
        softmax_scale=0.5,
        eps=1e-6,
        rope_dim=2,
        backward_op=backward,
    )
    output.sum().backward()

    assert seen["workspace"] is actual_workspace
    assert seen["indices"] is actual_indices
    assert seen["lengths"] is actual_lengths


def test_indexer_aux_loss_replays_fixed_topk_and_covers_indexer_inputs() -> None:
    torch.manual_seed(20260813)
    tokens, ratio, heads, index_heads = 8, 4, 2, 3
    dim, index_dim, rope_dim = 6, 4, 2
    output = torch.randn(tokens, heads, dim, requires_grad=True)
    q = torch.randn(tokens, heads, dim, requires_grad=True)
    index_q = torch.randn(tokens, index_heads, index_dim, requires_grad=True)
    index_score = torch.randn(tokens, 4 * index_dim, requires_grad=True)
    index_weights = torch.randn(tokens, index_heads, requires_grad=True)
    main_score = torch.randn(tokens, 4 * dim, requires_grad=True)
    index_ape = torch.randn(ratio, 2 * index_dim, requires_grad=True)
    index_norm = torch.randn(index_dim, requires_grad=True)
    main_ape = torch.randn(ratio, 2 * dim, requires_grad=True)
    main_norm = torch.randn(dim, requires_grad=True)
    positions = torch.arange(tokens)
    cache = torch.randn(tokens, rope_dim)
    topk = torch.full((tokens, 2), -1, dtype=torch.int32)
    topk[3:7, 0] = 0
    topk[7] = torch.tensor([1, 0], dtype=torch.int32)

    visible = attach_indexer_aux_loss(
        output,
        q,
        index_q,
        index_score,
        index_weights,
        main_score,
        index_ape,
        index_norm,
        main_ape,
        main_norm,
        positions,
        cache,
        topk,
        ratio=ratio,
        rope_dim=rope_dim,
        eps=1e-6,
        softmax_scale=dim**-0.5,
        loss_coeff=0.1,
    )
    assert torch.equal(visible, output)
    visible.square().mean().backward()
    for value in (index_q, index_score, index_weights, index_ape, index_norm):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize("ratio,tokens", [(4, 8), (128, 128)])
def test_attention_core_compressor_vjp_and_swa_workspace_offset(ratio, tokens) -> None:
    torch.manual_seed(20260813 + ratio)
    heads, dim, rope_dim = 2, 6, 4
    coff = 2 if ratio == 4 else 1
    q = torch.randn(tokens, heads, dim, requires_grad=True)
    kv = torch.randn(tokens, dim, requires_grad=True)
    score = torch.randn(tokens, 2 * coff * dim, requires_grad=True)
    ape = torch.randn(ratio, coff * dim, requires_grad=True)
    norm = torch.randn(dim, requires_grad=True)
    compressed_len = tokens // ratio
    workspace = torch.randn(compressed_len + tokens, 1, dim)
    indices = torch.zeros(tokens, 1, 2, dtype=torch.int32)
    lengths = torch.full((tokens,), 2, dtype=torch.int32)
    sink = torch.zeros(heads)
    slots = torch.arange(tokens)
    positions = torch.arange(tokens)
    cache = torch.randn(tokens + 1, rope_dim)
    q_visible = _rope_and_qnorm(q.detach(), positions, cache, rope_dim, 1e-6, normalize=True)
    dworkspace = torch.randn_like(workspace)

    output = attention_core(
        lambda *_: (q_visible, torch.zeros(tokens, heads), q_visible),
        q,
        kv,
        workspace,
        indices,
        lengths,
        sink,
        slots,
        positions,
        cache,
        softmax_scale=0.5,
        eps=1e-6,
        rope_dim=rope_dim,
        backward_op=lambda *_: (torch.zeros_like(q_visible), dworkspace),
        compressor_kv_score=score,
        compressor_ape=ape,
        compressor_norm=norm,
        compressor_ratio=ratio,
    )
    output.sum().backward()

    rkv = kv.detach().requires_grad_(True)
    functional_kv = _rope_and_qnorm(
        rkv, positions, cache, rope_dim, 1e-6, normalize=False
    )
    functional_kv.backward(dworkspace[compressed_len:, 0])
    torch.testing.assert_close(kv.grad, rkv.grad)
    refs = tuple(value.detach().requires_grad_(True) for value in (score, ape, norm))
    compressed = _compressed_sequence_graph(
        *refs,
        positions,
        cache,
        ratio=ratio,
        head_dim=dim,
        rope_dim=rope_dim,
        eps=1e-6,
    )
    compressed.backward(dworkspace[:compressed_len, 0])
    for candidate, reference in zip((score, ape, norm), refs, strict=True):
        assert candidate.grad is not None
        torch.testing.assert_close(candidate.grad, reference.grad)
