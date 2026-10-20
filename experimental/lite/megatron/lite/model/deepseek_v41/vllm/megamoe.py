# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MegaMoE forward with vLLM's fused DeepGEMM ``fp8_fp4_mega_moe``; owned backward.

Forward is the inference launch of ``DeepGemmMegaMoEExperts.forward`` (MXFP4
routed experts, fused MXFP8 shared expert, SwiGLU clamp, fast math): the
masters are quantized with the training codecs (MXFP4 / 32x32 E8M0 FP8, both
reproduce the release bytes), written into the kernel layout by vLLM's own
loader transforms, then ``prepare_megamoe_inputs`` and the kernel run over the
EP group's symmetric buffer. Dispatch, expert GEMMs, the L1 epilogue and the
slot-order combine are the kernel's, so EP>1 matches inference by construction.

Backward replays a differentiable FP32 reference of the same chain (decoded
operands, identity STE through every quantization), gathering tokens over the
EP group with Core's autograd collectives.
"""

import torch

from .linear import accumulate_wgrad, tc_mm

_BUFFERS = {}


def _layout():
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmMegaMoEExperts

    return DeepGemmMegaMoEExperts


def _write(dst, src, shard, is_scale):
    layout = _layout()
    rows = dst.shape[0] // (1 if shard == "w2" else 2)
    src = layout._pack_scale(src, rows, dst.shape[1] * 4) if is_scale else src.view(dst.dtype)
    dst, src = layout._kernel_layout_shard(dst, src, shard, is_scale)
    dst.copy_(src)


def _empty_weight(outer, n, k, values_per_element, device):
    weight = torch.zeros(
        *outer,
        n,
        k // values_per_element,
        dtype=torch.int8 if values_per_element == 2 else torch.float8_e4m3fn,
        device=device,
    )
    scale = torch.zeros(*outer, k // 128, n, dtype=torch.int32, device=device).transpose(-1, -2)
    return weight, scale


@torch.no_grad()
def kernel_weights(fc1, fc2, shared_fc1, shared_fc2):
    """Masters -> ``(l1, l2, shared_l1, shared_l2)`` in the MegaMoE kernel layout.

    ``fc1[e]`` is ``cat(w1, w3)`` [2I, H]; ``fc2[e]`` is ``w2`` [H, I].
    """
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

    device = fc1[0].device
    inter, hidden = fc1[0].shape[0] // 2, fc1[0].shape[1]
    w13, s13 = _empty_weight((len(fc1),), 2 * inter, hidden, 2, device)
    w2, s2 = _empty_weight((len(fc2),), hidden, inter, 2, device)
    for e, (a, b) in enumerate(zip(fc1, fc2)):
        for shard, part in (("w1", a[:inter]), ("w3", a[inter:])):
            codes, scale = quantize_mxfp4(part.detach().bfloat16())
            _write(w13[e], codes, shard, False)
            _write(s13[e], scale, shard, True)
        codes, scale = quantize_mxfp4(b.detach().bfloat16())
        _write(w2[e], codes, "w2", False)
        _write(s2[e], scale, "w2", True)
    shared = []
    s_inter = shared_fc1.shape[0] // 2
    l1, l1s = _empty_weight((), 2 * s_inter, hidden, 1, device)
    l2, l2s = _empty_weight((), hidden, s_inter, 1, device)
    for shard, part, (w, s) in (
        ("w1", shared_fc1[:s_inter], (l1, l1s)),
        ("w3", shared_fc1[s_inter:], (l1, l1s)),
        ("w2", shared_fc2, (l2, l2s)),
    ):
        codes, scale = quantize_block_fp8(part.detach().bfloat16(), (32, 32), scale_format="e8m0")
        _write(w, codes, shard, False)
        _write(s, scale, shard, True)
    shared = ((l1, l1s), (l2, l2s))
    return (w13, s13), (w2, s2), shared[0], shared[1]


def symm_buffer(group, num_experts, max_tokens, top_k, hidden, inter, shared_experts=1):
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    key = (id(group), torch.cuda.current_device(), num_experts, max_tokens, top_k, hidden, inter)
    if key not in _BUFFERS:
        torch.cuda.empty_cache()
        _BUFFERS[key] = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            max_tokens,
            top_k,
            hidden,
            inter,
            num_shared_experts=shared_experts,
            mma_type="fp8xfp4",
            activation="swiglu",
        )
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
    return _BUFFERS[key]


def launch(x, weights, ids, packed, buffer, limit, ep_size):
    """The inference launch over prepacked kernel weights. Returns bf16 [T, H]."""
    from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    tokens = x.shape[0]
    if tokens > buffer.num_max_tokens_per_rank:
        raise ValueError(
            f"MegaMoE buffer holds {buffer.num_max_tokens_per_rank} tokens, got {tokens}"
        )
    l1, l2, shared_l1, shared_l2 = packed
    num_experts = l1[0].shape[0] * ep_size
    block_m = deep_gemm.get_block_m_for_mega_moe(
        ep_size, num_experts, buffer.num_max_tokens_per_rank, tokens, ids.shape[1], "fp8xfp4"
    )
    prepare_megamoe_inputs(
        x.contiguous(),
        weights.contiguous(),
        ids.contiguous(),
        buffer.x[:tokens],
        buffer.x_sf[:tokens],
        buffer.topk_idx[:tokens],
        buffer.topk_weights[:tokens],
        is_padding=None,
        shared_x_sf=buffer.shared_l1_acts_sf,
        shared_block_m=block_m,
        hidden_quant=kMxfp8Dynamic,
    )
    y = torch.empty_like(x, dtype=torch.bfloat16)
    deep_gemm.fp8_fp4_mega_moe(
        y,
        l1,
        l2,
        buffer,
        fast_math=True,
        activation_clamp=limit,
        shared_l1_weights=shared_l1,
        shared_l2_weights=shared_l2,
    )
    return y


# ---- owned backward ------------------------------------------------------------


def _mx(w):
    from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4, quantize_mxfp4

    return dequantize_mxfp4(*quantize_mxfp4(w.detach().bfloat16())).bfloat16()  # exact


def _fp8(w):
    from megatron.lite.primitive.quantization.block_fp8 import (
        dequantize_block_fp8,
        quantize_block_fp8,
    )

    codes, scale = quantize_block_fp8(w.detach().bfloat16(), (32, 32), scale_format="e8m0")
    return dequantize_block_fp8(codes, scale, (32, 32)).bfloat16()  # exact


def _swiglu(h, limit):
    gate, up = h.chunk(2, -1)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return torch.nn.functional.silu(gate) * up


def _decode_activation(x):
    from .experts import dequantize_activation, quantize_activation

    return dequantize_activation(*quantize_activation(x.detach().bfloat16()))


def _requant_l2(h):
    """The kernel's L1-epilogue requant of the FP32 activation, decoded: 1x32 groups, UE8M0
    scale = smallest power of two with amax / scale <= 448 (DeepGEMM ``get_ue8m0_sf_exp``,
    exponent floor 2^-22), E4M3 round-to-nearest-even."""
    groups = h.detach().float().reshape(*h.shape[:-1], h.shape[-1] // 32, 32)
    bits = groups.abs().amax(-1, keepdim=True).view(torch.int32).to(torch.int64)
    exp = torch.clamp((bits + 0x7FFFFF - 0x600000) >> 23, min=113) - 8
    codes = (groups * torch.exp2((127 - exp).float())).to(torch.float8_e4m3fn)
    return (codes.float() * torch.exp2((exp - 127).float())).reshape(h.shape)


def _mlp_grads(a, score, grad, w1, w2, limit, need):
    """VJP of ``q(swiglu(bf16(a @ w1.T)) * score) @ w2.T`` over decoded operands, as the
    fused kernel computes it: the GEMM1 accumulator is rounded to BF16 before the clamp and
    SwiGLU, ``q`` is the FP8 requant of the FP32 activation (both identity STE).
    GEMMs take BF16 operands on tensor cores with FP32 accumulation (decoded operands are
    exact in BF16). Returns FP32 (da, dscore, dw1, dw2), None where unneeded."""
    need_a, need_s, need_w1, need_w2 = need
    h = tc_mm(a, w1.T).float()
    with torch.enable_grad():
        h_ = h.requires_grad_(need_a or need_w1)
        s_ = score.detach().float().requires_grad_(need_s)
        act = _swiglu(h_, limit) * s_.unsqueeze(-1)
    dw2 = tc_mm(grad.T, _requant_l2(act), torch.float32) if need_w2 else None
    dh = ds = None
    wanted = [t for t in (h_, s_) if t.requires_grad]
    if wanted:
        grads = iter(torch.autograd.grad(act, wanted, tc_mm(grad, w2, torch.float32)))
        dh = next(grads) if h_.requires_grad else None
        ds = next(grads) if s_.requires_grad else None
    da = tc_mm(dh, w1, torch.float32) if need_a else None
    dw1 = tc_mm(dh.T, a, torch.float32) if need_w1 else None
    return da, ds, dw1, dw2


def _all_gather(x, group):
    out = x.new_empty((group.size() * x.shape[0], *x.shape[1:]))
    torch.distributed.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _reduce_scatter(x, group):
    out = x.new_empty((x.shape[0] // group.size(), *x.shape[1:]))
    torch.distributed.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


class _MegaMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, ids, options, *masters):
        group, buffer, limit, packed, local = options
        ctx.options = options
        ctx.save_for_backward(x, weights, ids, *masters)
        ctx.mark_non_differentiable(ids)
        world = 1 if group is None else group.size()
        return launch(x, weights, ids, packed, buffer, limit, world)

    @staticmethod
    def backward(ctx, grad):
        """Decoded-operand VJP, one expert at a time (bounded memory). With EP > 1 tokens and
        grads are all-gathered (BF16: what the tensor-core GEMMs consume) and input/score
        grads reduce-scattered back (FP32). Expert weight gradients go to ``main_grad``."""
        x, weights, ids, *masters = ctx.saved_tensors
        group, _, limit, _, local = ctx.options
        need_x, need_w = ctx.needs_input_grad[:2]
        need_m = ctx.needs_input_grad[4:]
        fc1, fc2, s1, s2 = masters[:local], masters[local : 2 * local], *masters[2 * local :]
        grad = grad.float()
        with torch.autocast(device_type="cuda", enabled=False):
            a = _decode_activation(x)
            dx, ds, dw1s, dw2s = _mlp_grads(
                a,
                a.new_ones(a.shape[0]),
                grad,
                _fp8(s1),
                _fp8(s2),
                limit,
                (need_x, False, need_m[-2], need_m[-1]),
            )
            dx = torch.zeros_like(a) if dx is None else dx
            world = 1 if group is None else group.size()
            rank = 0 if group is None else group.rank()
            rows, k = ids.shape
            if world > 1:
                capacity = torch.tensor([rows], device=x.device)
                torch.distributed.all_reduce(capacity, torch.distributed.ReduceOp.MAX, group=group)
                pad = int(capacity) - rows
                padded = lambda t, v=0: torch.nn.functional.pad(
                    t, (0, 0, 0, pad), value=v
                )  # noqa: E731
                a_all = _all_gather(padded(a.bfloat16()), group)
                g_all = _all_gather(padded(grad.bfloat16()), group)
                w_all = _all_gather(padded(weights.float()), group)
                ids_all = _all_gather(padded(ids, -1), group)
            else:
                a_all, g_all, w_all, ids_all = a, grad, weights.float(), ids
            da_all = torch.zeros(a_all.shape, dtype=torch.float32, device=a_all.device)
            dw_all = torch.zeros_like(w_all)
            flat = ids_all.flatten()
            dfc1, dfc2 = [None] * local, [None] * local
            for e in range(local):
                pos = (flat == rank * local + e).nonzero().flatten()
                if not pos.numel():
                    if need_m[e]:
                        dfc1[e] = accumulate_wgrad((fc1[e],), None, fc1[e])
                    if need_m[local + e]:
                        dfc2[e] = accumulate_wgrad((fc2[e],), None, fc2[e])
                    continue
                token = pos // k
                da, dsc, d1, d2 = _mlp_grads(
                    a_all[token],
                    w_all.flatten()[pos],
                    g_all[token],
                    _mx(fc1[e]),
                    _mx(fc2[e]),
                    limit,
                    (need_x, need_w, need_m[e], need_m[local + e]),
                )
                if da is not None:
                    da_all.index_add_(0, token, da)
                if dsc is not None:
                    dw_all.view(-1)[pos] = dsc
                dfc1[e] = None if d1 is None else accumulate_wgrad((fc1[e],), d1, fc1[e])
                dfc2[e] = None if d2 is None else accumulate_wgrad((fc2[e],), d2, fc2[e])
            if world > 1:
                da_all = _reduce_scatter(da_all, group)[:rows]
                dw_all = _reduce_scatter(dw_all, group)[:rows]
            dx = (dx + da_all).to(x.dtype) if need_x else None
            dweights = dw_all.to(weights.dtype) if need_w else None
        return (
            dx,
            dweights,
            None,
            None,
            *dfc1,
            *dfc2,
            None if dw1s is None else accumulate_wgrad((s1,), dw1s, s1),
            None if dw2s is None else accumulate_wgrad((s2,), dw2s, s2),
        )


def mega_moe(
    x,
    weights,
    ids,
    fc1,
    fc2,
    shared_fc1,
    shared_fc2,
    *,
    limit,
    group=None,
    max_tokens=8192,
    packed=None,
):
    """Routed + shared MegaMoE output bf16 [T, H] for this rank's tokens.

    ``fc1``/``fc2``: this rank's local experts (global ids ``rank*L .. rank*L+L-1``).
    ``group``: EP process group (None: single-rank group of the default PG).
    ``max_tokens``: symmetric-buffer tokens per rank (inference max_num_batched_tokens);
    more rows run in chunks, every EP rank launching the same number of chunks.
    """
    if group is None:
        group = torch.distributed.group.WORLD
    world = group.size()
    if packed is None:
        packed = kernel_weights(fc1, fc2, shared_fc1, shared_fc2)
    buffer = symm_buffer(
        group, len(fc1) * world, max_tokens, ids.shape[1], x.shape[1], fc1[0].shape[0] // 2
    )
    options = (group if world > 1 else None, buffer, limit, packed, len(fc1))
    chunks = torch.tensor([max(1, -(-x.shape[0] // max_tokens))], device=x.device)
    if world > 1:
        torch.distributed.all_reduce(chunks, torch.distributed.ReduceOp.MAX, group=group)
    outputs = []
    for c in range(int(chunks)):
        rows = slice(c * max_tokens, (c + 1) * max_tokens)
        outputs.append(
            _MegaMoE.apply(
                x[rows], weights[rows], ids[rows], options, *fc1, *fc2, shared_fc1, shared_fc2
            )
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)
