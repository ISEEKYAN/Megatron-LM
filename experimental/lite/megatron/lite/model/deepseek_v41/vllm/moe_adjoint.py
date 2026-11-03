# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit routed-expert adjoint for the native MegaMoE LoRA hooks.

The backward no longer replays the whole MoE as an eager autograd graph. Routed rows
travel to their expert owner in one packed variable all-to-all (FP8 input codes and
scales, BF16 output adjoint, score, expert id, LoRA shrink), the owner recomputes only
GEMM1 (native FP8xFP4 grouped kernel + low-rank gate/up delta) and the BI SwiGLU, then
applies the closed-form VJP of the replayed graph: frozen-base dgrad by grouped GEMM
over a per-call decoded operand (freed on return), low-rank parameter gradients by
grouped wgrad, and one packed all-to-all brings (dX, dscore, d shrink) home. The
adjoint is the same straight-through approximation the replay computed: GEMM1 is
BF16 base plus delta, and the FP8 requant of the activations is the identity.
"""
import torch

from megatron.core.fusions.fused_bias_swiglu import (
    clamped_weighted_swiglu_back,
    weighted_swiglu_back,
)

from . import experts
from .batching import dense_mm, grouped_mm, packed_weights


def _quantize(x):
    return experts.quantize_activation(x)


def _decoded(weights):
    from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4

    codes, scales = weights
    if codes.is_cuda:
        return experts._decode_weight(codes, scales, torch.bfloat16)
    return dequantize_mxfp4(codes, scales).bfloat16()


def _ends(splits, device):
    return torch.tensor(splits, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)


def _gmm(left, right, splits):
    """``left[rows of e] @ right[e]`` for every local expert; BF16 operands and output."""
    left, right = left.bfloat16().contiguous(), right.bfloat16().contiguous()
    if not left.shape[0]:
        return left.new_empty((0, right.shape[-1]))
    if left.is_cuda:
        return torch._grouped_mm(left, right, offs=_ends(splits, left.device))
    return torch.cat(
        [(a.float() @ b.float()).bfloat16() for a, b in zip(left.split(splits), right.unbind(0))]
    )


def _fc1_base(codes, scale, weights, splits):
    """Native grouped FP8 x MXFP4 GEMM1, the kernel the replay's forward used."""
    if codes.is_cuda:
        return experts._grouped(codes, scale, *weights, splits)
    a = experts.dequantize_activation(codes.float(), scale).bfloat16()
    return _gmm(a, _decoded(weights).transpose(-1, -2), splits)


def _activation(fc1, scores, limit):
    """Native BF16 post-SwiGLU rows, the value GEMM2 and its low-rank path consumed."""
    if fc1.is_cuda:
        return experts.activation(fc1, scores, limit, native_input=True)[0].detach()
    gate, up = fc1.float().chunk(2, -1)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return (torch.nn.functional.silu(gate) * up * scores.float()[:, None]).bfloat16()


def _swiglu_back(grad, fc1, scores, limit):
    args = (grad.float(), fc1.float(), scores.float()[:, None])
    if limit is None:
        return weighted_swiglu_back(*args)
    return clamped_weighted_swiglu_back(*args, limit)


def _pack(fields):
    """Row-wise byte concatenation, padded to 16 bytes so the buffer stays BF16-viewable."""
    raw = [
        (
            f.contiguous().view(torch.uint8).reshape(f.shape[0], f[0].numel() * f.element_size())
            if f.shape[0]
            else f.new_empty((0, f.shape[1:].numel() * f.element_size()), dtype=torch.uint8)
        )
        for f in fields
    ]
    width = sum(r.shape[1] for r in raw)
    pad = -width % 16
    if pad:
        raw.append(raw[0].new_zeros((raw[0].shape[0], pad)))
    return torch.cat(raw, 1)


def _unpack(buffer, specs):
    """Strided views of ``buffer`` columns as ``(dtype, width)`` fields (no copy)."""
    out, start = [], 0
    for dtype, width in specs:
        size = width * torch.empty((), dtype=dtype).element_size()
        out.append(buffer[:, start : start + size].view(dtype))
        start += size
    return out


def _exchange(buffer, receives, sends, group, spare=False):
    """Variable all-to-all of packed rows; ``spare`` appends one zero row after them."""
    total = sum(receives)
    if group is None or group.size() == 1:
        if not spare:
            return buffer, None
        return torch.cat((buffer, buffer.new_zeros((1, buffer.shape[1])))), None
    out = buffer.new_empty((total + int(spare), buffer.shape[1]))
    if spare:
        out[total:].zero_()
    work = torch.distributed.all_to_all_single(
        out[:total], buffer, receives, sends, group=group, async_op=True
    )
    return out, work


def _counts(per_expert, group):
    world = 1 if group is None else group.size()
    if world == 1:
        return torch.stack((per_expert, per_expert)).cpu().view(2, 1, -1)
    received = torch.empty_like(per_expert)
    torch.distributed.all_to_all_single(received, per_expert, group=group)
    return torch.cat((per_expert, received)).cpu().view(2, world, -1)


def _expert_rows(codes, scale, grad, score, xa, splits, base, lora, limit):
    """Closed-form VJP of the replayed per-row expert graph, all local experts grouped.

    ``base`` is the stacked native (codes, scales) for FC1 and FC2; ``lora`` is None or
    (scaled gate/up B [L, 2I, r], down A [L, r, I], scaled down B [H, r]). Returns
    BF16 dX, FP32 dscore, BF16 d shrink and the raw low-rank gradients."""
    w1, w2 = base
    fc1 = _fc1_base(codes, scale, w1, splits)
    if lora is not None:
        b1, a2, b2 = lora
        fc1 = (fc1 + _gmm(xa, b1.transpose(-1, -2), splits)).bfloat16()
    hidden = _activation(fc1, score, limit)
    # Decode one operand at a time, only for this call; freed before the next.
    dact = _gmm(grad, _decoded(w2), splits).float()
    db1 = da2 = db2 = dxa = None
    if lora is not None:
        # Strided row views go straight to cuBLAS: no contiguous or transposed copies.
        dz = grad @ b2
        dact += _gmm(dz, a2, splits).float()
        da2 = grouped_mm(dz, hidden, splits, wgrad=True)
        db2 = (_gmm(hidden, a2.transpose(-1, -2), splits).T @ grad).T.float()
    dfc1, dscore = _swiglu_back(dact, fc1, score, limit)
    dfc1 = dfc1.bfloat16()
    if lora is not None:
        dxa = _gmm(dfc1, b1, splits)
        db1 = grouped_mm(dfc1, xa, splits, wgrad=True)
    dx = _gmm(dfc1, _decoded(w1), splits)
    return dx, dscore.flatten().float(), dxa, (db1, da2, db2)


def routed_adjoint(x, scores, ids, grad, fc1, fc2, lora, limit, group, overlap=None):
    """Routed experts' VJP for source rows ``x`` with top-k ``ids``/``scores``.

    ``lora`` is None or the SharedOuterExpertLoRA owner. Returns FP32 dX and dscore,
    the low-rank gradients ``(dA1, dB1 [L,2I,r], dA2 [L,r,I], dB2)`` (unscaled factor
    gradients, None without LoRA) and ``overlap()``'s result, which runs while the
    routed rows are in flight."""
    rows, k = ids.shape
    world = 1 if group is None else group.size()
    rank = 0 if group is None else group.rank()
    local, hidden = len(fc1), x.shape[1]
    flat = ids.flatten().long()
    key = torch.where(flat >= 0, flat, local * world)
    position = torch.argsort(key, stable=True)
    # One host sync per layer: per-peer send/receive rows and the local expert groups.
    per_expert = torch.bincount(key, minlength=local * world + 1)[: local * world]
    counts = _counts(per_expert, group)
    sends, receives = counts.sum(2).tolist()
    splits = counts[1].sum(0).tolist()
    position = position[: sum(sends)]
    token = position // k
    codes, scale = _quantize(x.detach())
    xa = None
    if lora is not None:
        xa = dense_mm(x.detach(), lora.fc1_lora_a.detach().T).bfloat16()
    # Pack once per token (with a slot-metadata placeholder), gather rows once.
    per_token = [codes.view(torch.uint8), scale, grad.bfloat16()] + ([xa] if xa is not None else [])
    per_token.append(scale.new_empty((x.shape[0], 2)))
    rank_r = 0 if lora is None else lora.rank
    specs = [
        (torch.uint8, hidden),
        (torch.float32, scale.shape[1]),
        (torch.bfloat16, hidden),
        *([(torch.bfloat16, rank_r)] if lora is not None else []),
        (torch.float32, 2),
    ]
    buffer = _pack(per_token).index_select(0, token)
    _unpack(buffer, specs)[-1].copy_(
        torch.stack((scores.detach().flatten()[position].float(), flat[position].float()), 1)
    )
    received, work = _exchange(buffer, receives, sends, group)
    shared = overlap() if overlap is not None else None
    if work is not None:
        work.wait()
    del buffer
    # Arrivals are source-major; a stable sort by local expert makes them expert-major.
    r_meta = _unpack(received, specs)[-1]
    order = torch.argsort(r_meta[:, 1].long() - rank * local, stable=True)
    received = received.index_select(0, order)
    fields = _unpack(received, specs)
    r_codes, r_scale, r_grad = fields[:3]
    # TE grouped wgrad reads data_ptr + shape: low-rank operands must be dense.
    r_xa = fields[3].contiguous() if lora is not None else None
    r_score = fields[-1][:, 0]
    base = (packed_weights(fc1), packed_weights(fc2))
    factors = None
    if lora is not None:
        dtype = lora.fc1_lora_b.weight0.dtype
        b1 = (torch.stack(lora.per_expert(lora.fc1_lora_b)).detach() * lora.scale).to(dtype)
        a2 = torch.stack(lora.per_expert(lora.fc2_lora_a)).detach()
        b2 = (lora.fc2_lora_b.detach() * lora.scale).to(lora.fc2_lora_b.dtype)
        factors = b1, a2, b2
    if r_grad.shape[0]:
        dx, dscore, dxa, (db1, da2, db2) = _expert_rows(
            r_codes.view(torch.float8_e4m3fn),
            r_scale,
            r_grad,
            r_score.contiguous(),
            r_xa,
            splits,
            base,
            factors,
            limit,
        )
    else:
        dx = r_grad.new_empty((0, hidden))
        dscore = r_score.new_empty((0,))
        dxa = None if lora is None else r_grad.new_empty((0, rank_r))
        db1 = da2 = db2 = None
        if lora is not None:
            db1 = torch.zeros(b1.shape, dtype=torch.float32, device=x.device)
            da2 = torch.zeros(a2.shape, dtype=torch.float32, device=x.device)
            db2 = torch.zeros(b2.shape, dtype=torch.float32, device=x.device)
    del received, fields, r_codes, r_scale, r_grad, r_xa
    # Scatter expert-major results straight into arrival order inside the send buffer.
    back_specs = [
        (torch.bfloat16, hidden),
        (torch.float32, 1),
        *([(torch.bfloat16, rank_r)] if lora is not None else []),
    ]
    width = sum(w * torch.empty((), dtype=d).element_size() for d, w in back_specs)
    back = dx.new_empty((dx.shape[0], width + (-width % 16)), dtype=torch.uint8)
    for view, value in zip(
        _unpack(back, back_specs), (dx, dscore[:, None], *([dxa] if lora is not None else []))
    ):
        view.index_copy_(0, order, value.to(view.dtype))
    back, work = _exchange(back, sends, receives, group, spare=True)
    if work is not None:
        work.wait()
    b_dx, b_ds, *b_xa = _unpack(back, back_specs)
    # Slots not routed anywhere read a spare zero row; the k-sum is in slot order.
    spare = position.shape[0]
    source = torch.full((rows * k,), spare, dtype=torch.long, device=x.device)
    source[position] = torch.arange(spare, device=x.device)

    def gather(t):
        return t.index_select(0, source).view(rows, k, t.shape[1])

    dx_x = gather(b_dx).sum(1, dtype=torch.float32)
    dscores = gather(b_ds).view(rows, k)
    grads = None
    if lora is not None:
        dxa_sum = gather(b_xa[0]).sum(1, dtype=torch.float32)
        dx_x += dense_mm(dxa_sum, lora.fc1_lora_a.detach())
        da1 = dense_mm(dxa_sum.T, x.detach())
        grads = (da1, db1 * lora.scale, da2, db2 * lora.scale)
    return dx_x, dscores, grads, shared
