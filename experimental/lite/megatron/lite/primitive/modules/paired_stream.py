# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed input validation and paired residual stream transport."""
import torch


def validate_input_ids(input_ids):
    if input_ids.ndim != 2 or input_ids.dtype != torch.int64 or not input_ids.shape[1]:
        raise ValueError('Expected nonempty int64 input_ids [B,S]')


def validate_packed_input(input_ids, cu_seqlens, cp_context, cp_size, row_group):
    if row_group is not None:
        # Each packed document visits both lookup collectives. Reject a
        # mismatched schedule before any rank enters the first lookup.
        count = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
        counts = torch.tensor([count, -count], device=input_ids.device)
        torch.distributed.all_reduce(
            counts, op=torch.distributed.ReduceOp.MAX, group=row_group
        )
        if counts[0] != -counts[1]:
            raise ValueError(
                'Sharded row lookup requires equal packed document counts across ranks'
            )
    if cp_context is None:
        validate_input_ids(input_ids)
    elif (
        input_ids.shape != (1, cp_context.local_length)
        or input_ids.dtype != torch.int64
    ):
        raise ValueError('CP input must match the local contiguous interval')
    if cp_context is not None and cu_seqlens is None:
        raise ValueError('CP requires explicit packed document boundaries')
    if cp_size > 1 and cp_context is None:
        raise ValueError('CP model requires protocol-owned contiguous input metadata')


def unpack_pair(carrier, input_shape, copies, width, dtype, message):
    if (
        carrier is None
        or carrier.dtype != torch.float32
        or carrier.shape != (*input_shape, copies * (width + 1))
    ):
        raise ValueError(message)
    pair = carrier.reshape(*input_shape, copies, width + 1)
    return pair[..., :-1].to(dtype).contiguous(), pair[..., -1].contiguous()


def pack_pair(hidden, pre):
    return torch.cat((hidden.float(), pre.unsqueeze(-1)), dim=-1).flatten(2)


def pack_state(hidden, pre, pending):
    """Carry materialized Engram input and deferred post operands in one graph.

    The FP32 wire is lossless for BF16 residual values. Post/pre coefficients
    remain FP32; no integer bit views or detached side messages cross PP.
    """
    if hidden.ndim != 4 or pre.shape != hidden.shape[:-1]:
        raise ValueError('PP state requires hidden [B,S,C,H] and pre [B,S,C]')
    if (
        hidden.dtype not in (torch.bfloat16, torch.float32)
        or pre.dtype != torch.float32
    ):
        raise ValueError('PP state requires BF16/FP32 hidden and FP32 pre')
    if pre.device != hidden.device:
        raise ValueError('PP state fields must share one device')
    pair = pack_pair(hidden, pre)
    if pending is None:
        return pair
    if not isinstance(pending, tuple) or len(pending) != 4:
        raise ValueError('PP pending requires (y,residual,post,comb)')
    y, residual, post, comb = pending
    batch, length, copies, width = hidden.shape
    shapes = (
        (batch, length, width),
        hidden.shape,
        pre.shape,
        (batch, length, copies, copies),
    )
    dtypes = (hidden.dtype, hidden.dtype, torch.float32, torch.float32)
    if any(
        value.shape != shape or value.dtype != dtype or value.device != hidden.device
        for value, shape, dtype in zip(pending, shapes, dtypes)
    ):
        raise ValueError('PP pending shape/dtype/device disagrees with hidden/pre')
    fields = [pair] + [value.float().flatten(2) for value in pending]
    return torch.cat(fields, dim=-1)


def unpack_state(carrier, input_shape, copies, width, dtype, *, require_pending=False):
    """Inverse of pack_state; reject legacy payloads for native fused stages."""
    base = copies * (width + 1)
    sizes = (width, copies * width, copies, copies * copies)
    if (
        carrier is None
        or carrier.dtype != torch.float32
        or carrier.shape[:-1] != tuple(input_shape)
        or carrier.shape[-1] not in (base, base + sum(sizes))
        or dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError('V4.1_PP_STATE: invalid packed hidden/pre/pending carrier')
    present = carrier.shape[-1] != base
    if require_pending and not present:
        raise ValueError(
            'V4.1_PP_STATE: native deployment requires deferred post operands'
        )
    hidden, pre = unpack_pair(
        carrier[..., :base],
        input_shape,
        copies,
        width,
        dtype,
        'V4.1_PP_STATE: invalid paired prefix',
    )
    if not present:
        return hidden, pre, None
    shapes = (
        (*input_shape, width),
        (*input_shape, copies, width),
        (*input_shape, copies),
        (*input_shape, copies, copies),
    )
    dtypes = (dtype, dtype, torch.float32, torch.float32)
    fields = carrier[..., base:].split(sizes, dim=-1)
    pending = tuple(
        value.reshape(shape).to(target).contiguous()
        for value, shape, target in zip(fields, shapes, dtypes)
    )
    return hidden, pre, pending
