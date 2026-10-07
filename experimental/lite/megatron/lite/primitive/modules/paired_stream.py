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


def pack_deployment(hidden, pre, pending):
    """Lossless FP32 PP carrier for the shifted deployment post/pre boundary.

    Materialized hidden is retained for Engram injection. The four pending post
    operands remain independent graph edges for the next layer's reference VJP.
    """
    shape = hidden.shape[:2]
    copies, width = hidden.shape[-2:]
    active = pending is not None
    if not active:
        pending = (
            hidden.new_zeros(*shape, width),
            torch.zeros_like(hidden),
            pre.new_zeros(*shape, copies),
            pre.new_zeros(*shape, copies, copies),
        )
    return torch.cat(
        [pack_pair(hidden, pre)]
        + [value.float().flatten(2) for value in pending]
        + [pre.new_full((*shape, 1), int(active))],
        dim=-1,
    )


def unpack_deployment(carrier, input_shape, copies, width, dtype):
    lengths = (copies * (width + 1), width, copies * width, copies, copies * copies, 1)
    if (
        carrier is None
        or carrier.dtype != torch.float32
        or carrier.shape != (*input_shape, sum(lengths))
    ):
        raise ValueError('V4.1_PP_DEPLOYMENT_INPUT: invalid shifted post/pre carrier')
    pair, output, residual, post, comb, flag = carrier.split(lengths, dim=-1)
    if not ((flag == 0).all() or (flag == 1).all()):
        raise ValueError('V4.1_PP_DEPLOYMENT_INPUT: inconsistent pending post flag')
    hidden, pre = unpack_pair(
        pair, input_shape, copies, width, dtype, 'Invalid paired carrier'
    )
    pending = None
    if bool(flag.flatten()[0]):
        pending = (
            output.to(dtype).contiguous(),
            residual.reshape(*input_shape, copies, width).to(dtype).contiguous(),
            post.contiguous(),
            comb.reshape(*input_shape, copies, copies).contiguous(),
        )
    return hidden, pre, pending
