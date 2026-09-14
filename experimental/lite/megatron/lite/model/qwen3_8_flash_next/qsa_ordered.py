# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Continue native indexed K/V adjoints in contiguous global query order."""

import torch
import torch.distributed as dist


def document_owners(context, start, end):
    width = context.local_sequence_length
    return tuple(range(start // width, (end - 1) // width + 1))


def accumulate_selected(state, grad_k, grad_v, batch, indices, *, document_start, rank):
    # Keep native per-occurrence scalar-dtype writebacks, including zero entries.
    # Summing independent BF16 partials cannot reconstruct this prefix state.
    for slot, grad in enumerate((grad_k, grad_v)):
        if grad is not None:
            torch.ops.aten._index_put_impl_(
                state[slot], [batch, indices], grad, True, True
            )


class _OrderedCPIndex(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, local_k, local_v, doc_k, doc_v, batch, indices, context, start, end
    ):
        ctx.save_for_backward(batch, indices)
        ctx.context, ctx.start, ctx.end = context, start, end
        ctx.local_shape, ctx.dtype, ctx.device = (
            local_k.shape,
            local_k.dtype,
            local_k.device,
        )
        return doc_k[batch, indices], doc_v[batch, indices]

    @staticmethod
    def backward(ctx, grad_k, grad_v):
        batch, indices = ctx.saved_tensors
        context, start, end = ctx.context, ctx.start, ctx.end
        owners = document_owners(context, start, end)
        position = owners.index(context.rank)
        shape = (2, ctx.local_shape[0], end - start, *ctx.local_shape[2:])
        state = torch.zeros(shape, dtype=ctx.dtype, device=ctx.device)

        def peer(owner):
            if context.group is None:
                raise RuntimeError('CP_GROUP_REQUIRED')
            return dist.get_global_rank(context.group, owner)

        if position:
            dist.recv(state, src=peer(owners[position - 1]), group=context.group)
        accumulate_selected(
            state,
            grad_k,
            grad_v,
            batch,
            indices,
            document_start=start,
            rank=context.rank,
        )
        local_start = max(start, context.local_sequence_start)
        local_end = min(end, context.local_sequence_end)
        if position + 1 < len(owners):
            dist.send(state, dst=peer(owners[position + 1]), group=context.group)
            completed = torch.empty(
                (2, ctx.local_shape[0], local_end - local_start, *ctx.local_shape[2:]),
                dtype=ctx.dtype,
                device=ctx.device,
            )
            dist.recv(completed, src=peer(owners[-1]), group=context.group)
        else:
            for owner in owners[:-1]:
                left = max(start, owner * context.local_sequence_length)
                right = min(end, (owner + 1) * context.local_sequence_length)
                dist.send(
                    state[:, :, left - start : right - start].contiguous(),
                    dst=peer(owner),
                    group=context.group,
                )
            completed = state[:, :, local_start - start : local_end - start]
        result = torch.zeros((2, *ctx.local_shape), dtype=ctx.dtype, device=ctx.device)
        result[
            :,
            :,
            local_start
            - context.local_sequence_start : local_end
            - context.local_sequence_start,
        ].copy_(completed)
        return result[0], result[1], None, None, None, None, None, None, None


ordered_select_kv = _OrderedCPIndex.apply
