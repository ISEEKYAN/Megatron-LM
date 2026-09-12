# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from megatron.lite.primitive.parallel.cp import (
    _all_gather_cp,
    _gather_contiguous_tail,
    contiguous_slice_for_cp,
)
from megatron.lite.primitive.parallel.thd import thd_pack_meta
from torch.nn import functional as F


@dataclass(frozen=True)
class Qwen3_8_FlashNextCPContext:
    group: object
    rank: int
    size: int
    global_input_ids: torch.Tensor
    global_padding_mask: torch.Tensor
    local_sequence_start: int
    local_sequence_length: int
    global_cu_seqlens: torch.Tensor | None = None

    def __post_init__(self):
        if (
            self.size < 1
            or not 0 <= self.rank < self.size
            or self.local_sequence_length < 1
            or self.local_sequence_start != self.rank * self.local_sequence_length
        ):
            raise ValueError('CP_CONTIGUOUS_INTERVAL')
        ids, mask = self.global_input_ids, self.global_padding_mask
        if (
            ids.ndim != 2
            or ids.dtype not in (torch.int32, torch.int64)
            or ids.shape[1] != self.size * self.local_sequence_length
            or mask.shape != ids.shape
            or mask.dtype != torch.bool
            or mask.device != ids.device
        ):
            raise ValueError('CP_GLOBAL_METADATA')
        cu = self.global_cu_seqlens
        if cu is not None and (
            ids.shape[0] != 1
            or cu.ndim != 1
            or cu.numel() < 2
            or int(cu[0]) != 0
            or int(cu[-1]) > ids.shape[1]
            or not bool((cu.diff() > 0).all())
        ):
            raise ValueError('CP_PACKED_BOUNDARIES')

    @property
    def global_sequence_length(self):
        return self.global_input_ids.shape[1]

    @property
    def local_sequence_end(self):
        return self.local_sequence_start + self.local_sequence_length

    @property
    def global_sequence_lengths(self):
        return (~self.global_padding_mask).sum(-1)


def qwen3_8_flash_next_cp_all_gather(
    tensor, context, *, sequence_dim=1, differentiable=True
):
    if context.size == 1:
        return tensor
    if context.group is None:
        raise RuntimeError('CP_GROUP_REQUIRED')
    with nullcontext() if differentiable else torch.no_grad():
        return torch.cat(_all_gather_cp(tensor, context.group), dim=sequence_dim)


def qwen3_8_flash_next_cp_left_halo(tensor, context, *, history):
    if (
        tensor.ndim != 3
        or tensor.shape[1] != context.local_sequence_length
        or history < 0
    ):
        raise ValueError('CP_HALO_SHAPE_HISTORY')
    if history == 0:
        return tensor[:, :0]
    if context.size == 1:
        return tensor.new_zeros((tensor.shape[0], history, tensor.shape[2]))
    parts = _gather_contiguous_tail(
        tensor,
        tail_len=min(history, tensor.shape[1]),
        cp_size=context.size,
        cp_group=context.group,
        seq_dim=1,
    )
    previous = (
        torch.cat(parts[: context.rank], 1)[:, -history:]
        if context.rank
        else tensor[:, :0]
    )
    # Every rank retains the collective in backward, including rank zero.
    anchor = sum(part[:, :0].sum() for part in parts)
    return F.pad(previous, (0, 0, history - previous.shape[1], 0)) + anchor


def packed_boundaries_from_seq_lens(seq_lens, *, total_tokens=None, sentinel=-1000):
    widths = seq_lens.flatten().long()
    widths = widths[widths != sentinel]
    if not widths.numel() or bool((widths <= 0).any()):
        raise ValueError('CP_PACKED_LENGTHS')
    cu = F.pad(widths.cumsum(0), (1, 0))
    if total_tokens is not None:
        if int(cu[-1]) > total_tokens:
            raise ValueError('CP_PACKED_OVERFLOW')
        if int(cu[-1]) < total_tokens:
            cu = torch.cat((cu, cu.new_tensor([total_tokens])))
    return cu


def shard_batch_for_qwen3_8_flash_next_cp(
    cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id=0, pad_multiple=4
):
    if tp_mesh is not None and tp_mesh.size() > 1:
        raise ValueError('CP_TP_UNSUPPORTED')
    fills = {
        'input_ids': padding_token_id,
        'labels': -100,
        'position_ids': 0,
        'attention_mask': 0,
        'padding_mask': True,
        'loss_mask': 0,
    }
    unknown = batch.keys() - fills.keys() - {'seq_lens', 'cu_seqlens'}
    if unknown:
        raise ValueError(f'CP_UNKNOWN_BATCH_KEYS: {sorted(unknown)}')
    size, rank = cp_mesh.size(), cp_mesh.get_local_rank()
    ids = batch['input_ids']
    length = ids.shape[1]
    if pad_multiple < 1:
        raise ValueError('CP_PAD_MULTIPLE')
    # Use one physical packed row: per-document padding would change global CP intervals.
    meta = thd_pack_meta(
        torch.tensor([length], device=ids.device),
        tp_size=pad_multiple,
        cp_size=size,
        contiguous=True,
    )
    total = int(meta.cu_seqlens_padded[-1])
    local = total // size
    cu = batch.get('cu_seqlens')
    if cu is None and batch.get('seq_lens') is not None:
        cu = packed_boundaries_from_seq_lens(batch['seq_lens'], total_tokens=length)
    mask = batch.get('padding_mask')
    if mask is None:
        mask = ~batch.get(
            'attention_mask', torch.ones_like(ids, dtype=torch.bool)
        ).bool()
    global_ids = F.pad(ids, (0, total - length), value=padding_token_id)
    global_mask = F.pad(mask, (0, total - length), value=True)
    context = Qwen3_8_FlashNextCPContext(
        cp_mesh.get_group() if size > 1 else None,
        rank,
        size,
        global_ids,
        global_mask,
        rank * local,
        local,
        cu,
    )
    output = dict(batch)
    if loss_mask is not None:
        output['loss_mask'] = loss_mask
    for key, value in list(output.items()):
        if key in fills and value is not None:
            output[key] = contiguous_slice_for_cp(
                F.pad(value, (0, total - length), value=fills[key]), rank, size
            )
    output['_qwen3_8_flash_next_cp_context'] = context
    return (
        nullcontext,
        output,
        SimpleNamespace(original_seq_len=length, padded_seq_len=total),
    )
