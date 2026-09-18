# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paired pipeline payload lifetime; callers own scheduling and layer binding."""

from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class PairedPayload:
    h: torch.Tensor
    p: torch.Tensor
    ced_h: torch.Tensor | None = None
    ced_p: torch.Tensor | None = None
    kv: torch.Tensor | None = None
    kv_values: torch.Tensor | None = None
    kv_scale: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    index_scale: torch.Tensor | None = None
    topk: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    latent: torch.Tensor | None = None
    candidates: torch.Tensor | None = None

    def __post_init__(self):
        if not isinstance(self.h, torch.Tensor) or not isinstance(self.p, torch.Tensor):
            raise ValueError('Require a current HC pair')
        if (self.ced_h is None) != (self.ced_p is None):
            raise ValueError('CED h/p must travel as a pair')
        for h, p in ((self.h, self.p), (self.ced_h, self.ced_p)):
            if h is not None and (
                p is None
                or h.ndim != 4
                or p.shape != h.shape[:-1]
                or not h.is_floating_point()
                or not p.is_floating_point()
            ):
                raise ValueError(
                    'HC pair requires h[batch,sequence,hc,hidden] and p[batch,sequence,hc]'
                )
        for name in (
            'kv_values',
            'kv_scale',
            'index_k',
            'index_scale',
            'topk',
            'positions',
            'candidates',
        ):
            value = getattr(self, name)
            if value is not None and value.requires_grad:
                raise ValueError(
                    'Published bytes, integer positions and frozen indexers have no gradient'
                )
        for name in ('topk', 'positions'):
            value = getattr(self, name)
            if value is not None and value.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    'Pipeline selection and position metadata must be integer tensors'
                )

    def tensors(self):
        return tuple(getattr(self, name) for name in PAYLOAD_FIELDS)

    def differentiable(self):
        return {
            name: value
            for name, value in zip(PAYLOAD_FIELDS, self.tensors())
            if value is not None and value.requires_grad
        }

    @classmethod
    def from_tensors(cls, tensors):
        if len(tensors) != len(PAYLOAD_FIELDS):
            raise ValueError('Pipeline payload field count differs')
        return cls(**dict(zip(PAYLOAD_FIELDS, tensors)))


PAYLOAD_FIELDS = tuple(field.name for field in fields(PairedPayload))


def packed_paired_forward(
    sequence_forward,
    hidden,
    pre_mix,
    cu_seqlens,
    *,
    input_ids=None,
    image_mask=None,
    cp_context=None,
):
    """Run a pure sequence callable over each logical sample, preserving its graph.

    The callable returns (hidden, next_pre_mix), creates fresh sequence state
    per invocation, and keeps bias/statistic publication outside forward. RNG is
    consumed in sequence order, exactly as for independent calls. This is a
    correctness path; CP transport and document ownership belong to the primitive.
    """
    from contextlib import nullcontext

    from megatron.lite.primitive.utils.packed_seq import packed_sequence_ranges

    from .router_replay import PackedRouterReplay

    if hidden.ndim != 4 or hidden.shape[0] != 1 or pre_mix.shape != hidden.shape[:-1]:
        raise ValueError("Expected packed hidden [1,T,HC,D] and pre_mix [1,T,HC]")
    for tensor in (input_ids, image_mask):
        if tensor is not None and tensor.shape != hidden.shape[:2]:
            raise ValueError("Token inputs must match packed [1,T] dimensions")
    outputs, mixes = [], []
    replay = PackedRouterReplay(hidden.shape[1]) if cp_context is None else None
    total = hidden.shape[1] if cp_context is None else cp_context.total_length
    offset = 0
    for begin, end in packed_sequence_ranges(cu_seqlens, total):
        kwargs = {}
        if cp_context is not None:
            document = cp_context.document(begin, end)
            kwargs['cp_context'] = document
            begin, end = offset, offset + document.local_length
            offset = end
        if input_ids is not None:
            kwargs['input_ids'] = input_ids[:, begin:end]
        if image_mask is not None:
            kwargs['image_mask'] = image_mask[:, begin:end]
        with replay.sequence(begin, end) if replay is not None else nullcontext():
            h, p = sequence_forward(
                hidden[:, begin:end], pre_mix[:, begin:end], **kwargs
            )
        outputs.append(h)
        mixes.append(p)
    if replay is not None:
        replay.finish()
    return torch.cat(outputs, dim=1), torch.cat(mixes, dim=1)
