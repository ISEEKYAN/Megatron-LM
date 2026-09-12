# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paired PP payload lifetime; the C4 protocol owns scheduling and layer binding."""

from dataclasses import dataclass, fields

import torch
from megatron.lite.primitive.parallel.tensor_payload import PipelineLedger, PipelineTag


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
