# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Whole-slice contract of a packed THD batch under contiguous CP.

One object fixes, for this CP rank, how the packed documents map onto its rows,
so a layer can run once over the slice and mark document boundaries inside its
kernels (cu_seqlens), as DS4's packed attention does, instead of being called
once per document. Positions and compressor groups come from Core's packed
CSA helpers; this module only owns the layout contract.

Layout contract (cp-a interface v1.2):
* Documents are packed back to back, unpadded; ``boundaries`` is the global
  ``cu_seqlens`` ``0 = b0 < b1 < ... < bD = T``. Empty documents are invalid.
* CP splits the whole pack contiguously: ``width = ceil(T / cp)`` rows per
  rank, rank ``r`` owning global rows ``[r * width, (r + 1) * width)``. Only
  the pack's tail is padded, to ``width * cp`` rows; the pad rows extend the
  last document, follow every real token and carry no labels.
* A rank may own no rows of a document; no per-document call exists, so this
  needs no special case.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
from megatron.lite.primitive.parallel.cp import contiguous_slice_for_cp

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    packed_layout,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils import (
    thd_cp_position_ids,
)


@dataclass(frozen=True)
class PackedCPGeometry:
    boundaries: tuple[int, ...]
    cp_rank: int = 0
    cp_size: int = 1
    cp_group: Any = None
    device: Any = 'cpu'

    def __post_init__(self):
        b = tuple(int(x) for x in self.boundaries)
        object.__setattr__(self, 'boundaries', b)
        object.__setattr__(self, 'device', torch.device(self.device))
        if len(b) < 2 or b[0] != 0 or any(e <= s for s, e in zip(b, b[1:])):
            raise ValueError('cu_seqlens must start at 0 and hold non-empty documents')
        if not (self.cp_size >= 1 and 0 <= self.cp_rank < self.cp_size):
            raise ValueError('Invalid CP rank or size')
        if self.cp_size > 1 and self.cp_group is None and self.device.type != 'cpu':
            raise ValueError('CP geometry needs its process group on device')

    @classmethod
    def from_cu_seqlens(cls, cu_seqlens, cp_rank=0, cp_size=1, cp_group=None):
        device = cu_seqlens.device if torch.is_tensor(cu_seqlens) else 'cpu'
        bounds = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else cu_seqlens
        return cls(tuple(bounds), cp_rank, cp_size, cp_group, device)

    @property
    def num_documents(self) -> int:
        return len(self.boundaries) - 1

    @property
    def total(self) -> int:
        return self.boundaries[-1]

    @property
    def width(self) -> int:
        return -(-self.total // self.cp_size)

    @property
    def padded_total(self) -> int:
        return self.width * self.cp_size

    @property
    def global_start(self) -> int:
        return self.cp_rank * self.width

    @property
    def padded_boundaries(self) -> tuple[int, ...]:
        return (*self.boundaries[:-1], self.padded_total)

    @cached_property
    def cu_seqlens(self) -> torch.Tensor:
        return torch.tensor(self.boundaries, dtype=torch.int32, device=self.device)

    @cached_property
    def cu_seqlens_padded(self) -> torch.Tensor:
        return torch.tensor(
            self.padded_boundaries, dtype=torch.int32, device=self.device
        )

    @cached_property
    def positions(self) -> torch.Tensor:
        """Position of each local row within its document (pad rows continue)."""
        return thd_cp_position_ids(
            self.cu_seqlens_padded, self.global_start, self.width
        ).long()

    @cached_property
    def valid(self) -> torch.Tensor:
        rows = torch.arange(self.width, device=self.device) + self.global_start
        return rows < self.total

    def owned(self, doc: int) -> tuple[int, int]:
        """Global ``[start, end)`` of document ``doc`` on this rank; may be empty."""
        begin, end = self.boundaries[doc], self.boundaries[doc + 1]
        start = min(max(self.global_start, begin), end)
        stop = max(min(self.global_start + self.width, end), start)
        return start, stop

    def lookback(self, rows: int) -> tuple[int, int]:
        """Global rows ``[start, global_start)`` just left of this rank."""
        return max(self.global_start - int(rows), 0), self.global_start

    def slice(self, tensor: torch.Tensor, *, seq_dim: int = 1) -> torch.Tensor:
        """This rank's rows of a full-pack tensor (pad rows are zeros)."""
        if tensor.shape[seq_dim] != self.total:
            raise ValueError('Geometry slice requires the whole pack')
        if self.cp_size == 1:
            return tensor
        shape = list(tensor.shape)
        shape[seq_dim] = self.padded_total - self.total
        padded = torch.cat((tensor, tensor.new_zeros(shape)), dim=seq_dim)
        return contiguous_slice_for_cp(padded, self.cp_rank, self.cp_size, seq_dim)

    def compressor_input(self, hidden, boundary, ratio, *, halo, capacity):
        """Core's packed compressor groups for this rank over the real tokens.

        ``hidden`` is ``[width, ...]``; ``boundary`` holds the ``lookback(halo)``
        rows. Returns Core's ``(gathered, group_ids, positions, cu_seqlens_
        compressed, row_map)``; ``positions`` are group starts, the RoPE
        positions the DS4.1 compressor uses.
        """
        return packed_layout.compact_compressor_input(
            hidden,
            boundary,
            self.cu_seqlens,
            self.global_start,
            ratio,
            halo,
            capacity,
            self.cp_size,
        )
