# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from megatron.lite.primitive.parallel.cp import (
    _all_gather_cp,
    _gather_contiguous_tail,
    contiguous_slice_for_cp,
)


@dataclass(frozen=True)
class ContiguousCPSequence:
    """A document's intersection with one globally contiguous packed shard.

    Only transport is padded. Queries, hashes and outputs contain true tokens;
    gathered KV/compressor inputs contain one complete, unpadded document.
    All ranks visit every document, including empty local intersections.
    """

    total_length: int
    rank: int
    size: int
    group: Any = None
    begin: int = 0
    end: int | None = None

    def __post_init__(self):
        if self.end is None:
            object.__setattr__(self, 'end', self.total_length)
        if not (
            self.size > 0
            and 0 <= self.rank < self.size
            and 0 <= self.begin < self.end <= self.total_length
        ):
            raise ValueError('Invalid contiguous CP sequence ownership')

    @property
    def width(self):
        return (self.total_length + self.size - 1) // self.size

    @property
    def length(self):
        return self.end - self.begin

    @property
    def start(self):
        return min(max(self.rank * self.width - self.begin, 0), self.length)

    @property
    def local_length(self):
        stop = min(max((self.rank + 1) * self.width - self.begin, 0), self.length)
        return stop - self.start

    def document(self, begin, end):
        return replace(self, begin=begin, end=end)

    def slice(self, tensor, *, seq_dim=1):
        if tensor.shape[seq_dim] != self.length:
            raise ValueError('CP slice requires the full document')
        if self.begin == 0 and self.end == self.total_length:
            shape = list(tensor.shape)
            shape[seq_dim] = self.width * self.size - self.length
            padded = torch.cat((tensor, tensor.new_zeros(shape)), dim=seq_dim)
            return contiguous_slice_for_cp(
                padded, self.rank, self.size, seq_dim
            ).narrow(seq_dim, 0, self.local_length)
        return tensor.narrow(seq_dim, self.start, self.local_length).contiguous()

    def gather(self, tensor, *, seq_dim=1):
        if tensor.shape[seq_dim] != self.local_length:
            raise ValueError('CP gather requires the local document intersection')
        if self.size == 1:
            return tensor
        if self.group is None:
            raise ValueError('CP gather requires an explicit group')
        shape = list(tensor.shape)
        shape[seq_dim] = self.width - self.local_length
        padded = torch.cat((tensor, tensor.new_zeros(shape)), dim=seq_dim)
        parts = _all_gather_cp(padded, self.group)
        return torch.cat(
            [
                part.narrow(seq_dim, 0, replace(self, rank=rank).local_length)
                for rank, part in enumerate(parts)
            ],
            dim=seq_dim,
        )


def iter_cp_sources(tensor, position_ids, *, cp_rank, cp_size, cp_group):
    if cp_size <= 1:
        yield cp_rank, tensor, position_ids
        return
    if cp_group is None:
        raise RuntimeError(
            "CP source iteration requires a context-parallel process group."
        )
    tensor_parts = _all_gather_cp(tensor, cp_group)
    position_parts = _all_gather_cp(position_ids.to(dtype=torch.long), cp_group)
    for rank, (source_tensor, source_positions) in enumerate(
        zip(tensor_parts, position_parts)
    ):
        yield rank, source_tensor, source_positions


def compress_contiguous_chunks_for_cp(
    compressor,
    tensor,
    *,
    position_ids,
    cp_rank,
    cp_size,
    cp_group,
    compress_kwargs: dict[str, Any] | None = None,
    seq_dim=1,
    compressed_seq_dim=2,
):
    kwargs = compress_kwargs or {}
    compress_ratio = int(compressor.compress_ratio)
    if cp_size <= 1:
        compressed = compressor(tensor, position_ids=position_ids, **kwargs)
        if compressed is None:
            return None
        cutoff = (tensor.size(seq_dim) // compress_ratio) * compress_ratio
        comp_pos = position_ids[:, :cutoff:compress_ratio]
        return compressed, comp_pos

    drop_prefix = 0
    tail_parts = None
    if compressor.overlap:
        tail_parts = _gather_contiguous_tail(
            tensor,
            tail_len=compress_ratio,
            cp_size=cp_size,
            cp_group=cp_group,
            seq_dim=seq_dim,
        )
        zero_tail = tensor.new_zeros(())
        for tail in tail_parts:
            zero_tail = zero_tail + tail.to(dtype=tensor.dtype).sum() * 0.0
        tensor = tensor + zero_tail
    if tail_parts is not None and cp_rank > 0:
        prefix = tail_parts[cp_rank - 1].to(device=tensor.device, dtype=tensor.dtype)
        prefix_pos = position_ids[:, :compress_ratio] - compress_ratio
        tensor = torch.cat([prefix, tensor], dim=seq_dim)
        position_ids = torch.cat([prefix_pos, position_ids], dim=1)
        drop_prefix = 1

    compressed = compressor(tensor, position_ids=position_ids, **kwargs)
    if compressed is None:
        return None
    cutoff = (tensor.size(seq_dim) // compress_ratio) * compress_ratio
    comp_pos = position_ids[:, :cutoff:compress_ratio]
    if drop_prefix:
        compressed = compressed.narrow(
            compressed_seq_dim,
            drop_prefix,
            compressed.size(compressed_seq_dim) - drop_prefix,
        )
        comp_pos = comp_pos[:, drop_prefix:]
    if compressed.size(compressed_seq_dim) == 0:
        return None
    return compressed, comp_pos
