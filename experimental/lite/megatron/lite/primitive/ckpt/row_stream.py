# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Borrowed row blocks: consume before advancing; never collect them in a list."""
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RowChunk:
    name: str
    offset: int
    total_rows: int
    weight: torch.Tensor
    scale: torch.Tensor | None = None


@torch.no_grad()
def stream_rows(
    name,
    weight,
    scale=None,
    *,
    boundaries=None,
    group=None,
    buffer_max_size_bytes=5 * 1024**3,
    quantize=False
):
    """Broadcast each owner's rows in global order through one byte buffer.

    All group members must drain the iterator. W/S share one collective and
    scratch allocation, including byte-valued FP8/E8M0 storage. Payload views
    expire on the next iteration. Resident inputs and consumer-owned output
    storage are excluded from the scratch limit; no full table is allocated.
    """
    if type(quantize) is not bool:
        raise TypeError('quantize must be bool')
    if type(buffer_max_size_bytes) is not int or buffer_max_size_bytes <= 0:
        raise ValueError('Export buffer must hold at least one row')
    rank = dist.get_rank(group) if group is not None else 0
    size = dist.get_world_size(group) if group is not None else 1
    boundaries = tuple(boundaries) if boundaries is not None else (0, weight.shape[0])
    if (
        len(boundaries) != size + 1
        or boundaries[0] != 0
        or any(a > b for a, b in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError('Invalid row ownership boundaries')
    if weight.ndim != 2 or not weight.is_contiguous() or weight.shape[1] == 0:
        raise ValueError('Expected contiguous 2D rows')
    if weight.shape[0] != boundaries[rank + 1] - boundaries[rank]:
        raise ValueError('Local rows differ from ownership boundaries')
    if scale is not None and (
        scale.ndim != 2
        or scale.shape[0] != weight.shape[0]
        or not scale.is_contiguous()
        or scale.device != weight.device
    ):
        raise ValueError('Expected colocated contiguous scale rows')
    output_dtype = weight.dtype
    scale_dtype = None if scale is None else scale.dtype
    scale_width = 0 if scale is None else scale.shape[1]
    if quantize:
        if (
            scale is not None
            or weight.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            or weight.shape[1] % 32
        ):
            raise ValueError(
                'Row quantization requires plain weights with block32 width'
            )
        output_dtype, scale_dtype = torch.float8_e4m3fn, torch.float8_e8m0fnu
        scale_width = weight.shape[1] // 32
    weight_bytes = weight.shape[1] * (1 if quantize else weight.element_size())
    scale_bytes = scale_width * (
        1 if quantize else 0 if scale is None else scale.element_size()
    )
    # CUDA's caching allocator rounds small allocations to 512-byte multiples.
    capacity = buffer_max_size_bytes // 512 * 512
    # Row-wise block-FP8 uses fewer than sixteen FP32 planes, including
    # source conversion, abs/reduction, exponent, expansion and rounding.
    # Leave allocator rounding room for each live workspace allocation too.
    working_row_bytes = weight.shape[1] * 64 if quantize else weight_bytes + scale_bytes
    rows = (capacity - (8192 if quantize else 0)) // working_row_bytes
    if rows < 1:
        raise ValueError(
            'Export buffer must hold at least one row (512-byte alignment)'
        )
    rows = min(rows, max(b - a for a, b in zip(boundaries, boundaries[1:])))
    if rows == 0:
        raise ValueError('Row streaming requires a nonempty global table')
    scratch = torch.empty(
        rows * (weight_bytes + scale_bytes), device=weight.device, dtype=torch.uint8
    )
    for owner, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        for offset in range(begin, end, rows):
            count = min(rows, end - offset)
            payload = scratch[: count * (weight_bytes + scale_bytes)]
            w = (
                payload[: count * weight_bytes]
                .view(output_dtype)
                .view(count, weight.shape[1])
            )
            s = (
                None
                if scale_dtype is None
                else payload[count * weight_bytes :]
                .view(scale_dtype)
                .view(count, scale_width)
            )
            if rank == owner:
                local = offset - begin
                if quantize:
                    from megatron.lite.primitive.quantization.block_fp8 import (
                        quantize_block_fp8,
                    )

                    encoded, scales = quantize_block_fp8(
                        weight[local : local + count], (1, 32), scale_format='e8m0'
                    )
                    w.view(torch.uint8).copy_(encoded.view(torch.uint8))
                    s.view(torch.uint8).copy_(scales.view(torch.uint8))
                    del encoded, scales
                else:
                    w.view(torch.uint8).copy_(
                        weight[local : local + count].view(torch.uint8)
                    )
                    if s is not None:
                        s.view(torch.uint8).copy_(
                            scale[local : local + count].view(torch.uint8)
                        )
            if group is not None:
                dist.broadcast(
                    payload, src=dist.get_global_rank(group, owner), group=group
                )
            yield RowChunk(name, offset, boundaries[-1], w, s)


class RowReceiver:
    """Copy intersections into a resident contiguous global row span.

    Suitable for a hash-head bucket span as well as ordinary row sharding.
    Both planes are validated before either is written. Scales may be stored
    as raw uint8 on the receiver without changing their bytes.
    """

    def __init__(self, name, total_rows, start, weight, scale=None):
        if (
            weight.ndim != 2
            or not weight.is_contiguous()
            or (
                scale is not None
                and (
                    scale.ndim != 2
                    or scale.shape[0] != weight.shape[0]
                    or not scale.is_contiguous()
                )
            )
        ):
            raise ValueError('Receiver requires contiguous matching row planes')
        if not 0 <= start <= start + weight.shape[0] <= total_rows:
            raise ValueError('Receiver span outside global rows')
        self.name, self.total_rows, self.start = name, total_rows, start
        self.weight, self.scale, self.next_row = weight, scale, 0

    @torch.no_grad()
    def copy(self, chunk):
        count = chunk.weight.shape[0]
        if (
            chunk.name != self.name
            or chunk.total_rows != self.total_rows
            or chunk.offset != self.next_row
            or count <= 0
            or chunk.offset + count > self.total_rows
        ):
            raise ValueError('Row stream name, shape or order mismatch')
        pairs = [(self.weight, chunk.weight), (self.scale, chunk.scale)]
        for target, source in pairs:
            if (target is None) != (source is None):
                raise ValueError('Row stream scale presence mismatch')
            if target is not None and (
                target.ndim != 2
                or source.ndim != 2
                or target.shape[1] != source.shape[1]
                or source.shape[0] != count
                or target.element_size() != source.element_size()
                or (target.dtype != source.dtype and target.dtype != torch.uint8)
            ):
                raise ValueError('Row stream plane shape or dtype mismatch')
        begin = max(self.start, chunk.offset)
        end = min(self.start + self.weight.shape[0], chunk.offset + count)
        if end > begin:
            for target, source in pairs:
                if target is not None:
                    target[begin - self.start : end - self.start].view(
                        torch.uint8
                    ).copy_(
                        source[begin - chunk.offset : end - chunk.offset].view(
                            torch.uint8
                        )
                    )
        self.next_row += count

    def finish(self):
        if self.next_row != self.total_rows:
            raise ValueError('Row stream incomplete')


def write_row_file(first, following, filename):
    """Write one complete row stream as ordinary safetensors, without a table buffer.

    Consume only this table's chunks from ``following``. The output is a normal
    full-shape HF tensor pair on disk; loaders need no row-offset extension.
    """
    import json
    import struct

    types = {
        torch.float32: 'F32',
        torch.float16: 'F16',
        torch.bfloat16: 'BF16',
        torch.float8_e4m3fn: 'F8_E4M3',
        torch.float8_e8m0fnu: 'F8_E8M0',
        torch.int8: 'I8',
        torch.uint8: 'U8',
    }
    if first.offset != 0 or first.weight.ndim != 2:
        raise ValueError('Row file must start with the first row')
    planes = [(first.name, first.weight)]
    if first.scale is not None:
        if not first.name.endswith('.weight'):
            raise ValueError('Paired row weights require a .weight suffix')
        planes.append((first.name[:-6] + 'scale', first.scale))
    header, offsets, size = {}, [], 0
    for name, tensor in planes:
        row_bytes = tensor.shape[1] * tensor.element_size()
        length = first.total_rows * row_bytes
        header[name] = dict(
            dtype=types[tensor.dtype],
            shape=[first.total_rows, tensor.shape[1]],
            data_offsets=[size, size + length],
        )
        offsets.append((size, row_bytes))
        size += length
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    start, cursor, chunk = 8 + len(encoded), 0, first
    with open(filename, 'wb') as output:
        output.write(struct.pack('<Q', len(encoded)))
        output.write(encoded)
        output.truncate(start + size)
        while True:
            current = [chunk.weight] + ([] if chunk.scale is None else [chunk.scale])
            if (
                chunk.name != first.name
                or chunk.total_rows != first.total_rows
                or chunk.offset != cursor
                or len(current) != len(planes)
                or chunk.weight.shape[0] <= 0
                or cursor + chunk.weight.shape[0] > first.total_rows
            ):
                raise ValueError('Row file stream name, shape or order mismatch')
            for tensor, (_, reference) in zip(current, planes):
                if (
                    tensor.shape != (chunk.weight.shape[0], reference.shape[1])
                    or tensor.dtype != reference.dtype
                ):
                    raise ValueError('Row file plane shape or dtype mismatch')
            for tensor, (offset, row_bytes) in zip(current, offsets):
                output.seek(start + offset + cursor * row_bytes)
                # memoryview avoids a second Python bytes copy of the CPU block.
                output.write(memoryview(tensor.view(torch.uint8).cpu().numpy()))
            cursor += chunk.weight.shape[0]
            if cursor == first.total_rows:
                break
            try:
                chunk = next(following)
            except StopIteration as exc:
                raise ValueError('Row file stream incomplete') from exc
    return list(header), size
