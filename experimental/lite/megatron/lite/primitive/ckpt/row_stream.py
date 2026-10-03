# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Borrowed row blocks: consume before advancing; never collect them in a list."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RowChunk:
    name: str
    offset: int
    total_rows: int
    weight: torch.Tensor
    scale: torch.Tensor | None = None
    scale_name: str | None = None


@torch.no_grad()
def stream_rows(name, weight, *, buffer_max_size_bytes=5 * 1024**3):
    """Copy local rows through one buffer; consume each before advancing.

    Resident inputs and consumer-owned outputs are excluded from the scratch
    limit. Payload views expire on the next iteration.
    """
    if type(buffer_max_size_bytes) is not int or buffer_max_size_bytes <= 0:
        raise ValueError("Export buffer must hold at least one row")
    if weight.ndim != 2 or not weight.is_contiguous() or weight.shape[1] == 0:
        raise ValueError("Expected contiguous 2D rows")
    rows = buffer_max_size_bytes // (weight.shape[1] * weight.element_size())
    if rows < 1:
        raise ValueError("Export buffer must hold at least one row")
    rows = min(rows, weight.shape[0])
    if rows == 0:
        raise ValueError("Row streaming requires a nonempty global table")
    scratch = torch.empty(
        (rows, weight.shape[1]), device=weight.device, dtype=weight.dtype
    )
    for offset in range(0, weight.shape[0], rows):
        count = min(rows, weight.shape[0] - offset)
        payload = scratch[:count]
        payload.view(torch.uint8).copy_(
            weight[offset : offset + count].view(torch.uint8)
        )
        yield RowChunk(name, offset, weight.shape[0], payload)


class RowReceiver:
    """Copy ordered rows into a complete resident contiguous table."""

    def __init__(self, name, weight):
        if weight.ndim != 2 or not weight.is_contiguous():
            raise ValueError("Receiver requires contiguous 2D rows")
        self.name, self.total_rows = name, weight.shape[0]
        self.weight, self.next_row = weight, 0

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
            raise ValueError("Row stream name, shape or order mismatch")
        source = chunk.weight
        if (
            chunk.scale is not None
            or source.ndim != 2
            or self.weight.shape[1] != source.shape[1]
            or self.weight.dtype != source.dtype
        ):
            raise ValueError("Row stream shape or dtype mismatch")
        self.weight[chunk.offset : chunk.offset + count].view(torch.uint8).copy_(
            source.view(torch.uint8)
        )
        self.next_row += count

    def finish(self):
        if self.next_row != self.total_rows:
            raise ValueError("Row stream incomplete")


def write_row_file(first, following, filename):
    """Write one complete row stream as ordinary safetensors, without a table buffer.

    Consume only this table's chunks from ``following``. The output is a normal
    full-shape HF tensor pair on disk; loaders need no row-offset extension.
    """
    import json
    import struct

    types = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e8m0fnu: "F8_E8M0",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.bool: "BOOL",
        torch.uint8: "U8",
    }
    if first.offset != 0 or first.weight.ndim != 2:
        raise ValueError("Row file must start with the first row")
    planes = [(first.name, first.weight)]
    if first.scale is not None:
        if not first.scale_name or first.scale_name == first.name:
            raise ValueError("Paired planes require a distinct scale key")
        planes.append((first.scale_name, first.scale))
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
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    start, cursor, chunk = 8 + len(encoded), 0, first
    with open(filename, "wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        output.truncate(start + size)
        while True:
            current = [chunk.weight] + ([] if chunk.scale is None else [chunk.scale])
            if (
                chunk.name != first.name
                or chunk.scale_name != first.scale_name
                or chunk.total_rows != first.total_rows
                or chunk.offset != cursor
                or len(current) != len(planes)
                or chunk.weight.shape[0] <= 0
                or cursor + chunk.weight.shape[0] > first.total_rows
            ):
                raise ValueError("Row file stream name, shape or order mismatch")
            for tensor, (_, reference) in zip(current, planes):
                if (
                    tensor.shape != (chunk.weight.shape[0], reference.shape[1])
                    or tensor.dtype != reference.dtype
                ):
                    raise ValueError("Row file plane shape or dtype mismatch")
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
                raise ValueError("Row file stream incomplete") from exc
    return list(header), size
