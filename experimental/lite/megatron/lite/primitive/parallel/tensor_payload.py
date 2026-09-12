# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Typed tensor P2P with exact integer generation headers and no dtype casts.

This transport is schedule-neutral. Both peers must execute matching operations;
callers retain autograd graphs and explicitly send cotangents in backward order.
A rejected generation is acknowledged before payload transfer so both peers fail.
"""

import math

import torch
import torch.distributed as dist

_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e8m0fnu,
    torch.int64,
    torch.int32,
    torch.int8,
    torch.uint8,
    torch.bool,
)
_MAGIC = 4101


def send_tensor_payload(tensors, tag, *, peer, group, device):
    tensors, tag = tuple(tensors), tuple(tag)
    if (
        not tag
        or len(tag) > 16
        or any(type(v) is not int for v in tag)
        or len(tensors) > 32
    ):
        raise ValueError('Invalid tensor payload tag or field count')
    metadata = [_MAGIC, len(tag), len(tensors), *tag]
    device = torch.device(device)
    for value in tensors:
        if value is None:
            metadata.extend((-1, 0, 0))
        else:
            if value.device != device or value.dtype not in _DTYPES or value.ndim > 8:
                raise ValueError('Unsupported tensor payload device/dtype/dimensions')
            metadata.extend(
                (
                    _DTYPES.index(value.dtype),
                    value.ndim,
                    int(value.requires_grad),
                    *value.shape,
                )
            )
    header = torch.tensor(metadata, dtype=torch.int64, device=device)
    size = torch.tensor([header.numel()], dtype=torch.int64, device=device)
    dist.send(size, peer, group=group)
    dist.send(header, peer, group=group)
    accepted = torch.empty((), dtype=torch.int64, device=device)
    dist.recv(accepted, peer, group=group)
    if accepted.item() != 1:
        raise RuntimeError('Peer rejected tensor payload generation or schema')
    for value in tensors:
        if value is not None and value.numel():
            # NCCL need not support the floating storage dtype: bytes are exact.
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
            dist.send(raw, peer, group=group)


def recv_tensor_payload(expected_tag, *, peer, group, device):
    device = torch.device(device)
    size = torch.empty(1, dtype=torch.int64, device=device)
    dist.recv(size, peer, group=group)
    if not 3 <= size.item() <= 1024:
        raise RuntimeError('Invalid tensor payload header length')
    header = torch.empty(int(size.item()), dtype=torch.int64, device=device)
    dist.recv(header, peer, group=group)
    metadata = header.tolist()  # Small shape/tag metadata only, never tensor data.
    descriptors = []
    error = None
    try:
        magic, tags, count = metadata[:3]
        if magic != _MAGIC or not 1 <= tags <= 16 or not 0 <= count <= 32:
            raise ValueError('Invalid tensor payload schema')
        if tuple(metadata[3 : 3 + tags]) != tuple(expected_tag):
            raise ValueError('Stale or wrong tensor payload generation')
        offset = 3 + tags
        for _ in range(count):
            dtype, ndim, gradient = metadata[offset : offset + 3]
            offset += 3
            shape = tuple(metadata[offset : offset + ndim])
            offset += ndim
            if dtype == -1 and ndim == 0 and gradient == 0:
                descriptors.append(None)
                continue
            if (
                not 0 <= dtype < len(_DTYPES)
                or not 0 <= ndim <= 8
                or len(shape) != ndim
                or any(d < 0 for d in shape)
                or gradient not in (0, 1)
            ):
                raise ValueError('Invalid tensor descriptor')
            if gradient and not _DTYPES[dtype].is_floating_point:
                raise ValueError('Integer tensor cannot carry an autograd edge')
            descriptors.append((_DTYPES[dtype], shape, bool(gradient)))
        if offset != len(metadata):
            raise ValueError('Unexpected tensor payload header tail')
    except (ValueError, IndexError) as exc:
        error = exc
    dist.send(
        torch.tensor(int(error is None), dtype=torch.int64, device=device),
        peer,
        group=group,
    )
    if error is not None:
        raise RuntimeError('Rejected tensor payload generation or schema') from error
    result = []
    for descriptor in descriptors:
        if descriptor is None:
            result.append(None)
            continue
        dtype, shape, gradient = descriptor
        length = math.prod(shape)
        raw = torch.empty(
            length * torch.empty((), dtype=dtype).element_size(),
            dtype=torch.uint8,
            device=device,
        )
        if length:
            dist.recv(raw, peer, group=group)
        value = raw.view(dtype).reshape(shape)
        value.requires_grad_(gradient)
        result.append(value)
    return tuple(result)
