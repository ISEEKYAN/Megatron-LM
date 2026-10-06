# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 deployment stream: bounded row pairs and byte-preserving transport."""
import json

import torch
from megatron.lite.primitive.ckpt.row_stream import RowChunk

_PREFIX = '__ds41_resync_v1__:'
_DTYPES = {'i8': torch.int8, 'e8m0': torch.float8_e8m0fnu}


def validate_target(target, options):
    if target is None and options is None:
        return False
    if target not in ('mxfp4', 'vllm'):
        raise ValueError('DS4.1 resync requires target=mxfp4 (or vllm)')
    options = dict(options or {})
    if options.keys() - {'expert_dtype'} or options.get('expert_dtype', 'fp4') != 'fp4':
        raise ValueError('DS4.1 resync supports only expert_dtype=fp4')
    return True


def transport_weights(weights, *, deployment=False):
    """Yield (name, tensor) for unmodified bucket transports.

    The consumer must copy each yielded payload before advancing the iterator.
    Row metadata is carried in names; weight and scale travel in one payload,
    so a bucket boundary cannot separate the two borrowed planes.
    """
    for item in weights:
        if isinstance(item, RowChunk):
            if item.weight.dtype != torch.float8_e4m3fn or item.scale is None:
                raise ValueError('DS4.1 row stream requires FP8 weight/E8M0 scale')
            if item.scale.dtype != torch.float8_e8m0fnu:
                raise ValueError('DS4.1 row stream requires E8M0 scales')
            meta = [
                item.name,
                'rows',
                item.offset,
                item.total_rows,
                item.weight.shape[0],
                item.weight.shape[1],
                item.scale.shape[1],
            ]
            payload = torch.cat(
                (
                    item.weight.view(torch.uint8).flatten(),
                    item.scale.view(torch.uint8).flatten(),
                )
            )
            yield _PREFIX + json.dumps(meta, separators=(',', ':')), payload
            del payload
        else:
            name, tensor = item
            if deployment and tensor.dtype in (
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ):
                # Official HF headers keep routing/sink/mHC control tensors
                # in FP32 while ordinary unquantized weights are BF16.
                controls = '.hc_' in name or name.endswith(
                    ('.attn.attn_sink', '.ffn.gate.bias', '.ffn.gate.bias_vl')
                )
                tensor = tensor.to(torch.float32 if controls else torch.bfloat16)
            code = next(
                (k for k, dtype in _DTYPES.items() if tensor.dtype == dtype), None
            )
            if code:
                yield _PREFIX + json.dumps(
                    [name, code], separators=(',', ':')
                ), tensor.view(torch.uint8)
            else:
                yield name, tensor
        del item
    if deployment:
        yield _PREFIX + '["end"]', torch.empty(0, dtype=torch.uint8)


def decoded_weights(weights):
    for pair in weights:
        item = decode_transport(*pair)
        if item is not None:
            yield item
        del item, pair


def decode_transport(name, tensor):
    """Restore bit views, or a borrowed global-row chunk. No numeric casts."""
    if not name.startswith(_PREFIX):
        return name, tensor
    meta = json.loads(name[len(_PREFIX) :])
    if tensor.dtype != torch.uint8 or not tensor.is_contiguous():
        raise ValueError('DS4.1 transport requires contiguous uint8 payloads')
    if meta == ['end'] and tensor.numel() == 0:
        return None
    if len(meta) == 2 and meta[1] in _DTYPES:
        return meta[0], tensor.view(_DTYPES[meta[1]])
    if len(meta) != 7 or meta[1] != 'rows':
        raise ValueError('Invalid DS4.1 transport metadata')
    name, _, offset, total, rows, width, scales = meta
    if (
        not all(type(x) is int for x in meta[2:])
        or rows <= 0
        or width <= 0
        or width % 32
        or scales != width // 32
        or not 0 <= offset < offset + rows <= total
        or tensor.ndim != 1
        or tensor.numel() != rows * (width + scales)
    ):
        raise ValueError('Invalid DS4.1 row payload')
    boundary = rows * width
    return RowChunk(
        name,
        offset,
        total,
        tensor[:boundary].view(torch.float8_e4m3fn).view(rows, width),
        tensor[boundary:].view(torch.float8_e8m0fnu).view(rows, scales),
    )
