"""Validate stage-local checkpoint bindings without reading tensor payloads."""

import json
import math
from pathlib import Path

import torch
from megatron.lite.primitive.ckpt.hf_weights import validate_bound_checkpoint_keys

DTYPES = {
    'BF16': torch.bfloat16,
    'F32': torch.float32,
    'F16': torch.float16,
    'F8_E4M3': torch.float8_e4m3fn,
    'F8_E8M0': torch.float8_e8m0fnu,
    'I8': torch.int8,
    'I32': torch.int32,
    'I64': torch.int64,
    'U8': torch.uint8,
}
PLAIN = (torch.bfloat16, torch.float32, torch.float16)


def read_headers(path):
    """{key: (shape, dtype_str)} for every tensor in the index, header bytes only."""
    path = Path(path)
    index = json.loads((path / 'model.safetensors.index.json').read_text())[
        'weight_map'
    ]
    table = {}
    for shard in sorted(set(index.values())):
        with open(path / shard, 'rb') as stream:
            size = int.from_bytes(stream.read(8), 'little')
            header = json.loads(stream.read(size))
        for key, meta in header.items():
            if key == '__metadata__':
                continue
            if key not in index:
                continue
            if key in table or index[key] != shard:
                raise ValueError(f'Header/index shard mismatch: {key}')
            table[key] = (list(meta['shape']), meta['dtype'])
    missing = set(index) - set(table)
    if missing:
        raise ValueError(f'index keys absent from shard headers: {sorted(missing)[:5]}')
    return {k: table[k] for k in index}


def _rows(lookup, shape):
    if lookup is None:
        return list(shape)
    cuts = lookup.boundaries
    return [cuts[lookup.rank + 1] - cuts[lookup.rank], *shape[1:]]


def check(model, path, spec, headers):
    path = Path(path)
    assert (
        json.loads((path / 'config.json').read_text()) == model.config.to_hf_dict()
    ), 'config'
    keys = set(headers)
    active, archive = validate_bound_checkpoint_keys(model, keys, spec)
    expected = active | archive
    checked, by_dtype, layers = 0, {}, set()
    for name, binding in model.tensor_bindings.items():
        if binding.role == 'scale':
            continue
        shape, dtype_s = headers[name]
        dtype = DTYPES[dtype_s]
        lookup = spec.row_shard(binding.owner)
        storage = spec.frozen_storage(binding.owner)
        scale_name = name[:-6] + 'scale'
        if storage is not None:
            for key, dest in ((name, storage[0]), (scale_name, storage[1])):
                s, d = headers[key]
                assert (
                    _rows(lookup, s) == list(dest.shape) and DTYPES[d] == dest.dtype
                ), (key, s, d, dest.shape, dest.dtype)
        elif not name.endswith('.weight') or dtype in PLAIN:
            assert scale_name not in keys or not name.endswith('.weight'), (
                'unexpected scale',
                name,
            )
            assert _rows(lookup, shape) == list(binding.tensor.shape), (
                name,
                shape,
                binding.tensor.shape,
            )
        else:
            assert scale_name in keys and headers[scale_name][1] == 'F8_E8M0', (
                'scale',
                name,
            )
            sshape = headers[scale_name][0]
            if dtype == torch.int8:
                logical = [*shape[:-1], shape[-1] * 2]
                assert sshape == [*shape[:-1], shape[-1] * 2 // 32], (
                    scale_name,
                    sshape,
                )
            elif dtype == torch.float8_e4m3fn:
                logical = list(shape)
                rb = spec.row_block(name) or 32
                assert sshape == [math.ceil(shape[0] / rb), math.ceil(shape[1] / 32)], (
                    scale_name,
                    sshape,
                )
            else:
                raise TypeError((name, dtype_s))
            assert _rows(lookup, logical) == list(binding.tensor.shape), (
                name,
                logical,
                binding.tensor.shape,
            )
        checked += 1
        by_dtype[dtype_s] = by_dtype.get(dtype_s, 0) + 1
        if name.startswith('layers.'):
            layers.add(int(name.split('.')[1]))
    return dict(
        index_keys=len(keys),
        expected_keys=len(expected),
        archival_keys=len(model.archival_bindings),
        local_bindings_checked=checked,
        local_by_dtype=by_dtype,
        layers_covered=sorted(layers),
        loader_rules='production key gate; codec shape checks on real shard headers',
    )
