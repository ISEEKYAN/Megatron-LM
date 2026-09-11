# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Numerical bindings of V4.1 checkpoint bytes; archival storage stays unchanged."""

import torch
from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4

_TORCH_DTYPES = {
    "I8": torch.int8,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


def _tensor(store, name):
    entry = store.entries[name]
    if entry.dtype not in _TORCH_DTYPES:
        raise TypeError(f"unsupported numerical dtype: {entry.dtype}")
    dtype = _TORCH_DTYPES[entry.dtype]
    if entry.byte_length == 0:
        return torch.empty(entry.shape, dtype=dtype)
    # Own the backing storage; neither the immutable entry nor a mapped file is mutated.
    return torch.frombuffer(bytearray(store.read(name)), dtype=dtype).reshape(entry.shape)


def load_weight(store, name, *, output_dtype=torch.bfloat16):
    """Decode one weight with its exact scale sibling, or reload a plain export.

    This is a CPU numerical binding, not the runtime FP8 activation/GEMM path.
    Engram tables use row-by-32 scales; ordinary matrices use 32-by-32 scales.
    """
    if output_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("output dtype must be F32, BF16 or F16")
    if not name.endswith(".weight"):
        raise ValueError("load_weight requires an explicit .weight key")
    weight = _tensor(store, name)
    scale_name = name[:-6] + "scale"
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        if scale_name in store.entries:
            raise ValueError(f"unexpected scale for plain weight: {name}")
        result = weight.float()
    else:
        if weight.ndim != 2:
            raise ValueError("quantized weights must be matrices")
        if scale_name not in store.entries:
            raise ValueError(f"missing scale: {scale_name}")
        scale = _tensor(store, scale_name)
        if scale.dtype != torch.float8_e8m0fnu:
            raise TypeError("release scales must be E8M0")
        if not torch.isfinite(scale.float()).all():
            raise ValueError("nonfinite release scale")
        if weight.dtype == torch.int8:
            if weight.shape[-1] % 16:
                raise ValueError("packed FP4 width must be divisible by 16")
            result = dequantize_mxfp4(weight, scale)
        elif weight.dtype == torch.float8_e4m3fn:
            rows, columns = weight.shape
            row_block = 1 if name.endswith(".engram.embed.weight") else 32
            expected = ((rows + row_block - 1) // row_block, (columns + 31) // 32)
            if tuple(scale.shape) != expected:
                raise ValueError(f"scale shape mismatch: {tuple(scale.shape)} != {expected}")
            # Reuse the aligned primitive, allowing a final partially occupied block.
            padded = torch.zeros(expected[0] * row_block, expected[1] * 32, dtype=weight.dtype)
            padded[:rows, :columns] = weight
            result = dequantize_block_fp8(padded, scale, (row_block, 32))[:rows, :columns]
        else:
            raise TypeError(f"unsupported weight dtype: {weight.dtype}")
    result = result.to(output_dtype)
    if not torch.isfinite(result).all():
        raise ValueError(f"nonfinite decoded weight: {name}")
    return result


# Assembly bindings retain exact module objects, including scale consumers.
# Header validation is separate from key-only topology checks: the latter never
# claim to have inspected release payloads or release tensor dimensions.
def bind_checkpoint(model, records, *, store=None, allow_missing_mtp=False):
    from dataclasses import replace

    records = list(records)
    names, headers = [], {}
    for record in records:
        name = record if isinstance(record, str) else record['name']
        names.append(name)
        if not isinstance(record, str):
            headers[name] = record
    if len(names) != len(set(names)):
        raise ValueError('duplicate checkpoint keys')
    available = {**model.tensor_bindings, **model.archival_bindings}
    expected = set(available)
    # The C reduced fixture intentionally excludes the complete inactive MTP tree.
    if allow_missing_mtp and not any(name.startswith('mtp.') for name in names):
        expected = {name for name in expected if not name.startswith('mtp.')}
    # Plain numerical exports have no quantization scale siblings.
    for name, header in headers.items():
        if (
            name in model.tensor_bindings
            and name.endswith('.weight')
            and _header_dtype(header['dtype']) in ('F32', 'BF16', 'F16')
        ):
            expected.discard(name[:-6] + 'scale')
    missing, extra = expected - set(names), set(names) - expected
    if missing or extra:
        raise ValueError(
            f'checkpoint coverage mismatch: missing={sorted(missing)[:8]} extra={sorted(extra)[:8]}'
        )
    if store is not None and set(store.entries) != set(names):
        raise ValueError('Store/header coverage mismatch')
    result = {}
    for name in names:
        binding = available[name]
        header = headers.get(name)
        if header is not None and binding.role != 'archival':
            dtype = _header_dtype(header['dtype'])
            if binding.role == 'scale':
                weight = model.tensor_bindings[name[:-5] + 'weight']
                rows, columns = weight.tensor.shape
                shape = (
                    (rows, (columns + 31) // 32)
                    if weight.encoding == 'I8' or weight.role == 'engram_table'
                    else ((rows + 31) // 32, (columns + 31) // 32)
                )
                if dtype != 'F8_E8M0':
                    raise ValueError(f'scale dtype mismatch: {name}')
            else:
                shape = tuple(binding.tensor.shape)
                if dtype == 'I8':
                    if binding.encoding != 'I8' or len(shape) != 2 or shape[-1] % 32:
                        raise ValueError(f'invalid packed weight: {name}')
                    shape = (*shape[:-1], shape[-1] // 2)
                elif dtype == 'F8_E4M3':
                    if binding.encoding != 'F8_E4M3':
                        raise ValueError(f'unexpected FP8 weight: {name}')
                elif dtype not in ('F32', 'BF16', 'F16'):
                    raise ValueError(f'unsupported dtype: {name}: {dtype}')
            if tuple(header['shape']) != shape:
                raise ValueError(f'checkpoint shape mismatch: {name}: {header["shape"]} != {shape}')
        result[name] = replace(binding, header=header, store=store)
    model.validate_parameter_bindings()
    model.checkpoint_bindings = result
    return result


def _header_dtype(dtype):
    return {
        'torch.float32': 'F32',
        'torch.bfloat16': 'BF16',
        'torch.float16': 'F16',
        'torch.int8': 'I8',
        'torch.float8_e4m3fn': 'F8_E4M3',
        'torch.float8_e8m0fnu': 'F8_E8M0',
    }.get(dtype, dtype)


def export_model(model):
    """Yield active numerical masters; immutable inactive bytes stay in the store.

    Plain masters intentionally drop encoded scale siblings, as load_weight's
    plain-export contract requires. Frozen Engram tables retain FP8 row storage.
    This is a lossless training export, not a quantized deployment conversion.
    """
    from .engram import EngramTable

    model.validate_parameter_bindings()
    for name, binding in model.tensor_bindings.items():
        if binding.role == 'scale':
            continue
        tensor = binding.tensor.detach()
        if tensor.is_meta:
            raise ValueError(f'Cannot export unmaterialized parameter: {name}')
        yield name, tensor
        if isinstance(binding.owner, EngramTable) and binding.owner.master is None:
            yield name[:-6] + 'scale', binding.owner.scale.detach()


def save_model(model, path):
    """Stream masters plus byte-identical archival entries to an atomic directory."""
    import hashlib
    import json
    import os
    import shutil
    import tempfile
    from pathlib import Path
    from .checkpoint_store import CheckpointTensorStore, TensorEntry

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    archive = model.archival_store
    required = set(model.archival_bindings)
    if archive is None or not required <= archive.entries.keys():
        raise ValueError('Complete MTP/vision/aligner archival storage is required for export')
    if archive.entries.keys() - model.archival_bindings.keys():
        raise ValueError('Unknown archival keys')
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.v41-export-', dir=path.parent))
    try:
        spool = staging / '.active-bytes'
        entries = dict(archive.entries)
        with spool.open('wb') as stream:
            for name, value in export_model(model):
                value = value.cpu().contiguous()
                raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
                dtype = _header_dtype(str(value.dtype))
                entries[name] = TensorEntry(
                    name,
                    dtype,
                    tuple(value.shape),
                    len(raw),
                    str(spool),
                    stream.tell(),
                    hashlib.sha256(raw).hexdigest(),
                )
                stream.write(raw)
        CheckpointTensorStore(entries).save(staging / 'model.safetensors')
        spool.unlink()
        (staging / 'config.json').write_text(json.dumps(model.config.to_hf_dict(), indent=2) + '\n')
        os.rename(staging, path)
    except BaseException:
        shutil.rmtree(staging)
        raise


@torch.no_grad()
def load_model(model, path):
    """Bind every header before loading any live tensor, retaining archive bytes."""
    import json
    from pathlib import Path
    from .checkpoint_store import CheckpointTensorStore
    from .engram import EngramTable

    path = Path(path)
    if json.loads((path / 'config.json').read_text()) != model.config.to_hf_dict():
        raise ValueError('Checkpoint config differs from the constructed model')
    index = path / 'model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_text())['weight_map']
        paths = [path / shard for shard in sorted(set(mapping.values()))]
        expected = list(mapping)
    else:
        import struct

        paths = [path / 'model.safetensors']
        with paths[0].open('rb') as source:
            size = struct.unpack('<Q', source.read(8))[0]
            expected = list(json.loads(source.read(size)))
        expected = [name for name in expected if name != '__metadata__']
    store = CheckpointTensorStore.load(paths, expected_keys=expected)
    records = [dict(name=name, dtype=e.dtype, shape=e.shape) for name, e in store.entries.items()]
    bindings = bind_checkpoint(model, records, store=store)
    for name, binding in bindings.items():
        if binding.role in ('archival', 'scale'):
            continue
        target = binding.tensor
        if target.is_meta:
            raise ValueError('Materialize the model before loading checkpoint tensors')
        if isinstance(binding.owner, EngramTable):
            table = binding.owner
            if table.master is None:
                if store.entries[name].dtype != 'F8_E4M3':
                    raise ValueError('Frozen Engram requires FP8 table storage')
                table.weight.copy_(_tensor(store, name).to(table.weight.device))
                table.scale.copy_(_tensor(store, name[:-6] + 'scale').to(table.scale.device))
            else:
                table.master.copy_(
                    load_weight(store, name, output_dtype=torch.float32).to(table.master.device)
                )
                table.refresh_storage()
        else:
            value = (
                load_weight(store, name, output_dtype=target.dtype)
                if name.endswith('.weight')
                else _tensor(store, name)
            )
            target.copy_(value.to(target.device))
    model.archival_store = CheckpointTensorStore(
        {name: e for name, e in store.entries.items() if bindings[name].role == 'archival'}
    )
