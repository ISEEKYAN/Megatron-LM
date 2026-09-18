# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint traversal for explicit bindings and model-supplied storage policy."""

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from .hf_weights import (
    DEFAULT_EXPORT_BUFFER_MAX_SIZE_BYTES,
    SafeTensorReader,
    _cast_export_tensor,
    _ep_all_gather,
    _resolve_export_dtype,
    stream_export_to_shards,
)
from .tensor_archive import (
    _PLAIN,
    CheckpointTensorStore,
    TensorEntry,
    _header_dtype,
    _tensor,
    load_weight,
)


# Assembly bindings retain exact module objects, including scale consumers.
# Header validation is separate from key-only topology checks: the latter never
# claim to have inspected release payloads or release tensor dimensions.
def bind_checkpoint(model, records, *, spec, store=None, allow_missing_archive=False):
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
    local_names = set(available)
    available = spec.expand_bindings(model, available)
    expected = set(available)
    # A model may declare one optional inactive subtree.
    if allow_missing_archive and not any(
        name.startswith(spec.optional_prefix) for name in names
    ):
        expected = {
            name for name in expected if not name.startswith(spec.optional_prefix)
        }
    # Plain numerical exports have no quantization scale siblings.
    for name, header in headers.items():
        if (
            name in available
            and available[name].role != 'archival'
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
                weight = available[name[:-5] + 'weight']
                rows, columns = _logical_shape(weight, spec)
                row_block, col_block = spec.scale_block(weight)
                shape = (
                    (rows + row_block - 1) // row_block,
                    (columns + col_block - 1) // col_block,
                )
                if dtype != 'F8_E8M0':
                    raise ValueError(f'scale dtype mismatch: {name}')
            else:
                shape = _logical_shape(binding, spec)
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
                raise ValueError(
                    f'checkpoint shape mismatch: {name}: {header["shape"]} != {shape}'
                )
        if name in local_names:
            result[name] = replace(binding, header=header, store=store)
    model.validate_parameter_bindings()
    model.checkpoint_bindings = result
    return result


def _logical_shape(binding, spec):
    shape = tuple(binding.tensor.shape)
    lookup = spec.row_shard(binding.owner)
    if lookup is not None:
        shape = (lookup.boundaries[-1], *shape[1:])
    return shape


def _export_rows(lookup, tensor):
    if lookup is None:
        return tensor
    sizes = [b - a for a, b in zip(lookup.boundaries, lookup.boundaries[1:])]
    # NCCL transports encoded FP8 storage as bytes. Padding is transport only.
    value = tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor
    padded = value.new_zeros(max(sizes), value.shape[1])
    padded[: value.shape[0]].copy_(value)
    chunks = [torch.empty_like(padded) for _ in sizes]
    _ep_all_gather(chunks, padded, lookup.group)
    return torch.cat([chunk[:size] for chunk, size in zip(chunks, sizes)]).view(
        tensor.dtype
    )


def _local_rows(lookup, tensor):
    if lookup is None:
        return tensor
    return tensor[lookup.boundaries[lookup.rank] : lookup.boundaries[lookup.rank + 1]]


def export_model(model, *, spec):
    """Yield active numerical masters; immutable inactive bytes stay in the store.

    Plain masters intentionally drop encoded scale siblings, as load_weight's
    plain-export contract requires. Frozen encoded tables retain their storage dtype.
    This is a lossless training export, not a quantized deployment conversion.
    """
    if model.local_layer_range != (0, len(model.layers)):
        raise NotImplementedError(
            'Pipeline stage export requires distributed checkpoint assembly'
        )
    model.validate_parameter_bindings()
    for name, binding in model.tensor_bindings.items():
        if binding.role == 'scale':
            continue
        if model.ps.ep_size > 1 and binding.role == 'expert':
            continue
        tensor = binding.tensor.detach()
        if tensor.is_meta:
            raise ValueError(f'Cannot export unmaterialized parameter: {name}')
        yield name, _export_rows(spec.row_shard(binding.owner), tensor)
        storage = spec.frozen_storage(binding.owner)
        if storage is not None:
            yield name[:-6] + 'scale', _export_rows(
                spec.row_shard(binding.owner), storage[1].detach()
            )
    if model.ps.ep_size > 1:
        import torch.distributed as dist

        local = {
            name: b.tensor.detach()
            for name, b in model.tensor_bindings.items()
            if b.role == 'expert'
        }
        metadata = [
            (name, tuple(t.shape), t.dtype, dist.get_rank())
            for name, t in local.items()
        ]
        gathered = [None] * model.ps.ep_size
        dist.all_gather_object(gathered, metadata, group=model.ps.ep_group)
        device = next(model.parameters()).device
        for name, shape, dtype, source in sorted(
            record for records in gathered for record in records
        ):
            value = (
                local[name].contiguous()
                if name in local
                else torch.empty(shape, dtype=dtype, device=device)
            )
            dist.broadcast(value, src=source, group=model.ps.ep_group)
            yield name, value


def export_checkpoint(
    model,
    *,
    spec,
    export_dtype=None,
    cpu=False,
    buffer_max_size_bytes=DEFAULT_EXPORT_BUFFER_MAX_SIZE_BYTES,
):
    """Stream complete tensors, bounding conversion copies by the buffer budget.

    A returned tensor is indivisible and may exceed the budget. Encoded
    tables/scales and inactive archival payloads retain their original bytes;
    export_dtype applies only to active plain floating-point weights.
    """
    dtype = _resolve_export_dtype(export_dtype)
    if dtype not in (None, *_PLAIN):
        raise ValueError(f'Unsupported export_dtype={export_dtype!r}')
    if type(cpu) is not bool:
        raise ValueError('cpu must be bool')
    if type(buffer_max_size_bytes) is not int or buffer_max_size_bytes < 4:
        raise ValueError('buffer_max_size_bytes must be an integer >= 4')
    if model.archival_store is None or set(model.archival_store.entries) != set(
        model.archival_bindings
    ):
        raise ValueError('Complete archival storage is required for export')
    for name, tensor in export_model(model, spec=spec):
        target_dtype = (
            (dtype or tensor.dtype) if tensor.dtype in _PLAIN else tensor.dtype
        )
        device = torch.device('cpu') if cpu else tensor.device
        yield name, _cast_export_tensor(
            tensor,
            target_dtype,
            device=device,
            buffer_max_size_bytes=buffer_max_size_bytes,
        )
    for name in model.archival_store.entries:
        tensor = _tensor(model.archival_store, name)
        if not cpu:
            tensor = tensor.to(next(iter(model.tensor_bindings.values())).tensor.device)
        yield name, tensor


def save_model(
    model, path, *, spec, export_dtype=None, cpu=True, buffer_max_size_bytes=None
):
    """Stream masters plus byte-identical archival entries to an atomic directory."""
    import shutil

    if type(cpu) is not bool:
        raise ValueError("cpu must be bool")
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(path)
    archive = model.archival_store
    required = set(model.archival_bindings)
    if archive is None or not required <= archive.entries.keys():
        raise ValueError(spec.archive_required_message)
    if archive.entries.keys() - model.archival_bindings.keys():
        raise ValueError('Unknown archival keys')
    if model.ps.dp_cp_size > 1 and torch.distributed.get_rank() != 0:
        # All ranks participate in row/expert export; only rank zero publishes.
        for _ in export_model(model, spec=spec):
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=spec.staging_prefix, dir=path.parent))
    try:
        if export_dtype is not None or buffer_max_size_bytes is not None or not cpu:
            buffer_size = (
                DEFAULT_EXPORT_BUFFER_MAX_SIZE_BYTES
                if buffer_max_size_bytes is None
                else buffer_max_size_bytes
            )
            stream_export_to_shards(
                export_checkpoint(
                    model,
                    spec=spec,
                    export_dtype=export_dtype,
                    cpu=cpu,
                    buffer_max_size_bytes=buffer_size,
                ),
                str(staging),
                shard_size_bytes=buffer_size,
            )
        else:
            spool = staging / '.active-bytes'
            entries = dict(archive.entries)
            with spool.open('wb') as stream:
                for name, value in export_model(model, spec=spec):
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
        (staging / 'config.json').write_text(
            json.dumps(model.config.to_hf_dict(), indent=2) + '\n'
        )
        os.rename(staging, path)
    except BaseException:
        shutil.rmtree(staging)
        raise


@torch.no_grad()
def load_model(model, path, *, spec, allow_missing_archive=False):
    """Bind every header before loading any live tensor, retaining archive bytes."""
    path = Path(path)
    if json.loads((path / 'config.json').read_text()) != model.config.to_hf_dict():
        raise ValueError('Checkpoint config differs from the constructed model')
    index = path / 'model.safetensors.index.json'
    if index.exists():
        mapping = SafeTensorReader(str(path)).index
        paths = [path / shard for shard in sorted(set(mapping.values()))]
        expected = list(mapping)
    else:
        paths = [path / 'model.safetensors']
        with paths[0].open('rb') as source:
            size = struct.unpack('<Q', source.read(8))[0]
            expected = list(json.loads(source.read(size)))
        expected = [name for name in expected if name != '__metadata__']
    store = CheckpointTensorStore.load(paths, expected_keys=expected)
    records = [
        dict(name=name, dtype=e.dtype, shape=e.shape)
        for name, e in store.entries.items()
    ]
    bindings = bind_checkpoint(
        model,
        records,
        spec=spec,
        store=store,
        allow_missing_archive=allow_missing_archive,
    )
    for name, binding in bindings.items():
        if binding.role in ('archival', 'scale'):
            continue
        target = binding.tensor
        if target.is_meta:
            raise ValueError('Materialize the model before loading checkpoint tensors')
        storage = spec.frozen_storage(binding.owner)
        lookup = spec.row_shard(binding.owner)
        if storage is not None:
            if store.entries[name].dtype != 'F8_E4M3':
                raise ValueError(spec.frozen_storage_message)
            for key, destination in (
                (name, storage[0]),
                (name[:-6] + 'scale', storage[1]),
            ):
                destination.copy_(
                    _local_rows(lookup, _tensor(store, key)).to(destination.device)
                )
        else:
            value = (
                load_weight(
                    store,
                    name,
                    output_dtype=target.dtype,
                    row_block=spec.row_block(name),
                )
                if name.endswith('.weight')
                else _tensor(store, name)
            )
            target.copy_(_local_rows(lookup, value).to(target.device))
            spec.refresh_storage(binding.owner)
    model.archival_store = CheckpointTensorStore(
        {
            name: e
            for name, e in store.entries.items()
            if name in model.archival_bindings
        }
    )
