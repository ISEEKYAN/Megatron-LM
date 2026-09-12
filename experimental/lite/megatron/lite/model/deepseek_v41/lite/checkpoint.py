# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Immutable file-backed checkpoint bytes, independent of model execution."""

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

from safetensors import SafetensorError, safe_open
import torch
from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4

_CHUNK = 8 * 1024 * 1024


def validate_execution(*, enable_dspark_execution: bool = False) -> None:
    if type(enable_dspark_execution) is not bool:
        raise TypeError("enable_dspark_execution must be bool")
    if enable_dspark_execution:
        raise NotImplementedError("DSpark execution is not implemented")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate header key: {key}")
        result[key] = value
    return result


def _coverage(entries, expected_keys):
    expected = list(expected_keys)
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate expected keys")
    missing, extra = set(expected) - entries.keys(), entries.keys() - set(expected)
    if missing or extra:
        raise ValueError(
            f"key coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )


@dataclass(frozen=True)
class TensorEntry:
    release_key: str
    dtype: str
    shape: tuple[int, ...]
    byte_length: int
    source_shard: str
    offset: int
    payload_digest: str


def _stream(entry, output=None):
    digest = hashlib.sha256()
    with open(entry.source_shard, "rb") as source:
        source.seek(entry.offset)
        remaining = entry.byte_length
        while remaining:
            raw = source.read(min(remaining, _CHUNK))
            if not raw:
                raise ValueError(f"truncated payload: {entry.release_key}")
            remaining -= len(raw)
            digest.update(raw)
            if output is not None:
                output.write(raw)
    return digest.hexdigest()


class CheckpointTensorStore:
    """Byte ranges with validated metadata; no tensor/model allocation or casting."""

    def __init__(self, entries):
        self.entries = MappingProxyType(dict(sorted(entries.items())))

    @classmethod
    def load(cls, paths, *, expected_keys, key_prefix=None):
        entries = {}
        for path in paths:
            path = Path(path).resolve()
            # Delegate shape, dtype, offset and complete-file validation to the
            # safetensors reader; retain byte ranges for lossless archival copies.
            try:
                with safe_open(path, framework="pt") as source:
                    keys = list(source.keys())
            except SafetensorError as error:
                raise ValueError(f"invalid safetensors file: {path}") from error
            with path.open("rb") as source:
                length = struct.unpack("<Q", source.read(8))[0]
                header = json.loads(source.read(length), object_pairs_hook=_unique_object)
            for name in keys:
                if key_prefix is not None and not name.startswith(key_prefix):
                    continue
                if name in entries:
                    raise ValueError(f"duplicate tensor: {name}")
                record = header[name]
                start, end = record["data_offsets"]
                entry = TensorEntry(
                    name, record["dtype"], tuple(record["shape"]),
                    end - start, str(path), 8 + length + start, "",
                )
                entries[name] = TensorEntry(
                    **{**asdict(entry), "payload_digest": _stream(entry)}
                )
        _coverage(entries, expected_keys)
        return cls(entries)

    def manifest(self):
        return {name: asdict(entry) for name, entry in self.entries.items()}

    def read(self, name):
        import io

        output = io.BytesIO()
        entry = self.entries[name]
        if _stream(entry, output) != entry.payload_digest:
            raise ValueError(f"payload digest mismatch: {name}")
        return output.getvalue()

    def shard(self, rank, world_size):
        if (
            type(world_size) is not int
            or world_size < 1
            or type(rank) is not int
            or not 0 <= rank < world_size
        ):
            raise ValueError("invalid archival rank/world size")
        return type(self)(
            {key: self.entries[key] for key in list(self.entries)[rank::world_size]}
        )

    @classmethod
    def merge(cls, stores, *, expected_keys):
        entries = {}
        for store in stores:
            for name, entry in store.entries.items():
                if name in entries:
                    raise ValueError(f"duplicate tensor: {name}")
                entries[name] = entry
        _coverage(entries, expected_keys)
        return cls(entries)

    def save(self, path):
        """Stream to a new file, publishing only after every payload digest agrees."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        header, cursor = {}, 0
        for name, entry in self.entries.items():
            header[name] = dict(
                dtype=entry.dtype,
                shape=list(entry.shape),
                data_offsets=[cursor, cursor + entry.byte_length],
            )
            cursor += entry.byte_length
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(struct.pack("<Q", len(encoded)))
                output.write(encoded)
                for name, entry in self.entries.items():
                    if _stream(entry, output) != entry.payload_digest:
                        raise ValueError(f"payload digest mismatch: {name}")
                output.flush()
                os.fsync(output.fileno())
            # Exclusive publication also prevents a concurrent writer from being overwritten.
            os.link(temporary, path)
        finally:
            os.unlink(temporary)



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
    return torch.frombuffer(bytearray(store.read(name)), dtype=dtype).reshape(
        entry.shape
    )


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
                raise ValueError(
                    f"scale shape mismatch: {tuple(scale.shape)} != {expected}"
                )
            # Reuse the aligned primitive, allowing a final partially occupied block.
            padded = torch.zeros(
                expected[0] * row_block, expected[1] * 32, dtype=weight.dtype
            )
            padded[:rows, :columns] = weight
            result = dequantize_block_fp8(padded, scale, (row_block, 32))[
                :rows, :columns
            ]
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
                raise ValueError(
                    f'checkpoint shape mismatch: {name}: {header["shape"]} != {shape}'
                )
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

    if model.local_layer_range != (0, len(model.layers)):
        raise NotImplementedError(
            'Pipeline stage export requires distributed checkpoint assembly'
        )
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


    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    archive = model.archival_store
    required = set(model.archival_bindings)
    if archive is None or not required <= archive.entries.keys():
        raise ValueError(
            'Complete MTP/vision/aligner archival storage is required for export'
        )
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
        (staging / 'config.json').write_text(
            json.dumps(model.config.to_hf_dict(), indent=2) + '\n'
        )
        os.rename(staging, path)
    except BaseException:
        shutil.rmtree(staging)
        raise


@torch.no_grad()
def load_model(model, path):
    """Bind every header before loading any live tensor, retaining archive bytes."""
    import json
    from pathlib import Path

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
    records = [
        dict(name=name, dtype=e.dtype, shape=e.shape)
        for name, e in store.entries.items()
    ]
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
                table.scale.copy_(
                    _tensor(store, name[:-6] + 'scale').to(table.scale.device)
                )
            else:
                table.master.copy_(
                    load_weight(store, name, output_dtype=torch.float32).to(
                        table.master.device
                    )
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
        {
            name: e
            for name, e in store.entries.items()
            if bindings[name].role == 'archival'
        }
    )
