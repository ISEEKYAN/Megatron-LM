# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validated immutable tensor archives and explicit numerical storage codecs."""

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import torch
from megatron.lite.primitive.ckpt.hf_weights import _dequantize_block_scaled_tensor
from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4
from safetensors import SafetensorError, safe_open

_CHUNK = 8 * 1024 * 1024


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
                header = json.loads(
                    source.read(length), object_pairs_hook=_unique_object
                )
            for name in keys:
                if key_prefix is not None and not name.startswith(key_prefix):
                    continue
                if name in entries:
                    raise ValueError(f"duplicate tensor: {name}")
                record = header[name]
                start, end = record["data_offsets"]
                entry = TensorEntry(
                    name,
                    record["dtype"],
                    tuple(record["shape"]),
                    end - start,
                    str(path),
                    8 + length + start,
                    "",
                )
                entries[name] = replace(entry, payload_digest=_stream(entry))
        _coverage(entries, expected_keys)
        return cls(entries)

    def read(self, name):
        import io

        output = io.BytesIO()
        entry = self.entries[name]
        if _stream(entry, output) != entry.payload_digest:
            raise ValueError(f"payload digest mismatch: {name}")
        return output.getvalue()

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


def _decode_fp8(weight, scale, row_block):
    return _dequantize_block_scaled_tensor(
        weight, scale, weight.shape, block_shape=(row_block, 32)
    )


def _decode_fp4(weight, scale, row_block):
    if weight.shape[-1] % 16:
        raise ValueError("packed FP4 width must be divisible by 16")
    return dequantize_mxfp4(weight, scale)


# Checkpoint format -> (default row block, decoder). Activations' group16/E4M3
# QAT is a different contract; serialized expert FP4 here uses group32/E8M0.
_CODECS = {torch.int8: (1, _decode_fp4), torch.float8_e4m3fn: (32, _decode_fp8)}
_PLAIN = (torch.float32, torch.bfloat16, torch.float16)


def load_weight(
    store, name, *, output_dtype=torch.bfloat16, row_block=None, read_tensor=None
):
    """Decode the declared storage format; plain exports have no scale sibling."""
    read_tensor = _tensor if read_tensor is None else read_tensor
    if output_dtype not in _PLAIN:
        raise TypeError("output dtype must be F32, BF16 or F16")
    if not name.endswith(".weight"):
        raise ValueError("load_weight requires an explicit .weight key")
    weight, scale_name = read_tensor(store, name), name[:-6] + "scale"
    if weight.dtype in _PLAIN:
        if scale_name in store.entries:
            raise ValueError(f"unexpected scale for plain weight: {name}")
        result = weight.float()
    else:
        if weight.ndim != 2:
            raise ValueError("quantized weights must be matrices")
        if scale_name not in store.entries:
            raise ValueError(f"missing scale: {scale_name}")
        scale = read_tensor(store, scale_name)
        if scale.dtype != torch.float8_e8m0fnu:
            raise TypeError("release scales must be E8M0")
        if not torch.isfinite(scale.float()).all():
            raise ValueError("nonfinite release scale")
        if weight.dtype not in _CODECS:
            raise TypeError(f"unsupported weight dtype: {weight.dtype}")
        default_block, decode = _CODECS[weight.dtype]
        result = decode(
            weight, scale, default_block if row_block is None else row_block
        )
    result = result.to(output_dtype)
    if not torch.isfinite(result).all():
        raise ValueError(f"nonfinite decoded weight: {name}")
    return result


def _header_dtype(dtype):
    return {str(value): key for key, value in _TORCH_DTYPES.items()}.get(dtype, dtype)
