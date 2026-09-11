# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Immutable file-backed checkpoint bytes, independent of model execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
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
            size = path.stat().st_size
            with path.open("rb") as source:
                prefix = source.read(8)
                if len(prefix) != 8:
                    raise ValueError("truncated safetensors header")
                length = struct.unpack("<Q", prefix)[0]
                if length > min(100_000_000, size - 8):
                    raise ValueError("invalid safetensors header length")
                header = json.loads(
                    source.read(length), object_pairs_hook=_unique_object
                )
            if not isinstance(header, dict):
                raise ValueError("header must be an object")
            cursor = 0
            records = []
            for name, record in header.items():
                if name == "__metadata__":
                    if not isinstance(record, dict) or not all(
                        isinstance(k, str) and isinstance(v, str)
                        for k, v in record.items()
                    ):
                        raise ValueError("invalid safetensors metadata")
                    continue
                if not isinstance(record, dict) or set(record) != {
                    "dtype",
                    "shape",
                    "data_offsets",
                }:
                    raise ValueError(f"invalid tensor header: {name}")
                dtype, shape, offsets = (
                    record["dtype"],
                    record["shape"],
                    record["data_offsets"],
                )
                if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
                    raise ValueError(f"unsupported header dtype: {dtype}")
                if not isinstance(shape, list) or any(
                    type(n) is not int or n < 0 for n in shape
                ):
                    raise ValueError(f"invalid shape: {name}")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(type(n) is not int or n < 0 for n in offsets)
                ):
                    raise ValueError(f"invalid offsets: {name}")
                start, end = offsets
                if end - start != math.prod(shape) * _DTYPE_BYTES[dtype]:
                    raise ValueError(f"shape/byte length mismatch: {name}")
                records.append((start, end, name, dtype, tuple(shape)))
            for start, end, name, dtype, shape in sorted(records):
                if start != cursor or end > size - 8 - length:
                    raise ValueError(f"noncontiguous or truncated payload: {name}")
                cursor = end
                if key_prefix is not None and not name.startswith(key_prefix):
                    continue
                if name in entries:
                    raise ValueError(f"duplicate tensor: {name}")
                entry = TensorEntry(
                    name, dtype, shape, end - start, str(path), 8 + length + start, ""
                )
                entries[name] = TensorEntry(
                    **{**asdict(entry), "payload_digest": _stream(entry)}
                )
            if cursor != size - 8 - length:
                raise ValueError("unaccounted trailing payload bytes")
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
