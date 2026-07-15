# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Executable reproduction of the resync IPC bucket byte-alignment defect.

The 128-GPU real-weight (FP8) DeepSeek-V4 RL run crashes during the vLLM weight
resync, while random-init BF16 proxy runs of identical geometry stay green. The
static source analysis (``docs/ds4-resync-bucket-byte-alignment.md``) traces this
to verl's bucketed IPC transfer:

  sender:   buffer[offset : offset + w.nbytes].copy_(w.view(-1).view(uint8))
            offset += w.nbytes            # <-- no alignment padding
  receiver: buffer[offset : offset + size].view(dtype=D).view(shape)

``Tensor.view(dtype=D)`` requires the byte ``storage_offset`` to be divisible by
``D.itemsize``. A pure BF16/FP32 stream keeps every offset a multiple of 2/4, so
the receive-side view never trips (proxy green). An FP8 tensor (itemsize 1) with
an odd byte count leaves ``offset`` odd, and the next BF16/FP32 tensor in the same
bucket hits the view at an unaligned offset -> RuntimeError.

These tests reproduce the mechanism on CPU (no GPU / no live vLLM) and pin the
Fix-A padding formula ``offset = (offset + 7) & ~7`` used to close it. The full
send->live-vLLM-receive path still requires a GPU verifier; this covers the
byte-arithmetic invariant that any of fix A/B/C must satisfy.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.mlite

FP8 = torch.float8_e4m3fn


def _pack_unaligned(buffer: torch.Tensor, tensors):
    """Model verl's sender loop with no inter-tensor padding (the defect)."""
    offset = 0
    meta = []
    for name, w in tensors:
        nbytes = w.nbytes
        buffer[offset : offset + nbytes].copy_(w.reshape(-1).view(torch.uint8))
        meta.append((name, offset, w.dtype, tuple(w.shape)))
        offset += nbytes
    return meta


def _pack_aligned(buffer: torch.Tensor, tensors, align: int = 8):
    """Fix A: pad each recorded offset up to an ``align``-byte boundary."""
    offset = 0
    meta = []
    for name, w in tensors:
        offset = (offset + align - 1) & ~(align - 1)
        nbytes = w.nbytes
        buffer[offset : offset + nbytes].copy_(w.reshape(-1).view(torch.uint8))
        meta.append((name, offset, w.dtype, tuple(w.shape)))
        offset += nbytes
    return meta


def _receive(buffer: torch.Tensor, name, offset, dtype, shape):
    """Model verl's receiver: ``buffer[off:off+size].view(dtype).view(shape)``."""
    size = dtype.itemsize * int(torch.tensor(shape).prod())
    return buffer[offset : offset + size].view(dtype=dtype).view(shape)


def _fp8_then_bf16():
    # numel=3 FP8 -> 3 bytes (odd) leaves the following BF16 tensor unaligned.
    fp8 = torch.arange(3, dtype=torch.int8).view(FP8)
    bf16 = torch.tensor([1.5, -2.0, 0.25, 8.0], dtype=torch.bfloat16)
    return [("expert.w_fp8", fp8), ("norm.weight", bf16)]


def test_fp8_odd_offset_crashes_receiver_without_padding() -> None:
    """Reproduces the exact 128-run crash: unaligned storage_offset on view."""
    tensors = _fp8_then_bf16()
    buffer = torch.zeros(64, dtype=torch.uint8)
    meta = _pack_unaligned(buffer, tensors)

    # The FP8 tensor lands at offset 0; the BF16 tensor at odd offset 3.
    assert meta[1][1] == 3

    with pytest.raises(RuntimeError, match="storage_offset"):
        _receive(buffer, *meta[1])


def test_pure_bf16_stream_stays_aligned_without_padding() -> None:
    """Why proxy (BF16-only) runs stay green: every offset is even."""
    tensors = [
        ("a", torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)),
        ("b", torch.tensor([4.0, 5.0], dtype=torch.bfloat16)),
    ]
    buffer = torch.zeros(64, dtype=torch.uint8)
    meta = _pack_unaligned(buffer, tensors)

    for name, offset, dtype, shape in meta:
        assert offset % dtype.itemsize == 0
        _receive(buffer, name, offset, dtype, shape)  # no raise


def test_fix_a_padding_restores_receiver_and_preserves_values() -> None:
    """Fix A: 8-byte offset padding makes every dtype view legal and lossless."""
    tensors = _fp8_then_bf16()
    buffer = torch.zeros(64, dtype=torch.uint8)
    meta = _pack_aligned(buffer, tensors, align=8)

    # The BF16 tensor is now pushed to the next 8-byte boundary.
    assert meta[1][1] == 8

    for (name, offset, dtype, shape), (_, original) in zip(meta, tensors):
        assert offset % dtype.itemsize == 0
        received = _receive(buffer, name, offset, dtype, shape)
        # Byte-exact round trip (compare as raw bytes; FP8 has no eq kernel on CPU).
        assert torch.equal(
            received.reshape(-1).view(torch.uint8),
            original.reshape(-1).view(torch.uint8),
        )


@pytest.mark.parametrize("align", [8])
@pytest.mark.parametrize(
    "dtypes",
    [
        (FP8, torch.bfloat16),
        (FP8, torch.float32),
        (FP8, FP8, torch.bfloat16),
        (torch.bfloat16, FP8, torch.float32),
    ],
)
def test_padding_covers_mixed_dtype_buckets(dtypes, align) -> None:
    """Any FP8/BF16/FP32 mix in one bucket is view-legal after padding."""
    tensors = []
    for i, dt in enumerate(dtypes):
        # Odd element counts maximise the chance of an unaligned tail.
        src = torch.arange(2 * i + 3, dtype=torch.int8)
        w = src.view(FP8) if dt is FP8 else src.to(dtype=dt)
        tensors.append((f"t{i}", w))

    buffer = torch.zeros(256, dtype=torch.uint8)
    meta = _pack_aligned(buffer, tensors, align=align)
    for name, offset, dtype, shape in meta:
        assert offset % dtype.itemsize == 0
        _receive(buffer, name, offset, dtype, shape)  # no raise


def _receive_from_producer_meta(staging, bucket_meta):
    """Model the receiver against the offsets the production sender recorded."""
    for entry in bucket_meta.values():
        offset = entry["offset"]
        dtype = entry["dtype"]
        shape = tuple(entry["shape"])
        yield entry["name"], _receive(staging, entry["name"], offset, dtype, shape)


def test_production_sender_pads_fp8_offsets_to_8_bytes() -> None:
    """The real MLite sender (_SyncBucketProducer) applies Fix-A end to end.

    Exercises the production packing path used at 128 GPUs (not the abstract
    formula): a same-layer FP8+BF16 mix must land every recorded ``offset`` on
    an 8-byte boundary so the receiver's ``view(dtype)`` never crashes, and the
    round trip must be byte-exact.
    """
    from verl_mlite.compat import _SyncBucketProducer

    # Two tensors that hash to the SAME layer-cluster key so they share a
    # bucket: an odd-numel FP8 weight followed by a BF16 weight. Without Fix-A
    # the BF16 tensor would land at odd offset 3 and crash the receiver.
    fp8 = torch.arange(3, dtype=torch.int8).view(FP8)
    bf16 = torch.tensor([1.5, -2.0, 0.25, 8.0], dtype=torch.bfloat16)
    weights = [
        ("model.layers.0.mlp.experts.w13.weight", fp8),
        ("model.layers.0.mlp.experts.w13.weight_scale", bf16),
    ]

    producer = _SyncBucketProducer(list(weights), bucket_size=256)
    staging = torch.zeros(256, dtype=torch.uint8)

    kind, bucket_meta, _direct, used_bytes, _ready, _is_last = producer.next_bucket(
        staging
    )
    assert kind == "bucket"
    assert set(bucket_meta) == {name for name, _ in weights}

    # Fix-A: the FP8 tensor at offset 0 has odd nbytes (3); the BF16 tensor must
    # be pushed to the next 8-byte boundary instead of odd offset 3.
    scale_entry = bucket_meta["model.layers.0.mlp.experts.w13.weight_scale"]
    assert scale_entry["offset"] == 8
    assert used_bytes >= 8 + bf16.nbytes

    originals = dict(weights)
    for name, received in _receive_from_producer_meta(staging, bucket_meta):
        assert received.reshape(-1).view(torch.uint8).equal(
            originals[name].reshape(-1).view(torch.uint8)
        )
