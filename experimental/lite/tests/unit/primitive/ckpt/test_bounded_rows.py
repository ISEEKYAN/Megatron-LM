# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
from megatron.lite.primitive.ckpt.hf_weights import (
    BoundedTensorReader,
    stream_export_to_shards,
)
from megatron.lite.primitive.ckpt.row_stream import RowChunk, RowReceiver, stream_rows


def test_paired_raw_bytes(tmp_path):
    weight = torch.arange(320, dtype=torch.int64).to(torch.uint8).reshape(10, 32)
    scale = torch.arange(10, dtype=torch.uint8).reshape(10, 1)
    stream_export_to_shards(
        stream_rows(
            "arbitrary", weight, scale, scale_name="other-key", buffer_max_size_bytes=70
        ),
        str(tmp_path),
    )
    reader = BoundedTensorReader(str(tmp_path))
    target, target_scale = torch.empty_like(weight), torch.empty_like(scale)
    receiver = RowReceiver("arbitrary", target, target_scale)
    for chunk in reader.rows("arbitrary", 2048, "other-key"):
        receiver.copy(chunk)
    receiver.finish()
    assert torch.equal(target, weight)
    assert torch.equal(target_scale, scale)


def test_invalid_scale_never_writes_weight():
    weight, scale = torch.zeros(3, 32), torch.zeros(3, 1)
    receiver = RowReceiver("x", weight, scale)
    with pytest.raises(ValueError, match="shape"):
        receiver.copy(RowChunk("x", 0, 3, torch.ones_like(weight), torch.ones(2, 1)))
    assert not weight.any() and not scale.any()
    with pytest.raises(ValueError, match="incomplete"):
        receiver.finish()


def test_byte_weight_and_float_scale_need_aligned_payload():
    weight, scale = torch.ones(7, 3, dtype=torch.uint8), torch.ones(7, 1)
    receiver = RowReceiver(
        'odd-width', torch.empty_like(weight), torch.empty_like(scale)
    )
    for chunk in stream_rows(
        'odd-width', weight, scale, scale_name='scale', buffer_max_size_bytes=19
    ):
        receiver.copy(chunk)
    receiver.finish()
    assert torch.equal(receiver.weight, weight)
    assert torch.equal(receiver.scale, scale)
