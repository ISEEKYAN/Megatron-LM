# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.lite.primitive.ckpt.hf_weights import (
    BoundedTensorReader,
    stream_export_to_shards,
)
from megatron.lite.primitive.ckpt.row_stream import RowChunk, RowReceiver, stream_rows


def test_paired_raw_bytes_and_reshard(tmp_path):
    weight = torch.arange(320, dtype=torch.int64).to(torch.uint8).reshape(10, 32)
    scale = torch.arange(10, dtype=torch.uint8).reshape(10, 1)
    stream_export_to_shards(
        stream_rows(
            "arbitrary", weight, scale, scale_name="other-key", buffer_max_size_bytes=70
        ),
        str(tmp_path),
    )
    reader = BoundedTensorReader(str(tmp_path))
    target, target_scale = torch.empty_like(weight[3:8]), torch.empty_like(scale[3:8])
    receiver = RowReceiver("arbitrary", 10, 3, target, target_scale)
    for chunk in reader.rows("arbitrary", 2048, "other-key"):
        receiver.copy(chunk)
    receiver.finish()
    assert torch.equal(target, weight[3:8])
    assert torch.equal(target_scale, scale[3:8])


def test_invalid_scale_never_writes_weight():
    weight, scale = torch.zeros(3, 32), torch.zeros(3, 1)
    receiver = RowReceiver("x", 3, 0, weight, scale)
    with pytest.raises(ValueError, match="shape"):
        receiver.copy(RowChunk("x", 0, 3, torch.ones_like(weight), torch.ones(2, 1)))
    assert not weight.any() and not scale.any()
    with pytest.raises(ValueError, match="incomplete"):
        receiver.finish()


def _two_rank_worker(rank, rendezvous, path):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        boundaries = (0, 7, 19)
        weight = torch.arange(19 * 32).reshape(19, 32).float()
        scale = torch.arange(19).reshape(19, 1).to(torch.uint8)
        a, b = boundaries[rank : rank + 2]
        trace = []

        def records():
            for chunk in stream_rows(
                "rows",
                weight[a:b],
                scale[a:b],
                scale_name="scales",
                boundaries=boundaries,
                group=dist.group.WORLD,
                buffer_max_size_bytes=512,
            ):
                trace.append((chunk.offset, chunk.weight.shape[0]))
                yield chunk

        stream_export_to_shards(records(), path)
        peers = [None, None]
        dist.all_gather_object(peers, trace)
        assert peers[0] == peers[1]
        reader = BoundedTensorReader(path)
        assert torch.equal(reader.read("rows", 65536), weight)
        assert torch.equal(reader.read("scales", 65536), scale)
    finally:
        dist.destroy_process_group()


def test_nonzero_rank_drains_same_stream(tmp_path):
    mp.spawn(
        _two_rank_worker,
        args=(f"file://{tmp_path}/init", str(tmp_path / "out")),
        nprocs=2,
        join=True,
    )


def test_raw_archival_passthrough_preserves_bytes(tmp_path):
    from megatron.lite.primitive.ckpt.hf_weights import export_raw_tensors

    raw = torch.arange(320).to(torch.uint8).reshape(10, 32)
    first, second = tmp_path / 'first', tmp_path / 'second'
    stream_export_to_shards(
        stream_rows('uninterpreted', raw, buffer_max_size_bytes=64), str(first)
    )
    reader = BoundedTensorReader(str(first))
    stream_export_to_shards(
        export_raw_tensors(reader, reader.keys(), buffer_max_size_bytes=4096),
        str(second),
    )
    assert torch.equal(
        BoundedTensorReader(str(second)).read('uninterpreted', 4096), raw
    )


def test_byte_weight_and_float_scale_need_aligned_payload():
    weight, scale = torch.ones(7, 3, dtype=torch.uint8), torch.ones(7, 1)
    receiver = RowReceiver(
        'odd-width', 7, 0, torch.empty_like(weight), torch.empty_like(scale)
    )
    for chunk in stream_rows(
        'odd-width', weight, scale, scale_name='scale', buffer_max_size_bytes=19
    ):
        receiver.copy(chunk)
    receiver.finish()
    assert torch.equal(receiver.weight, weight)
    assert torch.equal(receiver.scale, scale)
