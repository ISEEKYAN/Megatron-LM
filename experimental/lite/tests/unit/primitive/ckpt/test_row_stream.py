# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded row transport against independent global byte arrays."""
import weakref

import pytest
import torch
from megatron.lite.primitive.ckpt.row_stream import RowReceiver, stream_rows
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


class AllocationPeak(TorchDispatchMode):
    """Observe live ATen output storage bytes, excluding pre-existing inputs."""

    def __init__(self, inputs):
        self.existing = {x.untyped_storage()._cdata for x in inputs}
        self.live = {}
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for tensor in tree_leaves(result):
            if not isinstance(tensor, torch.Tensor):
                continue
            storage = tensor.untyped_storage()
            key = storage._cdata
            if key not in self.existing:
                self.live[key] = (weakref.ref(storage), storage.nbytes())
        self.live = {
            key: value for key, value in self.live.items() if value[0]() is not None
        }
        self.peak = max(self.peak, sum(size for _, size in self.live.values()))
        return result


@pytest.mark.parametrize('span', [(19, 93), (0, 7), (110, 113), (47, 47)])
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('master', [False, True])
def test_stream_matches_global_rows_and_bounds_live_allocations(
    monkeypatch, rank, master, span
):
    boundaries = (0, 47, 47, 113)  # uneven owners, including one empty owner
    weight = (torch.arange(113 * 32) % 127).to(torch.uint8).reshape(113, 32)
    weight = weight.float() if master else weight.view(torch.float8_e4m3fn)
    scale = (
        None
        if master
        else (100 + torch.arange(113) % 30)
        .to(torch.uint8)
        .unsqueeze(1)
        .view(torch.float8_e8m0fnu)
    )
    begin, end = boundaries[rank : rank + 2]
    local = weight[begin:end]
    local_scale = None if master else scale[begin:end]
    group, calls = object(), []
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda g: rank)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda g: 3)
    monkeypatch.setattr(torch.distributed, 'get_global_rank', lambda g, r: r + 10)
    offsets = list(boundaries[:-1])

    def broadcast(buffer, src, group):
        owner = src - 10
        row_bytes = weight[0].numel() * weight.element_size() + (0 if master else 1)
        count = buffer.numel() // row_bytes
        start = offsets[owner]
        w = weight[start : start + count].view(torch.uint8).flatten()
        buffer[: w.numel()].copy_(w)
        if not master:
            buffer[w.numel() :].copy_(
                scale[start : start + count].view(torch.uint8).flatten()
            )
        calls.append((src, start, count))
        offsets[owner] += count

    monkeypatch.setattr(torch.distributed, 'broadcast', broadcast)
    first, last = span
    target = torch.empty_like(weight[first:last])
    scales = None if master else torch.empty_like(scale[first:last])
    receiver = RowReceiver('table.weight', 113, first, target, scales)
    inputs = [weight, local, target] + ([] if master else [scale, local_scale, scales])
    with AllocationPeak(inputs) as allocation:
        for chunk in stream_rows(
            'table.weight',
            local,
            local_scale,
            boundaries=boundaries,
            group=group,
            buffer_max_size_bytes=1024,
        ):
            receiver.copy(chunk)
        receiver.finish()
    assert allocation.peak <= 1024, allocation.peak
    assert torch.equal(target.view(torch.uint8), weight[first:last].view(torch.uint8))
    if not master:
        assert torch.equal(
            scales.view(torch.uint8), scale[first:last].view(torch.uint8)
        )
    expected = []
    rows = 1024 // (128 if master else 33)
    for owner, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        expected.extend(
            (owner + 10, start, min(rows, b - start)) for start in range(a, b, rows)
        )
    assert calls == expected


def test_receiver_rejects_gaps_and_missing_tail():
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    target = torch.zeros(3, 32)
    receiver = RowReceiver('x', 7, 2, target)
    with pytest.raises(ValueError, match='order'):
        receiver.copy(RowChunk('x', 1, 7, torch.ones(2, 32)))
    assert torch.equal(target, torch.zeros_like(target))
    with pytest.raises(ValueError, match='incomplete'):
        receiver.finish()


def test_buffer_smaller_than_one_row_fails_before_collective(monkeypatch):
    monkeypatch.setattr(
        torch.distributed,
        'broadcast',
        lambda *a, **k: pytest.fail('collective before budget validation'),
    )
    with pytest.raises(ValueError, match='one row'):
        list(stream_rows('x', torch.ones(3, 32), buffer_max_size_bytes=64))


@pytest.mark.parametrize('master', [False, True])
def test_stream_writer_is_bounded_and_readable_by_safetensors(tmp_path, master):
    from megatron.lite.primitive.ckpt.hf_weights import stream_export_to_shards
    from safetensors.torch import load_file

    weight = torch.arange(4096 * 32, dtype=torch.float32).reshape(4096, 32)
    weight = (
        weight if master else (weight.to(torch.uint8) % 127).view(torch.float8_e4m3fn)
    )
    scale = (
        None
        if master
        else torch.full((4096, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    )
    with AllocationPeak([weight] + ([] if master else [scale])) as allocation:
        stream_export_to_shards(
            stream_rows('table.weight', weight, scale, buffer_max_size_bytes=4096),
            str(tmp_path),
            shard_size_bytes=4096,
        )
    assert allocation.peak <= 4096
    result = load_file(str(tmp_path / 'model.safetensors'))
    assert torch.equal(
        result['table.weight'].view(torch.uint8), weight.view(torch.uint8)
    )
    if not master:
        assert torch.equal(
            result['table.scale'].view(torch.uint8), scale.view(torch.uint8)
        )


def test_allocation_oracle_kills_full_table_materialization():
    """Restore the old gather/cat shape and show the actual bound assertion fails."""
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    source = torch.ones(4096, 32)
    receiver = RowReceiver('x', 4096, 0, source)
    with AllocationPeak([source]) as allocation:
        chunks = [torch.empty_like(source[:2048]) for _ in range(2)]
        for i, chunk in enumerate(chunks):
            chunk.copy_(source[i * 2048 : (i + 1) * 2048])
        whole = torch.cat(chunks)
        receiver.copy(RowChunk('x', 0, 4096, whole))
    with pytest.raises(AssertionError):
        assert allocation.peak <= 4096
    assert allocation.peak == 2 * source.numel() * source.element_size()


def _distributed_peak_worker(rank, rendezvous, output_dir, master, quantize):
    import json
    from pathlib import Path

    import torch.distributed as dist
    from megatron.lite.primitive.ckpt.row_stream import stream_rows

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2)
    try:
        boundaries = (0, 65536, 131073)
        total, width = boundaries[-1], 256
        dtype = torch.float32 if master else torch.float8_e4m3fn
        local = torch.full(
            (boundaries[rank + 1] - boundaries[rank], width),
            (rank + 1.0) * (1.3 if quantize else 1.0),
            device='cuda',
            dtype=torch.float32,
        ).to(dtype)
        scale = (
            None
            if master
            else torch.full(
                (len(local), width // 32), 127 + rank, device='cuda', dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
        )
        # Hash-head spans cross both transport blocks and owner boundaries.
        start, end = (30000, 70003) if rank == 0 else (70003, total)
        target = torch.empty(
            (end - start, width),
            dtype=torch.float8_e4m3fn if quantize else dtype,
            device='cuda',
        )
        scales = (
            None
            if master and not quantize
            else torch.empty(
                (end - start, width // 32), dtype=torch.uint8, device='cuda'
            )
        )
        receiver = RowReceiver('table.weight', total, start, target, scales)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        # Original _export_rows algorithm, independently kept as the regression arm.
        value = local.view(torch.uint8) if not master else local
        padded = value.new_zeros(65537, width)
        padded[: len(value)].copy_(value)
        chunks = [torch.empty_like(padded) for _ in range(2)]
        dist.all_gather(chunks, padded)
        full = torch.cat([chunks[0][:65536], chunks[1][:65537]])
        torch.cuda.synchronize()
        old_peak = torch.cuda.max_memory_allocated() - baseline
        del value, padded, chunks, full
        readings = []
        for budget in (512 * 1024, 1024 * 1024):
            receiver.next_row = 0
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            sequence, trace = [], []
            broadcast = dist.broadcast

            def traced_broadcast(tensor, *, src, group):
                trace.append(
                    (src, tensor.numel(), str(tensor.dtype), group is dist.group.WORLD)
                )
                return broadcast(tensor, src=src, group=group)

            dist.broadcast = traced_broadcast
            for chunk in stream_rows(
                'table.weight',
                local,
                scale,
                boundaries=boundaries,
                group=dist.group.WORLD,
                buffer_max_size_bytes=budget,
                quantize=quantize,
            ):
                sequence.append((chunk.offset, len(chunk.weight)))
                receiver.copy(chunk)
            del chunk
            dist.broadcast = broadcast
            receiver.finish()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - baseline
            assert peak <= budget, (rank, peak, budget)
            peers = [None, None]
            dist.all_gather_object(peers, (sequence, trace))
            assert peers[0] == peers[1]
            row_bytes = width + width // 32 if not master or quantize else width * 4
            assert trace == [
                (0 if offset < 65536 else 1, count * row_bytes, 'torch.uint8', True)
                for offset, count in sequence
            ]
            for owner in range(2):
                a, b = max(start, boundaries[owner]), min(end, boundaries[owner + 1])
                if b <= a:
                    continue
                expected = (
                    torch.full((b - a, width), 0x7A, dtype=torch.uint8, device='cuda')
                    if quantize
                    else torch.full((b - a, width), owner + 1.0, device='cuda').to(
                        dtype
                    )
                )
                assert torch.equal(
                    target[a - start : b - start].view(torch.uint8),
                    expected.view(torch.uint8),
                )
                del expected
                if scales is not None:
                    assert torch.equal(
                        scales[a - start : b - start],
                        torch.full_like(
                            scales[a - start : b - start],
                            (119 if quantize else 127) + owner,
                        ),
                    )
            readings.append(dict(budget=budget, peak=peak, blocks=len(sequence)))
        assert old_peak > max(r['budget'] for r in readings)
        Path(output_dir, f'peak-{master}-{rank}.json').write_text(
            json.dumps(
                dict(
                    rank=rank,
                    master=master,
                    quantize=quantize,
                    old_peak=old_peak,
                    readings=readings,
                )
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(2)
@pytest.mark.parametrize(
    'master, quantize', [(False, False), (True, False), (True, True)]
)
def test_real_nccl_sequence_bytes_and_cuda_peak(tmp_path, master, quantize):
    import torch.multiprocessing as mp

    mp.spawn(
        _distributed_peak_worker,
        args=(f'file://{tmp_path}/init', str(tmp_path), master, quantize),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        print(
            'ROW_STREAM_MEMORY_RESULT '
            + (tmp_path / f'peak-{master}-{rank}.json').read_text()
        )


def test_trainable_rows_quantize_with_bounded_workspace_and_external_bytes():
    values = (
        (1.3 * 2.0 ** (torch.arange(1024) % 5))
        .unsqueeze(1)
        .expand(1024, 32)
        .contiguous()
    )
    target = torch.empty(1024, 32, dtype=torch.float8_e4m3fn)
    scales = torch.empty(1024, 1, dtype=torch.uint8)
    receiver = RowReceiver('x.weight', 1024, 0, target, scales)
    with AllocationPeak([values, target, scales]) as allocation:
        for chunk in stream_rows(
            'x.weight', values, quantize=True, buffer_max_size_bytes=16384
        ):
            receiver.copy(chunk)
        receiver.finish()
    assert allocation.peak <= 16384
    # Independent E4M3 320 * 2^-8 and raw E8M0 exponent bytes for each row.
    assert torch.equal(
        target.view(torch.uint8), torch.full((1024, 32), 0x7A, dtype=torch.uint8)
    )
    assert torch.equal(
        scales, (119 + torch.arange(1024) % 5).to(torch.uint8).unsqueeze(1)
    )


def test_bad_scale_is_rejected_before_weight_copy():
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    target, scale = torch.zeros(3, 32), torch.zeros(3, 1)
    receiver = RowReceiver('x.weight', 3, 0, target, scale)
    with pytest.raises(ValueError, match='shape'):
        receiver.copy(RowChunk('x.weight', 0, 3, torch.ones(3, 32), torch.ones(2, 1)))
    assert torch.equal(target, torch.zeros_like(target))


def test_writer_rejects_missing_tail_and_can_mix_normal_tensors(tmp_path):
    import json

    from megatron.lite.primitive.ckpt.hf_weights import stream_export_to_shards
    from megatron.lite.primitive.ckpt.row_stream import RowChunk
    from safetensors.torch import load_file

    with pytest.raises(ValueError, match='incomplete'):
        stream_export_to_shards(
            iter([RowChunk('x.weight', 0, 5, torch.ones(1, 32))]), str(tmp_path)
        )

    weight = torch.arange(128 * 32, dtype=torch.float32).reshape(128, 32)

    def records():
        yield 'before', torch.tensor([7])
        yield from stream_rows('x.weight', weight, buffer_max_size_bytes=1024)
        yield from stream_rows('y.weight', weight, buffer_max_size_bytes=1024)
        yield 'after', torch.tensor([9])

    stream_export_to_shards(records(), str(tmp_path), shard_size_bytes=1024)
    index = json.loads((tmp_path / 'model.safetensors.index.json').read_text())
    assert list(index['weight_map']) == ['before', 'x.weight', 'y.weight', 'after']
    for name in ('x.weight', 'y.weight'):
        assert torch.equal(
            load_file(str(tmp_path / index['weight_map'][name]))[name], weight
        )
