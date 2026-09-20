# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment bytes, independently inspected rather than MLite round trips."""
import pytest
import torch
from tensor_allocations import AllocationPeak
from test_bound_row_export import make_model, read_hf


@pytest.mark.parametrize('trainable', [False, True])
def test_online_and_saved_deployment_bytes(v41_core_te, tmp_path, trainable):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.resync import decode_transport
    from megatron.lite.primitive.ckpt.row_stream import RowChunk, RowReceiver

    model = make_model(tmp_path / 'archive', trainable)
    with torch.no_grad():
        model.tensor_bindings['layers.0.ffn.experts.0.w1.weight'].tensor.copy_(
            (1.3 * (1 - 2 * (torch.arange(32) % 2)))[None, :]
        )
    from megatron.lite.runtime.contracts.weights import ResyncFormat

    opts = dict(
        target=ResyncFormat.parse('mxfp4').value,
        resync_config={'expert_dtype': 'fp4'},
        buffer_max_size_bytes=524288,
    )
    with AllocationPeak(list(model.parameters()) + list(model.buffers())) as save_meter:
        protocol.save_hf_weights(
            [model], tmp_path / 'saved', model.config, model.ps, **opts
        )
    assert save_meter.peak <= 524288, save_meter.peak
    expected = read_hf(tmp_path / 'saved')
    # Consumer-owned output is allocated before metering, as in a rollout model.
    received = {n: torch.empty_like(t) for n, t in expected.items()}
    receivers, names = {}, set()
    inputs = list(model.parameters()) + list(model.buffers()) + list(received.values())
    with AllocationPeak(inputs) as meter:
        for pair in protocol.export_hf_weights([model], model.config, model.ps, **opts):
            item = decode_transport(*pair)
            if item is None:
                continue
            if isinstance(item, RowChunk):
                name = item.name
                if name not in receivers:
                    receivers[name] = RowReceiver(
                        name,
                        item.total_rows,
                        0,
                        received[name],
                        received[name[:-6] + 'scale'],
                    )
                receivers[name].copy(item)
                names.update((name, name[:-6] + 'scale'))
            else:
                name, tensor = item
                received[name].copy_(tensor)
                names.add(name)
            del item, pair
    assert meter.peak <= 524288, meter.peak
    for receiver in receivers.values():
        receiver.finish()
    assert names == expected.keys()
    for name, tensor in expected.items():
        assert received[name].dtype == tensor.dtype
        assert torch.equal(
            received[name].view(torch.uint8), tensor.view(torch.uint8)
        ), name
    # Independent external layout: official expert weights pack two E2M1
    # values per signed byte; group32 scales use raw E8M0, not float masters.
    expert = 'layers.0.ffn.experts.0.w1'
    assert expected[expert + '.weight'].dtype == torch.int8
    assert expected[expert + '.weight'].shape == (32, 16)
    assert expected[expert + '.scale'].dtype == torch.float8_e8m0fnu
    assert expected[expert + '.scale'].shape == (32, 1)
    assert torch.equal(
        expected[expert + '.weight'].view(torch.uint8),
        torch.full((32, 16), 0xF7, dtype=torch.uint8),
    )
    assert torch.equal(
        expected[expert + '.scale'].view(torch.uint8),
        torch.full((32, 1), 125, dtype=torch.uint8),
    )
    assert expected['embed.weight'].dtype == torch.bfloat16
    for name in (
        'layers.0.attn.attn_sink',
        'layers.0.ffn.gate.bias',
        'layers.0.hc_attn_fn',
    ):
        assert expected[name].dtype == torch.float32
        assert torch.equal(expected[name], model.tensor_bindings[name].tensor)
    print(
        f'RESYNC_EXPORT_MEMORY trainable={trainable} budget=524288 peak={meter.peak} save_peak={save_meter.peak}'
    )
    assert not (tmp_path / 'saved' / 'mlite_masters').exists()


@pytest.mark.parametrize(
    'options',
    [
        {'target': 'vllm', 'limit': 1},
        {'target': 'vllm', 'row_chunks': False},
        {'target': 'vllm', 'export_dtype': 'float16'},
        {'target': 'vllm', 'buffer_max_size_bytes': 65536},
    ],
)
def test_both_entries_reject_partial_or_incompatible_deployment(
    v41_core_te, tmp_path, options
):
    from megatron.lite.model.deepseek_v41.lite import protocol

    model = make_model(tmp_path / 'archive', False)
    for entry in ('save', 'online'):
        with pytest.raises(ValueError):
            if entry == 'save':
                protocol.save_hf_weights(
                    [model], tmp_path / 'save', model.config, model.ps, **options
                )
            else:
                list(
                    protocol.export_hf_weights(
                        [model], model.config, model.ps, **options
                    )
                )


def test_transport_preserves_all_integer_and_exponent_bytes():
    from megatron.lite.model.deepseek_v41.lite.resync import (
        decode_transport,
        transport_weights,
    )

    raw = torch.arange(256).to(torch.uint8)
    for dtype in (torch.int8, torch.float8_e8m0fnu):
        name, payload = next(transport_weights([('weight', raw.view(dtype))]))
        assert payload.dtype == torch.uint8 and torch.equal(payload, raw)
        # Emulate reuse after transport copied out of sender storage.
        actual_name, actual = decode_transport(name, payload.clone())
        assert actual_name == 'weight' and actual.dtype == dtype
        assert torch.equal(actual.view(torch.uint8), raw)


def test_rows_reject_missing_tail_and_out_of_order():
    from megatron.lite.model.deepseek_v41.lite.resync import (
        decode_transport,
        transport_weights,
    )
    from megatron.lite.primitive.ckpt.row_stream import RowReceiver, stream_rows

    weight = torch.zeros(40, 32, dtype=torch.float8_e4m3fn)
    scale = torch.full((40, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    receiver = RowReceiver(
        'table.weight',
        40,
        4,
        torch.empty_like(weight[:7]),
        torch.empty((7, 1), dtype=torch.uint8),
    )
    stream = transport_weights(
        stream_rows('table.weight', weight, scale, buffer_max_size_bytes=512)
    )
    name, payload = next(stream)
    chunk = decode_transport(name, payload)
    receiver.copy(chunk)
    with pytest.raises(ValueError, match='incomplete'):
        receiver.finish()
    with pytest.raises(ValueError, match='order mismatch'):
        receiver.copy(chunk)


def _resync_rows_worker(rank, rendezvous, output, trainable):
    import json
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.resync import decode_transport
    from megatron.lite.primitive.ckpt.row_stream import RowChunk, RowReceiver
    from megatron.lite.primitive.modules.engram_lookup import RowLookup

    torch.cuda.set_device(rank)
    model = make_model(Path(output) / f'archive-{rank}', trainable)
    dist.init_process_group(
        'nccl',
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        sizes, destinations = {}, {}
        for layer in (1, 14):
            table = model.layers[layer].engram.embed
            rows = len(table.weight)
            sizes[layer] = rows
            boundaries = (0, rows // 2, rows)
            start, end = boundaries[rank : rank + 2]
            table.weight = table.weight[start:end].cuda()
            table.scale = table.scale[start:end].cuda()
            if trainable:
                table.master = torch.nn.Parameter(table.master[start:end].cuda())
            table.lookup = RowLookup(boundaries, dist.group.WORLD)
            # Rollout ownership deliberately differs from trainer's row split.
            a, b = (0, rows // 3) if rank == 0 else (rows // 3, rows)
            destinations[layer] = (
                a,
                torch.empty(b - a, 32, device='cuda', dtype=torch.float8_e4m3fn),
                torch.empty(b - a, 1, device='cuda', dtype=torch.uint8),
            )
        budget = 524288
        peaks, chunks_seen = [], []
        for generation in range(2):
            if generation:
                with torch.no_grad():
                    for layer in (1, 14):
                        table = model.layers[layer].engram.embed
                        if trainable:
                            table.master.mul_(2)
                        else:
                            table.scale.view(torch.uint8).add_(1)
            receivers = {
                f'layers.{layer}.engram.embed.weight': RowReceiver(
                    f'layers.{layer}.engram.embed.weight',
                    sizes[layer],
                    start,
                    weight,
                    scale,
                )
                for layer, (start, weight, scale) in destinations.items()
            }
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            count, terminated = 0, False
            for pair in protocol.export_hf_weights(
                [model],
                model.config,
                model.ps,
                target='mxfp4',
                buffer_max_size_bytes=budget,
            ):
                item = decode_transport(*pair)
                if isinstance(item, RowChunk):
                    receivers[item.name].copy(item)
                    count += 1
                elif item is None:
                    terminated = True
                del item, pair
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - baseline
            assert 0 < peak <= budget, (rank, trainable, peak, budget)
            assert count > 2 and terminated
            for receiver in receivers.values():
                receiver.finish()
            for layer, (start, weight, scale) in destinations.items():
                row = torch.arange(start, start + len(weight), device='cuda')
                expected_weight = (
                    torch.full_like(weight.view(torch.uint8), 0x7A)
                    if trainable
                    else (row[:, None] + torch.arange(32, device='cuda'))
                    .remainder(127)
                    .to(torch.uint8)
                )
                expected_scale = (
                    (119 if trainable else 100)
                    + row % (5 if trainable else 30)
                    + generation
                ).to(torch.uint8)[:, None]
                assert torch.equal(weight.view(torch.uint8), expected_weight)
                assert torch.equal(scale, expected_scale)
            peaks.append(peak)
            chunks_seen.append(count)
        result = dict(
            rank=rank,
            trainable=trainable,
            generations=2,
            peaks=peaks,
            budget=budget,
            row_chunks=chunks_seen,
            bitwise=True,
        )
        (Path(output) / f'resync-{rank}.json').write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(2)
@pytest.mark.parametrize('trainable', [False, True])
def test_resync_real_two_rank_rows_and_second_generation(tmp_path, trainable):
    import json

    import torch.multiprocessing as mp

    mp.spawn(
        _resync_rows_worker,
        args=(f'file://{tmp_path}/init', str(tmp_path), trainable),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        result = json.loads((tmp_path / f'resync-{rank}.json').read_text())
        assert result['bitwise'] and result['generations'] == 2
        print('DS41_RESYNC_ROWS_RESULT ' + json.dumps(result))
