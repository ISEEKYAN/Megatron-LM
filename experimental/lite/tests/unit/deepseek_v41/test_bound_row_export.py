# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real bound exporter: independent HF bytes and measured staging limits."""
import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from tensor_allocations import AllocationPeak
from test_redo_parity import release_config


def make_model(path, trainable):
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

    config = release_config().to_hf_dict()
    from megatron.lite.primitive.modules.engram_lookup import prime_buckets

    config['text_config']['engram_vocab_size'] = 4096
    config['text_config']['engram_num_embeddings'] = (
        prime_buckets([1, 14], 3, 1, 4096).flatten(1).sum(1).tolist()
    )
    model = protocol.build_model(
        DeepseekV41Config(config),
        impl_cfg=protocol.ImplConfig(
            device='cpu',
            dtype=torch.float32,
            quantized=False,
            token_map=list(range(64)),
            trainable_engram=trainable,
        ),
    ).chunks[0]
    path.mkdir(parents=True)
    save_file(
        {name: torch.arange(7, dtype=torch.uint8) for name in model.archival_bindings},
        str(path / 'model.safetensors'),
    )
    model.archival_store = SafeTensorReader(str(path))
    model.archival_keys = sorted(model.archival_bindings)
    for layer in (1, 14):
        table = model.layers[layer].engram.embed
        rows = len(table.weight)
        if trainable:
            with torch.no_grad():
                table.master.copy_((1.3 * 2.0 ** (torch.arange(rows) % 5))[:, None])
        else:
            table.weight.view(torch.uint8).copy_(
                (torch.arange(rows)[:, None] + torch.arange(32))
                .remainder(127)
                .to(torch.uint8)
            )
            table.scale.view(torch.uint8).copy_(
                (100 + torch.arange(rows) % 30).to(torch.uint8)[:, None]
            )
    return model


def read_hf(path):
    index = path / 'model.safetensors.index.json'
    names = (
        set(json.loads(index.read_text())['weight_map'].values())
        if index.exists()
        else ['model.safetensors']
    )
    return {
        name: tensor
        for shard in names
        for name, tensor in load_file(str(path / shard)).items()
    }


@pytest.mark.parametrize('trainable', [False, True])
def test_bound_save_streams_rows_and_exact_masters_under_budget(
    v41_core_te, tmp_path, trainable
):
    from megatron.lite.model.deepseek_v41.lite import checkpoint

    model = make_model(tmp_path / 'archive', trainable)
    inputs = list(model.parameters()) + list(model.buffers())
    with AllocationPeak(inputs) as meter:
        checkpoint.save_model(model, tmp_path / 'saved', buffer_max_size_bytes=262144)
    assert meter.peak <= 262144, meter.peak
    weights = read_hf(tmp_path / 'saved')
    masters = read_hf(tmp_path / 'saved' / 'mlite_masters')
    for layer in (1, 14):
        name = f'layers.{layer}.engram.embed'
        table = model.layers[layer].engram.embed
        expected = (
            torch.full_like(table.weight.view(torch.uint8), 0x7A)
            if trainable
            else table.weight.view(torch.uint8)
        )
        scales = (
            (119 + torch.arange(len(table.weight)) % 5).to(torch.uint8)[:, None]
            if trainable
            else table.scale.view(torch.uint8)
        )
        assert torch.equal(weights[name + '.weight'].view(torch.uint8), expected)
        assert torch.equal(weights[name + '.scale'].view(torch.uint8), scales)
        if trainable:
            assert torch.equal(masters[name + '.weight'], table.master)
        else:
            assert name + '.weight' not in masters
    print(
        'BOUND_ROW_MEMORY_RESULT '
        + json.dumps(dict(trainable=trainable, budget=262144, peak=meter.peak))
    )


def test_oversized_rows_require_explicit_row_consumer(v41_core_te, tmp_path):
    from megatron.lite.model.deepseek_v41.lite import checkpoint

    model = make_model(tmp_path / 'archive', True)
    with pytest.raises(NotImplementedError, match='ROW_STREAM_REQUIRED'):
        next(checkpoint.export_checkpoint(model, buffer_max_size_bytes=262144))


def _bound_save_worker(rank, rendezvous, output, trainable):
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist
    from megatron.lite.model.deepseek_v41.lite import checkpoint
    from megatron.lite.primitive.modules.engram_lookup import RowLookup

    path = Path(output)
    torch.cuda.set_device(rank)
    model = make_model(path / f'archive-{rank}', trainable)
    expected = {}
    for layer in (1, 14):
        table = model.layers[layer].engram.embed
        expected[layer] = (
            table.weight.clone(),
            table.scale.clone(),
            None if table.master is None else table.master.detach().clone(),
        )
    dist.init_process_group(
        'nccl',
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        for layer in (1, 14):
            table = model.layers[layer].engram.embed
            rows = len(table.weight)
            boundaries = (0, rows // 2, rows)
            a, b = boundaries[rank : rank + 2]
            table.weight = table.weight[a:b].cuda()
            table.scale = table.scale[a:b].cuda()
            if trainable:
                table.master = torch.nn.Parameter(table.master[a:b].cuda())
            table.lookup = RowLookup(boundaries, dist.group.WORLD)
        table = model.layers[14].engram.embed
        local = table.master if trainable else table.weight.view(torch.uint8)
        # Measure the original padded all-gather + cat, including its live copies.
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        maximum = max(
            b - a for a, b in zip(table.lookup.boundaries, table.lookup.boundaries[1:])
        )
        padded = local.new_zeros(maximum, local.shape[1])
        padded[: len(local)].copy_(local)
        gathered = [torch.empty_like(padded) for _ in range(2)]
        dist.all_gather(gathered, padded)
        full = torch.cat(
            [
                part[: b - a]
                for part, a, b in zip(
                    gathered, table.lookup.boundaries, table.lookup.boundaries[1:]
                )
            ]
        )
        torch.cuda.synchronize()
        old_peak = torch.cuda.max_memory_allocated() - baseline
        del padded, gathered, full
        budget = 262144
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        checkpoint.save_model(model, path / 'saved', buffer_max_size_bytes=budget)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        assert 0 < peak <= budget < old_peak, (rank, peak, budget, old_peak)
        if rank == 0:
            weights, masters = read_hf(path / 'saved'), read_hf(
                path / 'saved' / 'mlite_masters'
            )
            for layer, (weight, scale, master) in expected.items():
                name = f'layers.{layer}.engram.embed'
                w = (
                    torch.full_like(weight.view(torch.uint8), 0x7A)
                    if trainable
                    else weight.view(torch.uint8)
                )
                s = (
                    (119 + torch.arange(len(weight)) % 5).to(torch.uint8)[:, None]
                    if trainable
                    else scale.view(torch.uint8)
                )
                assert torch.equal(weights[name + '.weight'].view(torch.uint8), w)
                assert torch.equal(weights[name + '.scale'].view(torch.uint8), s)
                if trainable:
                    assert torch.equal(masters[name + '.weight'], master)
                else:
                    assert name + '.weight' not in masters
        (path / f'bound-{rank}.json').write_text(
            json.dumps(
                dict(
                    rank=rank,
                    trainable=trainable,
                    budget=budget,
                    old_peak=old_peak,
                    peak=peak,
                )
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(2)
@pytest.mark.parametrize('trainable', [False, True])
def test_real_sharded_bound_save_cuda_peak_and_independent_hf(tmp_path, trainable):
    import torch.multiprocessing as mp

    mp.spawn(
        _bound_save_worker,
        args=(f'file://{tmp_path}/init', str(tmp_path), trainable),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        result = json.loads((tmp_path / f'bound-{rank}.json').read_text())
        assert result['rank'] == rank and result['trainable'] == trainable
        assert 0 < result['peak'] <= result['budget'] < result['old_peak']
        print('BOUND_ROW_CUDA_RESULT ' + json.dumps(result))
