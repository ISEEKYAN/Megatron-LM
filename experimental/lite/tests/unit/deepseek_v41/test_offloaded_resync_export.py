# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Resync from offloaded masters equals the resync of a reloaded model.

The engine skips the whole-model reload when the DS4.1 PP1 exporter takes an
export_device: each binding is copied there just before use, frozen Engram
tables stream from host, and experts are broadcast from the export device.
Both paths must emit the same transport stream, item by item and byte for
byte, over two generations (initial census, then reuse), on every EP rank;
the export must not repoint any parameter or buffer storage.
"""

import hashlib
import json
import os
import runpy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

LAYERS = 4


def _prefix_config(experts):
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    release = runpy.run_path(str(Path(__file__).with_name('test_redo_parity.py')))
    hf = release['release_config']().to_hf_dict()
    text = hf['text_config']
    text.update(
        num_hidden_layers=LAYERS,
        compress_ratios=text['compress_ratios'][:LAYERS] + text['compress_ratios'][-3:],
        candidate_source_layer_id=-1,
        kv_source_layer_ids=[2],
        index_source_layer_ids=[2],
        engram_layer_ids=[1],
        engram_num_embeddings=text['engram_num_embeddings'][:1],
        n_routed_experts=experts,
    )
    return DeepseekV41Config(hf)


def _digest(pair):
    from megatron.lite.model.deepseek_v41.lite.resync import decode_transport
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    name, tensor = pair
    item = decode_transport(name, tensor)
    hasher = hashlib.sha256(name.encode())
    hasher.update(str((tuple(tensor.shape), tensor.dtype)).encode())
    hasher.update(tensor.detach().reshape(-1).cpu().view(torch.uint8).numpy())
    kind = type(item).__name__ if item is not None else 'end'
    if isinstance(item, RowChunk):
        kind = f'RowChunk:{item.name}:{item.offset}:{item.total_rows}'
    return [kind, hasher.hexdigest()]


def _generations(model, export_device):
    from megatron.lite.model.deepseek_v41.lite.checkpoint import export_hf_weights

    if hasattr(model, '_resync_frozen_tables'):
        del model._resync_frozen_tables
    streams = []
    for _ in range(2):
        options = {}
        if export_device is not None:
            options['export_device'] = export_device
        streams.append(
            [
                _digest(pair)
                for pair in export_hf_weights(
                    [model],
                    model.config,
                    model.ps,
                    target='mxfp4',
                    resync_config={'expert_dtype': 'fp4', 'freeze_engram': True},
                    include_archival=False,
                    buffer_max_size_bytes=1 << 20,
                    **options,
                )
            ]
        )
    return streams


def _storages(model):
    tensors = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))
    return {
        name: (tensor.device.type, tensor.data_ptr())
        for name, tensor in tensors.items()
    }


def _worker(rank, directory, ep, experts, backend):
    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    monkeypatch = pytest.MonkeyPatch()
    fixtures['transformer_engine_import_stub'].__wrapped__(monkeypatch)()
    import megatron.lite.primitive.ckpt.hf_weights as hf_weights
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    gpu = backend == 'nccl'
    if gpu:
        torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        backend,
        init_method=f'file://{directory}/store',
        rank=rank,
        world_size=ep,
        timeout=timedelta(seconds=300),
    )
    record = {}
    try:
        impl = protocol.ImplConfig(
            device='cuda' if gpu else 'cpu',
            dtype=torch.float32,
            token_map=list(range(64)),
            trainable_engram=False,
            # Sharded Engram lookups are CUDA-only; the CPU run replicates the
            # frozen table and the GPU run row-shards it over NCCL.
            shard_engram=gpu,
            parallel=ParallelConfig(ep=ep),
        )
        torch.manual_seed(351)
        bundle = protocol.build_model(
            _prefix_config(experts), impl_cfg=replace(impl, quantized=False)
        )
        model = bundle.chunks[0]
        assert model.ps.ep_size == ep
        tables = [
            b.tensor.data_ptr()
            for b in model.tensor_bindings.values()
            if b.role == 'engram_table'
        ]
        assert tables
        # Reference: today's path, every tensor resident where the exporter
        # reads it (the reloaded model on GPU; the CPU model as is).
        reference = _generations(model, None)
        export_device = torch.device('cuda', rank) if gpu else torch.device('cpu')
        if gpu:
            # Offload like param_offload; parameters keep pointing at host.
            model.to('cpu')
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        storages = _storages(model)
        moved = []
        original = hf_weights._on_export_device

        def spy(tensor, device):
            moved.append(tensor.data_ptr())
            return original(tensor, device)

        monkeypatch.setattr(hf_weights, '_on_export_device', spy)
        streamed = _generations(model, export_device)
        if gpu:
            torch.cuda.synchronize()
            record['memory'] = dict(
                before=before,
                peak=torch.cuda.max_memory_allocated(),
                after=torch.cuda.memory_allocated(),
            )
        record.update(
            equal=reference == streamed,
            items=[len(stream) for stream in streamed],
            first_unequal=next(
                (
                    (g, i, a, b)
                    for g, (ra, rb) in enumerate(zip(reference, streamed))
                    for i, (a, b) in enumerate(zip(ra, rb))
                    if a != b
                ),
                None,
            ),
            storages_unchanged=storages == _storages(model),
            moved=len(moved),
            frozen_moved=bool(set(tables) & set(moved)),
            row_chunks=sum(kind.startswith('RowChunk') for kind, _ in streamed[0]),
        )
    except Exception:
        import traceback

        record['error'] = traceback.format_exc()[-6000:]
    finally:
        Path(f'{directory}/record-{rank}.json').write_text(json.dumps(record))
        dist.destroy_process_group()
        monkeypatch.undo()


def _run(directory, ep, experts, backend):
    mp.start_processes(
        _worker,
        args=(str(directory), ep, experts, backend),
        nprocs=ep,
        join=True,
        start_method='spawn',
    )
    records = [
        json.loads((Path(directory) / f'record-{r}.json').read_text())
        for r in range(ep)
    ]
    print('OFFLOADED_RESYNC_EXPORT', backend, ep, json.dumps(records))
    assert not [r['error'] for r in records if 'error' in r], records
    for record in records:
        assert record['equal'], record['first_unequal']
        assert record['storages_unchanged']
        assert record['moved'] > 0 and not record['frozen_moved']
        # The reuse generation skips the frozen tables the first one sent.
        assert record['row_chunks'] > 0
        assert record['items'][0] > record['items'][1] > 0
    return records


@pytest.mark.parametrize('ep,experts', [(2, 2), (4, 4)])
def test_offloaded_export_stream_equals_reloaded_export(
    v41_core_te, tmp_path, ep, experts
):
    _run(tmp_path, ep, experts, 'gloo')


@pytest.mark.gpus(4, min_architecture='blackwell')
def test_offloaded_export_on_gpu_returns_to_offloaded_memory(tmp_path):
    output = Path(os.environ.get('DS41_OFFLOAD_EXPORT_OUTPUT', tmp_path))
    output.mkdir(parents=True, exist_ok=True)
    records = _run(output, 4, 4, 'nccl')
    for record in records:
        memory = record['memory']
        # The export holds no device tensor once its stream is drained.
        assert memory['after'] == memory['before'], memory
        assert memory['peak'] > memory['before'], memory
