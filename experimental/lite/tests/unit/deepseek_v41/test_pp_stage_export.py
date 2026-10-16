# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Stage export combines frozen metadata and preserves borrowed payloads."""
import pytest
import torch


def test_pp_stream_copies_borrowed_codec_storage(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt import pipeline_stream

    ps = SimpleNamespace(
        pp_size=2, pp_rank=0, pp_global_ranks=[0, 1], pp_group=object()
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    def header(values, *, src, **kwargs):
        if src == 1:
            values[0] = []

    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', header)
    monkeypatch.setattr(torch.distributed, 'broadcast', lambda *args, **kwargs: None)
    storage = torch.empty(1, dtype=torch.uint8)

    def borrowed():
        for value in range(3):
            storage.fill_(value)
            yield str(value), storage

    result = list(
        pipeline_stream.stream_pipeline_tensors(borrowed(), ps, buffer_max_size_bytes=2)
    )
    assert [(n, int(t.item())) for n, t in result] == [('0', 0), ('1', 1), ('2', 2)]


@pytest.mark.parametrize('mismatch', [False, True])
def test_pp_export_publishes_one_generation_wide_frozen_header(
    v41_core_te, monkeypatch, mismatch
):
    from types import SimpleNamespace

    from megatron.lite.model.deepseek_v41.lite import checkpoint, resync
    from megatron.lite.primitive.ckpt import pipeline_stream

    ps = SimpleNamespace(pp_size=2, pp_rank=0, pp_is_last=False, pp_group=object())
    seen = []

    def export(*args, **kwargs):
        assert kwargs['include_archival'] is False
        yield resync.frozen_tables_transport({'first': ['hash-a']}, False)
        yield 'local', torch.tensor([3])

    def gather(result, local, *, group):
        assert group is ps.pp_group
        result[:] = [local, resync.FrozenTables(mismatch, {'second': ['hash-b']})]

    def stream(weights, state, **kwargs):
        seen.append(kwargs['buffer_max_size_bytes'])
        yield from weights
        yield 'remote', torch.tensor([7])

    monkeypatch.setattr(checkpoint, '_export_stage_hf_weights', export)
    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    monkeypatch.setattr(pipeline_stream, 'stream_pipeline_tensors', stream)
    call = lambda: list(
        checkpoint.export_hf_weights(
            [object()],
            object(),
            ps,
            target='mxfp4',
            resync_config={'freeze_engram': True},
            buffer_max_size_bytes=256,
        )
    )
    if mismatch:
        with pytest.raises(ValueError, match='generations disagree'):
            call()
        assert not seen
    else:
        result = call()
        first = resync.decode_transport(*result[0])
        assert first.manifest == {'first': ['hash-a'], 'second': ['hash-b']}
        assert first.reuse is False
        assert [name for name, _ in result[1:]] == ['local', 'remote']
        assert seen == [64]


def _pp_export_worker(rank, directory):
    import runpy
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    from megatron.lite.model.deepseek_v41.lite import checkpoint, resync
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/store',
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        ps = init_parallel(ParallelConfig(pp=2))
        original = checkpoint._export_stage_hf_weights
        for reuse in [False, True]:

            def stage_export(*args, **kwargs):
                if rank == 0:
                    assert kwargs['include_archival'] is False
                yield resync.frozen_tables_transport(
                    {f'table-{rank}': [f'hash-{rank}']}, reuse
                )
                payload = torch.empty(1, dtype=torch.int64)
                for index in range(3):
                    payload.fill_(rank * 10 + index)
                    yield f'layer-{rank}/weight-{index}', payload

            checkpoint._export_stage_hf_weights = stage_export
            result = list(
                checkpoint.export_hf_weights(
                    [object()],
                    object(),
                    ps,
                    target='mxfp4',
                    resync_config={'freeze_engram': True},
                    buffer_max_size_bytes=32,
                )
            )
            census = resync.decode_transport(*result[0])
            assert isinstance(census, resync.FrozenTables)
            assert census.reuse is reuse
            assert census.manifest == {'table-0': ['hash-0'], 'table-1': ['hash-1']}
            assert [(name, tensor.item()) for name, tensor in result[1:]] == [
                (f'layer-{stage}/weight-{index}', stage * 10 + index)
                for stage in range(2)
                for index in range(3)
            ]
        checkpoint._export_stage_hf_weights = original
        torch.save(
            {'rank': rank, 'complete_stream': True, 'initial_and_reuse': True},
            Path(directory) / f'rank-{rank}.pt',
        )
    finally:
        dist.destroy_process_group()


def test_real_pp_stream_merges_initial_and_reused_frozen_generations(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_pp_export_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(tmp_path / f'rank-{rank}.pt', weights_only=True)
        assert result['complete_stream'] and result['initial_and_reuse']
