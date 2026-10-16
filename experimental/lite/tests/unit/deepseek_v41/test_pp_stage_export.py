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


@pytest.mark.parametrize('mismatch', ['none', 'reuse', 'resend'])
@pytest.mark.parametrize('resend', [False, True])
def test_pp_export_publishes_one_generation_wide_frozen_header(
    v41_core_te, monkeypatch, mismatch, resend
):
    from types import SimpleNamespace

    from megatron.lite.model.deepseek_v41.lite import checkpoint, resync
    from megatron.lite.primitive.ckpt import pipeline_stream

    ps = SimpleNamespace(pp_size=2, pp_rank=0, pp_is_last=False, pp_group=object())
    seen = []

    def export(*args, **kwargs):
        assert kwargs['include_archival'] is False
        yield resync.frozen_tables_transport({'first': ['hash-a']}, False, resend)
        yield 'local', torch.tensor([3])

    def gather(result, local, *, group):
        assert group is ps.pp_group
        result[:] = [
            local,
            resync.FrozenTables(
                mismatch == 'reuse',
                {'second': ['hash-b']},
                resend ^ (mismatch == 'resend'),
            ),
        ]

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
            resync_config={'freeze_engram': True, 'resend_frozen_engram': resend},
            buffer_max_size_bytes=256,
        )
    )
    if mismatch != 'none':
        with pytest.raises(ValueError, match='generations disagree'):
            call()
        assert not seen
    else:
        result = call()
        first = resync.decode_transport(*result[0])
        assert first.manifest == {'first': ['hash-a'], 'second': ['hash-b']}
        assert first.reuse is False
        assert first.resend is resend
        assert [name for name, _ in result[1:-1]] == ['local', 'remote']
        assert resync.decode_transport(*result[-1]) is None
        assert seen == [64]


def test_real_pp_stream_merges_initial_and_reused_frozen_generations(tmp_path):
    import json

    import torch.multiprocessing as mp
    from test_pp_bound_export import _real_pp_export_worker

    mp.spawn(_real_pp_export_worker, args=(2, 1, str(tmp_path)), nprocs=2, join=True)
    rows = [
        json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(2)
    ]
    assert all(row['passed'] and row['receiver_loaded'] for row in rows)
    assert len({row['sha256'] for row in rows}) == 1
