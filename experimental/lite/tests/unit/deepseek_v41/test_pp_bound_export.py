# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real stage bindings reach the codec, rather than a mocked stage iterator."""
import pytest
import torch
from test_pp_stage_training import pp_models


def test_bound_stage_stream_union_matches_pp1_bytes(pp_models):
    from megatron.lite.model.deepseek_v41.lite import checkpoint
    from megatron.lite.model.deepseek_v41.lite.resync import decoded_weights

    full, stages = pp_models
    reference = full.state_dict()
    for stage in stages:
        stage.load_state_dict({k: reference[k] for k in stage.state_dict()})

    def export(model):
        pairs = checkpoint._export_stage_hf_weights(
            [model],
            model.config,
            model.ps,
            target='mxfp4',
            buffer_max_size_bytes=524288,
            include_archival=False,
        )
        return {n: t.clone() for n, t in decoded_weights(pairs)}

    expected = export(full)
    actual = {}
    for stage in stages:
        local = export(stage)
        assert not actual.keys() & local.keys()
        actual.update(local)
    assert actual.keys() == expected.keys()
    for name, tensor in actual.items():
        assert tensor.dtype == expected[name].dtype
        assert torch.equal(
            tensor.view(torch.uint8), expected[name].view(torch.uint8)
        ), name


@pytest.mark.parametrize('entry', ['export', 'save'])
def test_global_bound_entries_still_reject_incomplete_stage(pp_models, tmp_path, entry):
    from megatron.lite.model.deepseek_v41.lite import checkpoint

    _, stages = pp_models
    with pytest.raises(NotImplementedError, match='distributed checkpoint assembly'):
        if entry == 'export':
            list(checkpoint.export_checkpoint(stages[0], include_archival=False))
        else:
            checkpoint.save_model(stages[0], tmp_path / 'partial')
    assert not (tmp_path / 'partial').exists()


def _real_pp_export_worker(rank, world, ep, directory):
    import hashlib
    import json
    import runpy
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist
    from safetensors.torch import save_file

    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils
    import megatron.core.transformer.experimental_attention_variant.csa
    import megatron.core.transformer.hyper_connection

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.model.deepseek_v41.lite import checkpoint
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model
    from megatron.lite.model.deepseek_v41.lite.resync import (
        FrozenTables,
        decode_transport,
    )
    from megatron.lite.primitive.ckpt.hf_weights import load_bound_model
    from megatron.lite.primitive.ckpt.row_stream import RowChunk, RowReceiver
    from megatron.lite.primitive.parallel.state import ParallelState, init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig
    from test_pp_stage_training import prefix_config

    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/store',
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=180),
    )
    try:
        ps = init_parallel(ParallelConfig(pp=2, ep=ep))
        raw = prefix_config().to_hf_dict()
        raw['text_config'].update(
            n_routed_experts=8, engram_layer_ids=[1], engram_num_embeddings=[18]
        )
        cfg = DeepseekV41Config(raw)
        torch.manual_seed(146)
        full = DeepseekV41Model(
            cfg,
            quantized=False,
            token_map=list(range(64)),
            parallel_state=ParallelState(),
        )
        table = full.layers[1].engram.embed
        table.weight.view(torch.uint8).copy_(
            torch.arange(table.weight.numel())
            .remainder(120)
            .to(torch.uint8)
            .reshape_as(table.weight)
        )
        table.scale.view(torch.uint8).fill_(125)
        span = (ps.pp_rank * 2, ps.pp_rank * 2 + 2)
        model = DeepseekV41Model(
            cfg,
            quantized=False,
            token_map=list(range(64)),
            parallel_state=ps,
            layer_range=span,
            pipeline_split_layer=2,
        )
        reference = full.state_dict()
        local_state = {}
        for name, value in model.state_dict().items():
            source = reference[name]
            if source.shape != value.shape:
                assert 'engram.embed.' in name
                lookup = model.layers[1].engram.embed.lookup
                source = source[
                    lookup.boundaries[lookup.rank] : lookup.boundaries[lookup.rank + 1]
                ]
            local_state[name] = source
        model.load_state_dict(local_state)
        options = dict(
            target='mxfp4',
            buffer_max_size_bytes=524288,
            include_archival=False,
            resync_config={'freeze_engram': True},
        )

        def receive(pairs, *, reuse=False, resend=False):
            received, ended, row_receivers, census = {}, False, {}, None
            for pair in pairs:
                assert not ended, 'Tensors after the generation terminator'
                item = decode_transport(*pair)
                if item is None:
                    ended = True
                    continue
                if isinstance(item, FrozenTables):
                    assert census is None and not received
                    census = item
                    continue
                if isinstance(item, RowChunk):
                    name = item.name
                    if name not in row_receivers:
                        received[name] = torch.empty(
                            (item.total_rows, item.weight.shape[1]),
                            dtype=item.weight.dtype,
                        )
                        received[name[:-6] + 'scale'] = torch.empty(
                            (item.total_rows, item.scale.shape[1]),
                            dtype=item.scale.dtype,
                        )
                        row_receivers[name] = RowReceiver(
                            name,
                            item.total_rows,
                            0,
                            received[name],
                            received[name[:-6] + 'scale'],
                        )
                    row_receivers[name].copy(item)
                    continue
                name, value = item
                assert name not in received, ('Duplicate exported key', name)
                received[name] = value.clone()
            assert ended, 'Missing generation terminator'
            assert census is not None and census.reuse is reuse
            assert census.resend is resend
            assert bool(row_receivers) is not reuse
            for receiver in row_receivers.values():
                receiver.finish()
            return received

        expected = receive(
            checkpoint.export_hf_weights([full], cfg, full.ps, **options)
        )
        received = receive(checkpoint.export_hf_weights([model], cfg, ps, **options))
        assert received.keys() == expected.keys()
        digest = hashlib.sha256()
        for name in sorted(expected):
            a, b = received[name], expected[name]
            assert a.shape == b.shape and a.dtype == b.dtype
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
            digest.update(str((name, tuple(a.shape), a.dtype)).encode())
            digest.update(a.contiguous().view(torch.uint8).numpy().tobytes())
        # 09f6's production binding scope must partition exactly the same HF keys.
        bindings = checkpoint._SPEC.expand_bindings(model, dict(model.tensor_bindings))
        local_keys = set(received) & set(bindings)
        stage_keys = [None] * ps.pp_size
        dist.all_gather_object(stage_keys, local_keys, group=ps.pp_group)
        assert not stage_keys[0] & stage_keys[1]
        assert set.union(*stage_keys) == set(expected)
        target = Path(directory) / f'receiver-{rank}'
        target.mkdir()
        save_file(received, str(target / 'model.safetensors'))
        (target / 'config.json').write_text(json.dumps(cfg.to_hf_dict()))
        restored = DeepseekV41Model(
            cfg,
            quantized=False,
            token_map=list(range(64)),
            parallel_state=ps,
            layer_range=span,
            pipeline_split_layer=2,
        )
        load_bound_model(restored, target, checkpoint._SPEC, allow_missing_archive=True)
        # Load a PP1 reference through the same HF decoder, then compare local owners.
        baseline = Path(directory) / f'baseline-{rank}'
        baseline.mkdir()
        save_file(expected, str(baseline / 'model.safetensors'))
        (baseline / 'config.json').write_text(json.dumps(cfg.to_hf_dict()))
        load_bound_model(full, baseline, checkpoint._SPEC, allow_missing_archive=True)
        for name, binding in restored.tensor_bindings.items():
            if name in full.tensor_bindings:
                expected_owner = full.tensor_bindings[name].tensor
                if expected_owner.shape != binding.tensor.shape:
                    lookup = checkpoint._SPEC.row_shard(binding.owner)
                    assert lookup is not None, name
                    expected_owner = expected_owner[
                        lookup.boundaries[lookup.rank] : lookup.boundaries[
                            lookup.rank + 1
                        ]
                    ]
                assert binding.tensor.shape == expected_owner.shape, name
                assert binding.tensor.dtype == expected_owner.dtype, name
                assert torch.equal(
                    binding.tensor.view(torch.uint8), expected_owner.view(torch.uint8)
                ), name
        # A second real generation must reuse the frozen table, not resend it.
        reused_expected = receive(
            checkpoint.export_hf_weights([full], cfg, full.ps, **options), reuse=True
        )
        reused = receive(
            checkpoint.export_hf_weights([model], cfg, ps, **options), reuse=True
        )
        assert reused.keys() == reused_expected.keys()
        for name in reused:
            assert torch.equal(
                reused[name].view(torch.uint8), reused_expected[name].view(torch.uint8)
            ), name
        # Production resends frozen rows every generation (sleep/offload).
        resend_options = dict(
            options, resync_config={'freeze_engram': True, 'resend_frozen_engram': True}
        )
        for _ in range(2):
            resent_expected = receive(
                checkpoint.export_hf_weights([full], cfg, full.ps, **resend_options),
                resend=True,
            )
            resent = receive(
                checkpoint.export_hf_weights([model], cfg, ps, **resend_options),
                resend=True,
            )
            assert resent.keys() == resent_expected.keys() == expected.keys()
            for name in resent:
                assert torch.equal(
                    resent[name].view(torch.uint8),
                    resent_expected[name].view(torch.uint8),
                ), name
        receipt = dict(
            passed=True,
            rank=rank,
            world=world,
            pp=2,
            ep=ep,
            stage_range=span,
            keys=len(received),
            bytes=sum(t.numel() * t.element_size() for t in received.values()),
            sha256=digest.hexdigest(),
            stage_local_keys=len(local_keys),
            initial_and_reuse=True,
            production_resend_twice=True,
            pp1_bytes_equal=True,
            receiver_loaded=True,
            payload='real CPU tensors',
            scope='scaled 4L/8expert; no forward, optimizer, vLLM Engine or GPU',
        )
        (Path(directory) / f'rank-{rank}.json').write_text(json.dumps(receipt))
    finally:
        dist.destroy_process_group()


def test_real_pp_bound_generation_and_stage_load(tmp_path):
    import json

    import torch.multiprocessing as mp

    mp.spawn(_real_pp_export_worker, args=(2, 1, str(tmp_path)), nprocs=2, join=True)
    rows = [
        json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(2)
    ]
    assert all(
        row['passed'] and row['receiver_loaded'] and row['production_resend_twice']
        for row in rows
    )
    assert len({row['sha256'] for row in rows}) == 1
