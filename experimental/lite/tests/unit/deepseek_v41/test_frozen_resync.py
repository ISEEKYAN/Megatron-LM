# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Freeze, replay and corruption controls for actual exporter/receiver contracts."""
from types import SimpleNamespace as NS

import pytest
import torch
from reload_fixture import install_reload
from test_bound_row_export import make_model
from test_receiver_staging import adapter


def exports(model, frozen=True):
    import megatron.lite.model.deepseek_v41.lite.checkpoint as ckpt

    export_hf_weights = ckpt.export_hf_weights

    options = {'expert_dtype': 'fp4'}
    if frozen:
        options['freeze_engram'] = True
    return list(
        export_hf_weights(
            [model],
            model.config,
            model.ps,
            target='mxfp4',
            resync_config=options,
            include_archival=False,
            buffer_max_size_bytes=524288,
        )
    )


@pytest.mark.parametrize('hasher', [True, False])
def test_frozen_export_and_mutation(v41_core_te, tmp_path, hasher):
    import megatron.lite.model.deepseek_v41.lite.resync as resync

    decode_transport = resync.decode_transport
    FrozenTables = resync.FrozenTables
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    model = make_model(tmp_path / 'archive', False)
    if not hasher:
        model.engram_hash = None
    first = [decode_transport(*x) for x in exports(model)]
    second = [decode_transport(*x) for x in exports(model)]
    assert isinstance(first[0], FrozenTables) and not first[0].reuse
    assert isinstance(second[0], FrozenTables) and second[0].reuse
    assert first[0].manifest == second[0].manifest
    assert all(
        isinstance(piece, dict)
        for pieces in first[0].manifest.values()
        for piece in pieces
    )
    assert any(isinstance(x, RowChunk) for x in first)
    assert not any(isinstance(x, RowChunk) for x in second)
    # Actual table is a buffer: no master, gradients or AdamW/Muon owner/state.
    tables = [model.layers[i].engram.embed for i in (1, 14)]
    for table in tables:
        assert table.master is None and list(table.parameters()) == []
        assert table.weight.grad is table.scale.grad is None
    # .data bypasses version counters: byte fingerprint must still fail loudly.
    tables[0].weight.data.view(torch.uint8)[0, 0] ^= 1
    with pytest.raises(ValueError, match='storage changed'):
        exports(model)


def test_trainable_legacy_preserved_and_opt_in_rejected(v41_core_te, tmp_path):
    from megatron.lite.model.deepseek_v41.lite.resync import decode_transport
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    model = make_model(tmp_path / 'archive', True)
    assert any(
        isinstance(decode_transport(*x), RowChunk) for x in exports(model, False)
    )
    with pytest.raises(ValueError, match='without FP32 masters'):
        exports(model)
    table = model.layers[1].engram.embed
    loss = table(torch.tensor([0, 1])).float().sum()
    loss.backward()
    assert table.master.grad is not None


def receiver_model(monkeypatch):
    model = torch.nn.Module()
    model.table = torch.nn.Module()
    model.table.weight = torch.nn.Parameter(
        torch.zeros((4, 32), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    model.table.weight.engram_vocab_start = 0
    model.table.weight_scale_inv = torch.nn.Parameter(
        torch.ones((4, 1)).to(torch.float8_e8m0fnu), requires_grad=False
    )
    model.hf_to_vllm_mapper = NS(
        apply=lambda pairs: [('table.weight', tensor) for _, tensor in pairs]
    )
    install_reload(monkeypatch, model)
    return model


def receiver_manifest(model):
    from megatron.lite.primitive.ckpt.frozen_storage import storage_census

    return {
        'layers.1.engram.embed.weight': storage_census(
            model.table.weight,
            model.table.weight_scale_inv,
            start=0,
            boundaries=(0, len(model.table.weight)),
        )
    }


def test_receiver_census_reuse_and_byte_corruption(v41_core_te, monkeypatch):
    import megatron.lite.model.deepseek_v41.lite.resync as resync

    frozen_tables_transport = resync.frozen_tables_transport
    transport_weights = resync.transport_weights
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    consumer = adapter(v41_core_te)
    model = receiver_model(monkeypatch)
    manifest = receiver_manifest(model)
    first = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    first.receive([frozen_tables_transport(manifest, False)])
    rows = RowChunk(
        'layers.1.engram.embed.weight',
        0,
        4,
        model.table.weight.detach().clone(),
        model.table.weight_scale_inv.detach().clone(),
    )
    first.receive(list(transport_weights([rows], deployment=True)))
    first.finish()
    second = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    second.receive(
        [
            frozen_tables_transport(manifest, True),
            *transport_weights([], deployment=True),
        ]
    )
    second.finish()
    assert second.received_tables == set()
    model.table.weight.data.view(torch.uint8)[0, 0] ^= 1
    broken = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    with pytest.raises(ValueError, match='storage changed'):
        broken.receive([frozen_tables_transport(manifest, True)])
    broken.abort()


def test_receiver_rejects_cold_reuse_and_missing_initial_rows(v41_core_te, monkeypatch):
    import megatron.lite.model.deepseek_v41.lite.resync as resync

    frozen_tables_transport = resync.frozen_tables_transport
    transport_weights = resync.transport_weights

    consumer = adapter(v41_core_te)
    model = receiver_model(monkeypatch)
    manifest = receiver_manifest(model)
    receiver = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    with pytest.raises(ValueError, match='matching initial'):
        receiver.receive([frozen_tables_transport(manifest, True)])
    receiver.abort()
    receiver = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    receiver.receive(
        [
            frozen_tables_transport(manifest, False),
            *transport_weights([], deployment=True),
        ]
    )
    with pytest.raises(ValueError, match='missing Engram'):
        receiver.finish()
    receiver.abort()


def test_saved_hf_is_standalone_after_online_freeze(v41_core_te, tmp_path):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from test_bound_row_export import read_hf

    model = make_model(tmp_path / 'archive', False)
    exports(model)
    protocol.save_hf_weights(
        [model],
        tmp_path / 'saved',
        model.config,
        model.ps,
        target='mxfp4',
        resync_config={'freeze_engram': True},
        buffer_max_size_bytes=524288,
    )
    saved = read_hf(tmp_path / 'saved')
    for index in (1, 14):
        assert f'layers.{index}.engram.embed.weight' in saved
        assert f'layers.{index}.engram.embed.scale' in saved


def test_frozen_table_has_no_optimizer_state(v41_core_te, tmp_path):
    import megatron.lite.model.deepseek_v41.lite.optimizer_groups as groups

    V41Optimizer = groups.V41Optimizer
    from megatron.lite.model.deepseek_v41.vision_config import OptimizerConfig

    model = make_model(tmp_path / 'archive', False)
    optimizer = V41Optimizer(
        model, OptimizerConfig(lr=1e-6, ns_steps=2, coefficient_type='quintic')
    )
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    for index in (1, 14):
        table = model.layers[index].engram.embed
        assert table.master is None
        assert not any(p is table.weight or p is table.scale for p in parameters)
    # Exercise actual Muon and AdamW publication with all owners finite.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.01)
            if p.requires_grad:
                p.grad = torch.ones_like(p)
    assert optimizer.step()[0]
    for backend in optimizer.optimizers:
        assert all(any(p is owner for owner in parameters) for p in backend.state)
    for index in (1, 14):
        table = model.layers[index].engram.embed
        assert table.weight.grad is table.scale.grad is None


def test_frozen_tables_survive_native_level2_buffer_lifecycle(v41_core_te, monkeypatch):
    import megatron.lite.model.deepseek_v41.lite.resync as resync

    frozen_tables_transport = resync.frozen_tables_transport
    transport_weights = resync.transport_weights
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    consumer = adapter(v41_core_te)
    model = receiver_model(monkeypatch)
    model.table.weight.data.view(torch.uint8).fill_(56)
    manifest = receiver_manifest(model)
    first = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    first.receive([frozen_tables_transport(manifest, False)])
    first.receive(
        list(
            transport_weights(
                [
                    RowChunk(
                        'layers.1.engram.embed.weight',
                        0,
                        4,
                        model.table.weight.detach().clone(),
                        model.table.weight_scale_inv.detach().clone(),
                    )
                ],
                deployment=True,
            )
        )
    )
    first.finish()
    original = first._table_digests()
    # These are the exact named_buffers save/copy operations in GPUWorker
    # sleep(level=2)/wake_up(tags=['weights']); emulate discarded storage.
    saved = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}
    assert len(saved) == 2
    assert not any('_ds41_frozen_' in name for name in model.state_dict())
    model.table.weight.data.view(torch.uint8).zero_()
    model.table.weight_scale_inv.data.view(torch.uint8).zero_()
    assert first._table_digests() != original
    for name, buffer in model.named_buffers():
        buffer.data.copy_(saved[name].data)
    assert first._table_digests() == original
    second = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    second.receive(
        [
            frozen_tables_transport(manifest, True),
            *transport_weights([], deployment=True),
        ]
    )
    second.finish()
    assert second.received_tables == set()
    assert len(list(model.named_buffers())) == 2
    # Wake restoration must not weaken the subsequent byte corruption guard.
    model.table.weight.data.view(torch.uint8)[0, 0] ^= 1
    broken = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    with pytest.raises(ValueError, match='storage changed'):
        broken.receive([frozen_tables_transport(manifest, True)])
    broken.abort()


def test_receiver_rejects_fabricated_initial_shard_census(v41_core_te, monkeypatch):
    from megatron.lite.model.deepseek_v41.lite.resync import (
        frozen_tables_transport,
        transport_weights,
    )
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    consumer = adapter(v41_core_te)
    model = receiver_model(monkeypatch)
    receiver = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    receiver.receive(
        [
            frozen_tables_transport(
                {
                    'layers.1.engram.embed.weight': [
                        dict(start=0, end=4, sha256='a' * 64)
                    ]
                },
                False,
            )
        ]
    )
    receiver.receive(
        list(
            transport_weights(
                [
                    RowChunk(
                        'layers.1.engram.embed.weight',
                        0,
                        4,
                        model.table.weight.detach().clone(),
                        model.table.weight_scale_inv.detach().clone(),
                    )
                ],
                deployment=True,
            )
        )
    )
    with pytest.raises(ValueError, match='shard census digest mismatch'):
        receiver.finish()
    receiver.abort()


def _ep8_frozen_census_worker(rank, directory, source_partitions, receiver_layout):
    import hashlib
    import runpy
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils
    import megatron.core.transformer.experimental_attention_variant.csa
    import megatron.core.transformer.hyper_connection

    monkeypatch = pytest.MonkeyPatch()
    fixtures['transformer_engine_import_stub'].__wrapped__(monkeypatch)()
    from megatron.lite.model.deepseek_v41.lite.checkpoint import export_hf_weights
    from megatron.lite.model.deepseek_v41.lite.resync import (
        FrozenTables,
        decode_transport,
    )
    from megatron.lite.primitive.ckpt.binding_records import (
        TensorBinding,
        validate_parameter_bindings,
    )
    from megatron.lite.primitive.ckpt.frozen_storage import verify_storage_census
    from megatron.lite.primitive.ckpt.row_stream import RowChunk
    from megatron.lite.primitive.modules.engram_lookup import (
        RowLookup,
        ShardedEngramTable,
    )

    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo',
        init_method='file://' + str(Path(directory) / 'rendezvous'),
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=120),
    )
    try:
        world_cuts = tuple(17 * i // 8 for i in range(9))
        cuts = tuple(17 * i // source_partitions for i in range(source_partitions + 1))
        group = dist.group.WORLD
        if source_partitions == 4:
            groups = [dist.new_group(list(range(a, a + 4))) for a in (0, 4)]
            group = groups[rank // 4]
        raw = ((torch.arange(17 * 32).reshape(17, 32) % 63) + 32).to(torch.uint8)
        scale = (118 + torch.arange(17) % 17).to(torch.uint8).reshape(17, 1)
        local_rank = rank % source_partitions
        a, b = cuts[local_rank : local_rank + 2]
        table = ShardedEngramTable(
            raw[a:b].clone().view(torch.float8_e4m3fn),
            scale[a:b].clone().view(torch.float8_e8m0fnu),
            RowLookup(cuts, group),
        )
        model = torch.nn.Module()
        model.anchor = torch.nn.Parameter(torch.ones(2))
        model.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Module()])
        model.layers[1].engram = torch.nn.Module()
        model.layers[1].engram.embed = table
        model.local_layer_range = (0, 2)
        model.archival_bindings = {}
        model.archival_store = None
        model.ps = NS(pp_size=1, pp_rank=0, ep_size=8, ep_group=dist.group.WORLD)
        model.config = NS(to_hf_dict=lambda: {'text_config': {'n_routed_experts': 8}})
        model.engram_layer_ids = (1,)
        model.engram_hash = NS(primes=torch.tensor([[[3, 4, 5, 5]]]))
        key = 'layers.1.engram.embed.weight'
        model.tensor_bindings = {
            'anchor': TensorBinding('anchor', model, 'anchor', 'norm'),
            key: TensorBinding(key, table, 'weight', 'engram_table'),
        }
        model.parameter_bindings = lambda: [model.tensor_bindings['anchor']]
        model.validate_parameter_bindings = lambda: validate_parameter_bindings(model)
        consumer = adapter(None)
        target = receiver_model(monkeypatch)
        head_cuts = (0, 3, 7, 12, 17)
        begin, end = (
            head_cuts[rank % 4 : rank % 4 + 2]
            if receiver_layout == 'head'
            else world_cuts[rank : rank + 2]
        )
        target.table.weight = torch.nn.Parameter(
            torch.zeros_like(raw[begin:end]).view(torch.float8_e4m3fn),
            requires_grad=False,
        )
        target.table.weight.engram_vocab_start = begin
        target.table.weight_scale_inv = torch.nn.Parameter(
            torch.zeros_like(scale[begin:end]), requires_grad=False
        )

        def independent_digest(begin, end):
            digest = hashlib.sha256()
            for value, dtype in (
                (raw[begin:end], torch.float8_e4m3fn),
                (scale[begin:end], torch.float8_e8m0fnu),
            ):
                digest.update(str((tuple(value.shape), dtype)).encode())
                digest.update(value.numpy().tobytes())
            return digest.hexdigest()

        census_cuts = sorted(set(cuts).union(head_cuts, world_cuts))
        expected = [
            dict(start=a, end=b, sha256=independent_digest(a, b))
            for a, b in zip(census_cuts, census_cuts[1:])
        ]
        manifest = None
        for reuse in (False, True):
            receiver = consumer.ResyncReceiver(target, NS(cpu_offload_gb=0))
            row_count = 0
            received = []
            for pair in export_hf_weights(
                [model],
                model.config,
                model.ps,
                target='mxfp4',
                resync_config={'expert_dtype': 'fp4', 'freeze_engram': True},
                include_archival=False,
                buffer_max_size_bytes=65536,
            ):
                item = decode_transport(*pair)
                if isinstance(item, FrozenTables):
                    assert item.reuse == reuse
                    manifest = item.manifest
                    receiver.receive([pair])
                    received.append(pair)
                elif isinstance(item, RowChunk):
                    row_count += 1
                    receiver.receive([pair])
                    received.append(pair)
                elif pair[0].startswith('__'):
                    receiver.receive([pair])
                    received.append(pair)
            verify_storage_census(
                manifest[key],
                target.table.weight,
                target.table.weight_scale_inv,
                start=begin,
                total_rows=17,
            )
            assert manifest == {key: expected}
            # The census comes from the real exporter on both generations.
            # Damage only one installed DP shard; other ranks remain valid.
            for plane in (target.table.weight, target.table.weight_scale_inv):
                if rank == 7:
                    plane.view(torch.uint8)[0, 0] ^= 1
                    with pytest.raises(
                        ValueError, match='shard census digest mismatch'
                    ):
                        verify_storage_census(
                            manifest[key],
                            target.table.weight,
                            target.table.weight_scale_inv,
                            start=begin,
                            total_rows=17,
                        )
                    if not reuse:
                        with pytest.raises(
                            ValueError, match='shard census digest mismatch'
                        ):
                            receiver.finish()
                    plane.view(torch.uint8)[0, 0] ^= 1
                    if not reuse:
                        receiver = consumer.ResyncReceiver(target, NS(cpu_offload_gb=0))
                        receiver.receive(received)
                else:
                    verify_storage_census(
                        manifest[key],
                        target.table.weight,
                        target.table.weight_scale_inv,
                        start=begin,
                        total_rows=17,
                    )
            receiver.finish()
            assert bool(row_count) != reuse
            assert torch.equal(target.table.weight.view(torch.uint8), raw[begin:end])
            assert torch.equal(
                target.table.weight_scale_inv.view(torch.uint8), scale[begin:end]
            )
        target.table.weight.data.view(torch.uint8)[0, 0] ^= 1
        from megatron.lite.model.deepseek_v41.lite.resync import frozen_tables_transport

        broken = consumer.ResyncReceiver(target, NS(cpu_offload_gb=0))
        with pytest.raises(ValueError, match='storage changed'):
            broken.receive([frozen_tables_transport(manifest, True)])
        broken.abort()
        for plane in (table.weight, table.scale):
            if local_rank == source_partitions - 1:
                plane.view(torch.uint8)[0, 0] ^= 1
            with pytest.raises(ValueError, match='storage changed'):
                next(
                    export_hf_weights(
                        [model],
                        model.config,
                        model.ps,
                        target='mxfp4',
                        resync_config={'expert_dtype': 'fp4', 'freeze_engram': True},
                        include_archival=False,
                        buffer_max_size_bytes=65536,
                    )
                )
            if local_rank == source_partitions - 1:
                plane.view(torch.uint8)[0, 0] ^= 1

    finally:
        dist.destroy_process_group()
        monkeypatch.undo()


@pytest.mark.parametrize(
    'source_partitions,receiver_layout', [(8, 'head'), (8, 'dp'), (4, 'dp')]
)
def test_ep8_frozen_census_roundtrip_reuse_and_rank7_corruption(
    tmp_path, source_partitions, receiver_layout
):
    """Actual exports cover PP-local DP4 -> rollout DP8, reuse and W/S damage."""
    torch.multiprocessing.spawn(
        _ep8_frozen_census_worker,
        args=(str(tmp_path), source_partitions, receiver_layout),
        nprocs=8,
        join=True,
    )


@pytest.mark.parametrize('span', [(0, 3), (3, 7), (7, 12), (12, 17)])
@pytest.mark.parametrize('damage', [None, 'weight', 'scale', 'gap'])
def test_sharded_initial_census_checks_installed_bytes(
    v41_core_te, monkeypatch, span, damage
):
    """Check actual resident DP slices, including corruption after the copy."""
    import hashlib

    from megatron.lite.model.deepseek_v41.lite.resync import (
        frozen_tables_transport,
        transport_weights,
    )
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    raw = ((torch.arange(17 * 32).reshape(17, 32) % 63) + 32).to(torch.uint8)
    scales = (118 + torch.arange(17) % 17).to(torch.uint8).reshape(17, 1)
    cuts = sorted(set(17 * i // 8 for i in range(9)).union((0, 3, 7, 12, 17)))
    census = []
    for a, b in zip(cuts, cuts[1:]):
        digest = hashlib.sha256()
        for value, dtype in [
            (raw[a:b], torch.float8_e4m3fn),
            (scales[a:b], torch.float8_e8m0fnu),
        ]:
            digest.update(str((tuple(value.shape), dtype)).encode())
            digest.update(value.numpy().tobytes())
        census.append(dict(start=a, end=b, sha256=digest.hexdigest()))
    consumer = adapter(v41_core_te)
    model = receiver_model(monkeypatch)
    a, b = span
    model.table.weight = torch.nn.Parameter(
        torch.zeros_like(raw[a:b]).view(torch.float8_e4m3fn), requires_grad=False
    )
    model.table.weight.engram_vocab_start = a
    model.table.weight_scale_inv = torch.nn.Parameter(
        torch.zeros_like(scales[a:b]), requires_grad=False
    )
    receiver = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    key = 'layers.1.engram.embed.weight'
    if damage == 'gap':
        with pytest.raises(ValueError, match='Invalid frozen Engram shard census'):
            receiver.receive([frozen_tables_transport({key: census[1:]}, False)])
        receiver.abort()
        return
    receiver.receive([frozen_tables_transport({key: census}, False)])
    receiver.receive(
        list(
            transport_weights(
                [
                    RowChunk(
                        key,
                        0,
                        17,
                        raw.view(torch.float8_e4m3fn),
                        scales.view(torch.float8_e8m0fnu),
                    )
                ],
                deployment=True,
            )
        )
    )
    if damage is not None:
        tensor = (
            model.table.weight if damage == 'weight' else model.table.weight_scale_inv
        )
        tensor.data.view(torch.uint8)[0, 0] ^= 1
        with pytest.raises(ValueError, match='shard census digest mismatch'):
            receiver.finish()
        return
    receiver.finish()
    assert torch.equal(model.table.weight.view(torch.uint8), raw[a:b])
    assert torch.equal(model.table.weight_scale_inv.view(torch.uint8), scales[a:b])


def test_resent_frozen_tables_keep_no_sleep_copy(v41_core_te, monkeypatch):
    """Resend: every generation carries rows, equal bytes, no level-2 buffers."""
    import megatron.lite.model.deepseek_v41.lite.resync as resync
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    consumer = adapter(v41_core_te)
    model = receiver_model(monkeypatch)
    model.table.weight.data.view(torch.uint8).fill_(56)
    manifest = receiver_manifest(model)
    rows = model.table.weight.detach().clone()
    scales = model.table.weight_scale_inv.detach().clone()

    def generation(weight=rows):
        receiver = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
        item = resync.decode_transport(
            *resync.frozen_tables_transport(manifest, False, resend=True)
        )
        assert item.resend and not item.reuse
        receiver.receive([resync.frozen_tables_transport(manifest, False, resend=True)])
        receiver.receive(
            list(
                resync.transport_weights(
                    [
                        RowChunk(
                            'layers.1.engram.embed.weight',
                            0,
                            4,
                            weight.clone(),
                            scales.clone(),
                        )
                    ],
                    deployment=True,
                )
            )
        )
        return receiver

    for _ in range(2):
        receiver = generation()
        receiver.finish()
        assert receiver.received_tables == receiver.expected_tables
        # Nothing for GPUWorker sleep(level=2) to save to host.
        assert list(model.named_buffers()) == []
    # Level-2 sleep discards the table; the next generation rewrites it.
    model.table.weight.data.view(torch.uint8).zero_()
    generation().finish()
    # Resent bytes that differ from the first generation are refused.
    changed = rows.clone()
    changed.view(torch.uint8)[0, 0] ^= 1
    broken = generation(changed)
    with pytest.raises(ValueError):
        broken.finish()
    broken.abort()
    # A resent generation without its table rows is refused.
    missing = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    missing.receive(
        [
            resync.frozen_tables_transport(manifest, False, resend=True),
            *resync.transport_weights([], deployment=True),
        ]
    )
    with pytest.raises(ValueError, match='missing Engram tables'):
        missing.finish()
    missing.abort()
    with pytest.raises(ValueError, match='never reused'):
        resync.frozen_tables_transport(manifest, True, resend=True)


def test_resend_option_requires_freeze():
    import megatron.lite.model.deepseek_v41.lite.resync as resync

    assert resync.validate_target(
        'mxfp4', {'freeze_engram': True, 'resend_frozen_engram': True}
    )
    with pytest.raises(ValueError, match='requires freeze_engram'):
        resync.validate_target('mxfp4', {'resend_frozen_engram': True})
