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


def test_frozen_export_and_mutation(v41_core_te, tmp_path):
    import megatron.lite.model.deepseek_v41.lite.resync as resync

    decode_transport = resync.decode_transport
    FrozenTables = resync.FrozenTables
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    model = make_model(tmp_path / 'archive', False)
    first = [decode_transport(*x) for x in exports(model)]
    second = [decode_transport(*x) for x in exports(model)]
    assert isinstance(first[0], FrozenTables) and not first[0].reuse
    assert isinstance(second[0], FrozenTables) and second[0].reuse
    assert first[0].manifest == second[0].manifest
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


def test_receiver_census_reuse_and_byte_corruption(v41_core_te, monkeypatch):
    import megatron.lite.model.deepseek_v41.lite.resync as resync

    frozen_tables_transport = resync.frozen_tables_transport
    transport_weights = resync.transport_weights
    from megatron.lite.primitive.ckpt.row_stream import RowChunk

    consumer = adapter(v41_core_te)
    model = receiver_model(monkeypatch)
    manifest = {'layers.1.engram.embed.weight': ['a' * 64]}
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
    manifest = {'layers.1.engram.embed.weight': ['a' * 64]}
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
    manifest = {'layers.1.engram.embed.weight': ['a' * 64]}
    model.table.weight.data.view(torch.uint8).fill_(56)
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
