# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize('cp_size', [1, 2])
def test_cp_experts_use_ddp_and_cannot_reach_expert_finalize(
    v41_core_te, monkeypatch, cp_size
):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.ckpt.binding_records import TensorBinding
    from megatron.lite.primitive.optimizers.headwise_muon import MixedOptimizer
    from megatron.lite.primitive.parallel.owned_ddp import wrap_owned_ddp
    from megatron.lite.primitive.parallel.state import ParallelState
    from megatron.lite.runtime.contracts import ParallelConfig

    dp, dp_cp, row_group = object(), object(), object()
    ps = ParallelState(
        ep_size=1,
        cp_size=cp_size,
        dp_size=2,
        dp_cp_size=2 * cp_size,
        dp_group=dp,
        dp_cp_group=dp_cp,
    )
    model = torch.nn.Module()
    model.expert = torch.nn.Parameter(torch.ones(2))
    model.row = torch.nn.Parameter(torch.ones(2))
    model.parameter_bindings = lambda: [
        TensorBinding('expert', model, 'expert', 'expert')
    ]
    seen = []

    def ddp(chunk, *, process_group, **kwargs):
        seen.append(process_group)
        assert 'expert' not in chunk._ddp_params_and_buffers_to_ignore
        assert chunk._ddp_params_and_buffers_to_ignore == ['row']
        return chunk

    monkeypatch.setattr(torch.nn.parallel, 'DistributedDataParallel', ddp)
    wrap_owned_ddp(
        model,
        ps,
        optimizing=True,
        external_device=None,
        row_tables=[SimpleNamespace(master=model.row, buffers=lambda: ())],
        shard_group=row_group,
    )
    assert seen == [dp_cp if cp_size > 1 else dp]
    model.expert.grad = model.expert.main_grad = torch.tensor([6.0, 10.0])
    model.row.grad = model.row.main_grad = torch.tensor([8.0, 16.0])
    expert_finalize = Mock(
        side_effect=AssertionError('EP=1 reached expert-only reduction')
    )
    optimizer = SimpleNamespace(
        ps=ps,
        host_update=None,
        row_parameters=[model.row],
        finalize_expert_grads=expert_finalize,
    )
    MixedOptimizer.finalize_grads(optimizer)
    expert_finalize.assert_not_called()
    assert torch.equal(model.expert.grad, torch.tensor([6.0, 10.0]))
    assert torch.equal(model.row.grad, torch.tensor([8.0, 16.0]) / (2 * cp_size))

    # EP-over-CP is accepted; numerical update equivalence is covered by
    # test_cp_ep_parity and test_redo_ep_finalize with real distributed ranks.
    from megatron.lite.model.deepseek_v41.lite import model as model_module
    from megatron.lite.primitive.modules import native_fp32_linear
    from test_w4a8_fp32 import tiny_config

    combined = ParallelState(ep_size=2, cp_size=2, dp_size=2, dp_cp_size=4)
    chunk = torch.nn.Module()
    chunk.engram_hash = chunk.engram_group = chunk.vision_schedule = None
    chunk.layers = []
    chunk.parameter_bindings = lambda: []
    constructor = Mock(return_value=chunk)
    parallel = Mock(return_value=combined)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 4)
    monkeypatch.setattr(protocol, 'init_parallel', parallel)
    monkeypatch.setattr(model_module, 'DeepseekV41Model', constructor)
    monkeypatch.setattr(
        protocol, 'wrap_owned_ddp', lambda value, *args, **kwargs: value
    )
    monkeypatch.setattr(
        native_fp32_linear, 'configure_residual_projections', lambda *args: None
    )
    bundle = protocol.build_model(
        tiny_config(),
        impl_cfg=protocol.ImplConfig(device='cpu', parallel=ParallelConfig(cp=2, ep=2)),
    )
    parallel.assert_called_once_with(ParallelConfig(cp=2, ep=2, etp=1))
    assert constructor.call_args.kwargs['parallel_state'] is combined
    assert bundle.parallel_state is combined and bundle.chunks == [chunk]
    assert bundle.extras['prepare_microbatches'].keywords['cp_size'] == 2


def _owned_ddp_subgroup_worker(rank, directory, manual_dense_sync):
    import runpy
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    from megatron.lite.primitive.ckpt.binding_records import TensorBinding
    from megatron.lite.primitive.optimizers.headwise_muon import MixedOptimizer
    from megatron.lite.primitive.parallel.owned_ddp import wrap_owned_ddp
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/store',
        world_size=4,
        rank=rank,
        timeout=timedelta(seconds=60),
    )
    try:
        # Dense and expert groups [1,3] exclude global rank zero.
        ps = init_parallel(ParallelConfig(tp=2, ep=2, etp=1))
        model = torch.nn.Module()
        model.dense = torch.nn.Parameter(torch.tensor([rank + 1.0]))
        model.expert = torch.nn.Parameter(torch.tensor([rank + 10.0]))
        model.register_buffer(
            'encoded', torch.tensor([rank + 1.0]).to(torch.float8_e4m3fn)
        )
        model.table = torch.nn.Module()
        model.table.master = None
        model.table.register_buffer(
            'weight', torch.tensor([rank + 30], dtype=torch.uint8)
        )
        model.table.register_buffer(
            'scale', torch.tensor([rank + 40], dtype=torch.uint8)
        )
        model.parameter_bindings = lambda: [
            TensorBinding('expert', model, 'expert', 'expert')
        ]
        for p in model.parameters():
            p.main_grad = None
        row_bytes = {name: value.clone() for name, value in model.table.named_buffers()}
        wrapped = wrap_owned_ddp(
            model,
            ps,
            optimizing=True,
            external_device=None,
            row_tables=[model.table],
            shard_group=ps.ep_group,
            manual_dense_sync=manual_dense_sync,
        )
        source = dist.get_global_rank(ps.dp_group, 0)
        assert torch.equal(model.encoded.float(), torch.tensor([source + 1.0]))
        assert torch.equal(model.dense, torch.tensor([source + 1.0]))
        assert torch.equal(model.expert, torch.tensor([source + 10.0]))
        for name, value in model.table.named_buffers():
            assert torch.equal(value, row_bytes[name])
        assert set(model._ddp_params_and_buffers_to_ignore) == {
            'expert',
            'table.weight',
            'table.scale',
        }
        if manual_dense_sync:
            assert wrapped is model
        else:
            assert isinstance(wrapped, torch.nn.parallel.DistributedDataParallel)
        # Use the wrapper's real DDP forward and reducer, not a mocked constructor.
        model.forward = lambda x: (model.dense + model.expert) * x
        wrapped(torch.tensor([rank + 1.0])).sum().backward()
        optimizer = MixedOptimizer.__new__(MixedOptimizer)
        optimizer.ps, optimizer.model, optimizer.dp_group = ps, model, ps.dp_group
        optimizer.host_update = object() if manual_dense_sync else None
        optimizer.expert_ids, optimizer.row_ids = {id(model.expert)}, set()
        optimizer.expert_parameters, optimizer.row_parameters = [model.expert], []
        optimizer.finalize_grads()
        mean = (
            sum(r + 1 for r in dist.get_process_group_ranks(ps.dp_group))
            / ps.dp_cp_size
        )
        assert torch.equal(model.dense.grad, torch.tensor([mean]))
        assert torch.equal(model.expert.grad, torch.tensor([mean]))
        reference = [
            torch.nn.Parameter(torch.tensor([source + offset]))
            for offset in (1.0, 10.0)
        ]
        for p in reference:
            p.grad = torch.tensor([mean])
        actual = torch.optim.AdamW(
            list(model.parameters()), lr=0.05, eps=1e-3, foreach=False
        )
        oracle = torch.optim.AdamW(reference, lr=0.05, eps=1e-3, foreach=False)
        actual.step()
        oracle.step()
        for p, q in zip(model.parameters(), reference, strict=True):
            assert torch.equal(p, q)
            for key, value in actual.state[p].items():
                assert torch.equal(value, oracle.state[q][key])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('manual_dense_sync', [False, True])
def test_owned_ddp_subgroups_sync_bytes_experts_and_manual_dense(
    v41_core_te, tmp_path, manual_dense_sync
):
    """Real collectives preserve row shards and choose group-local rank-zero owners."""
    torch.multiprocessing.spawn(
        _owned_ddp_subgroup_worker,
        args=(str(tmp_path), manual_dense_sync),
        nprocs=4,
        join=True,
    )
