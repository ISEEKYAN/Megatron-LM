# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Stage-local PP ownership and training against the unpartitioned model."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from test_redo_parity import compare_execution, release_config


def prefix_config():
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    source = release_config().to_hf_dict()
    text = source['text_config']
    text.update(
        num_hidden_layers=4,
        compress_ratios=[0, 0, 2, 2] + [0] * 3,
        kv_source_layer_ids=[2],
        index_source_layer_ids=[2],
        candidate_source_layer_id=-1,
        engram_layer_ids=[],
        engram_num_embeddings=[],
    )
    return DeepseekV41Config(source)


def test_pp2_ep4_passes_parallel_guards(monkeypatch):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 8)
    protocol._validate_parallel(
        protocol.ImplConfig(optimizer='muon', pipeline_split_layer=2),
        ParallelConfig(pp=2, ep=4),
        cut=2,
    )


@pytest.mark.parametrize('cut', [2, 8, 14, 20])
def test_pp_cut_keeps_attention_owners_with_readers(cut):
    from megatron.lite.model.deepseek_v41.topology import pipeline_cut

    assert pipeline_cut(release_config().topology, cut) == cut


@pytest.mark.parametrize('cut', [0, 3, 10, 19, 21, 40])
def test_pp_cut_rejects_cross_stage_state(cut):
    from megatron.lite.model.deepseek_v41.topology import pipeline_cut

    with pytest.raises((ValueError, NotImplementedError)):
        pipeline_cut(release_config().topology, cut)


@pytest.fixture
def pp_models(v41_core_te):
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model
    from megatron.lite.primitive.parallel.state import ParallelState

    cfg = prefix_config()
    torch.manual_seed(381)
    full = DeepseekV41Model(cfg, quantized=False)
    stages = [
        DeepseekV41Model(
            cfg,
            quantized=False,
            layer_range=span,
            parallel_state=ParallelState(
                pp_size=2, pp_rank=i, pp_is_first=i == 0, pp_is_last=i == 1
            ),
            pipeline_split_layer=2,
        )
        for i, span in enumerate([(0, 2), (2, 4)])
    ]
    for model in [full, *stages]:
        model.float()
        for binding in model.parameter_bindings():
            if binding.role in ("vision", "aligner", "image_delimiter"):
                binding.tensor.requires_grad_(False)
    for stage in stages:
        stage.pipeline_residual_dtype = torch.float32
    return full, stages


def test_pp_stage_owner_inventory_is_disjoint_and_complete(pp_models):
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import (
        _optimizer_owners,
        parameter_groups,
    )

    full, stages = pp_models
    whole = {g['owner_key'] for g in parameter_groups(full, lr=1e-4)}
    inventories = []
    for stage in stages:
        groups = parameter_groups(stage, lr=1e-4)
        inventories.append({g['owner_key'] for g in groups})
        experts, routers, tables, _ = _optimizer_owners(stage)
        assert len(routers) == 2 and experts and not tables
        assert all(group['params'][0].requires_grad for group in groups)
    assert not inventories[0] & inventories[1]
    assert inventories[0] | inventories[1] == whole


def test_pp_paired_forward_backward_and_per_stage_router_stats(pp_models):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts.data import PackedBatch

    full, stages = pp_models
    ids = torch.tensor([[2, 3, 9, 4]])

    def execute(reference, pieces):
        expected = reference(ids)['logits']
        outgoing = pieces[0](ids)
        assert len(outgoing['modality_loads']) == 2
        assert all(len(entries) == 1 for entries in outgoing['modality_loads'])
        assert outgoing['hidden_states'].dtype == torch.float32
        pieces[1].set_input_tensor(outgoing['hidden_states'])
        incoming = pieces[1](ids)
        assert len(incoming['modality_loads']) == 2
        assert all(len(entries) == 1 for entries in incoming['modality_loads'])
        return expected, incoming['logits']

    compare_execution(full, stages, execute)
    batch = PackedBatch(ids[0], labels=None, seq_lens=torch.tensor([4]))
    optimizers = [Mock(), Mock()]
    outgoing = protocol._forward_step(stages[0], batch, optimizer=optimizers[0])
    stages[1].set_input_tensor(outgoing['hidden_states'])
    protocol._forward_step(stages[1], batch, optimizer=optimizers[1])
    for optimizer in optimizers:
        loads = optimizer.accumulate_modality_loads.call_args.args[0]
        assert len(loads) == 2
        optimizer.accumulate_modality_loads.assert_called_once()


@pytest.mark.parametrize('use_deepep', [False, True])
def test_official_pp2_passes_deployment_guard_before_allocation(
    v41_core_te, monkeypatch, use_deepep
):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.parallel.state import ParallelState
    from megatron.lite.runtime.contracts import ParallelConfig

    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(protocol, 'init_parallel', lambda _: ParallelState(pp_size=2))

    def construct(*args):
        raise RuntimeError('official PP reached stage constructor')

    monkeypatch.setattr(protocol, '_construct_model', construct)
    with pytest.raises(RuntimeError, match='official PP reached stage constructor'):
        protocol.build_model(
            prefix_config(),
            impl_cfg=protocol.ImplConfig(
                device='cpu',
                parallel=ParallelConfig(pp=2),
                pipeline_split_layer=2,
                deployment_math=True,
                w4a8_experts=True,
                use_deepep=use_deepep,
            ),
        )


@pytest.mark.parametrize('rank', range(8))
def test_pp2_ep4_uses_stage_local_dense_and_expert_groups(monkeypatch, rank):
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 8)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: rank)
    monkeypatch.setattr(
        torch.distributed, 'new_group', lambda ranks, **kw: tuple(ranks)
    )
    ps = init_parallel(ParallelConfig(pp=2, ep=4))
    stage = tuple(range(0, 4)) if rank < 4 else tuple(range(4, 8))
    assert ps.ep_group == ps.dp_group == stage
    assert ps.ep_dp_group == (rank,)
    assert ps.pp_group == (rank % 4, rank % 4 + 4)
    assert ps.dp_size == 4 and ps.expert_dp_size == 1
    assert ps.pp_rank == rank // 4 and ps.ep_rank == rank % 4


def test_stage_host_owners_update_and_restore_like_unpartitioned(pp_models):
    # Real Muon/Sinkhorn/AdamW and segmented host owners; global PP votes and
    # clipping are exercised by distributed integration, not simulated here.
    from copy import deepcopy

    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import V41Optimizer
    from megatron.lite.model.deepseek_v41.vision_config import OptimizerConfig

    full, stages = pp_models
    for stage in stages:
        own = stage.state_dict()
        stage.load_state_dict({k: v for k, v in full.state_dict().items() if k in own})
    cfg = OptimizerConfig(
        lr=1e-4,
        ns_steps=2,
        coefficient_type='quintic',
        clip_grad=0,
        segmented_host=True,
    )
    all_models = [full, *stages]
    optimizers = [V41Optimizer(model, cfg) for model in all_models]
    ids = torch.tensor([[2, 3, 9, 4]])
    for iteration in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad()
        expected = full(ids)
        outgoing = stages[0](ids)
        stages[1].set_input_tensor(outgoing['hidden_states'])
        actual = stages[1](ids)
        assert torch.equal(actual['logits'], expected['logits'])
        expected['logits'].square().mean().backward()
        actual['logits'].square().mean().backward()
        for optimizer, output in zip(optimizers, [expected, outgoing, actual]):
            optimizer.accumulate_modality_loads(output['modality_loads'])
            assert optimizer.step()[0]
        reference = dict(full.named_parameters())
        for stage_id, (stage, optimizer) in enumerate(zip(stages, optimizers[1:])):
            for name, value in stage.named_parameters():
                assert torch.equal(value, reference[name]), (iteration, name)
            saved = deepcopy(optimizer.state_dict())
            restored = V41Optimizer(stage, cfg)
            restored.load_state_dict(saved)
            assert set(restored.host_update.masters) == set(
                optimizer.host_update.masters
            )
            optimizers[stage_id + 1] = restored


def test_pp_router_bias_reduces_only_same_stage_owners(v41_core_te, monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.model.deepseek_v41.lite import moe

    config = SimpleNamespace(**prefix_config().to_hf_dict()['text_config'])
    stage_group = object()
    router = moe.ModalityRouter(
        config, SimpleNamespace(tp_size=1), bias_group=stage_group
    )
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 8)
    seen = []

    def updated(counts, bias, rate, group):
        assert group is stage_group
        seen.append(group)
        return bias + torch.sign(counts.mean() - counts) * rate

    monkeypatch.setattr(moe, 'get_updated_expert_bias', updated)
    router.update_bias(
        moe.ModalityLoad(torch.tensor([[8, 0], [0, 8]]), torch.tensor([8, 8]))
    )
    assert seen == [stage_group, stage_group]
    assert router.bias[0] < router.bias[1]
    assert router.bias_vl[0] > router.bias_vl[1]


def _pp_host_worker(rank, directory, segmented):
    import runpy
    from datetime import timedelta
    from functools import partial
    from pathlib import Path
    from types import SimpleNamespace

    import torch.distributed as dist

    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.vision_config import OptimizerConfig
    from megatron.lite.primitive.parallel.pipeline import forward_backward_pipelining
    from megatron.lite.runtime.contracts import ParallelConfig
    from megatron.lite.runtime.contracts.data import PackedBatch

    torch.set_num_threads(1)
    cfg = prefix_config()
    optimizer_cfg = OptimizerConfig(
        lr=1e-4,
        ns_steps=2,
        coefficient_type='quintic',
        clip_grad=1e-4,
        segmented_host=segmented,
    )
    impl = protocol.ImplConfig(
        device='cpu',
        dtype=torch.float32,
        quantized=False,
        shard_engram=False,
        optimizer='muon',
        optimizer_config=optimizer_cfg,
    )
    torch.manual_seed(91)
    reference = protocol.build_model(cfg, impl_cfg=impl)
    full, ref_opt = reference.chunks[0], reference.optimizer
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/store',
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        bundle = protocol.build_model(
            cfg,
            impl_cfg=replace(
                impl, parallel=ParallelConfig(pp=2), pipeline_split_layer=2
            ),
        )
        stage, optimizer, ps = bundle.chunks[0], bundle.optimizer, bundle.parallel_state
        own = stage.state_dict()
        stage.load_state_dict({k: v for k, v in full.state_dict().items() if k in own})
        batches = [
            PackedBatch(
                torch.tensor(ids), labels=None, seq_lens=torch.tensor([len(ids)])
            )
            for ids in ([2, 3, 9], [4, 8, 7, 11, 5])
        ]
        for iteration in range(2):
            ref_opt.zero_grad()
            optimizer.zero_grad()
            expected = []
            for batch in batches:
                out = protocol._forward_step(full, batch, optimizer=ref_opt)
                expected.append(out['logits'].detach().clone())
                (out['logits'].square().mean() / len(batches)).backward()
            checked = []

            def loss_fn(out, batch):
                assert torch.equal(out['logits'], expected[len(checked)])
                checked.append(True)
                return out['logits'].square().mean(), {}

            forward_backward_pipelining(
                partial(protocol._forward_step, optimizer=optimizer),
                [stage],
                iter(batches),
                SimpleNamespace(num_microbatches=2, pipeline_dtype=torch.float32),
                ps,
                tensor_shape=(1, 1, 66),
                loss_fn=loss_fn,
            )
            assert len(checked) == (2 if rank else 0)
            expected_params = dict(full.named_parameters())
            for name, p in stage.named_parameters():
                wanted = expected_params[name].grad
                assert (p.grad is None) == (wanted is None), name
                if wanted is not None:
                    assert torch.equal(p.grad, wanted), name
            optimizer.finalize_grads()
            # The clip norm is global across stages, so the clipped update
            # must reproduce the unpartitioned model's bits.
            stage_result, ref_result = optimizer.step(), ref_opt.step()
            assert stage_result[0] and ref_result[0]
            assert stage_result[1] == ref_result[1] > optimizer_cfg.clip_grad, (
                stage_result[1],
                ref_result[1],
            )
            for name, p in stage.named_parameters():
                assert torch.equal(p, expected_params[name]), (iteration, name)
        # One bad stage must reject publication on both PP peers, despite
        # different stage owner inventories and local host transaction lengths.
        before = {n: p.detach().clone() for n, p in stage.named_parameters()}
        optimizer.zero_grad()
        for p in stage.parameters():
            if p.requires_grad:
                p.grad = p.main_grad = torch.ones_like(p)
        if rank == 0:
            next(p for p in stage.parameters() if p.requires_grad).grad.flatten()[0] = (
                float('nan')
            )
        assert not optimizer.step()[0]
        assert all(torch.equal(p, before[n]) for n, p in stage.named_parameters())
        torch.save(
            {'rank': rank, 'two_updates': True, 'pp_rejection': True},
            Path(directory) / f'rank-{rank}.pt',
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('segmented', [True, False], ids=['host', 'resident'])
def test_real_pp2_1f1b_update_and_shared_failure_vote(tmp_path, segmented):
    import torch.multiprocessing as mp

    mp.spawn(_pp_host_worker, args=(str(tmp_path), segmented), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(tmp_path / f'rank-{rank}.pt', weights_only=True)
        assert result['two_updates'] and result['pp_rejection']


@pytest.mark.parametrize('fail', [False, True])
def test_pipeline_honors_and_restores_transport_dtype(monkeypatch, fail):
    from types import SimpleNamespace

    from megatron.lite.primitive.parallel import pipeline

    previous = pipeline._PIPELINE_TENSOR_DTYPE

    def schedule(*args, **kwargs):
        assert pipeline._PIPELINE_TENSOR_DTYPE == torch.float32
        if fail:
            raise RuntimeError('schedule failure')
        return ['scheduled']

    monkeypatch.setattr(pipeline, '_1f1b_schedule', schedule)
    call = lambda: pipeline.forward_backward_pipelining(
        None,
        [None],
        None,
        SimpleNamespace(pipeline_dtype=torch.float32, num_microbatches=1),
        SimpleNamespace(pp_size=2),
        tensor_shape=(1, 1, 66),
    )
    if fail:
        with pytest.raises(RuntimeError, match='schedule failure'):
            call()
    else:
        assert call() == ['scheduled']
    assert pipeline._PIPELINE_TENSOR_DTYPE == previous
