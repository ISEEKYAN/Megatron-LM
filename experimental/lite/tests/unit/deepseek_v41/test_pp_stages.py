# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text PP2 must preserve the complete model's native HC values and gradients."""
from dataclasses import replace

import pytest
import torch
# Load the real layout before the CPU-only TE import stub is installed.
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig


def _impl(dtype=torch.float32, trainable=False, device='cpu'):
    return protocol.ImplConfig(
        device=device, dtype=dtype, quantized=False, token_map=list(range(256)),
        trainable_engram=trainable,
    )


def _local_stages(config, impl, monkeypatch):
    assert PipelineParallelLayerLayout is not None
    stages = []
    with monkeypatch.context() as patch:
        patch.setattr(torch.distributed, 'is_initialized', lambda: True)
        patch.setattr(torch.distributed, 'get_world_size', lambda: 2)
        for rank in range(2):
            ps = ParallelState(pp_size=2, pp_rank=rank,
                pp_is_first=rank == 0, pp_is_last=rank == 1)
            patch.setattr(protocol, 'init_parallel', lambda p, ps=ps: ps)
            stages.append(protocol.build_model(config,
                impl_cfg=replace(impl, parallel=ParallelConfig(pp=2))))
    return stages


def _load_stage(stage, reference):
    state = reference.state_dict()
    local = stage.state_dict()
    stage.load_state_dict({key: state[key] for key in local})


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('trainable', [False, True])
def test_pp2_stages_match_monolithic(moe, model_config, monkeypatch, dtype, trainable):
    torch.manual_seed(17)
    impl = _impl(dtype, trainable)
    serial = protocol.build_model(model_config, impl_cfg=impl)
    reference = serial.chunks[0]
    _seed_engram(reference)
    stages = _local_stages(model_config, impl, monkeypatch)
    models = [b.chunks[0] for b in stages]
    for rank, (bundle, model) in enumerate(zip(stages, models)):
        assert bundle.extras['pipeline_dtype'] == torch.float32, 'PP_NATIVE_WIRE_DTYPE'
        assert model.local_layer_range == (rank * 20, (rank + 1) * 20), 'PP_LAYER_OWNERSHIP'
        assert (model.vision is not None) == (rank == 0), 'PP_VISION_OWNER'
        assert (model.embed is not None) == (rank == 0), 'PP_EMBED_OWNER'
        assert (model.head is not None) == (rank == 1), 'PP_HEAD_OWNER'
        assert [i for i, block in enumerate(model.layers)
            if block is not None and block.engram is not None] == ([1, 14] if rank == 0 else []), 'PP_ENGRAM_OWNER'
        model.validate_parameter_bindings()
        _load_stage(model, reference)
    owned = [set(dict(m.named_parameters())) for m in models]
    assert not owned[0] & owned[1], 'PP_DISJOINT_PARAMETERS'
    assert owned[0] | owned[1] == set(dict(reference.named_parameters())), 'PP_COMPLETE_PARAMETERS'
    for lengths in ([5, 3], [3, 6]):
        ids = torch.arange(3, 3 + sum(lengths))
        batch = PackedBatch(ids, ids.roll(-1), torch.tensor(lengths), torch.arange(len(ids)).float() % 3)
        expected = serial.forward_step(reference, batch)
        outgoing = stages[0].forward_step(models[0], batch)
        assert 'backward' not in outgoing and 'loss' not in outgoing, 'PP_NO_CALLBACK'
        wire = outgoing['hidden_states']
        assert wire.dtype == torch.float32 and wire.shape == (1, len(ids), 99), 'PP_PAIRED_WIRE'
        received = wire.detach().clone().requires_grad_()
        models[1].set_input_tensor(received)
        actual = stages[1].forward_step(models[1], batch)
        for key in ('logits', 'log_probs', 'loss'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg='PP_PAIRED_' + key)
        expected['loss'].backward()
        actual['loss'].backward()
        wire.backward(received.grad)
        for model in models:
            for name, p in model.named_parameters():
                q = dict(reference.named_parameters())[name]
                assert (p.grad is None) == (q.grad is None), 'PP_GRAD_OWNER:' + name
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg='PP_GRAD:' + name)
        for model in [reference, *models]:
            model.zero_grad(set_to_none=True)


@pytest.mark.parametrize('cut', [15, 19, 21, 0, 40])
def test_pp2_rejects_untransported_csa2_state(model_config, cut):
    with pytest.raises(NotImplementedError, match='^V4.1_PP_CSA2_PAYLOAD_UNSUPPORTED:'):
        protocol.build_model(model_config, impl_cfg=replace(_impl(),
            parallel=ParallelConfig(pp=2), pipeline_split_layer=cut))


def _seed_engram(model):
    with torch.no_grad():
        for block in model.layers:
            if block is not None and block.engram is not None:
                table = block.engram.embed
                rows = torch.linspace(-0.25, 0.25, table.weight.numel(), device=table.weight.device).reshape(table.weight.shape)
                table.weight.copy_(rows.to(table.weight.dtype))
                if table.master is not None:
                    table.master.copy_(table.weight.float())


def _pp_worker(rank, config, dtype, trainable, directory):
    import json
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist
    from megatron.lite.primitive.train_step import run_microbatch_loop
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
    from megatron.lite.runtime.contracts.handle import ModelHandle

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.manual_seed(17)
    impl = _impl(dtype, trainable, f'cuda:{rank}')
    serial = protocol.build_model(config, impl_cfg=impl)
    reference = serial.chunks[0]
    _seed_engram(reference)
    dist.init_process_group('nccl', init_method=(Path(directory) / 'rendezvous').as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        bundle = protocol.build_model(config, impl_cfg=replace(impl, parallel=ParallelConfig(pp=2)))
        model = bundle.chunks[0]
        _load_stage(model, reference)
        ps = bundle.parallel_state
        assert ps.pp_size == 2 and ps.dp_size == ps.cp_size == ps.ep_size == 1, 'PP_REAL_WORLD'
        counts = {'forward': 0, 'backward': 0}
        seen_logits, seen_loss = [], []

        def count_backward(grad):
            counts['backward'] += 1
            return grad

        def forward(module, batch):
            counts['forward'] += 1
            output = bundle.forward_step(module, batch)
            assert 'backward' not in output, 'PP_TEXT_NO_CALLBACK'
            if 'logits' in output:
                seen_logits.append(output['logits'].detach().clone())
                seen_loss.append(output['loss'].detach().clone())
            tensor = output['loss'] if ps.pp_is_last else output['hidden_states']
            if tensor.requires_grad:
                tensor.register_hook(count_backward)
            return output

        handle = ModelHandle(model=model, parallel_state=ps,
            _extras={**bundle.extras, 'forward_step': forward, 'model_chunks': [model]})
        runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
        batches = []
        for lengths in ([5, 3], [3, 6]):
            ids = torch.arange(3, 3 + sum(lengths), device=rank)
            batches.append(PackedBatch(ids, ids.roll(-1), torch.tensor(lengths, device=rank),
                torch.arange(len(ids), device=rank).float() % 3))
        with torch.no_grad():
            expected_logits = [serial.forward_step(reference, batch)['logits'] for batch in batches]
            runtime.forward_backward(handle, iter(batches), num_microbatches=2, forward_only=True)
        if ps.pp_is_last:
            for actual, expected in zip(seen_logits, expected_logits, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg='PP_GPU_FORWARD_EXACT')
        counts.update(forward=0, backward=0)
        seen_logits.clear()
        seen_loss.clear()
        reference_losses = []

        def reference_forward(module, batch):
            out = serial.forward_step(module, batch)
            reference_losses.append(out['loss'].detach().clone())
            return out

        run_microbatch_loop(reference, iter(batches), 2, reference_forward,
            prepare_microbatches=serial.extras['prepare_microbatches'])
        runtime.forward_backward(handle, iter(batches), num_microbatches=2)
        assert counts == {'forward': 2, 'backward': 2}, 'PP_NO_IMAGE_PARTICIPATION'
        if ps.pp_is_last:
            for actual, expected in zip(seen_loss, reference_losses, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg='PP_TOKEN_NORMALIZATION')
        gradient_max_abs = 0.0
        for name, p in model.named_parameters():
            q = dict(reference.named_parameters())[name]
            assert (p.grad is None) == (q.grad is None), 'PP_GPU_GRAD_OWNER:' + name
            if p.grad is not None:
                gradient_max_abs = max(gradient_max_abs, float((p.grad - q.grad).abs().max()))
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg='PP_GPU_GRAD_EXACT:' + name)
        for block in model.layers:
            if block is not None and block.engram is not None:
                table = block.engram.embed
                assert table.weight.is_cuda and table.scale.is_cuda, 'PP_ENGRAM_RESIDENCY'
                if trainable:
                    assert table.master.is_cuda and table.master.grad is not None, 'PP_ENGRAM_TRAINABLE'
        Path(directory, f'rank-{rank}.json').write_text(json.dumps(dict(
            gradient_max_abs=gradient_max_abs, logits_max_abs=0.0, loss_max_abs=0.0,
            counts=counts, layer_range=model.local_layer_range, pipeline_dtype=str(bundle.extras['pipeline_dtype']))))
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(2)
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('trainable', [False, True])
def test_pp2_runtime_forward_backward(model_config, dtype, trainable, tmp_path):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() >= 2, 'PP requires two Slurm-allocated GPUs'
    mp.spawn(_pp_worker, args=(model_config, dtype, trainable, str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        print(f'PP2 dtype={dtype} trainable={trainable} rank={rank}: '
            f'{(tmp_path / f"rank-{rank}.json").read_text()}')
