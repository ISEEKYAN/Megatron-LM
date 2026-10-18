# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run with PYTHONPATH=/tmp/mrg:experimental/lite for nv/dev core.

New model features require its hyper_connection operators; the reference is
the same concrete unpartitioned model. Run preservation tests separately.
"""
from dataclasses import replace

import pytest
import torch


def release_config():
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    text = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=40,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=4,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=1,
        index_n_heads=1,
        index_head_dim=32,
        index_topk=2,
        sliding_window=4,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        candidate_source_layer_id=20,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        hidden_act='silu',
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        norm_topk_prob=True,
        topk_method='noaux_tc',
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_scaling=dict(
            rope_type='yarn',
            factor=16,
            beta_fast=32,
            beta_slow=1,
            original_max_position_embeddings=65536,
        ),
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        scoring_func='sqrtsoftplus',
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        rms_norm_eps=1e-20,
        hc_mult=2,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=3,
        engram_n_heads=1,
        engram_vocab_size=7,
        engram_num_embeddings=[18, 30],
        engram_head_dim=32,
        engram_compressed_vocab_size=64,
        engram_pad_token_id=0,
        num_nextn_predict_layers=3,
        dspark_n_routed_experts=2,
    )
    return DeepseekV41Config(
        dict(
            model_type='deepseek_v41',
            text_config=text,
            vision_config=dict(
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=1,
                intermediate_size=16,
                patch_size=2,
                rope_theta=10000,
                downsample_ratio=2,
            ),
            quantization_config=dict(
                quant_method='fp8',
                activation_scheme='dynamic',
                weight_block_size=[32, 32],
                scale_fmt='ue8m0',
                expert_dtype='fp4',
            ),
        )
    )


@pytest.fixture(
    params=[False, pytest.param(True, marks=pytest.mark.gpus(1))],
    ids=['floating', 'default_quantized'],
)
def bundle(v41_core_te, request, monkeypatch):
    from megatron.lite.model.deepseek_v41.lite import protocol

    torch.manual_seed(351)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    impl = protocol.ImplConfig(
        device='cuda' if request.param else 'cpu',
        dtype=torch.float32,
        token_map=list(range(64)),
        trainable_engram=True,
    )
    assert impl.quantized is True
    if not request.param:
        impl = replace(impl, quantized=False)
    return protocol.build_model(release_config(), impl_cfg=impl), impl


def compare_execution(reference, candidates, execute):
    """One value/gradient criterion, independently supplied partition execution."""
    for candidate in candidates:
        owned = candidate.state_dict()
        candidate.load_state_dict(
            {k: v for k, v in reference.state_dict().items() if k in owned}
        )
    expected, actual = execute(reference, candidates)
    assert torch.equal(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    originals = dict(reference.named_parameters())
    for candidate in candidates:
        for name, value in candidate.named_parameters():
            wanted = originals[name].grad
            assert (wanted is None) == (value.grad is None), name
            if wanted is not None:
                assert torch.equal(value.grad, wanted), name


def test_real_pp2_boundary_restarts_attention_state(bundle, monkeypatch):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.parallel.state import ParallelState
    from megatron.lite.runtime.contracts import ParallelConfig

    reference_bundle, impl = bundle
    reference = reference_bundle.chunks[0]
    pieces = []
    with monkeypatch.context() as patch:
        patch.setattr(torch.distributed, 'is_initialized', lambda: True)
        patch.setattr(torch.distributed, 'get_world_size', lambda *args: 2)
        for rank in range(2):
            ps = ParallelState(
                pp_size=2, pp_rank=rank, pp_is_first=rank == 0, pp_is_last=rank == 1
            )
            patch.setattr(protocol, 'init_parallel', lambda _, ps=ps: ps)
            pieces.append(
                protocol.build_model(
                    release_config(),
                    impl_cfg=replace(impl, parallel=ParallelConfig(pp=2)),
                ).chunks[0]
            )
    seen = []
    pieces[1].layers[20].attn.register_forward_pre_hook(
        lambda _, args: seen.append(args[1])
    )
    ids = torch.tensor([[2, 3, 9, 4, 11, 7]], device=reference.head.weight.device)

    def execute(full, stages):
        expected = full(ids)['logits']
        outgoing = stages[0](ids)['hidden_states']
        assert outgoing.shape == (1, 6, 66)
        stages[1].set_input_tensor(outgoing)
        return expected, stages[1](ids)['logits']

    compare_execution(reference, pieces, execute)
    assert len(seen) == 1
    assert all(value is None for value in vars(seen[0]).values())

    # Every stage publishes its own router statistics to the optimizer.
    from unittest.mock import Mock

    from megatron.lite.runtime.contracts.data import PackedBatch

    batch = PackedBatch(
        ids[0], labels=None, seq_lens=torch.tensor([6], device=ids.device)
    )
    for stage in pieces:
        if stage is pieces[1]:
            stage.set_input_tensor(
                protocol._forward_step(pieces[0], batch)['hidden_states']
            )
        optimizer = Mock()
        protocol._forward_step(stage, batch, optimizer=optimizer)
        optimizer.accumulate_modality_loads.assert_called_once()
        assert len(optimizer.accumulate_modality_loads.call_args.args[0]) == 20


def test_optimizer_two_steps_and_nonfinite_transaction(bundle, tmp_path):
    import torch.distributed as dist
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import V41Optimizer
    from megatron.lite.model.deepseek_v41.vision_config import OptimizerConfig

    full, _ = bundle
    model = full.chunks[0]
    optimizer = V41Optimizer(
        model, OptimizerConfig(lr=1e-4, ns_steps=2, coefficient_type='quintic')
    )
    assert optimizer.owns_param_group_policy is True
    ids = torch.tensor([[1, 8, 3, 6]], device=model.head.weight.device)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(2):
        optimizer.zero_grad()
        output = model(ids)
        optimizer.accumulate_modality_loads(output['modality_loads'])
        output['logits'].square().mean().backward()
        assert optimizer.step()[0]
    assert any(not torch.equal(p, initial[n]) for n, p in model.named_parameters())
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    next(p for p in model.parameters() if p.grad is not None).grad.flatten()[0] = float(
        'nan'
    )
    assert not optimizer.step()[0]
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())


def test_archival_export_is_byte_preserving_and_reloadable(bundle, tmp_path):
    from megatron.lite.model.deepseek_v41.lite import checkpoint
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader
    from safetensors.torch import save_file

    full, impl = bundle
    model = full.chunks[0]
    archive_dir = tmp_path / 'archive'
    archive_dir.mkdir()
    archive = {
        name: torch.arange(17, dtype=torch.uint8) for name in model.archival_bindings
    }
    save_file(archive, str(archive_dir / 'model.safetensors'))
    model.archival_store = SafeTensorReader(str(archive_dir))
    model.archival_keys = sorted(archive)
    exported = dict(
        checkpoint.export_checkpoint(model, export_dtype='float32', cpu=True)
    )
    assert all(
        torch.equal(exported[name], original) for name, original in archive.items()
    )
    assert all(exported[name].dtype == torch.uint8 for name in archive)
    with pytest.raises(TypeError):
        list(checkpoint.export_checkpoint(model, invented_option=True))
    checkpoint.save_model(model, tmp_path / 'saved', buffer_max_size_bytes=65536)
    from megatron.lite.model.deepseek_v41.lite import protocol

    loaded = protocol.build_model(release_config(), impl_cfg=impl).chunks[0]
    checkpoint.load_model(loaded, tmp_path / 'saved')
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, dict(loaded.named_parameters())[name]), name
    ids = torch.tensor([[2, 4, 7, 3]], device=model.head.weight.device)
    assert torch.equal(model(ids)['logits'], loaded(ids)['logits'])


def test_packed_loss_head_gradient_with_ddp_unused_detection(bundle, tmp_path):
    import torch.distributed as dist
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts.data import PackedBatch
    from torch.nn.parallel import DistributedDataParallel

    model = bundle[0].chunks[0]
    ids = torch.tensor([2, 4, 7, 3], device=model.head.weight.device)
    batch = PackedBatch(ids, ids.clone(), torch.tensor([4], dtype=torch.int32))
    protocol._forward_step(model, batch)['loss'].backward()
    expected = model.head.weight.grad.clone()
    dist.init_process_group(
        'gloo', init_method=f'file://{tmp_path}/ddp', rank=0, world_size=1
    )
    try:
        ddp = DistributedDataParallel(
            model, broadcast_buffers=False, find_unused_parameters=True
        )
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            protocol._forward_step(model, batch, execution_model=ddp)['loss'].backward()
            assert torch.equal(model.head.weight.grad, expected)
    finally:
        dist.destroy_process_group()


def _remote_nonfinite_worker(rank, directory, parallel, scenario):
    """Two real Gloo ranks hold the same replicated owners; rank 1 alone fails."""
    import runpy
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist

    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import V41Optimizer
    from megatron.lite.model.deepseek_v41.vision_config import OptimizerConfig

    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/store',
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=300),
    )
    try:
        torch.manual_seed(351)
        impl = protocol.ImplConfig(
            device='cpu',
            dtype=torch.float32,
            token_map=list(range(64)),
            trainable_engram=True,
            # Sharded Engram rows need CUDA; replicate the table on CPU ranks.
            shard_engram=False,
            quantized=False,
        )
        model = protocol.build_model(release_config(), impl_cfg=impl).chunks[0]
        opt = V41Optimizer(
            model,
            OptimizerConfig(lr=1e-3, ns_steps=2, coefficient_type='quintic'),
            dp_group=dist.group.WORLD,
            ps=replace(model.ps, **parallel),
        )
        output = model(torch.tensor([[1, 3, 5, 7]]))
        opt.accumulate_modality_loads(output['modality_loads'])
        output['logits'].square().mean().backward()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        biases = [block.ffn.gate.bias.clone() for block in model.layers]
        if scenario == 'reduced_nan':
            # A non-finite gradient on one replica reaches every replica
            # through the DP/CP gradient reduction that precedes the step.
            if rank == 1:
                next(p for p in model.parameters() if p.grad is not None).grad[...] = (
                    float('nan')
                )
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad.div_(2)
        elif scenario == 'remote_candidate' and rank == 1:
            # Only rank 1's prepared update is non-finite (the remote vote).
            opt._local_finite = lambda tensors: False
        stepped = opt.step()[0]
        unchanged = all(torch.equal(p, before[n]) for n, p in model.named_parameters())
        biases_unchanged = all(
            torch.equal(b.ffn.gate.bias, old) for b, old in zip(model.layers, biases)
        )
        torch.save(
            dict(stepped=stepped, unchanged=unchanged, biases=biases_unchanged),
            f'{directory}/{rank}.pt',
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('scenario', ['control', 'reduced_nan', 'remote_candidate'])
@pytest.mark.parametrize('parallel', [dict(cp_size=2), dict(dp_size=2)])
def test_remote_nonfinite_skips_replicated_optimizer(
    v41_core_te, tmp_path, parallel, scenario
):
    """Real two-rank Gloo: one rank's non-finite update skips the step everywhere.

    The vote is one MIN all-reduce over every training rank, so no rank
    publishes while another holds a non-finite gradient or update.
    """
    import torch.multiprocessing as mp

    mp.start_processes(
        _remote_nonfinite_worker,
        args=(str(tmp_path), parallel, scenario),
        nprocs=2,
        join=True,
        start_method='spawn',
    )
    results = [torch.load(tmp_path / f'{rank}.pt') for rank in range(2)]
    if scenario == 'control':
        assert all(r['stepped'] and not r['unchanged'] for r in results), results
    else:
        assert all(
            not r['stepped'] and r['unchanged'] and r['biases'] for r in results
        ), results


def test_image_tokens_backpropagate_into_vision_and_aligner(bundle):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.modules.image_data import ImageInput, image_token_types
    from megatron.lite.primitive.modules.vision_training import VisionTrainability

    impl = replace(
        bundle[1],
        text_only=False,
        vision_trainability=VisionTrainability(True, True, True, True),
    )
    model = protocol.build_model(release_config(), impl_cfg=impl).chunks[0]
    ids = torch.arange(8, device=impl.device)[None]
    types = image_token_types(1, 1).to(impl.device)
    image = ImageInput(1, torch.randn(4, 3, 2, 2, device=impl.device), 2, 2, types)
    model(ids, images=[[image]])['logits'].square().sum().backward()
    for module in (model.vision, model.aligner):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) > 0 for g in gradients)
