# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PP shifted mHC boundary: values, independent graph edges and packed order."""
from dataclasses import replace

import pytest
import torch
from test_redo_parity import compare_execution, release_config


@pytest.mark.parametrize('inject_engram', [False, True])
def test_shifted_boundary_preserves_value_and_reference_vjp(inject_engram):
    import megatron.lite.primitive.modules.paired_stream as _imports_paired_stream
    from megatron.lite.primitive.modules import deployment_math as math

    pack_deployment = _imports_paired_stream.pack_deployment
    unpack_deployment = _imports_paired_stream.unpack_deployment

    torch.manual_seed(47)
    values = [
        torch.randn(1, 7, 4, 32, dtype=torch.bfloat16),
        torch.randn(1, 7, 4),
        torch.randn(1, 7, 32, dtype=torch.bfloat16),
        torch.randn(1, 7, 4, 32, dtype=torch.bfloat16),
        torch.randn(1, 7, 4),
        torch.randn(1, 7, 4, 4),
    ]
    originals = [x.detach().clone().requires_grad_() for x in values]
    transported = [x.detach().clone().requires_grad_() for x in values]
    hidden, pre, *pending = originals
    stream = hidden if inject_engram else math.reference_post(*pending)
    expected = (stream.float() * pre.unsqueeze(-1)).sum(-2)
    carrier = pack_deployment(transported[0], transported[1], tuple(transported[2:]))
    h, p, pending = unpack_deployment(carrier, (1, 7), 4, 32, torch.bfloat16)
    stream = h if inject_engram else math.reference_post(*pending)
    actual = (stream.float() * p.unsqueeze(-1)).sum(-2)
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for wanted, got in zip(originals, transported, strict=True):
        # An unused side of the concatenated carrier receives zero rather than
        # None; its numerical VJP must remain zero, never another residual edge.
        a = torch.zeros_like(wanted) if wanted.grad is None else wanted.grad
        b = torch.zeros_like(got) if got.grad is None else got.grad
        assert torch.equal(a, b)


def test_carrier_rejects_invalid_pending_contract():
    import megatron.lite.primitive.modules.paired_stream as _imports_paired_stream

    pack_deployment = _imports_paired_stream.pack_deployment
    unpack_deployment = _imports_paired_stream.unpack_deployment

    hidden = torch.ones(1, 3, 4, 32, dtype=torch.bfloat16)
    pre = torch.ones(1, 3, 4)
    carrier = pack_deployment(hidden, pre, None)
    h, p, pending = unpack_deployment(carrier, (1, 3), 4, 32, hidden.dtype)
    assert torch.equal(hidden, h) and torch.equal(pre, p) and pending is None
    carrier[:, 0, -1] = 1
    with pytest.raises(ValueError, match='inconsistent pending post flag'):
        unpack_deployment(carrier, (1, 3), 4, 32, hidden.dtype)
    with pytest.raises(ValueError, match='invalid shifted post/pre carrier'):
        unpack_deployment(carrier[..., :-1], (1, 3), 4, 32, hidden.dtype)


def test_packed_pending_slices_each_document_once(v41_core_te):
    import megatron.lite.model.deepseek_v41.lite.protocol as _imports_protocol

    packed_paired_forward = _imports_protocol.packed_paired_forward

    hidden = torch.arange(5.0).reshape(1, 5, 1, 1).requires_grad_()
    pre = torch.ones(1, 5, 1)
    pending = tuple(hidden.reshape(1, 5, 1) * i for i in range(1, 5))
    seen = []

    def sequence(h, p, *, pending):
        seen.append(tuple(value.flatten().tolist() for value in pending))
        return h + pending[0].unsqueeze(-1), p, pending

    h, p, post = packed_paired_forward(
        sequence, hidden, pre, torch.tensor([0, 2, 5]), pending=pending
    )
    assert seen[0][0] == [0.0, 1.0] and seen[1][0] == [2.0, 3.0, 4.0]
    assert torch.equal(h, hidden * 2)
    assert all(torch.equal(a, b) for a, b in zip(post, pending, strict=True))
    h.sum().backward()
    assert torch.equal(hidden.grad, torch.full_like(hidden, 2))


def test_proxy_split_one_preserves_engram_and_packed_vjp(v41_core_te, monkeypatch):
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.parallel.state import ParallelState
    from megatron.lite.runtime.contracts import ParallelConfig

    cfg = release_config().to_hf_dict()
    text = cfg['text_config']
    text.update(
        num_hidden_layers=2,
        compress_ratios=[0] * 5,
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        candidate_source_layer_id=-1,
        engram_layer_ids=[1],
        engram_num_embeddings=[18],
    )
    cfg = DeepseekV41Config(cfg)
    impl = protocol.ImplConfig(
        device='cpu',
        dtype=torch.float32,
        quantized=False,
        token_map=list(range(64)),
        trainable_engram=False,
    )
    reference = protocol.build_model(cfg, impl_cfg=impl).chunks[0]
    stages = []
    with monkeypatch.context() as patch:
        patch.setattr(torch.distributed, 'is_initialized', lambda: True)
        patch.setattr(torch.distributed, 'get_world_size', lambda *args: 2)
        for rank in range(2):
            ps = ParallelState(
                pp_size=2, pp_rank=rank, pp_is_first=rank == 0, pp_is_last=rank == 1
            )
            patch.setattr(protocol, 'init_parallel', lambda _, ps=ps: ps)
            stages.append(
                protocol.build_model(
                    cfg,
                    impl_cfg=replace(
                        impl, parallel=ParallelConfig(pp=2), pipeline_split_layer=1
                    ),
                ).chunks[0]
            )
    ids = torch.tensor([[2, 3, 9, 4, 11]])
    cu = torch.tensor([0, 2, 5])

    def execute(full, pieces):
        expected = full(ids, cu_seqlens=cu)['logits']
        pieces[1].set_input_tensor(pieces[0](ids, cu_seqlens=cu)['hidden_states'])
        return expected, pieces[1](ids, cu_seqlens=cu)['logits']

    compare_execution(reference, stages, execute)
    assert stages[0].layers[1] is None
    assert stages[1].layers[1].engram.embed.master is None


def _stream_worker(rank, directory):
    from datetime import timedelta
    from types import SimpleNamespace

    import megatron.lite.primitive.ckpt.pipeline_stream as _imports_pipeline_stream
    import torch.distributed as dist

    broadcast_stage_stream = _imports_pipeline_stream.broadcast_stage_stream

    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/stream',
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        ps = SimpleNamespace(
            pp_rank=rank, pp_global_ranks=[0, 1], pp_group=dist.group.WORLD
        )

        def local():
            for i in range(3):
                dtype = (torch.float32, torch.float8_e4m3fn, torch.int8)[i]
                yield f'stage{rank}.{i}', (
                    torch.arange(i + 1, dtype=torch.float32) + rank
                ).to(dtype)

        actual = [
            (name, value.clone()) for name, value in broadcast_stage_stream(local(), ps)
        ]
        assert [name for name, _ in actual] == [
            f'stage{s}.{i}' for s in range(2) for i in range(3)
        ]
        for (name, value), (s, i) in zip(
            actual, [(s, i) for s in range(2) for i in range(3)], strict=True
        ):
            dtype = (torch.float32, torch.float8_e4m3fn, torch.int8)[i]
            wanted = (torch.arange(i + 1, dtype=torch.float32) + s).to(dtype)
            assert value.dtype == dtype
            assert torch.equal(value.view(torch.uint8), wanted.view(torch.uint8))
    finally:
        dist.destroy_process_group()


def test_encoded_pp_stream_actual_gloo(tmp_path):
    torch.multiprocessing.spawn(
        _stream_worker, args=(str(tmp_path),), nprocs=2, join=True
    )


def _optimizer_worker(rank, directory):
    import runpy
    from datetime import timedelta
    from pathlib import Path

    import megatron.core.fp8_utils
    import megatron.core.transformer.experimental_attention_variant.csa
    import megatron.core.transformer.hyper_connection
    import torch.distributed as dist

    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    import megatron.lite.model.deepseek_v41.lite.optimizer_groups as _imports_optimizer_groups
    from megatron.lite.model.deepseek_v41.lite import protocol

    OptimizerConfig = _imports_optimizer_groups.OptimizerConfig
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.set_num_threads(1)
    torch.manual_seed(14)
    impl = protocol.ImplConfig(
        device='cpu',
        dtype=torch.float32,
        quantized=False,
        token_map=list(range(64)),
        trainable_engram=False,
        optimizer='muon',
        optimizer_config=OptimizerConfig(
            lr=0.001, clip_grad=0.25, ns_steps=2, coefficient_type='quintic'
        ),
    )
    reference = protocol.build_model(release_config(), impl_cfg=impl)
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/optimizer',
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        stage = protocol.build_model(
            release_config(), impl_cfg=replace(impl, parallel=ParallelConfig(pp=2))
        )
        full, piece = reference.chunks[0], stage.chunks[0]
        state = full.state_dict()
        piece.load_state_dict({name: state[name] for name in piece.state_dict()})
        for model in (full, piece):
            for p in model.parameters():
                if p.requires_grad:
                    p.grad = p.main_grad = torch.full_like(p, 1 / 64)
        expected = reference.optimizer.step()
        actual = stage.optimizer.step()
        assert actual == expected and actual[0]
        for name, p in piece.named_parameters():
            assert torch.equal(p, dict(full.named_parameters())[name]), name
        before = {name: p.detach().clone() for name, p in piece.named_parameters()}
        for p in piece.parameters():
            if p.requires_grad:
                p.grad = p.main_grad = torch.full_like(
                    p, float('nan') if rank == 1 else 1.0
                )
        result = stage.optimizer.step()
        assert not result[0]
        assert all(torch.equal(p, before[name]) for name, p in piece.named_parameters())
    finally:
        dist.destroy_process_group()


def test_pp_global_clipping_and_failed_transaction_actual_gloo(tmp_path):
    torch.multiprocessing.spawn(
        _optimizer_worker, args=(str(tmp_path),), nprocs=2, join=True
    )


@pytest.mark.gpus(1)
def test_native_shifted_pending_carrier_value_and_vjp():
    import megatron.lite.primitive.modules.paired_stream as _imports_paired_stream
    from megatron.lite.primitive.modules import deployment_math
    from megatron.lite.primitive.modules.attention.mhc import HCMixes

    pack_deployment = _imports_paired_stream.pack_deployment
    unpack_deployment = _imports_paired_stream.unpack_deployment

    if not torch.cuda.is_available():
        pytest.skip('Native PP deployment boundary requires CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(79)
    values = [
        torch.randn(*shape, device='cuda', dtype=dtype)
        for shape, dtype in [
            ((1, 3, 4, 5120), torch.bfloat16),
            ((1, 3, 4), torch.float32),
            ((5120,), torch.float32),
            ((1, 3, 5120), torch.bfloat16),
            ((1, 3, 4, 5120), torch.bfloat16),
            ((1, 3, 4), torch.float32),
            ((1, 3, 4, 4), torch.float32),
        ]
    ]
    a = [value.detach().clone().requires_grad_() for value in values]
    b = [value.detach().clone().requires_grad_() for value in values]
    first = HCMixes(5120, 4, iterations=20).cuda()
    second = HCMixes(5120, 4, iterations=20).cuda()
    first.broadcast_projection = second.broadcast_projection = False
    second.load_state_dict(first.state_dict())
    expected = deployment_math.mhc_joint(a[0], a[1], a[2], first, tuple(a[3:]))
    carrier = pack_deployment(b[0], b[1], tuple(b[3:]))
    h, pre, pending = unpack_deployment(carrier, (1, 3), 4, 5120, torch.bfloat16)
    actual = deployment_math.mhc_joint(h, pre, b[2], second, pending)
    assert all(torch.equal(x, y) for x, y in zip(actual, expected, strict=True))
    sum(x.float().square().sum() for x in expected).backward()
    sum(x.float().square().sum() for x in actual).backward()
    for x, y in zip(
        a + list(first.parameters()), b + list(second.parameters()), strict=True
    ):
        gx = torch.zeros_like(x) if x.grad is None else x.grad
        gy = torch.zeros_like(y) if y.grad is None else y.grad
        assert torch.equal(gx, gy)
    print(
        'NATIVE_PP2_PENDING_VALUE_VJP_PASS width=5120 hc=4 outputs=5 all_live_operand_gradients_equal=true'
    )
