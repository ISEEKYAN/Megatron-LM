# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PP2 optimizer clipping and finite vote equal the unpartitioned model's.

Both stages receive exactly the whole model's gradients (no pipeline numerics),
so the global norm, the clip factor and the updated parameters must be bitwise
PP1's, for resident and segmented-host owners alike; one non-finite stage must
stop publication on every stage.
"""
from dataclasses import replace

import pytest
import torch
from test_pp_stage_training import prefix_config

CLIP = 0.5


def _clip_factor(norm):
    return min(1.0, CLIP / (norm + 1e-6))


def _worker(rank, directory, segmented):
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
    from megatron.lite.model.deepseek_v41.vision_config import OptimizerConfig
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.set_num_threads(1)
    impl = protocol.ImplConfig(
        device='cpu',
        dtype=torch.float32,
        quantized=False,
        shard_engram=False,
        optimizer='muon',
        optimizer_config=OptimizerConfig(
            lr=1e-4,
            ns_steps=2,
            coefficient_type='quintic',
            clip_grad=CLIP,
            segmented_host=segmented,
        ),
    )
    torch.manual_seed(91)
    reference = protocol.build_model(prefix_config(), impl_cfg=impl)
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
            prefix_config(),
            impl_cfg=replace(
                impl, parallel=ParallelConfig(pp=2), pipeline_split_layer=2
            ),
        )
        stage, optimizer = bundle.chunks[0], bundle.optimizer
        own = stage.state_dict()
        stage.load_state_dict({k: v for k, v in full.state_dict().items() if k in own})
        whole = dict(full.named_parameters())
        local = dict(stage.named_parameters())
        generator = torch.Generator().manual_seed(7)
        norms = []
        for _ in range(2):
            ref_opt.zero_grad()
            optimizer.zero_grad()
            grads = {
                name: torch.randn(p.shape, generator=generator)
                for name, p in whole.items()
                if p.requires_grad
            }
            for params in (whole, local):
                for name, p in params.items():
                    if name in grads:
                        p.grad = p.main_grad = grads[name].clone()
            ref_opt.finalize_grads()
            optimizer.finalize_grads()
            expected, actual = ref_opt.step(), optimizer.step()
            assert expected[0] and actual[0]
            # Bitwise: the same float64 norm, hence the same clip factor.
            assert actual[1] == expected[1], (actual[1], expected[1])
            assert _clip_factor(actual[1]) == _clip_factor(expected[1]) < 1
            for name, p in local.items():
                assert torch.equal(p, whole[name]), name
            norms.append(actual[1])
        # One non-finite stage stops publication on both stages.
        before = {n: p.detach().clone() for n, p in local.items()}
        optimizer.zero_grad()
        for p in stage.parameters():
            if p.requires_grad:
                p.grad = p.main_grad = torch.ones_like(p)
        if rank == 0:
            next(p for p in stage.parameters() if p.requires_grad).grad.flatten()[0] = (
                float('nan')
            )
        optimizer.finalize_grads()
        assert not optimizer.step()[0]
        assert all(torch.equal(p, before[n]) for n, p in local.items())
        torch.save({'norms': norms}, Path(directory) / f'rank-{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('segmented', [False, True], ids=['resident', 'host'])
def test_pp2_clip_and_finite_vote_match_unpartitioned(v41_core_te, tmp_path, segmented):
    import torch.multiprocessing as mp

    mp.spawn(_worker, args=(str(tmp_path), segmented), nprocs=2, join=True)
    norms = [
        torch.load(tmp_path / f'rank-{rank}.pt', weights_only=True)['norms']
        for rank in range(2)
    ]
    assert norms[0] == norms[1]
