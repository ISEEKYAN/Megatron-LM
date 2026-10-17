# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Resident optimizer under PP: one global clip norm and one non-finite vote.

PP2 stages own disjoint owners; their clipped update must equal PP1's bits.
"""
from types import SimpleNamespace

import torch
from megatron.lite.primitive.optimizers.headwise_muon import MixedOptimizer

OWNERS = [
    ('layers.0.attn.wq', 'muon', (8, 6)),
    ('layers.0.norm', 'adamw', (6,)),
    ('layers.1.attn.wq', 'muon', (6, 8)),
    ('layers.2.attn.wq', 'muon', (8, 8)),
    ('layers.3.ffn.w1', 'muon', (4, 8)),
    ('layers.3.norm', 'adamw', (8,)),
]
STAGES = (range(0, 3), range(3, 6))


def _build(indices, ps=None):
    model = torch.nn.ParameterDict(
        {
            str(i): torch.nn.Parameter(
                torch.linspace(-0.4 + i / 10, 0.6, torch.Size(OWNERS[i][2]).numel())
                .reshape(OWNERS[i][2])
                .double()
                .float()
            )
            for i in indices
        }
    )
    groups = lambda: [
        dict(
            params=[model[str(i)]],
            algorithm=OWNERS[i][1],
            owner_key=OWNERS[i][0],
            matrix_shape=OWNERS[i][2],
            weight_decay=0.1,
        )
        for i in indices
    ]
    opt = MixedOptimizer(
        model,
        SimpleNamespace(
            lr=1e-2, clip_grad=0.05, ns_steps=2, coefficient_type='quintic'
        ),
        group_builder=groups,
        owners=lambda: ([], [], [], None),
        stats_factory=SimpleNamespace,
        ps=ps,
    )
    return model, opt


def _grads(model, step, poison=None):
    for name, p in model.items():
        i = int(name)
        g = torch.sin(torch.arange(p.numel(), dtype=torch.float32) * (i + 1) + step)
        g = g.reshape(p.shape) * (3.0**i)
        if poison == i:
            g[0] = float('inf')
        p.grad = p.main_grad = g


def _pp2(rank, directory, reference):
    import torch.distributed as dist

    dist.init_process_group(
        'gloo', init_method=f'file://{directory}/init', rank=rank, world_size=2
    )
    try:
        ps = SimpleNamespace(
            pp_size=2,
            pp_rank=rank,
            pp_group=dist.group.WORLD,
            ep_size=1,
            dp_size=1,
            dp_cp_size=1,
            ep_group=None,
        )
        model, opt = _build(STAGES[rank], ps)
        for step in range(3):
            _grads(model, step)
            ok, norm, _ = opt.step()
            assert ok and norm == reference['norms'][step]
            for name, p in model.items():
                assert torch.equal(p, reference['params'][step][name]), (step, name)
        weights = {k: v.detach().clone() for k, v in model.items()}
        # A late non-finite owner on stage1 alone rejects both stages.
        _grads(model, 3, poison=4)
        assert not opt.step()[0]
        for name, p in model.items():
            assert torch.equal(p, weights[name])
    finally:
        dist.destroy_process_group()


def test_pp2_resident_clip_and_vote_match_pp1(tmp_path):
    model, opt = _build(range(len(OWNERS)))
    reference = {'norms': [], 'params': []}
    for step in range(3):
        _grads(model, step)
        ok, norm, _ = opt.step()
        assert ok and norm > opt.config.clip_grad
        reference['norms'].append(norm)
        reference['params'].append({k: v.detach().clone() for k, v in model.items()})
    torch.multiprocessing.spawn(
        _pp2, args=(str(tmp_path), reference), nprocs=2, join=True
    )
