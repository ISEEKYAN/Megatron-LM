# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent resident-reference updates, numerical rejection and new-process restore."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.primitive.optimizers.headwise_muon import MixedOptimizer


def build(segmented, device='cpu'):
    model = torch.nn.ParameterDict(
        {
            'matrix': torch.nn.Parameter(
                torch.linspace(-0.3, 0.4, 32, device=device).reshape(2, 4, 4)
            ),
            'rows': torch.nn.Parameter(
                torch.linspace(-0.2, 0.5, 12, device=device).reshape(3, 4)
            ),
            'norm': torch.nn.Parameter(torch.linspace(0.1, 0.4, 4, device=device)),
            'late': torch.nn.Parameter(
                torch.linspace(-0.7, 0.9, 16, device=device).reshape(4, 4)
            ),
        }
    )
    config = SimpleNamespace(
        lr=1e-3,
        clip_grad=0.25,
        ns_steps=2,
        coefficient_type='quintic',
        segmented_host=segmented,
    )
    groups = lambda: [
        dict(
            params=[model[n]],
            algorithm=a,
            owner_key=n,
            matrix_shape=tuple(model[n].shape),
            weight_decay=0.1,
        )
        for n, a in [
            ('matrix', 'muon'),
            ('late', 'muon'),
            ('rows', 'sinkhorn'),
            ('norm', 'adamw'),
        ]
    ]
    opt = MixedOptimizer(
        model,
        config,
        group_builder=groups,
        owners=lambda: ([], [], [], None),
        stats_factory=SimpleNamespace,
    )
    return model, opt


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert (
            a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
        )
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


def update(model, opt, iteration):
    opt.zero_grad()
    sum(
        (i + 1) * (p + iteration * 0.1).square().sum()
        for i, p in enumerate(model.parameters())
    ).backward()
    return opt.step()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_five_updates_and_resume_bitwise_against_original(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('actual CUDA update/offload equality requires a GPU')
    model, host = build(True, device)
    reference, resident = build(False, device)
    for iteration in range(5):
        assert update(model, host, iteration) == update(reference, resident, iteration)
        equal(model.state_dict(), reference.state_dict())
        equal(host.state_dict(), resident.state_dict())
        assert all(v.device.type == 'cpu' for v in host.host_update.masters.values())
        assert all(
            v.device.type == 'cpu'
            for b in host.optimizers
            for s in b.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )
        host.load_state_to_device()
        assert all(
            v.device.type == 'cpu'
            for b in host.optimizers
            for s in b.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )
    torch.save(
        {'model': model.state_dict(), 'optimizer': host.state_dict()},
        tmp_path / 'checkpoint.pt',
    )
    restarted, restarted_opt = build(True, device)
    checkpoint = torch.load(
        tmp_path / 'checkpoint.pt', map_location=device, weights_only=False
    )
    restarted.load_state_dict(checkpoint['model'])
    restarted_opt.load_state_dict(checkpoint['optimizer'])
    equal(restarted_opt.state_dict(), host.state_dict())
    for iteration in range(5, 8):
        assert update(restarted, restarted_opt, iteration) == update(
            reference, resident, iteration
        )
        equal(restarted.state_dict(), reference.state_dict())
        equal(restarted_opt.state_dict(), resident.state_dict())
    import megatron.lite.runtime.backends.mlite.runtime as runtime

    MegatronLiteRuntime = runtime.MegatronLiteRuntime

    handle = SimpleNamespace(
        _model=restarted,
        _optimizer=restarted_opt,
        _extras={
            'model_chunks': [restarted],
            'pre_model_device_transfer_hook': restarted_opt.host_update.prepare_model_transfer,
        },
    )
    MegatronLiteRuntime.__new__(MegatronLiteRuntime).to(
        handle, 'cpu', model=True, optimizer=True, grad=True
    )
    assert all(
        p.grad is None and p.main_grad is None and p.device.type == 'cpu'
        for p in restarted.parameters()
    )
    assert all(
        p.data_ptr() == restarted_opt.host_update.masters[id(p)].data_ptr()
        for p in restarted.parameters()
    )


def test_late_nonfinite_candidate_does_not_publish_any_owner():
    model, opt = build(True)
    assert update(model, opt, 0)[0]
    opt.zero_grad()
    for p in model.parameters():
        p.grad = p.main_grad = torch.ones_like(p)
    backend = next(b for b in opt.optimizers if isinstance(b, torch.optim.AdamW))
    # Finite gradients/norm but a late AdamW candidate overflows. Earlier Muon
    # and Sinkhorn owners must retain their original masters and moments.
    backend.param_groups[0]['lr'] = 1e10
    backend.param_groups[0]['weight_decay'] = 1e10
    with torch.no_grad():
        model['norm'].fill_(1e30)
    weights = deepcopy(model.state_dict())
    state = deepcopy(opt.state_dict())
    result = opt.step()
    assert not result[0] and torch.isfinite(torch.tensor(result[1]))
    equal(model.state_dict(), weights)
    equal(opt.state_dict(), state)
    assert not opt.host_update.broken


def test_nonfinite_gradient_and_checkpoint_while_busy_fail_closed():
    model, opt = build(True)
    opt.zero_grad()
    for p in model.parameters():
        p.grad = p.main_grad = torch.ones_like(p)
    next(model.parameters()).grad.flatten()[0] = float('inf')
    original = deepcopy(model.state_dict())
    assert not opt.step()[0]
    equal(model.state_dict(), original)
    assert all(not b.state for b in opt.optimizers)
    opt.host_update.busy = True
    with pytest.raises(RuntimeError, match='busy'):
        opt.state_dict()


def _distributed_rejection(rank, directory):
    import torch.distributed as dist

    dist.init_process_group(
        'gloo', init_method=f'file://{directory}/init', rank=rank, world_size=4
    )
    try:
        dense = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        pipeline = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        stage = rank // 2
        ps = SimpleNamespace(
            pp_size=2,
            pp_rank=stage,
            pp_group=pipeline[rank % 2],
            ep_size=1,
            dp_size=2,
            dp_cp_size=2,
            dp_group=dense[stage],
        )
        model = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.full((2,), 1e30 if stage == 1 and i == 2 else 0.5)
                )
                for i in range(2 + stage)
            ]
        )
        groups = lambda: [
            dict(
                params=[p],
                algorithm='adamw',
                owner_key=str(i),
                lr=1e10 if stage == 1 and i == 2 else 0.001,
                weight_decay=1e10 if stage == 1 and i == 2 else 0.1,
            )
            for i, p in enumerate(model)
        ]
        opt = MixedOptimizer(
            model,
            SimpleNamespace(lr=0.001, clip_grad=0.25, segmented_host=True),
            group_builder=groups,
            owners=lambda: ([], [], [], None),
            stats_factory=None,
            ps=ps,
            dp_group=dense[stage],
        )
        for p in model:
            p.grad = p.main_grad = torch.ones_like(p)
        weights = deepcopy(model.state_dict())
        result = opt.step()
        assert not result[0] and torch.isfinite(torch.tensor(result[1]))
        equal(weights, model.state_dict())
        assert all(not backend.state for backend in opt.optimizers)
        all_results = [None] * 4
        dist.all_gather_object(all_results, result)
        assert all(r == result for r in all_results)

        import megatron.lite.primitive.ckpt.dcp as dcp

        save_training_checkpoint = dcp.save_training_checkpoint
        load_training_checkpoint = dcp.load_training_checkpoint

        with torch.no_grad():
            for parameter in model:
                parameter.fill_(0.5)
        for backend in opt.optimizers:
            for group in backend.param_groups:
                group['lr'], group['weight_decay'] = 0.001, 0.1
        assert opt.step()[0]
        opt.host_update.prepare_model_transfer('cpu')
        published, state = deepcopy(model.state_dict()), deepcopy(opt.state_dict())
        save_training_checkpoint(
            model,
            opt,
            1,
            str(directory) + '/distributed-cp',
            config=SimpleNamespace(),
            ps=ps,
            save_rng=False,
        )
        with torch.no_grad():
            for parameter in model:
                parameter.fill_(3)
        assert (
            load_training_checkpoint(
                model,
                opt,
                str(directory) + '/distributed-cp',
                config=SimpleNamespace(),
                ps=ps,
                load_rng=False,
            )
            == 1
        )
        equal(published, model.state_dict())
        equal(state, opt.state_dict())
    finally:
        dist.destroy_process_group()


def test_late_remote_stage_rejects_all_dense_and_pipeline_ranks(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_rejection, args=(str(tmp_path),), nprocs=4, join=True
    )


def test_real_DCP_host_checkpoint_and_restored_updates_match_reference(tmp_path):
    import megatron.lite.primitive.ckpt.dcp as dcp

    save_training_checkpoint = dcp.save_training_checkpoint
    load_training_checkpoint = dcp.load_training_checkpoint

    model, opt = build(True)
    reference, resident = build(False)
    for i in range(3):
        assert update(model, opt, i) == update(reference, resident, i)
    opt.host_update.prepare_model_transfer('cpu')
    ps = SimpleNamespace(pp_size=1, pp_rank=0)
    save_training_checkpoint(
        model,
        opt,
        3,
        str(tmp_path / 'dcp'),
        config=SimpleNamespace(),
        ps=ps,
        save_rng=False,
    )
    restored, restored_opt = build(True)
    assert (
        load_training_checkpoint(
            restored,
            restored_opt,
            str(tmp_path / 'dcp'),
            config=SimpleNamespace(),
            ps=ps,
            load_rng=False,
        )
        == 3
    )
    equal(model.state_dict(), restored.state_dict())
    equal(opt.state_dict(), restored_opt.state_dict())
    assert all(
        p.data_ptr() == restored_opt.host_update.masters[id(p)].data_ptr()
        for p in restored.parameters()
    )
    for i in range(3, 6):
        assert update(restored, restored_opt, i) == update(reference, resident, i)
        equal(restored.state_dict(), reference.state_dict())
        equal(restored_opt.state_dict(), resident.state_dict())
