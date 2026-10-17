# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 training resume through DCP equals continuous training, bit for bit.

Ported from the -clean chain's ``test_v41_g1_training_checkpoint_bitwise``: three
continuous steps against one step, a training checkpoint, and two steps resumed in
a fresh process group that shares nothing with the first but the checkpoint on disk.
The resumed model, optimizer, schedulers, RNG-driven batches, losses and learning
rates must equal the continuous run's. Reconstruction consumes RNG before the load,
so a missing RNG restore changes the batches. A corrupted load and a vanished head
gradient must be caught.

The checkpoint goes through ``save_training_checkpoint`` / ``load_training_checkpoint``
with DCP as the training engine calls them. With the segmented host optimizer (the
recipe's), parameters are offloaded to their FP32 host masters before the save and
DCP takes the host path; that runs on CPU over gloo, at world 1, EP2 and EP2 x CP2.
Without it DCP builds CUDA device meshes, so those cases run on GPU.
"""

import json
import os
import random
import runpy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

STEPS, SAVED_AT = 3, 1
TOKENS = 8


def _prefix_config(experts, hidden_size=None):
    parity = runpy.run_path(str(Path(__file__).with_name('test_cp_ep_parity.py')))
    config = parity['_prefix_config'](experts)
    if hidden_size is None:
        return config
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    hf = config.to_hf_dict()
    hf['text_config']['hidden_size'] = hidden_size
    return DeepseekV41Config(hf)


def _reduced_config():
    """The -clean chain's reduced 40L fixture (#218 _assembly_config)."""
    import importlib.util

    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    root = Path(__file__).parents[2] / 'fixtures/deepseek_v41'
    release = json.loads((root / 'reference/config.json').read_text())
    modules = {}
    for name in ('config_mapping', 'fixtures'):
        path = Path(__file__).parents[3] / 'tools/deepseek_v41' / (name + '.py')
        spec = importlib.util.spec_from_file_location('c4_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    overrides = modules['fixtures'].reduced_overrides()
    for path, target in modules['config_mapping'].MAPPING.items():
        if target in overrides:
            parts = path.split('.')
            section = release
            for part in parts[:-1]:
                section = section[part]
            section[parts[-1]] = overrides[target]
    return DeepseekV41Config(release)


def _impl(case, parallel):
    from dataclasses import replace

    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.vision_config import OptimizerConfig

    impl = protocol.ImplConfig(
        device=case['device'],
        dtype=torch.float32 if case['device'] == 'cpu' else torch.bfloat16,
        token_map=list(range(256 if case.get('reduced') else 64)),
        trainable_engram=case['trainable'],
        # Sharded Engram lookups are CUDA-only (test_cp_ep_parity).
        shard_engram=case['device'] != 'cpu',
        parallel=parallel,
        optimizer='muon',
        optimizer_config=OptimizerConfig(
            lr=1e-3,
            ns_steps=5,
            coefficient_type='quintic',
            segmented_host=case['segmented'],
        ),
    )
    return replace(impl, quantized=case['quantized'])


def _seed():
    import numpy as np

    random.seed(915)
    np.random.seed(915)
    torch.manual_seed(915)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(915)


def _build(case):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    parallel = ParallelConfig(ep=case['ep'], cp=case['cp'])
    config = (
        _reduced_config()
        if case.get('reduced')
        else _prefix_config(case['experts'], case.get('hidden_size'))
    )
    bundle = protocol.build_model(config, impl_cfg=_impl(case, parallel))
    schedulers = [
        torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=0.8)
        for o in bundle.optimizer.optimizers
    ]
    return bundle, schedulers, parallel


def _digest(tensors):
    import hashlib

    hasher = hashlib.sha256()
    for tensor in tensors:
        if tensor is not None:
            hasher.update(
                tensor.detach().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            )
    return hasher.hexdigest()[:16]


def _grad(p):
    return p.main_grad if getattr(p, 'main_grad', None) is not None else p.grad


def _grads(model):
    return [_grad(p) for p in model.parameters()]


def _masters(optimizer):
    host = optimizer.host_update
    return [] if host is None else list(host.masters.values())


def _rng_digests():
    import hashlib
    import pickle

    import numpy as np

    streams = dict(
        python=pickle.dumps(random.getstate()),
        numpy=pickle.dumps(np.random.get_state()),
        torch=torch.get_rng_state().numpy().tobytes(),
    )
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        streams['cuda'] = torch.cuda.get_rng_state().numpy().tobytes()
    return {k: hashlib.sha256(v).hexdigest()[:16] for k, v in streams.items()}


def _step(bundle, schedulers):
    import numpy as np
    from megatron.lite.runtime.contracts import PackedBatch
    from megatron.lite.runtime.contracts.loss import LossContext, use_loss_context

    model, optimizer = bundle.chunks[0], bundle.optimizer
    model.train()
    # Every RNG stream entering the step: a divergence names the stream.
    rng = dict(start=_rng_digests())
    # Consume all three host RNG streams and torch's: a reset or a missing restore
    # changes the batch.
    offset = (
        random.randrange(30) + int(np.random.randint(30)) + int(torch.randint(30, (1,)))
    ) % 30
    device = next(model.parameters()).device
    ids = torch.randint(1, 32, (TOKENS,)) + offset
    ids = ids.to(device)
    batch = PackedBatch(ids, ids, torch.tensor([TOKENS], device=device))
    optimizer.zero_grad()
    context = LossContext(normalization_denominator=float(TOKENS))
    with use_loss_context(context):
        output = bundle.forward_step(model, batch)
    rng['forward'] = _rng_digests()
    loss = output['loss']
    assert loss.requires_grad and torch.isfinite(loss), 'RESUME_LOSS'
    loss.backward()
    rng['backward'] = _rng_digests()
    # Stage fingerprints: a run-to-run difference names the first stage that differs.
    embed = model.tensor_bindings['embed.weight'].tensor
    stages = dict(
        embed_grad=_digest(_grads(torch.nn.ModuleList([model.embed]))),
        # Per parameter, so a difference names the parameter whose backward differs.
        grads={name: _digest([_grad(p)]) for name, p in model.named_parameters()},
    )
    head = model.tensor_bindings['head.weight'].tensor
    grad = head.main_grad if getattr(head, 'main_grad', None) is not None else head.grad
    assert (
        grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    ), 'RESUME_GRADIENT: missing native gradient at head.weight'
    if bundle.finalize_grads is not None:
        bundle.finalize_grads()
    stages['finalized_grads'] = _digest(_grads(model))
    stages['masters_before'] = _digest(_masters(optimizer))
    accepted, norm = optimizer.step()[:2]
    rng['optimizer'] = _rng_digests()
    stages['masters_after'] = _digest(_masters(optimizer))
    stages['embed_after'] = _digest([embed])
    stages['parameters_after'] = _digest(list(model.parameters()))
    assert accepted, 'RESUME_UPDATE: the training step must publish an update'
    for scheduler in schedulers:
        scheduler.step()
    return dict(
        stages=stages,
        rng=rng,
        ids=ids.cpu().tolist(),
        # The global gradient norm the update was clipped with.
        norm=torch.as_tensor(norm, dtype=torch.float32).cpu().view(torch.int32).item(),
        loss=loss.detach().float().cpu().view(torch.int32).item(),
        lr=[g['lr'] for g in optimizer.param_groups],
    )


def _state(bundle, schedulers):
    from copy import deepcopy

    model = bundle.chunks[0]
    return deepcopy(
        dict(
            parameters={n: p.detach().cpu() for n, p in model.named_parameters()},
            buffers={n: b.detach().cpu() for n, b in model.named_buffers()},
            optimizer=bundle.optimizer.state_dict(),
            schedulers=[s.state_dict() for s in schedulers],
        )
    )


def _first_difference(actual, expected, label):
    """The first differing leaf, as #218's _g1_equal walks it; None if equal."""
    if isinstance(expected, torch.Tensor):
        if not (
            isinstance(actual, torch.Tensor)
            and actual.dtype == expected.dtype
            and actual.shape == expected.shape
        ):
            return label
        same = torch.equal(
            actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )
        return None if same else label
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return f'{label}/keys'
        for key in expected:
            found = _first_difference(actual[key], expected[key], f'{label}/{key}')
            if found:
                return found
        return None
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return f'{label}/len'
        for i, (a, e) in enumerate(zip(actual, expected)):
            found = _first_difference(a, e, f'{label}/{i}')
            if found:
                return found
        return None
    return None if actual == expected else label


def _offload(bundle):
    """The engine's param_offload: trainable parameters onto their host masters."""
    host = bundle.optimizer.host_update
    if host is not None:
        host.prepare_model_transfer('cpu')
        bundle.chunks[0].to('cpu')


def _reload(bundle, case):
    if case['device'] != 'cpu':
        bundle.chunks[0].to(case['device'])


def _checkpoint(bundle, parallel):
    return dict(config=parallel, ps=bundle.parallel_state, use_dcp=True)


def _worker(rank, directory, case, phase):
    from megatron.lite.primitive.ckpt import dcp as checkpoint

    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    world = case['ep'] * case['cp'] if case['ep'] * case['cp'] > 1 else 1
    gpu = case['device'] != 'cpu'
    if gpu:
        # As MegatronLiteEngine with full_determinism (the recipe): before CUDA init.
        from verl.workers.engine.utils import enable_full_determinism

        enable_full_determinism(seed=915)
        torch.cuda.set_device(rank)
        case = dict(case, device=f'cuda:{rank}')
    torch.set_num_threads(1)
    if not gpu and case['cp'] > 1:
        # Gloo-only shim for all_gather backward in CP subgroups (test_cp_ep_parity).
        parity = runpy.run_path(str(Path(__file__).with_name('test_cp_ep_parity.py')))
        parity['_patch_gloo_all_gather_backward']()
    root = Path(directory)
    dist.init_process_group(
        'nccl' if gpu else 'gloo',
        init_method=f'file://{root}/store-{phase}',
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=600),
    )
    record = dict(rank=rank, phase=phase)
    # The determinism control repeats the continuous run with its own checkpoint.
    tag = '' if phase != 'repeat' else '-repeat'
    path = str(root / f'checkpoint{tag}')
    try:
        _seed()
        bundle, schedulers, parallel = _build(case)
        if phase in ('continuous', 'repeat'):
            history = [_step(bundle, schedulers)]
            _offload(bundle)
            checkpoint.save_training_checkpoint(
                bundle.chunks[0],
                bundle.optimizer,
                SAVED_AT,
                path,
                **_checkpoint(bundle, parallel),
            )
            torch.save(
                [s.state_dict() for s in schedulers],
                root / f'schedulers{tag}-{rank}.pt',
            )
            saved = _state(bundle, schedulers)
            _reload(bundle, case)
            history += [_step(bundle, schedulers) for _ in range(STEPS - SAVED_AT)]
            torch.save(
                dict(saved=saved, expected=_state(bundle, schedulers), history=history),
                root / f'continuous{tag}-{rank}.pt',
            )
        else:
            reference = torch.load(root / f'continuous-{rank}.pt', weights_only=False)
            _offload(bundle)
            step = checkpoint.load_training_checkpoint(
                bundle.chunks[0],
                bundle.optimizer,
                path,
                **_checkpoint(bundle, parallel),
            )
            for scheduler, payload in zip(
                schedulers,
                torch.load(root / f'schedulers-{rank}.pt', weights_only=False),
            ):
                scheduler.load_state_dict(payload)
            record['step'] = int(step)
            record['restore'] = _first_difference(
                _state(bundle, schedulers), reference['saved'], 'RESTORE'
            )
            _reload(bundle, case)
            history = [_step(bundle, schedulers) for _ in range(SAVED_AT, STEPS)]
            record['schedule'] = _first_difference(
                history, reference['history'][SAVED_AT:], 'RNG_SCHEDULE_LOSS'
            )
            record['resume'] = _first_difference(
                _state(bundle, schedulers), reference['expected'], 'RESUME_BITWISE'
            )
            # The restore boundary itself must be checked: corrupt what load wrote.
            _offload(bundle)
            checkpoint.load_training_checkpoint(
                bundle.chunks[0],
                bundle.optimizer,
                path,
                **_checkpoint(bundle, parallel),
            )
            with torch.no_grad():
                bundle.chunks[0].embed.weight.view(-1)[0].add_(1)
            record['corrupt_caught'] = (
                _first_difference(
                    _state(bundle, schedulers)['parameters'],
                    reference['saved']['parameters'],
                    'RESTORE/parameters',
                )
                == 'RESTORE/parameters/embed.weight'
            )
            checkpoint.load_training_checkpoint(
                bundle.chunks[0],
                bundle.optimizer,
                path,
                **_checkpoint(bundle, parallel),
            )
            _reload(bundle, case)
            hook = bundle.chunks[0].head.weight.register_hook(torch.zeros_like)
            try:
                _step(bundle, schedulers)
                record['zero_gradient_caught'] = False
            except AssertionError as error:
                record['zero_gradient_caught'] = 'RESUME_GRADIENT' in str(error)
            finally:
                hook.remove()
    except Exception:
        import traceback

        record['error'] = traceback.format_exc()[-6000:]
    finally:
        (root / f'{phase}-{rank}.json').write_text(json.dumps(record))
        dist.destroy_process_group()


def _run(directory, case):
    world = case['ep'] * case['cp']
    # GPU cases first check that continuous training repeats bit for bit: a
    # nondeterministic run makes the resume comparison meaningless.
    phases = (
        ('continuous', 'repeat', 'resume')
        if case['device'] != 'cpu'
        else ('continuous', 'resume')
    )
    for phase in phases:
        # A fresh process group per phase: only the checkpoint on disk is shared.
        mp.start_processes(
            _worker,
            args=(str(directory), case, phase),
            nprocs=world,
            join=True,
            start_method='spawn',
        )
        records = [
            json.loads((Path(directory) / f'{phase}-{r}.json').read_text())
            for r in range(world)
        ]
        errors = [r['error'] for r in records if 'error' in r]
        assert not errors, errors[0]
    repeat = {}
    if 'repeat' in phases:
        for rank in range(world):
            first, second = (
                torch.load(Path(directory) / f'{name}-{rank}.pt', weights_only=False)
                for name in ('continuous', 'continuous-repeat')
            )
            repeat[rank] = _first_difference(
                [second['history'], second['expected']],
                [first['history'], first['expected']],
                'GPU_NONDETERMINISTIC',
            )
    print(
        'RESUME_BITWISE',
        json.dumps(dict(case=case, repeat=repeat, records=records)),
        flush=True,
    )
    assert not any(repeat.values()), repeat
    for record in records:
        assert record['step'] == SAVED_AT, record
        assert record['restore'] is None, record
        assert record['schedule'] is None, record
        assert record['resume'] is None, record
        assert record['corrupt_caught'] and record['zero_gradient_caught'], record


CPU_TOPOLOGIES = [(1, 1, 2), (2, 1, 2), (2, 2, 2)]


@pytest.mark.parametrize('trainable', [False, True])
@pytest.mark.parametrize('ep,cp,experts', CPU_TOPOLOGIES)
def test_dcp_resume_is_bitwise_host_optimizer(
    v41_core_te, tmp_path, ep, cp, experts, trainable
):
    case = dict(
        device='cpu',
        ep=ep,
        cp=cp,
        experts=experts,
        trainable=trainable,
        segmented=True,
        quantized=False,
    )
    _run(tmp_path, case)


@pytest.mark.gpus(2, min_architecture='blackwell')
@pytest.mark.parametrize('segmented', [False, True])
@pytest.mark.parametrize('trainable', [False, True])
@pytest.mark.parametrize('ep', [1, 2])
def test_dcp_resume_is_bitwise_on_gpu(tmp_path, ep, trainable, segmented):
    # VERL's fused cross entropy needs hidden_size % 128 == 0 (the reduced 40L
    # fixture's width); the CPU prefix is narrower.
    case = dict(
        device='cuda',
        ep=ep,
        cp=1,
        experts=2,
        trainable=trainable,
        segmented=segmented,
        quantized=False,
        hidden_size=128,
    )
    _run(tmp_path, case)


@pytest.mark.gpus(1, min_architecture='blackwell')
@pytest.mark.parametrize('segmented', [False, True])
@pytest.mark.parametrize('trainable', [False, True])
def test_dcp_resume_is_bitwise_quantized_on_gpu(tmp_path, trainable, segmented):
    """The quantized path on the reduced 40L fixture (#218's GPU configuration)."""
    case = dict(
        device='cuda',
        ep=1,
        cp=1,
        experts=8,
        trainable=trainable,
        segmented=segmented,
        quantized=True,
        reduced=True,
    )
    _run(tmp_path, case)
