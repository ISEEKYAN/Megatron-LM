# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exclusive adapter publication must park actor before any rollout wake."""
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def product():
    path = Path(__file__).parents[3] / 'examples/verl/verl_mlite/rollout/exclusive.py'
    spec = importlib.util.spec_from_file_location('exclusive_actor_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(exclusive=True, adapter=True, host=True):
    trace = []
    values = [('layers.20.lora_a.weight', torch.tensor([-0.0, 1.0], dtype=torch.bfloat16))]
    peft = {'manifest': 'unchanged'} if adapter else None

    class Engine:
        engine_config = SimpleNamespace(exclusive_rollout=exclusive)
        is_param_offload_enabled = False

        def get_per_tensor_param(self, **kwargs):
            cached = getattr(self, '_exclusive_adapter_snapshot', None)
            if cached is not None:
                self._exclusive_adapter_snapshot = None
                trace.append('consume_host')
                return iter(cached[0]), cached[1]
            trace.append('export')
            if not host:
                return iter([('bad', torch.empty(1, device='meta'))]), peft
            return iter(values), peft

        def to(self, device, **kwargs):
            assert device == 'cpu' and kwargs == dict(model=True, optimizer=True, grad=True)
            trace.append('offload')

    class Worker:
        actor = SimpleNamespace(engine=Engine())
        config = SimpleNamespace(
            rollout=SimpleNamespace(checkpoint_engine=SimpleNamespace(backend='naive'))
        )
        base_sync_done = True
        layered_summon = False

        async def update_weights(self, global_steps=None, mode='auto'):
            trace.append('wake')
            ws, cfg = self.actor.engine.get_per_tensor_param(base_sync_done=True)
            if exclusive:
                assert trace.index('offload') < trace.index('wake')
            assert cfg is peft
            received = list(ws)
            assert torch.equal(received[0][1].view(torch.uint8), values[0][1].view(torch.uint8))
            trace.extend(['native_publish', 'all_ACK'])
            return global_steps

    Worker.update_weights._dispatch_marker = 'preserved'
    return Worker, trace


def install(module, cls):
    workers = ModuleType('verl.workers')
    workers.engine_workers = SimpleNamespace(ActorRolloutRefWorker=cls)
    with patch.dict(sys.modules, {'verl.workers': workers}):
        module.install()


@pytest.mark.parametrize('enabled', [False, True])
def test_native_order_and_host_bytes(enabled):
    m = product()
    cls, trace = fixture(exclusive=enabled)
    install(m, cls)
    assert cls.update_weights._dispatch_marker == 'preserved'
    with patch.object(torch.cuda, 'is_available', return_value=False):
        assert asyncio.run(cls().update_weights(global_steps=3)) == 3
    assert trace == (
        ['export', 'offload', 'wake', 'consume_host', 'native_publish', 'all_ACK']
        if enabled
        else ['wake', 'export', 'native_publish', 'all_ACK']
    )
    assert getattr(cls.actor.engine, '_exclusive_adapter_snapshot', None) is None


@pytest.mark.parametrize(
    'case', ['base', 'backend', 'offload', 'no_adapter', 'non_host', 'pending']
)
def test_invalid_state_stops_before_wake(case):
    m = product()
    cls, trace = fixture(adapter=case != 'no_adapter', host=case != 'non_host')
    if case == 'base':
        cls.base_sync_done = False
    if case == 'backend':
        cls.config.rollout.checkpoint_engine.backend = 'other'
    if case == 'offload':
        cls.actor.engine.is_param_offload_enabled = True
    if case == 'pending':
        cls.actor.engine._exclusive_adapter_snapshot = ('pending', {})
    install(m, cls)
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(cls().update_weights())
    assert 'wake' not in trace


def test_repeat_cpu_transfer_does_not_drop_restore_paths():
    m = product()
    owner = torch.nn.Linear(2, 2)
    engine = SimpleNamespace(module=owner, _exclusive_extra_storage=[(owner, 'weight', False)])
    m.extra_storage(engine, 'cpu')
    assert engine._exclusive_extra_storage == [(owner, 'weight', False)]


def test_host_snapshot_consumed_once_without_onload():
    m = product()
    values = [('layers.20.lora_b.weight', torch.tensor([-0.0, 2.0], dtype=torch.bfloat16))]
    engine = SimpleNamespace(_exclusive_initial_adapter=(values, {'policy_version': 0}))
    got, config = m.consume_host_snapshot(engine)
    assert config == {'policy_version': 0}
    assert torch.equal(list(got)[0][1].view(torch.uint8), values[0][1].view(torch.uint8))
    assert m.consume_host_snapshot(engine) is None


def test_ddp_onload_preserves_original_host_parameter():
    path = Path(__file__).parents[3] / 'megatron/lite/runtime/megatron_utils.py'
    spec = importlib.util.spec_from_file_location('native_runtime_moves_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = torch.nn.Module()
    model.host = torch.nn.Parameter(torch.ones(3), requires_grad=False)
    model.base = torch.nn.Parameter(torch.ones(5), requires_grad=False)
    model.host._mlite_keep_on_cpu = True
    owner = SimpleNamespace(module=model, buffers=[], expert_parallel_buffers=[])
    calls = []
    original = torch.Tensor.to

    def storage_adapter(value, device, **kwargs):
        assert device == 'cuda'
        calls.append(value.data_ptr())
        return original(value, device='cpu', **kwargs)

    with (
        patch.object(module, '_is_megatron_ddp', return_value=True),
        patch.object(torch.Tensor, 'to', storage_adapter),
    ):
        module.load_model_to_gpu([owner])
    assert calls == [model.base.data_ptr()]
    assert model.host.device.type == 'cpu' and torch.equal(model.host, torch.ones(3))


def test_all_actor_ranks_park_before_shared_server_wake():
    m = product()
    cls, trace = fixture()
    install(m, cls)
    with (
        patch.object(torch.cuda, 'is_available', return_value=False),
        patch.object(torch.distributed, 'is_initialized', return_value=True),
        patch.object(
            torch.distributed, 'barrier', side_effect=lambda: trace.append('all_actors_parked')
        ),
    ):
        asyncio.run(cls().update_weights())
    assert trace.index('offload') < trace.index('all_actors_parked') < trace.index('wake')
