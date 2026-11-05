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


def test_residency_check_handles_core_ddp_buffers_list(monkeypatch):
    monkeypatch.setenv('MLITE_EXCLUSIVE_GUARD', 'raise')
    m = product()
    owner = torch.nn.Module()
    owner.buffers = []  # Core DDP shadows nn.Module.buffers with buffer groups.
    owner.register_buffer('kept_on_host', torch.ones(2))
    nvml = SimpleNamespace(nvmlInit=lambda: None, nvmlDeviceGetCount=lambda: 0)
    with (
        patch.dict(sys.modules, {'pynvml': nvml}),
        patch.object(torch.cuda, 'is_available', return_value=True),
        patch.object(torch.cuda, 'synchronize'),
        patch.object(torch.cuda, 'memory_allocated', return_value=0),
        patch.object(torch.cuda, 'memory_reserved', return_value=0),
        patch.object(torch.cuda, 'max_memory_allocated', return_value=0),
        patch.object(torch.cuda, 'max_memory_reserved', return_value=0),
        patch.object(torch.cuda, 'mem_get_info', return_value=(170 * 2**30, 186 * 2**30)),
    ):
        m.record(SimpleNamespace(module=owner), 'before_rollout_wake')
        with patch.object(torch.cuda, 'memory_allocated', return_value=5 * 2**30):
            with pytest.raises(AssertionError, match='ACTOR_GPU_RESIDENCY_ABOVE_4GIB'):
                m.record(SimpleNamespace(module=owner), 'before_rollout_wake')


@pytest.mark.parametrize(
    ('process_gib', 'free_gib', 'total_gib', 'error'),
    [
        (12, 159, 186, None),
        (17, 159, 186, 'ABOVE_16GIB'),
        (12, 143, 186, 'WAKE_FREE_MARGIN'),
        (12, 159, 150, 'CAPACITY_BUDGET'),
    ],
)
@pytest.mark.parametrize('mode', ['raise', 'warn', None])
def test_native_process_residency_and_rollout_headroom(
    process_gib, free_gib, total_gib, error, mode, monkeypatch, capsys
):
    if mode is None:
        monkeypatch.delenv('MLITE_EXCLUSIVE_GUARD', raising=False)
    else:
        monkeypatch.setenv('MLITE_EXCLUSIVE_GUARD', mode)
    m = product()
    owner = torch.nn.Module()
    nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetComputeRunningProcesses=lambda h: [
            SimpleNamespace(pid=m.os.getpid(), usedGpuMemory=process_gib * 2**30)
        ],
    )
    with (
        patch.dict(sys.modules, {'pynvml': nvml}),
        patch.object(torch.cuda, 'is_available', return_value=True),
        patch.object(torch.cuda, 'synchronize'),
        patch.object(torch.cuda, 'memory_allocated', return_value=2**30),
        patch.object(torch.cuda, 'memory_reserved', return_value=2 * 2**30),
        patch.object(torch.cuda, 'max_memory_allocated', return_value=130 * 2**30),
        patch.object(torch.cuda, 'max_memory_reserved', return_value=140 * 2**30),
        patch.object(
            torch.cuda, 'mem_get_info', return_value=(free_gib * 2**30, total_gib * 2**30)
        ),
    ):
        if error and mode == 'raise':
            with pytest.raises(AssertionError, match=error):
                m.record(SimpleNamespace(module=owner), 'before_rollout_wake')
        else:
            m.record(SimpleNamespace(module=owner), 'before_rollout_wake')
    warned = [line for line in capsys.readouterr().out.splitlines() if 'GUARD_WARN' in line]
    assert len(warned) == (1 if error and mode != 'raise' else 0)
    if warned:
        assert error in warned[0]


class _DeviceTensor(torch.Tensor):
    """CPU stand-in for a CUDA tensor; each ``to`` is a real copy, like D2H/H2D."""

    is_cuda = True
    copies = 0

    def to(self, device, *args, **kwargs):
        _DeviceTensor.copies += 1
        moved = self.as_subclass(torch.Tensor).clone()
        return moved.as_subclass(_HostTensor if device == 'cpu' else _DeviceTensor)


class _HostTensor(_DeviceTensor):
    is_cuda = False


def test_round_trip_keeps_attribute_shared_by_modules():
    # DS4.1 deployment attention caches vLLM's process-wide RoPE table on every layer
    # (``_deployment_rope``): one tensor referenced by many modules.
    m = product()
    _DeviceTensor.copies = 0
    shared = torch.arange(8.0).as_subclass(_DeviceTensor)
    root = torch.nn.Module()
    layers = [torch.nn.Module() for _ in range(3)]
    for index, layer in enumerate(layers):
        root.add_module(str(index), layer)
        layer._deployment_rope = shared
    engine = SimpleNamespace(module=root)

    m.extra_storage(engine, 'cpu')
    assert len({id(layer._deployment_rope) for layer in layers}) == 1
    m.extra_storage(engine, 'cuda')
    assert len({id(layer._deployment_rope) for layer in layers}) == 1
    assert torch.equal(layers[0]._deployment_rope.as_subclass(torch.Tensor), torch.arange(8.0))
    assert _DeviceTensor.copies == 2


def test_round_trip_is_bitwise_and_keeps_buffer_and_attribute_sharing():
    m = product()
    gen = torch.Generator().manual_seed(7)

    def device(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen).to(dtype).as_subclass(_DeviceTensor)

    rope = device(16, 4, dtype=torch.float32)  # one table for every layer
    root = torch.nn.Module()
    layers = []
    for index in range(4):
        layer = torch.nn.Linear(4, 4)
        layer.register_buffer('bias_vl', device(4))
        layer.register_buffer('absent', None)
        layer._deployment_rope = rope
        layer.scale = 0.5  # non-tensor attributes are left alone
        root.add_module(str(index), layer)
        layers.append(layer)
    root.register_buffer('mask', device(8, dtype=torch.int64))
    engine = SimpleNamespace(module=root)
    plain = lambda value: value.as_subclass(torch.Tensor).clone()
    params = {n: (p, plain(p)) for n, p in root.named_parameters()}
    buffers = {n: plain(b) for n, b in root.named_buffers()}
    rope_bits = plain(rope)

    m.extra_storage(engine, 'cpu')
    assert all(not b.is_cuda for b in root.buffers())
    assert len({id(layer._deployment_rope) for layer in layers}) == 1
    assert not layers[0]._deployment_rope.is_cuda
    m.extra_storage(engine, 'cuda')

    for name, (owner, bits) in params.items():  # Core DDP owns parameter moves
        assert dict(root.named_parameters())[name] is owner
        assert torch.equal(owner, bits)
    for name, value in root.named_buffers():
        assert value.is_cuda and torch.equal(value.as_subclass(torch.Tensor), buffers[name])
    assert all(layer.absent is None and layer.scale == 0.5 for layer in layers)
    restored = {id(layer._deployment_rope) for layer in layers}
    pointers = {layer._deployment_rope.data_ptr() for layer in layers}
    assert len(restored) == 1 and len(pointers) == 1
    assert layers[0]._deployment_rope.is_cuda
    assert torch.equal(layers[0]._deployment_rope.as_subclass(torch.Tensor), rope_bits)
    assert engine._exclusive_extra_storage == []
