# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU lifecycle controls; GPU residency and numerical parity require RL gates."""
import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples/verl'))
from verl_mlite import compat

pytestmark = pytest.mark.optional


@pytest.fixture
def server(monkeypatch):
    module = ModuleType('verl.workers.rollout.vllm_rollout.vllm_async_server')

    class Server:
        async def _sleep_hybrid(self):
            await self.engine.sleep(level=2)
            await self.engine.reset_encoder_cache()

        def _get_wake_up_tags(self):
            return ['kv_cache', 'weights']

        async def wake_up(self, tags=None):
            self.events.append(('wake', tags or self._get_wake_up_tags()))

    module.vLLMHttpServer = Server
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(compat, '_vllm_importable', lambda: True)
    monkeypatch.delenv('VERL_MLITE_ROLLOUT_SLEEP_LEVEL', raising=False)
    obj = Server()
    obj.lora_as_adapter = True
    obj.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type='deepseek_v41'))
    obj.events = []

    async def sleep(*, level):
        obj.events.append(('sleep', level))

    async def release():
        obj.events.append(('release',))

    async def reset():
        obj.events.append(('reset',))

    obj.engine = SimpleNamespace(
        sleep=sleep, release_kv_cache_memory=release, reset_encoder_cache=reset
    )
    return obj


def test_kv_only_is_ds41_adapter_default_and_idempotent(server, monkeypatch):
    monkeypatch.setenv('VERL_MLITE_ROLLOUT_SLEEP_LEVEL', '1')
    assert compat._patch_vllm_rollout_sleep_level()
    patched = type(server)._sleep_hybrid
    assert compat._patch_vllm_rollout_sleep_level()
    assert patched is type(server)._sleep_hybrid
    asyncio.run(server._sleep_hybrid())
    assert server.events == [('sleep', 0), ('release',), ('reset',)]
    assert server._get_wake_up_tags() == ['kv_cache']


def test_missing_api_fails_before_pause(server):
    del server.engine.release_kv_cache_memory
    assert compat._patch_vllm_rollout_sleep_level()
    with pytest.raises(RuntimeError, match='KV-only release API'):
        asyncio.run(server._sleep_hybrid())
    assert server.events == []


@pytest.mark.parametrize('failure', ['sleep', 'release_kv_cache_memory'])
def test_original_exception_is_not_hidden(server, failure):
    error = RuntimeError('original-engine-error')
    setattr(server.engine, failure, AsyncMock(side_effect=error))
    compat._patch_vllm_rollout_sleep_level()
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(server._sleep_hybrid())
    assert exc.value is error
    assert ('reset',) not in server.events


@pytest.mark.parametrize('adapter,model', [(False, 'deepseek_v41'), (True, 'other')])
@pytest.mark.parametrize('override', [None, '1', '2'])
def test_foreign_or_full_parameter_rollouts_keep_original_policy(
    server, monkeypatch, adapter, model, override
):
    server.lora_as_adapter = adapter
    server.model_config.hf_config.model_type = model
    if override:
        monkeypatch.setenv('VERL_MLITE_ROLLOUT_SLEEP_LEVEL', override)
    compat._patch_vllm_rollout_sleep_level()
    asyncio.run(server._sleep_hybrid())
    assert server.events == [('sleep', int(override or '2')), ('reset',)]
    assert server._get_wake_up_tags() == ['kv_cache', 'weights']


@pytest.mark.parametrize('legacy', [None, '1', '2'])
def test_explicit_weight_offload_keeps_base_and_restores_both_tags(server, monkeypatch, legacy):
    server.config = SimpleNamespace(custom={'ds41_lora_sleep_mode': 'weights_cpu'})
    if legacy:
        monkeypatch.setenv('VERL_MLITE_ROLLOUT_SLEEP_LEVEL', legacy)
    assert compat._patch_vllm_rollout_sleep_level()
    asyncio.run(server._sleep_hybrid())
    assert server.events == [('sleep', 1), ('reset',)]
    assert server._get_wake_up_tags() == ['weights', 'kv_cache']


def test_unknown_adapter_mode_fails_before_engine_calls(server):
    server.config = SimpleNamespace(custom={'ds41_lora_sleep_mode': 'discard'})
    compat._patch_vllm_rollout_sleep_level()
    with pytest.raises(ValueError, match='sleep mode'):
        asyncio.run(server._sleep_hybrid())
    with pytest.raises(ValueError, match='sleep mode'):
        server._get_wake_up_tags()
    assert server.events == []


def test_weight_offload_preserves_original_engine_exception(server):
    server.config = SimpleNamespace(custom={'ds41_lora_sleep_mode': 'weights_cpu'})
    error = RuntimeError('native-sleep-error')
    server.engine.sleep = AsyncMock(side_effect=error)
    compat._patch_vllm_rollout_sleep_level()
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(server._sleep_hybrid())
    assert exc.value is error and ('reset',) not in server.events


@pytest.mark.parametrize('adapter,model', [(False, 'deepseek_v41'), (True, 'other')])
def test_adapter_offload_option_does_not_change_foreign_rollouts(server, adapter, model):
    server.lora_as_adapter = adapter
    server.model_config.hf_config.model_type = model
    server.config = SimpleNamespace(custom={'ds41_lora_sleep_mode': 'weights_cpu'})
    compat._patch_vllm_rollout_sleep_level()
    asyncio.run(server._sleep_hybrid())
    assert server.events == [('sleep', 2), ('reset',)]
    assert server._get_wake_up_tags() == ['kv_cache', 'weights']


def reload_server(server):
    server.config = SimpleNamespace(
        custom={'ds41_lora_sleep_mode': 'weights_reload'},
        data_parallel_size=4,
        tensor_model_parallel_size=1,
    )

    async def utility(*args):
        server.events.append(('reload', args))
        return [[None] for _ in range(4)]

    server.engine.engine_core = SimpleNamespace(call_utility_all_async=utility)
    return server


def test_l2_reload_all_engines_once_before_adapter_republish(server):
    reload_server(server)
    compat._patch_vllm_rollout_sleep_level()

    async def cycle():
        await server._sleep_hybrid()
        await server.wake_up(['weights'])
        server.events.append(('publish_current_adapter_all_ack',))
        await server.wake_up(['kv_cache'])

    asyncio.run(cycle())
    assert server.events == [
        ('sleep', 2),
        ('reset',),
        ('wake', ['weights']),
        ('reload', ('collective_rpc', 'reload_weights', None, (), {})),
        ('publish_current_adapter_all_ack',),
        ('wake', ['kv_cache']),
    ]
    assert server._verl_mlite_reload_receipt['engines'] == 4
    assert not server._verl_mlite_reload_pending


@pytest.mark.parametrize('replies', [[], [[None]] * 3, [None] * 4, [[None, None]] * 4])
def test_l2_missing_engine_fails_closed_and_remains_pending(server, replies):
    reload_server(server)
    server.engine.engine_core.call_utility_all_async = AsyncMock(return_value=replies)
    compat._patch_vllm_rollout_sleep_level()
    asyncio.run(server._sleep_hybrid())
    with pytest.raises(RuntimeError, match='engine census'):
        asyncio.run(server.wake_up())
    assert server._verl_mlite_reload_pending


def test_l2_native_loader_failure_is_not_hidden(server):
    reload_server(server)
    error = RuntimeError('native-loader-error')
    server.engine.engine_core.call_utility_all_async = AsyncMock(side_effect=error)
    compat._patch_vllm_rollout_sleep_level()
    asyncio.run(server._sleep_hybrid())
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(server.wake_up())
    assert caught.value is error
    assert server._verl_mlite_reload_pending


def test_l2_requires_all_dp_api_before_discard(server):
    reload_server(server)
    del server.engine.engine_core.call_utility_all_async
    compat._patch_vllm_rollout_sleep_level()
    with pytest.raises(RuntimeError, match='all-DP'):
        asyncio.run(server._sleep_hybrid())
    assert server.events == []


def test_l2_kv_only_wake_first_also_restores_weights(server):
    reload_server(server)
    compat._patch_vllm_rollout_sleep_level()
    asyncio.run(server._sleep_hybrid())
    asyncio.run(server.wake_up(['kv_cache']))
    assert server.events[2] == ('wake', ['weights', 'kv_cache'])
    assert server._get_wake_up_tags() == ['weights', 'kv_cache']
