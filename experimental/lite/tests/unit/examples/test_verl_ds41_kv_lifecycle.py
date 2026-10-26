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


def test_weights_cpu_is_ds41_adapter_default_and_idempotent(server, monkeypatch):
    monkeypatch.setenv('VERL_MLITE_ROLLOUT_SLEEP_LEVEL', '1')
    assert compat._patch_vllm_rollout_sleep_level()
    patched = type(server)._sleep_hybrid
    assert compat._patch_vllm_rollout_sleep_level()
    assert patched is type(server)._sleep_hybrid
    asyncio.run(server._sleep_hybrid())
    assert server.events == [('sleep', 1), ('reset',)]
    assert server._get_wake_up_tags() == ['weights', 'kv_cache']


def test_missing_api_fails_before_pause(server):
    server.config = SimpleNamespace(custom={"ds41_lora_sleep_mode": "kv_only"})
    del server.engine.release_kv_cache_memory
    assert compat._patch_vllm_rollout_sleep_level()
    with pytest.raises(RuntimeError, match='KV-only release API'):
        asyncio.run(server._sleep_hybrid())
    assert server.events == []


@pytest.mark.parametrize('failure', ['sleep', 'release_kv_cache_memory'])
def test_original_exception_is_not_hidden(server, failure):
    server.config = SimpleNamespace(custom={'ds41_lora_sleep_mode': 'kv_only'})
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


@pytest.mark.parametrize('legacy_level', ['1', '2'])
def test_explicit_kv_only_remains_available(server, monkeypatch, legacy_level):
    server.config = SimpleNamespace(custom={'ds41_lora_sleep_mode': 'kv_only'})
    monkeypatch.setenv('VERL_MLITE_ROLLOUT_SLEEP_LEVEL', legacy_level)
    compat._patch_vllm_rollout_sleep_level()
    asyncio.run(server._sleep_hybrid())
    assert server.events == [('sleep', 0), ('release',), ('reset',)]
    assert server._get_wake_up_tags() == ['kv_cache']


def test_cpu_backup_does_not_require_kv_only_api(server):
    del server.engine.release_kv_cache_memory
    server.config = SimpleNamespace(custom={'ds41_lora_sleep_mode': 'weights_cpu'})
    compat._patch_vllm_rollout_sleep_level()
    asyncio.run(server._sleep_hybrid())
    assert server.events == [('sleep', 1), ('reset',)]


def test_bad_mode_fails_before_sleep(server):
    server.config = SimpleNamespace(custom={'ds41_lora_sleep_mode': 'typo'})
    compat._patch_vllm_rollout_sleep_level()
    with pytest.raises(ValueError, match='sleep mode'):
        asyncio.run(server._sleep_hybrid())
    assert server.events == []
