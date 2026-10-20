# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))

from verl_mlite import compat

pytestmark = pytest.mark.optional


def test_ds4_uses_native_vllm_layerwise_reload(monkeypatch) -> None:
    events = []
    fp8_utils = ModuleType("verl.utils.vllm.vllm_fp8_utils")
    fp8_utils.prepare_quanted_weights_for_loading = (
        lambda runner: events.append("legacy-prepare") or False
    )
    fp8_utils.process_quanted_weights_after_loading = (
        lambda runner, state: events.append(("legacy-process", state))
    )
    captured_weights = []
    fp8_utils.load_quanted_weights = (
        lambda weights, runner: captured_weights.extend(weights) or "loaded"
    )
    dsv4_utils = ModuleType("verl.utils.vllm.vllm_dsv4_fp8_utils")
    dsv4_utils.is_deepseek_v4_model = lambda model: model.is_ds4
    vllm_config = ModuleType("vllm.config")
    reload_api = ModuleType("vllm.model_executor.model_loader.reload")
    reload_meta = ModuleType("vllm.model_executor.model_loader.reload.meta")
    rollout_utils = ModuleType("verl.workers.rollout.vllm_rollout.utils")
    reload_meta.SKIP_TENSORS = {"existing"}

    @contextmanager
    def set_current_vllm_config(config):
        events.append(("enter-config", config))
        try:
            yield
        finally:
            events.append(("exit-config", config))

    vllm_config.set_current_vllm_config = set_current_vllm_config
    reload_api.initialize_layerwise_reload = (
        lambda model: events.append(("initialize", model))
    )
    reload_api.finalize_layerwise_processing = (
        lambda model, config: events.append(("finalize", model, config))
    )
    rollout_utils.load_quanted_weights = fp8_utils.load_quanted_weights
    for module in (
        fp8_utils,
        dsv4_utils,
        vllm_config,
        reload_api,
        reload_meta,
        rollout_utils,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(compat, "_vllm_importable", lambda: True)
    monkeypatch.setattr(
        compat,
        "_restore_dsv4_attn_sink_padding",
        lambda model: events.append(("restore-attention", model)) or 4,
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: events.append("empty-device-cache")
    )
    model = SimpleNamespace(is_ds4=True)
    config = SimpleNamespace(model_config=object())
    runner = SimpleNamespace(model=model, vllm_config=config)
    assert compat._patch_verl_dsv4_native_layerwise_reload()
    state = fp8_utils.prepare_quanted_weights_for_loading(runner)
    assert state is compat._DSV4_LAYERWISE_RELOAD_STATE
    source = torch.arange(4)
    assert rollout_utils.load_quanted_weights([("weight", source)], runner) == "loaded"
    assert captured_weights[0][1].data_ptr() != source.data_ptr()
    assert model._verl_mlite_ds4_layerwise_reload_active is True
    fp8_utils.process_quanted_weights_after_loading(runner, state)
    assert model._verl_mlite_ds4_layerwise_reload_active is False
    assert "legacy-prepare" not in events
    assert not any(isinstance(event, tuple) and event[0] == "legacy-process" for event in events)
    assert events == [
        ("enter-config", config),
        ("initialize", model),
        ("exit-config", config),
        ("enter-config", config),
        ("finalize", model, config.model_config),
        ("exit-config", config),
        ("restore-attention", model),
        "empty-device-cache",
    ]
    assert reload_meta.SKIP_TENSORS == {
        "existing",
        "tid2eid",
        "expert_bias",
        "e_score_correction_bias",
        "attn_sink",
    }


def test_non_ds4_keeps_verl_reload_path(monkeypatch) -> None:
    events = []
    fp8_utils = ModuleType("verl.utils.vllm.vllm_fp8_utils")
    fp8_utils.prepare_quanted_weights_for_loading = lambda runner: "legacy-state"
    fp8_utils.process_quanted_weights_after_loading = (
        lambda runner, state: events.append(state)
    )
    fp8_utils.load_quanted_weights = lambda weights, runner: list(weights)
    dsv4_utils = ModuleType("verl.utils.vllm.vllm_dsv4_fp8_utils")
    dsv4_utils.is_deepseek_v4_model = lambda model: False
    rollout_utils = ModuleType("verl.workers.rollout.vllm_rollout.utils")
    rollout_utils.load_quanted_weights = fp8_utils.load_quanted_weights
    monkeypatch.setitem(sys.modules, fp8_utils.__name__, fp8_utils)
    monkeypatch.setitem(sys.modules, dsv4_utils.__name__, dsv4_utils)
    monkeypatch.setitem(sys.modules, rollout_utils.__name__, rollout_utils)
    monkeypatch.setattr(compat, "_vllm_importable", lambda: True)

    runner = SimpleNamespace(model=object())
    assert compat._patch_verl_dsv4_native_layerwise_reload()
    state = fp8_utils.prepare_quanted_weights_for_loading(runner)
    assert state == "legacy-state"
    source = torch.arange(2)
    loaded = rollout_utils.load_quanted_weights([("weight", source)], runner)
    assert loaded[0][1] is source
    fp8_utils.process_quanted_weights_after_loading(runner, state)
    assert events == ["legacy-state"]


def test_level2_retention_keeps_weights_pool_bytes_not_rewritten_by_sync(
    monkeypatch,
) -> None:
    """Drafter weights, bias_vl and plain tensors share the pool with synced params."""

    class _Storage:
        def __init__(self, ptr, nbytes):
            self._ptr, self._nbytes = ptr, nbytes

        def data_ptr(self):
            return self._ptr

        def nbytes(self):
            return self._nbytes

    class _Param:
        is_cuda = True

        def __init__(self, ptr, nbytes):
            self._st = _Storage(ptr, nbytes)

        def untyped_storage(self):
            return self._st

    def chunk(ptr, size, tag="weights"):
        return SimpleNamespace(handle=(0, size, ptr), tag=tag)

    allocator = SimpleNamespace(
        pointer_to_data={
            0: chunk(1000, 100),
            1: chunk(2000, 50),
            2: chunk(5000, 10, tag="kv_cache"),
        }
    )
    cumem = ModuleType("vllm.device_allocator.cumem")
    cumem.CuMemAllocator = SimpleNamespace(get_instance=lambda: allocator)
    monkeypatch.setitem(sys.modules, "vllm.device_allocator.cumem", cumem)
    params = [
        ("layers.0.w", _Param(1010, 30)),
        ("layers.0.w_alias", _Param(1010, 30)),
        ("layers.1.w", _Param(1060, 20)),
        ("layers.0.gate.bias_vl", _Param(1080, 10)),
        ("layers.2.w", _Param(2000, 50)),
    ]
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=SimpleNamespace(named_parameters=lambda: iter(params))
        )
    )

    assert compat._weights_pool_gaps(worker) == [
        (1000, 10),
        (1040, 20),
        (1080, 20),
    ]


def test_val_max_tokens_caps_only_validation(monkeypatch) -> None:
    import asyncio

    seen = []

    class AgentLoopWorker:
        async def _run_agent_loop(self, sampling_params, trajectory, *, agent_name, trace=True):
            seen.append(dict(sampling_params))

    module = ModuleType("verl.experimental.agent_loop.agent_loop")
    module.AgentLoopWorker = AgentLoopWorker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("VERL_MLITE_VAL_MAX_TOKENS", "4096")

    assert compat._patch_verl_val_max_tokens()
    worker = AgentLoopWorker()
    params = {"temperature": 1.0}
    asyncio.run(worker._run_agent_loop(params, {"validate": True}, agent_name="a"))
    asyncio.run(worker._run_agent_loop(params, {"validate": False}, agent_name="a"))

    assert seen == [{"temperature": 1.0, "max_tokens": 4096}, {"temperature": 1.0}]
    assert params == {"temperature": 1.0}
