# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from verl_mlite.engine.config import MegatronLiteEngineConfig
from verl_mlite.engine.mlite_engine import MegatronLiteEngine, _build_lr_scheduler
from megatron.lite.runtime.contracts import LossContext


def _optimizer_config(**override_optimizer_config) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adam",
        lr=1e-6,
        min_lr=None,
        min_lr_ratio=None,
        clip_grad=1.0,
        weight_decay=0.1,
        lr_warmup_steps_ratio=0.0,
        total_training_steps=10,
        lr_warmup_steps=0,
        lr_warmup_init=0.0,
        lr_decay_steps=None,
        lr_decay_style="constant",
        weight_decay_incr_style="constant",
        lr_wsd_decay_style="exponential",
        lr_wsd_decay_steps=None,
        use_checkpoint_opt_param_scheduler=False,
        betas=(0.9, 0.95),
        override_optimizer_config=override_optimizer_config,
    )


def _engine(
    *, engine_config: MegatronLiteEngineConfig, optimizer_config: SimpleNamespace | None = None
) -> MegatronLiteEngine:
    return MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path="/tmp/qwen35", hf_config={"model_type": "qwen3_5_moe"}, mtp=None
        ),
        engine_config=engine_config,
        optimizer_config=optimizer_config or _optimizer_config(),
        checkpoint_config={},
    )


def _engine_config(**kwargs) -> MegatronLiteEngineConfig:
    values = {"custom_backend_module": None, "impl_cfg": {"use_thd": True}}
    values.update(kwargs)
    return MegatronLiteEngineConfig(**values)


def _fake_handle(*, model_chunks=None, optimizer=None) -> SimpleNamespace:
    """Minimal ModelHandle stand-in for the resync memory protocol."""
    return SimpleNamespace(
        _extras={"model_chunks": model_chunks if model_chunks is not None else []},
        _optimizer=optimizer,
    )


@pytest.mark.parametrize("num_microbatches", [1, 4])
def test_verl_loss_hook_preserves_gradient_and_micro_outputs(num_microbatches):
    engine = _engine(engine_config=_engine_config())
    weight = torch.nn.Parameter(torch.tensor(1.0))
    outputs = []
    engine._build_verl_model_output = lambda **_kwargs: {"log_probs": weight * 3}
    engine.get_data_parallel_group = lambda: None

    hook = engine._make_runtime_loss_fn(
        lambda model_output, **_kwargs: (model_output["log_probs"] / num_microbatches, {}),
        num_microbatches=num_microbatches,
        output_lst=outputs,
    )
    for _ in range(num_microbatches):
        loss, _ = hook({}, object(), LossContext(source_batch=object()))
        (loss / num_microbatches).backward()

    torch.testing.assert_close(weight.grad, torch.tensor(3.0))
    assert [output["loss"] for output in outputs] == [3.0 / num_microbatches] * num_microbatches


def test_optimizer_offload_enables_full_optimizer_state_offload_by_default() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=True),
        optimizer_config=_optimizer_config(
            use_precision_aware_optimizer=True, decoupled_weight_decay=True
        ),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 1.0
    assert optimizer.use_precision_aware_optimizer is True
    assert optimizer.decoupled_weight_decay is True
    assert optimizer.adam_beta1 == 0.9
    assert optimizer.adam_beta2 == 0.95


def test_explicit_optimizer_offload_fraction_overrides_engine_default() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=True),
        optimizer_config=_optimizer_config(offload_fraction=0.25),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 0.25


def test_optimizer_cpu_offload_alias_maps_to_full_offload_fraction() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=False),
        optimizer_config=_optimizer_config(optimizer_cpu_offload=True),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 1.0


def test_mlite_config_threads_rl_parallel_and_impl_settings() -> None:
    engine = _engine(
        engine_config=_engine_config(
            tp=2,
            ep=8,
            etp=1,
            pp=1,
            cp=1,
            optimizer_offload=True,
            attention_backend_override="flash",
            impl_cfg={"use_thd": True, "deterministic": False},
        )
    )

    config = engine._build_mlite_config()

    assert config.model_name == "qwen3_5"
    assert config.impl == "lite"
    assert config.parallel.tp == 2
    assert config.parallel.ep == 8
    assert config.parallel.etp == 1
    assert config.optimizer.offload_fraction == 1.0
    assert config.attention_backend_override == "flash"
    assert config.impl_cfg["use_thd"] is True
    assert config.impl_cfg["deterministic"] is False


def test_resync_format_is_validated_without_model_specific_engine_logic() -> None:
    assert _engine_config(resync_format="vllm_checkpoint").resync_format == "vllm_checkpoint"
    with pytest.raises(ValueError, match="resync_format"):
        _engine_config(resync_format="fp8")

    with pytest.raises(ValueError, match="resync_config requires resync_format"):
        _engine_config(resync_config={"expert_dtype": "fp8"})


def test_checkpoint_resync_format_is_forwarded_to_model_export(monkeypatch) -> None:
    engine = _engine(
        engine_config=_engine_config(
            resync_format="vllm_checkpoint",
            resync_config={"expert_dtype": "fp8"},
        )
    )
    calls = []

    class Runtime:
        def export_weights(self, handle, **kwargs):
            calls.append((handle, kwargs))
            return iter(())

    engine.runtime = Runtime()
    engine.handle = _fake_handle()
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.aggressive_empty_cache", lambda **_: None)

    weights, metadata = engine.get_per_tensor_param()

    assert list(weights) == []
    assert metadata is None
    assert calls == [
        (
            engine.handle,
            {
                "target": "vllm_checkpoint",
                "resync_config": {"expert_dtype": "fp8"},
                "export_dtype": "bfloat16",
            },
        )
    ]


def test_online_weight_export_uses_device_resident_runtime_defaults() -> None:
    engine = _engine(
        engine_config=_engine_config(model_name="qwen3_5", export_dtype="bfloat16")
    )
    captured = {}

    class Runtime:
        @staticmethod
        def export_weights(handle, **kwargs):
            captured["handle"] = handle
            captured["kwargs"] = kwargs
            return iter(())

    engine.runtime = Runtime()
    engine.handle = _fake_handle()
    engine._initial_sync_cache_cleared = True

    weights, metadata = engine.get_per_tensor_param(limit=3)

    assert list(weights) == []
    assert metadata is None
    assert captured == {
        "handle": engine.handle,
        "kwargs": {
            "export_dtype": "bfloat16",
            "limit": 3,
            "target": "vllm",
        },
    }


def test_resync_memory_protocol_evicts_before_export_and_restores_after(monkeypatch) -> None:
    """Optimizer/grad are evicted before the gather and restored after drain."""
    import megatron.lite.runtime.megatron_utils as mu

    engine = _engine(
        engine_config=_engine_config(param_offload=True, optimizer_offload=True)
    )
    events: list[str] = []

    class Runtime:
        @staticmethod
        def export_weights(handle, **kwargs):
            def gen():
                events.append("export:begin")
                yield ("w0", torch.zeros(1))
                events.append("export:end")

            return gen()

    engine.runtime = Runtime()
    engine.handle = _fake_handle(optimizer=object(), model_chunks=[object()])
    engine._initial_sync_cache_cleared = True

    monkeypatch.setattr(mu, "optimizer_states_on_gpu", lambda opt: True)
    monkeypatch.setattr(mu, "model_grads_resident", lambda chunks: True)
    monkeypatch.setattr(mu, "free_grad_buffers", lambda chunks: events.append("free_grad"))
    monkeypatch.setattr(
        mu,
        "load_model_to_gpu",
        lambda chunks, load_grad=True: events.append(f"reload_grad={load_grad}"),
    )

    def fake_to(device, model=True, optimizer=True, grad=True):
        events.append(f"to:{device}:model={model}:opt={optimizer}:grad={grad}")

    engine.to = fake_to

    weights, metadata = engine.get_per_tensor_param()
    assert metadata is None

    # Eviction happens eagerly, before the lazy export body runs.
    assert "to:cpu:model=False:opt=True:grad=False" in events
    assert "free_grad" in events
    assert "export:begin" not in events

    drained = list(weights)
    assert [name for name, _ in drained] == ["w0"]

    # After the export generator drains: grad reload then optimizer restore.
    assert events.index("export:end") < events.index("reload_grad=True")
    assert events.index("reload_grad=True") < events.index(
        "to:cuda:model=False:opt=True:grad=False"
    )


def test_resync_memory_protocol_is_noop_when_state_already_offloaded(monkeypatch) -> None:
    """When optimizer/grad are already off-GPU, no eviction/restore is issued."""
    import megatron.lite.runtime.megatron_utils as mu

    engine = _engine(engine_config=_engine_config(param_offload=False))
    to_calls: list[str] = []

    class Runtime:
        @staticmethod
        def export_weights(handle, **kwargs):
            return iter([("w", torch.zeros(1))])

    engine.runtime = Runtime()
    engine.handle = _fake_handle()
    engine._initial_sync_cache_cleared = True

    monkeypatch.setattr(mu, "optimizer_states_on_gpu", lambda opt: False)
    monkeypatch.setattr(mu, "model_grads_resident", lambda chunks: False)

    def fail_reload(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("no grad reload expected when nothing was freed")

    monkeypatch.setattr(mu, "load_model_to_gpu", fail_reload)
    engine.to = lambda device, **kwargs: to_calls.append(device)

    weights, _ = engine.get_per_tensor_param()
    assert [name for name, _ in weights] == ["w"]
    # is_param_offload default is False here, so no reload-to-cuda was issued and
    # no optimizer offload/restore round-trip occurred.
    assert to_calls == []


def test_resync_smoke_exit_after_first_successful_sync(monkeypatch) -> None:
    """MLITE_RESYNC_SMOKE_EXIT_AFTER exits cleanly once the peak survives."""
    monkeypatch.setenv("MLITE_RESYNC_SMOKE_EXIT_AFTER", "1")
    engine = _engine(engine_config=_engine_config(param_offload=False))

    class Runtime:
        @staticmethod
        def export_weights(handle, **kwargs):
            return iter([("w", torch.zeros(1))])

    engine.runtime = Runtime()
    engine.handle = _fake_handle()
    engine._initial_sync_cache_cleared = True

    weights, _ = engine.get_per_tensor_param()
    with pytest.raises(SystemExit) as excinfo:
        list(weights)
    assert excinfo.value.code == 0


def test_resync_no_smoke_exit_without_env(monkeypatch) -> None:
    monkeypatch.delenv("MLITE_RESYNC_SMOKE_EXIT_AFTER", raising=False)
    engine = _engine(engine_config=_engine_config(param_offload=False))

    class Runtime:
        @staticmethod
        def export_weights(handle, **kwargs):
            return iter([("w", torch.zeros(1))])

    engine.runtime = Runtime()
    engine.handle = _fake_handle()
    engine._initial_sync_cache_cleared = True

    weights, _ = engine.get_per_tensor_param()
    assert [name for name, _ in weights] == ["w"]


def test_resync_export_empty_caches_and_tracks_worst_tensor_per_tensor(monkeypatch) -> None:
    """Per-tensor residency control: after each exported tensor is consumed the
    caching allocator is reclaimed and the worst single-tensor peak is tracked,
    so a failed colocated resync pins the single-tensor lower bound."""
    engine = _engine(engine_config=_engine_config(param_offload=False))

    class Runtime:
        @staticmethod
        def export_weights(handle, **kwargs):
            return iter(
                [("a", torch.zeros(1)), ("b", torch.zeros(2)), ("c", torch.zeros(3))]
            )

    engine.runtime = Runtime()
    engine.handle = _fake_handle()
    engine._initial_sync_cache_cleared = True

    calls = {"empty_cache": 0, "reset_peak": 0}
    # peak established when each tensor is produced (b is the worst)
    peaks = iter([10, 40, 25])
    current = {"peak": 10}

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            calls["empty_cache"] += 1

        @staticmethod
        def reset_peak_memory_stats():
            calls["reset_peak"] += 1
            current["peak"] = next(peaks, 0)

        @staticmethod
        def max_memory_allocated():
            return current["peak"]

        @staticmethod
        def memory_allocated():
            return 0

        @staticmethod
        def memory_reserved():
            return 0

    import verl_mlite.engine.mlite_engine as me

    monkeypatch.setattr(me.torch, "cuda", _FakeCuda)

    printed: list[str] = []
    monkeypatch.setattr(
        "builtins.print", lambda *a, **k: printed.append(" ".join(str(x) for x in a))
    )

    weights, _ = engine.get_per_tensor_param()
    drained = list(weights)

    assert [name for name, _ in drained] == ["a", "b", "c"]
    assert [t.numel() for _, t in drained] == [1, 2, 3]
    # one empty_cache per exported tensor; one reset at export_begin + one/tensor
    assert calls["empty_cache"] == 3
    assert calls["reset_peak"] == 4
    curve_line = next(p for p in printed if "MLITE_RESYNC_MEMCURVE" in p)
    assert "worst_tensor=b" in curve_line


def test_local_lr_scheduler_warmup_decay_and_state_roundtrip() -> None:
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.1}])
    opt = SimpleNamespace(
        total_training_steps=4,
        lr_warmup_steps=1,
        lr_warmup_steps_ratio=0.0,
        lr_warmup_init=0.0,
        lr=1.0,
        min_lr=0.1,
        lr_decay_steps=4,
        lr_decay_style="linear",
        weight_decay=0.1,
        weight_decay_incr_style="constant",
        lr_wsd_decay_steps=None,
        lr_wsd_decay_style="exponential",
    )

    scheduler = _build_lr_scheduler(optimizer, opt)

    assert optimizer.param_groups[0]["lr"] == 0.0
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == 1.0
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.7)

    state = scheduler.state_dict()
    scheduler.step(10)
    scheduler.load_state_dict(state)

    assert scheduler.state_dict() == state
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.7)
