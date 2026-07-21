# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Train-mode context exit hands the post-step residual back before the vLLM wake.

The colocated resync OOM fix is internalized: instead of a
verl-called hook, the engine releases the backend export scratch and parks the
optimizer at its own offload exit -- ``_MegatronLiteModeCtx.__exit__`` for the
training-mode context, which verl's fit loop always runs at the end of
``update_actor`` and before the subsequent ``update_weights`` weight-pool wake.
These tests pin that the release fires exactly on the clean train exit (not on
eval, not on a faulted exit) and keeps the sharded weights resident.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from verl_mlite.engine.config import MegatronLiteEngineConfig
from verl_mlite.engine.mlite_engine import MegatronLiteEngine


@pytest.fixture(autouse=True)
def _single_process_dist(monkeypatch):
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.dist.is_initialized", lambda: False)


def _optimizer_config() -> SimpleNamespace:
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
        override_optimizer_config={},
    )


def _engine_config(**kwargs) -> MegatronLiteEngineConfig:
    values = {"custom_backend_module": None, "impl_cfg": {"use_thd": True}}
    values.update(kwargs)
    return MegatronLiteEngineConfig(**values)


def _initialized_engine(*, param_offload=False, optimizer_offload=False):
    engine = MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path="/tmp/qwen35", hf_config={"model_type": "qwen3_5_moe"}, mtp=None
        ),
        engine_config=_engine_config(
            param_offload=param_offload, optimizer_offload=optimizer_offload
        ),
        optimizer_config=_optimizer_config(),
        checkpoint_config={},
    )

    release_calls = []

    @contextmanager
    def _mode_ctx(_handle):
        yield

    runtime = SimpleNamespace(
        train_mode=_mode_ctx,
        eval_mode=_mode_ctx,
        release_export_scratch=lambda handle: release_calls.append(handle),
    )
    handle = SimpleNamespace(_optimizer=object(), _lr_scheduler=object())
    engine.runtime = runtime
    engine.handle = handle
    return engine, release_calls, handle


def _wire_recorders(engine, monkeypatch):
    to_calls = []
    empty_cache_calls = []
    monkeypatch.setattr(engine, "to", lambda **kwargs: to_calls.append(kwargs))
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.aggressive_empty_cache",
        lambda **kwargs: empty_cache_calls.append(kwargs),
    )
    return to_calls, empty_cache_calls


def test_train_mode_exit_releases_scratch_and_parks_optimizer(monkeypatch):
    # param_offload on, optimizer_offload off: the generic offload leaves the Adam
    # moments on GPU, so the internalized release must park them before the wake.
    engine, release_calls, handle = _initialized_engine(
        param_offload=True, optimizer_offload=False
    )
    to_calls, empty_cache_calls = _wire_recorders(engine, monkeypatch)

    with engine.train_mode():
        pass

    # Scratch handed back to the driver + allocator drained exactly once.
    assert release_calls == [handle]
    assert empty_cache_calls == [{"force_sync": True}]
    # The optimizer park is the only cpu-move with optimizer=True/model=False; the
    # sharded weights are NOT moved by the release (they stay resident as the
    # export gather source -- the generic context offload owns model movement).
    park_calls = [c for c in to_calls if c.get("optimizer") and not c.get("model")]
    assert park_calls == [{"device": "cpu", "model": False, "optimizer": True, "grad": False}]


def test_train_mode_exit_skips_optimizer_park_when_already_offloaded(monkeypatch):
    # optimizer_offload on: the generic exit already parks the Adam moments, so the
    # release must not double-park; it still drops the scratch + drains.
    engine, release_calls, handle = _initialized_engine(
        param_offload=True, optimizer_offload=True
    )
    to_calls, empty_cache_calls = _wire_recorders(engine, monkeypatch)

    with engine.train_mode():
        pass

    assert release_calls == [handle]
    assert empty_cache_calls == [{"force_sync": True}]
    park_calls = [c for c in to_calls if c.get("optimizer") and not c.get("model")]
    assert park_calls == []


def test_eval_mode_exit_does_not_release_scratch(monkeypatch):
    engine, release_calls, _ = _initialized_engine(param_offload=True)
    _to_calls, empty_cache_calls = _wire_recorders(engine, monkeypatch)

    with engine.eval_mode():
        pass

    assert release_calls == []
    assert empty_cache_calls == []


def test_faulted_train_exit_does_not_release_scratch(monkeypatch):
    # A training pass that raises must not run the resync release: the state may be
    # inconsistent and no weight wake follows a failed update_actor.
    engine, release_calls, _ = _initialized_engine(param_offload=True)
    _to_calls, empty_cache_calls = _wire_recorders(engine, monkeypatch)

    with pytest.raises(RuntimeError, match="boom"):
        with engine.train_mode():
            raise RuntimeError("boom")

    assert release_calls == []
    assert empty_cache_calls == []
