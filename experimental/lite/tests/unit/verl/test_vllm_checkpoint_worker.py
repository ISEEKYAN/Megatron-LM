from types import SimpleNamespace

import torch


def test_vllm_server_profile_isolated_to_ray_actor_options(monkeypatch) -> None:
    from verl_mlite.compat import _RayActorClassProfile, _vllm_server_profile_env

    monkeypatch.setenv("PYTHONPATH", "/training")
    monkeypatch.setenv("LD_PRELOAD", "/training/base.so")
    monkeypatch.setenv("VERL_MLITE_VLLM_SITE", "/rollout")
    monkeypatch.setenv("VERL_MLITE_VLLM_LD_PRELOAD", "/rollout/shim.so")
    profile = _vllm_server_profile_env()
    calls = []

    class ActorClass:
        def options(self, **kwargs):
            calls.append(kwargs)
            return "configured-actor"

    wrapped = _RayActorClassProfile(ActorClass(), profile)
    result = wrapped.options(
        name="server",
        runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
    )

    assert result == "configured-actor"
    assert calls == [
        {
            "name": "server",
            "runtime_env": {
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "PYTHONPATH": "/rollout:/training",
                    "LD_PRELOAD": "/rollout/shim.so:/training/base.so",
                    "PYTHONNOUSERSITE": "1",
                }
            },
        }
    ]


def test_vllm_server_profile_is_disabled_without_explicit_site(monkeypatch) -> None:
    from verl_mlite.compat import _vllm_server_profile_env

    monkeypatch.delenv("VERL_MLITE_VLLM_SITE", raising=False)
    monkeypatch.delenv("VERL_MLITE_VLLM_LD_PRELOAD", raising=False)

    assert _vllm_server_profile_env() == {}


def test_checkpoint_bucket_reload_has_one_lifecycle_for_all_buckets() -> None:
    from verl_mlite.rollout.vllm_worker import reload_checkpoint_buckets

    events = []

    class Model:
        def load_weights(self, weights):
            events.append(("load", [name for name, _ in weights]))

    model = Model()
    runner = SimpleNamespace(model=model, model_config=object())

    def receive(callback):
        callback([("w1.weight", object()), ("w1.scale", object())])
        callback([("w2.weight", object()), ("w2.scale", object())])

    reload_checkpoint_buckets(
        runner,
        receive,
        initialize=lambda value: events.append(("begin", value)),
        finalize=lambda value, config: events.append(("finish", value, config)),
    )

    assert events == [
        ("begin", model),
        ("load", ["w1.weight", "w1.scale"]),
        ("load", ["w2.weight", "w2.scale"]),
        ("finish", model, runner.model_config),
    ]


def test_checkpoint_bucket_reload_does_not_finalize_failed_partial_update() -> None:
    import pytest

    from verl_mlite.rollout.vllm_worker import reload_checkpoint_buckets

    events = []

    class Model:
        def load_weights(self, weights):
            del weights
            raise RuntimeError("broken bucket")

    runner = SimpleNamespace(model=Model(), model_config=object())

    with pytest.raises(RuntimeError, match="broken bucket"):
        reload_checkpoint_buckets(
            runner,
            lambda callback: callback([("w", object())]),
            initialize=lambda model: events.append("begin"),
            finalize=lambda model, config: events.append("finish"),
        )

    assert events == ["begin"]


def test_checkpoint_path_reload_uses_vllm_native_checkpoint_lifecycle() -> None:
    from verl_mlite.rollout.vllm_worker import VllmCheckpointPathWorkerExtension

    calls = []
    extension = SimpleNamespace(
        model_runner=SimpleNamespace(
            reload_weights=lambda **kwargs: calls.append(kwargs),
        )
    )

    VllmCheckpointPathWorkerExtension.reload_checkpoint_from_path(
        extension, "/tmp/resync"
    )

    assert calls == [{"weights_path": "/tmp/resync", "is_checkpoint_format": True}]


def test_checkpoint_state_fingerprints_detect_parameter_changes() -> None:
    from verl_mlite.rollout.vllm_worker import checkpoint_state_fingerprints

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
            self.register_buffer("weight_scale", torch.tensor([0.5]))

    model = Model()
    runner = SimpleNamespace(model=model)
    before = checkpoint_state_fingerprints(runner, chunk_bytes=3)
    with torch.no_grad():
        model.weight[2] += 1
    after = checkpoint_state_fingerprints(runner, chunk_bytes=3)

    assert [record["name"] for record in before] == ["weight", "weight_scale"]
    assert before[0]["kind"] == "parameter"
    assert before[0]["dtype"] == "float32"
    assert before[0]["shape"] == [8]
    assert before[0]["nbytes"] == 32
    assert len(before[0]["sha256"]) == 64
    assert before[0]["sha256"] != after[0]["sha256"]
    assert before[1]["sha256"] == after[1]["sha256"]


def test_proxy_worker_module_does_not_import_verl(monkeypatch) -> None:
    import builtins
    import importlib
    import sys

    module_name = "verl_mlite.rollout.vllm_worker"
    sys.modules.pop(module_name, None)
    original_import = builtins.__import__

    def reject_verl(name, *args, **kwargs):
        if name == "verl" or name.startswith("verl."):
            raise AssertionError(f"proxy worker imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_verl)
    module = importlib.import_module(module_name)

    assert module.VllmCheckpointPathWorkerExtension is not None
