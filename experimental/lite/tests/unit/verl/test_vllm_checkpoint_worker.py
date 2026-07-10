import inspect
import sys
from types import SimpleNamespace

import torch


def test_unknown_hf_model_type_is_registered_as_opaque_config(monkeypatch) -> None:
    from verl_mlite import compat

    registrations = []

    class PretrainedConfig:
        pass

    class AutoConfig:
        @classmethod
        def register(cls, model_type, config_cls):
            registrations.append((model_type, config_cls))

    monkeypatch.setenv("VERL_MLITE_HF_CONFIG_MODEL_TYPE", "deepseek_v4")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=AutoConfig, PretrainedConfig=PretrainedConfig),
    )
    monkeypatch.setattr(compat, "_REGISTERED_HF_CONFIG_TYPES", set())

    assert compat._register_opaque_hf_config()
    assert not compat._register_opaque_hf_config()
    assert registrations[0][0] == "deepseek_v4"
    assert issubclass(registrations[0][1], PretrainedConfig)
    assert registrations[0][1].model_type == "deepseek_v4"


def test_vllm_server_profile_keeps_rollout_dependency_closure_first(
    monkeypatch,
) -> None:
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


def test_vllm_server_profile_keeps_shared_rollout_dependencies_on_thin_site(
    monkeypatch, tmp_path
) -> None:
    from verl_mlite import compat

    training_site = tmp_path / "training"
    rollout_site = tmp_path / "rollout"
    for site in (training_site, rollout_site):
        (site / "transformers").mkdir(parents=True)
        (site / "transformers/__init__.py").write_text("")
        (site / "vllm").mkdir()
        (site / "vllm/__init__.py").write_text("")
        (site / "compressed_tensors").mkdir()
        (site / "compressed_tensors/__init__.py").write_text("")

    monkeypatch.setenv("PYTHONPATH", str(training_site))
    monkeypatch.setenv("VERL_MLITE_VLLM_SITE", str(rollout_site))
    profile = compat._vllm_server_profile_env()
    search_path = profile["PYTHONPATH"].split(compat.os.pathsep)

    transformers_spec = compat.importlib.machinery.PathFinder.find_spec(
        "transformers", search_path
    )
    vllm_spec = compat._VllmThinFinder(str(rollout_site)).find_spec("vllm", None, None)
    dependency_spec = compat.importlib.machinery.PathFinder.find_spec(
        "compressed_tensors", search_path
    )

    assert transformers_spec is not None
    assert transformers_spec.origin == str(rollout_site / "transformers/__init__.py")
    assert vllm_spec is not None
    assert vllm_spec.origin == str(rollout_site / "vllm/__init__.py")
    assert dependency_spec is not None
    assert dependency_spec.origin == str(
        rollout_site / "compressed_tensors/__init__.py"
    )


def test_transformers_vision2seq_alias_uses_v5_replacement(monkeypatch) -> None:
    from verl_mlite import compat

    replacement = object()
    transformers = SimpleNamespace(AutoModelForImageTextToText=replacement)
    monkeypatch.setenv("VERL_MLITE_VLLM_SITE", "/rollout")
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    assert compat._patch_transformers_vision2seq_alias()
    assert transformers.AutoModelForVision2Seq is replacement
    assert not compat._patch_transformers_vision2seq_alias()


def test_transformers_vision2seq_alias_preserves_v4_class(monkeypatch) -> None:
    from verl_mlite import compat

    original = object()
    transformers = SimpleNamespace(
        AutoModelForVision2Seq=original,
        AutoModelForImageTextToText=object(),
    )
    monkeypatch.setenv("VERL_MLITE_VLLM_SITE", "/rollout")
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    assert not compat._patch_transformers_vision2seq_alias()
    assert transformers.AutoModelForVision2Seq is original


def test_transformers_vision2seq_alias_requires_explicit_rollout_site(
    monkeypatch,
) -> None:
    from verl_mlite import compat

    transformers = SimpleNamespace(AutoModelForImageTextToText=object())
    monkeypatch.delenv("VERL_MLITE_VLLM_SITE", raising=False)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    assert not compat._patch_transformers_vision2seq_alias()
    assert not hasattr(transformers, "AutoModelForVision2Seq")


def test_transformers_vision2seq_alias_requires_v5_replacement(monkeypatch) -> None:
    from verl_mlite import compat

    transformers = SimpleNamespace()
    monkeypatch.setenv("VERL_MLITE_VLLM_SITE", "/rollout")
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    assert not compat._patch_transformers_vision2seq_alias()
    assert not hasattr(transformers, "AutoModelForVision2Seq")


def test_transformers_alias_precedes_verl_importing_runtime_patches() -> None:
    from verl_mlite import compat

    source = inspect.getsource(compat.apply_runtime_patches)

    assert source.index("_patch_transformers_vision2seq_alias()") < source.index(
        "_register_opaque_hf_config()"
    )
    assert source.index("_patch_transformers_vision2seq_alias()") < source.index(
        "_patch_verl_vllm_device_uuid()"
    )
    assert source.index("_patch_transformers_vision2seq_alias()") < source.index(
        "_patch_vllm_server_profile()"
    )


def test_vllm_server_profile_is_disabled_without_explicit_site(monkeypatch) -> None:
    from verl_mlite.compat import _vllm_server_profile_env

    monkeypatch.delenv("VERL_MLITE_VLLM_SITE", raising=False)
    monkeypatch.delenv("VERL_MLITE_VLLM_LD_PRELOAD", raising=False)

    assert _vllm_server_profile_env() == {}


def test_vllm_thin_finder_routes_only_the_vllm_top_level(tmp_path) -> None:
    from verl_mlite.compat import _VllmThinFinder

    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("ORIGIN = 'thin'\n")
    finder = _VllmThinFinder(str(tmp_path))

    assert finder.find_spec("transformers", None, None) is None
    spec = finder.find_spec("vllm", None, None)
    assert spec is not None
    assert spec.origin == str(package / "__init__.py")


def test_vllm_triton_kernels_alias_prefers_vendored_package(monkeypatch) -> None:
    from verl_mlite import compat

    external = SimpleNamespace(__file__="/base/triton_kernels/__init__.py")
    vendored = SimpleNamespace(__file__="/rollout/vllm/third_party/triton_kernels/__init__.py")
    monkeypatch.setenv("VERL_MLITE_VLLM_SITE", "/rollout")
    monkeypatch.setitem(sys.modules, "triton_kernels", external)
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda name: (
            vendored
            if name == "vllm.third_party.triton_kernels"
            else (_ for _ in ()).throw(AssertionError(name))
        ),
    )

    assert compat._install_vllm_triton_kernels_alias()
    assert sys.modules["triton_kernels"] is vendored
    assert not compat._install_vllm_triton_kernels_alias()


def test_vllm_device_uuid_normalizes_physical_id_for_one_visible_gpu(
    monkeypatch,
) -> None:
    from verl_mlite import compat

    calls = []

    def get_device_uuid(device_id):
        calls.append(device_id)
        return f"GPU-{device_id}"

    utils = SimpleNamespace(get_device_uuid=get_device_uuid)
    monkeypatch.setenv("VERL_MLITE_VLLM_SITE", "/rollout")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.utils",
        utils,
    )

    assert compat._patch_verl_vllm_device_uuid()
    assert not compat._patch_verl_vllm_device_uuid()
    assert utils.get_device_uuid(5) == "GPU-0"
    assert utils.get_device_uuid(0) == "GPU-0"
    assert utils.get_device_uuid(3) == "GPU-3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-opaque")
    assert utils.get_device_uuid(5) == "GPU-5"
    assert calls == [0, 0, 3, 5]


def test_vllm_device_uuid_maps_physical_id_with_multiple_visible_gpus(
    monkeypatch,
) -> None:
    from verl_mlite import compat

    calls = []

    def get_device_uuid(device_id):
        calls.append(device_id)
        return f"GPU-{device_id}"

    utils = SimpleNamespace(get_device_uuid=get_device_uuid)
    monkeypatch.setenv("VERL_MLITE_VLLM_SITE", "/rollout")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.utils",
        utils,
    )

    assert compat._patch_verl_vllm_device_uuid()
    assert utils.get_device_uuid(5) == "GPU-1"
    assert utils.get_device_uuid(1) == "GPU-1"
    assert utils.get_device_uuid(3) == "GPU-3"
    assert calls == [1, 1, 3]


def test_vllm_device_uuid_patch_repairs_loaded_consumer_alias(monkeypatch) -> None:
    from verl_mlite import compat

    calls = []

    def get_device_uuid(device_id):
        calls.append(device_id)
        return f"GPU-{device_id}"

    utils = SimpleNamespace(get_device_uuid=get_device_uuid)
    consumer = SimpleNamespace(get_device_uuid=get_device_uuid)
    monkeypatch.setenv("VERL_MLITE_VLLM_SITE", "/rollout")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.utils",
        utils,
    )
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.vllm_rollout",
        consumer,
    )

    assert compat._patch_verl_vllm_device_uuid()
    assert consumer.get_device_uuid is utils.get_device_uuid
    assert consumer.get_device_uuid(5) == "GPU-0"

    consumer.get_device_uuid = get_device_uuid
    assert compat._patch_verl_vllm_device_uuid()
    assert consumer.get_device_uuid is utils.get_device_uuid
    assert consumer.get_device_uuid(5) == "GPU-0"
    assert not compat._patch_verl_vllm_device_uuid()
    assert calls == [0, 0]


def test_vllm_device_uuid_patch_requires_explicit_rollout_site(monkeypatch) -> None:
    from verl_mlite import compat

    monkeypatch.delenv("VERL_MLITE_VLLM_SITE", raising=False)

    assert not compat._patch_verl_vllm_device_uuid()


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
