from types import SimpleNamespace


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
    from verl_mlite.rollout.vllm_worker import VllmCheckpointWorkerExtension

    calls = []
    extension = SimpleNamespace(
        model_runner=SimpleNamespace(
            reload_weights=lambda **kwargs: calls.append(kwargs),
        )
    )

    VllmCheckpointWorkerExtension.reload_checkpoint_from_path(extension, "/tmp/resync")

    assert calls == [{"weights_path": "/tmp/resync", "is_checkpoint_format": True}]
