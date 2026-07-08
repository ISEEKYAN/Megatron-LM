import os


def test_proxy_impl_config_is_concrete_and_uses_local_attention() -> None:
    from examples.verl.ds4_resync_proxy import _proxy_impl_config

    impl = _proxy_impl_config()
    parallel = impl.parallel

    assert (
        parallel.tp,
        parallel.etp,
        parallel.ep,
        parallel.pp,
        parallel.vpp,
        parallel.cp,
    ) == (
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert impl.attention_backend_override == "local"
    assert impl.optimizer is None
    assert impl.mtp_enable is False


def test_proxy_initializes_a_local_single_gpu_process_group(monkeypatch) -> None:
    from examples.verl import ds4_resync_proxy

    calls = []
    monkeypatch.setattr(ds4_resync_proxy.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(ds4_resync_proxy.torch.cuda, "set_device", calls.append)
    monkeypatch.setattr(
        ds4_resync_proxy.dist,
        "init_process_group",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    ds4_resync_proxy._initialize_single_gpu_process_group()

    assert calls[0] == 0
    args, kwargs = calls[1]
    assert args == ("nccl",)
    assert kwargs["init_method"].startswith("file://")
    assert kwargs["rank"] == 0
    assert kwargs["world_size"] == 1
    assert kwargs["device_id"] == ds4_resync_proxy.torch.device("cuda:0")


def test_proxy_handoff_removes_torchrun_state(monkeypatch) -> None:
    from examples.verl import ds4_resync_proxy

    calls = []
    monkeypatch.setattr(ds4_resync_proxy.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        ds4_resync_proxy.dist, "destroy_process_group", lambda: calls.append("destroy")
    )
    monkeypatch.setattr(
        ds4_resync_proxy.torch.cuda, "synchronize", lambda: calls.append("sync")
    )
    monkeypatch.setattr(
        ds4_resync_proxy.torch.cuda, "empty_cache", lambda: calls.append("empty")
    )
    for name in ds4_resync_proxy._TORCHRUN_ENVIRONMENT:
        monkeypatch.setenv(name, "stale")

    ds4_resync_proxy._handoff_to_vllm()

    assert calls == ["destroy", "sync", "empty"]
    assert all(name not in os.environ for name in ds4_resync_proxy._TORCHRUN_ENVIRONMENT)
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
