import os
import sys


def test_proxy_config_covers_vllm_ds4_constructor_fields() -> None:
    from examples.verl.ds4_resync_proxy import tiny_config

    required = {
        "expert_dtype",
        "hc_eps",
        "hc_mult",
        "hc_sinkhorn_iters",
        "hidden_act",
        "hidden_size",
        "index_topk",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "norm_topk_prob",
        "num_attention_heads",
        "num_experts_per_tok",
        "num_hash_layers",
        "num_hidden_layers",
        "rms_norm_eps",
        "rope_parameters",
        "swiglu_limit",
        "topk_method",
        "vocab_size",
    }

    config = tiny_config()
    assert required <= config.keys()
    assert config["hidden_act"] == "silu"
    assert config["topk_method"] == "noaux_tc"
    assert config["rope_parameters"]["rope_type"] != "default"
    assert (
        config["rope_parameters"]["factor"]
        * config["rope_parameters"]["original_max_position_embeddings"]
        == config["max_position_embeddings"]
    )


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


def test_proxy_isolates_vllm_warmup_from_unrelated_models(monkeypatch) -> None:
    from examples.verl import ds4_resync_proxy

    module_name = "vllm.model_executor.warmup.minimax_m3_msa_warmup"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    ds4_resync_proxy._isolate_vllm_warmup_from_unrelated_models()

    shim = sys.modules[module_name]
    assert shim.minimax_m3_msa_warmup(object()) is None
