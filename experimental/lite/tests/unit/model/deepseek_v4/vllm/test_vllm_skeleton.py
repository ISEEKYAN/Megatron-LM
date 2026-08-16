from __future__ import annotations

import importlib.util
import inspect
import sys
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    _canonicalize_sparse_topk_indices,
)

from megatron.lite.model import registry
from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm import protocol
from megatron.lite.model.deepseek_v4.vllm.model import (
    AttentionAdapters,
    AttentionKernelMetadata,
    DeepseekV4Model,
    _AttentionState,
    _RMSNormState,
)
from megatron.lite.model.deepseek_v4.vllm.moe import (
    DeepseekV4MoE,
    MoEKernelMetadata,
    _kernel_topk_weights,
)
from megatron.lite.primitive.autograd import inference_only
from megatron.lite.primitive.modules.router_replay import (
    RouterReplay,
    RouterReplayAction,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig


def test_sparse_prefill_topk_has_canonical_reduction_order() -> None:
    unordered = torch.tensor(
        [[5, -1, 2, 9], [3, 7, -1, 1]], dtype=torch.int32
    )
    actual = _canonicalize_sparse_topk_indices(unordered)

    torch.testing.assert_close(
        actual,
        torch.tensor([[9, 5, 2, -1], [7, 3, 1, -1]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_final_rmsnorm_uses_vllm_batch_invariant_kernel(monkeypatch) -> None:
    from vllm.model_executor.layers import batch_invariant

    norm = _RMSNormState(8)
    value = torch.randn(3, 8, dtype=torch.bfloat16)
    sentinel = torch.full_like(value, 7)
    calls = []

    def fake_rms_norm(input_value, weight, eps):
        calls.append((input_value, weight, eps))
        return sentinel

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(batch_invariant, "rms_norm_batch_invariant", fake_rms_norm)

    assert norm(value, 1e-6) is sentinel
    assert calls == [(value, norm.weight, 1e-6)]


def test_attention_restores_fp32_rope_after_fsdp_input_cast() -> None:
    attention = _AttentionState.__new__(_AttentionState)
    torch.nn.Module.__init__(attention)
    metadata = SimpleNamespace(
        cos_sin_cache=torch.empty(8, 4, dtype=torch.bfloat16)
    )

    class _StopAfterBoundary(Exception):
        pass

    def stop(_stage):
        raise _StopAfterBoundary

    attention._selected = stop
    with pytest.raises(_StopAfterBoundary):
        attention.forward(torch.empty(2, 4), metadata=metadata)
    assert metadata.cos_sin_cache.dtype == torch.float32


def test_grouped_moe_restores_fp32_router_weights_after_fsdp_cast() -> None:
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.bfloat16)
    restored = _kernel_topk_weights(weights)
    assert restored.dtype == torch.float32
    torch.testing.assert_close(restored, weights.float(), rtol=0, atol=0)


def _tiny_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=16,
        hidden_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=4,
        qk_rope_head_dim=2,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        hc_mult=2,
        num_nextn_predict_layers=1,
    )


def _tiny_4l_config() -> DeepseekV4Config:
    config = _tiny_config()
    config.num_hidden_layers = 4
    config.num_hash_layers = 3
    config.compress_ratios = [0, 0, 4, 128]
    config.index_head_dim = 4
    config.index_n_heads = 2
    config.index_topk = 2
    return config


def test_registry_exposes_distinct_vllm_runtime() -> None:
    assert registry.resolve_runtime_model_name("deepseek_v4", "vllm") == "deepseek_v4_vllm"
    assert (
        registry.TRAIN_RUNTIME_MODULES["deepseek_v4_vllm"]
        == "megatron.lite.model.deepseek_v4.vllm.protocol"
    )


def test_package_has_no_sibling_lite_imports() -> None:
    package = Path(inspect.getfile(protocol)).parent
    forbidden = "megatron.lite.model.deepseek_v4." + "lite."
    for source in package.glob("*.py"):
        assert forbidden not in source.read_text(), source


def test_selector_is_frozen_normalized_and_explicit() -> None:
    selector = protocol.SelectorConfig([1, 0], ["moe", "attn"])
    assert selector.global_layer_ids == (1, 0)
    assert selector.module_names == (
        "router_moe",
        "deepep",
        "mhc",
        "linear",
        "kv_flashmla",
        "o_proj",
    )
    assert selector.selects(1, "linear")
    assert not selector.selects(1, "router")
    with pytest.raises(FrozenInstanceError):
        selector.module_names = ()  # type: ignore[misc]
    with pytest.raises(ValueError):
        protocol.SelectorConfig((2,), ("router",))


def test_impl_config_normalizes_hydra_selector_mapping() -> None:
    config = protocol.ImplConfig(
        selector={
            "global_layer_ids": [0, 1],
            "module_names": ["attn", "moe"],
        }
    )

    assert isinstance(config.selector, protocol.SelectorConfig)
    assert config.selector.global_layer_ids == (0, 1)
    assert config.selector.selects(1, "kv_flashmla")
    assert config.selector.selects(1, "deepep")


def test_r3_protocol_uses_contiguous_ds4_token_layout() -> None:
    model = nn.Module()
    model.ps = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        cp_size=2,
        cp_rank=1,
        cp_group=None,
    )
    batch = PackedBatch(
        input_ids=torch.arange(8),
        labels=torch.arange(8),
        seq_lens=torch.tensor([4, 4]),
        r3_replay_mask=torch.tensor(
            [True, False, True, False, False, True, False, True]
        ),
    )
    routes = torch.arange(8).view(2, 4, 1, 1)

    packed = protocol.pack_routed_experts(model, batch, routes)
    mask = protocol.pack_r3_replay_mask(model, batch)

    assert len(packed) == 1
    assert torch.equal(packed[0], torch.arange(4, 8).view(4, 1))
    assert torch.equal(mask, batch.r3_replay_mask[4:])


@pytest.mark.parametrize("hash_layer", [True, False])
def test_r3_replays_ids_and_recomputes_live_actor_weights(hash_layer: bool) -> None:
    config = _tiny_config()
    moe = DeepseekV4MoE(
        config,
        ParallelState(),
        layer_idx=0 if hash_layer else 1,
    )
    replay = RouterReplay()
    moe.gate.router_replay = replay
    logits = torch.tensor(
        [[-2.0, -0.5, 0.5, 2.0], [1.5, -1.0, 0.25, -0.25]],
        dtype=torch.float32,
    )
    native_ids = torch.tensor([[0, 1], [2, 3]])
    native_weights = torch.full((2, 2), -1.0)
    target = torch.tensor([[3, 2], [0, 1]])
    replay.target_topk_idx = target
    replay.target_replay_mask = torch.tensor([True, False])
    replay.router_replay_action = RouterReplayAction.REPLAY_FORWARD

    weights, ids = moe._replay_route(logits, native_weights, native_ids)

    expected_ids = torch.tensor([[3, 2], [2, 3]])
    dense = torch.sqrt(torch.nn.functional.softplus(logits))
    expected_weights = dense.gather(-1, expected_ids)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    expected_weights *= config.routed_scaling_factor
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


@pytest.mark.parametrize(
    "parallel",
    [
        ParallelConfig(tp=2),
        ParallelConfig(etp=2),
        ParallelConfig(pp=2),
        ParallelConfig(vpp=2),
    ],
)
def test_parallel_contract_fails_closed(parallel: ParallelConfig) -> None:
    with pytest.raises(NotImplementedError):
        protocol._validate_contract(
            _tiny_config(),
            protocol.ImplConfig(parallel=parallel),
        )


def test_parallel_contract_accepts_cp_with_required_ep_deepep() -> None:
    protocol._validate_contract(
        _tiny_config(),
        protocol.ImplConfig(
            parallel=ParallelConfig(cp=2, ep=2),
            use_deepep=True,
        ),
    )


def test_cp_forward_inputs_match_lite_contiguous_padded_layout() -> None:
    model = nn.Module()
    model.ps = ParallelState(cp_size=2, cp_rank=1, tp_size=1)
    batch = PackedBatch(
        input_ids=torch.arange(7),
        labels=torch.arange(10, 17),
        loss_mask=torch.ones(7),
        seq_lens=torch.tensor([3, 4]),
    )

    inputs, padded_lengths = protocol._prepare_cp_forward_inputs(model, batch)

    assert padded_lengths == [4, 4]
    assert inputs["input_ids"].shape == (4,)
    assert inputs["position_ids"].shape == (4,)
    assert inputs["labels"].shape == (4,)
    assert inputs["loss_mask"].shape == (4,)


def test_attention_cp_gathers_through_model_owned_group(monkeypatch) -> None:
    attention = _AttentionState.__new__(_AttentionState)
    nn.Module.__init__(attention)
    group = object()
    attention.ps = SimpleNamespace(cp_size=2, cp_rank=0, cp_group=group)
    metadata = SimpleNamespace(cos_sin_cache=torch.empty(8, 4, dtype=torch.float32))
    local = torch.arange(8, dtype=torch.float32).view(2, 4)
    gathered = []

    def fake_all_gather(value, *, group):
        assert group is attention.ps.cp_group
        return (value, value + 100)

    monkeypatch.setattr(torch.distributed.nn.functional, "all_gather", fake_all_gather)

    class _Stop(Exception):
        pass

    def capture(value):
        gathered.append(value)
        raise _Stop

    attention._selected = lambda _stage: None
    attention._input_projections = capture
    with pytest.raises(_Stop):
        attention.forward(local, metadata=metadata)

    assert torch.equal(gathered[0], torch.cat((local, local + 100), dim=0))


def test_build_contract_preserves_release_master_dtypes(monkeypatch) -> None:
    monkeypatch.setattr(
        protocol,
        "initialize_ds4_vllm_batch_invariance",
        lambda: None,
    )
    monkeypatch.setattr(
        protocol,
        "init_parallel",
        lambda _config: ParallelState(ep_size=1, ep_rank=0),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        SimpleNamespace(VllmConfig=lambda: object()),
    )
    impl = protocol.ImplConfig(
        parallel=ParallelConfig(ep=2),
        use_deepep=True,
        selector=protocol.SelectorConfig((0,), ("attn", "moe")),
    )
    bundle = protocol.build_model(_tiny_config(), impl_cfg=impl)
    assert bundle.optimizer is None
    assert bundle.finalize_grads is None
    assert "pre_forward_hook" not in bundle.extras
    assert len(bundle.chunks[0].mtp) == 0
    floating = {
        name: value.dtype
        for name, value in bundle.chunks[0].state_dict().items()
        if value.is_floating_point()
    }
    assert floating
    assert set(floating.values()) == {torch.bfloat16, torch.float32}
    assert all(
        dtype == torch.float32
        for name, dtype in floating.items()
        if name.endswith((".hc_fn", ".hc_base", ".hc_scale", ".attn_sink"))
    )


def test_build_contract_can_construct_training_optimizer(monkeypatch) -> None:
    monkeypatch.setattr(protocol, "initialize_ds4_vllm_batch_invariance", lambda: None)
    monkeypatch.setattr(
        protocol,
        "init_parallel",
        lambda _config: ParallelState(ep_size=1, ep_rank=0),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        SimpleNamespace(VllmConfig=lambda: object()),
    )
    impl = protocol.ImplConfig(
        parallel=ParallelConfig(ep=2),
        use_deepep=True,
        optimizer="adamw",
        optimizer_config=protocol.OptimizerConfig(lr=2e-4, weight_decay=0.1),
        selector=protocol.SelectorConfig((0,), ("attn", "moe")),
    )
    bundle = protocol.build_model(_tiny_config(), impl_cfg=impl)
    assert isinstance(bundle.optimizer, torch.optim.AdamW)
    assert bundle.finalize_grads is not None
    assert bundle.extras["optimizer_backend"] == "adamw"
    assert bundle.optimizer.param_groups[0]["lr"] == 2e-4


def test_build_contract_defers_fsdp2_until_after_weight_load(monkeypatch) -> None:
    monkeypatch.setattr(protocol, "initialize_ds4_vllm_batch_invariance", lambda: None)
    parallel_state = ParallelState(ep_size=1, ep_rank=0)
    monkeypatch.setattr(protocol, "init_parallel", lambda _config: parallel_state)
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        SimpleNamespace(VllmConfig=lambda: object()),
    )
    calls = {}

    import megatron.lite.primitive.optimizers.fsdp2 as fsdp2

    def fake_build(chunks, config, ps, **kwargs):
        calls.update(chunks=chunks, config=config, ps=ps, kwargs=kwargs)
        return "fsdp2-optimizer"

    monkeypatch.setattr(fsdp2, "build_fsdp2_training_optimizer", fake_build)
    optimizer_config = protocol.OptimizerConfig(lr=2e-4, offload_fraction=1.0)
    impl = protocol.ImplConfig(
        parallel=ParallelConfig(ep=2),
        use_deepep=True,
        optimizer="fsdp2",
        optimizer_config=optimizer_config,
        attention_backend_override="flash",
        selector=protocol.SelectorConfig((0,), ("attn", "moe")),
    )
    bundle = protocol.build_model(_tiny_config(), impl_cfg=impl)
    assert bundle.optimizer is None
    assert bundle.finalize_grads is None
    assert bundle.extras["optimizer_backend"] == "fsdp2"
    updates = bundle.extras["post_model_load_hook"]()
    assert updates == {"optimizer": "fsdp2-optimizer"}
    assert calls["config"] is optimizer_config
    assert calls["ps"] is parallel_state
    assert calls["kwargs"]["use_fp32_shards"] is True
    assert calls["kwargs"]["cast_forward_inputs"] is False


def test_verl_packed_batch_builds_per_layer_runtime_metadata(monkeypatch) -> None:
    calls = {}

    class AttentionBuilder:
        @classmethod
        def from_hf(cls, *_args, **_kwargs):
            return cls()

        def build_prefill_batch(self, token_counts):
            calls["token_counts"] = token_counts
            return "attention"

    class MoEBuilder:
        def __init__(self, *_args, **_kwargs):
            pass

        def build(self):
            return "moe"

    monkeypatch.setattr(protocol, "initialize_ds4_vllm_batch_invariance", lambda: None)
    monkeypatch.setattr(
        protocol,
        "init_parallel",
        lambda _config: ParallelState(ep_size=1, ep_rank=0),
    )
    monkeypatch.setattr(protocol, "DS4SparseAttentionMetadataBuilderAdapter", AttentionBuilder)
    monkeypatch.setattr(protocol, "DS4MoEKernelMetadataBuilderAdapter", MoEBuilder)
    monkeypatch.setattr(protocol, "ds4_vllm_forward_context", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        SimpleNamespace(VllmConfig=lambda: object()),
    )
    monkeypatch.setattr(
        protocol,
        "_forward_step",
        lambda _model, _batch, **kwargs: calls.update(kwargs) or {"loss": torch.tensor(0.0)},
    )
    impl = protocol.ImplConfig(
        parallel=ParallelConfig(ep=2),
        use_deepep=True,
        hf_path="/proxy",
        selector=protocol.SelectorConfig((0,), ("attn", "moe")),
    )
    bundle = protocol.build_model(_tiny_config(), impl_cfg=impl)
    batch = type(
        "Batch",
        (),
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "seq_lens": torch.tensor([1, 2]),
        },
    )()

    bundle.forward_step(bundle.chunks[0], batch)

    assert calls["token_counts"] == [1, 2]
    assert calls["attention_metadata"] == {0: "attention"}
    assert calls["moe_metadata"] == {0: "moe"}


def test_forward_step_reuses_caller_owned_runtime_assets(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(protocol, "initialize_ds4_vllm_batch_invariance", lambda: None)
    monkeypatch.setattr(
        protocol,
        "init_parallel",
        lambda _config: ParallelState(ep_size=1, ep_rank=0),
    )
    monkeypatch.setattr(
        protocol.DS4SparseAttentionMetadataBuilderAdapter,
        "from_hf",
        lambda *_args, **_kwargs: pytest.fail("rebuilt caller-owned attention assets"),
    )
    monkeypatch.setattr(
        protocol.DS4MoEKernelMetadataBuilderAdapter,
        "create_deepep_buffer",
        lambda *_args, **_kwargs: pytest.fail("created a second DeepEP buffer"),
    )
    monkeypatch.setattr(
        protocol,
        "ds4_vllm_forward_context",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        SimpleNamespace(VllmConfig=lambda: object()),
    )
    monkeypatch.setattr(
        protocol,
        "_forward_step",
        lambda _model, _batch, **kwargs: captured.update(kwargs)
        or {"loss": torch.tensor(0.0)},
    )
    impl = protocol.ImplConfig(
        parallel=ParallelConfig(ep=2),
        use_deepep=True,
        hf_path="/proxy",
        selector=protocol.SelectorConfig((0,), ("attn", "moe")),
    )
    bundle = protocol.build_model(_tiny_config(), impl_cfg=impl)
    attention_metadata = {0: object()}
    moe_metadata = {0: object()}
    batch = SimpleNamespace(
        input_ids=torch.tensor([1, 2, 3]),
        attention_metadata=attention_metadata,
        moe_metadata=moe_metadata,
    )

    bundle.forward_step(bundle.chunks[0], batch)

    assert captured["attention_metadata"] is attention_metadata
    assert captured["moe_metadata"] is moe_metadata


def test_forward_step_passes_training_batch_fields(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        protocol,
        "add_loss_context_kwargs",
        lambda kwargs: kwargs.update(calculate_entropy=True),
    )

    class Model:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {"loss": torch.tensor(1.0)}

    batch = type(
        "Batch",
        (),
        {
            "input_ids": torch.tensor([1]),
            "position_ids": torch.tensor([0]),
            "attention_metadata": object(),
            "moe_metadata": object(),
            "labels": torch.tensor([2]),
            "loss_mask": torch.tensor([1.0]),
            "temperature": 0.75,
        },
    )()
    protocol._forward_step(Model(), batch)
    assert captured["labels"] is batch.labels
    assert captured["loss_mask"] is batch.loss_mask
    assert captured["temperature"] == 0.75
    assert captured["calculate_entropy"] is True


def test_forward_step_rolls_packed_targets_without_crossing_sequences() -> None:
    captured = {}

    class Model:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {"log_probs": torch.zeros(5)}

    batch = type(
        "Batch",
        (),
        {
            "input_ids": torch.tensor([11, 12, 21, 22, 23]),
            "seq_lens": torch.tensor([2, 3]),
            "labels": torch.tensor([11, 12, 21, 22, 23]),
            "loss_mask": torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
        },
    )()
    protocol._forward_step(Model(), batch)
    torch.testing.assert_close(captured["labels"], torch.tensor([12, 0, 22, 23, 0]))
    torch.testing.assert_close(
        captured["loss_mask"], torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    )


def test_unpack_forward_output_restores_jagged_rows() -> None:
    batch = type("Batch", (), {"seq_lens": torch.tensor([2, 3])})()
    unpacked = protocol.unpack_forward_output(
        None,
        batch,
        torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0]),
    )
    assert unpacked.is_nested
    torch.testing.assert_close(unpacked[0], torch.tensor([-1.0, -2.0]))
    torch.testing.assert_close(unpacked[1], torch.tensor([-3.0, -4.0, -5.0]))


def test_missing_attention_metadata_fails_closed() -> None:
    model = DeepseekV4Model(
        _tiny_config(),
        selected_layer_ids=(0,),
        selected_module_names=("linear", "kv_flashmla", "o_proj"),
    )
    value = torch.randn(2, 8, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="explicit metadata"):
        model.layers["0"].self_attn(value, metadata=None)


class _FakeFlash:
    def __init__(self, calls):
        self.calls = calls

    def sparse(self, q, kv, indices, **kwargs):
        self.calls.append(("flash", q.shape, kv.shape, indices.dtype))
        kwargs["out"].copy_(q + 3)
        return [kwargs["out"], torch.empty(0), torch.empty(0)]

    def paged(self, *args, **kwargs):
        raise AssertionError("unexpected decode")


def test_attention_calls_every_adapter_and_rejects_backward() -> None:
    cfg = _tiny_config()
    cfg.num_hidden_layers = 1
    calls = []

    def linear(x, weight):
        calls.append(("linear", tuple(weight.shape), weight.dtype))
        width = weight.shape[0]
        return x[:, :1].expand(x.shape[0], width).contiguous() + 1

    def norm(q, kv, qw, kw, eps):
        calls.append(("norm", q.shape, kv.shape, qw.dtype, kw.dtype, eps))
        return q + 1, kv + 1

    def insert(q, kv, cache, slots, positions, cos, **kwargs):
        calls.append(("insert", q.shape, kv.shape, slots.dtype, positions.dtype))
        return q + 1

    def o_project(o, positions, cos_sin_cache, wo_a, wo_b, **kwargs):
        del cos_sin_cache, kwargs
        calls.append(("o_proj", o.shape, wo_a.dtype, wo_b.dtype))
        return o.flatten(1)[:, : cfg.hidden_size] + 5

    adapters = AttentionAdapters(
        fused_linear=linear,
        q_linear=linear,
        indexer_q_linear=linear,
        norm=norm,
        kv_insert=insert,
        flash=_FakeFlash(calls),
        o_project=o_project,
    )
    model = DeepseekV4Model(
        cfg,
        selected_layer_ids=(0,),
        selected_module_names=("linear", "kv_flashmla", "o_proj"),
        attention_adapters=adapters,
    )
    value = torch.randn(2, cfg.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    metadata = AttentionKernelMetadata(
        positions=torch.arange(2, dtype=torch.int64),
        slot_mapping=torch.arange(2, dtype=torch.int64),
        cos_sin_cache=torch.zeros(4, 4, dtype=torch.bfloat16),
        swa_cache=torch.zeros(1, 2 * cfg.head_dim, dtype=torch.uint8),
        block_size=2,
        flash_kind="prefill",
        indices=torch.zeros(2, 1, 1, dtype=torch.int32),
        topk_length=torch.ones(2, dtype=torch.int32),
        output=torch.empty(2, cfg.num_attention_heads, cfg.head_dim, dtype=torch.bfloat16),
        kv_workspace=torch.zeros(2, 1, cfg.head_dim, dtype=torch.bfloat16),
    )
    output = model.layers["0"].self_attn(value, metadata=metadata)
    assert not torch.equal(output, value)
    assert [call[0] for call in calls] == [
        "linear",
        "norm",
        "linear",
        "insert",
        "flash",
        "o_proj",
    ]
    assert output.grad_fn is not None


def test_hyperconnection_release_shapes_use_fp32_masters() -> None:
    cfg = _tiny_config()
    model = DeepseekV4Model(cfg)
    mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
    layer = model.layers["0"]
    assert layer.attn_hc.hc_fn.shape == (mix_hc, cfg.hc_mult * cfg.hidden_size)
    assert layer.attn_hc.hc_base.shape == (mix_hc,)
    assert layer.attn_hc.hc_scale.shape == (3,)
    assert model.hc_head.hc_fn.shape == (cfg.hc_mult, cfg.hc_mult * cfg.hidden_size)
    for name, value in model.state_dict().items():
        if name.endswith((".hc_fn", ".hc_base", ".hc_scale")):
            assert value.dtype == torch.float32, name


def test_four_layer_attention_and_router_release_structure() -> None:
    cfg = _tiny_4l_config()
    model = DeepseekV4Model(cfg)
    assert [model.layers[str(i)].self_attn.compress_ratio for i in range(4)] == [
        1,
        1,
        4,
        128,
    ]
    assert model.layers["0"].self_attn.compressor is None
    assert model.layers["1"].self_attn.compressor is None

    layer2 = model.layers["2"].self_attn
    assert layer2.compressor.fused_wkv_wgate.shape == (16, cfg.hidden_size)
    assert layer2.compressor.ape.shape == (4, 8)
    assert layer2.indexer.wq_b.shape == (
        cfg.index_n_heads * cfg.index_head_dim,
        cfg.q_lora_rank,
    )
    assert layer2.indexer.weights_proj.shape == (
        cfg.index_n_heads,
        cfg.hidden_size,
    )
    assert layer2.indexer.compressor.fused_wkv_wgate.shape == (
        16,
        cfg.hidden_size,
    )

    layer3 = model.layers["3"]
    assert layer3.self_attn.compressor.fused_wkv_wgate.shape == (
        2 * cfg.head_dim,
        cfg.hidden_size,
    )
    assert layer3.self_attn.indexer is None
    assert hasattr(layer3.mlp.gate, "expert_bias")
    assert not hasattr(layer3.mlp.gate, "tid2eid")
    floating = {
        name: value.dtype
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }
    assert set(floating.values()) == {torch.bfloat16, torch.float32}
    fp32_suffixes = (
        ".hc_fn",
        ".hc_base",
        ".hc_scale",
        ".attn_sink",
        ".compressor.ape",
        ".mlp.gate.expert_bias",
    )
    assert {
        name for name, dtype in floating.items() if dtype == torch.float32
    } == {name for name in floating if name.endswith(fp32_suffixes)}


def test_layer2_attention_calls_compressor_and_indexer_in_order() -> None:
    cfg = _tiny_4l_config()
    calls = []

    def linear(x, weight):
        calls.append("linear")
        return torch.ones(x.shape[0], weight.shape[0], dtype=torch.bfloat16)

    def bf16_linear(x, weight):
        calls.append("bf16_linear")
        return torch.ones(x.shape[0], weight.shape[0], dtype=torch.bfloat16)

    def indexer_q_linear(x, weight):
        calls.append("indexer_q_linear")
        return torch.ones(x.shape[0], weight.shape[0], dtype=torch.bfloat16)

    def fp32_linear(x, weight):
        calls.append("fp32_linear")
        return torch.ones(x.shape[0], weight.shape[0], dtype=torch.float32)

    def norm(q, kv, *_args):
        calls.append("norm")
        return q, kv

    def compressor_operation(**kwargs):
        calls.append(("compressor", kwargs["head_dim"]))

    def indexer_operation(**_kwargs):
        calls.append("indexer")
        return (
            torch.zeros(2, 1, 1, dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
        )

    def insert(q, *_args, **_kwargs):
        calls.append("insert")
        return q

    def o_project(o, *_args, **_kwargs):
        calls.append("o_proj")
        return o.flatten(1)

    class Flash:
        def sparse(self, q, _kv, _indices, **kwargs):
            calls.append("flash")
            kwargs["out"].copy_(q)
            return kwargs["out"], torch.zeros(
                q.shape[0], q.shape[1], dtype=torch.float32
            )

    adapters = AttentionAdapters(
        fused_linear=linear,
        q_linear=linear,
        indexer_q_linear=indexer_q_linear,
        bf16_linear=bf16_linear,
        fp32_linear=fp32_linear,
        norm=norm,
        kv_insert=insert,
        flash=Flash(),
        o_project=o_project,
    )
    model = DeepseekV4Model(
        cfg,
        selected_layer_ids=(0, 1, 2),
        selected_module_names=("linear", "kv_flashmla", "o_proj"),
        attention_adapters=adapters,
    )
    metadata = AttentionKernelMetadata(
        positions=torch.arange(2, dtype=torch.int64),
        slot_mapping=torch.arange(2, dtype=torch.int64),
        cos_sin_cache=torch.zeros(4, 4, dtype=torch.bfloat16),
        swa_cache=torch.zeros(1, 2 * cfg.head_dim, dtype=torch.uint8),
        block_size=2,
        flash_kind="prefill",
        indices=torch.full((2, 1, 1), -1, dtype=torch.int32),
        topk_length=torch.zeros(2, dtype=torch.int32),
        output=torch.empty(
            2, cfg.num_attention_heads, cfg.head_dim, dtype=torch.bfloat16
        ),
        kv_workspace=torch.zeros(2, 1, cfg.head_dim, dtype=torch.bfloat16),
        compressor_operation=compressor_operation,
        indexer_operation=indexer_operation,
    )
    output = model.layers["2"].self_attn(
        torch.zeros(2, cfg.hidden_size, dtype=torch.bfloat16),
        metadata=metadata,
    )
    assert output.shape == (2, cfg.hidden_size)
    assert calls == [
        "linear",
        "fp32_linear",
        "bf16_linear",
        "fp32_linear",
        "norm",
        "linear",
        "indexer_q_linear",
        ("compressor", cfg.head_dim),
        ("compressor", cfg.index_head_dim),
        "indexer",
        "insert",
        "flash",
        "o_proj",
    ]
    assert torch.all(metadata.indices == 0)
    assert torch.all(metadata.topk_length == 1)


def test_four_layer_selector_requires_release_config() -> None:
    cfg = _tiny_4l_config()
    impl = protocol.ImplConfig(
        parallel=ParallelConfig(ep=2),
        use_deepep=True,
        selector=protocol.SelectorConfig((0, 1, 2, 3), ("attn", "moe")),
    )
    protocol._validate_contract(cfg, impl)
    cfg.compress_ratios = [1, 1, 1, 1]
    with pytest.raises(ValueError, match="compress_ratios"):
        protocol._validate_contract(cfg, impl)


@pytest.mark.parametrize("num_hash_layers", (-1, 5))
def test_num_hash_layers_is_bounded_prefix_length(num_hash_layers: int) -> None:
    cfg = _tiny_4l_config()
    cfg.num_hash_layers = num_hash_layers
    with pytest.raises(ValueError, match="zero-based prefix length"):
        protocol._validate_contract(cfg, protocol.ImplConfig())


def test_layer0_audit_rejects_missing_stage() -> None:
    with pytest.raises(ValueError, match="missing"):
        protocol._validate_contract(
            _tiny_config(),
            protocol.ImplConfig(
                parallel=ParallelConfig(ep=2),
                use_deepep=True,
                selector=protocol.SelectorConfig((0,), ("linear",)),
            ),
        )


def test_hash_moe_calls_gate_route_and_training_dispatcher() -> None:
    cfg = _tiny_config()
    cfg.n_shared_experts = 0
    ps = ParallelState(ep_size=1, ep_rank=0)
    model = DeepseekV4Model(
        cfg,
        ps=ps,
        use_deepep=True,
        selected_layer_ids=(0,),
        selected_module_names=("router_moe", "deepep"),
    )
    moe = model.layers["0"].mlp
    assert moe.dispatcher.deepep_align_to_low_latency is True
    calls = []

    class Route(torch.nn.Module):
        def forward(self, logits, tokens, table, **kwargs):
            calls.append(("route", logits.dtype, tokens.dtype, table.dtype))
            shape = (logits.shape[0], cfg.num_experts_per_tok)
            return (
                torch.ones(shape, dtype=torch.float32),
                torch.zeros(shape, dtype=torch.int64),
            )

    class Dispatcher:
        _local_tpe_list = [2]

        def dispatch(self, hidden_states, weights, ids):
            calls.append(("dispatch", weights.dtype, ids.dtype))
            return hidden_states, torch.tensor([2]), None

        def wait_dispatch_event(self):
            calls.append(("wait",))

        def combine(self, hidden_states):
            calls.append(("combine",))
            return hidden_states

    class Experts(torch.nn.Module):
        def forward(self, hidden_states, *_args, **_kwargs):
            calls.append(("experts",))
            return hidden_states + 1

    moe.hash_route_adapter = Route()
    moe.shared_experts = None
    moe.dispatcher = Dispatcher()
    moe.experts = Experts()
    metadata = MoEKernelMetadata(
        gate_linear=lambda x: torch.ones(
            x.shape[0], cfg.n_routed_experts, dtype=torch.float32
        ),
    )
    value = torch.randn(2, cfg.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    output = moe(value, input_ids=torch.tensor([1, 2]), metadata=metadata)
    assert output.shape == value.shape
    assert not torch.equal(output, value)
    assert ("dispatch", torch.float32, torch.int64) in calls
    assert ("experts",) in calls
    assert ("combine",) in calls
    assert ("route", torch.float32, torch.int32, torch.int32) in calls
    output.sum().backward()
    assert value.grad is not None


def test_learned_moe_uses_fp32_logits_and_bias_without_token_ids() -> None:
    cfg = _tiny_config()
    cfg.num_hash_layers = 1
    cfg.n_shared_experts = 0
    moe = DeepseekV4Model(
        cfg,
        ps=ParallelState(ep_size=1, ep_rank=0),
        use_deepep=True,
        selected_layer_ids=(0, 1),
        selected_module_names=("router_moe", "deepep"),
    ).layers["1"].mlp
    calls = []

    class LearnedRoute(torch.nn.Module):
        def forward(self, logits, correction_bias, **kwargs):
            calls.append(
                (
                    "learned",
                    logits.dtype,
                    correction_bias.dtype,
                    kwargs["indices_dtype"],
                    kwargs["routed_scaling_factor"],
                )
            )
            shape = (logits.shape[0], cfg.num_experts_per_tok)
            return (
                torch.ones(shape, dtype=torch.float32),
                torch.zeros(shape, dtype=torch.int32),
            )

    class Experts(torch.nn.Module):
        def forward(self, hidden_states, *_args, **kwargs):
            return hidden_states + 1

    class Dispatcher:
        _local_tpe_list = [2]

        def dispatch(self, hidden_states, weights, ids):
            return hidden_states, torch.tensor([2]), None

        def wait_dispatch_event(self):
            pass

        def combine(self, hidden_states):
            return hidden_states

    moe.learned_route_adapter = LearnedRoute()
    moe.experts = Experts()
    moe.dispatcher = Dispatcher()
    metadata = MoEKernelMetadata(
        gate_linear=lambda x: torch.ones(
            x.shape[0], cfg.n_routed_experts, dtype=torch.bfloat16
        ),
    )
    value = torch.randn(2, cfg.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    output = moe(value, input_ids=None, metadata=metadata)

    assert not hasattr(moe.gate, "tid2eid")
    assert moe.gate.gate.weight.dtype == torch.bfloat16
    assert moe.gate.expert_bias.dtype == torch.float32
    assert (
        "learned",
        torch.float32,
        torch.float32,
        torch.int64,
        cfg.routed_scaling_factor,
    ) in calls
    output.sum().backward()
    assert value.grad is not None


def test_moe_missing_deepep_metadata_fails_closed() -> None:
    cfg = _tiny_config()
    moe = DeepseekV4Model(
        cfg,
        selected_layer_ids=(0,),
        selected_module_names=("router_moe", "deepep"),
    ).layers["0"].mlp
    with pytest.raises(NotImplementedError, match="GateLinear"):
        moe(
            torch.zeros(1, cfg.hidden_size, dtype=torch.bfloat16),
            input_ids=torch.zeros(1, dtype=torch.int32),
            metadata=None,
        )


def test_model_agnostic_boundary_rejects_backward() -> None:
    value = torch.randn(4, requires_grad=True)
    output = inference_only(value)
    assert torch.equal(output, value)
    with pytest.raises(NotImplementedError, match="inference-only"):
        output.sum().backward()


def test_protocol_does_not_install_training_features() -> None:
    source = inspect.getsource(protocol.build_model)
    for forbidden in ("apply_qat", "apply_recompute", "apply_offload", "register_training_hooks"):
        assert forbidden not in source


def test_protocol_owns_vllm_workspace_lifecycle() -> None:
    source = inspect.getsource(protocol.build_model)
    assert "init_workspace_manager" in source
    assert 'extras["close_hook"] = reset_workspace_manager' in source


@pytest.mark.gpus(1)
def test_production_kernel_fixture_contract() -> None:
    if not torch.cuda.is_available():
        pytest.xfail("CUDA is unavailable for the production vLLM kernel fixture")
    if importlib.util.find_spec("vllm") is None:
        pytest.xfail("the matching vLLM build is unavailable")
    pytest.xfail(
        "official FlashMLA scheduler/cache metadata fixture is not available in CPU CI"
    )
