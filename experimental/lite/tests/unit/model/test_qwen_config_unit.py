# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.registry import (
    resolve_model_type_from_hf,
    resolve_runtime_model_name,
)
from megatron.lite.runtime.contracts.config import ParallelConfig

pytestmark = pytest.mark.mlite

LITE_ROOT = Path(__file__).resolve().parents[3]


def _tiny_qwen3_hf_dict() -> dict:
    return {
        "model_type": "qwen3_moe",
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 64,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 8,
        "rope_parameters": {"rope_theta": 12345.0},
    }


def _tiny_qwen35_text_config() -> dict:
    return {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "num_hidden_layers": 2,
        "vocab_size": 64,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 4,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 2,
        "num_nextn_predict_layers": 1,
        "layer_types": ["linear_attention", "full_attention", "full_attention"],
        "rope_parameters": {"partial_rotary_factor": 1.0, "mrope_section": [1, 1, 0]},
    }


def test_registry_resolves_qwen_lite_model_names():
    assert resolve_model_type_from_hf({"model_type": "qwen3_moe"}) == "qwen3"
    assert resolve_model_type_from_hf({"model_type": "qwen3_5_moe"}) == "qwen3_5"
    assert resolve_runtime_model_name("qwen3", "lite") == "qwen3"
    assert resolve_runtime_model_name("qwen3_moe", "lite") == "qwen3_moe"
    assert resolve_runtime_model_name("qwen3_5", "lite") == "qwen3_5"


def test_qwen3_config_from_hf_dict_derives_head_dim_and_rope_theta():
    cfg = Qwen3MoEConfig._from_hf_dict(_tiny_qwen3_hf_dict())

    assert cfg.hidden_size == 16
    assert cfg.head_dim == 4
    assert cfg.layer_types == ["full_attention"]
    assert cfg.rope_theta == 12345.0


def test_qwen3_config_rejects_invalid_expert_topk():
    hf = _tiny_qwen3_hf_dict()
    hf["num_experts_per_tok"] = 3

    with pytest.raises(ValueError, match="num_experts_per_tok"):
        Qwen3MoEConfig._from_hf_dict(hf)


def test_qwen35_config_from_text_config_splits_mtp_layer_types():
    cfg = Qwen35Config._from_hf_dict(
        {"model_type": "qwen3_5_moe", "text_config": _tiny_qwen35_text_config()}
    )

    assert cfg.layer_types == ["linear_attention", "full_attention"]
    assert cfg.mtp_layer_types == ["full_attention"]
    assert cfg.rotary_dim == 4
    assert cfg.mrope_section == [1, 1, 0]


def test_qwen_lite_protocols_build_configs_from_hf_dicts(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()

    from megatron.lite.model.qwen3_5.lite import protocol as qwen35_protocol
    from megatron.lite.model.qwen3_moe.lite import protocol as qwen3_protocol

    qwen3_cfg = qwen3_protocol.build_model_config(_tiny_qwen3_hf_dict(), vocab_size=128)
    qwen35_cfg = qwen35_protocol.build_model_config(
        {"model_type": "qwen3_5_moe", "text_config": _tiny_qwen35_text_config()},
        vocab_size=128,
    )

    assert qwen3_cfg.vocab_size == 128
    assert qwen35_cfg.vocab_size == 128
    assert qwen35_cfg.layer_type_at(0) == "linear_attention"
    assert qwen35_cfg.layer_type_at(1) == "full_attention"


def test_qwen3_chunked_ep_is_one_boolean_fixed_profile(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import protocol

    cfg = protocol.ImplConfig(
        parallel=ParallelConfig(ep=8),
        use_deepep=True,
        enable_ep_chunk_overlap=True,
        ep_chunk_max_token_rows_per_rank=4096,
    )

    assert cfg.enable_ep_chunk_overlap is True
    assert cfg.ep_chunk_max_token_rows_per_rank == 4096
    assert not any("num_chunks" in name or "schedule" in name for name in vars(cfg))


def test_qwen3_layer_builds_lazy_selected_ops_from_real_token_capacity(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import model

    workspaces = []
    released = []

    class FakeWorkspace:
        def __init__(self, key):
            self.key = key
            self.materialize_devices = []
            self.prepare_scratch_devices = []

        def reset_tensors(self, *, stream=None):
            del stream

        def materialize(self, *, device=None):
            self.materialize_devices.append(device)

        def prepare_scratch(self, *, device=None):
            self.prepare_scratch_devices.append(device)

    class FakeOp:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.workspace = kwargs["workspace"]

    def get_workspace(key, _factory):
        workspace = FakeWorkspace(key)
        workspaces.append(workspace)
        return workspace

    monkeypatch.setattr(
        model, "TopKRouter", lambda *_args, **_kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(model, "Experts", lambda *_args, **_kwargs: torch.nn.Identity())
    monkeypatch.setattr(model, "get_ep_chunk_workspace", get_workspace)
    monkeypatch.setattr(
        model,
        "release_ep_chunk_workspace",
        lambda key, stream=None: released.append((key.op, stream)),
    )
    monkeypatch.setattr(model, "EPChunkForwardOp", FakeOp)
    monkeypatch.setattr(model, "EPChunkBackwardOp", FakeOp)
    monkeypatch.setattr(model, "EPChunkFusedForwardBackwardOp", FakeOp)
    monkeypatch.setattr(
        model.torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(
            AssertionError("Qwen construction must not bind a CUDA device")
        ),
    )

    config = SimpleNamespace(
        num_experts=128,
        hidden_size=64,
        num_experts_per_tok=8,
        max_position_embeddings=17,
    )
    ps = SimpleNamespace(ep_size=8, tp_ep_group=object())
    layer = model.MoELayer(
        config,
        ps,
        use_deepep=True,
        enable_ep_chunk_overlap=True,
        ep_chunk_max_token_rows_per_rank=33,
    )

    assert {workspace.key.op for workspace in workspaces} == {"forward", "backward"}
    assert all(
        workspace.key.shape_profile.max_input_rows == 33
        and workspace.key.shape_profile.max_recv_rows == 17 * 8
        and workspace.key.device_index is None
        for workspace in workspaces
    )
    assert all(workspace.materialize_devices == [] for workspace in workspaces)
    by_op = {workspace.key.op: workspace for workspace in workspaces}
    layer.materialize_ep_chunk_workspaces(device=model.torch.device("cuda", 3))
    assert by_op["forward"].materialize_devices == [model.torch.device("cuda", 3)]
    assert by_op["backward"].materialize_devices == []

    layer.materialize_ep_chunk_workspaces(
        phase="backward", device=model.torch.device("cuda", 3)
    )
    assert by_op["backward"].materialize_devices == []
    assert by_op["backward"].prepare_scratch_devices == [model.torch.device("cuda", 3)]

    stream = object()
    layer.release_ep_chunk_workspaces(phase="forward", stream=stream)
    layer.release_ep_chunk_workspaces(phase="backward", stream=stream)
    assert released == [("forward", stream), ("backward", stream)]
    with pytest.raises(ValueError, match="expected 'forward' or 'backward'"):
        layer.materialize_ep_chunk_workspaces(phase="unused")


@pytest.mark.parametrize(
    ("mtp_enable", "requested_chunk_count", "expected_chunk_count", "expected_moe_layers"),
    [(False, None, 2, 1), (True, 3, 3, 3)],
)
def test_qwen3_model_build_propagates_logical_chunk_count_to_decoder_and_mtp(
    mtp_enable,
    requested_chunk_count,
    expected_chunk_count,
    expected_moe_layers,
    monkeypatch,
    transformer_engine_import_stub,
):
    """Exercise the real Qwen composition, including MTP's nested TransformerLayer."""
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import model
    from megatron.lite.primitive.parallel import ParallelState

    class NoOpModule(torch.nn.Identity):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    class FakeWorkspace:
        def __init__(self, key):
            self.key = key

    class FakeOp:
        def __init__(self, **kwargs):
            self.workspace = kwargs["workspace"]

    for name in (
        "GQAttention",
        "TopKRouter",
        "Experts",
        "VocabParallelEmbedding",
        "VocabParallelOutput",
        "VanillaColumnParallelLinear",
    ):
        monkeypatch.setattr(model, name, NoOpModule)
    monkeypatch.setattr(model.te, "RMSNorm", NoOpModule)
    monkeypatch.setattr(
        model, "get_ep_chunk_workspace", lambda key, _factory: FakeWorkspace(key)
    )
    monkeypatch.setattr(model, "EPChunkForwardOp", FakeOp)
    monkeypatch.setattr(model, "EPChunkBackwardOp", FakeOp)
    monkeypatch.setattr(model, "EPChunkFusedForwardBackwardOp", FakeOp)

    hf = _tiny_qwen3_hf_dict()
    if mtp_enable:
        hf["num_nextn_predict_layers"] = 2
    config = Qwen3MoEConfig._from_hf_dict(hf)
    model_kwargs = dict(
        use_deepep=True,
        mtp_enable=mtp_enable,
        enable_ep_chunk_overlap=True,
        ep_chunk_max_token_rows_per_rank=8,
    )
    if requested_chunk_count is not None:
        model_kwargs["ep_chunk_count"] = requested_chunk_count
    built = model.Qwen3MoEModel(
        config,
        ParallelState(ep_size=2, tp_ep_group=object()),
        **model_kwargs,
    )

    moe_layers = [layer.moe for layer in built.layers]
    if mtp_enable:
        assert built.mtp is not None
        moe_layers.extend(layer.transformer_layer.moe for layer in built.mtp.layers)
    assert len(moe_layers) == expected_moe_layers
    assert {
        moe.ep_chunk_forward.workspace.key.shape_profile.chunk_count for moe in moe_layers
    } == {expected_chunk_count}
    assert {
        moe.ep_chunk_backward.workspace.key.shape_profile.chunk_count for moe in moe_layers
    } == {expected_chunk_count}


@pytest.mark.parametrize(
    "full_recompute,expected_ops",
    [
        (False, {"forward", "backward"}),
        (True, {"forward", "fused_forward_backward"}),
    ],
)
def test_qwen3_builds_only_two_cross_layer_workspaces_for_48_layers(
    full_recompute,
    expected_ops,
    monkeypatch,
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import model

    registry = {}
    dispatcher_count = 0

    class FakeWorkspace:
        def __init__(self, key, factory):
            self.key = key
            self.factory = factory
            self.dispatchers = []

        def materialize(self, *, device=None):
            if not self.dispatchers:
                self.dispatchers = [self.factory(0), self.factory(1)]

        def prepare_scratch(self, *, device=None):
            del device

    class FakeOp:
        def __init__(self, **kwargs):
            self.workspace = kwargs["workspace"]

    def fake_dispatcher(*_args, **_kwargs):
        nonlocal dispatcher_count
        dispatcher_count += 1
        return object()

    def get_workspace(key, factory):
        if key not in registry:
            registry[key] = FakeWorkspace(key, factory)
        return registry[key]

    monkeypatch.setattr(model, "TopKRouter", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(model, "Experts", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(model, "TokenDispatcher", fake_dispatcher)
    monkeypatch.setattr(model, "get_ep_chunk_workspace", get_workspace)
    monkeypatch.setattr(model, "EPChunkForwardOp", FakeOp)
    monkeypatch.setattr(model, "EPChunkBackwardOp", FakeOp)
    monkeypatch.setattr(model, "EPChunkFusedForwardBackwardOp", FakeOp)
    monkeypatch.setattr(model.torch.cuda, "current_device", lambda: 0)
    config = SimpleNamespace(
        num_experts=128,
        hidden_size=64,
        num_experts_per_tok=8,
    )
    ps = SimpleNamespace(ep_size=8, tp_ep_group=object())

    layers = [
        model.MoELayer(
            config,
            ps,
            use_deepep=True,
            enable_ep_chunk_overlap=True,
            ep_chunk_max_token_rows_per_rank=16384,
            ep_chunk_full_recompute=full_recompute,
        )
        for _ in range(48)
    ]

    assert {key.op for key in registry} == expected_ops
    assert len(registry) == 2
    assert dispatcher_count == 0
    layers[0].materialize_ep_chunk_workspaces(device="cpu")
    assert dispatcher_count == 2
    assert (
        registry[next(key for key in registry if key.op != "forward")].dispatchers == []
    )
    layers[0].materialize_ep_chunk_workspaces(phase="backward", device="cpu")
    assert dispatcher_count == (4 if full_recompute else 2)
    assert len({id(layer.ep_chunk_forward.workspace) for layer in layers}) == 1
    companion = "ep_chunk_fused" if full_recompute else "ep_chunk_backward"
    assert len({id(getattr(layer, companion).workspace) for layer in layers}) == 1


def test_qwen3_chunked_ep_requires_explicit_per_rank_token_capacity(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import protocol

    model_cfg = Qwen3MoEConfig._from_hf_dict(_tiny_qwen3_hf_dict())
    with pytest.raises(ValueError, match="ep_chunk_max_token_rows_per_rank"):
        protocol.build_model(
            model_cfg,
            impl_cfg=protocol.ImplConfig(
                parallel=ParallelConfig(ep=8),
                use_deepep=True,
                enable_ep_chunk_overlap=True,
            ),
        )


@pytest.mark.parametrize(
    "topk,max_rows,match",
    [
        (3, 8, "top-k must not exceed EP size"),
        (2, 1, "ep_chunk_max_token_rows_per_rank >= 2"),
    ],
)
def test_qwen3_moe_layer_direct_construction_reuses_chunk_policy_validation(
    topk, max_rows, match, monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import model

    monkeypatch.setattr(
        model,
        "TopKRouter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("policy validation must run before module construction")
        ),
    )
    config = SimpleNamespace(
        num_experts=8,
        hidden_size=4,
        num_experts_per_tok=topk,
    )
    ps = SimpleNamespace(ep_size=2, tp_ep_group=object())

    with pytest.raises(ValueError, match=match):
        model.MoELayer(
            config,
            ps,
            use_deepep=True,
            enable_ep_chunk_overlap=True,
            ep_chunk_max_token_rows_per_rank=max_rows,
        )


@pytest.mark.parametrize(
    "requested",
    [False, True],
)
def test_qwen3_protocol_exposes_explicit_chunk_full_recompute_policy(
    requested, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import protocol

    impl = protocol.ImplConfig(ep_chunk_full_recompute=requested)

    assert impl.ep_chunk_full_recompute is requested
    assert "_ep_chunk_full_recompute_requested" not in inspect.getsource(
        protocol.build_model
    )


@pytest.mark.parametrize(
    "overlap,full_recompute,recompute,error",
    [
        (False, True, [], "requires enable_ep_chunk_overlap=True"),
        (True, False, ["moe"], "conflicts with outer MoE recompute"),
        (True, False, ["full"], "conflicts with outer MoE recompute"),
    ],
)
def test_qwen3_protocol_rejects_conflicting_chunk_recompute_composition(
    overlap, full_recompute, recompute, error, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import protocol

    impl = protocol.ImplConfig(
        enable_ep_chunk_overlap=overlap,
        ep_chunk_full_recompute=full_recompute,
        recompute=recompute,
    )

    with pytest.raises(ValueError, match=error):
        protocol.validate_qwen3_ep_chunk_recompute_composition(
            enable_ep_chunk_overlap=impl.enable_ep_chunk_overlap,
            ep_chunk_full_recompute=impl.ep_chunk_full_recompute,
            recompute_modules=impl.recompute,
        )


@pytest.mark.parametrize(
    "overlap,full_recompute,recompute",
    [
        (False, False, ["moe"]),
        (True, False, ["attn"]),
        (True, True, ["moe"]),
        (True, True, ["full"]),
    ],
)
def test_qwen3_protocol_accepts_unambiguous_chunk_recompute_composition(
    overlap, full_recompute, recompute, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import protocol

    protocol.validate_qwen3_ep_chunk_recompute_composition(
        enable_ep_chunk_overlap=overlap,
        ep_chunk_full_recompute=full_recompute,
        recompute_modules=recompute,
    )


@pytest.mark.parametrize(
    "overlap,full_recompute,recompute,error",
    [
        (False, True, [], "requires enable_ep_chunk_overlap=True"),
        (True, False, ["moe"], "conflicts with outer MoE recompute"),
        (True, False, ["full"], "conflicts with outer MoE recompute"),
    ],
)
def test_qwen3_direct_model_construction_rejects_ambiguous_recompute(
    overlap,
    full_recompute,
    recompute,
    error,
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel

    with pytest.raises(ValueError, match=error):
        Qwen3MoEModel(
            SimpleNamespace(),
            SimpleNamespace(),
            enable_ep_chunk_overlap=overlap,
            ep_chunk_full_recompute=full_recompute,
            recompute_modules=recompute,
        )


@pytest.mark.parametrize(
    "modules,full_recompute,expected",
    [
        (["moe_act"], False, True),
        (["moe_act"], True, False),
        (["moe"], False, False),
    ],
)
def test_qwen3_compose_layer_owns_moe_activation_recompute_selection(
    modules, full_recompute, expected, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import model

    assert (
        model._qwen3_moe_act_recompute_requested(
            modules, ep_chunk_full_recompute=full_recompute
        )
        is expected
    )


def test_qwen3_full_chunked_recompute_owns_the_entire_layer_checkpoint(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import protocol

    requested = ["core_attn", "moe", "mlp"]
    assert (
        protocol._qwen3_recompute_modules_for_ep_chunk_overlap(requested, enabled=False)
        == requested
    )
    assert (
        protocol._qwen3_recompute_modules_for_ep_chunk_overlap(requested, enabled=True)
        == []
    )
    assert (
        protocol._qwen3_recompute_modules_for_ep_chunk_overlap(["full"], enabled=True)
        == []
    )


@pytest.mark.parametrize(
    "full_recompute,expected_counts",
    [
        (False, {"forward": 1, "backward": 1, "fused": 0}),
    ],
)
def test_qwen3_training_mode_matches_no_recompute_and_full_recompute_parity(
    full_recompute,
    expected_counts,
    monkeypatch,
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import model

    calls = {"forward": 0, "backward": 0, "fused": 0}

    class FakeWorkspace:
        def __init__(self, key):
            self.key = key

        def reset_tensors(self, *, stream=None):
            del stream

    class FakeForward:
        def __init__(self, *, backward_op=None, **_kwargs):
            self.backward_op = backward_op

        def __call__(self, x):
            calls["forward"] += 1

            class SavedForward(torch.autograd.Function):
                @staticmethod
                def forward(ctx, value, backward_op):
                    ctx.backward_op = backward_op
                    return value * 2

                @staticmethod
                def backward(ctx, grad_output):
                    grad_x, _router, _experts = ctx.backward_op.backward(
                        object(), grad_output
                    )
                    return grad_x, None

            return SavedForward.apply(x, self.backward_op)

    class FakeBackward:
        def __init__(self, **_kwargs):
            pass

        def backward(self, _context, grad_output):
            calls["backward"] += 1
            return grad_output * 2, [], []

    class FakeFused:
        def __init__(self, **kwargs):
            self.workspace = kwargs["workspace"]

        def forward_backward(self, _x_saved, grad_output, _routing_input=None):
            calls["fused"] += 1
            return grad_output * 2, [], []

    monkeypatch.setattr(
        model, "TopKRouter", lambda *_args, **_kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(model, "Experts", lambda *_args, **_kwargs: torch.nn.Identity())
    monkeypatch.setattr(
        model,
        "get_ep_chunk_workspace",
        lambda key, _factory: FakeWorkspace(key),
    )
    monkeypatch.setattr(model, "EPChunkForwardOp", FakeForward)
    monkeypatch.setattr(model, "EPChunkBackwardOp", FakeBackward)
    monkeypatch.setattr(model, "EPChunkFusedForwardBackwardOp", FakeFused)
    monkeypatch.setattr(model.torch.cuda, "current_device", lambda: 0)
    config = SimpleNamespace(
        num_experts=8,
        hidden_size=4,
        num_experts_per_tok=2,
    )
    ps = SimpleNamespace(ep_size=2, tp_ep_group=object())
    layer = model.MoELayer(
        config,
        ps,
        use_deepep=True,
        enable_ep_chunk_overlap=True,
        ep_chunk_max_token_rows_per_rank=8,
        ep_chunk_full_recompute=full_recompute,
    )
    value = torch.randn(2, 4, requires_grad=True)
    expected = value * 2

    actual = layer(value)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 2))
    assert calls == expected_counts


def test_qwen3_full_layer_recompute_runs_no_grad_once_and_recomputes_moe_once(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite.model import (
        _Qwen3TransformerLayerFullRecomputeFunction,
    )

    calls = {"forward": 0, "fused": 0}

    class Scale(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

        def forward(self, value, **_kwargs):
            return value * self.scale

    class FakeForward:
        def __call__(self, x):
            assert torch.is_grad_enabled() is False
            calls["forward"] += 1
            return x * 2

    class FakeFused:
        def forward_backward(self, _x_saved, grad_output):
            calls["fused"] += 1
            return grad_output * 5, [torch.tensor(7.0)], [torch.tensor(8.0)]

    class FakeLayer:
        def __init__(self):
            self.attn = Scale(2)
            self.mlp_norm = Scale(3)
            self.moe = SimpleNamespace(
                router=Scale(1),
                experts=Scale(1),
                ep_chunk_forward=FakeForward(),
                ep_chunk_fused=FakeFused(),
            )

        def _ep_chunk_full_recompute_forward(
            self, x, *, position_ids, packed_seq_params
        ):
            del position_ids, packed_seq_params
            residual = x
            x = residual + self.attn(x)
            residual = x
            h = self.mlp_norm(x)
            return residual + self.moe.ep_chunk_forward(h)

    layer = FakeLayer()
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    params = (
        *layer.attn.parameters(),
        *layer.mlp_norm.parameters(),
        *layer.moe.router.parameters(),
        *layer.moe.experts.parameters(),
    )
    actual = _Qwen3TransformerLayerFullRecomputeFunction.apply(
        value, None, None, layer, *params
    )
    torch.testing.assert_close(actual, value.detach() * 21)
    actual.sum().backward()

    torch.testing.assert_close(value.grad, torch.full_like(value, 48))
    assert calls == {"forward": 1, "fused": 1}
    torch.testing.assert_close(layer.attn.scale.grad, torch.tensor(160.0))
    torch.testing.assert_close(layer.mlp_norm.scale.grad, torch.tensor(150.0))
    torch.testing.assert_close(layer.moe.router.scale.grad, torch.tensor(7.0))
    torch.testing.assert_close(layer.moe.experts.scale.grad, torch.tensor(8.0))


def test_qwen3_full_recompute_custom_function_forward_is_framework_no_grad(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite.model import (
        _Qwen3TransformerLayerFullRecomputeFunction,
    )

    class Forward:
        def __call__(self, value):
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "custom autograd Function.forward unexpectedly enabled grad"
                )
            return value * 2

    layer = SimpleNamespace(
        _ep_chunk_full_recompute_forward=lambda value, **_kwargs: Forward()(value)
    )
    value = torch.randn(2, 4, requires_grad=True)

    output = _Qwen3TransformerLayerFullRecomputeFunction.apply(value, None, None, layer)

    assert output.requires_grad
    assert output.grad_fn is not None


def test_qwen3_full_layer_recompute_rejects_differentiable_position_ids(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite.model import (
        _Qwen3TransformerLayerFullRecomputeFunction,
    )

    value = torch.ones(1, 1, requires_grad=True)
    position_ids = torch.ones(1, 1, requires_grad=True)
    with pytest.raises(RuntimeError, match="position_ids must not require gradients"):
        _Qwen3TransformerLayerFullRecomputeFunction.apply(
            value, position_ids, None, SimpleNamespace()
        )


def test_qwen3_full_layer_recompute_restores_rng_for_dropout_replay(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite.model import (
        _Qwen3TransformerLayerFullRecomputeFunction,
    )

    samples = []

    class RandomAttention(torch.nn.Module):
        def forward(self, value, **_kwargs):
            sample = torch.rand_like(value)
            samples.append(sample)
            return value + sample

    class IdentityScale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, value):
            return value * self.scale

    class Forward:
        def __call__(self, value):
            return value

    class Fused:
        def forward_backward(self, _value, grad_output):
            return grad_output, [], []

    class Layer:
        def __init__(self):
            self.attn = RandomAttention()
            self.mlp_norm = IdentityScale()
            self.moe = SimpleNamespace(
                router=torch.nn.Identity(),
                experts=torch.nn.Identity(),
                ep_chunk_forward=Forward(),
                ep_chunk_fused=Fused(),
            )

        def _ep_chunk_full_recompute_forward(self, value, **_kwargs):
            residual = value + self.attn(value)
            return residual + self.moe.ep_chunk_forward(self.mlp_norm(residual))

    layer = Layer()
    value = torch.ones(2, 2, requires_grad=True)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda *_args, **_kwargs: pytest.fail(
            "CPU recompute must not initialize CUDA RNG"
        ),
    )
    torch.manual_seed(20260810)
    expected_sample = torch.rand_like(value)
    expected_after = torch.get_rng_state()
    torch.manual_seed(20260810)

    output = _Qwen3TransformerLayerFullRecomputeFunction.apply(
        value, None, None, layer, *layer.mlp_norm.parameters()
    )
    output.sum().backward()

    assert len(samples) == 2
    torch.testing.assert_close(samples[0], expected_sample)
    torch.testing.assert_close(samples[1], expected_sample)
    assert torch.equal(torch.get_rng_state(), expected_after)


def test_qwen3_chunked_ep_fails_loud_without_deepep_or_ep(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import protocol

    model_cfg = Qwen3MoEConfig._from_hf_dict(_tiny_qwen3_hf_dict())
    with pytest.raises(ValueError, match="DeepEP"):
        protocol.build_model(
            model_cfg,
            impl_cfg=protocol.ImplConfig(
                parallel=ParallelConfig(ep=8),
                use_deepep=False,
                enable_ep_chunk_overlap=True,
            ),
        )
    with pytest.raises(ValueError, match="EP > 1"):
        protocol.build_model(
            model_cfg,
            impl_cfg=protocol.ImplConfig(
                parallel=ParallelConfig(ep=1),
                use_deepep=True,
                enable_ep_chunk_overlap=True,
            ),
        )


def test_qwen_lite_protocols_reexport_checkpoint_hook_names():
    protocol_paths = [
        LITE_ROOT / "megatron/lite/model/qwen3_moe/lite/protocol.py",
        LITE_ROOT / "megatron/lite/model/qwen3_5/lite/protocol.py",
    ]

    for path in protocol_paths:
        tree = ast.parse(path.read_text())
        exported = _string_list_assignment(tree, "__all__")
        checkpoint_imports = _checkpoint_import_names(tree)

        assert "EXPERT_CLASSIFIER" in exported
        assert "PLACEMENT_FN" in exported
        assert "EXPERT_CLASSIFIER" in checkpoint_imports
        assert "PLACEMENT_FN" in checkpoint_imports


def _string_list_assignment(tree: ast.Module, name: str) -> set[str]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            return set()
        return {
            item.value for item in node.value.elts if isinstance(item, ast.Constant)
        }
    return set()


def _checkpoint_import_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None or not node.module.endswith(".lite.checkpoint"):
            continue
        names.update(alias.name for alias in node.names)
    return names
