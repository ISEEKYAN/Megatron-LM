# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import sys
import types
from dataclasses import field as dataclass_field
from dataclasses import fields, make_dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.optimizers.megatron_wrap import (
    _build_pg_collection,
    build_dist_opt_optimizer_config,
    build_dist_opt_stack,
    validate_dist_opt_config,
)
from megatron.lite.primitive.optimizers.muon_routing import tag_muon_parameter_metadata
from megatron.lite.primitive.parallel.state import init_parallel
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts.config import OptimizerConfig


class _SyntheticEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.embedding.weight.is_embedding_or_output_parameter = True


class _OutputColumn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 8, bias=False)


class _SyntheticOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.col = _OutputColumn()
        self.col.linear.weight.is_embedding_or_output_parameter = True


class _QKVProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 12, bias=False)


class _SyntheticAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads_local = 2
        self.num_kv_heads_local = 1
        self.head_dim = 2
        self._output_gate = True
        self.qkv = _QKVProjection()
        self.qkv.linear.weight.is_qkv = True
        self.qkv.linear.weight.qkv_split_shapes = [4, 4, 2, 2]


class _SyntheticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = _SyntheticEmbedding()
        self.attn = _SyntheticAttention()
        self.experts = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.dense = nn.Linear(4, 4, bias=False)
        self.head = _SyntheticOutput()


def _install_fake_core_optimizer_config(
    monkeypatch, *, name: str, excluded_fields: set[str] | None = None
):
    field_names = {
        "optimizer",
        "lr",
        "min_lr",
        "weight_decay",
        "clip_grad",
        "use_distributed_optimizer",
        "use_layer_wise_distributed_optimizer",
        "bf16",
        "params_dtype",
        *(item.name for item in fields(OptimizerConfig)),
    }
    field_names.difference_update(excluded_fields or set())
    fake_core_config = make_dataclass(
        name,
        [
            (field_name, object, dataclass_field(default=None))
            for field_name in sorted(field_names)
        ],
    )
    fake_module = types.ModuleType("megatron.core.optimizer.optimizer_config")
    fake_module.OptimizerConfig = fake_core_config
    monkeypatch.setitem(
        sys.modules, "megatron.core.optimizer.optimizer_config", fake_module
    )
    return fake_core_config


def test_central_routing_preserves_module_owned_optimizer_metadata() -> None:
    model = _SyntheticModel()

    tag_muon_parameter_metadata(
        [model], is_expert_param=lambda name: name.startswith("experts.")
    )

    embedding = model.embed.embedding.weight
    output = model.head.col.linear.weight
    qkv = model.attn.qkv.linear.weight

    assert embedding.is_embedding_or_output_parameter is True
    assert output.is_embedding_or_output_parameter is True
    assert embedding.is_managed_by_layer_wise_optimizer is False
    assert output.is_managed_by_layer_wise_optimizer is False
    assert qkv.is_qkv is True
    assert qkv.qkv_split_shapes == [4, 4, 2, 2]
    assert qkv.is_managed_by_layer_wise_optimizer is True


def test_central_routing_preserves_expert_and_tensor_parallel_metadata() -> None:
    model = _SyntheticModel()
    expert = model.experts[0].weight
    dense = model.dense.weight
    expert.tensor_model_parallel = True
    expert.partition_dim = 0
    expert.partition_stride = 2
    expert.allreduce = False
    dense.tensor_model_parallel = False
    dense.sequence_parallel = True

    tag_muon_parameter_metadata(
        [model], is_expert_param=lambda name: name.startswith("experts.")
    )

    assert expert.expert_tp is True
    assert expert.tensor_model_parallel is True
    assert expert.partition_dim == 0
    assert expert.partition_stride == 2
    assert expert.allreduce is False
    assert dense.tensor_model_parallel is False
    assert dense.sequence_parallel is True


def test_routing_does_not_infer_qkv_metadata_from_module_internals() -> None:
    model = _SyntheticModel()
    model.attn.qkv.linear.weight = nn.Parameter(torch.empty(15, 4))

    tag_muon_parameter_metadata([model], is_expert_param=lambda _name: False)

    assert not getattr(model.attn.qkv.linear.weight, "is_qkv", False)


def test_compact_dist_opt_lowering_follows_upstream_order(monkeypatch) -> None:
    model = _SyntheticModel()
    events: list[str] = []
    owner_group = SimpleNamespace(size=lambda: 2)
    expert_owner_group = SimpleNamespace(size=lambda: 1)
    pg_collection = SimpleNamespace(dp_cp=owner_group, expt_dp=expert_owner_group)

    import megatron.lite.primitive.optimizers.megatron_wrap as wrap_module
    import megatron.lite.primitive.optimizers.muon_routing as routing_module

    original_tag_metadata = routing_module.tag_muon_parameter_metadata

    def tag_metadata(model_chunks, *, is_expert_param):
        events.append("metadata")
        original_tag_metadata(model_chunks, is_expert_param=is_expert_param)

    monkeypatch.setattr(routing_module, "tag_muon_parameter_metadata", tag_metadata)
    monkeypatch.setattr(
        wrap_module,
        "_build_transformer_config",
        lambda _model_cfg, _engine_cfg: SimpleNamespace(),
    )
    monkeypatch.setattr(
        wrap_module,
        "_build_pg_collection",
        lambda _ps, _engine_cfg: pg_collection,
    )
    monkeypatch.setattr(
        wrap_module,
        "_ensure_dist_opt_mpu_parallel_state",
        lambda _engine_cfg: events.append("mpu"),
    )

    class FakeDistributedDataParallelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.bucket_size = None
            self.num_distributed_optimizer_instances = 1

    class FakeDistributedDataParallel(nn.Module):
        def __init__(
            self,
            _config,
            ddp_config,
            module,
            *,
            disable_bucketing,
            pg_collection,
            full_param_layout,
        ):
            super().__init__()
            events.append("ddp")
            self.module = module
            self.ddp_config = ddp_config
            self.disable_bucketing = disable_bucketing
            self.pg_collection = pg_collection
            self.full_param_layout = full_param_layout

    distributed_module = types.ModuleType("megatron.core.distributed")
    distributed_module.DistributedDataParallel = FakeDistributedDataParallel
    distributed_module.DistributedDataParallelConfig = FakeDistributedDataParallelConfig
    monkeypatch.setitem(sys.modules, "megatron.core.distributed", distributed_module)

    finalize_module = types.ModuleType("megatron.core.distributed.finalize_model_grads")
    finalize_module.finalize_model_grads = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "megatron.core.distributed.finalize_model_grads", finalize_module
    )

    def tag_params_for_buffer_routing(model_chunks) -> None:
        events.append("buffer_route")
        assert model_chunks == [model]
        assert model.embed.embedding.weight.is_embedding_or_output_parameter is True
        assert model.embed.embedding.weight.is_managed_by_layer_wise_optimizer is False
        for param in model.parameters():
            param.is_managed_by_layer_wise_optimizer = param.dim() == 2 and not getattr(
                param, "is_embedding_or_output_parameter", False
            )

    class FakeLayerWiseDistributedOptimizer:
        @staticmethod
        def compute_full_param_layout(
            params,
            bucket_size,
            data_parallel_world_size,
            ddp_config,
            *,
            expert_data_parallel_world_size,
        ):
            events.append("layout")
            assert list(params) == list(model.parameters())
            assert bucket_size is None
            assert data_parallel_world_size == 2
            assert expert_data_parallel_world_size == 1
            assert ddp_config.use_layer_wise_param_layout is False
            return "compact-layout"

    layer_wise_module = types.ModuleType("megatron.core.optimizer.layer_wise_optimizer")
    layer_wise_module.LayerWiseDistributedOptimizer = FakeLayerWiseDistributedOptimizer
    layer_wise_module.tag_params_for_buffer_routing = tag_params_for_buffer_routing
    monkeypatch.setitem(
        sys.modules, "megatron.core.optimizer.layer_wise_optimizer", layer_wise_module
    )

    _install_fake_core_optimizer_config(
        monkeypatch, name="FakeCoreOptimizerConfigForLowering"
    )

    facade = SimpleNamespace(chained_optimizers=["muon", "adam"])

    def get_megatron_optimizer(*, config, model_chunks, **kwargs):
        events.append("optimizer")
        assert config.use_distributed_optimizer is False
        assert config.use_layer_wise_distributed_optimizer is True
        assert config.muon_num_ns_steps == 7
        assert isinstance(model_chunks[0], FakeDistributedDataParallel)
        assert kwargs == {
            "use_gloo_process_groups": False,
            "pg_collection": pg_collection,
        }
        return facade

    optimizer_module = types.ModuleType("megatron.core.optimizer")
    optimizer_module.get_megatron_optimizer = get_megatron_optimizer
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer", optimizer_module)

    class FakeModelType:
        encoder_or_decoder = "encoder_or_decoder"

    enum_module = types.ModuleType("megatron.core.transformer.enums")
    enum_module.ModelType = FakeModelType
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.enums", enum_module)

    parallel = SimpleNamespace(vpp=1, pp=1, tp=1, ep=1, etp=1, cp=1)
    optimizer_config = OptimizerConfig(optimizer="muon")
    optimizer_config.override_optimizer_config = {"muon_num_ns_steps": 7}
    engine_config = SimpleNamespace(
        parallel=parallel,
        optimizer=optimizer_config,
        deterministic=False,
    )

    wrapped, optimizer = build_dist_opt_stack(
        [model],
        model_cfg=SimpleNamespace(),
        engine_cfg=engine_config,
        ps=SimpleNamespace(),
        is_expert=lambda name: name.startswith("experts."),
    )

    assert events == ["metadata", "mpu", "buffer_route", "layout", "ddp", "optimizer"]
    assert wrapped[0].full_param_layout == "compact-layout"
    assert wrapped[0].pg_collection is pg_collection
    assert optimizer is facade
    assert optimizer._dist_opt_pg_collection is pg_collection


@pytest.mark.parametrize(
    ("optimizer_overrides", "message"),
    [
        ({"use_layer_wise_param_layout": True}, "padded"),
        ({"overlap_grad_reduce": True}, "overlap_grad_reduce"),
        ({"overlap_param_gather": True}, "overlap_param_gather"),
        ({"overlap_param_gather_with_optimizer_step": True}, "optimizer step"),
        ({"optimizer_cpu_offload": True}, "offload"),
        ({"optimizer_offload_fraction": 0.5}, "offload"),
        ({"offload_optimizer_states": True}, "offload"),
        ({"fp8_param_gather": True}, "fp8_param_gather"),
        ({"fp4_param_gather": True}, "fp4_param_gather"),
        ({"use_precision_aware_optimizer": True}, "precision-aware"),
    ],
)
def test_compact_muon_rejects_deferred_layout_overlap_and_offload(
    optimizer_overrides, message
) -> None:
    optimizer = SimpleNamespace(optimizer="muon", **optimizer_overrides)
    engine_config = SimpleNamespace(
        parallel=SimpleNamespace(vpp=1, pp=1), optimizer=optimizer
    )

    with pytest.raises(ValueError, match=message):
        validate_dist_opt_config(engine_config)


@pytest.mark.parametrize(
    ("optimizer_overrides", "message"),
    [
        ({"overlap_grad_reduce": True}, "overlap_grad_reduce"),
        ({"overlap_param_gather": True}, "overlap_param_gather"),
        ({"optimizer_cpu_offload": True}, "offload"),
        ({"fp8_param_gather": True}, "fp8_param_gather"),
        ({"fp4_param_gather": True}, "fp4_param_gather"),
    ],
)
def test_compact_muon_rejects_runtime_optimizer_overrides(
    optimizer_overrides, message
) -> None:
    runtime_config = MegatronLiteConfig.from_dict(
        "/models/Qwen3",
        {
            "optimizer": {
                "optimizer": "muon",
                "override_optimizer_config": optimizer_overrides,
            }
        },
    )
    engine_config = SimpleNamespace(
        parallel=runtime_config.parallel,
        optimizer=runtime_config.optimizer,
    )

    with pytest.raises(ValueError, match=message):
        validate_dist_opt_config(engine_config)


def test_prewrapped_muon_rejection_does_not_mutate_model_metadata() -> None:
    model = _SyntheticModel()
    metadata_before = {
        name: dict(param.__dict__) for name, param in model.named_parameters()
    }
    engine_config = SimpleNamespace(
        parallel=SimpleNamespace(vpp=1, pp=1, tp=1, ep=1, etp=1, cp=1),
        optimizer=OptimizerConfig(optimizer="muon"),
        deterministic=False,
    )

    with pytest.raises(ValueError, match="unwrapped model chunks"):
        build_dist_opt_stack(
            [model],
            model_cfg=SimpleNamespace(),
            engine_cfg=engine_config,
            ps=SimpleNamespace(),
            is_expert=lambda _name: False,
            skip_ddp_wrap=True,
        )

    assert {
        name: dict(param.__dict__) for name, param in model.named_parameters()
    } == metadata_before


def test_invalid_muon_override_rejects_before_metadata_mutation(monkeypatch) -> None:
    import megatron.lite.primitive.optimizers.muon_routing as routing_module

    _install_fake_core_optimizer_config(
        monkeypatch, name="FakeCoreOptimizerConfigForEarlyRejection"
    )
    monkeypatch.setattr(
        routing_module,
        "tag_muon_parameter_metadata",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid lowering override must fail before metadata mutation"
        ),
    )
    optimizer_config = OptimizerConfig(optimizer="muon")
    optimizer_config.override_optimizer_config = {"use_distributed_optimizer": True}
    engine_config = SimpleNamespace(
        parallel=SimpleNamespace(vpp=1, pp=1, tp=1, ep=1, etp=1, cp=1),
        optimizer=optimizer_config,
        deterministic=False,
    )

    with pytest.raises(ValueError, match="lowering-owned field"):
        build_dist_opt_stack(
            [_SyntheticModel()],
            model_cfg=SimpleNamespace(),
            engine_cfg=engine_config,
            ps=SimpleNamespace(),
            is_expert=lambda _name: False,
        )


def test_pg_collection_uses_explicit_owner_groups_without_world_or_singletons(
    monkeypatch,
) -> None:
    group_field_names = (
        "tp",
        "cp",
        "pp",
        "ep",
        "mp",
        "dp",
        "dp_cp",
        "expt_dp",
        "expt_tp",
        "tp_ep",
        "tp_ep_pp",
        "intra_dist_opt",
        "embd",
        "pos_embd",
    )
    FakeProcessGroupCollection = make_dataclass(
        "FakeProcessGroupCollection",
        [(name, object, dataclass_field(default=None)) for name in group_field_names],
    )

    process_groups_module = types.ModuleType("megatron.core.process_groups_config")
    process_groups_module.ProcessGroupCollection = FakeProcessGroupCollection
    monkeypatch.setitem(
        sys.modules, "megatron.core.process_groups_config", process_groups_module
    )

    monkeypatch.setattr(
        torch.distributed,
        "new_group",
        lambda *_args, **_kwargs: pytest.fail(
            "compact PP1 must not create singleton groups"
        ),
    )
    dense_dp = object()
    expert_owner = object()
    optimizer_owner = object()
    ps = SimpleNamespace(
        pp_group=object(),
        tp_group=object(),
        cp_group=object(),
        ep_group=object(),
        dp_group=object(),
        dp_cp_group=dense_dp,
        ep_dp_group=expert_owner,
        intra_dist_opt_group=optimizer_owner,
        etp_group=object(),
        tp_ep_group=object(),
    )
    engine_config = SimpleNamespace(
        parallel=SimpleNamespace(pp=1), optimizer=SimpleNamespace(optimizer="muon")
    )

    groups = _build_pg_collection(ps, engine_config)

    assert groups.intra_dist_opt is optimizer_owner
    assert groups.intra_dist_opt is not groups.dp_cp
    assert groups.embd is None
    assert groups.pos_embd is None


def test_intra_dist_opt_owner_rank_order_matches_upstream_single_instance(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 16)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "new_group",
        lambda ranks, **_kwargs: tuple(ranks),
    )

    state = init_parallel(SimpleNamespace(tp=2, ep=2, etp=2, cp=1, pp=2))

    assert state.intra_dist_opt_group == (
        0,
        1,
        2,
        3,
        8,
        9,
        10,
        11,
        4,
        5,
        6,
        7,
        12,
        13,
        14,
        15,
    )


def test_native_fields_route_without_rewriting_and_legacy_offload_keeps_compat(
    monkeypatch,
) -> None:
    _install_fake_core_optimizer_config(monkeypatch, name="FakeCoreOptimizerConfig")

    config = OptimizerConfig(
        optimizer="Muon",
        muon_momentum=0.9,
        muon_split_qkv=False,
        muon_nesterov=True,
        muon_scale_mode="unit_rms_norm",
        muon_fp32_matmul_prec="high",
        muon_coefficient_type="simple",
        muon_num_ns_steps=7,
        muon_tp_mode="duplicated",
        muon_extra_scale_factor=0.2,
        use_layer_wise_param_layout=True,
        overlap_param_gather=True,
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=0.25,
        use_torch_optimizer_for_cpu_offload=True,
        overlap_cpu_optimizer_d2h_h2d=False,
        pin_cpu_grads=False,
        pin_cpu_params=False,
        offload_optimizer_states=True,
    )

    core = build_dist_opt_optimizer_config(config, complete_muon_lowering=True)

    assert core.muon_momentum == 0.9
    assert core.optimizer == "muon"
    assert core.muon_split_qkv is False
    assert core.muon_nesterov is True
    assert core.muon_scale_mode == "unit_rms_norm"
    assert core.muon_fp32_matmul_prec == "high"
    assert core.muon_coefficient_type == "simple"
    assert core.muon_num_ns_steps == 7
    assert core.muon_tp_mode == "duplicated"
    assert core.muon_extra_scale_factor == 0.2
    assert core.muon_scalar_optimizer == "adam"
    assert core.use_distributed_optimizer is False
    assert core.use_layer_wise_distributed_optimizer is True
    assert core.use_layer_wise_param_layout is True
    assert core.overlap_param_gather is True
    assert core.overlap_param_gather_with_optimizer_step is False
    assert core.optimizer_cpu_offload is True
    assert core.optimizer_offload_fraction == 0.25
    assert core.use_torch_optimizer_for_cpu_offload is True
    assert core.overlap_cpu_optimizer_d2h_h2d is False
    assert core.pin_cpu_grads is False
    assert core.pin_cpu_params is False
    assert core.offload_optimizer_states is True

    legacy_core = build_dist_opt_optimizer_config(
        OptimizerConfig(offload_fraction=0.5, overlap_cpu_optimizer_d2h_h2d=False)
    )

    assert legacy_core.optimizer_cpu_offload is True
    assert legacy_core.optimizer_offload_fraction == 0.5
    assert legacy_core.overlap_cpu_optimizer_d2h_h2d is True


def test_optimizer_overrides_cannot_change_lowering_owned_fields(monkeypatch) -> None:
    _install_fake_core_optimizer_config(
        monkeypatch, name="FakeCoreOptimizerConfigForOverrides"
    )

    with pytest.raises(ValueError, match="lowering-owned field 'optimizer'"):
        build_dist_opt_optimizer_config(
            OptimizerConfig(optimizer="adam"),
            override_optimizer_config={"optimizer": "muon"},
        )


def test_direct_builder_rejects_muon_without_complete_lowering(monkeypatch) -> None:
    _install_fake_core_optimizer_config(
        monkeypatch, name="FakeCoreOptimizerConfigForDirectMuon"
    )

    with pytest.raises(ValueError, match="complete.*lowering"):
        build_dist_opt_optimizer_config(OptimizerConfig(optimizer="muon"))


def test_muon_requires_pinned_layerwise_core_contract(monkeypatch) -> None:
    _install_fake_core_optimizer_config(
        monkeypatch,
        name="FakeOldCoreOptimizerConfig",
        excluded_fields={"use_layer_wise_distributed_optimizer"},
    )

    with pytest.raises(
        RuntimeError,
        match="pinned Megatron d64ba4ccb.*use_layer_wise_distributed_optimizer",
    ):
        build_dist_opt_optimizer_config(
            OptimizerConfig(optimizer="muon"), complete_muon_lowering=True
        )
