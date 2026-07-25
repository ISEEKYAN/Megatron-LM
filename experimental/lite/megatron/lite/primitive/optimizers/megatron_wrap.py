# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-Core optimizer wrap backend for Megatron Lite."""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.protocols import (
    ExpertClassifierFn,
    default_expert_classifier,
)


def _optimizer_overrides(opt) -> dict[str, Any]:
    overrides = getattr(opt, "override_optimizer_config", None)
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise TypeError("override_optimizer_config must be a dict")
    return overrides


def _optimizer_value(opt, name: str, default=None):
    overrides = _optimizer_overrides(opt)
    if name in overrides:
        return overrides[name]
    return getattr(opt, name, default)


def validate_dist_opt_config(engine_cfg) -> None:
    """Validate dist_opt constraints owned by this optimizer primitive."""
    p = engine_cfg.parallel
    if p.vpp > 1 and p.pp == 1:
        raise ValueError("dist_opt requires pp>1 when vpp>1.")

    opt = engine_cfg.optimizer
    if str(opt.optimizer).lower() != "muon":
        return
    if bool(_optimizer_value(opt, "use_layer_wise_param_layout", False)):
        raise ValueError(
            "Muon padded LayerWise layout is deferred; use compact layout."
        )
    if bool(_optimizer_value(opt, "overlap_grad_reduce", False)):
        raise ValueError(
            "Muon compact lowering does not yet support overlap_grad_reduce."
        )
    if bool(_optimizer_value(opt, "overlap_param_gather", False)):
        raise ValueError(
            "Muon compact lowering does not yet support overlap_param_gather."
        )
    if bool(_optimizer_value(opt, "overlap_param_gather_with_optimizer_step", False)):
        raise ValueError(
            "Muon does not support parameter gather overlap with optimizer step."
        )
    if bool(_optimizer_value(opt, "fp8_param_gather", False)):
        raise ValueError("Muon compact lowering does not support fp8_param_gather.")
    if bool(_optimizer_value(opt, "fp4_param_gather", False)):
        raise ValueError("Muon compact lowering does not support fp4_param_gather.")
    if bool(_optimizer_value(opt, "use_precision_aware_optimizer", False)):
        raise ValueError(
            "Muon does not support the Adam-only precision-aware optimizer."
        )

    offload_requested = any(
        (
            bool(_optimizer_value(opt, "optimizer_cpu_offload", False)),
            float(_optimizer_value(opt, "optimizer_offload_fraction", 0.0) or 0.0)
            > 0.0,
            bool(_optimizer_value(opt, "use_torch_optimizer_for_cpu_offload", False)),
            bool(_optimizer_value(opt, "overlap_cpu_optimizer_d2h_h2d", False)),
            bool(_optimizer_value(opt, "offload_optimizer_states", False)),
            float(_optimizer_value(opt, "offload_fraction", 0.0) or 0.0) > 0.0,
        )
    )
    if offload_requested:
        raise ValueError(
            "Muon optimizer offload is deferred to the dedicated offload lowering."
        )


def _effective_etp(parallel) -> int:
    return int(parallel.etp if parallel.etp is not None else 1)


def _ensure_dist_opt_mpu_parallel_state(engine_cfg) -> None:
    """Initialize Megatron-Core mpu globals when dist_opt fallback groups are used."""

    from megatron.core import parallel_state as mpu  # pyright: ignore[reportMissingImports]

    p = engine_cfg.parallel
    expected = (int(p.tp), int(p.ep), _effective_etp(p), int(p.pp), int(p.cp))
    if mpu.is_initialized():
        current = (
            int(mpu.get_tensor_model_parallel_world_size()),
            int(mpu.get_expert_model_parallel_world_size()),
            int(mpu.get_expert_tensor_parallel_world_size() or 1),
            int(mpu.get_pipeline_model_parallel_world_size()),
            int(mpu.get_context_parallel_world_size()),
        )
        if current != expected:
            raise RuntimeError(
                "dist_opt found an incompatible existing Megatron-Core parallel state: "
                f"current={current}, expected={expected}."
            )
        return

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=p.tp,
        pipeline_model_parallel_size=p.pp,
        virtual_pipeline_model_parallel_size=None if int(p.vpp or 1) <= 1 else p.vpp,
        context_parallel_size=p.cp,
        expert_model_parallel_size=p.ep,
        expert_tensor_parallel_size=_effective_etp(p),
        create_gloo_process_groups=bool(getattr(engine_cfg, "deterministic", False)),
    )


def build_dist_opt_optimizer_config(
    opt,
    *,
    override_optimizer_config: dict[str, Any] | None = None,
    complete_muon_lowering: bool = False,
):
    """Build Megatron-Core OptimizerConfig from user's OptimizerConfig (duck-typed).

    Single source of truth for Megatron Lite's Megatron-Core optimizer stack.

    Works on either `runtime.contracts.config.OptimizerConfig` (real dataclass)
    or a `SimpleNamespace` with the same field names (legacy lite path).
    Direct Muon callers are rejected because only ``build_dist_opt_stack`` owns
    the required metadata, layout, and DDP lowering sequence.
    """
    optimizer_name = str(opt.optimizer).lower()
    use_muon = optimizer_name == "muon"
    if use_muon and not complete_muon_lowering:
        raise ValueError(
            "Muon requires the complete metadata, layout, DDP, and optimizer lowering; "
            "use build_dist_opt_stack instead of constructing its optimizer config directly."
        )

    from megatron.core.optimizer.optimizer_config import (
        OptimizerConfig as CoreOptimizerConfig,  # pyright: ignore[reportMissingImports]
    )

    legacy_offload = getattr(opt, "offload_fraction", None)
    native_offload = getattr(opt, "optimizer_offload_fraction", None)
    if legacy_offload is not None and native_offload not in (None, 0.0, legacy_offload):
        raise ValueError(
            "offload_fraction compatibility alias conflicts with optimizer_offload_fraction"
        )
    offload = native_offload if native_offload is not None else legacy_offload
    offload = float(offload or 0.0)
    args: dict[str, Any] = {
        "optimizer": optimizer_name,
        "lr": opt.lr,
        "min_lr": getattr(opt, "min_lr", 0.0),
        "weight_decay": opt.weight_decay,
        "clip_grad": opt.clip_grad,
        # Megatron keeps this False for the LayerWise facade. DDP itself is still
        # configured with use_distributed_optimizer=True so the Adam fallback gets
        # its byte-sharded buffers.
        "use_distributed_optimizer": not use_muon,
        "bf16": True,
        "params_dtype": torch.bfloat16,
    }
    core_fields = {field.name for field in fields(CoreOptimizerConfig)}
    native_fields = (
        "muon_momentum",
        "muon_split_qkv",
        "muon_nesterov",
        "muon_scale_mode",
        "muon_fp32_matmul_prec",
        "muon_coefficient_type",
        "muon_num_ns_steps",
        "muon_tp_mode",
        "muon_extra_scale_factor",
        "muon_scalar_optimizer",
        "use_layer_wise_param_layout",
        "overlap_param_gather",
        "overlap_param_gather_with_optimizer_step",
        "optimizer_cpu_offload",
        "optimizer_offload_fraction",
        "use_torch_optimizer_for_cpu_offload",
        "overlap_cpu_optimizer_d2h_h2d",
        "pin_cpu_grads",
        "pin_cpu_params",
        "offload_optimizer_states",
    )
    for name in native_fields:
        if name in core_fields and hasattr(opt, name):
            args[name] = getattr(opt, name)
    if legacy_offload is not None and offload > 0:
        # Preserve the legacy alias semantics. Canonical Megatron fields above are
        # otherwise forwarded verbatim, including an explicitly disabled overlap.
        args["optimizer_offload_fraction"] = offload
        args["overlap_cpu_optimizer_d2h_h2d"] = True
        args["optimizer_cpu_offload"] = True
    if getattr(opt, "adam_beta1", None) is not None:
        args["adam_beta1"] = opt.adam_beta1
    if getattr(opt, "adam_beta2", None) is not None:
        args["adam_beta2"] = opt.adam_beta2
    if getattr(opt, "adam_eps", None) is not None:
        args["adam_eps"] = opt.adam_eps
    if getattr(opt, "use_precision_aware_optimizer", None) is not None:
        args["use_precision_aware_optimizer"] = opt.use_precision_aware_optimizer
    if getattr(opt, "decoupled_weight_decay", None) is not None:
        args["decoupled_weight_decay"] = opt.decoupled_weight_decay
    if use_muon:
        required_muon_fields = {
            "muon_momentum",
            "muon_split_qkv",
            "muon_nesterov",
            "muon_scale_mode",
            "muon_fp32_matmul_prec",
            "muon_coefficient_type",
            "muon_num_ns_steps",
            "muon_tp_mode",
            "muon_extra_scale_factor",
            "muon_scalar_optimizer",
            "use_layer_wise_distributed_optimizer",
            "use_layer_wise_param_layout",
        }
        missing = sorted(required_muon_fields - core_fields)
        if missing:
            raise RuntimeError(
                "Muon requires the pinned Megatron d64ba4ccb optimizer contract; "
                f"runtime is missing fields: {missing}"
            )
        args["use_layer_wise_distributed_optimizer"] = True
    if override_optimizer_config:
        normalized_overrides = dict(override_optimizer_config)
        if "optimizer" in normalized_overrides:
            normalized_overrides["optimizer"] = str(
                normalized_overrides["optimizer"]
            ).lower()
        lowering_owned = {
            "optimizer",
            "use_distributed_optimizer",
            "use_layer_wise_distributed_optimizer",
            "use_layer_wise_param_layout",
        }
        for name in lowering_owned.intersection(normalized_overrides):
            value = normalized_overrides[name]
            if name not in args or value != args[name]:
                raise ValueError(
                    f"override_optimizer_config cannot change lowering-owned field '{name}'"
                )
        args.update(normalized_overrides)
    return CoreOptimizerConfig(**args)


def build_dist_opt_stack(
    model_chunks: list[nn.Module],
    *,
    model_cfg,
    engine_cfg,
    ps,
    is_expert: ExpertClassifierFn | None = None,
    proto=None,
    skip_ddp_wrap: bool = False,
):
    """Wrap ML model chunks with Megatron-Core DDP and build the matching dist_opt optimizer.

    Args:
        skip_ddp_wrap: when True, ``model_chunks`` are assumed to already be
            Megatron-Core ``DistributedDataParallel``-wrapped; we skip our own wrapping
            and feed them directly to the optimizer. The bucket layout
            influences optimizer master-grad sharding, so callers that prewrap
            chunks own the DDP config compatibility.
    """
    validate_dist_opt_config(engine_cfg)

    p = engine_cfg.parallel
    opt = engine_cfg.optimizer
    if is_expert is not None:
        is_expert_param = is_expert
    elif proto is not None and hasattr(proto, "EXPERT_CLASSIFIER"):
        is_expert_param = proto.EXPERT_CLASSIFIER
    else:
        is_expert_param = default_expert_classifier

    use_muon = str(opt.optimizer).lower() == "muon"
    if use_muon and skip_ddp_wrap:
        raise ValueError(
            "Muon×dist_opt requires unwrapped model chunks for buffer routing."
        )

    # Validate and construct the complete Core optimizer contract before metadata,
    # process-group, or DDP mutation. In particular, lowering-owned overrides must
    # fail without leaving a partially wrapped model behind.
    opt_config = build_dist_opt_optimizer_config(
        opt,
        override_optimizer_config=_optimizer_overrides(opt),
        complete_muon_lowering=use_muon,
    )

    if use_muon:
        from megatron.lite.primitive.optimizers.muon_routing import (
            tag_muon_parameter_metadata,
        )

        tag_muon_parameter_metadata(model_chunks, is_expert_param=is_expert_param)

    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.transformer.enums import ModelType

    dist_opt_transformer_cfg = _build_transformer_config(model_cfg, engine_cfg)
    dist_opt_transformer_cfg.finalize_model_grads_func = finalize_model_grads
    use_mpu_groups = bool(getattr(engine_cfg, "deterministic", False)) and not use_muon
    if use_muon or use_mpu_groups:
        # Pinned LayerWise still consults the ambient MCore model-parallel group
        # for grad statistics. Keep explicit pg_collection ownership below, but
        # initialize this residual upstream dependency before the first step.
        _ensure_dist_opt_mpu_parallel_state(engine_cfg)
    pg_collection = None if use_mpu_groups else _build_pg_collection(ps, engine_cfg)

    if skip_ddp_wrap:
        # Caller already wrapped and marked every param. Our helper setting
        # `param.allreduce` on dense params could clash with Megatron-Core code paths that
        # distinguish `hasattr(param,'allreduce')` from `getattr(..., True)`.
        wrapped_chunks = list(model_chunks)
    else:
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=bool(
                _optimizer_value(opt, "overlap_grad_reduce", False)
            ),
            overlap_param_gather=bool(
                _optimizer_value(opt, "overlap_param_gather", False)
            ),
            use_layer_wise_param_layout=bool(
                _optimizer_value(opt, "use_layer_wise_param_layout", False)
            ),
            grad_reduce_in_fp32=True,
        )
        per_chunk_layouts = [None] * len(model_chunks)
        if use_muon:
            from megatron.core.optimizer.layer_wise_optimizer import (
                LayerWiseDistributedOptimizer,
                tag_params_for_buffer_routing,
            )

            for chunk in model_chunks:
                _mark_dist_opt_parallel_attrs(chunk, is_expert_param, tp_size=p.tp)
            tag_params_for_buffer_routing(model_chunks)

            data_parallel_world_size = pg_collection.dp_cp.size()
            expert_data_parallel_world_size = pg_collection.expt_dp.size()
            for chunk_idx, chunk in enumerate(model_chunks):
                params = [param for param in chunk.parameters() if param.requires_grad]
                per_chunk_layouts[chunk_idx] = (
                    LayerWiseDistributedOptimizer.compute_full_param_layout(
                        params,
                        None,
                        data_parallel_world_size,
                        ddp_config,
                        expert_data_parallel_world_size=expert_data_parallel_world_size,
                    )
                )

        wrapped_chunks = []
        for chunk_idx, chunk in enumerate(model_chunks):
            chunk.model_type = ModelType.encoder_or_decoder
            if not use_muon:
                _mark_dist_opt_parallel_attrs(chunk, is_expert_param, tp_size=p.tp)
            ddp_kwargs = {}
            if pg_collection is not None:
                ddp_kwargs["pg_collection"] = pg_collection
            if per_chunk_layouts[chunk_idx] is not None:
                ddp_kwargs["full_param_layout"] = per_chunk_layouts[chunk_idx]
            wrapped_chunks.append(
                DistributedDataParallel(
                    dist_opt_transformer_cfg,
                    ddp_config,
                    chunk,
                    disable_bucketing=(chunk_idx > 0),
                    **ddp_kwargs,
                )
            )

    # This branch falls back to Megatron-Core mpu globals for the optimizer's process
    # groups. Long term, this primitive should always pass its own
    # `pg_collection`.
    if skip_ddp_wrap or use_mpu_groups:
        optimizer = get_megatron_optimizer(
            config=opt_config, model_chunks=wrapped_chunks
        )
        optimizer._dist_opt_pg_collection = None  # pyright: ignore[reportAttributeAccessIssue]
    else:
        optimizer = get_megatron_optimizer(
            config=opt_config,
            model_chunks=wrapped_chunks,
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )
        optimizer._dist_opt_pg_collection = pg_collection  # pyright: ignore[reportAttributeAccessIssue]
    return wrapped_chunks, optimizer


def build_dist_opt_training_optimizer(
    model_chunks: list[nn.Module],
    *,
    model_cfg,
    impl_cfg,
    ps,
    model_name: str,
    is_expert: ExpertClassifierFn | None = None,
    skip_ddp_wrap: bool = False,
    deterministic: bool | None = None,
):
    """Build the dist_opt DDP+optimizer stack from a Megatron Lite model ImplConfig."""

    opt = impl_cfg.optimizer_config
    if opt is None:
        opt = SimpleNamespace(
            optimizer="adam",
            lr=1e-4,
            weight_decay=0.01,
            clip_grad=1.0,
            offload_fraction=None,
            adam_beta1=None,
            adam_beta2=None,
            adam_eps=None,
        )
    if deterministic is None:
        from megatron.lite.primitive.deterministic import deterministic_requested

        deterministic = deterministic_requested()

    engine_cfg = SimpleNamespace(
        model_name=model_name,
        parallel=impl_cfg.parallel,
        optimizer=opt,
        deterministic=bool(deterministic),
    )
    model_chunks[:], optimizer = build_dist_opt_stack(
        model_chunks,
        model_cfg=model_cfg,
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=is_expert,
        skip_ddp_wrap=skip_ddp_wrap,
    )

    def finalize_grads() -> None:
        finalize_dist_opt_grads(model_chunks, optimizer)

    return optimizer, finalize_grads


def finalize_dist_opt_grads(model_chunks: list[nn.Module], optimizer) -> None:
    """Run Megatron-Core gradient finalization to match the optimizer's expected contract."""
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads

    finalize_model_grads(model_chunks, pg_collection=optimizer._dist_opt_pg_collection)


def _build_transformer_config(model_cfg, engine_cfg):
    from megatron.core.transformer.transformer_config import TransformerConfig

    p = engine_cfg.parallel
    kwargs = dict(
        num_layers=max(getattr(model_cfg, "num_hidden_layers", 1), 1),
        hidden_size=max(getattr(model_cfg, "hidden_size", 1), 1),
        num_attention_heads=max(getattr(model_cfg, "num_attention_heads", 1), 1),
        num_query_groups=getattr(model_cfg, "num_key_value_heads", None),
        num_moe_experts=getattr(model_cfg, "num_experts", None),
        moe_ffn_hidden_size=getattr(model_cfg, "moe_intermediate_size", None),
        tensor_model_parallel_size=p.tp,
        pipeline_model_parallel_size=p.pp,
        context_parallel_size=p.cp,
        expert_model_parallel_size=p.ep,
        expert_tensor_parallel_size=p.etp if p.etp is not None else 1,
        sequence_parallel=p.tp > 1,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    if hasattr(model_cfg, "add_bias_linear"):
        kwargs["add_bias_linear"] = bool(model_cfg.add_bias_linear)
    elif kwargs["num_moe_experts"] is not None:
        kwargs["add_bias_linear"] = False
    if p.pp > 1:
        kwargs["pipeline_dtype"] = torch.bfloat16
    return TransformerConfig(**kwargs)


def _mark_dist_opt_parallel_attrs(
    model: nn.Module, is_expert_param: ExpertClassifierFn, *, tp_size: int
) -> None:
    """Mark per-param optimizer metadata (allreduce / tensor_model_parallel / sequence_parallel).

    IMPORTANT: respect attrs that are already set. Prewrapped Megatron-Core models may
    mark these correctly per-param (e.g. `moe.router.weight` is 2D but
    TP-replicated, and must NOT have `tensor_model_parallel=True`). Blind
    override would cause dist_opt grad-norm to over-count replicated params.
    """
    sp_param_ids = {id(param) for param in getattr(model, "sp_params", [])}
    for name, param in model.named_parameters():
        # Megatron-Core uses `allreduce=False` to route expert params into expert-DP buffers.
        if not hasattr(param, "allreduce"):
            param.allreduce = not is_expert_param(name)
        if tp_size > 1 and id(param) not in sp_param_ids and param.ndim > 1:
            # vision params are replicated across TP (AVG all-reduce, not TP-split).
            # tensor_model_parallel=True would cause dist_opt to wrong-account their grad-norm.
            if getattr(param, "average_gradients_across_tp_domain", False):
                continue
            # Skip params already marked sequence_parallel=True: they are TP-replicated
            # with SP-sharded input (e.g. shared_experts.gate_weight, RMSNorm weights).
            # Stacking tensor_model_parallel=True on top would cause double all-reduce.
            if getattr(param, "sequence_parallel", False):
                continue
            # Distopt excludes TP replicas from grad-norm accounting via this metadata.
            if not hasattr(param, "tensor_model_parallel"):
                param.tensor_model_parallel = True

    for param in getattr(model, "sp_params", []):
        if not hasattr(param, "sequence_parallel"):
            param.sequence_parallel = True
        param.allreduce = True
        param.tensor_model_parallel = False


def _build_pg_collection(ps, engine_cfg):
    import torch.distributed as dist  # pyright: ignore[reportMissingImports]

    from megatron.core.process_groups_config import ProcessGroupCollection

    if ps.pp_group is None:
        raise ValueError("dist_opt requires a local pp_group.")
    if ps.intra_dist_opt_group is None:
        raise ValueError("dist_opt requires an explicit intra_dist_opt owner group.")

    def _dense_rank(tp_i: int, cp_i: int, dp_i: int, pp_i: int) -> int:
        return ((pp_i * ps.dp_size + dp_i) * ps.cp_size + cp_i) * ps.tp_size + tp_i

    def _expert_rank(etp_i: int, ep_i: int, edp_i: int, pp_i: int) -> int:
        return (
            (pp_i * ps.expert_dp_size + edp_i) * ps.ep_size + ep_i
        ) * ps.etp_size + etp_i

    if engine_cfg.parallel.pp == 1:
        mp_group = ps.tp_group
        tp_ep_pp_group = ps.tp_ep_group
    else:
        rank = dist.get_rank()
        mp_group = None
        for dp_idx in range(ps.dp_size):
            for cp_idx in range(ps.cp_size):
                ranks = [
                    _dense_rank(tp_idx, cp_idx, dp_idx, pp_idx)
                    for pp_idx in range(ps.pp_size)
                    for tp_idx in range(ps.tp_size)
                ]
                group = dist.new_group(ranks)
                if rank in ranks:
                    mp_group = group

        tp_ep_pp_group = None
        for expert_dp_idx in range(ps.expert_dp_size):
            ranks = [
                _expert_rank(etp_idx, ep_idx, expert_dp_idx, pp_idx)
                for pp_idx in range(ps.pp_size)
                for ep_idx in range(ps.ep_size)
                for etp_idx in range(ps.etp_size)
            ]
            group = dist.new_group(ranks)
            if rank in ranks:
                tp_ep_pp_group = group

        if mp_group is None or tp_ep_pp_group is None:
            raise RuntimeError(
                "Failed to construct dist_opt pipeline-aware process groups."
            )

    pg_kwargs = dict(
        tp=ps.tp_group,
        cp=ps.cp_group,
        pp=ps.pp_group,
        ep=ps.ep_group,
        mp=mp_group,
        dp=ps.dp_group,
        dp_cp=ps.dp_cp_group,
        expt_dp=ps.ep_dp_group,
        expt_tp=ps.etp_group,
        tp_ep=ps.tp_ep_group,
        tp_ep_pp=tp_ep_pp_group,
        # Optimizer-wide grad statistics use the explicit single-instance group.
        # This is distinct from dense ``dp_cp`` and expert ``expt_dp`` owner domains.
        intra_dist_opt=ps.intra_dist_opt_group,
        # Native MLite models do not expose MCore's tied pipeline-embedding surface.
        # ``None`` is the explicit ProcessGroupCollection contract for an unused group.
        embd=None,
        pos_embd=None,
    )
    supported_fields = {field.name for field in fields(ProcessGroupCollection)}
    return ProcessGroupCollection(
        **{key: value for key, value in pg_kwargs.items() if key in supported_fields}
    )


# ---------------------------------------------------------------------------
# Backend adapter (consumed by runtime/session.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DistOptBackend:
    name: str = "dist_opt"
    runtime_backend: str = "dist_opt"

    def zero_grad(self, optimizer: Any) -> None:
        optimizer.zero_grad()

    def finish_grad_sync(self, optimizer: Any) -> None:
        if hasattr(optimizer, "finish_grad_sync"):
            optimizer.finish_grad_sync()

    def clip_grad_norm(self, optimizer: Any):
        if hasattr(optimizer, "clip_grad_norm"):
            return optimizer.clip_grad_norm()
        return None

    def step(self, optimizer: Any):
        return optimizer.step()

    def state_dict(self, optimizer: Any) -> dict:
        return optimizer.state_dict()

    def load_state_dict(self, optimizer: Any, state_dict: dict) -> None:
        optimizer.load_state_dict(state_dict)

    def finalize_grads(
        self, finalize_fn, model_chunks: list[Any], optimizer: Any
    ) -> None:
        finalize_fn(model_chunks, optimizer)


BACKEND = DistOptBackend()

__all__ = [
    "BACKEND",
    "DistOptBackend",
    "build_dist_opt_optimizer_config",
    "build_dist_opt_stack",
    "build_dist_opt_training_optimizer",
    "finalize_dist_opt_grads",
    "validate_dist_opt_config",
]
