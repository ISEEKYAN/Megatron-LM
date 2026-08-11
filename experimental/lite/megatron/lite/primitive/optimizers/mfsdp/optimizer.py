# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Self-contained Megatron-FSDP construction for Megatron Lite."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.lite.primitive.optimizers.mfsdp.config import (
    MFSDPConfig,
    annotate_parallel_parameters,
    build_mfsdp_config,
    build_mfsdp_process_groups,
    validate_mfsdp_config,
)
from megatron.lite.primitive.optimizers.mfsdp.cpu_offload import CpuAdamGroup
from megatron.lite.primitive.optimizers.mfsdp.fully_shard import fully_shard_model
from megatron.lite.primitive.optimizers.mfsdp.fused_ops import (
    OptimizerFactory,
    build_optimizer,
)
from megatron.lite.primitive.optimizers.mfsdp.grad_norm import (
    all_reduce_scalar_,
    local_grad_sq_sum,
    resolve_torch_dtype,
)
from megatron.lite.primitive.optimizers.mfsdp.wrapper import MFSdpModule

logger = logging.getLogger(__name__)

ExpertClassifierFn = Callable[[str], bool]
_MFSDP_PARAM_VALUES_KEY = "_mfsdp_param_values"


class _NullOptimizer:
    """Torch-optimizer surface for the all-CPU-update case."""

    param_groups: list[dict[str, Any]] = []
    state: dict[Any, Any] = {}

    def step(self) -> None:
        pass

    def zero_grad(self, *args, **kwargs) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {"state": {}, "param_groups": []}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict not in ({}, {"state": {}, "param_groups": []}):
            raise ValueError("Invalid empty GPU optimizer state.")

def _override(opt: Any, name: str, default: Any) -> Any:
    values = dict(getattr(opt, "override_optimizer_config", None) or {})
    return values.get(name, getattr(opt, name, default))


class _StandaloneOptimizer:
    """Torch optimizer adapter with M-FSDP-aware norm and state movement."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: list[nn.Parameter],
        *,
        ps: Any,
        clip_grad: float,
        grad_norm_accum_dtype: str | torch.dtype,
        expert_params: Iterable[nn.Parameter],
        expert_grad_scale: float,
        use_decoupled_grad: bool = False,
        cpu_group: CpuAdamGroup | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params = params
        self.ps = ps
        self.clip_grad = float(clip_grad)
        self.grad_norm_accum_dtype = resolve_torch_dtype(grad_norm_accum_dtype)
        self.expert_params = list(expert_params)
        self._expert_param_ids = {id(param) for param in self.expert_params}
        self.tp_replicated_params = [
            param
            for param in self.params
            if id(param) not in self._expert_param_ids
            and (
                bool(getattr(param, "sequence_parallel", False))
                or bool(getattr(param, "average_gradients_across_tp_domain", False))
            )
            and not bool(getattr(param, "shared", False))
        ]
        self._tp_replicated_param_ids = {
            id(param) for param in self.tp_replicated_params
        }
        self.expert_grad_scale = float(expert_grad_scale)
        self.use_decoupled_grad = bool(use_decoupled_grad)
        self._expert_grads_scaled = False
        self._tp_replicated_grads_synced = False
        self._offloaded_state_devices: dict[tuple[int, str], torch.device] = {}
        self.cpu_group = cpu_group
        self._cpu_param_ids = (
            set() if cpu_group is None else {id(param) for param in cpu_group.gpu_params}
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups = list(self.optimizer.param_groups)
        if self.cpu_group is not None:
            groups.extend(self.cpu_group.param_groups)
        return groups

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()
        self._expert_grads_scaled = False
        self._tp_replicated_grads_synced = False

    def step(self) -> tuple[bool, float, int]:
        self._sync_tp_replicated_grads_once()
        self._scale_expert_grads_once()
        grad_norm = self.clip_grad_norm()
        if not math.isfinite(grad_norm):
            return False, grad_norm, 0
        self.optimizer.step()
        if self.cpu_group is not None:
            self.cpu_group.step()
        return True, grad_norm, 0

    @torch.no_grad()
    def update_optimizer_grads(self) -> None:
        """Install reduced shards using MCore's standard or decoupled grad path."""
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if id(param) in self._cpu_param_ids:
                    # CPU AdamW consumes the FP32 M-FSDP main-grad directly;
                    # installing it as a BF16 .grad would discard precision.
                    param.grad = None
                    if hasattr(param, "decoupled_grad"):
                        param.decoupled_grad = None
                    continue
                main_grad = getattr(param, "main_grad", None)
                if main_grad is None or main_grad.numel() == 0:
                    continue
                if self.use_decoupled_grad:
                    param.grad = None
                    param.decoupled_grad = main_grad
                elif param.grad is None:
                    param.grad = main_grad.to(dtype=param.dtype)

    def clip_grad_norm(self) -> float:
        self._sync_tp_replicated_grads_once()
        self._scale_expert_grads_once()
        dense_params = [
            param
            for param in self.params
            if id(param) not in self._expert_param_ids
            and id(param) not in self._tp_replicated_param_ids
            and _include_dense_param_in_norm(param, getattr(self.ps, "tp_group", None))
        ]
        default_device = self.params[0].device if self.params else torch.device("cpu")
        dense_sq = local_grad_sq_sum(
            dense_params,
            dtype=self.grad_norm_accum_dtype,
            default_device=default_device,
        )
        dense_dp_group = getattr(self.ps, "dp_cp_group", None) or getattr(
            self.ps, "dp_group", None
        )
        _sum_if_distributed(dense_sq, dense_dp_group)
        _sum_if_distributed(dense_sq, getattr(self.ps, "tp_group", None))

        tp_replicated_sq = local_grad_sq_sum(
            self.tp_replicated_params,
            dtype=self.grad_norm_accum_dtype,
            default_device=dense_sq.device,
        )
        _sum_if_distributed(tp_replicated_sq, dense_dp_group)

        expert_sq = local_grad_sq_sum(
            self.expert_params,
            dtype=self.grad_norm_accum_dtype,
            default_device=dense_sq.device,
        )
        _sum_if_distributed(expert_sq, getattr(self.ps, "ep_dp_group", None))
        _sum_if_distributed(expert_sq, getattr(self.ps, "etp_group", None))
        _sum_if_distributed(expert_sq, getattr(self.ps, "ep_group", None))
        total_sq = (
            dense_sq
            + tp_replicated_sq.to(dense_sq.device)
            + expert_sq.to(dense_sq.device)
        )
        _sum_if_distributed(total_sq, getattr(self.ps, "pp_group", None))
        total_norm = total_sq.sqrt()
        if bool(torch.isfinite(total_norm).item()) and self.clip_grad > 0.0:
            coefficient = min(1.0, self.clip_grad / (float(total_norm.item()) + 1.0e-6))
            if coefficient < 1.0:
                for param in self.params:
                    grad = _optimizer_grad(param)
                    if grad is not None:
                        grad.mul_(coefficient)
        return float(total_norm.float().item())

    def state_dict(self) -> dict[str, Any]:
        state = (
            self.optimizer.state_dict()
            if self.cpu_group is None
            else {
                "gpu": self.optimizer.state_dict(),
                "cpu": self.cpu_group.state_dict(),
            }
        )
        state[_MFSDP_PARAM_VALUES_KEY] = [
            [param.detach().cpu().clone() for param in group["params"]]
            for group in self.optimizer.param_groups
        ]
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = dict(state_dict)
        param_values = state.pop(_MFSDP_PARAM_VALUES_KEY, None)
        if self.cpu_group is None:
            self.optimizer.load_state_dict(state.get("gpu", state))
        elif "gpu" not in state or "cpu" not in state:
            raise ValueError(
                "M-FSDP CPU-offloaded optimizer checkpoint requires both "
                "'gpu' and 'cpu' states."
            )
        else:
            self.optimizer.load_state_dict(state["gpu"])
            self.cpu_group.load_state_dict(state["cpu"])
        if param_values is not None:
            with torch.no_grad():
                for group, group_values in zip(
                    self.optimizer.param_groups, param_values, strict=True
                ):
                    for param, value in zip(group["params"], group_values, strict=True):
                        param.copy_(value.to(device=param.device, dtype=param.dtype))

    def offload_state_to_cpu(self) -> None:
        if self.cpu_group is not None:
            self.cpu_group.release_transfer_state()
        self._offloaded_state_devices.clear()
        for param, state in self.optimizer.state.items():
            for key, value in tuple(state.items()):
                if not isinstance(value, torch.Tensor) or value.device.type == "cpu":
                    continue
                self._offloaded_state_devices[(id(param), str(key))] = value.device
                state[key] = value.cpu()

    def load_state_to_device(self) -> None:
        for param, state in self.optimizer.state.items():
            for key, value in tuple(state.items()):
                device = self._offloaded_state_devices.get((id(param), str(key)))
                if device is not None and isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        self._offloaded_state_devices.clear()

    def _scale_expert_grads_once(self) -> None:
        if self._expert_grads_scaled:
            return
        if self.expert_grad_scale != 1.0:
            for param in self.expert_params:
                grad = _optimizer_grad(param)
                if grad is not None:
                    grad.mul_(self.expert_grad_scale)
        self._expert_grads_scaled = True

    def _sync_tp_replicated_grads_once(self) -> None:
        if self._tp_replicated_grads_synced:
            return
        group = getattr(self.ps, "tp_group", None)
        for param in self.tp_replicated_params:
            grad = _optimizer_grad(param)
            if grad is not None:
                _all_reduce_grad_if_distributed(
                    grad,
                    group,
                    average=bool(
                        getattr(param, "average_gradients_across_tp_domain", False)
                    ),
                )
        self._tp_replicated_grads_synced = True


def _sum_if_distributed(value: torch.Tensor, group: dist.ProcessGroup | None) -> None:
    if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return
    all_reduce_scalar_(value, op=dist.ReduceOp.SUM, group=group)


def _optimizer_grad(param: nn.Parameter) -> torch.Tensor | None:
    """Prefer the dtype-compatible optimizer gradient installed by M-FSDP."""
    decoupled_grad = getattr(param, "decoupled_grad", None)
    if decoupled_grad is not None:
        return decoupled_grad
    if param.grad is not None:
        return param.grad
    return getattr(param, "main_grad", None)


def _all_reduce_grad_if_distributed(
    grad: torch.Tensor, group: dist.ProcessGroup | None, *, average: bool = False
) -> None:
    if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return
    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
    if average:
        grad.div_(dist.get_world_size(group))


def _include_dense_param_in_norm(
    param: nn.Parameter, tp_group: dist.ProcessGroup | None
) -> bool:
    if bool(getattr(param, "shared", False)):
        return False
    if bool(getattr(param, "tensor_model_parallel", False)):
        return True
    if tp_group is None or not dist.is_initialized():
        return True
    return dist.get_rank(tp_group) == 0


class MFSdpOptimizer:
    """Join the standalone optimizer and M-FSDP communication lifecycle."""

    name = "megatron_fsdp"

    def __init__(self, optimizer: Any, model_chunks: list[MFSdpModule]) -> None:
        self._inner_optimizer = optimizer
        self._model_chunks = model_chunks
        self._grad_sync_enabled = False
        self._rollout_offloaded = False
        self._pre_rollout_devices: list[torch.device] = []

    @property
    def grad_sync_enabled(self) -> bool:
        return self._grad_sync_enabled

    @grad_sync_enabled.setter
    def grad_sync_enabled(self, enabled: bool) -> None:
        self._grad_sync_enabled = bool(enabled)
        for chunk in self._model_chunks:
            chunk.param_sync.set_grad_sync_enabled(self._grad_sync_enabled)

    @property
    def param_groups(self):
        return self._inner_optimizer.param_groups

    def zero_grad(self) -> None:
        self._require_training_resident("zero_grad")
        self.grad_sync_enabled = False
        self._inner_optimizer.zero_grad()
        for chunk in self._model_chunks:
            chunk.zero_grad_buffer()

    def finish_grad_sync(self, *, update_optimizer_grads: bool = True) -> None:
        self._require_training_resident("finish_grad_sync")
        for chunk in self._model_chunks:
            chunk.finish_grad_sync()
        if update_optimizer_grads:
            self.update_optimizer_grads()

    def update_optimizer_grads(self) -> None:
        self._inner_optimizer.update_optimizer_grads()

    def clip_grad_norm(self) -> float:
        return self._inner_optimizer.clip_grad_norm()

    def step(self) -> tuple[bool, float, int]:
        self._require_training_resident("step")
        result = self._inner_optimizer.step()
        for chunk in self._model_chunks:
            chunk.param_sync.copy_main_weights_to_model_weights()
            chunk.param_sync.invalidate_parameters()
        self.grad_sync_enabled = False
        return result

    def state_dict(self) -> dict[str, Any]:
        return self._inner_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._inner_optimizer.load_state_dict(state_dict)
        for chunk in self._model_chunks:
            chunk.param_sync.copy_main_weights_to_model_weights()
            chunk.param_sync.invalidate_parameters()

    def offload_state_to_cpu(self) -> None:
        self._inner_optimizer.offload_state_to_cpu()

    def load_state_to_device(self) -> None:
        self._inner_optimizer.load_state_to_device()

    def offload_for_rollout(self) -> None:
        """Move the complete training state out of GPU at an optimizer boundary."""
        if self._rollout_offloaded:
            return
        if self.grad_sync_enabled:
            raise RuntimeError(
                "M-FSDP train-to-rollout offload requires an optimizer-step boundary; "
                "gradient synchronization is still enabled."
            )
        self._pre_rollout_devices = [
            (
                chunk.param_sync.buckets[0].device
                if chunk.param_sync.buckets
                else torch.device("cpu")
            )
            for chunk in self._model_chunks
        ]
        for chunk in self._model_chunks:
            chunk.move_model_state("cpu", load_grad=False)
        self._inner_optimizer.offload_state_to_cpu()
        self._rollout_offloaded = True

    def load_from_rollout(self) -> None:
        """Restore training shards and state after colocated rollout sleeps."""
        if not self._rollout_offloaded:
            return
        for chunk, device in zip(
            self._model_chunks, self._pre_rollout_devices, strict=True
        ):
            chunk.move_model_state(device, load_grad=True)
        self._inner_optimizer.load_state_to_device()
        self._pre_rollout_devices = []
        self._rollout_offloaded = False

    def _require_training_resident(self, operation: str) -> None:
        if self._rollout_offloaded:
            raise RuntimeError(
                f"M-FSDP cannot {operation} while training state is offloaded for rollout."
            )


def build_mfsdp_stack(
    model_chunks: list[nn.Module],
    *,
    engine_cfg,
    ps,
    is_expert: ExpertClassifierFn | None = None,
    fsdp_unit_modules: tuple[type[nn.Module] | str, ...] | None = None,
    enable_fine_grained_param_gather_hook: bool = False,
    enable_fine_grained_param_gather_backward_hook: bool = False,
    fine_grained_recurse_module_types: tuple[type[nn.Module], ...] | None = None,
    suggested_communication_unit_size: int | None = None,
    optimizer_factory: OptimizerFactory | None = None,
    calculate_per_token_loss: bool = False,
):
    """Wrap chunks with the native M-FSDP path and build its local optimizer."""
    validate_mfsdp_config(
        engine_cfg, has_optimizer_factory=optimizer_factory is not None
    )
    opt = engine_cfg.optimizer
    classifier = is_expert or (lambda _name: False)
    offload_fraction = float(getattr(opt, "offload_fraction", 0.0) or 0.0)
    if not math.isfinite(offload_fraction) or not 0.0 <= offload_fraction <= 1.0:
        raise ValueError(
            f"M-FSDP offload_fraction must be in [0, 1], got {offload_fraction}."
        )
    config = build_mfsdp_config(
        opt, calculate_per_token_loss=calculate_per_token_loss
    )
    if offload_fraction == 1.0:
        config = replace(config, full_optimizer_offload=True)
    if (
        config.suggested_communication_unit_size is None
        and suggested_communication_unit_size is not None
    ):
        if suggested_communication_unit_size <= 0:
            raise ValueError("M-FSDP communication unit size must be positive.")
        config = replace(
            config,
            suggested_communication_unit_size=int(suggested_communication_unit_size),
        )
    config = _order_param_gathers_for_parallel_collectives(config, ps)
    groups = build_mfsdp_process_groups(ps)

    wrapped_chunks = []
    for chunk in model_chunks:
        _mark_mfsdp_parallel_attrs(
            chunk,
            classifier,
            tp_size=int(getattr(engine_cfg.parallel, "tp", 1) or 1),
            etp_size=int(getattr(engine_cfg.parallel, "etp", 1) or 1),
        )
        wrapped_chunks.append(
            fully_shard_model(
                chunk,
                groups=groups,
                config=config,
                is_expert=classifier,
                unit_modules=fsdp_unit_modules,
                enable_fine_grained_param_gather_hook=(
                    enable_fine_grained_param_gather_hook
                ),
                enable_fine_grained_param_gather_backward_hook=(
                    enable_fine_grained_param_gather_backward_hook
                ),
                fine_grained_recurse_module_types=fine_grained_recurse_module_types,
            )
        )

    params, param_groups, expert_params = _build_param_groups(
        wrapped_chunks,
        classifier=classifier,
        weight_decay=float(getattr(opt, "weight_decay", 0.01)),
        apply_wd_to_qk_layernorm=bool(getattr(opt, "apply_wd_to_qk_layernorm", False)),
    )
    cpu_group = None
    if offload_fraction > 0.0:
        optimizer_name = str(getattr(opt, "optimizer", "adam")).lower()
        if optimizer_name not in {"adam", "adamw"}:
            raise ValueError(
                "M-FSDP CPU optimizer offload supports AdamW only; "
                f"got {optimizer_name!r}."
            )
        if optimizer_factory is not None:
            raise ValueError(
                "M-FSDP CPU optimizer offload cannot preserve an injected "
                "optimizer_factory algorithm."
            )
        gpu_param_groups, cpu_param_groups = _split_param_groups_by_fraction(
            param_groups, offload_fraction
        )
        torch_optimizer = (
            _build_optimizer_algorithm(
                gpu_param_groups,
                opt,
                optimizer_factory=None,
                use_decoupled_grad=config.use_decoupled_grad,
            )
            if gpu_param_groups
            else _NullOptimizer()
        )
        cpu_group = _build_cpu_adam_group(
            cpu_param_groups, opt, bucket_size=config.bucket_size
        )
    else:
        torch_optimizer = _build_optimizer_algorithm(
            param_groups,
            opt,
            optimizer_factory=optimizer_factory,
            use_decoupled_grad=config.use_decoupled_grad,
        )

    standalone_optimizer = _StandaloneOptimizer(
        torch_optimizer,
        params,
        ps=ps,
        clip_grad=float(getattr(opt, "clip_grad", 1.0)),
        grad_norm_accum_dtype=_override(opt, "grad_norm_accum_dtype", "float32"),
        expert_params=expert_params,
        expert_grad_scale=(
            1.0
            if calculate_per_token_loss
            else (
                float(getattr(ps, "expert_dp_size", 1))
                / float(getattr(ps, "dp_cp_size", 1))
                if expert_params
                else 1.0
            )
        ),
        use_decoupled_grad=config.use_decoupled_grad,
        cpu_group=cpu_group,
    )
    optimizer = MFSdpOptimizer(standalone_optimizer, wrapped_chunks)
    return wrapped_chunks, optimizer


def _order_param_gathers_for_parallel_collectives(
    config: MFSDPConfig, ps: Any
) -> MFSDPConfig:
    """Avoid unordered async collectives across intersecting process groups.

    The standalone overlap pipeline has no dependency handshake with model-side
    TP/CP/EP/ETP collectives.  When parameter shards and model-parallel groups
    both span ranks, prefetching the next bucket can enqueue collectives on
    intersecting process groups in different orders.  Keep overlap for pure
    data parallelism, but make multidimensional compositions use ordered bucket
    gathers until that cross-group handshake exists.
    """
    if not config.overlap_param_gather:
        return config

    model_parallel_size = math.prod(
        max(int(getattr(ps, name, 1) or 1), 1)
        for name in ("tp_size", "cp_size", "ep_size", "etp_size")
    )
    sharded_group_size = max(
        int(getattr(ps, "dp_cp_size", 1) or 1),
        int(getattr(ps, "expert_dp_size", 1) or 1),
    )
    if model_parallel_size == 1 or sharded_group_size == 1:
        return config

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.warning(
            "Disabling M-FSDP parameter-gather overlap because parameter shards "
            "and model-parallel collectives span intersecting process groups."
        )
    return replace(config, overlap_param_gather=False)


def _split_param_groups_by_fraction(
    param_groups: list[dict[str, Any]], offload_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match HDO's deterministic leading-numel split without splitting a tensor."""
    total_numel = sum(param.numel() for group in param_groups for param in group["params"])
    cpu_target = int(total_numel * offload_fraction)
    cpu_numel = 0
    gpu_groups: list[dict[str, Any]] = []
    cpu_groups: list[dict[str, Any]] = []
    for group in param_groups:
        gpu_params: list[nn.Parameter] = []
        cpu_params: list[nn.Parameter] = []
        for param in group["params"]:
            if cpu_numel < cpu_target:
                cpu_params.append(param)
                cpu_numel += param.numel()
            else:
                gpu_params.append(param)
        for params, target in ((gpu_params, gpu_groups), (cpu_params, cpu_groups)):
            if params:
                copied = dict(group)
                copied["params"] = params
                target.append(copied)
    return gpu_groups, cpu_groups


def _build_cpu_adam_group(
    cpu_param_groups: list[dict[str, Any]], opt: Any, *, bucket_size: int | None
) -> CpuAdamGroup:
    beta1 = getattr(opt, "adam_beta1", None)
    beta2 = getattr(opt, "adam_beta2", None)
    eps = getattr(opt, "adam_eps", None)
    capacity = bucket_size or max(
        1,
        max(
            param.numel()
            for group in cpu_param_groups
            for param in group["params"]
        ),
    )
    return CpuAdamGroup(
        cpu_param_groups,
        lr=float(getattr(opt, "lr", 1.0e-4)),
        betas=(0.9 if beta1 is None else beta1, 0.999 if beta2 is None else beta2),
        eps=1.0e-8 if eps is None else eps,
        bucket_size=capacity,
    )


def _mark_mfsdp_parallel_attrs(
    model: nn.Module, classifier: ExpertClassifierFn, *, tp_size: int, etp_size: int
) -> None:
    """Compatibility entry point for the active metadata normalization pass."""
    annotate_parallel_parameters(model, classifier, tp_size=tp_size, etp_size=etp_size)


def build_mfsdp_training_optimizer(
    model_chunks: list[nn.Module],
    *,
    impl_cfg,
    ps,
    is_expert: ExpertClassifierFn | None = None,
    fsdp_unit_modules: tuple[type[nn.Module] | str, ...] | None = None,
    enable_fine_grained_param_gather_hook: bool = False,
    enable_fine_grained_param_gather_backward_hook: bool = False,
    fine_grained_recurse_module_types: tuple[type[nn.Module], ...] | None = None,
    suggested_communication_unit_size: int | None = None,
    optimizer_factory: OptimizerFactory | None = None,
    calculate_per_token_loss: bool = False,
):
    """Build M-FSDP with Adam by default or an injected optimizer algorithm."""
    opt = impl_cfg.optimizer_config
    if opt is None:
        opt = SimpleNamespace(
            optimizer="adam",
            lr=1e-4,
            min_lr=0.0,
            weight_decay=0.01,
            clip_grad=1.0,
            offload_fraction=0.0,
            adam_beta1=None,
            adam_beta2=None,
            adam_eps=None,
            override_optimizer_config={},
        )
    engine_cfg = SimpleNamespace(parallel=impl_cfg.parallel, optimizer=opt)
    model_chunks[:], optimizer = build_mfsdp_stack(
        model_chunks,
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=is_expert,
        fsdp_unit_modules=fsdp_unit_modules,
        enable_fine_grained_param_gather_hook=(
            enable_fine_grained_param_gather_hook
        ),
        enable_fine_grained_param_gather_backward_hook=(
            enable_fine_grained_param_gather_backward_hook
        ),
        fine_grained_recurse_module_types=fine_grained_recurse_module_types,
        suggested_communication_unit_size=suggested_communication_unit_size,
        optimizer_factory=optimizer_factory,
        calculate_per_token_loss=calculate_per_token_loss,
    )

    def finalize_grads(*, num_tokens: torch.Tensor | None = None) -> None:
        optimizer.finish_grad_sync(
            update_optimizer_grads=not calculate_per_token_loss
        )
        if calculate_per_token_loss:
            if num_tokens is None:
                raise ValueError(
                    "M-FSDP per-token loss requires the local token count."
                )
            _scale_gradients_by_global_tokens(model_chunks, num_tokens, ps)
            optimizer.update_optimizer_grads()

    return optimizer, finalize_grads


def _scale_gradients_by_global_tokens(
    model_chunks: list[nn.Module], num_tokens: torch.Tensor, ps: Any
) -> None:
    """Mirror MCore finalize_model_grads per-token normalization."""
    total_num_tokens = num_tokens.detach().to(dtype=torch.int64)
    if total_num_tokens.numel() != 1:
        raise ValueError("M-FSDP num_tokens must be a scalar tensor.")
    pp_group = getattr(ps, "pp_group", None)
    pp_global_ranks = getattr(ps, "pp_global_ranks", None)
    if pp_group is not None and pp_global_ranks is not None:
        dist.broadcast(total_num_tokens, src=pp_global_ranks[-1], group=pp_group)
    dp_cp_group = getattr(ps, "dp_cp_group", None)
    if dp_cp_group is not None:
        dist.all_reduce(total_num_tokens, group=dp_cp_group)
    scaling = total_num_tokens.clamp_min(1).to(dtype=torch.float32).reciprocal()
    for chunk in model_chunks:
        chunk.scale_gradients(scaling)


def _build_param_groups(
    model_chunks: Iterable[nn.Module],
    *,
    classifier: ExpertClassifierFn,
    weight_decay: float,
    apply_wd_to_qk_layernorm: bool,
) -> tuple[list[nn.Parameter], list[dict[str, Any]], list[nn.Parameter]]:
    params: list[nn.Parameter] = []
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    expert_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for chunk in model_chunks:
        named_params = (
            chunk.named_optimizer_parameters()
            if isinstance(chunk, MFSdpModule)
            else chunk.named_parameters()
        )
        for name, param in named_params:
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            params.append(param)
            normalized_name = name.removeprefix("module.")
            if classifier(normalized_name):
                expert_params.append(param)
            original_ndim = int(getattr(param, "_mfsdp_original_ndim", param.ndim))
            no_weight_decay = original_ndim == 1 or normalized_name.endswith(".bias")
            if apply_wd_to_qk_layernorm and any(
                marker in normalized_name
                for marker in ("q_layernorm.", "k_layernorm.", "q_norm.", "k_norm.")
            ):
                no_weight_decay = False
            (no_decay_params if no_weight_decay else decay_params).append(param)

    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append(
            {"params": decay_params, "weight_decay": weight_decay, "wd_mult": 1.0}
        )
    if no_decay_params:
        param_groups.append(
            {"params": no_decay_params, "weight_decay": 0.0, "wd_mult": 0.0}
        )
    if not param_groups:
        raise ValueError("M-FSDP found no trainable parameters.")
    return params, param_groups, expert_params


def _build_optimizer_algorithm(
    param_groups: list[dict[str, Any]],
    opt: Any,
    *,
    optimizer_factory: OptimizerFactory | None = None,
    use_decoupled_grad: bool = False,
) -> torch.optim.Optimizer:
    return build_optimizer(
        param_groups,
        opt,
        optimizer_factory=optimizer_factory,
        use_decoupled_grad=use_decoupled_grad,
    )
