# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Self-contained Megatron-FSDP construction for Megatron Lite."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.optimizers.mfsdp.config import (
    annotate_parallel_parameters,
    build_mfsdp_config,
    build_mfsdp_process_groups,
    validate_mfsdp_config,
)
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
from megatron.lite.primitive.optimizers.mfsdp.wrapper import (
    MFSdpModule,
    mark_optimizer_built,
)

ExpertClassifierFn = Callable[[str], bool]


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
        self._expert_grads_scaled = False
        self._tp_replicated_grads_synced = False
        self._offloaded_state_devices: dict[tuple[int, str], torch.device] = {}

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._expert_grads_scaled = False
        self._tp_replicated_grads_synced = False

    def step(self) -> tuple[bool, float, int]:
        self._sync_tp_replicated_grads_once()
        self._scale_expert_grads_once()
        grad_norm = self.clip_grad_norm()
        if not math.isfinite(grad_norm):
            return False, grad_norm, 0
        self.optimizer.step()
        return True, grad_norm, 0

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
        _sum_if_distributed(
            dense_sq,
            dense_dp_group,
        )
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
                    if param.grad is not None:
                        param.grad.mul_(coefficient)
        return float(total_norm.float().item())

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def offload_state_to_cpu(self) -> None:
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
                if param.grad is not None:
                    param.grad.mul_(self.expert_grad_scale)
        self._expert_grads_scaled = True

    def _sync_tp_replicated_grads_once(self) -> None:
        if self._tp_replicated_grads_synced:
            return
        group = getattr(self.ps, "tp_group", None)
        for param in self.tp_replicated_params:
            if param.grad is not None:
                _all_reduce_grad_if_distributed(
                    param.grad,
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


def _all_reduce_grad_if_distributed(
    grad: torch.Tensor,
    group: dist.ProcessGroup | None,
    *,
    average: bool = False,
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
        self.grad_sync_enabled = False
        self._inner_optimizer.zero_grad()
        for chunk in self._model_chunks:
            chunk.zero_grad_buffer()

    def finish_grad_sync(self) -> None:
        for chunk in self._model_chunks:
            chunk.finish_grad_sync()

    def clip_grad_norm(self) -> float:
        return self._inner_optimizer.clip_grad_norm()

    def step(self) -> tuple[bool, float, int]:
        result = self._inner_optimizer.step()
        for chunk in self._model_chunks:
            chunk.param_sync.release_all()
        self.grad_sync_enabled = False
        return result

    def state_dict(self) -> dict[str, Any]:
        return self._inner_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._inner_optimizer.load_state_dict(state_dict)

    def offload_state_to_cpu(self) -> None:
        self._inner_optimizer.offload_state_to_cpu()

    def load_state_to_device(self) -> None:
        self._inner_optimizer.load_state_to_device()


def build_mfsdp_stack(
    model_chunks: list[nn.Module],
    *,
    engine_cfg,
    ps,
    is_expert: ExpertClassifierFn | None = None,
    fsdp_unit_modules: tuple[type[nn.Module] | str, ...] | None = None,
    optimizer_factory: OptimizerFactory | None = None,
):
    """Wrap chunks with the native M-FSDP path and build its local optimizer."""
    validate_mfsdp_config(
        engine_cfg,
        has_optimizer_factory=optimizer_factory is not None,
    )
    opt = engine_cfg.optimizer
    classifier = is_expert or (lambda _name: False)
    config = build_mfsdp_config(opt)
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
            )
        )

    params, param_groups, expert_params = _build_param_groups(
        wrapped_chunks,
        classifier=classifier,
        weight_decay=float(getattr(opt, "weight_decay", 0.01)),
        apply_wd_to_qk_layernorm=bool(getattr(opt, "apply_wd_to_qk_layernorm", False)),
    )
    torch_optimizer = _build_optimizer_algorithm(
        param_groups,
        opt,
        optimizer_factory=optimizer_factory,
    )
    standalone_optimizer = _StandaloneOptimizer(
        torch_optimizer,
        params,
        ps=ps,
        clip_grad=float(getattr(opt, "clip_grad", 1.0)),
        grad_norm_accum_dtype=_override(opt, "grad_norm_accum_dtype", "float32"),
        expert_params=expert_params,
        expert_grad_scale=(
            float(getattr(ps, "expert_dp_size", 1))
            / float(getattr(ps, "dp_cp_size", 1))
            if expert_params
            else 1.0
        ),
    )
    for chunk in wrapped_chunks:
        mark_optimizer_built(chunk)
    optimizer = MFSdpOptimizer(standalone_optimizer, wrapped_chunks)
    return wrapped_chunks, optimizer


def _mark_mfsdp_parallel_attrs(
    model: nn.Module,
    classifier: ExpertClassifierFn,
    *,
    tp_size: int,
    etp_size: int,
) -> None:
    """Compatibility entry point for the active metadata normalization pass."""
    annotate_parallel_parameters(
        model,
        classifier,
        tp_size=tp_size,
        etp_size=etp_size,
    )


def build_mfsdp_training_optimizer(
    model_chunks: list[nn.Module],
    *,
    impl_cfg,
    ps,
    is_expert: ExpertClassifierFn | None = None,
    fsdp_unit_modules: tuple[type[nn.Module] | str, ...] | None = None,
    optimizer_factory: OptimizerFactory | None = None,
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
            offload_fraction=None,
            adam_beta1=None,
            adam_beta2=None,
            adam_eps=None,
            override_optimizer_config={},
        )
    engine_cfg = SimpleNamespace(
        parallel=impl_cfg.parallel,
        optimizer=opt,
    )
    model_chunks[:], optimizer = build_mfsdp_stack(
        model_chunks,
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=is_expert,
        fsdp_unit_modules=fsdp_unit_modules,
        optimizer_factory=optimizer_factory,
    )

    def finalize_grads() -> None:
        optimizer.finish_grad_sync()

    return optimizer, finalize_grads


def _build_param_groups(
    model_chunks: Iterable[nn.Module],
    *,
    classifier: ExpertClassifierFn,
    weight_decay: float,
    apply_wd_to_qk_layernorm: bool,
) -> tuple[
    list[nn.Parameter],
    list[dict[str, Any]],
    list[nn.Parameter],
]:
    params: list[nn.Parameter] = []
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    expert_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
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
) -> torch.optim.Optimizer:
    return build_optimizer(
        param_groups,
        opt,
        optimizer_factory=optimizer_factory,
    )
