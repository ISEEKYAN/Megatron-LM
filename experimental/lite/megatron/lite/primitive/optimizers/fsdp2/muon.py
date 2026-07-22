# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Muon optimizer lowering for the FSDP2 primitive.

This is the FSDP2-native counterpart of Megatron's ``TensorParallelMuon``: it
reuses the same Megatron Muon configuration semantics (momentum EMA, decoupled
weight decay, Newton-Schulz orthogonalization, spectral update scaling, QKV
split) but lowers them onto FSDP2's per-parameter DTensor shards instead of the
Megatron-Core distributed optimizer.

Sharding contract (per 2D Muon-managed parameter):

* momentum EMA, weight decay and the final parameter update are element-wise on
  the **local shard** — sharding commutes with them, so no communication;
* orthogonalization is *not* element-wise, so the momentum update is passed to
  ``emerging_optimizers.newton_schulz_tp`` in **distributed** mode: the FSDP2
  DTensor mesh supplies ``tp_group``, the shard placement supplies
  ``partition_dim``, and NS work is split across ranks via all-reduce rather
  than gathering the full matrix and redundantly re-running NS on every rank;
* fused QKV weights that are not evenly split across complete query groups fall
  back to a bounded gather → per-head ``newton_schulz`` → reshard path (the
  gather is required for correct Q/K/V separation, not for NS redundancy).

Non-matrix / embedding / output parameters fall back to :class:`FP32AdamW`; the
two children are composed under a single :class:`FSDP2Optimizer` facade so the
runtime sees one ``step``/``grad-norm``/``zero``/``state`` surface.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.optimizers.fsdp2.adamw import (
    ChainedOptimizer,
    copy_local_tensor_to_param_,
    dtensor_from_local,
    fsdp2_model_param_dtype,
    is_dtensor_like,
    to_local_tensor,
)

_EMERGING_OPT_SITE = os.environ.get("EMERGING_OPT_SITE")
if _EMERGING_OPT_SITE and _EMERGING_OPT_SITE not in sys.path:
    sys.path.insert(0, _EMERGING_OPT_SITE)

from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
    newton_schulz,
    newton_schulz_tp,
)


@contextlib.contextmanager
def _fp32_matmul_precision(precision: str):
    """Temporarily set the global fp32 matmul precision (restored on exit)."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def newton_schulz_orthogonalize(
    x: torch.Tensor, steps: int, coefficient_type: str = "quintic", eps: float = 1e-7
) -> torch.Tensor:
    """Non-distributed Newton-Schulz orthogonalization (``emerging_optimizers``)."""
    del eps  # emerging_optimizers uses a fixed internal default.
    if x.ndim != 2:
        raise ValueError(f"newton_schulz_orthogonalize expects a 2D matrix, got {x.ndim}D")
    if x.dtype != torch.float32:
        raise TypeError(f"newton_schulz_orthogonalize requires float32 input, got {x.dtype}")
    if steps < 1:
        raise ValueError(f"num_ns_steps must be at least 1, got {steps}")
    return newton_schulz(x, steps=steps, coefficient_type=coefficient_type)


def muon_update_scale(size_out: int, size_in: int, mode: str = "spectral") -> float:
    """Update scale factor for the orthogonalized update (Megatron semantics)."""
    return get_muon_scale_factor(size_out, size_in, mode=mode)  # type: ignore[arg-type]


def _placement_name(placement: Any) -> str:
    return type(placement).__name__


def _shard_context(
    master: torch.Tensor,
) -> tuple[dist.ProcessGroup | None, int | None]:
    """Return (tp_group, partition_dim) for a DTensor master shard, else (None, None)."""
    if not is_dtensor_like(master):
        return None, None
    for mesh_dim, placement in enumerate(master.placements):
        if _placement_name(placement) == "Shard":
            return master.device_mesh.get_group(mesh_dim), int(placement.dim)
    return None, None


def _global_matrix_size(
    local: torch.Tensor,
    *,
    tp_group: dist.ProcessGroup | None,
    partition_dim: int | None,
) -> tuple[int, int]:
    size_out, size_in = int(local.size(-2)), int(local.size(-1))
    if tp_group is not None and partition_dim is not None and dist.get_world_size(tp_group) > 1:
        if partition_dim == 0:
            size_out *= dist.get_world_size(tp_group)
        elif partition_dim == 1:
            size_in *= dist.get_world_size(tp_group)
    return size_out, size_in


def _should_use_distributed_ns(
    local: torch.Tensor,
    *,
    tp_group: dist.ProcessGroup | None,
    partition_dim: int | None,
) -> bool:
    """Return True when ``newton_schulz_tp`` distributed mode matches full-matrix NS.

    FSDP2 row-shards standard ``[out, in]`` weights (``out >= in``) along dim 0.
    Column-shard and Q/K/V head slices use the bounded gather fallback so the
    unsharded reference stays bitwise-aligned in unit tests.
    """
    if tp_group is None or partition_dim is None or partition_dim != 0:
        return False
    if dist.get_world_size(tp_group) <= 1:
        return False
    size_out, size_in = _global_matrix_size(
        local, tp_group=tp_group, partition_dim=partition_dim
    )
    return size_out >= size_in


class FP32Muon:
    """Muon with FP32 master params over FSDP2 (possibly DTensor) shards.

    State per parameter mirrors :class:`FP32AdamW` (``master_param`` +
    ``momentum_buffer`` + ``step``) so the shared offload / checkpoint helpers in
    ``fsdp2.state`` and ``fsdp2.optimizer`` treat both children uniformly.
    """

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        nesterov: bool = False,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = True,
        num_ns_steps: int = 5,
        coefficient_type: str = "quintic",
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        fp32_matmul_prec: str = "medium",
        qkv_split_shapes: list[int] | None = None,
        model_param_dtypes: dict[int, torch.dtype] | None = None,
    ):
        self.param_groups = param_groups
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.use_decoupled_weight_decay = bool(use_decoupled_weight_decay)
        self.split_qkv = bool(split_qkv)
        self.num_ns_steps = int(num_ns_steps)
        self.coefficient_type = str(coefficient_type)
        self.scale_mode = str(scale_mode)
        self.extra_scale_factor = float(extra_scale_factor)
        self.fp32_matmul_prec = str(fp32_matmul_prec)
        self.qkv_split_shapes = list(qkv_split_shapes) if qkv_split_shapes is not None else None
        self._model_dtype_for_param = dict(model_param_dtypes or {})

        self.params: list[nn.Parameter] = []
        self.state: dict[nn.Parameter, dict[str, Any]] = {}
        for group in self.param_groups:
            group.setdefault("lr", self.lr)
            group.setdefault("weight_decay", self.weight_decay)
            group.setdefault("momentum", self.momentum)
            for param in group["params"]:
                self.params.append(param)
                master = self._init_master_param(param)
                self.state[param] = {
                    "master_param": master,
                    "momentum_buffer": torch.zeros_like(master),
                    "step": 0,
                }

    def _init_master_param(self, param: nn.Parameter) -> torch.Tensor:
        if param.dtype is torch.float32 and fsdp2_model_param_dtype(param) is None:
            return param.detach()
        return param.detach().to(dtype=torch.float32).clone()

    def _model_param_dtype(self, param: nn.Parameter) -> torch.dtype | None:
        return self._model_dtype_for_param.get(id(param)) or fsdp2_model_param_dtype(param)

    @property
    def defaults(self) -> dict[str, Any]:  # torch.optim-compatible shim
        return {"lr": self.lr, "weight_decay": self.weight_decay, "momentum": self.momentum}

    def zero_grad(self, *args, **kwargs) -> None:
        set_to_none = kwargs.get("set_to_none", False)
        if args:
            set_to_none = bool(args[0])
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            group_lr = float(group.get("lr", self.lr))
            group_wd = float(group.get("weight_decay", self.weight_decay))
            group_momentum = float(group.get("momentum", self.momentum))
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                master = state["master_param"]
                buffer = state["momentum_buffer"]
                master_local = to_local_tensor(master)
                buffer_local = to_local_tensor(buffer)
                grad_local = to_local_tensor(grad).detach().to(dtype=torch.float32)

                # (1) weight decay -- decoupled acts on the master, l2 on the grad.
                if group_wd != 0.0:
                    if self.use_decoupled_weight_decay:
                        master_local.mul_(1.0 - group_lr * group_wd)
                    else:
                        grad_local = grad_local.add(master_local, alpha=group_wd)

                # (2) momentum EMA and (3) optional Nesterov look-ahead (element-wise).
                buffer_local.lerp_(grad_local, 1.0 - group_momentum)
                if self.nesterov:
                    update_local = grad_local.lerp(buffer_local, group_momentum)
                else:
                    update_local = buffer_local

                # (4) distributed Newton-Schulz on the local shard (no redundant gather).
                with _fp32_matmul_precision(self.fp32_matmul_prec):
                    orth_local = self._orthogonalize(param, update_local, master)

                # (5) apply the update to the master shard, then mirror to the param.
                master_local.add_(orth_local, alpha=-group_lr)
                self._copy_master_to_param(param, master)
                state["step"] = int(state["step"]) + 1

    def _gather_full(self, update_local: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        if not is_dtensor_like(master):
            return update_local
        update = dtensor_from_local(
            update_local,
            master.device_mesh,
            master.placements,
            shape=master.shape,
            stride=master.stride(),
        )
        return update.full_tensor()

    def _reshard_local(self, orth_full: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        if not is_dtensor_like(master):
            return orth_full
        from torch.distributed.tensor import distribute_tensor

        orth = distribute_tensor(orth_full, master.device_mesh, master.placements)
        return orth.to_local()

    def _orthogonalize(
        self, param: nn.Parameter, update_local: torch.Tensor, master: torch.Tensor
    ) -> torch.Tensor:
        tp_group, partition_dim = _shard_context(master)
        if self.split_qkv and getattr(param, "is_qkv", False):
            split_shapes = getattr(param, "qkv_split_shapes", None) or self.qkv_split_shapes
            if not split_shapes:
                raise RuntimeError("Muon QKV split requested but qkv_split_shapes is unset.")
            full_update = self._gather_full(update_local, master)
            return self._reshard_local(self._orthogonalize_qkv(full_update, split_shapes), master)
        if _should_use_distributed_ns(
            update_local, tp_group=tp_group, partition_dim=partition_dim
        ):
            return self._scaled_orthogonalize_distributed(
                update_local, tp_group=tp_group, partition_dim=partition_dim
            )
        full_update = self._gather_full(update_local, master)
        return self._reshard_local(
            self._scaled_orthogonalize(full_update), master
        )

    def _scaled_orthogonalize_distributed(
        self,
        grad: torch.Tensor,
        *,
        tp_group: dist.ProcessGroup | None,
        partition_dim: int | None,
    ) -> torch.Tensor:
        if partition_dim is None:
            orth = newton_schulz_orthogonalize(grad, self.num_ns_steps, self.coefficient_type)
        else:
            if tp_group is None:
                raise RuntimeError("DTensor shard placement requires a process group for Muon NS.")
            orth = newton_schulz_tp(
                grad,
                steps=self.num_ns_steps,
                coefficient_type=self.coefficient_type,  # type: ignore[arg-type]
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="distributed",
            )
        size_out, size_in = _global_matrix_size(
            grad, tp_group=tp_group, partition_dim=partition_dim
        )
        scale = muon_update_scale(size_out, size_in, self.scale_mode)
        return orth * scale * self.extra_scale_factor

    def _scaled_orthogonalize(self, grad: torch.Tensor) -> torch.Tensor:
        return self._scaled_orthogonalize_distributed(grad, tp_group=None, partition_dim=None)

    def _orthogonalize_qkv(self, grad: torch.Tensor, split_shapes: list[int]) -> torch.Tensor:
        grad_shape = grad.shape
        qkv_split_dim = sum(split_shapes)
        if grad_shape[0] % qkv_split_dim != 0:
            raise RuntimeError(
                f"Muon QKV split shape mismatch: grad_shape={tuple(grad_shape)}, "
                f"split_shapes={split_shapes}"
            )
        num_query_groups = grad_shape[0] // qkv_split_dim
        qkv_grads = torch.split(
            grad.view(num_query_groups, qkv_split_dim, -1), split_shapes, dim=1
        )
        qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]
        qkv_orth = [
            self._scaled_orthogonalize(g).view(num_query_groups, -1, grad_shape[-1])
            for g in qkv_grads
        ]
        return torch.cat(qkv_orth, dim=1).view(grad_shape)

    def _copy_master_to_param(self, param: nn.Parameter, master: torch.Tensor) -> None:
        model_dtype = self._model_param_dtype(param)
        source = master
        if model_dtype is not None:
            source = master.to(dtype=model_dtype).to(dtype=master.dtype)
        copy_local_tensor_to_param_(param, to_local_tensor(source))

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "fp32_muon",
            "master_params": [self.state[param]["master_param"] for param in self.params],
            "momentum_buffers": [self.state[param]["momentum_buffer"] for param in self.params],
            "steps": [int(self.state[param]["step"]) for param in self.params],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("type") != "fp32_muon":
            raise ValueError("Invalid FP32 Muon state_dict.")
        for target_name, key in (
            ("master_params", "master_param"),
            ("momentum_buffers", "momentum_buffer"),
        ):
            loaded = state_dict.get(target_name)
            if not isinstance(loaded, list) or len(loaded) != len(self.params):
                raise ValueError(f"Invalid FP32 Muon {target_name} state.")
            for param, src in zip(self.params, loaded, strict=True):
                target_local = to_local_tensor(self.state[param][key])
                src_local = to_local_tensor(src).to(
                    device=target_local.device, dtype=target_local.dtype
                )
                target_local.copy_(src_local)
        loaded_steps = state_dict.get("steps")
        if loaded_steps is not None:
            if not isinstance(loaded_steps, list) or len(loaded_steps) != len(self.params):
                raise ValueError("Invalid FP32 Muon steps state.")
            for param, step in zip(self.params, loaded_steps, strict=True):
                self.state[param]["step"] = int(step)
        for param in self.params:
            self._copy_master_to_param(param, self.state[param]["master_param"])


def split_muon_and_fallback_params(
    model_chunks: Iterable[nn.Module],
) -> tuple[list[nn.Parameter], list[nn.Parameter], dict[int, str]]:
    """Partition trainable params into Muon-managed matrices and Adam fallback.

    Routing follows the ``is_managed_by_layer_wise_optimizer`` tag attached by
    ``muon_routing.tag_muon_parameter_metadata`` before FSDP2 wrapping (the tag
    survives ``fully_shard`` via ``FSDP2Config.preserve_param_attrs``).
    """
    muon_params: list[nn.Parameter] = []
    fallback_params: list[nn.Parameter] = []
    param_names: dict[int, str] = {}
    seen: set[int] = set()
    for chunk_idx, chunk in enumerate(model_chunks):
        for name, param in chunk.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            param_names[id(param)] = f"chunk{chunk_idx}.{name}"
            if getattr(param, "is_managed_by_layer_wise_optimizer", False):
                muon_params.append(param)
            else:
                fallback_params.append(param)
    return muon_params, fallback_params, param_names


def build_fp32_muon_child(
    muon_params: list[nn.Parameter],
    opt,
    *,
    model_param_dtypes: dict[int, torch.dtype] | None = None,
) -> FP32Muon:
    """Build the :class:`FP32Muon` child from the shared OptimizerConfig-like object."""
    decoupled = getattr(opt, "decoupled_weight_decay", None)
    return FP32Muon(
        [{"params": muon_params, "weight_decay": float(getattr(opt, "weight_decay", 0.01))}],
        lr=float(getattr(opt, "lr", 1.0e-4)),
        momentum=float(getattr(opt, "muon_momentum", 0.95)),
        weight_decay=float(getattr(opt, "weight_decay", 0.01)),
        nesterov=bool(getattr(opt, "muon_nesterov", False)),
        use_decoupled_weight_decay=True if decoupled is None else bool(decoupled),
        split_qkv=bool(getattr(opt, "muon_split_qkv", True)),
        num_ns_steps=int(getattr(opt, "muon_num_ns_steps", 5)),
        coefficient_type=str(getattr(opt, "muon_coefficient_type", "quintic")),
        scale_mode=str(getattr(opt, "muon_scale_mode", "spectral")),
        extra_scale_factor=float(getattr(opt, "muon_extra_scale_factor", 1.0)),
        fp32_matmul_prec=str(getattr(opt, "muon_fp32_matmul_prec", "medium")),
        model_param_dtypes=model_param_dtypes,
    )


def build_muon_chained_optimizer(
    muon_child: FP32Muon,
    fallback_optimizer: Any | None,
) -> Any:
    """Compose the Muon child with its Adam fallback under one facade."""
    if fallback_optimizer is None:
        return ChainedOptimizer([muon_child])
    return ChainedOptimizer([muon_child, fallback_optimizer])


__all__ = [
    "FP32Muon",
    "build_fp32_muon_child",
    "build_muon_chained_optimizer",
    "muon_update_scale",
    "newton_schulz_orthogonalize",
    "split_muon_and_fallback_params",
]
