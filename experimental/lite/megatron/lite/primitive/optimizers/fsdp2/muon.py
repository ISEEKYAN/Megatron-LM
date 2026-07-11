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
* orthogonalization is *not* element-wise, so the momentum is gathered into the
  full matrix **one parameter at a time** (bounded peak memory), the same
  Newton-Schulz runs on that full matrix, then the orthogonalized update is
  resharded back to the local shard;
* the gather/reshard reuse the parameter's own DTensor mesh + placement, so
  dense params gather over the dense DP/CP mesh and expert params over the
  expert-DP mesh with no new transport API, and the next forward keeps using
  FSDP2's ordinary parameter all-gather (we never gather the parameter itself).

Non-matrix / embedding / output parameters fall back to :class:`FP32AdamW`; the
two children are composed under a single :class:`FSDP2Optimizer` facade so the
runtime sees one ``step``/``grad-norm``/``zero``/``state`` surface.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from itertools import chain, cycle, islice, repeat
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.primitive.optimizers.fsdp2.adamw import (
    ChainedOptimizer,
    copy_local_tensor_to_param_,
    dtensor_from_local,
    fsdp2_model_param_dtype,
    is_dtensor_like,
    to_local_tensor,
)

# Newton-Schulz quintic-iteration coefficients, mirrored verbatim from the
# pinned Megatron ``emerging_optimizers`` reference (rounded to fp32). Values are
# data selected by ``muon_coefficient_type``; see ``newton_schulz_orthogonalize``.
_NS_COEFFICIENT_SETS: dict[str, list[tuple[float, float, float]]] = {
    "simple": [(3.4445, -4.7750, 2.0315)],
    "quintic": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
    "cans": [
        (8.4703, -25.1081, 18.6293),
        (4.1828, -3.1087, 0.5806),
        (3.9619, -2.9541, 0.5630),
        (3.2866, -2.4647, 0.5074),
        (2.2737, -1.6447, 0.4162),
    ],
    "aol": [
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
    "deepseekv4": [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2,
}
# These coefficient sets keep applying the last tuple after they are exhausted;
# the rest cycle from the beginning (matches the reference iteration mode).
_NS_REPEAT_LAST_TYPES = frozenset({"polar_express", "cans", "deepseekv4"})


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
    """Newton-Schulz orthogonalization of a full 2D matrix (Megatron semantics).

    Mirrors ``emerging_optimizers`` ``newton_schulz`` for the non-distributed
    path: whiten on the smaller dimension, normalize to spectral norm <= 1, then
    iterate the quintic polynomial. Under ``"medium"`` fp32 matmul precision the
    iteration runs in bf16 (matching the reference's explicit bf16 cast) since
    PyTorch has no fp32-I/O bf16-compute kernel for that precision.
    """
    if x.ndim != 2:
        raise ValueError(f"newton_schulz_orthogonalize expects a 2D matrix, got {x.ndim}D")
    if x.dtype != torch.float32:
        raise TypeError(f"newton_schulz_orthogonalize requires float32 input, got {x.dtype}")
    if steps < 1:
        raise ValueError(f"num_ns_steps must be at least 1, got {steps}")

    coefficient_sets = _NS_COEFFICIENT_SETS.get(coefficient_type)
    if coefficient_sets is None:
        raise ValueError(f"Unsupported muon coefficient type: {coefficient_type!r}")

    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)

    if coefficient_type in _NS_REPEAT_LAST_TYPES:
        coeff_iter = islice(chain(coefficient_sets, repeat(coefficient_sets[-1])), steps)
    else:
        coeff_iter = islice(cycle(coefficient_sets), steps)

    if torch.get_float32_matmul_precision() == "medium":
        X = X.to(torch.bfloat16)

    for a, b, c in coeff_iter:
        A = X @ X.mT
        B = torch.addmm(A, A, A, alpha=c, beta=b)
        X = torch.addmm(X, B, X, alpha=1.0, beta=a)

    X = X.to(torch.float32)
    if transpose:
        X = X.mT
    return X


def muon_update_scale(size_out: int, size_in: int, mode: str = "spectral") -> float:
    """Update scale factor for the orthogonalized update (Megatron semantics)."""
    if mode == "spectral":
        return float(max(size_out, size_in)) ** 0.5
    if mode == "shape_scaling":
        return max(1.0, size_out / size_in) ** 0.5
    if mode == "unit_rms_norm":
        return (size_out / size_in) ** 0.5
    raise ValueError(f"Invalid muon update scale mode: {mode!r}")


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

                # (4) bounded full-matrix gather -> Newton-Schulz -> local reshard.
                full_update = self._gather_full(update_local, master)
                with _fp32_matmul_precision(self.fp32_matmul_prec):
                    orth_full = self._orthogonalize(param, full_update)
                orth_local = self._reshard_local(orth_full, master)

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

    def _orthogonalize(self, param: nn.Parameter, full: torch.Tensor) -> torch.Tensor:
        if self.split_qkv and getattr(param, "is_qkv", False):
            split_shapes = getattr(param, "qkv_split_shapes", None) or self.qkv_split_shapes
            if not split_shapes:
                raise RuntimeError("Muon QKV split requested but qkv_split_shapes is unset.")
            return self._orthogonalize_qkv(full, split_shapes)
        return self._scaled_orthogonalize(full)

    def _scaled_orthogonalize(self, grad: torch.Tensor) -> torch.Tensor:
        orth = newton_schulz_orthogonalize(grad, self.num_ns_steps, self.coefficient_type)
        scale = muon_update_scale(grad.size(-2), grad.size(-1), self.scale_mode)
        return orth * scale * self.extra_scale_factor

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
