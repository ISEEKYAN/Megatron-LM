# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LoRA helpers for Megatron Lite native model implementations.

This module is intentionally narrow: it supports the Qwen3-MoE lite path's
Megatron-style sharded linear surfaces, not arbitrary PEFT injection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_TARGET_MODULES = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
_TARGET_ALIASES = {
    "qkv": "linear_qkv",
    "proj": "linear_proj",
    "fc1": "linear_fc1",
    "fc2": "linear_fc2",
}


def lora_scaling(rank: int, alpha: int | None, use_rslora: bool = False) -> float:
    """LoRA scaling factor applied to ``B @ A``.

    Standard LoRA uses ``alpha / rank``; rsLoRA (Kalajdzievski, 2023) uses
    ``alpha / sqrt(rank)`` so the effective update magnitude stays rank-stable,
    enabling learning-rate reuse across ranks.
    """
    numerator = float(rank if alpha is None else alpha)
    return numerator / (math.sqrt(rank) if use_rslora else float(rank))


def olora_tail_factors(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """OLoRA-tail init factors ``(B0, A0)`` from a frozen base weight.

    Per arXiv:2606.02437 (§4.1.2): let ``W0 = U Σ Vᵀ``; OLoRA-tail uses the singular
    vectors of the **smallest** ``rank`` singular values (the minor / "tail" subspace)
    with **NO** singular-value scaling: ``B0 = U₋ᵣ``, ``A0 = V₋ᵣᵀ``. The minor subspace
    spans directions that are comparatively inert in the pretrained model, so the early
    on-policy update stays inside the RL "KL leash"; dropping Σ (vs MiLoRA's
    ``U₋ᵣΣ₋ᵣ^½`` / ``Σ₋ᵣ^½V₋ᵣᵀ``) avoids amplifying that early update.

    ``weight`` is ``[out, in]``; returns ``B0 [out, rank]``, ``A0 [rank, in]``.
    SVD is computed in float32 for stability regardless of the weight dtype.
    """
    w = weight.detach().to(torch.float32)
    if w.dim() != 2:
        raise ValueError(f"olora_tail_factors expects a 2D weight, got shape {tuple(w.shape)}.")
    k = min(w.shape)
    if rank > k:
        raise ValueError(f"OLoRA-tail rank {rank} exceeds min(out, in) = {k}.")
    # torch.linalg.svd returns S in DESCENDING order -> smallest r are the last r.
    U, _S, Vh = torch.linalg.svd(w, full_matrices=False)
    B0 = U[:, -rank:].contiguous()  # [out, rank]  left vectors of the smallest r values
    A0 = Vh[-rank:, :].contiguous()  # [rank, in]   right vectors (V₋ᵣᵀ)
    return B0, A0


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 0
    alpha: int | None = None
    dropout: float = 0.0
    use_rslora: bool = False
    init: str = "default"  # "default" = kaiming(A)+zeros(B); "olora_tail" = minor-SVD init (post-load)
    target_modules: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_TARGET_MODULES)

    @property
    def enabled(self) -> bool:
        return self.rank > 0

    @property
    def olora_tail(self) -> bool:
        return self.init == "olora_tail"

    @property
    def scale(self) -> float:
        return lora_scaling(self.rank, self.alpha, self.use_rslora)

    def targets(self) -> set[str]:
        out = set()
        for target in self.target_modules:
            out.add(_TARGET_ALIASES.get(target, target))
        return out

    def targets_module(self, name: str) -> bool:
        canonical = _TARGET_ALIASES.get(name, name)
        return canonical in self.targets()


def normalize_lora_config(config: LoraConfig | dict[str, Any] | None) -> LoraConfig:
    if config is None:
        return LoraConfig()
    if isinstance(config, LoraConfig):
        return config
    if not isinstance(config, dict):
        raise TypeError(f"LoRA config must be LoraConfig, dict, or None, got {type(config)!r}.")
    values = dict(config)
    enabled = values.pop("enabled", None)
    if enabled is False:
        values["rank"] = 0
    if "targets" in values and "target_modules" not in values:
        values["target_modules"] = values.pop("targets")
    else:
        values.pop("targets", None)
    if "target_modules" in values and not isinstance(values["target_modules"], tuple):
        values["target_modules"] = tuple(values["target_modules"])
    return LoraConfig(**values)


def freeze_non_lora_params(model: nn.Module) -> dict[str, int]:
    """Freeze base parameters and leave adapter parameters trainable."""

    lora_tensors = 0
    lora_numel = 0
    frozen_tensors = 0
    frozen_numel = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "adapter" in name.lower():
            param.requires_grad_(True)
            lora_tensors += 1
            lora_numel += param.numel()
        else:
            param.requires_grad_(False)
            frozen_tensors += 1
            frozen_numel += param.numel()
    return {
        "lora_tensors": lora_tensors,
        "lora_numel": lora_numel,
        "frozen_tensors": frozen_tensors,
        "frozen_numel": frozen_numel,
    }


def apply_olora_tail_init(model: nn.Module) -> dict[str, int]:
    """Apply OLoRA-tail init to every mlite LoRA adapter found under ``model``.

    Pairs each adapter with its frozen base weight by the mlite attribute convention:
    attention ``qkv_lora``/``proj_lora`` ← ``qkv.linear.weight``/``proj.linear.weight``;
    expert ``fc1_lora``/``fc2_lora`` ← the GroupedLinear per-expert weights ``weight{e}``.
    Must run AFTER base weights load and BEFORE the optimizer captures params (mlite's
    fsdp2 ``post_model_load_hook``). tp=1 only — see ``LinearLoRA.olora_tail_init_``.
    The expert path is best-effort: if the GroupedLinear weight layout is unrecognized
    it is skipped (those adapters keep the standard zero-delta init, which also preserves
    the layer's output at init, so mixing the two inits is safe).
    """
    n_attn = n_expert = 0
    for module in model.modules():
        qkv_lora = getattr(module, "qkv_lora", None)
        if qkv_lora is not None and getattr(module, "qkv", None) is not None:
            qkv_lora.olora_tail_init_(module.qkv.linear.weight)
            n_attn += 1
        proj_lora = getattr(module, "proj_lora", None)
        if proj_lora is not None and getattr(module, "proj", None) is not None:
            proj_lora.olora_tail_init_(module.proj.linear.weight)
            n_attn += 1
        for lora_attr, base_attr in (("fc1_lora", "fc1"), ("fc2_lora", "fc2")):
            adapter = getattr(module, lora_attr, None)
            base = getattr(module, base_attr, None)
            n_local = int(getattr(module, "num_local_experts", 0) or 0)
            if adapter is None or base is None or n_local <= 0:
                continue
            weights = []
            for e in range(n_local):
                w = getattr(base, f"weight{e}", None)
                if w is None:
                    weights = []
                    break
                weights.append(w)
            if weights:
                adapter.olora_tail_init_(weights)
                n_expert += 1
    return {"olora_attn_adapters": n_attn, "olora_expert_adapters": n_expert}


def trainable_param_stats(model: nn.Module) -> dict[str, int]:
    tensors = 0
    numel = 0
    for param in model.parameters():
        if param.requires_grad:
            tensors += 1
            numel += param.numel()
    return {"trainable_tensors": tensors, "trainable_numel": numel}


def _gather_sequence_parallel(x: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _AllGatherSequence.apply(x, group)


def _reduce_scatter_sequence_parallel(x: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _ReduceScatterSequence.apply(x, group)


def _scatter_sequence_parallel(x: torch.Tensor, group, group_rank: int) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _ScatterSequence.apply(x, group, group_rank)


def _all_reduce_sum(x: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _AllReduceSum.apply(x, group)


class _AllGatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.local_seq = x.shape[0]
        out = torch.empty((x.shape[0] * world_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty((ctx.local_seq, *grad.shape[1:]), dtype=grad.dtype, device=grad.device)
        dist.reduce_scatter_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None


class _ReduceScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        if x.shape[0] % world_size != 0:
            raise ValueError(
                f"Cannot reduce-scatter sequence dim {x.shape[0]} over TP={world_size}."
            )
        ctx.group = group
        ctx.world_size = world_size
        out = torch.empty((x.shape[0] // world_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty(
            (grad.shape[0] * ctx.world_size, *grad.shape[1:]), dtype=grad.dtype, device=grad.device
        )
        dist.all_gather_into_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None


class _ScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group, group_rank: int) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        if x.shape[0] % world_size != 0:
            raise ValueError(f"Cannot scatter sequence dim {x.shape[0]} over TP={world_size}.")
        ctx.group = group
        ctx.world_size = world_size
        local_seq = x.shape[0] // world_size
        start = int(group_rank) * local_seq
        return x[start : start + local_seq].contiguous()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty(
            (grad.shape[0] * ctx.world_size, *grad.shape[1:]), dtype=grad.dtype, device=grad.device
        )
        dist.all_gather_into_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None, None


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        out = x.contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = grad.contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)
        return out, None


def _all_gather_last_dim(x: torch.Tensor, group, *, reduce_backward: bool = False) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _AllGatherLastDim.apply(x, group, reduce_backward)


class _AllGatherLastDim(torch.autograd.Function):
    """All-gather last dim with Megatron tensor-parallel split backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, reduce_backward: bool) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.local_width = x.shape[-1]
        ctx.group_rank = dist.get_rank(group)
        ctx.reduce_backward = bool(reduce_backward)
        flat = x.movedim(-1, 0).contiguous().view(ctx.local_width, -1)
        gathered = torch.empty(
            (ctx.local_width * world_size, flat.shape[1]), dtype=x.dtype, device=x.device
        )
        dist.all_gather_into_tensor(gathered, flat, group=group)
        return (
            gathered.view(ctx.local_width * world_size, *x.shape[:-1]).movedim(0, -1).contiguous()
        )

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        flat = grad.movedim(-1, 0).contiguous().view(grad.shape[-1], -1)
        start = ctx.group_rank * ctx.local_width
        out = flat.narrow(0, start, ctx.local_width).contiguous()
        if ctx.reduce_backward:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)
        return out.view(ctx.local_width, *grad.shape[:-1]).movedim(0, -1).contiguous(), None, None


class _SequenceParallelRankPartitionedLoRA(torch.autograd.Function):
    """QKV LoRA path that recomputes gathered activations in backward.

    The ordinary composition of all-gather + matmul saves the full
    sequence-parallel gathered input for every layer. For QKV LoRA that input
    is much larger than the low-rank hidden activation. This function saves
    only the local input plus LoRA weights, then repeats the small gather/matmul
    sequence during backward.
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float, group
    ):
        world_size = dist.get_world_size(group) if group is not None else 1
        if world_size > 1:
            gathered = _all_gather_sequence_forward(x, group, world_size)
        else:
            gathered = x
        hidden_local = gathered.matmul(lora_a.t())
        hidden = _all_gather_last_dim_forward(hidden_local, group, world_size)
        out = hidden.matmul(lora_b.t()) * scale
        ctx.save_for_backward(x, lora_a, lora_b)
        ctx.group = group
        ctx.world_size = world_size
        ctx.local_seq = x.shape[0]
        ctx.local_rank_width = hidden_local.shape[-1]
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, lora_a, lora_b = ctx.saved_tensors
        world_size = ctx.world_size
        group = ctx.group
        if world_size > 1:
            gathered = _all_gather_sequence_forward(x, group, world_size)
        else:
            gathered = x
        hidden_local = gathered.matmul(lora_a.t())
        hidden = _all_gather_last_dim_forward(hidden_local, group, world_size)

        grad_out_scaled = grad_out * ctx.scale
        grad_b = (
            grad_out_scaled.reshape(-1, grad_out_scaled.shape[-1])
            .t()
            .matmul(hidden.reshape(-1, hidden.shape[-1]))
        )
        grad_hidden = grad_out_scaled.matmul(lora_b)
        if world_size > 1:
            grad_hidden_local = _split_last_dim(
                grad_hidden, dist.get_rank(group), ctx.local_rank_width
            )
            dist.all_reduce(grad_hidden_local, op=dist.ReduceOp.SUM, group=group)
        else:
            grad_hidden_local = grad_hidden
        grad_a = (
            grad_hidden_local.reshape(-1, grad_hidden_local.shape[-1])
            .t()
            .matmul(gathered.reshape(-1, gathered.shape[-1]))
        )
        grad_gathered = grad_hidden_local.matmul(lora_a)
        if world_size > 1:
            grad_x = _reduce_scatter_sequence_forward(grad_gathered, group, ctx.local_seq)
        else:
            grad_x = grad_gathered
        return grad_x, grad_a, grad_b, None, None


def _all_gather_sequence_forward(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    out = torch.empty((x.shape[0] * world_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _reduce_scatter_sequence_forward(x: torch.Tensor, group, local_seq: int) -> torch.Tensor:
    out = torch.empty((local_seq, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


def _all_gather_last_dim_forward(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return x
    local_width = x.shape[-1]
    flat = x.movedim(-1, 0).contiguous().view(local_width, -1)
    gathered = torch.empty(
        (local_width * world_size, flat.shape[1]), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(gathered, flat, group=group)
    return gathered.view(local_width * world_size, *x.shape[:-1]).movedim(0, -1).contiguous()


def _split_last_dim(x: torch.Tensor, group_rank: int, local_width: int) -> torch.Tensor:
    start = int(group_rank) * local_width
    return x.narrow(-1, start, local_width).contiguous()


class LinearLoRA(nn.Module):
    """Low-rank delta for a sharded linear layer.

    `a` is replicated unless the caller feeds a row-parallel local input. `b`
    has the local output shard for column-parallel surfaces, and the replicated
    full output for row-parallel surfaces.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: int | None = None,
        dropout: float = 0.0,
        use_rslora: bool = False,
        sequence_parallel_input: bool = False,
        row_parallel_output: bool = False,
        sequence_parallel_scatter_output: bool = False,
        tp_group=None,
        tp_rank: int = 0,
        rank_partition_size: int | None = None,
        rank_partitioned_a: bool = False,
        input_parallel_reduce: bool = False,
        output_partition_size: int | None = None,
        output_partitioned_b: bool = False,
        a_tensor_model_parallel: bool = False,
        b_tensor_model_parallel: bool = False,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive for LinearLoRA.")
        self.rank = int(rank)
        self.rank_partitioned_a = bool(rank_partitioned_a)
        if self.rank_partitioned_a:
            partition_size = (
                int(rank_partition_size)
                if rank_partition_size is not None
                else (dist.get_world_size(tp_group) if tp_group is not None else 1)
            )
            if partition_size <= 0:
                raise ValueError("LoRA rank partition size must be positive.")
            if self.rank % partition_size != 0:
                raise ValueError(
                    f"LoRA rank {self.rank} must be divisible by rank partition size {partition_size}."
                )
            self.rank_partition_size = partition_size
            self.local_rank = self.rank // partition_size
        else:
            self.rank_partition_size = 1
            self.local_rank = self.rank
        self.scale = lora_scaling(rank, alpha, use_rslora)
        self.use_rslora = bool(use_rslora)
        self.dropout_p = float(dropout)
        self.sequence_parallel_input = bool(sequence_parallel_input)
        self.row_parallel_output = bool(row_parallel_output)
        self.sequence_parallel_scatter_output = bool(sequence_parallel_scatter_output)
        if self.row_parallel_output and self.sequence_parallel_scatter_output:
            raise ValueError(
                "Use either row_parallel_output or sequence_parallel_scatter_output, not both."
            )
        self.tp_group = tp_group
        self.tp_rank = int(tp_rank)
        self.input_parallel_reduce = bool(input_parallel_reduce)
        self.output_partitioned_b = bool(output_partitioned_b)
        if self.output_partitioned_b:
            partition_size = (
                int(output_partition_size)
                if output_partition_size is not None
                else (dist.get_world_size(tp_group) if tp_group is not None else 1)
            )
            if partition_size <= 0:
                raise ValueError("LoRA output partition size must be positive.")
            if out_features % partition_size != 0:
                raise ValueError(
                    f"LoRA output features {out_features} must be divisible by {partition_size}."
                )
            self.output_partition_size = partition_size
            self.local_out_features = out_features // partition_size
        else:
            self.output_partition_size = 1
            self.local_out_features = out_features
        self.lora_a = nn.Parameter(torch.empty(self.local_rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(self.local_out_features, rank))
        self.lora_a.tensor_model_parallel = bool(a_tensor_model_parallel)
        self.lora_b.tensor_model_parallel = bool(b_tensor_model_parallel)
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel_input and self.rank_partitioned_a and not self.training:
            # Keep eval/inference on the simple path; the memory optimization
            # matters only when autograd needs to retain forward activations.
            pass
        elif (
            self.sequence_parallel_input
            and self.rank_partitioned_a
            and not self.input_parallel_reduce
            and not self.output_partitioned_b
            and not self.row_parallel_output
            and not self.sequence_parallel_scatter_output
            and self.dropout_p == 0.0
        ):
            return _SequenceParallelRankPartitionedLoRA.apply(
                x, self.lora_a, self.lora_b, self.scale, self.tp_group
            )
        if self.sequence_parallel_input:
            x = _gather_sequence_parallel(x, self.tp_group)
        dropped = F.dropout(x, p=self.dropout_p, training=self.training) if self.dropout_p else x
        hidden = dropped.matmul(self.lora_a.t())
        if self.rank_partitioned_a:
            hidden = _all_gather_last_dim(hidden, self.tp_group, reduce_backward=True)
        if self.input_parallel_reduce:
            hidden = _all_reduce_sum(hidden, self.tp_group)
        out = hidden.matmul(self.lora_b.t()) * self.scale
        if self.output_partitioned_b:
            out = _all_gather_last_dim(out, self.tp_group)
        if self.row_parallel_output:
            out = _reduce_scatter_sequence_parallel(out, self.tp_group)
        if self.sequence_parallel_scatter_output:
            out = _scatter_sequence_parallel(out, self.tp_group, self.tp_rank)
        return out

    @torch.no_grad()
    def olora_tail_init_(self, base_weight: torch.Tensor) -> None:
        """OLoRA-tail init from the (loaded) frozen base weight, in place.

        Sets ``lora_b = U₋ᵣ``, ``lora_a = V₋ᵣᵀ`` from the minor SVD subspace, then
        subtracts ``scale · B0 @ A0`` from ``base_weight`` (PiSSA-style residual) so
        the layer output is UNCHANGED at init: ``W0 x = (W0 - scale·B0A0)x + scale·B0A0x``.
        tp=1 only — a sharded base weight would need a distributed SVD.
        """
        tp_world = dist.get_world_size(self.tp_group) if self.tp_group is not None else 1
        if self.rank_partitioned_a or self.output_partitioned_b or tp_world > 1:
            raise NotImplementedError(
                "OLoRA-tail init supports tp=1 (unsharded base weight) only; "
                f"got tp_world={tp_world}, rank_partitioned_a={self.rank_partitioned_a}, "
                f"output_partitioned_b={self.output_partitioned_b}."
            )
        expected = (self.lora_b.shape[0], self.lora_a.shape[1])
        if tuple(base_weight.shape) != expected:
            raise ValueError(
                f"OLoRA-tail base weight shape {tuple(base_weight.shape)} != expected {expected}."
            )
        b0, a0 = olora_tail_factors(base_weight, self.rank)
        # SVD factors carry sign/degeneracy ambiguity that differs across ranks.
        # The init runs pre-sharding on replicated weights, but fsdp2 later shards
        # lora_a/lora_b dim-0 across data-parallel ranks: the concatenated factors
        # must reproduce the exact B0@A0 every rank's residual write-back used, so
        # all ranks must hold identical factors.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(b0, src=0)
            dist.broadcast(a0, src=0)
        self.lora_b.copy_(b0.to(self.lora_b.dtype))
        self.lora_a.copy_(a0.to(self.lora_a.dtype))
        base_weight.sub_((b0 @ a0).to(base_weight.dtype), alpha=self.scale)


class GroupedLinearLoRA(nn.Module):
    """Per-local-expert LoRA delta for `te.GroupedLinear` expert surfaces."""

    def __init__(
        self,
        num_local_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: int | None = None,
        dropout: float = 0.0,
        use_rslora: bool = False,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive for GroupedLinearLoRA.")
        self.num_local_experts = int(num_local_experts)
        self.rank = int(rank)
        self.scale = lora_scaling(rank, alpha, use_rslora)
        self.use_rslora = bool(use_rslora)
        self.dropout_p = float(dropout)
        self.lora_a = nn.Parameter(torch.empty(num_local_experts, rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(num_local_experts, out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor, splits: list[int]) -> torch.Tensor:
        if len(splits) != self.num_local_experts:
            raise ValueError(
                f"GroupedLinearLoRA expected {self.num_local_experts} splits, got {len(splits)}."
            )
        outputs = []
        offset = 0
        for expert_idx, size in enumerate(splits):
            x_i = x[offset : offset + size]
            if size == 0:
                outputs.append(x_i.new_empty((0, self.lora_b.shape[1])))
            else:
                dropped = (
                    F.dropout(x_i, p=self.dropout_p, training=self.training)
                    if self.dropout_p
                    else x_i
                )
                h_i = dropped.matmul(self.lora_a[expert_idx].t())
                outputs.append(h_i.matmul(self.lora_b[expert_idx].t()) * self.scale)
            offset += size
        return torch.cat(outputs, dim=0) if outputs else x.new_empty((0, self.lora_b.shape[1]))


class SharedGroupedLinearLoRA(nn.Module):
    """LoRA delta shared by all local experts in a GroupedLinear."""

    def __init__(
        self,
        num_local_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: int | None = None,
        dropout: float = 0.0,
        use_rslora: bool = False,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive for SharedGroupedLinearLoRA.")
        self.num_local_experts = int(num_local_experts)
        self.rank = int(rank)
        self.scale = lora_scaling(rank, alpha, use_rslora)
        self.use_rslora = bool(use_rslora)
        self.dropout_p = float(dropout)
        self.shared_across_experts = True
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank))
        self.lora_a.tensor_model_parallel = False
        self.lora_b.tensor_model_parallel = False
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor, splits: list[int]) -> torch.Tensor:
        if len(splits) != self.num_local_experts:
            raise ValueError(
                f"SharedGroupedLinearLoRA expected {self.num_local_experts} splits, got {len(splits)}."
            )
        dropped = F.dropout(x, p=self.dropout_p, training=self.training) if self.dropout_p else x
        return dropped.matmul(self.lora_a.t()).matmul(self.lora_b.t()) * self.scale

    @torch.no_grad()
    def olora_tail_init_(self, expert_weights: list[torch.Tensor]) -> None:
        """OLoRA-tail init for a single adapter SHARED across local experts.

        Each expert has its own base weight, but the adapter is shared, so there is no
        single ``W0``. We SVD the MEAN expert weight (the natural proxy for a shared
        correction) for the minor factors, then subtract the SAME ``scale·B0@A0`` from
        EVERY expert weight — which preserves each expert's output at init, because the
        shared adapter adds that same delta back uniformly:
        ``W_e x = (W_e - scale·B0A0)x + scale·B0A0x`` for every expert ``e``.
        """
        if not expert_weights:
            raise ValueError("OLoRA-tail expert init requires at least one expert weight.")
        expected = (self.lora_b.shape[0], self.lora_a.shape[1])
        for w in expert_weights:
            if tuple(w.shape) != expected:
                raise ValueError(
                    f"OLoRA-tail expert weight shape {tuple(w.shape)} != expected {expected}."
                )
        mean_w = torch.stack([w.detach().to(torch.float32) for w in expert_weights]).mean(dim=0)
        b0, a0 = olora_tail_factors(mean_w, self.rank)
        self.lora_b.copy_(b0.to(self.lora_b.dtype))
        self.lora_a.copy_(a0.to(self.lora_a.dtype))
        delta = (b0 @ a0) * self.scale
        for w in expert_weights:
            w.sub_(delta.to(w.dtype))


__all__ = [
    "GroupedLinearLoRA",
    "LinearLoRA",
    "LoraConfig",
    "SharedGroupedLinearLoRA",
    "apply_olora_tail_init",
    "freeze_non_lora_params",
    "normalize_lora_config",
    "olora_tail_factors",
    "trainable_param_stats",
]
