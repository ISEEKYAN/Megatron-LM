# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LoRA helpers for Megatron Lite native model implementations.

The primitive owns generic Megatron-style sharded linear adapters. Individual
models declare extra target attributes next to their model implementation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

LORA_DEFAULT_RANK = 0
LORA_DEFAULT_ALPHA = None
LORA_DEFAULT_DROPOUT = 0.0
LORA_DEFAULT_TARGET_MODULES = (
    "linear_qkv",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
)
LORA_DEFAULT_USE_RSLORA = False
_DEFAULT_IGNORE_PATTERNS = (
    "lm_head",
    "output_layer",
    "router",
    "gate",
    "embedding",
    "word_embeddings",
)
_TARGET_ALIASES = {
    "qkv": "linear_qkv",
    "proj": "linear_proj",
    "fc1": "linear_fc1",
    "fc2": "linear_fc2",
}

# Initializations in this set replace the pretrained weight with a residual
# base.  Any consumer that combines a separately loaded base with the adapter
# must consult this contract.  Keep the older OLoRA/PiSSA spellings fail-safe
# even though the only base-mutating initialization currently implemented by
# MLite is ``olora_tail``.
_LORA_INITS_WITH_RESIDUAL_BASE = frozenset({"olora", "olora_tail", "pissa"})


def lora_init_uses_residual_base(init: str | None) -> bool:
    """Whether ``init`` replaces the pretrained weight with a residual base."""

    normalized = str(init or "default").lower()
    return normalized in _LORA_INITS_WITH_RESIDUAL_BASE


def resolve_lora_alpha(rank: int, alpha: int | None) -> int:
    """Resolve the shared training/export alpha contract."""

    return int(rank if alpha is None else alpha)


def lora_scaling(rank: int, alpha: int | None, *, use_rslora: bool = False) -> float:
    effective_alpha = float(resolve_lora_alpha(rank, alpha))
    denom = float(rank) ** 0.5 if use_rslora else float(rank)
    return effective_alpha / denom


def assert_lora_alpha_scaling_consistent(
    rank: int,
    alpha: int | None,
    *,
    applied_alpha: int,
    applied_use_rslora: bool,
    use_rslora: bool = False,
) -> None:
    """Fail if a consumer would apply a different LoRA alpha or scale."""

    expected_alpha = resolve_lora_alpha(rank, alpha)
    expected_scaling = lora_scaling(rank, alpha, use_rslora=use_rslora)
    applied_scaling = lora_scaling(
        rank,
        applied_alpha,
        use_rslora=applied_use_rslora,
    )
    if applied_alpha != expected_alpha or applied_scaling != expected_scaling:
        raise RuntimeError(
            "LoRA alpha/scaling disagrees between training and rollout: "
            f"training alpha={expected_alpha}, scaling={expected_scaling}; "
            f"rollout alpha={applied_alpha}, scaling={applied_scaling}."
        )


@dataclass(frozen=True)
class LoraSpec:
    enabled: bool = False
    rank: int = LORA_DEFAULT_RANK
    alpha: int | None = LORA_DEFAULT_ALPHA
    dropout: float = LORA_DEFAULT_DROPOUT
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: LORA_DEFAULT_TARGET_MODULES
    )
    use_rslora: bool = LORA_DEFAULT_USE_RSLORA
    init: str = "default"
    # How the rollout engine receives the adapter.
    #
    # ``merge`` folds the delta into the base weight in the rollout dtype, which
    # rounds away every update below half an ulp of the base weight. It is kept
    # for initializations whose base is not the pretrained weight (OLoRA/PiSSA),
    # where adapter-only sync would apply the delta to the wrong operand.
    #
    # ``adapter`` ships the LoRA factors and lets the inference engine apply
    # them. This is the default.
    #
    # NOTE: this field is declarative. The rollout export path resolves the mode
    # from the raw engine config, not from this dataclass -- see
    # ``MegatronLiteEngine._lora_rollout_sync_is_merge``. Keep the two defaults
    # equal; tests/unit/verl/test_mlite_engine_lora_sync.py pins the resolver,
    # which is the site that actually decides, and asserts the two agree.
    #
    # An earlier revision of this comment argued for a ``merge`` default from a
    # run whose adapter arm produced a near-uniform policy. That failure was the
    # rollout engine starting from random weights under ``load_format=dummy``
    # plus an expert-parallel misattribution; both are fixed, so the argument no
    # longer applies and is not restated here.
    rollout_sync: str = "adapter"
    ignore_patterns: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_IGNORE_PATTERNS
    )

    @property
    def scale(self) -> float:
        return lora_scaling(self.rank, self.alpha, use_rslora=self.use_rslora)

    def targets(self) -> set[str]:
        out = set()
        for target in self.target_modules:
            out.add(_TARGET_ALIASES.get(target, target))
        return out

    def targets_module(self, name: str) -> bool:
        canonical = _TARGET_ALIASES.get(name, name)
        return canonical in self.targets()

    def ignores_module(self, name: str) -> bool:
        """Match exact, case-insensitive dotted path components."""
        components = {component.lower() for component in name.split(".")}
        return any(pattern.lower() in components for pattern in self.ignore_patterns)


def normalize_lora_spec(config: LoraSpec | dict[str, Any] | None) -> LoraSpec:
    if config is None:
        return LoraSpec()
    if isinstance(config, LoraSpec):
        return config
    if not isinstance(config, dict):
        raise TypeError(
            f"LoRA spec must be LoraSpec, dict, or None, got {type(config)!r}."
        )
    values = dict(config)
    if (
        "enabled" not in values
        and int(values.get("rank", LORA_DEFAULT_RANK) or 0) > 0
    ):
        warnings.warn(
            "LoRA rank alone is inert; LoRA requires enabled=True.",
            UserWarning,
            stacklevel=2,
        )
    if values.get("enabled") is False:
        values["rank"] = LORA_DEFAULT_RANK
    if "targets" in values and "target_modules" not in values:
        values["target_modules"] = values.pop("targets")
    else:
        values.pop("targets", None)
    if "target_modules" in values and not isinstance(values["target_modules"], tuple):
        values["target_modules"] = tuple(values["target_modules"])
    if "ignore_patterns" in values and not isinstance(values["ignore_patterns"], tuple):
        values["ignore_patterns"] = tuple(values["ignore_patterns"])
    return LoraSpec(**values)


LoraConfig = LoraSpec
normalize_lora_config = normalize_lora_spec


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
        out = torch.empty(
            (x.shape[0] * world_size, *x.shape[1:]), dtype=x.dtype, device=x.device
        )
        dist.all_gather_into_tensor(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty(
            (ctx.local_seq, *grad.shape[1:]), dtype=grad.dtype, device=grad.device
        )
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
        out = torch.empty(
            (x.shape[0] // world_size, *x.shape[1:]), dtype=x.dtype, device=x.device
        )
        dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty(
            (grad.shape[0] * ctx.world_size, *grad.shape[1:]),
            dtype=grad.dtype,
            device=grad.device,
        )
        dist.all_gather_into_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None


class _ScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group, group_rank: int) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        if x.shape[0] % world_size != 0:
            raise ValueError(
                f"Cannot scatter sequence dim {x.shape[0]} over TP={world_size}."
            )
        ctx.group = group
        ctx.world_size = world_size
        local_seq = x.shape[0] // world_size
        start = int(group_rank) * local_seq
        return x[start : start + local_seq].contiguous()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty(
            (grad.shape[0] * ctx.world_size, *grad.shape[1:]),
            dtype=grad.dtype,
            device=grad.device,
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
        # Row-parallel LoRA partitions both A's input and B's output. After
        # _AllGatherLastDim.backward selects the local B-output slice, each TP
        # rank owns a different contribution to d(hidden). Sum those
        # contributions before propagating through the input-sharded A.
        # This is required by the chain rule; it does not rescale a replicated
        # gradient from the forward all-reduce.
        out = grad.contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)
        return out, None


def _all_gather_last_dim(
    x: torch.Tensor, group, *, reduce_backward: bool = False
) -> torch.Tensor:
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
            (ctx.local_width * world_size, flat.shape[1]),
            dtype=x.dtype,
            device=x.device,
        )
        dist.all_gather_into_tensor(gathered, flat, group=group)
        return (
            gathered.view(ctx.local_width * world_size, *x.shape[:-1])
            .movedim(0, -1)
            .contiguous()
        )

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        flat = grad.movedim(-1, 0).contiguous().view(grad.shape[-1], -1)
        start = ctx.group_rank * ctx.local_width
        out = flat.narrow(0, start, ctx.local_width).contiguous()
        if ctx.reduce_backward:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)
        return (
            out.view(ctx.local_width, *grad.shape[:-1]).movedim(0, -1).contiguous(),
            None,
            None,
        )


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
        ctx,
        x: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
        group,
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
            grad_x = _reduce_scatter_sequence_forward(
                grad_gathered, group, ctx.local_seq
            )
        else:
            grad_x = grad_gathered
        return grad_x, grad_a, grad_b, None, None


def _all_gather_sequence_forward(
    x: torch.Tensor, group, world_size: int
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0] * world_size, *x.shape[1:]), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _reduce_scatter_sequence_forward(
    x: torch.Tensor, group, local_seq: int
) -> torch.Tensor:
    out = torch.empty((local_seq, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


def _all_gather_last_dim_forward(
    x: torch.Tensor, group, world_size: int
) -> torch.Tensor:
    if world_size == 1:
        return x
    local_width = x.shape[-1]
    flat = x.movedim(-1, 0).contiguous().view(local_width, -1)
    gathered = torch.empty(
        (local_width * world_size, flat.shape[1]), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(gathered, flat, group=group)
    return (
        gathered.view(local_width * world_size, *x.shape[:-1])
        .movedim(0, -1)
        .contiguous()
    )


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
        alpha: int | None = LORA_DEFAULT_ALPHA,
        dropout: float = LORA_DEFAULT_DROPOUT,
        use_rslora: bool = LORA_DEFAULT_USE_RSLORA,
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
        self.use_rslora = bool(use_rslora)
        self.alpha = resolve_lora_alpha(rank, alpha)
        self.scale = lora_scaling(rank, alpha, use_rslora=use_rslora)
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
        if (
            self.sequence_parallel_input
            and self.rank_partitioned_a
            and not self.training
        ):
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
        dropped = (
            F.dropout(x, p=self.dropout_p, training=self.training)
            if self.dropout_p
            else x
        )
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

    def materialized_lora_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return TP-gathered ``(lora_a, lora_b)`` with **no** scaling applied.

        Adapter-only rollout sync ships the factors themselves, so the scale
        must stay separable: the consumer (vLLM) applies its own scaling and the
        exporter compensates for any mismatch. ``materialized_delta_weight`` is
        the merged-export counterpart and multiplies ``self.scale`` back in.
        """
        if self.dropout_p != 0.0:
            raise ValueError("materialized_lora_factors requires dropout=0.")
        lora_a = self.lora_a
        lora_b = self.lora_b
        if hasattr(lora_a, "full_tensor"):
            lora_a = lora_a.full_tensor()
        if hasattr(lora_b, "full_tensor"):
            lora_b = lora_b.full_tensor()
        if self.rank_partitioned_a:
            lora_a = _all_gather_last_dim_forward(
                lora_a.t(), self.tp_group, self.rank_partition_size
            ).t()
        if self.output_partitioned_b:
            lora_b = _all_gather_last_dim_forward(
                lora_b.t(), self.tp_group, self.output_partition_size
            ).t()
        return lora_a, lora_b

    def materialized_delta_weight(self) -> torch.Tensor:
        # Merge/export uses eval semantics, so activation dropout is irrelevant.
        # Adapter-only export is stricter because the consumer owns dropout and
        # therefore continues to reject non-zero dropout in
        # ``materialized_lora_factors``.
        lora_a = self.lora_a
        lora_b = self.lora_b
        if hasattr(lora_a, "full_tensor"):
            lora_a = lora_a.full_tensor()
        if hasattr(lora_b, "full_tensor"):
            lora_b = lora_b.full_tensor()
        if self.rank_partitioned_a:
            lora_a = _all_gather_last_dim_forward(
                lora_a.t(), self.tp_group, self.rank_partition_size
            ).t()
        if self.output_partitioned_b:
            lora_b = _all_gather_last_dim_forward(
                lora_b.t(), self.tp_group, self.output_partition_size
            ).t()
        return (lora_b @ lora_a) * self.scale

    def olora_tail_init_(self, base_weight: torch.Tensor) -> None:
        with torch.no_grad():
            delta = self.materialized_delta_weight()
            if base_weight.shape != delta.shape:
                raise ValueError(
                    f"OLoRA-tail base shape {base_weight.shape} != delta {delta.shape}."
                )
            base_weight.sub_(delta.to(base_weight.dtype))


class SharedGroupedLinearLoRA(nn.Module):
    """LoRA delta shared by all local experts in a GroupedLinear."""

    def __init__(
        self,
        num_local_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: int | None = LORA_DEFAULT_ALPHA,
        dropout: float = LORA_DEFAULT_DROPOUT,
        use_rslora: bool = LORA_DEFAULT_USE_RSLORA,
        tp_group=None,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive for SharedGroupedLinearLoRA.")
        self.num_local_experts = int(num_local_experts)
        self.rank = int(rank)
        self.use_rslora = bool(use_rslora)
        self.alpha = resolve_lora_alpha(rank, alpha)
        self.scale = lora_scaling(rank, alpha, use_rslora=use_rslora)
        self.dropout_p = float(dropout)
        self.tp_group = tp_group
        self.shared_across_experts = True
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank))
        self.lora_a.tensor_model_parallel = False
        self.lora_b.tensor_model_parallel = False
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)
        if self.tp_group is not None:
            self.lora_a.register_hook(self._all_reduce_tp_grad)
            self.lora_b.register_hook(self._all_reduce_tp_grad)

    def _all_reduce_tp_grad(self, grad: torch.Tensor) -> torch.Tensor:
        return _all_reduce_sum(grad, self.tp_group)

    def forward(self, x: torch.Tensor, splits: list[int]) -> torch.Tensor:
        if len(splits) != self.num_local_experts:
            raise ValueError(
                f"SharedGroupedLinearLoRA expected {self.num_local_experts} splits, got {len(splits)}."
            )
        dropped = (
            F.dropout(x, p=self.dropout_p, training=self.training)
            if self.dropout_p
            else x
        )
        return dropped.matmul(self.lora_a.t()).matmul(self.lora_b.t()) * self.scale

    def materialized_lora_factors(
        self, expert_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(lora_a, lora_b)`` with **no** scaling applied.

        The adapter is shared by every local expert, so ``expert_idx`` only
        documents the caller's intent; the returned factors are identical for
        all experts (see the class docstring).
        """
        if self.dropout_p != 0.0:
            raise ValueError("materialized_lora_factors requires dropout=0.")
        del expert_idx
        lora_a = self.lora_a
        lora_b = self.lora_b
        if hasattr(lora_a, "full_tensor"):
            lora_a = lora_a.full_tensor()
        if hasattr(lora_b, "full_tensor"):
            lora_b = lora_b.full_tensor()
        return lora_a, lora_b

    def materialized_delta_weight(self, expert_idx: int = 0) -> torch.Tensor:
        del expert_idx
        lora_a = self.lora_a
        lora_b = self.lora_b
        if hasattr(lora_a, "full_tensor"):
            lora_a = lora_a.full_tensor()
        if hasattr(lora_b, "full_tensor"):
            lora_b = lora_b.full_tensor()
        return (lora_b @ lora_a) * self.scale


def _weight_owner(module: nn.Module) -> nn.Module | None:
    inner = getattr(module, "linear", None)
    if isinstance(inner, nn.Module) and isinstance(
        getattr(inner, "weight", None), nn.Parameter
    ):
        if inner.weight.dim() == 2:
            return inner
    if (
        isinstance(getattr(module, "weight", None), nn.Parameter)
        and module.weight.dim() == 2
    ):
        return module
    return None


def apply_olora_tail_init(model: nn.Module) -> dict[str, int]:
    """Initialize supported dense adapters and warn for unsupported grouped experts."""

    from megatron.lite.primitive.modules.lora_apply import (
        LoRAWrappedGroupedLinear,
        LoRAWrappedLinear,
    )

    stats = {"initialized": 0, "skipped": 0}
    skipped_grouped_experts = 0
    for module in model.modules():
        if isinstance(module, LoRAWrappedLinear):
            owner = _weight_owner(module.base)
            if owner is None:
                stats["skipped"] += 1
                continue
            module.adapter.olora_tail_init_(owner.weight)
            stats["initialized"] += 1
        elif isinstance(module, LoRAWrappedGroupedLinear):
            stats["skipped"] += 1
            skipped_grouped_experts += 1
    if skipped_grouped_experts:
        warnings.warn(
            "OLoRA-tail does not support MoE grouped expert adapters; "
            f"skipped {skipped_grouped_experts} grouped adapter(s) while "
            "initializing supported dense adapters.",
            UserWarning,
            stacklevel=2,
        )
    return stats


__all__ = [
    "LORA_DEFAULT_ALPHA",
    "LORA_DEFAULT_DROPOUT",
    "LORA_DEFAULT_RANK",
    "LORA_DEFAULT_TARGET_MODULES",
    "LORA_DEFAULT_USE_RSLORA",
    "LinearLoRA",
    "LoraConfig",
    "LoraSpec",
    "SharedGroupedLinearLoRA",
    "apply_olora_tail_init",
    "assert_lora_alpha_scaling_consistent",
    "freeze_non_lora_params",
    "lora_init_uses_residual_base",
    "lora_scaling",
    "normalize_lora_config",
    "normalize_lora_spec",
    "resolve_lora_alpha",
    "trainable_param_stats",
]
