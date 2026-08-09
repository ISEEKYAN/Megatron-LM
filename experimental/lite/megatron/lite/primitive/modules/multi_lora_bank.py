# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense multi-LoRA banks and explicit named-adapter persistence.

The kernel-facing :class:`DenseLoraBank` deliberately knows only slots.  This
module keeps the tenant/name registry above that boundary: every surface uses
the same immutable ``name -> slot`` table, so a directory name can never
accidentally choose a different slot on another surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
from megatron.lite.primitive.ckpt import hf_weights
from megatron.lite.primitive.ckpt.hf_weights import export_hf_lora_bank_adapter
from megatron.lite.primitive.modules import multi_lora_kernel
from megatron.lite.primitive.modules.lora import (
    LoraSpec,
    _all_gather_last_dim,
    _all_reduce_sum,
    _gather_sequence_parallel,
    _scatter_sequence_parallel,
    resolve_lora_alpha,
)
from megatron.lite.primitive.modules.multi_lora import BatchedLoraDelta


@dataclass(frozen=True)
class LoraBankPartition:
    tp_size: int = 1
    rank_partitioned_a: bool = False
    output_partitioned_b: bool = False


@dataclass(frozen=True)
class DenseLoraBank:
    """A zero-copy homogeneous ``A``/``B`` bank for one LoRA surface."""

    a_bank: torch.Tensor
    b_bank: torch.Tensor
    partition: LoraBankPartition = LoraBankPartition()

    def __post_init__(self) -> None:
        if self.a_bank.ndim != 3 or self.b_bank.ndim != 3:
            raise ValueError(
                "DenseLoraBank requires three-dimensional A_bank and B_bank."
            )
        if self.partition.tp_size < 1 or (
            self.partition.rank_partitioned_a and self.partition.output_partitioned_b
        ):
            raise ValueError("invalid multi-LoRA TP partition metadata.")
        ranks_match = self.a_bank.shape[1] == self.b_bank.shape[2]
        if self.partition.rank_partitioned_a:
            ranks_match = (
                self.a_bank.shape[1] * self.partition.tp_size == self.b_bank.shape[2]
            )
        if self.a_bank.shape[0] != self.b_bank.shape[0] or not ranks_match:
            raise ValueError(
                "DenseLoraBank A_bank and B_bank must agree on slots and rank."
            )

    @property
    def slots(self) -> int:
        return self.a_bank.shape[0]

    def delta(
        self, x: torch.Tensor, lora_indices: torch.Tensor, *, scale: float
    ) -> torch.Tensor:
        return BatchedLoraDelta.apply(x, self.a_bank, self.b_bank, lora_indices, scale)


def apply_batched_lora_delta(
    bank: DenseLoraBank,
    x: torch.Tensor,
    lora_indices: torch.Tensor,
    *,
    scale: float,
    tp_group=None,
    tp_rank: int = 0,
    sequence_parallel_input: bool = False,
    input_parallel_reduce: bool = False,
    sequence_parallel_scatter_output: bool = False,
) -> torch.Tensor:
    """Apply a selected bank with the established LinearLoRA TP/SP collectives."""
    if x.ndim < 2:
        raise ValueError("batched LoRA input must have a feature dimension.")
    rows = x.reshape(-1, x.shape[-1])
    slots = lora_indices.reshape(-1)
    if slots.numel() != rows.shape[0]:
        raise ValueError("batched LoRA indices must have one entry per token.")
    if sequence_parallel_input:
        rows = _gather_sequence_parallel(rows, tp_group)
        slots = _gather_sequence_parallel(slots[:, None], tp_group).squeeze(-1)
    partition = bank.partition
    if rows.shape[0] == 0:
        width = bank.b_bank.shape[1] * (
            partition.tp_size if partition.output_partitioned_b else 1
        )
        return rows.new_zeros((rows.shape[0], width)).reshape(-1, *x.shape[1:-1], width)
    if not partition.rank_partitioned_a and not partition.output_partitioned_b:
        if slots.numel() < 2 or bool(torch.all(slots[1:] >= slots[:-1])):
            delta = bank.delta(rows, slots, scale=scale)
        else:
            order = torch.argsort(slots, stable=True)
            restore = torch.empty_like(order)
            restore[order] = torch.arange(order.numel(), device=order.device)
            delta = bank.delta(
                rows.index_select(0, order), slots.index_select(0, order), scale=scale
            ).index_select(0, restore)
    else:
        order = torch.argsort(slots, stable=True)
        restore = torch.empty_like(order)
        restore[order] = torch.arange(order.numel(), device=order.device)
        rows, slots = rows.index_select(0, order), slots.index_select(0, order)
        hidden = multi_lora_kernel.batched_lora_linear_stage(
            rows,
            bank.a_bank,
            slots,
            output_dtype=torch.float32,
            max_g_size_hint=rows.shape[0],
        )
        if partition.rank_partitioned_a:
            hidden = _all_gather_last_dim(hidden, tp_group, reduce_backward=True)
        if input_parallel_reduce:
            hidden = _all_reduce_sum(hidden, tp_group)
        delta = multi_lora_kernel.batched_lora_linear_stage(
            hidden,
            bank.b_bank,
            slots,
            scale=scale,
            output_dtype=rows.dtype,
            max_g_size_hint=rows.shape[0],
        )
        if partition.output_partitioned_b:
            delta = _all_gather_last_dim(delta, tp_group, reduce_backward=False)
        delta = delta.index_select(0, restore)
    if sequence_parallel_scatter_output:
        delta = _scatter_sequence_parallel(delta, tp_group, tp_rank)
    # SP collectives may change the number of token rows.  The output layout
    # must follow the post-collective delta, not the caller's pre-gather input.
    return delta.reshape(-1, *x.shape[1:-1], delta.shape[-1])


@dataclass(frozen=True)
class MultiLoraSpec:
    """Declarative construction contract for trainable named dense banks."""

    names: tuple[str, ...] = ()
    rank: int = 0
    alpha: int | None = None
    use_rslora: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.names)


def normalize_multi_lora_spec(
    config: MultiLoraSpec | Mapping[str, Any] | None
) -> MultiLoraSpec:
    """Normalize runtime config without allowing an unowned registry injection."""
    if config is None:
        return MultiLoraSpec()
    if isinstance(config, MultiLoraSpec):
        spec = config
    elif isinstance(config, Mapping):
        values = dict(config)
        if "names" in values:
            values["names"] = tuple(values["names"])
        spec = MultiLoraSpec(**values)
    else:
        raise TypeError("multi_lora must be MultiLoraSpec, mapping, or None.")
    if not spec.names:
        if spec.rank:
            raise ValueError("disabled multi_lora must not set rank.")
        return spec
    if spec.rank < 1:
        raise ValueError("enabled multi_lora requires rank >= 1.")
    if len(set(spec.names)) != len(spec.names) or any(not name for name in spec.names):
        raise ValueError("multi_lora adapter names must be unique non-empty strings.")
    if resolve_lora_alpha(spec.rank, spec.alpha) < 1:
        raise ValueError("multi_lora alpha must resolve to a positive value.")
    return spec


def validate_multi_lora_parallel_support(
    spec: MultiLoraSpec, *, tp_size: int, etp_size: int | None, use_deepep: bool
) -> None:
    """Reject unsupported parallel modes before model or optimizer construction."""
    if not spec.enabled:
        return
    if etp_size is not None and etp_size > 1:
        raise ValueError("multi-LoRA model-owned sidecars do not support ETP.")
    if tp_size > 1 and spec.rank % tp_size:
        raise ValueError("multi-LoRA rank must be divisible by TP size.")
    if use_deepep:
        raise ValueError("multi-LoRA model-owned sidecars do not support DeepEP.")


class MultiLoraTrainingState(nn.Module):
    """Model-owned bank parameters plus the only production sidecar factory."""

    def __init__(
        self,
        registry: "NamedLoraBankRegistry",
        layer_surfaces: Mapping[int, tuple[str, str]],
        attention_surfaces: Mapping[int, tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.registry = registry
        self._layer_surfaces = dict(layer_surfaces)
        self._attention_surfaces = dict(attention_surfaces or {})
        registered_tensor_names: dict[int, str] = {}
        for surface, bank in registry.banks.items():
            for factor, tensor in (("a", bank.a_bank), ("b", bank.b_bank)):
                if not isinstance(tensor, nn.Parameter):
                    raise TypeError(
                        "production multi-LoRA banks must own nn.Parameter tensors."
                    )
                parameter_name = self.parameter_name(surface, factor)
                prior_name = registered_tensor_names.get(id(tensor))
                if prior_name is not None:
                    raise ValueError(
                        "one model-owned multi-LoRA Parameter cannot represent two "
                        f"native surfaces: {prior_name!r} and {parameter_name!r}."
                    )
                self.register_parameter(parameter_name, tensor)
                registered_tensor_names[id(tensor)] = parameter_name

    @staticmethod
    def parameter_name(native_surface: str, factor: str) -> str:
        """Return a dot-free, reversible checkpoint key fragment for one factor."""
        if factor not in {"a", "b"}:
            raise ValueError("multi-LoRA parameter factor must be 'a' or 'b'.")
        return f"bank_{native_surface.encode('utf-8').hex()}_{factor}"

    def banks_for_layer(self, layer_idx: int) -> tuple[DenseLoraBank, DenseLoraBank]:
        """Return model-owned native-surface banks without model-layer imports."""
        try:
            fc1_surface, fc2_surface = self._layer_surfaces[layer_idx]
        except KeyError as exc:
            raise KeyError(f"multi-LoRA slots name unknown layer: {layer_idx}") from exc
        return self.registry.banks[fc1_surface], self.registry.banks[fc2_surface]

    def attention_banks_for_layer(
        self, layer_idx: int
    ) -> tuple[DenseLoraBank, DenseLoraBank] | None:
        """Return the QKV and output-projection banks for one model layer."""
        surfaces = self._attention_surfaces.get(layer_idx)
        if surfaces is None:
            return None
        qkv_surface, proj_surface = surfaces
        return self.registry.banks[qkv_surface], self.registry.banks[proj_surface]

    @property
    def local_layer_indices(self) -> tuple[int, ...]:
        """Global layer indices physically owned by this pipeline stage."""
        return tuple(sorted(self._layer_surfaces))

    def checkpoint_identity_metadata(self) -> dict[str, Any]:
        """Return the versioned tenant identity required to reload these slots.

        Bank tensors alone are insufficient: equal-shaped banks with a changed
        name-to-slot order would silently assign one tenant's adapter to
        another.  Training checkpoint bridges persist and compare this record
        before copying any bank tensor.
        """
        spec = self.registry.lora_spec
        return {
            "schema_version": 1,
            "names_by_slot": tuple(
                name
                for name, _slot in sorted(
                    self.registry.names.items(), key=lambda item: item[1]
                )
            ),
            "rank": self.registry.rank,
            "alpha": self.registry.alpha,
            "use_rslora": bool(spec.use_rslora) if spec is not None else False,
        }

    @property
    def scale(self) -> float:
        scale = (
            self.registry.lora_spec
            or LoraSpec(
                enabled=True, rank=self.registry.rank, alpha=self.registry.alpha
            )
        ).scale
        return scale


@dataclass(frozen=True)
class NamedLoraBankRegistry:
    """One stable adapter-name registry shared by homogeneous LoRA surfaces.

    ``banks`` maps a native model weight name (for example
    ``layers.0.moe.experts._fc1_weight_0``) to its three-dimensional
    ``DenseLoraBank``. ``names`` is the sole authority for choosing a bank
    slot; the HF exporter performs naming, TP/EP gather, and scale conversion.
    """

    banks: Mapping[str, DenseLoraBank]
    names: Mapping[str, int]
    rank: int
    alpha: int | None
    base_model_identity: Mapping[str, Any]
    revision: int = 1
    lora_spec: LoraSpec | None = None

    def __post_init__(self) -> None:
        if not self.banks:
            raise ValueError(
                "NamedLoraBankRegistry requires at least one surface bank."
            )
        if not self.names:
            raise ValueError(
                "NamedLoraBankRegistry requires at least one adapter name."
            )
        if (
            self.rank < 1
            or resolve_lora_alpha(self.rank, self.alpha) < 1
            or self.revision < 1
        ):
            raise ValueError("rank, resolved alpha, and revision must be positive.")
        if self.lora_spec is not None:
            if not self.lora_spec.enabled:
                raise ValueError(
                    "named multi-LoRA persistence requires an enabled LoraSpec."
                )
            if self.lora_spec.rank != self.rank:
                raise ValueError("LoraSpec rank must match the bank rank.")
            if resolve_lora_alpha(
                self.rank, self.lora_spec.alpha
            ) != resolve_lora_alpha(self.rank, self.alpha):
                raise ValueError("LoraSpec alpha must match the bank alpha.")
        slots = list(self.names.values())
        if any(not isinstance(slot, int) for slot in slots):
            raise TypeError("adapter slots must be integers.")
        if len(set(slots)) != len(slots):
            raise ValueError("NamedLoraBankRegistry rejects duplicate adapter slots.")
        if min(slots) < 0:
            raise ValueError("adapter slots must be non-negative.")
        slot_count = len(self.names)
        if set(slots) != set(range(slot_count)):
            raise ValueError("adapter slots must be a dense 0..K-1 mapping.")
        for surface, bank in self.banks.items():
            if not surface:
                raise ValueError("surface names must be non-empty.")
            if bank.slots != slot_count:
                raise ValueError(
                    f"surface {surface!r} has {bank.slots} slots; expected {slot_count}."
                )
            a_rank = (
                bank.a_bank.shape[1] * bank.partition.tp_size
                if bank.partition.rank_partitioned_a
                else bank.a_bank.shape[1]
            )
            if a_rank != self.rank or bank.b_bank.shape[2] != self.rank:
                raise ValueError(
                    f"surface {surface!r} rank does not match registry rank {self.rank}."
                )

    def slot_for(self, name: str) -> int:
        try:
            return self.names[name]
        except KeyError as exc:
            raise KeyError(f"Unknown multi-LoRA adapter name {name!r}.") from exc

    def select(self, name: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return two-dimensional factors for exactly ``name``'s explicit slot."""
        slot = self.slot_for(name)
        selected: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for surface, bank in self.banks.items():
            index = torch.tensor([slot], dtype=torch.long, device=bank.a_bank.device)
            a = bank.a_bank.index_select(0, index).squeeze(0)
            b = bank.b_bank.index_select(0, index).squeeze(0)
            selected[surface] = (a, b)
        return selected

    def export_hf_state(
        self, name: str, spec, ps, *, export_dtype=None
    ) -> dict[str, torch.Tensor]:
        """Export one named slot through the production HF adapter pipeline."""
        lora_spec = self.lora_spec or LoraSpec(
            enabled=True, rank=self.rank, alpha=self.alpha
        )
        selected = self.select(name)
        items = []
        for surface, bank in self.banks.items():
            active_partition = (
                bank.partition.rank_partitioned_a or bank.partition.output_partitioned_b
            )
            if (active_partition and bank.partition.tp_size != ps.tp_size) or (
                not active_partition and bank.partition.tp_size not in (1, ps.tp_size)
            ):
                raise ValueError(
                    "multi-LoRA bank TP metadata does not match export parallel state."
                )
            a, b = selected[surface]
            # DTensor must become its local tensor before the LoRA-internal
            # gather.  The normal exporter then performs only the base-weight
            # gather for this same surface.
            a = hf_weights._materialize_dtensor(a)
            b = hf_weights._materialize_dtensor(b)
            if ps.tp_size > 1 and bank.partition.rank_partitioned_a:
                a = hf_weights.allgather_concat(a, ps.tp_size, ps.tp_group, dim=0)
            if ps.tp_size > 1 and bank.partition.output_partitioned_b:
                b = hf_weights.allgather_concat(b, ps.tp_size, ps.tp_group, dim=0)
            items.extend(
                export_hf_lora_bank_adapter(
                    {surface: (a, b)},
                    spec=spec,
                    ps=ps,
                    train_scale=lora_spec.scale,
                    rank=self.rank,
                    alpha=self.alpha,
                    use_rslora=lora_spec.use_rslora,
                    export_dtype=export_dtype,
                )
            )
        state = dict(items)
        if len(state) != len(items):
            raise RuntimeError(
                "named multi-LoRA export produced duplicate HF keys; each grouped "
                "native surface must be registered exactly once before expert expansion."
            )
        return state


__all__ = [
    "DenseLoraBank",
    "apply_batched_lora_delta",
    "MultiLoraSpec",
    "MultiLoraTrainingState",
    "NamedLoraBankRegistry",
    "normalize_multi_lora_spec",
    "validate_multi_lora_parallel_support",
]
