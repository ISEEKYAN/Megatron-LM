# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense multi-LoRA banks and explicit named-adapter persistence.

The kernel-facing :class:`DenseLoraBank` deliberately knows only slots.  This
module keeps the tenant/name registry above that boundary: every surface uses
the same immutable ``name -> slot`` table, so a directory name can never
accidentally choose a different slot on another surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from megatron.lite.primitive.ckpt.hf_weights import (
    VLLM_LORA_NAME_PREFIX,
    export_hf_lora_bank_adapter,
)
from megatron.lite.primitive.modules.lora import LoraSpec, resolve_lora_alpha
from megatron.lite.primitive.modules.multi_lora import BatchedLoraDelta


@dataclass(frozen=True)
class DenseLoraBank:
    """A zero-copy homogeneous ``A``/``B`` bank for one LoRA surface."""

    a_bank: torch.Tensor
    b_bank: torch.Tensor

    def __post_init__(self) -> None:
        if self.a_bank.ndim != 3 or self.b_bank.ndim != 3:
            raise ValueError(
                "DenseLoraBank requires three-dimensional A_bank and B_bank."
            )
        if (
            self.a_bank.shape[0] != self.b_bank.shape[0]
            or self.a_bank.shape[1] != self.b_bank.shape[2]
        ):
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


@dataclass(frozen=True)
class NamedLoraBankRegistry:
    """One stable adapter-name registry shared by homogeneous LoRA surfaces.

    ``banks`` maps a PEFT module name (for example ``layers.0.q_proj``) to its
    three-dimensional ``DenseLoraBank``.  ``names`` is the sole authority for
    choosing a bank slot; file/directory enumeration is intentionally absent
    from both import and export.
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
            if bank.a_bank.shape[1] != self.rank or bank.b_bank.shape[2] != self.rank:
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

    def export_state(self, name: str, *, cpu: bool = True) -> dict[str, torch.Tensor]:
        """Produce ordinary single-adapter PEFT keys for one explicit name."""
        state: dict[str, torch.Tensor] = {}
        for surface, (a, b) in self.select(name).items():
            if a.ndim != 2 or b.ndim != 2:
                raise AssertionError(
                    "select() must reduce a bank to two-dimensional factors."
                )
            state[f"{VLLM_LORA_NAME_PREFIX}{surface}.lora_A.weight"] = (
                a.detach().cpu().contiguous() if cpu else a.detach().contiguous()
            )
            state[f"{VLLM_LORA_NAME_PREFIX}{surface}.lora_B.weight"] = (
                b.detach().cpu().contiguous() if cpu else b.detach().contiguous()
            )
        return state

    def export_hf_state(
        self, name: str, spec, ps, *, export_dtype=None
    ) -> dict[str, torch.Tensor]:
        """Export one named slot through the production HF adapter pipeline."""
        lora_spec = self.lora_spec or LoraSpec(
            enabled=True, rank=self.rank, alpha=self.alpha
        )
        return dict(
            export_hf_lora_bank_adapter(
                self.select(name),
                spec=spec,
                ps=ps,
                train_scale=lora_spec.scale,
                rank=self.rank,
                alpha=self.alpha,
                use_rslora=lora_spec.use_rslora,
                export_dtype=export_dtype,
            )
        )

    def manifest(self, name: str) -> dict[str, Any]:
        """Return self-validating metadata for one exported slot."""
        slot = self.slot_for(name)
        return {
            "format": "megatron.lite_multi_lora_peft_v1",
            "name": name,
            "slot": slot,
            "revision": self.revision,
            "rank": self.rank,
            "alpha": resolve_lora_alpha(self.rank, self.alpha),
            "targets": sorted(self.banks),
            "base_model_identity": dict(self.base_model_identity),
            "shapes": {
                surface: {
                    "A": list(bank.a_bank.shape[1:]),
                    "B": list(bank.b_bank.shape[1:]),
                }
                for surface, bank in self.banks.items()
            },
        }

    def load_state(
        self, name: str, state: Mapping[str, torch.Tensor], manifest: Mapping[str, Any]
    ) -> None:
        """Validate then write a directory's factors into one explicit slot only."""
        expected = self.manifest(name)
        for key in (
            "format",
            "name",
            "slot",
            "revision",
            "rank",
            "alpha",
            "targets",
            "base_model_identity",
            "shapes",
        ):
            if manifest.get(key) != expected[key]:
                raise ValueError(
                    f"multi-LoRA manifest {key!r} does not match registered adapter {name!r}."
                )
        expected_keys = {
            f"{VLLM_LORA_NAME_PREFIX}{surface}.lora_{factor}.weight"
            for surface in self.banks
            for factor in ("A", "B")
        }
        if set(state) != expected_keys:
            raise ValueError(
                "adapter state keys do not exactly match registered target surfaces."
            )
        slot = self.slot_for(name)
        for surface, bank in self.banks.items():
            for factor, target in (("A", bank.a_bank), ("B", bank.b_bank)):
                value = state[f"{VLLM_LORA_NAME_PREFIX}{surface}.lora_{factor}.weight"]
                if tuple(value.shape) != tuple(target.shape[1:]):
                    raise ValueError(
                        f"adapter {name!r} {surface!r} lora_{factor} shape {tuple(value.shape)} "
                        f"does not match {tuple(target.shape[1:])}."
                    )
                # Adapter import is a state update, never part of the caller's
                # autograd graph.  Keep this explicit instead of using
                # ``.data``, which can silently invalidate autograd views.
                with torch.no_grad():
                    target[slot].copy_(
                        value.to(device=target.device, dtype=target.dtype)
                    )


def save_named_lora_adapter(
    registry: NamedLoraBankRegistry, name: str, output_dir: str | Path
) -> dict[str, Any]:
    """Save one named bank slot as a normal PEFT adapter directory."""
    from safetensors.torch import save_file

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = registry.export_state(name)
    save_file(state, str(output / "adapter_model.safetensors"))
    manifest = registry.manifest(name)
    (output / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": str(
                    registry.base_model_identity.get("name", "")
                ),
                "inference_mode": False,
                "r": registry.rank,
                "lora_alpha": resolve_lora_alpha(registry.rank, registry.alpha),
                "lora_dropout": (
                    registry.lora_spec.dropout
                    if registry.lora_spec is not None
                    else 0.0
                ),
                "target_modules": sorted(
                    {surface.rsplit(".", 1)[-1] for surface in manifest["targets"]}
                ),
                "bias": "none",
                "fan_in_fan_out": False,
                "init_lora_weights": True,
                "modules_to_save": None,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "multi_lora_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return {
        "path": str(output),
        "name": name,
        "slot": manifest["slot"],
        "manifest": manifest,
    }


def export_named_lora_adapter_state(
    registry: NamedLoraBankRegistry, name: str, spec, ps, *, export_dtype=None
) -> dict[str, torch.Tensor]:
    """Production export entry for one named multi-LoRA bank slot."""
    return registry.export_hf_state(name, spec, ps, export_dtype=export_dtype)


def load_named_lora_adapter(
    registry: NamedLoraBankRegistry, name: str, adapter_dir: str | Path
) -> dict[str, Any]:
    """Load one named directory into its registered slot; never infer the slot."""
    from safetensors.torch import load_file

    output = Path(adapter_dir)
    manifest = json.loads((output / "multi_lora_manifest.json").read_text())
    state = load_file(str(output / "adapter_model.safetensors"), device="cpu")
    registry.load_state(name, state, manifest)
    return {"path": str(output), "name": name, "slot": registry.slot_for(name)}


__all__ = [
    "DenseLoraBank",
    "NamedLoraBankRegistry",
    "export_named_lora_adapter_state",
    "load_named_lora_adapter",
    "save_named_lora_adapter",
]
