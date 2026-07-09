# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime-facing contract for the self-contained M-FSDP optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MegatronFSDPBackend:
    name: str = "megatron_fsdp"
    runtime_backend: str = "megatron_fsdp"

    def zero_grad(self, optimizer: Any) -> None:
        optimizer.zero_grad()

    def finish_grad_sync(self, optimizer: Any) -> None:
        optimizer.finish_grad_sync()

    def clip_grad_norm(self, optimizer: Any):
        return optimizer.clip_grad_norm()

    def step(self, optimizer: Any):
        return optimizer.step()

    def state_dict(self, optimizer: Any) -> dict[str, Any]:
        return optimizer.state_dict()

    def dcp_state_dict(
        self,
        optimizer: Any,
        *,
        is_loading: bool,
        include_main_params: bool = False,
    ) -> dict[str, Any]:
        del is_loading, include_main_params
        return optimizer.state_dict()

    def load_state_dict(self, optimizer: Any, state_dict: dict[str, Any]) -> None:
        optimizer.load_state_dict(state_dict)

    def state_dict_has_main_params(self, state_dict: Any) -> bool:
        del state_dict
        return False

    def sync_model_weights_to_main_weights(self, optimizer: Any) -> bool:
        sync = getattr(optimizer, "sync_model_weights_to_main_weights", None)
        return bool(sync()) if callable(sync) else False

    def finalize_grads(
        self, finalize_fn, model_chunks: list[Any], optimizer: Any
    ) -> None:
        finalize_fn(model_chunks, optimizer)


BACKEND = MegatronFSDPBackend()


def is_megatron_fsdp_optimizer(optimizer: Any) -> bool:
    return getattr(optimizer, "name", None) == "megatron_fsdp"


__all__ = ["BACKEND", "MegatronFSDPBackend", "is_megatron_fsdp_optimizer"]
