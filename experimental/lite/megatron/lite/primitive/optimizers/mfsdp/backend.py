# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime backend contract for standalone M-FSDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MegatronFSDPBackend:
    name: str = "mfsdp"
    runtime_backend: str = "megatron_fsdp"

    def zero_grad(self, optimizer: Any) -> None:
        optimizer.zero_grad()

    def step(self, optimizer: Any):
        return optimizer.step()

    def state_dict(self, optimizer: Any) -> dict[str, Any]:
        return optimizer.state_dict()

    def load_state_dict(self, optimizer: Any, state_dict: dict[str, Any]) -> None:
        optimizer.load_state_dict(state_dict)

    def sync_model_weights_to_main_weights(self, optimizer: Any) -> bool:
        for chunk in optimizer._model_chunks:
            chunk.param_sync.copy_full_parameters_to_shards()
        return True


BACKEND = MegatronFSDPBackend()


__all__ = [
    "BACKEND",
    "MegatronFSDPBackend",
]
