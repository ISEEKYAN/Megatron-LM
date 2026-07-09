# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime backend contract for standalone M-FSDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from megatron.lite.primitive.optimizers.mfsdp.uneven_dtensor import (
    preprocess_state_dict_for_uneven_dtensor,
)


@dataclass(frozen=True, slots=True)
class MegatronFSDPBackend:
    name: str = "mfsdp"
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

    def dcp_state_dict(self, optimizer: Any, *, is_loading: bool) -> dict[str, Any]:
        state = optimizer.state_dict()
        return preprocess_state_dict_for_uneven_dtensor(state)

    def load_state_dict(self, optimizer: Any, state_dict: dict[str, Any]) -> None:
        optimizer.load_state_dict(state_dict)

    def sync_model_weights_to_main_weights(self, optimizer: Any) -> bool:
        for chunk in optimizer._model_chunks:
            chunk.param_sync.copy_full_parameters_to_shards()
        return True

    def finalize_grads(
        self, finalize_fn, model_chunks: list[Any], optimizer: Any
    ) -> None:
        finalize_fn()


BACKEND = MegatronFSDPBackend()


__all__ = ["BACKEND", "MegatronFSDPBackend"]
