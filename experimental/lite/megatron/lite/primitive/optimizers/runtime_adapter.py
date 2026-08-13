# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime-facing optimizer lifecycle adapters.

The runtime delegates execution-wrapper ownership and device transitions through
this contract. Optimizer implementations may override either operation without
teaching the runtime about a concrete sharding backend.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class DefaultOptimizerRuntimeAdapter:
    name: str = "none"
    runtime_backend: str = "none"

    def pipeline_model_chunks(self, model_chunks: list[Any]) -> list[Any]:
        from megatron.lite.primitive.ckpt.hf_weights import unwrap_model

        return [unwrap_model(chunk) for chunk in model_chunks]

    def release_export_scratch(self, model_chunks: list[Any]) -> None:
        for chunk in model_chunks:
            release = getattr(chunk, "release_export_scratch", None)
            if callable(release):
                release()

    def transfer_training_state(
        self,
        optimizer: Any | None,
        model_chunks: list[Any],
        device: str,
        *,
        model: bool,
        optimizer_state: bool,
        grad: bool,
    ) -> None:
        from megatron.lite.runtime.megatron_utils import (
            load_model_to_gpu,
            load_optimizer,
            offload_model_to_cpu,
            offload_optimizer,
        )

        training_transfer = model and grad
        if device == "cpu":
            if model:
                offload_model_to_cpu(model_chunks)
            if (optimizer_state or training_transfer) and optimizer is not None:
                offload_state = getattr(optimizer, "offload_state_to_cpu", None)
                if callable(offload_state):
                    offload_state()
                else:
                    offload_optimizer(optimizer)
            if training_transfer:
                self.release_export_scratch(model_chunks)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
        elif device == "cuda":
            if model:
                load_model_to_gpu(model_chunks, load_grad=grad)
            if (optimizer_state or training_transfer) and optimizer is not None:
                load_state = getattr(optimizer, "load_state_to_device", None)
                if callable(load_state):
                    load_state()
                else:
                    load_optimizer(optimizer)


DEFAULT_RUNTIME_ADAPTER = DefaultOptimizerRuntimeAdapter()


__all__ = ["DEFAULT_RUNTIME_ADAPTER", "DefaultOptimizerRuntimeAdapter"]
