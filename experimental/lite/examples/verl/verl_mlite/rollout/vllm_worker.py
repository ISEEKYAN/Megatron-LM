# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""vLLM checkpoint-format weight reload extension for veRL."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

try:
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
except ModuleNotFoundError:
    vLLMColocateWorkerExtension = object


def _runner_model(model_runner: Any) -> Any:
    get_model = getattr(model_runner, "get_model", None)
    return get_model() if callable(get_model) else model_runner.model


def _runner_model_config(model_runner: Any) -> Any:
    model_config = getattr(model_runner, "model_config", None)
    if model_config is not None:
        return model_config
    return model_runner.vllm_config.model_config


def reload_checkpoint_buckets(
    model_runner: Any,
    receive: Callable[[Callable[[list[tuple[str, Any]]], None]], None],
    *,
    initialize: Callable[[Any], None] | None = None,
    finalize: Callable[[Any, Any], None] | None = None,
) -> None:
    """Stream all IPC buckets through one vLLM layerwise-reload lifecycle."""
    if initialize is None or finalize is None:
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        initialize = initialize or initialize_layerwise_reload
        finalize = finalize or finalize_layerwise_reload

    model = _runner_model(model_runner)
    initialize(model)
    receive(model.load_weights)
    finalize(model, _runner_model_config(model_runner))


class VllmCheckpointWorkerExtension(vLLMColocateWorkerExtension):
    """Consume already-serialized checkpoint tensors without online quantization."""

    def __new__(cls, **kwargs):
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            raise RuntimeError(
                "vllm_checkpoint resync is incompatible with veRL online FP8 quantization; "
                "set VERL_VLLM_FP8_QUANT_ENABLED=0"
            )
        return super().__new__(cls, **kwargs)

    def update_weights_from_ipc(
        self,
        peft_config: dict | None = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
    ) -> None:
        if peft_config or base_sync_done:
            raise NotImplementedError(
                "checkpoint-format resync does not support LoRA delta updates"
            )

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
            BucketedWeightReceiver,
        )

        if self.device is None:
            raise RuntimeError("vLLM worker device is not initialized")
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(), device=self.device, use_shm=use_shm
        )
        reload_checkpoint_buckets(self.model_runner, receiver.receive_weights)


__all__ = ["VllmCheckpointWorkerExtension", "reload_checkpoint_buckets"]
