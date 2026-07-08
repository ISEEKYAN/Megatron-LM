# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""veRL IPC consumer for serialized vLLM checkpoint weight streams."""

from __future__ import annotations

import os

try:
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
except ModuleNotFoundError:
    vLLMColocateWorkerExtension = object

from verl_mlite.rollout.vllm_worker import reload_checkpoint_buckets


class VllmCheckpointWorkerExtension(vLLMColocateWorkerExtension):
    """Consume serialized checkpoint tensors without veRL online quantization."""

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


__all__ = ["VllmCheckpointWorkerExtension"]
