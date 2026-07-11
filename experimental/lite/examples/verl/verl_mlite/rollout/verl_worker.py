# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""veRL IPC consumer for serialized vLLM checkpoint weight streams."""

from __future__ import annotations

import os

try:
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
except ModuleNotFoundError:
    vLLMColocateWorkerExtension = object

from verl_mlite.rollout.vllm_worker import reload_checkpoint_buckets

_UPSTREAM_UPDATE_WEIGHTS_FROM_IPC = getattr(
    vLLMColocateWorkerExtension, "update_weights_from_ipc", None
)


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
        if peft_config:
            raise NotImplementedError(
                "checkpoint-format resync does not support PEFT/LoRA weight updates"
            )
        if self.device is None:
            raise RuntimeError("vLLM worker device is not initialized")
        if _UPSTREAM_UPDATE_WEIGHTS_FROM_IPC is None:
            raise RuntimeError("veRL vLLM IPC weight synchronization is unavailable")

        def receive(load_bucket) -> None:
            self._checkpoint_load_bucket = load_bucket
            try:
                _UPSTREAM_UPDATE_WEIGHTS_FROM_IPC(
                    self,
                    peft_config=None,
                    base_sync_done=base_sync_done,
                    use_shm=use_shm,
                )
            finally:
                del self._checkpoint_load_bucket

        reload_checkpoint_buckets(self.model_runner, receive)

    def _update_weights(
        self,
        weights,
        peft_config: dict | None,
        base_sync_done: bool,
    ) -> None:
        del base_sync_done
        if peft_config:
            raise NotImplementedError(
                "checkpoint-format resync does not support PEFT/LoRA weight updates"
            )
        load_bucket = getattr(self, "_checkpoint_load_bucket", None)
        if load_bucket is None:
            raise RuntimeError("checkpoint reload lifecycle is not initialized")
        load_bucket(weights)


__all__ = ["VllmCheckpointWorkerExtension"]
