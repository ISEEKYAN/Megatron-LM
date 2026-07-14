# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dependency-free vLLM checkpoint reload helpers and proxy extension."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from verl_mlite.rollout.layer_cluster import LayerClusterBuffer


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
    # IPC byte buckets may split one HF layer across multiple ZMQ rounds.
    # vLLM's layerwise reload defers processing until each submodule has received
    # all of its weights; feeding multiple layers in one load_weights call leaves
    # many submodules in the deferred staging state at once (r1-r11 receiver peak).
    cluster = LayerClusterBuffer(model.load_weights)
    receive(cluster.ingest_bucket)
    cluster.finalize()
    finalize(model, _runner_model_config(model_runner))


def _tensor_sha256(tensor: Any, *, chunk_bytes: int) -> str:
    """Hash one device tensor without materializing the full value on the host."""
    import torch

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    value = tensor.detach()
    if value.device.type == "meta":
        raise ValueError("cannot fingerprint a meta tensor")
    if not value.is_contiguous():
        value = value.contiguous()
    byte_view = value.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, byte_view.numel(), chunk_bytes):
        host = byte_view[start : start + chunk_bytes].cpu().numpy()
        digest.update(host.tobytes())
    return digest.hexdigest()


def checkpoint_state_fingerprints(
    model_runner: Any,
    *,
    chunk_bytes: int = 64 * 1024**2,
) -> list[dict[str, Any]]:
    """Return deterministic fingerprints for all model parameters and buffers."""
    model = _runner_model(model_runner)
    state = [
        *(("parameter", name, tensor) for name, tensor in model.named_parameters()),
        *(("buffer", name, tensor) for name, tensor in model.named_buffers()),
    ]
    records = []
    for kind, name, tensor in sorted(state, key=lambda item: (item[1], item[0])):
        records.append(
            {
                "name": name,
                "kind": kind,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "nbytes": tensor.numel() * tensor.element_size(),
                "sha256": _tensor_sha256(tensor, chunk_bytes=chunk_bytes),
            }
        )
    return records


class VllmCheckpointPathWorkerExtension:
    """Minimal vLLM-only extension used by the single-GPU proxy."""

    def __new__(cls, **kwargs):
        del kwargs
        return super().__new__(cls)

    def reload_checkpoint_from_path(self, path: str) -> None:
        """Reload a serialized checkpoint directory for validation or recovery."""
        self.model_runner.reload_weights(weights_path=path, is_checkpoint_format=True)

    def checkpoint_state_fingerprints(self) -> list[dict[str, Any]]:
        """Fingerprint the loaded TP shard for cold-vs-online validation."""
        return checkpoint_state_fingerprints(self.model_runner)


__all__ = [
    "VllmCheckpointPathWorkerExtension",
    "checkpoint_state_fingerprints",
    "reload_checkpoint_buckets",
]
