# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dependency-free vLLM checkpoint reload helpers and proxy extension."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


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


class VllmCheckpointPathWorkerExtension:
    """Minimal vLLM-only extension used by the single-GPU proxy."""

    def __new__(cls, **kwargs):
        del kwargs
        return super().__new__(cls)

    def reload_checkpoint_from_path(self, path: str) -> None:
        """Reload a serialized checkpoint directory for validation or recovery."""
        self.model_runner.reload_weights(weights_path=path, is_checkpoint_format=True)


__all__ = ["VllmCheckpointPathWorkerExtension", "reload_checkpoint_buckets"]
