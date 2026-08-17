# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fail-closed model identity sidecars for training checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch.nn as nn


IDENTITY_STATE_KEY = "mlite_model_identity"


def model_checkpoint_identity_metadata(
    model: nn.Module | Iterable[nn.Module],
) -> dict[str, Any]:
    """Collect explicit identity contracts from model-owned state modules."""
    chunks = [model] if isinstance(model, nn.Module) else list(model)
    metadata: dict[str, Any] = {}
    for chunk_idx, chunk in enumerate(chunks):
        if not isinstance(chunk, nn.Module):
            raise TypeError("checkpoint model chunks must be nn.Module instances.")
        for module_name, module in chunk.named_modules():
            get_identity = getattr(module, "checkpoint_identity_metadata", None)
            if callable(get_identity):
                key = (
                    f"chunk{chunk_idx}.{module_name}"
                    if module_name
                    else f"chunk{chunk_idx}"
                )
                metadata[key] = get_identity()
    return metadata


def require_checkpoint_identity_match(
    model: nn.Module | Iterable[nn.Module], loaded: Any
) -> None:
    """Reject missing or different identity metadata before tensor restore."""
    expected = model_checkpoint_identity_metadata(model)
    if not expected:
        return
    if loaded != expected:
        raise ValueError(
            "model checkpoint identity mismatch or missing metadata; refusing "
            "to restore tensors against a different model identity contract."
        )


__all__ = [
    "IDENTITY_STATE_KEY",
    "model_checkpoint_identity_metadata",
    "require_checkpoint_identity_match",
]
