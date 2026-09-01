# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native GLM-5 lite implementation."""

__all__ = ["Glm5Model"]


def __getattr__(name: str):
    """Keep protocol-only CPU checks independent of Transformer Engine."""
    if name == "Glm5Model":
        from megatron.lite.model.glm5.lite.model import Glm5Model

        return Glm5Model
    raise AttributeError(name)
