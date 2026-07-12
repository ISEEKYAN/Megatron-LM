# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Legacy delayed-scaling FP8 recipe for pre-blockwise model implementations.

This helper is intentionally **not** part of the shared precision primitive. The
closed blockwise precision system (``megatron.lite.primitive.precision``) owns
the frozen ``Float8BlockScaling`` recipe and routes precision exclusively through
its three closed names. The delayed-scaling ``HYBRID`` recipe below belongs to
the older model implementations (DeepSeek-V4, GLM5, Kimi-K2, Qwen3.5) that gate
FP8 on a ``train_config.fp8`` boolean and have not yet migrated to the closed
profiles. Keeping it here rather than in ``primitive/utils`` prevents a shared
primitive from silently handing back a recipe that violates the blockwise
contract.

Follow-up: migrate the four legacy models onto the closed
``hopper_blockwise_*`` profiles and delete this shim.
"""

from __future__ import annotations


def build_fp8_recipe(train_config=None):
    """Build the legacy TE FP8 recipe (DelayedScaling, HYBRID format, H100)."""
    from transformer_engine.common.recipe import DelayedScaling, Format

    return DelayedScaling(margin=0, fp8_format=Format.HYBRID)


__all__ = ["build_fp8_recipe"]
