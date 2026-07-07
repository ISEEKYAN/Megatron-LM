# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Process-wide compatibility hooks for the VERL MLite examples."""

import os

from verl_mlite.compat import _patch_bucketed_weight_sender, apply_runtime_patches

if os.getenv("MLITE_WEIGHT_SYNC_PROBE"):
    # The probe is also used by the Megatron/mbridge control. Keep that path
    # limited to sender instrumentation instead of importing MLite's optional
    # Transformers compatibility layer into every Ray and vLLM process.
    _patch_bucketed_weight_sender()
else:
    apply_runtime_patches()
