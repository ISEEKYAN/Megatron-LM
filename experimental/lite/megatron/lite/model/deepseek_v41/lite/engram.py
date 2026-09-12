# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model-facing Engram assembly interfaces."""
from megatron.lite.primitive.modules.engram_lookup import (
    Engram,
    EngramTable,
    NgramHash,
    build_compressed_token_map,
    hash_multipliers,
    prime_buckets,
)
