# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""V4.1 layer policy over shared CSA2 computation."""
from megatron.lite.primitive.modules.csa2 import AttentionState, Compressor
from megatron.lite.primitive.modules.csa2 import CSA2Attention as _CSA2Attention
from megatron.lite.primitive.modules.csa2 import CSA2Config, Indexer, Linear, rotate


class CSA2Attention(_CSA2Attention):
    def __init__(self, config, layer_id):
        if not 0 <= layer_id < 40:
            raise ValueError(
                "Only the 40 backbone layers are supported; DSpark is excluded"
            )
        owner = (
            20
            if layer_id >= 20
            else 14 if layer_id >= 14 else 8 if layer_id >= 8 else 2
        )
        super().__init__(
            config,
            layer_id,
            ratio=0 if layer_id < 2 else 2 if layer_id < 20 else 1,
            kv_owner=owner,
            index_owner=20 + ((layer_id - 20) // 4) * 4 if layer_id >= 20 else owner,
            candidate_mode=(
                'reuse' if layer_id > 20 else 'build' if layer_id == 20 else 'none'
            ),
        )
