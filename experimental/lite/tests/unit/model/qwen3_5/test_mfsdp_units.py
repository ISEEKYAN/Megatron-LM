# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.lite.model.qwen3_5.lite.model import Qwen35Layer
from megatron.lite.model.qwen3_5.lite.protocol import _mfsdp_unit_modules
from megatron.lite.primitive.parallel import VocabParallelEmbedding, VocabParallelOutput


def test_qwen35_mfsdp_shards_all_compute_weight_boundaries():
    assert _mfsdp_unit_modules() == (
        Qwen35Layer,
        VocabParallelEmbedding,
        VocabParallelOutput,
    )
