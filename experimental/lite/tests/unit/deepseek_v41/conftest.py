# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from pathlib import Path
import os, json, hashlib

@pytest.fixture
def moe(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v41.lite import moe

    return moe



@pytest.fixture
def model_config():
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.model.deepseek_v41.lite.engram import prime_buckets
    path = Path(os.environ.get('DS41_REFERENCE_DIR', '/tmp/ds41-fixture-reference')) / 'config.json'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879'
    cfg = json.loads(raw)
    cfg['text_config'].update(hidden_size=128, vocab_size=256, num_attention_heads=8,
        head_dim=64, qk_rope_head_dim=64, q_lora_rank=64, o_lora_rank=32,
        index_n_heads=4, index_head_dim=64, n_routed_experts=8, moe_intermediate_size=64,
        engram_head_dim=32, engram_vocab_size=31, engram_compressed_vocab_size=256,
        engram_num_embeddings=prime_buckets([1, 14], 4, 8, 31).flatten(1).sum(1).tolist())
    cfg['vision_config'].update(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128)
    return DeepseekV41Config(cfg)
