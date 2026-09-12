# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

REFERENCE_SHA256 = {
    'model.py': '4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65',
    'kernel.py': '1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455',
    'engram.py': '11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897',
    'vision.py': '5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c',
    'image_processor.py': '482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272',
    'inference_config.json': '2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809',
    'config.json': '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879',
}


@pytest.fixture
def official(monkeypatch):
    root = Path(os.environ['DS41_REFERENCE_DIR'])
    for name, digest in REFERENCE_SHA256.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name

    def load(name, cls=None, method=None):
        path = root / name
        if cls:
            def named(nodes, kind, label):
                return next(n for n in nodes if isinstance(n, kind) and n.name == label)

            node = named(ast.parse(path.read_text()).body, ast.ClassDef, cls)
            node = named(node.body, ast.FunctionDef, method)
            namespace = {'torch': torch, 'nn': torch.nn}
            tree = ast.Module(body=[node], type_ignores=[])
            exec(compile(tree, str(path), 'exec'), namespace)
            return namespace[method]
        spec = importlib.util.spec_from_file_location('official_' + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def moe(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v41.lite import moe

    return moe


@pytest.fixture
def model_config():
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    release = json.loads('''
    {"model_type": "deepseek_v41", "text_config": {"vocab_size": 256, "hidden_size": 32, "num_hidden_layers": 40,
     "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
     "qk_rope_head_dim": 4, "q_lora_rank": 32, "o_lora_rank": 4, "o_groups": 2,
     "index_n_heads": 2, "index_head_dim": 32, "index_topk": 2, "sliding_window": 3,
     "candidate_topk_blocks": 1, "candidate_block_size": 2, "candidate_source_layer_id": 20,
     "kv_source_layer_ids": [2,8,14,20], "index_source_layer_ids": [2,8,14,20,24,28,32,36],
     "hidden_act": "silu", "attention_bias": false, "attention_dropout": 0.0,
     "tie_word_embeddings": false, "norm_topk_prob": true, "topk_method": "noaux_tc",
     "rope_theta": 10000, "compress_rope_theta": 160000,
     "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32,
                      "beta_slow": 1, "original_max_position_embeddings": 65536},
     "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
     "moe_intermediate_size": 32, "scoring_func": "sqrtsoftplus",
     "routed_scaling_factor": 1.5, "swiglu_limit": 10.0, "rms_norm_eps": 1e-20,
     "hc_mult": 3, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
     "engram_layer_ids": [1,14], "engram_max_ngram_size": 3, "engram_n_heads": 2,
     "engram_vocab_size": 31, "engram_num_embeddings": [152,220], "engram_head_dim": 32,
     "engram_compressed_vocab_size": 256, "engram_pad_token_id": 2,
     "num_nextn_predict_layers": 3, "dspark_n_routed_experts": 4},
     "vision_config": {"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
                       "intermediate_size": 32, "patch_size": 14, "rope_theta": 10000,
                       "downsample_ratio": 3},
     "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                             "weight_block_size": [32,32], "scale_fmt": "ue8m0",
                             "expert_dtype": "fp4"}}
    ''')
    release['text_config']['compress_ratios'] = [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
    return DeepseekV41Config(release)
