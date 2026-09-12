# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config


@pytest.mark.parametrize('field', ['kv_source_layer_ids', 'index_source_layer_ids',
    'candidate_source_layer_id', 'compress_ratios', 'num_hidden_layers'])
def test_topology_guards(model_config, field):
    config = model_config.to_hf_dict()
    value = config['text_config'][field]
    config['text_config'][field] = value[:-1] if isinstance(value, list) else value + 1
    with pytest.raises(ValueError, match=field):
        DeepseekV41Config(config)
