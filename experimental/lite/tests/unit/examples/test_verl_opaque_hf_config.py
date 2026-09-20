# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import json
import sys
from pathlib import Path

import pytest
from transformers import AutoConfig, GenerationConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "verl"))


def test_nested_text_config_supports_generation_and_roundtrip(monkeypatch, tmp_path):
    from verl_mlite.compat import _register_opaque_hf_config

    monkeypatch.setenv("VERL_MLITE_HF_CONFIG_MODEL_TYPE", "mlite_nested_test")
    config = {
        "model_type": "mlite_nested_test",
        "text_config": {
            "vocab_size": 64,
            "hidden_size": 32,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "custom_architecture": [0, 2, 1],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert _register_opaque_hf_config()
    loaded = AutoConfig.from_pretrained(tmp_path)
    generation = GenerationConfig.from_model_config(loaded)
    assert generation.eos_token_id == 2
    loaded.save_pretrained(tmp_path)
    restored = AutoConfig.from_pretrained(tmp_path)
    for key, value in config["text_config"].items():
        assert restored.to_dict()["text_config"][key] == value


@pytest.mark.parametrize("value", [None, "", "  "])
def test_missing_opaque_config_is_observable(monkeypatch, capsys, value):
    from verl_mlite.compat import (
        _REGISTERED_HF_CONFIG_TYPES,
        _register_opaque_hf_config,
    )

    monkeypatch.delenv("VERL_MLITE_HF_CONFIG_MODEL_TYPE", raising=False)
    if value is not None:
        monkeypatch.setenv("VERL_MLITE_HF_CONFIG_MODEL_TYPE", value)
    before = _REGISTERED_HF_CONFIG_TYPES.copy()
    assert _register_opaque_hf_config() is False
    assert (
        "VERL_MLITE_OPAQUE_CONFIG disabled: VERL_MLITE_HF_CONFIG_MODEL_TYPE is unset"
        in capsys.readouterr().err
    )
    assert _REGISTERED_HF_CONFIG_TYPES == before
