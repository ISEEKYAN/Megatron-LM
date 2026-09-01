from __future__ import annotations

import ast
import inspect
from pathlib import Path

import torch

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm.model import DeepseekV4Model
from megatron.lite.model.deepseek_v4.vllm import protocol


def _sources() -> list[Path]:
    return sorted(Path(inspect.getfile(protocol)).parent.glob("*.py"))


def _tiny_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=4,
        qk_rope_head_dim=2,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        num_hash_layers=1,
        hc_mult=2,
        num_nextn_predict_layers=0,
    )


def test_vllm_implementation_does_not_import_sibling_lite() -> None:
    forbidden = "megatron.lite.model.deepseek_v4." + "lite"
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ] + [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not any(name == forbidden or name.startswith(forbidden + ".") for name in imports), path


def test_model_is_native_composition_not_whole_model_wrapper() -> None:
    forbidden_bases = {
        "DeepseekV4ForCausalLM",
        "DeepseekV4Model",
        "MegatronModel",
        "TransformerLanguageModel",
    }
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                base.id if isinstance(base, ast.Name) else base.attr
                for base in node.bases
                if isinstance(base, (ast.Name, ast.Attribute))
            }
            assert not bases & forbidden_bases, (path, node.name, bases)
    assert DeepseekV4Model.__module__.endswith(".deepseek_v4.vllm.model")


def test_small_model_state_dict_preserves_release_master_dtypes() -> None:
    model = DeepseekV4Model(_tiny_config())
    floating = {
        name: value.dtype
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }
    assert floating
    assert set(floating.values()) == {torch.bfloat16, torch.float32}
    fp32_suffixes = (
        ".hc_fn",
        ".hc_base",
        ".hc_scale",
        ".attn_sink",
        ".compressor.ape",
        ".mlp.gate.expert_bias",
    )
    assert {
        name for name, dtype in floating.items() if dtype == torch.float32
    } == {name for name in floating if name.endswith(fp32_suffixes)}
    assert sum(value.numel() for value in model.state_dict().values()) < 100_000


def test_static_suite_never_constructs_release_dimensions() -> None:
    config = _tiny_config()
    assert config.vocab_size <= 32
    assert config.hidden_size <= 16
    assert config.moe_intermediate_size <= 8
    assert config.num_hidden_layers == 1
    assert config.n_routed_experts <= 4
