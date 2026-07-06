# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exported router ``expert_bias`` must bypass the opt-in ``export_dtype`` cast.

The bias steers top-k expert selection: hub checkpoints for these families
store it as the only fp32 tensor (HF ``_keep_in_fp32_modules_strict``) and the
load path deliberately restores it fp32, so bf16-rounding it on export flips
routing. Every other floating tensor must still honor the opt-in cast.

The glm5/deepseek_v4 cases install the shared ``transformer_engine_import_stub``
fixture (their ``lite`` package eagerly imports the TE-backed model on import);
kimi_k2 imports TE-free and runs without it.
"""
from types import SimpleNamespace

import torch
import torch.nn as nn


def _single_rank_parallel_state() -> SimpleNamespace:
    return SimpleNamespace(
        pp_size=1, tp_size=1, tp_group=None, ep_size=1, ep_group=None, etp_size=1, etp_group=None
    )


def _bias_pattern(num_experts: int) -> torch.Tensor:
    # Distinctive fp32 values that do NOT survive a bf16 round-trip bitwise.
    return torch.arange(num_experts, dtype=torch.float32) * 1e-3 + 0.05


class _TinyRoutedModule(nn.Module):
    """Exactly what the bias-export loop consumes: a ``layers.0`` submodule
    holding the fp32 router bias buffer, plus a ``norm.weight`` parameter so
    the shared exporter has one ordinary float tensor to cast (the positive
    half of the opt-in cast contract)."""

    def __init__(self, bias_parent_path: str, num_experts: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.layers = nn.ModuleList([nn.Module()])
        parent = self.layers[0]
        for part in bias_parent_path.split("."):
            child = nn.Module()
            setattr(parent, part, child)
            parent = child
        parent.register_buffer("expert_bias", _bias_pattern(num_experts))


def _assert_bias_fp32_and_norm_cast(
    exported: dict[str, torch.Tensor],
    source_bias: torch.Tensor,
    bias_key: str,
    norm_key: str,
) -> None:
    assert bias_key in exported
    bias = exported[bias_key]
    assert bias.dtype == torch.float32
    assert torch.equal(bias, source_bias)
    # The opt-in cast contract must stay intact for every non-bias float.
    assert exported[norm_key].dtype == torch.bfloat16


def test_kimi_k2_export_keeps_expert_bias_fp32_under_bf16_cast() -> None:
    from megatron.lite.model.kimi_k2.config import KimiK2Config
    from megatron.lite.model.kimi_k2.lite.checkpoint import export_hf_weights

    cfg = KimiK2Config(
        num_hidden_layers=1, n_routed_experts=4, num_experts_per_tok=2, first_k_dense_replace=0
    )
    model = _TinyRoutedModule("moe.router", cfg.num_experts)

    exported = dict(
        export_hf_weights(model, cfg, _single_rank_parallel_state(), export_dtype="bfloat16")
    )

    _assert_bias_fp32_and_norm_cast(
        exported,
        model.layers[0].moe.router.expert_bias,
        "model.layers.0.mlp.gate.e_score_correction_bias",
        "model.norm.weight",
    )


def test_glm5_export_keeps_expert_bias_fp32_under_bf16_cast(
    transformer_engine_import_stub,
) -> None:
    transformer_engine_import_stub()
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import export_hf_weights

    cfg = Glm5Config(
        num_hidden_layers=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        num_nextn_predict_layers=0,
    )
    model = _TinyRoutedModule("moe.router", cfg.num_experts)

    exported = dict(
        export_hf_weights(model, cfg, _single_rank_parallel_state(), export_dtype="bfloat16")
    )

    _assert_bias_fp32_and_norm_cast(
        exported,
        model.layers[0].moe.router.expert_bias,
        "model.layers.0.mlp.gate.e_score_correction_bias",
        "model.norm.weight",
    )


def test_deepseek_v4_export_keeps_expert_bias_fp32_under_bf16_cast(
    transformer_engine_import_stub,
) -> None:
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite.checkpoint import export_hf_weights

    cfg = DeepseekV4Config(
        num_hidden_layers=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hash_layers=0,
        num_nextn_predict_layers=0,
    )
    # Synthetic fp32 buffer pins the exporter contract (never cast the bias); real
    # DS4 runtime buffers are bf16 today (SigmoidTopKRouter lacks the _apply fp32 pin).
    model = _TinyRoutedModule("mlp.gate", cfg.num_experts)

    exported = dict(
        export_hf_weights(model, cfg, _single_rank_parallel_state(), export_dtype="bfloat16")
    )

    _assert_bias_fp32_and_norm_cast(
        exported,
        model.layers[0].mlp.gate.expert_bias,
        "layers.0.ffn.gate.bias",
        "norm.weight",
    )
