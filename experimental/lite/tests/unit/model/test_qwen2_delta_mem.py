# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""qwen2 dense × δ-mem integration contract tests.

Injection contract (reference delta_impl.py:1849-1869): Δq is added to the RAW
fused-qkv query slice — before head reshape and RoPE; Δo after linear_proj.
v1 training wiring = full-sequence steering with a fresh zero state per forward
(packed rows unsupported; documented in Qwen2Attention.forward).
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.model.qwen2.config import Qwen2Config
from megatron.lite.model.qwen2.lite.model import Qwen2Attention, Qwen2ForCausalLM
from megatron.lite.primitive.modules.delta_mem import (
    DeltaMemory,
    apply_delta_mem_base_slice_init,
)
from megatron.lite.primitive.modules.lora import freeze_non_lora_params

pytestmark = pytest.mark.mlite


def _tiny_config(num_layers=2):
    return Qwen2Config(
        num_hidden_layers=num_layers,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        intermediate_size=32,
        max_position_embeddings=16,
    )


def _paired_models(delta_mem, num_layers=2, seed=7):
    """A base model and a δ-mem model sharing identical base weights."""
    torch.manual_seed(seed)
    base = Qwen2ForCausalLM(_tiny_config(num_layers))
    delta = Qwen2ForCausalLM(_tiny_config(num_layers), delta_mem_config=delta_mem)
    missing, unexpected = delta.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert all("delta_mem" in name for name in missing)
    return base, delta


def test_disabled_config_adds_nothing():
    torch.manual_seed(1)
    model = Qwen2ForCausalLM(_tiny_config(), delta_mem_config={"rank": 0})
    assert model.model.layers[0].self_attn.delta_mem is None
    assert not any("delta_mem" in name for name, _ in model.named_parameters())


def test_zero_output_init_preserves_base_logits_bitwise():
    # rank>0 with the default zero Δ-heads must be a bitwise no-op on the base
    # forward (F.linear against zero weights is exactly zero; x + 0 == x).
    base, delta = _paired_models({"rank": 4})
    input_ids = torch.randint(0, 32, (2, 6), generator=torch.Generator().manual_seed(8))
    out_base = base(input_ids)["logits"]
    out_delta = delta(input_ids)["logits"]
    assert torch.equal(out_base, out_delta)


def test_base_slice_hook_fills_heads_from_fused_slices():
    _, delta = _paired_models({"rank": 4, "output_init": "base_slice_fixed"}, num_layers=2)
    stats = apply_delta_mem_base_slice_init(delta)
    assert stats == {"delta_mem_base_slice_inits": 2}
    attn = delta.model.layers[0].self_attn
    adapter = attn.delta_mem
    width = min(8, 4, attn.qkv.weight.shape[1])
    expected_q = torch.nn.functional.normalize(
        attn.qkv.weight[: attn.q_size, :width].float(), dim=0, eps=1e-6
    ) * 0.05
    expected_o = torch.nn.functional.normalize(
        attn.proj.weight[:, :width].float(), dim=0, eps=1e-6
    ) * 0.05
    assert torch.equal(adapter.delta_q_proj[:, :width], expected_q)
    assert torch.equal(adapter.delta_o_proj[:, :width], expected_o)


def test_first_token_invariant_and_later_tokens_steered():
    # Causal attention + S_0 = 0 + read-before-write: position 0 must be bitwise
    # identical to the base model even with non-zero Δ-heads; later positions
    # must actually be steered.
    base, delta = _paired_models({"rank": 4, "output_init": "base_slice_fixed"})
    apply_delta_mem_base_slice_init(delta)
    input_ids = torch.randint(0, 32, (2, 5), generator=torch.Generator().manual_seed(9))
    out_base = base(input_ids)["logits"]
    out_delta = delta(input_ids)["logits"]
    assert torch.equal(out_base[:, 0], out_delta[:, 0])
    assert torch.count_nonzero(out_base[:, 1:] - out_delta[:, 1:]) > 0


def test_freeze_marks_delta_mem_trainable_and_grads_flow():
    _, delta = _paired_models({"rank": 4, "output_init": "base_slice_fixed"})
    apply_delta_mem_base_slice_init(delta)
    stats = freeze_non_lora_params(delta)
    per_layer = sum(
        p.numel() for p in delta.model.layers[0].self_attn.delta_mem.parameters()
    )
    assert stats["lora_numel"] == per_layer * 2
    for name, param in delta.named_parameters():
        assert param.requires_grad == ("delta_mem" in name)

    input_ids = torch.randint(0, 32, (1, 6), generator=torch.Generator().manual_seed(10))
    labels = torch.randint(0, 32, (1, 6), generator=torch.Generator().manual_seed(11))
    delta(input_ids, labels=labels)["loss"].backward()
    attn = delta.model.layers[0].self_attn
    for name in ("memory_q_proj", "memory_k_proj", "memory_v_proj", "beta_proj",
                 "delta_q_proj", "delta_o_proj"):
        grad = getattr(attn.delta_mem, name).grad
        assert grad is not None and torch.count_nonzero(grad) > 0, name
    assert attn.qkv.weight.grad is None


def test_attention_level_injection_matches_manual_composition():
    # White-box: the attention forward with δ-mem equals base attention run on a
    # manually Δq-patched qkv plus Δo — pinning the pre-RoPE injection point.
    torch.manual_seed(12)
    cfg = _tiny_config(num_layers=1)
    attn = Qwen2Attention(cfg, delta_mem_config={"rank": 4, "output_init": "base_slice_fixed"})
    apply_delta_mem_base_slice_init(attn)  # helper walks any module tree
    x = torch.randn(5, 2, cfg.hidden_size, generator=torch.Generator().manual_seed(13))

    out = attn(x)

    x_bth = x.transpose(0, 1)
    state = attn.delta_mem.init_state(2, dtype=x.dtype)
    delta_q, delta_o, _ = attn.delta_mem(x_bth, state)
    ref = Qwen2Attention(cfg)
    ref.load_state_dict(attn.state_dict(), strict=False)
    qkv = ref.qkv(x)
    q, k, v = torch.split(qkv, [ref.q_size, ref.kv_size, ref.kv_size], dim=-1)
    q = q + delta_q.transpose(0, 1)
    s, b = x.shape[:2]
    q = q.view(s, b, ref.num_attention_heads, ref.head_dim)
    k = k.view(s, b, ref.num_key_value_heads, ref.head_dim)
    v = v.view(s, b, ref.num_key_value_heads, ref.head_dim)
    q, k = ref._apply_rope(q, k, None)
    repeat = ref.num_attention_heads // ref.num_key_value_heads
    k = k.repeat_interleave(repeat, dim=2)
    v = v.repeat_interleave(repeat, dim=2)
    a = torch.nn.functional.scaled_dot_product_attention(
        q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3), v.permute(1, 2, 0, 3), is_causal=True
    )
    a = a.permute(2, 0, 1, 3).contiguous().view(s, b, ref.q_size)
    expected = ref.proj(a) + delta_o.transpose(0, 1)
    assert torch.equal(out, expected)
