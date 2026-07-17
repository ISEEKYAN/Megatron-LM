# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Minimal dense Qwen2 lite model.

This is the first exact-route runtime slice for DeepSeek-R1-Distill-Qwen-1.5B:
TP=1 dense Qwen2 forward/backward with LoRA injection.  HF checkpoint loading,
adapter import/export, OLoRA-tail init, and distributed layouts are intentionally
left to follow-up slices.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.model.qwen2.config import Qwen2Config
from megatron.lite.primitive.modules.delta_mem import (
    DeltaMemConfig,
    DeltaMemory,
    normalize_delta_mem_config,
)
from megatron.lite.primitive.modules.lora import LinearLoRA, LoraConfig, normalize_lora_config
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.parallel import ParallelState


def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _targets_any_lora(lora: LoraConfig, *names: str) -> bool:
    return lora.enabled and any(lora.targets_module(name) for name in names)


def _normalize_delta_mem(
    config: DeltaMemConfig | Mapping[str, Any] | None,
) -> DeltaMemConfig:
    # Convert non-dict Mappings (e.g. OmegaConf DictConfig) before the strict
    # primitive normalizer — the isinstance(dict) trap bit us once (9b9da6d34).
    if (
        config is not None
        and not isinstance(config, (DeltaMemConfig, dict))
        and isinstance(config, Mapping)
    ):
        config = dict(config)
    return normalize_delta_mem_config(config)


def _temperature_to_float(temperature: float | torch.Tensor) -> float:
    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError("Qwen2ForCausalLM supports scalar temperature only.")
        return float(temperature.detach().float().item())
    return float(temperature)


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        *,
        lora_config: LoraConfig | Mapping[str, Any] | None = None,
        delta_mem_config: DeltaMemConfig | Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = config.num_attention_heads * config.head_dim
        self.kv_size = config.num_key_value_heads * config.head_dim
        self.qkv = nn.Linear(
            config.hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=config.attention_bias,
        )
        self.proj = nn.Linear(self.q_size, config.hidden_size, bias=False)

        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / float(config.head_dim)
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        lora = normalize_lora_config(lora_config)
        self.qkv_lora: LinearLoRA | None = None
        self.proj_lora: LinearLoRA | None = None
        if _targets_any_lora(lora, "linear_qkv"):
            self.qkv_lora = LinearLoRA(
                config.hidden_size,
                self.q_size + 2 * self.kv_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                use_rslora=lora.use_rslora,
            )
        if _targets_any_lora(lora, "linear_proj"):
            self.proj_lora = LinearLoRA(
                self.q_size,
                config.hidden_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                use_rslora=lora.use_rslora,
            )

        delta_mem = _normalize_delta_mem(delta_mem_config)
        self.delta_mem: DeltaMemory | None = None
        if delta_mem.enabled:
            self.delta_mem = DeltaMemory(
                config.hidden_size,
                self.q_size,
                config.hidden_size,
                delta_mem,
            )

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        # x: [S, B, H]
        qkv = self.qkv(x)
        if self.qkv_lora is not None:
            qkv = qkv + self.qkv_lora(x)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        delta_o = None
        if self.delta_mem is not None:
            # v1 training wiring = full-sequence steering: a fresh zero state per
            # forward, per layer (the reference's non-default "dialogue" mode).
            # Packed rows are NOT supported — the recurrent state would leak
            # across documents packed into one row; feed padded batches.
            x_bth = x.transpose(0, 1)
            state = self.delta_mem.init_state(x.size(1), device=x.device, dtype=x.dtype)
            delta_q, delta_o, _ = self.delta_mem(x_bth, state)
            if delta_q is not None:
                # Reference injection point: the RAW q-projection output, before
                # head reshape and RoPE (delta_impl.py:1849-1869).
                q = q + delta_q.transpose(0, 1)
        seq_len, batch_size, _ = x.shape
        q = q.view(seq_len, batch_size, self.num_attention_heads, self.head_dim)
        k = k.view(seq_len, batch_size, self.num_key_value_heads, self.head_dim)
        v = v.view(seq_len, batch_size, self.num_key_value_heads, self.head_dim)
        q, k = self._apply_rope(q, k, position_ids)

        if self.num_key_value_heads != self.num_attention_heads:
            repeat = self.num_attention_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        q_bhsd = q.permute(1, 2, 0, 3)
        k_bhsd = k.permute(1, 2, 0, 3)
        v_bhsd = v.permute(1, 2, 0, 3)
        attn = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, is_causal=True)
        attn = attn.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, self.q_size)
        out = self.proj(attn)
        if self.proj_lora is not None:
            out = out + self.proj_lora(attn)
        if delta_o is not None:
            out = out + delta_o.transpose(0, 1)
        return out

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len, batch_size = q.shape[:2]
        if position_ids is None:
            positions = (
                torch.arange(seq_len, device=q.device)
                .view(seq_len, 1)
                .expand(-1, batch_size)
            )
        else:
            positions = position_ids.to(device=q.device)
            if positions.shape == (batch_size, seq_len):
                positions = positions.transpose(0, 1)
            if positions.shape != (seq_len, batch_size):
                raise ValueError(
                    f"position_ids must have shape [B, S] or [S, B], got {tuple(position_ids.shape)}."
                )
        freqs = torch.einsum("sb,d->sbd", positions.float(), self.inv_freq.to(q.device))
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype=q.dtype)
        cos = emb.cos().unsqueeze(2)
        sin = emb.sin().unsqueeze(2)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        *,
        lora_config: LoraConfig | Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.gate_up = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        lora = normalize_lora_config(lora_config)
        self.gate_up_lora: LinearLoRA | None = None
        self.down_lora: LinearLoRA | None = None
        if _targets_any_lora(lora, "linear_fc1"):
            self.gate_up_lora = LinearLoRA(
                config.hidden_size,
                config.intermediate_size * 2,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                use_rslora=lora.use_rslora,
            )
        if _targets_any_lora(lora, "linear_fc2"):
            self.down_lora = LinearLoRA(
                config.intermediate_size,
                config.hidden_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                use_rslora=lora.use_rslora,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up(x)
        if self.gate_up_lora is not None:
            gate_up = gate_up + self.gate_up_lora(x)
        hidden = _swiglu(gate_up)
        out = self.down(hidden)
        if self.down_lora is not None:
            out = out + self.down_lora(hidden)
        return out


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        *,
        lora_config: LoraConfig | Mapping[str, Any] | None = None,
        delta_mem_config: DeltaMemConfig | Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(
            config, lora_config=lora_config, delta_mem_config=delta_mem_config
        )
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(config, lora_config=lora_config)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), position_ids=position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        ps: ParallelState | None = None,
        *,
        lora_config: LoraConfig | Mapping[str, Any] | None = None,
        delta_mem_config: DeltaMemConfig | Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.config = config
        self.ps = ps or ParallelState()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    config, lora_config=lora_config, delta_mem_config=delta_mem_config
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids).transpose(0, 1).contiguous()
        for layer in self.layers:
            hidden = layer(hidden, position_ids=position_ids)
        return self.norm(hidden)


class Qwen2ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        ps: ParallelState | None = None,
        *,
        lora_config: LoraConfig | Mapping[str, Any] | None = None,
        delta_mem_config: DeltaMemConfig | Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.config = config
        self.ps = ps or ParallelState()
        self.model = Qwen2Model(
            config, self.ps, lora_config=lora_config, delta_mem_config=delta_mem_config
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        calculate_entropy: bool = False,
        return_log_probs: bool = True,
    ) -> dict[str, torch.Tensor]:
        hidden_sbh = self.model(input_ids=input_ids, position_ids=position_ids)
        logits_bsv = self.lm_head(hidden_sbh).transpose(0, 1).contiguous()
        output: dict[str, torch.Tensor] = {"hidden_states": hidden_sbh}
        if labels is None:
            output["logits"] = logits_bsv
            return output

        temperature_value = _temperature_to_float(temperature)
        scaled_logits_bsv = logits_bsv.float()
        if temperature_value != 1.0:
            scaled_logits_bsv = scaled_logits_bsv / temperature_value
        token_loss = F.cross_entropy(
            scaled_logits_bsv.view(-1, scaled_logits_bsv.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        if return_log_probs:
            output["log_probs"] = -token_loss
        if calculate_entropy:
            output["entropy"] = vocab_parallel_entropy(
                scaled_logits_bsv, getattr(self.ps, "tp_group", None)
            )
        if loss_mask is not None:
            masked_loss = token_loss * loss_mask.to(dtype=token_loss.dtype)
            denom = loss_mask.to(dtype=token_loss.dtype).sum().clamp_min(1.0)
            output["loss"] = masked_loss.sum() / denom
        else:
            output["loss"] = token_loss.mean()
        return output


__all__ = [
    "Qwen2Attention",
    "Qwen2DecoderLayer",
    "Qwen2ForCausalLM",
    "Qwen2MLP",
    "Qwen2Model",
]
