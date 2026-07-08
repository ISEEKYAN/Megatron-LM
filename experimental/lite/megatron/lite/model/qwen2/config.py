# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense Qwen2 architecture config for the exact Fig.14 PEFT target family."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any

from megatron.lite.primitive.config import load_hf_config_dict


@dataclass
class Qwen2Config:
    """Pure dense Qwen2 architecture parameters.

    The defaults are intentionally small enough to be ordinary Qwen2-like
    values; paper-target values are read from the pinned HF config snapshot.
    Implementation knobs live in a future dense-Qwen2 protocol, not here.
    """

    num_hidden_layers: int = 28
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 151936
    intermediate_size: int = 8960
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131072
    attention_dropout: float = 0.0
    attention_bias: bool = True
    tie_word_embeddings: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    max_window_layers: int | None = 21
    torch_dtype: str | None = "bfloat16"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        errors: list[str] = []

        def _check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        _check(self.num_hidden_layers >= 1, "num_hidden_layers must be >= 1")
        _check(self.hidden_size > 0, "hidden_size must be > 0")
        _check(self.num_attention_heads >= 1, "num_attention_heads must be >= 1")
        _check(self.num_key_value_heads >= 1, "num_key_value_heads must be >= 1")
        _check(
            self.hidden_size % self.num_attention_heads == 0,
            "hidden_size must be divisible by num_attention_heads",
        )
        _check(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "num_attention_heads must be divisible by num_key_value_heads",
        )
        _check(self.head_dim > 0, "head_dim must be > 0")
        _check(
            self.head_dim * self.num_attention_heads == self.hidden_size,
            "head_dim * num_attention_heads must equal hidden_size",
        )
        _check(self.intermediate_size > 0, "intermediate_size must be > 0")
        _check(self.vocab_size > 0, "vocab_size must be > 0")
        _check(self.rope_theta > 0, "rope_theta must be > 0")
        _check(self.rms_norm_eps > 0, "rms_norm_eps must be > 0")
        _check(self.max_position_embeddings > 0, "max_position_embeddings must be > 0")
        _check(
            0.0 <= self.attention_dropout < 1.0,
            "attention_dropout must be in [0, 1)",
        )
        _check(isinstance(self.attention_bias, bool), "attention_bias must be a boolean")
        if self.sliding_window is not None:
            _check(self.sliding_window > 0, "sliding_window must be > 0 when set")
        if self.max_window_layers is not None:
            _check(self.max_window_layers >= 0, "max_window_layers must be >= 0 when set")

        if errors:
            raise ValueError(
                f"Invalid Qwen2Config ({len(errors)} errors):\n  " + "\n  ".join(errors)
            )

    @property
    def qkv_size(self) -> int:
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in dc_fields(self)}

    @classmethod
    def from_hf(cls, path_or_name: str, **overrides) -> Qwen2Config:
        hf_dict = load_hf_config_dict(path_or_name)
        return cls._from_hf_dict(hf_dict, **overrides)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> Qwen2Config:
        hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)
        return cls._from_hf_dict(hf_dict, **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict[str, Any], **overrides) -> Qwen2Config:
        valid_fields = {field.name for field in dc_fields(cls)}
        kwargs = {key: value for key, value in hf.items() if key in valid_fields}

        if "rope_theta" not in kwargs:
            rope_scaling = hf.get("rope_parameters")
            if isinstance(rope_scaling, dict) and "rope_theta" in rope_scaling:
                kwargs["rope_theta"] = float(rope_scaling["rope_theta"])
        if "head_dim" not in kwargs or kwargs["head_dim"] is None:
            hidden_size = kwargs.get("hidden_size", cls.hidden_size)
            num_heads = kwargs.get("num_attention_heads", cls.num_attention_heads)
            kwargs["head_dim"] = hidden_size // num_heads

        kwargs.update(overrides)
        return cls(**kwargs)
