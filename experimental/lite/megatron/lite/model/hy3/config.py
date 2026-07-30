# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tencent Hy3 architecture configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from typing import Any

from megatron.lite.primitive.config import load_hf_config_dict


@dataclass
class Hy3Config:
    num_hidden_layers: int = 80
    hidden_size: int = 4096
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 120832
    intermediate_size: int = 13312
    moe_intermediate_size: int = 1536
    num_experts: int = 192
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    router_scaling_factor: float = 2.826
    rms_norm_eps: float = 1e-5
    rope_theta: float = 11_158_840.0
    max_position_embeddings: int = 262144
    hidden_act: str = "silu"
    qk_norm: bool = True
    moe_router_use_sigmoid: bool = True
    moe_router_enable_expert_bias: bool = True
    route_norm: bool = True
    enable_moe_fp32_combine: bool = False
    router_aux_loss_coef: float = 0.0
    num_nextn_predict_layers: int = 1
    mtp_loss_scaling_factor: float = 0.1
    mtp_use_repeated_layer: bool = False
    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = [
                "dense" if index < self.first_k_dense_replace else "sparse"
                for index in range(self.num_hidden_layers)
            ]
        self._validate()

    @property
    def shared_expert_intermediate_size(self) -> int:
        return self.moe_intermediate_size * self.num_shared_experts

    @property
    def n_routed_experts(self) -> int:
        """Compatibility view consumed by the model-agnostic router primitive."""
        return self.num_experts

    def _validate(self) -> None:
        errors: list[str] = []

        def check(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        check(self.hidden_size > 0, "hidden_size must be positive")
        check(self.num_hidden_layers > 0, "num_hidden_layers must be positive")
        check(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "num_attention_heads must be divisible by num_key_value_heads",
        )
        check(self.head_dim > 0, "head_dim must be positive")
        check(
            0 <= self.first_k_dense_replace <= self.num_hidden_layers,
            "first_k_dense_replace is out of range",
        )
        check(
            len(self.layer_types) == self.num_hidden_layers,
            "layer_types length must equal num_hidden_layers",
        )
        check(
            all(kind in {"dense", "sparse"} for kind in self.layer_types),
            "layer_types must contain dense or sparse",
        )
        check(self.num_experts > 0, "num_experts must be positive")
        check(
            1 <= self.num_experts_per_tok <= self.num_experts,
            "num_experts_per_tok is out of range",
        )
        check(self.num_shared_experts == 1, "num_shared_experts must be 1")
        check(self.hidden_act == "silu", "hidden_act must be silu")
        check(self.qk_norm, "qk_norm must be enabled")
        check(self.moe_router_use_sigmoid, "moe_router_use_sigmoid must be enabled")
        check(
            self.moe_router_enable_expert_bias,
            "moe_router_enable_expert_bias must be enabled",
        )
        check(self.route_norm, "route_norm must be enabled")
        check(self.router_scaling_factor > 0, "router_scaling_factor must be positive")
        check(
            self.num_nextn_predict_layers >= 0,
            "num_nextn_predict_layers must be non-negative",
        )
        if errors:
            raise ValueError("Invalid Hy3Config:\n  " + "\n  ".join(errors))

    @classmethod
    def from_hf(cls, path_or_name: str, **overrides) -> "Hy3Config":
        return cls._from_hf_dict(load_hf_config_dict(path_or_name), **overrides)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> "Hy3Config":
        source = (
            hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)
        )
        return cls._from_hf_dict(source, **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict[str, Any], **overrides) -> "Hy3Config":
        if hf.get("model_type", "hy_v3") != "hy_v3":
            raise ValueError(
                f"model_type must be 'hy_v3', got {hf.get('model_type')!r}"
            )
        valid = {item.name for item in dc_fields(cls)}
        kwargs = {key: value for key, value in hf.items() if key in valid}
        rope = hf.get("rope_parameters")
        if isinstance(rope, dict):
            if rope.get("rope_type", "default") != "default":
                raise ValueError("rope_parameters.rope_type must be default")
            kwargs["rope_theta"] = float(rope.get("rope_theta", cls.rope_theta))
        kwargs.update(overrides)
        return cls(**kwargs)


__all__ = ["Hy3Config"]
