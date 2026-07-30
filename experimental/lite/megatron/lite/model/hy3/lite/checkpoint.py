# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hy3 HF/native checkpoint mapping."""

from __future__ import annotations

import re

import torch
from torch.distributed.tensor import Replicate, Shard

from megatron.lite.model.hy3.config import Hy3Config
from megatron.lite.primitive.checkpoint_transforms import (
    pack_grouped_query_qkv,
    unpack_grouped_query_qkv,
)


class Hy3WeightSpec:
    def __init__(self, config: Hy3Config):
        self.config = config

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    def _add_attention(
        self,
        weight_map: dict[str, list[str]],
        native_prefix: str,
        hf_prefix: str,
    ) -> None:
        attention = f"{hf_prefix}.self_attn"
        weight_map.update(
            {
                f"{native_prefix}.attn.qkv.linear.layer_norm_weight": [
                    f"{hf_prefix}.input_layernorm.weight"
                ],
                f"{native_prefix}.attn.qkv.linear.weight": [
                    f"{attention}.q_proj.weight",
                    f"{attention}.k_proj.weight",
                    f"{attention}.v_proj.weight",
                ],
                f"{native_prefix}.attn.q_norm.weight": [f"{attention}.q_norm.weight"],
                f"{native_prefix}.attn.k_norm.weight": [f"{attention}.k_norm.weight"],
                f"{native_prefix}.attn.proj.linear.weight": [
                    f"{attention}.o_proj.weight"
                ],
                f"{native_prefix}.mlp_norm.weight": [
                    f"{hf_prefix}.post_attention_layernorm.weight"
                ],
            }
        )

    def _add_sparse_mlp(
        self,
        weight_map: dict[str, list[str]],
        native_prefix: str,
        hf_prefix: str,
    ) -> None:
        mlp = f"{hf_prefix}.mlp"
        weight_map.update(
            {
                f"{native_prefix}.moe.router.gate.weight": [
                    f"{mlp}.router.gate.weight"
                ],
                f"{native_prefix}.moe.router.expert_bias": [f"{mlp}.expert_bias"],
                f"{native_prefix}.moe.shared_mlp.gate_up.linear.weight": [
                    f"{mlp}.shared_mlp.gate_proj.weight",
                    f"{mlp}.shared_mlp.up_proj.weight",
                ],
                f"{native_prefix}.moe.shared_mlp.down.linear.weight": [
                    f"{mlp}.shared_mlp.down_proj.weight"
                ],
            }
        )
        for expert in range(self.config.num_experts):
            weight_map[f"{native_prefix}.moe.experts._fc1_weight_{expert}"] = [
                f"{mlp}.experts.{expert}.gate_proj.weight",
                f"{mlp}.experts.{expert}.up_proj.weight",
            ]
            weight_map[f"{native_prefix}.moe.experts._fc2_weight_{expert}"] = [
                f"{mlp}.experts.{expert}.down_proj.weight"
            ]

    def weight_map(self) -> dict[str, list[str]]:
        config = self.config
        result: dict[str, list[str]] = {
            "embed.embedding.weight": ["model.embed_tokens.weight"],
            "mtp_embed.embedding.weight": ["model.embed_tokens.weight"],
            "norm.weight": ["model.norm.weight"],
            "head.col.linear.weight": ["lm_head.weight"],
        }
        for layer in range(config.num_hidden_layers):
            native = f"layers.{layer}"
            hf = f"model.layers.{layer}"
            self._add_attention(result, native, hf)
            if config.layer_types[layer] == "dense":
                result[f"{native}.mlp.gate_up.linear.weight"] = [
                    f"{hf}.mlp.gate_proj.weight",
                    f"{hf}.mlp.up_proj.weight",
                ]
                result[f"{native}.mlp.down.linear.weight"] = [
                    f"{hf}.mlp.down_proj.weight"
                ]
            else:
                self._add_sparse_mlp(result, native, hf)
        for mtp_index in range(config.num_nextn_predict_layers):
            hf_layer = config.num_hidden_layers + mtp_index
            native = f"mtp.layers.{mtp_index}"
            transformer = f"{native}.transformer_layer"
            hf = f"model.layers.{hf_layer}"
            result.update(
                {
                    f"{native}.enorm.weight": [f"{hf}.enorm.weight"],
                    f"{native}.hnorm.weight": [f"{hf}.hnorm.weight"],
                    f"{native}.eh_proj.linear.weight": [f"{hf}.eh_proj.weight"],
                    f"{native}.final_layernorm.weight": [
                        f"{hf}.final_layernorm.weight"
                    ],
                }
            )
            self._add_attention(result, transformer, hf)
            self._add_sparse_mlp(result, transformer, hf)
        return result

    def hf_to_native(
        self, native_name: str, tensors: list[torch.Tensor]
    ) -> torch.Tensor:
        if len(tensors) == 3:
            return pack_grouped_query_qkv(
                *tensors,
                num_attention_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
            )
        if len(tensors) == 2:
            return torch.cat(tensors, dim=0)
        return tensors[0]

    @staticmethod
    def _canonical_expert_name(native_name: str) -> str:
        """Restore GroupedLinear's local name to the shared weight-map key."""
        return re.sub(
            r"(\.moe\.experts)\.(fc[12])\.weight(\d+)$",
            r"\1._\2_weight_\3",
            native_name,
        )

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        native_name = self._canonical_expert_name(native_name)
        if native_name == "mtp_embed.embedding.weight":
            return []
        targets = self.weight_map().get(native_name)
        if targets is None:
            return [(native_name, tensor)]
        if len(targets) == 3:
            tensors = unpack_grouped_query_qkv(
                tensor,
                num_attention_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
            )
            return list(zip(targets, tensors))
        if len(targets) == 2:
            return list(zip(targets, tensor.chunk(2, dim=0)))
        return [(targets[0], tensor)]

    def qkv_spec(self, native_name: str) -> tuple[int, int, int] | None:
        return None

    def tp_spec(self, native_name: str) -> tuple[int, int] | None:
        if self.is_expert(native_name):
            if "fc1" in native_name:
                return (0, 1)
            if "fc2" in native_name:
                return (1, 1)
            return None
        if "eh_proj" in native_name:
            return (0, 0)
        if "qkv" in native_name and "layer_norm" not in native_name:
            return (0, 0)
        if "attn.proj" in native_name:
            return (1, 0)
        if "gate_up" in native_name:
            return (0, 0)
        if ".down." in native_name:
            return (1, 0)
        if "embed" in native_name or "head" in native_name:
            return (0, 0)
        return None

    def is_expert(self, native_name: str) -> bool:
        return ".experts." in native_name and ".router." not in native_name

    def expert_global_id(self, native_name: str) -> int | None:
        native_name = self._canonical_expert_name(native_name)
        if "_fc1_weight_" in native_name or "_fc2_weight_" in native_name:
            return int(native_name.rsplit("_", 1)[1])
        return None

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        prefix = native_name.rsplit("._fc", 1)[0]
        fc_tag = "fc1" if "_fc1_weight_" in native_name else "fc2"
        return f"{prefix}.{fc_tag}.weight{local_idx}"


def EXPERT_CLASSIFIER(name: str) -> bool:
    return ".experts." in name and ".router." not in name


def PLACEMENT_FN(param_name: str) -> list:
    if EXPERT_CLASSIFIER(param_name):
        if "fc1" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(0)]
        if "fc2" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(1)]
    if "qkv" in param_name and "layer_norm" not in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "attn.proj" in param_name or ".down." in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "gate_up" in param_name or "embed" in param_name or "head" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    return [Replicate(), Replicate(), Replicate(), Replicate()]


def load_hf_weights(model, path: str, config: Hy3Config, ps) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import load_hf_weights as load

    load(model, path, Hy3WeightSpec(config), ps, vocab_size=config.vocab_size)


def export_hf_weights(model, config: Hy3Config, ps, **kwargs):
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights as export

    yield from export(
        model, Hy3WeightSpec(config), ps, vocab_size=config.vocab_size, **kwargs
    )


def save_hf_weights(model, path: str, config: Hy3Config, ps) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import save_hf_weights as save

    save(model, path, Hy3WeightSpec(config), ps, vocab_size=config.vocab_size)


__all__ = [
    "EXPERT_CLASSIFIER",
    "Hy3WeightSpec",
    "PLACEMENT_FN",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
]
