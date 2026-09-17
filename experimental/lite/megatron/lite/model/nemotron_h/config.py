"""HF architecture mapping for the Nemotron-H Lightning BF16 implementation."""

from __future__ import annotations

from dataclasses import dataclass, fields

from megatron.lite.primitive.config import load_hf_config_dict


@dataclass
class NemotronHConfig:
    hidden_size: int
    vocab_size: int
    layers_block_type: list[str]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    mamba_num_heads: int
    mamba_head_dim: int
    n_groups: int
    ssm_state_size: int
    conv_kernel: int
    chunk_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    moe_shared_expert_intermediate_size: int
    n_shared_experts: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    layer_norm_epsilon: float
    residual_in_fp32: bool
    mlp_hidden_act: str
    mamba_hidden_act: str
    attention_bias: bool
    mlp_bias: bool
    mamba_proj_bias: bool
    use_bias: bool
    use_conv_bias: bool
    tie_word_embeddings: bool
    n_group: int
    topk_group: int

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layers_block_type)

    @property
    def num_experts(self) -> int:
        return self.n_routed_experts

    @property
    def pipeline_hidden_size(self) -> int:
        return 2 * self.hidden_size

    @property
    def mamba_inner_size(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def mamba_conv_dim(self) -> int:
        return self.mamba_inner_size + 2 * self.n_groups * self.ssm_state_size

    @property
    def mamba_in_proj_size(self) -> int:
        return self.mamba_inner_size + self.mamba_conv_dim + self.mamba_num_heads

    @property
    def attention_qkv_size(self) -> int:
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim

    def __post_init__(self):
        if not self.layers_block_type or set(self.layers_block_type) - {
            "linear_attention",
            "full_attention",
            "moe",
            "mlp",
        }:
            raise ValueError("Invalid Nemotron layers_block_type")
        for name in (
            "hidden_size",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "mamba_num_heads",
            "mamba_head_dim",
            "n_groups",
            "ssm_state_size",
            "conv_kernel",
            "chunk_size",
            "n_routed_experts",
            "moe_intermediate_size",
            "n_group",
            "topk_group",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Attention heads must divide into KV groups")
        if self.mamba_num_heads % self.n_groups:
            raise ValueError("Mamba heads must divide into SSM groups")
        if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("Invalid num_experts_per_tok")
        if (
            self.n_routed_experts % self.n_group
            or not 1 <= self.topk_group <= self.n_group
        ):
            raise ValueError("Invalid expert routing groups")
        if self.num_experts_per_tok > self.topk_group * (
            self.n_routed_experts // self.n_group
        ):
            raise ValueError("Selected expert groups cannot supply top-k experts")
        if self.n_shared_experts < 0 or (
            self.n_shared_experts and self.moe_shared_expert_intermediate_size <= 0
        ):
            raise ValueError("Invalid shared expert dimensions")
        if self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")

    @classmethod
    def from_hf(cls, path: str) -> NemotronHConfig:
        return cls._from_hf_dict(load_hf_config_dict(path))

    @classmethod
    def _from_hf_dict(cls, hf: dict) -> NemotronHConfig:
        if hf.get("model_type") != "nemotron_h":
            raise ValueError("Expected model_type=nemotron_h")
        limit = hf.get("time_step_limit", (0.0, float("inf")))
        upper = limit[1]
        if isinstance(upper, dict) and upper == {"__float__": "Infinity"}:
            upper = float("inf")
        if tuple((limit[0], upper)) != (0.0, float("inf")):
            raise ValueError("Aligned SSD currently requires time_step_limit=(0, inf)")
        names = {item.name for item in fields(cls)}
        missing = names - hf.keys()
        if missing:
            raise ValueError(
                f"Missing explicit Nemotron architecture fields: {sorted(missing)}"
            )
        config = cls(**{name: hf[name] for name in names})
        if (
            hf.get("num_hidden_layers", config.num_hidden_layers)
            != config.num_hidden_layers
        ):
            raise ValueError("num_hidden_layers disagrees with layers_block_type")
        return config
