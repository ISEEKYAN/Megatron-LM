# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native Hy3 composition over Megatron Lite primitives."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.model.hy3.config import Hy3Config
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.gqa import GQAttention
from megatron.lite.primitive.modules.mlp import SwiGLUMLP
from megatron.lite.primitive.modules.mtp import (
    MTPBlock,
    MTPDecoderLayer,
    MTPLossAutoScaler,
)
from megatron.lite.primitive.modules.router import SigmoidTopKRouter
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.parallel import (
    ParallelState,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
    gather_from_sequence_parallel,
    roll_packed_thd_left,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.utils import build_fp8_recipe


class Hy3MoELayer(nn.Module):
    """Hy3-specific composition of generic routing, expert and MLP primitives."""

    def __init__(
        self,
        config: Hy3Config,
        ps: ParallelState,
        *,
        use_deepep: bool,
        fp8: bool,
        moe_act_recompute: bool,
    ) -> None:
        super().__init__()
        self.combine_in_fp32 = config.enable_moe_fp32_combine
        self.router = SigmoidTopKRouter(
            config,
            ps,
            compute_aux_loss=False,
            expert_bias_persistent=True,
        )
        self.experts = Experts(
            config,
            ps,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
        )
        self.dispatcher = TokenDispatcher(
            config.num_experts,
            config.hidden_size,
            ps,
            use_deepep=use_deepep,
        )
        self.shared_mlp = SwiGLUMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        tokens = x.reshape(-1, x.size(-1))
        shared = self.shared_mlp(tokens)
        scores, indices = self.router(tokens)
        dispatched, tokens_per_expert, permuted_probs = self.dispatcher.dispatch(
            tokens, scores, indices
        )
        self.dispatcher.wait_dispatch_event()
        expert_output = self.experts(
            dispatched,
            tokens_per_expert,
            permuted_probs,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        routed = self.dispatcher.combine(expert_output)
        if self.combine_in_fp32:
            output = (routed.float() + shared.float()).to(x.dtype)
        else:
            output = routed + shared
        return output.reshape(input_shape).to(x.dtype)


class Hy3TransformerLayer(nn.Module):
    def __init__(
        self,
        config: Hy3Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        layer_type: str | None = None,
        use_deepep: bool = False,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        use_thd: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type or config.layer_types[layer_idx]
        self.attn = GQAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            ps=ps,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            use_thd=use_thd,
            qkv_layout="mcore",
        )
        self.mlp_norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp: SwiGLUMLP | None = None
        self.moe: Hy3MoELayer | None = None
        if self.layer_type == "dense":
            self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
        elif self.layer_type == "sparse":
            self.moe = Hy3MoELayer(
                config,
                ps,
                use_deepep=use_deepep,
                fp8=fp8,
                moe_act_recompute=moe_act_recompute,
            )
        else:
            raise ValueError(f"Unsupported Hy3 layer type: {self.layer_type!r}")

    def forward(self, x, position_ids=None, packed_seq_params=None):
        x = x + self.attn(
            x,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )
        normalized = self.mlp_norm(x)
        feed_forward = self.mlp if self.mlp is not None else self.moe
        assert feed_forward is not None
        return x + feed_forward(normalized)


def _collect_sp_grad_params(model: nn.Module) -> list[nn.Parameter]:
    suffixes = (
        ".attn.qkv.linear.layer_norm_weight",
        ".attn.q_norm.weight",
        ".attn.k_norm.weight",
        ".mlp_norm.weight",
        ".moe.router.gate.weight",
        ".moe.router.expert_bias",
        ".enorm.weight",
        ".hnorm.weight",
        ".final_layernorm.weight",
    )
    return [
        parameter
        for name, parameter in model.named_parameters()
        if name == "norm.weight" or any(name.endswith(suffix) for suffix in suffixes)
    ]


class Hy3Model(nn.Module):
    def __init__(
        self,
        config: Hy3Config,
        ps: ParallelState,
        vpp: int | None = None,
        vpp_chunk_id: int | None = None,
        *,
        use_deepep: bool = False,
        fp8: bool = False,
        recompute_modules: list[str] | None = None,
        use_thd: bool = False,
        mtp_enable: bool = False,
        mtp_enable_train: bool = False,
        mtp_detach_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.ps = ps
        self.fp8 = fp8
        self.mtp_enable_train = mtp_enable and mtp_enable_train
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self._input_tensor: torch.Tensor | None = None
        layout = build_pipeline_chunk_layout(
            config.num_hidden_layers, ps, vpp, vpp_chunk_id
        )
        self.layer_indices = layout.layer_indices
        self.embed = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)
            if layout.has_embed
            else None
        )
        recompute = recompute_modules or []
        moe_act_recompute = "moe_act" in recompute and "moe" not in recompute
        self.layers = nn.ModuleList(
            [
                Hy3TransformerLayer(
                    config,
                    ps,
                    index,
                    use_deepep=use_deepep,
                    fp8=fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
                )
                for index in self.layer_indices
            ]
        )
        self.norm = (
            te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if layout.has_head
            else None
        )
        self.head = (
            VocabParallelOutput(config.vocab_size, config.hidden_size, ps)
            if layout.has_head
            else None
        )
        self.mtp_embed: VocabParallelEmbedding | None = None
        self.mtp: MTPBlock | None = None
        if mtp_enable and config.num_nextn_predict_layers > 0 and self.head is not None:
            mtp_embedding = self.embed
            if mtp_embedding is None:
                mtp_embedding = VocabParallelEmbedding(
                    config.vocab_size, config.hidden_size, ps
                )
                self.mtp_embed = mtp_embedding

            def make_mtp_layer(index: int) -> MTPDecoderLayer:
                transformer = Hy3TransformerLayer(
                    config,
                    ps,
                    config.num_hidden_layers + index,
                    layer_type="sparse",
                    use_deepep=use_deepep,
                    fp8=fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
                )
                return MTPDecoderLayer(
                    hidden_size=config.hidden_size,
                    rms_norm_eps=config.rms_norm_eps,
                    ps=ps,
                    embedding=mtp_embedding,
                    transformer_layer=transformer,
                    detach_encoder=mtp_detach_encoder,
                    zero_centered_gamma=False,
                )

            self.mtp = MTPBlock(
                num_layers=config.num_nextn_predict_layers,
                repeated_layer=config.mtp_use_repeated_layer,
                layer_factory=make_mtp_layer,
            )
        self.sp_params = _collect_sp_grad_params(self) if ps.tp_size > 1 else []

    def set_input_tensor(self, input_tensor) -> None:
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("Hy3Model expects one pipeline input tensor")
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def forward(
        self,
        input_ids=None,
        hidden_states=None,
        position_ids=None,
        packed_seq_params=None,
        labels=None,
        loss_mask=None,
        temperature: float | torch.Tensor = 1.0,
        use_fused_kernels: bool = False,
        calculate_entropy: bool = False,
        return_log_probs: bool = True,
    ) -> dict:
        if self.embed is not None:
            if input_ids is None:
                raise ValueError("input_ids are required on the embedding stage")
            hidden = scatter_to_sequence_parallel(self.embed(input_ids), self.ps)
        else:
            hidden = hidden_states if hidden_states is not None else self._input_tensor
            if hidden is None:
                raise ValueError("hidden states are required on a non-embedding stage")
        context = (
            te.fp8_autocast(enabled=True, fp8_recipe=build_fp8_recipe())
            if self.fp8
            else nullcontext()
        )
        with context:
            for layer in self.layers:
                hidden = layer(
                    hidden,
                    position_ids=position_ids,
                    packed_seq_params=packed_seq_params,
                )
        output = {"hidden_states": hidden}
        if self.head is None:
            return output
        assert self.norm is not None
        hidden_for_head = self.norm(hidden)
        if labels is None:
            output["logits"] = self.head.gather(self.head(hidden_for_head))
            return output
        temperature_value = (
            float(temperature.detach().float().item())
            if isinstance(temperature, torch.Tensor)
            else float(temperature)
        )
        mtp_result = self._apply_mtp_loss(
            hidden_for_head,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
            temperature=temperature_value,
            use_fused_kernels=use_fused_kernels,
        )
        if mtp_result is not None:
            hidden_for_head, output["mtp_loss"] = mtp_result
        labels_sb = labels.transpose(0, 1).contiguous()
        if use_fused_kernels:
            hidden_full = gather_from_sequence_parallel(hidden_for_head, self.ps)
            log_probs, entropy = linear_cross_entropy(
                hidden_full,
                self._head_weight(hidden_full),
                labels_sb,
                temperature_value,
                self.ps.tp_group,
            )
            token_loss = -log_probs
        else:
            logits = self.head(hidden_for_head)
            if temperature_value != 1.0:
                logits = logits / temperature_value
            token_loss = vocab_parallel_cross_entropy(
                logits, labels_sb, self.ps.tp_group
            )
            log_probs = -token_loss
            entropy = (
                vocab_parallel_entropy(logits, self.ps.tp_group)
                if calculate_entropy
                else None
            )
        output["loss"] = token_loss.mean()
        if return_log_probs:
            output["log_probs"] = log_probs.transpose(0, 1).contiguous()
        if calculate_entropy and entropy is not None:
            output["entropy"] = entropy.transpose(0, 1).contiguous()
        return output

    def _head_weight(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.head is not None
        weight = self.head.col.linear.weight
        return weight if weight.dtype == hidden.dtype else weight.to(hidden.dtype)

    def _apply_mtp_loss(
        self,
        hidden_states,
        *,
        input_ids,
        position_ids,
        labels,
        loss_mask,
        packed_seq_params,
        temperature,
        use_fused_kernels,
    ):
        if self.mtp is None or not self.mtp_enable_train:
            return None
        if input_ids is None:
            raise ValueError("MTP training requires input_ids")
        mask = (
            torch.ones_like(labels, dtype=torch.float32)
            if loss_mask is None
            else loss_mask.float()
        )
        mtp_hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
        )
        shifted_labels = labels.clone()
        losses = []
        for mtp_hidden in mtp_hidden_states:
            shifted_labels, _ = roll_packed_thd_left(
                shifted_labels, packed_seq_params=packed_seq_params, dims=-1
            )
            mask, num_tokens = roll_packed_thd_left(
                mask, packed_seq_params=packed_seq_params, dims=-1
            )
            labels_sb = shifted_labels.transpose(0, 1).contiguous()
            if use_fused_kernels:
                hidden_full = gather_from_sequence_parallel(mtp_hidden, self.ps)
                log_probs, _ = linear_cross_entropy(
                    hidden_full,
                    self._head_weight(hidden_full),
                    labels_sb,
                    temperature,
                    self.ps.tp_group,
                )
                token_loss = -log_probs
            else:
                assert self.head is not None
                logits = self.head(mtp_hidden)
                token_loss = vocab_parallel_cross_entropy(
                    logits / temperature if temperature != 1.0 else logits,
                    labels_sb,
                    self.ps.tp_group,
                )
            token_loss = token_loss * mask.transpose(0, 1).to(token_loss.dtype)
            denominator = num_tokens.to(token_loss.dtype).clamp_min(1.0)
            losses.append(token_loss.sum() / denominator)
            scale = self.mtp_loss_scaling_factor / max(len(mtp_hidden_states), 1)
            hidden_states = MTPLossAutoScaler.apply(
                hidden_states, scale * token_loss / denominator
            )
        return hidden_states, torch.stack(
            [loss.detach().float() for loss in losses]
        ).mean()


__all__ = ["Hy3MoELayer", "Hy3Model", "Hy3TransformerLayer"]
