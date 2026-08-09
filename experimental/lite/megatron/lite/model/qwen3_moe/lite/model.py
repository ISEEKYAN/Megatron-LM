# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native Qwen3MoE: TransformerLayer + Qwen3MoEModel.

Attention and MoE come from primitive/modules; this file only
defines the model-specific composition (Layer stacking, PP layout,
loss computation).
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.gqa import GQAttention
from megatron.lite.primitive.modules.lora import LoraConfig
from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
    EPChunkBackwardOp,
    EPChunkForwardOp,
    EPChunkFusedForwardBackwardOp,
    EPChunkShapeProfile,
    EPChunkWorkspaceKey,
    get_ep_chunk_workspace,
    release_ep_chunk_workspace,
)
from megatron.lite.primitive.modules.moe_ep_chunk_overlap_policy import (
    validate_ep_chunk_overlap_config,
)
from megatron.lite.primitive.modules.router import TopKRouter
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.parallel import (
    ParallelState,
    VanillaColumnParallelLinear,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
    gather_from_sequence_parallel,
    roll_packed_thd_left,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.utils import build_fp8_recipe

# ---------------------------------------------------------------------------
# MoE Layer (thin assembly over megatron.lite.primitive.modules)
# ---------------------------------------------------------------------------


class _Qwen3EPChunkFullRecomputeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, layer: "MoELayer", *params: torch.Tensor):
        # torch.autograd.Function.apply runs forward with grad disabled. Keep this
        # fail-loud guard because the initial full-recompute forward must not build a graph.
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Qwen3 full-recompute custom forward must run with grad disabled"
            )
        del params
        ctx.layer = layer
        ctx.save_for_backward(x.detach())
        assert layer.ep_chunk_forward is not None
        return layer.ep_chunk_forward(x).detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x_saved,) = ctx.saved_tensors
        layer = ctx.layer
        assert layer.ep_chunk_fused is not None
        grad_x, router_grads, expert_grads = layer.ep_chunk_fused.forward_backward(
            x_saved, grad_output
        )
        return grad_x, None, *router_grads, *expert_grads


class MoELayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        ps: ParallelState,
        *,
        use_deepep: bool = True,
        router_bias_rate: float = 0.0,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        enable_ep_chunk_overlap: bool = False,
        ep_chunk_max_token_rows_per_rank: int | None = None,
        ep_chunk_count: int = 2,
        ep_chunk_full_recompute: bool = False,
        lora_config: LoraConfig | dict | None = None,
    ):
        super().__init__()
        validate_qwen3_ep_chunk_recompute_composition(
            enable_ep_chunk_overlap=enable_ep_chunk_overlap,
            ep_chunk_full_recompute=ep_chunk_full_recompute,
            recompute_modules=[],
        )
        validate_ep_chunk_overlap_config(
            enable_ep_chunk_overlap,
            use_deepep=use_deepep,
            ep_size=ps.ep_size,
            topk=config.num_experts_per_tok,
            max_token_rows_per_rank=ep_chunk_max_token_rows_per_rank,
        )
        # Match Qwen3-MoE's `load_balancing_type="none"` setting: no aux loss.
        self.router = TopKRouter(
            config, ps, router_bias_rate=router_bias_rate, compute_aux_loss=False
        )
        self.experts = Experts(
            config,
            ps,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            delay_wgrad_compute=enable_ep_chunk_overlap,
            lora_config=lora_config,
        )
        self.dispatcher: TokenDispatcher | None = None
        self.ep_chunk_forward: EPChunkForwardOp | None = None
        self.ep_chunk_backward: EPChunkBackwardOp | None = None
        self.ep_chunk_fused: EPChunkFusedForwardBackwardOp | None = None
        self.ep_chunk_full_recompute = ep_chunk_full_recompute
        if enable_ep_chunk_overlap:
            assert ep_chunk_max_token_rows_per_rank is not None
            shape_profile = EPChunkShapeProfile.for_two_slot_chunked_ep(
                max_input_rows=ep_chunk_max_token_rows_per_rank,
                hidden_size=config.hidden_size,
                topk=config.num_experts_per_tok,
                ep_size=ps.ep_size,
                chunk_count=ep_chunk_count,
            )
            common_key = dict(
                device_type="cuda",
                # Bind to the actual runtime tensor/explicit materialize device,
                # after the CPU-constructed module has been moved to its rank device.
                device_index=None,
                ep_group_id=id(ps.tp_ep_group),
                dtype=torch.bfloat16,
                shape_profile=shape_profile,
            )

            def workspace(op):
                key = EPChunkWorkspaceKey(op=op, **common_key)
                workspace = get_ep_chunk_workspace(
                    key,
                    lambda _slot: TokenDispatcher(
                        config.num_experts,
                        config.hidden_size,
                        ps,
                        use_deepep=True,
                    ),
                )
                return workspace

            op_kwargs = dict(
                router=self.router,
                experts=self.experts,
            )
            if ep_chunk_full_recompute:
                self.ep_chunk_forward = EPChunkForwardOp(
                    workspace=workspace("forward"), **op_kwargs
                )
                self.ep_chunk_fused = EPChunkFusedForwardBackwardOp(
                    workspace=workspace("fused_forward_backward"), **op_kwargs
                )
            else:
                self.ep_chunk_backward = EPChunkBackwardOp(
                    workspace=workspace("backward"), **op_kwargs
                )
                self.ep_chunk_forward = EPChunkForwardOp(
                    workspace=workspace("forward"),
                    backward_op=self.ep_chunk_backward,
                    **op_kwargs,
                )
        else:
            self.dispatcher = TokenDispatcher(
                config.num_experts, config.hidden_size, ps, use_deepep=use_deepep
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ep_chunk_forward is not None:
            if self.ep_chunk_full_recompute:
                params = tuple(self.router.parameters()) + tuple(
                    self.experts.parameters()
                )
                return _Qwen3EPChunkFullRecomputeFunction.apply(x, self, *params)
            assert self.ep_chunk_forward is not None
            return self.ep_chunk_forward(x)

        assert self.dispatcher is not None
        input_shape = x.shape
        if x.dim() == 3:
            x_2d = x.view(-1, x.size(-1))
        else:
            x_2d = x

        scores, indices = self.router(x_2d)
        dispatched, tpe, permuted_probs = self.dispatcher.dispatch(
            x_2d, scores, indices
        )
        del scores, indices
        self.dispatcher.wait_dispatch_event()
        expert_out = self.experts(
            dispatched,
            tpe,
            permuted_probs,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        del dispatched, tpe, permuted_probs
        combined = self.dispatcher.combine(expert_out)
        del expert_out

        return combined.view(input_shape).to(x.dtype)

    def _ep_chunk_workspaces(self):
        """Return the unique lightweight workspaces selected by Qwen composition."""
        workspaces = []
        for op in (
            self.ep_chunk_forward,
            self.ep_chunk_backward,
            self.ep_chunk_fused,
        ):
            if op is not None and all(
                op.workspace is not workspace for workspace in workspaces
            ):
                workspaces.append(op.workspace)
        return tuple(workspaces)

    def _ep_chunk_workspaces_for_phase(self, phase: str):
        """Select only workspaces first used by one execution phase."""
        return tuple(
            workspace
            for workspace, _require_dispatcher in self._ep_chunk_requirements_for_phase(
                phase
            )
        )

    def _ep_chunk_requirements_for_phase(self, phase: str):
        """Pair phase workspaces with their dispatcher materialization requirement."""
        if phase == "forward":
            requirements = ((self.ep_chunk_forward, True),)
        elif phase == "backward":
            requirements = (
                (self.ep_chunk_fused, True)
                if self.ep_chunk_full_recompute
                else (self.ep_chunk_backward, False),
            )
        else:
            raise ValueError(
                f"Unsupported EP chunk workspace phase {phase!r}; expected 'forward' or 'backward'"
            )
        return tuple(
            (op.workspace, require_dispatcher)
            for op, require_dispatcher in requirements
            if op is not None
        )

    def materialize_ep_chunk_workspaces(
        self,
        *,
        phase: str = "forward",
        device: torch.device | str | None = None,
    ) -> None:
        """Materialize only the workspace needed by the requested execution phase."""
        for workspace, require_dispatcher in self._ep_chunk_requirements_for_phase(
            phase
        ):
            if require_dispatcher:
                workspace.materialize(device=device)
            else:
                workspace.prepare_scratch(device=device)

    def ep_chunk_workspace_evidence(self) -> dict[str, dict]:
        return {
            workspace.key.op: workspace.evidence()
            for workspace in self._ep_chunk_workspaces()
        }

    def release_ep_chunk_workspaces(
        self, *, phase: str | None = None, stream=None
    ) -> None:
        """Release one phase or all selected workspaces for mode switch/teardown."""
        workspaces = (
            self._ep_chunk_workspaces()
            if phase is None
            else self._ep_chunk_workspaces_for_phase(phase)
        )
        for workspace in workspaces:
            release_ep_chunk_workspace(workspace.key, stream=stream)

    def reset_ep_chunk_workspace_tensors(self, *, phase: str, stream=None) -> None:
        """Drop phase scratch at an explicit safe boundary, retaining DeepEP state."""
        for workspace in self._ep_chunk_workspaces_for_phase(phase):
            workspace.reset_tensors(stream=stream)


# ---------------------------------------------------------------------------
# Transformer Layer + Model
# ---------------------------------------------------------------------------

_SP_GRAD_SUFFIXES: tuple[str, ...] = (
    ".attn.qkv.linear.layer_norm_weight",
    ".mlp_norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".moe.router.gate.weight",
    ".enorm.weight",
    ".hnorm.weight",
    ".final_layernorm.weight",
)


def validate_qwen3_ep_chunk_recompute_composition(
    *,
    enable_ep_chunk_overlap: bool,
    ep_chunk_full_recompute: bool,
    recompute_modules: list[str] | tuple[str, ...],
) -> None:
    """Validate Qwen3 recompute composition without leaking policy to primitives."""
    if ep_chunk_full_recompute and not enable_ep_chunk_overlap:
        raise ValueError(
            "ep_chunk_full_recompute=True requires enable_ep_chunk_overlap=True"
        )
    if (
        enable_ep_chunk_overlap
        and not ep_chunk_full_recompute
        and any(module in {"moe", "full"} for module in recompute_modules)
    ):
        raise ValueError(
            "normal ChunkedEP conflicts with outer MoE recompute; enable "
            "ep_chunk_full_recompute or remove moe/full recompute"
        )


def _qwen3_moe_act_recompute_requested(
    recompute_modules: list[str], *, ep_chunk_full_recompute: bool
) -> bool:
    """Keep recompute-policy interpretation in the Qwen composition layer."""
    return (
        "moe_act" in recompute_modules
        and "moe" not in recompute_modules
        and not ep_chunk_full_recompute
    )


def _collect_sp_grad_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect non-TP-sharded params needing coalesced all_reduce after backward."""
    params = []
    for name, p in model.named_parameters():
        if any(name.endswith(s) for s in _SP_GRAD_SUFFIXES) or name == "norm.weight":
            params.append(p)
    return params


class TransformerLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        ps: ParallelState,
        layer_idx: int,
        *,
        use_deepep: bool = True,
        router_bias_rate: float = 0.0,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        use_thd: bool = False,
        enable_ep_chunk_overlap: bool = False,
        ep_chunk_max_token_rows_per_rank: int | None = None,
        ep_chunk_count: int = 2,
        ep_chunk_full_recompute: bool = False,
        lora_config: LoraConfig | dict | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Declaration order follows MC's TransformerLayer (self_attention →
        # pre_mlp_layernorm → mlp). `named_parameters()` iterates in
        # declaration order, and MC's `DistributedDataParallel` lays out
        # gradient buckets by that order; mismatching it changes fp32 master
        # shard layouts and breaks bitwise alignment from step 1 onwards.
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
            lora_config=lora_config,
        )
        self.mlp_norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = MoELayer(
            config,
            ps,
            use_deepep=use_deepep,
            router_bias_rate=router_bias_rate,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            enable_ep_chunk_overlap=enable_ep_chunk_overlap,
            ep_chunk_max_token_rows_per_rank=ep_chunk_max_token_rows_per_rank,
            ep_chunk_count=ep_chunk_count,
            ep_chunk_full_recompute=ep_chunk_full_recompute,
            lora_config=lora_config,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        residual = x
        h = self.attn(x, position_ids=position_ids, packed_seq_params=packed_seq_params)
        x = residual + h

        residual = x
        h = self.mlp_norm(x)
        moe_out = self.moe(h)
        x = residual + moe_out

        return x


class MTPLossAutoScaler(torch.autograd.Function):
    """Attach MTP loss gradients to the main LM hidden state."""

    main_loss_backward_scale: float = 1.0

    @staticmethod
    def forward(ctx, output: torch.Tensor, mtp_loss: torch.Tensor):
        ctx.save_for_backward(mtp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (mtp_loss,) = ctx.saved_tensors
        scaled_mtp_grad = (
            torch.ones_like(mtp_loss) * MTPLossAutoScaler.main_loss_backward_scale
        )
        return grad_output, scaled_mtp_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor | float) -> None:
        if isinstance(scale, torch.Tensor):
            scale = float(scale.detach().float().item())
        MTPLossAutoScaler.main_loss_backward_scale = float(scale)


class MultiTokenPredictionLayer(nn.Module):
    """MCore-style MTP layer for the THD SFT lite path."""

    def __init__(
        self,
        config: Qwen3MoEConfig,
        ps: ParallelState,
        layer_idx: int,
        *,
        embedding: VocabParallelEmbedding,
        use_deepep: bool,
        router_bias_rate: float,
        fp8: bool,
        moe_act_recompute: bool,
        use_thd: bool,
        enable_ep_chunk_overlap: bool,
        ep_chunk_max_token_rows_per_rank: int | None,
        ep_chunk_count: int,
        ep_chunk_full_recompute: bool,
        detach_encoder: bool,
        lora_config: LoraConfig | dict | None,
    ):
        super().__init__()
        self.ps = ps
        self.embedding = embedding
        self.detach_encoder = detach_encoder
        self.enorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = VanillaColumnParallelLinear(
            config.hidden_size * 2,
            config.hidden_size,
            ps,
            sp=ps.tp_size > 1,
            gather_output=True,
        )
        self.transformer_layer = TransformerLayer(
            config,
            ps,
            config.num_hidden_layers + layer_idx,
            use_deepep=use_deepep,
            router_bias_rate=router_bias_rate,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            use_thd=use_thd,
            enable_ep_chunk_overlap=enable_ep_chunk_overlap,
            ep_chunk_max_token_rows_per_rank=ep_chunk_max_token_rows_per_rank,
            ep_chunk_count=ep_chunk_count,
            ep_chunk_full_recompute=ep_chunk_full_recompute,
            lora_config=lora_config,
        )
        self.final_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        hidden_states: torch.Tensor,
        rotary_position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        attention_position_ids = (
            rotary_position_ids if rotary_position_ids is not None else position_ids
        )
        input_ids, _ = roll_packed_thd_left(
            input_ids, packed_seq_params=packed_seq_params, dims=-1
        )
        if position_ids is not None:
            position_ids, _ = roll_packed_thd_left(
                position_ids, packed_seq_params=packed_seq_params, dims=-1
            )
        decoder_input = self.embedding(input_ids)
        decoder_input = scatter_to_sequence_parallel(decoder_input, self.ps)

        if self.detach_encoder:
            decoder_input = decoder_input.detach()
            hidden_states = hidden_states.detach()

        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = torch.cat((decoder_input, hidden_states), dim=-1)
        hidden_states = self.eh_proj(hidden_states)
        hidden_states = scatter_to_sequence_parallel(hidden_states, self.ps)
        hidden_states = self.transformer_layer(
            hidden_states,
            position_ids=attention_position_ids,
            packed_seq_params=packed_seq_params,
        )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, input_ids, position_ids


class MultiTokenPredictionBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        ps: ParallelState,
        *,
        embedding: VocabParallelEmbedding,
        use_deepep: bool,
        router_bias_rate: float,
        fp8: bool,
        moe_act_recompute: bool,
        use_thd: bool,
        enable_ep_chunk_overlap: bool,
        ep_chunk_max_token_rows_per_rank: int | None,
        ep_chunk_count: int,
        ep_chunk_full_recompute: bool,
        detach_encoder: bool,
        repeated_layer: bool,
        lora_config: LoraConfig | dict | None,
    ):
        super().__init__()
        self.num_layers = config.num_nextn_predict_layers
        self.repeated_layer = repeated_layer
        layers_to_build = 1 if repeated_layer else self.num_layers
        self.layers = nn.ModuleList(
            [
                MultiTokenPredictionLayer(
                    config,
                    ps,
                    idx,
                    embedding=embedding,
                    use_deepep=use_deepep,
                    router_bias_rate=router_bias_rate,
                    fp8=fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
                    enable_ep_chunk_overlap=enable_ep_chunk_overlap,
                    ep_chunk_max_token_rows_per_rank=ep_chunk_max_token_rows_per_rank,
                    ep_chunk_count=ep_chunk_count,
                    ep_chunk_full_recompute=ep_chunk_full_recompute,
                    detach_encoder=detach_encoder,
                    lora_config=lora_config,
                )
                for idx in range(layers_to_build)
            ]
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        hidden_states: torch.Tensor,
        packed_seq_params=None,
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        rotary_position_ids = position_ids
        for depth in range(self.num_layers):
            layer = self.layers[0] if self.repeated_layer else self.layers[depth]
            hidden_states, input_ids, position_ids = layer(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                rotary_position_ids=rotary_position_ids,
                packed_seq_params=packed_seq_params,
            )
            outputs.append(hidden_states)
        return outputs


def _temperature_to_float(temperature: float | torch.Tensor) -> float:
    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError(
                "Megatron Lite fused/MTP SFT currently supports scalar temperature only."
            )
        return float(temperature.detach().float().item())
    return float(temperature)


class Qwen3MoEModel(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        ps: ParallelState,
        vpp: int | None = None,
        vpp_chunk_id: int | None = None,
        *,
        use_deepep: bool = False,
        fp8: bool = False,
        recompute_modules: list[str] | None = None,
        router_bias_rate: float = 0.0,
        use_thd: bool = False,
        mtp_enable: bool = False,
        mtp_enable_train: bool = False,
        mtp_detach_encoder: bool = False,
        enable_ep_chunk_overlap: bool = False,
        ep_chunk_max_token_rows_per_rank: int | None = None,
        ep_chunk_count: int = 2,
        ep_chunk_full_recompute: bool = False,
        lora_config: LoraConfig | dict | None = None,
    ):
        super().__init__()
        validate_qwen3_ep_chunk_recompute_composition(
            enable_ep_chunk_overlap=enable_ep_chunk_overlap,
            ep_chunk_full_recompute=ep_chunk_full_recompute,
            recompute_modules=recompute_modules or [],
        )
        self.config = config
        self.ps = ps
        self.fp8 = fp8
        self.mtp_enable_train = bool(mtp_enable and mtp_enable_train)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self._input_tensor: torch.Tensor | None = None
        layout = build_pipeline_chunk_layout(
            config.num_hidden_layers, ps, vpp, vpp_chunk_id
        )
        self.layer_indices = layout.layer_indices
        has_embed = layout.has_embed
        has_head = layout.has_head
        self.pre_process = has_embed
        self.post_process = has_head
        self.share_embeddings_and_output_weights = False

        self.embed: VocabParallelEmbedding | None = None
        if has_embed:
            self.embed = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, ps
            )

        _recompute = recompute_modules or []
        moe_act_recompute = _qwen3_moe_act_recompute_requested(
            _recompute,
            ep_chunk_full_recompute=ep_chunk_full_recompute,
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    config,
                    ps,
                    idx,
                    use_deepep=use_deepep,
                    router_bias_rate=router_bias_rate,
                    fp8=fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
                    enable_ep_chunk_overlap=enable_ep_chunk_overlap,
                    ep_chunk_max_token_rows_per_rank=ep_chunk_max_token_rows_per_rank,
                    ep_chunk_count=ep_chunk_count,
                    ep_chunk_full_recompute=ep_chunk_full_recompute,
                    lora_config=lora_config,
                )
                for idx in self.layer_indices
            ]
        )

        self.norm: nn.Module | None = None
        self.head: VocabParallelOutput | None = None
        if has_head:
            self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)

        self.mtp_embed: VocabParallelEmbedding | None = None
        self.mtp: MultiTokenPredictionBlock | None = None
        if mtp_enable and config.num_nextn_predict_layers > 0 and self.head is not None:
            mtp_embedding = self.embed
            if mtp_embedding is None:
                mtp_embedding = VocabParallelEmbedding(
                    config.vocab_size, config.hidden_size, ps
                )
                self.mtp_embed = mtp_embedding
            self.mtp = MultiTokenPredictionBlock(
                config,
                ps,
                embedding=mtp_embedding,
                use_deepep=use_deepep,
                router_bias_rate=router_bias_rate,
                fp8=fp8,
                moe_act_recompute=moe_act_recompute,
                use_thd=use_thd,
                enable_ep_chunk_overlap=enable_ep_chunk_overlap,
                ep_chunk_max_token_rows_per_rank=ep_chunk_max_token_rows_per_rank,
                ep_chunk_count=ep_chunk_count,
                ep_chunk_full_recompute=ep_chunk_full_recompute,
                detach_encoder=mtp_detach_encoder,
                repeated_layer=config.mtp_use_repeated_layer,
                lora_config=lora_config,
            )

        self.sp_params: list[nn.Parameter] = []
        if ps.tp_size > 1:
            self.sp_params = _collect_sp_grad_params(self)

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError(
                    "Qwen3MoEModel expects a single pipeline input tensor."
                )
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        use_fused_kernels: bool = False,
        calculate_entropy: bool = False,
        return_log_probs: bool = True,
    ) -> dict:
        if self.embed is not None:
            assert input_ids is not None
            h = self.embed(input_ids)
        else:
            if hidden_states is None:
                hidden_states = self._input_tensor
            assert hidden_states is not None
            h = hidden_states

        fp8_ctx = (
            te.fp8_autocast(enabled=True, fp8_recipe=build_fp8_recipe())
            if self.fp8
            else nullcontext()
        )

        with fp8_ctx:
            if self.embed is not None:
                h = scatter_to_sequence_parallel(h, self.ps)
            for layer in self.layers:
                h = layer(
                    h, position_ids=position_ids, packed_seq_params=packed_seq_params
                )
            # Head path is SP-aware: norm runs on SP-sharded [S/tp, B, H] and
            # head's internal all-gather happens inside VocabParallelOutput.
            # Mirrors MC GPTModel's final_layernorm → output_layer(sp=True).

        output = {"hidden_states": h}

        if self.head is not None:
            hidden_for_head = self.norm(h)

            if labels is not None:
                temperature_value = _temperature_to_float(temperature)
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
                    hidden_for_head, mtp_loss = mtp_result
                    output["mtp_loss"] = mtp_loss
                labels_sb = labels.transpose(0, 1).contiguous()
                if use_fused_kernels:
                    hidden_full = gather_from_sequence_parallel(
                        hidden_for_head, self.ps
                    )
                    log_probs, entropy = linear_cross_entropy(
                        hidden_full,
                        self._head_weight_for_fused_ce(hidden_full),
                        labels_sb,
                        temperature_value,
                        self.ps.tp_group,
                    )
                    token_loss = -log_probs
                    output["loss"] = token_loss.mean()
                    if return_log_probs:
                        output["log_probs"] = log_probs.transpose(0, 1).contiguous()
                    if calculate_entropy:
                        output["entropy"] = entropy.transpose(0, 1).contiguous()
                else:
                    logits = self.head(hidden_for_head)
                    if temperature_value != 1.0:
                        logits = logits / temperature_value
                    token_loss = vocab_parallel_cross_entropy(
                        logits, labels_sb, self.ps.tp_group
                    )
                    output["loss"] = token_loss.mean()
                    if return_log_probs:
                        output["log_probs"] = (-token_loss).transpose(0, 1).contiguous()
                    if calculate_entropy:
                        entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
                        output["entropy"] = entropy.transpose(0, 1).contiguous()

            if labels is None:
                logits = self.head(hidden_for_head)
                output["logits"] = self.head.gather(logits)

        return output

    def _apply_mtp_loss(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        labels: torch.Tensor,
        loss_mask: torch.Tensor | None,
        packed_seq_params,
        temperature: float,
        use_fused_kernels: bool,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.mtp is None:
            return None
        if not self.mtp_enable_train:
            return None
        if input_ids is None:
            raise ValueError("MTP training requires input_ids.")
        if loss_mask is None:
            loss_mask = torch.ones_like(labels, dtype=torch.float32)
        else:
            loss_mask = loss_mask.to(dtype=torch.float32)

        mtp_hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
        )

        mtp_labels = labels.clone()
        mtp_loss_mask = loss_mask.clone()
        mtp_loss_values = []
        for mtp_hidden in mtp_hidden_states:
            mtp_labels, _ = roll_packed_thd_left(
                mtp_labels, packed_seq_params=packed_seq_params, dims=-1
            )
            mtp_loss_mask, num_tokens = roll_packed_thd_left(
                mtp_loss_mask, packed_seq_params=packed_seq_params, dims=-1
            )
            labels_sb = mtp_labels.transpose(0, 1).contiguous()
            mask_sb = mtp_loss_mask.transpose(0, 1).contiguous()

            if use_fused_kernels:
                mtp_hidden_full = gather_from_sequence_parallel(mtp_hidden, self.ps)
                log_probs, _entropy = linear_cross_entropy(
                    mtp_hidden_full,
                    self._head_weight_for_fused_ce(mtp_hidden_full),
                    labels_sb,
                    temperature,
                    self.ps.tp_group,
                )
                token_loss = -log_probs
            else:
                logits = self.head(mtp_hidden)
                if temperature != 1.0:
                    logits = logits / temperature
                token_loss = vocab_parallel_cross_entropy(
                    logits, labels_sb, self.ps.tp_group
                )
            token_loss = token_loss * mask_sb.to(dtype=token_loss.dtype)
            num_tokens = num_tokens.to(dtype=token_loss.dtype).clamp_min(1.0)
            mtp_loss_values.append(token_loss.sum() / num_tokens)

            mtp_loss_scale = self.mtp_loss_scaling_factor / max(
                len(mtp_hidden_states), 1
            )
            hidden_states = MTPLossAutoScaler.apply(
                hidden_states, mtp_loss_scale * token_loss / num_tokens
            )

        if not mtp_loss_values:
            return None
        return (
            hidden_states,
            torch.stack([loss.detach().float() for loss in mtp_loss_values]).mean(),
        )

    def _head_weight_for_fused_ce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.head is not None
        weight = self.head.col.linear.weight
        if weight.dtype == hidden_states.dtype:
            return weight
        return weight.to(dtype=hidden_states.dtype)


__all__ = [
    "MoELayer",
    "MTPLossAutoScaler",
    "MultiTokenPredictionBlock",
    "MultiTokenPredictionLayer",
    "Qwen3MoEModel",
    "TransformerLayer",
]
