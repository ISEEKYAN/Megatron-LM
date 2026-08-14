"""Native, fail-closed DeepSeek-V4 layer-0 vLLM execution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm.moe import DeepseekV4MoE, MoEKernelMetadata
from megatron.lite.model.deepseek_v4.vllm.primitive import (
    attention_core,
    attach_indexer_aux_loss,
    block_fp8_linear,
    fused_qkv_rms_norm,
    mhc_head,
    mhc_post,
    mhc_post_pre,
    mhc_pre_broadcast,
    o_projection,
)
from megatron.lite.primitive.autograd import inference_only
from megatron.lite.primitive.kernels.vllm_ds4 import (
    CompressorKernelAdapter,
    DS4KVInsertAdapter,
    FlashMLAAdapter,
    FusedQKVRMSNormAdapter,
    IndexerKernelAdapter,
    KVCacheLayout,
    MHCKernel,
    MHCTileLangAdapter,
    OProjectionAdapter,
)
from megatron.lite.primitive.quantization.deployment_block_fp8 import (
    DeploymentBlockFP8Adapter,
)


_STAGES = frozenset({"mhc", "linear", "kv_flashmla", "o_proj", "router_moe", "deepep"})


@dataclass
class AttentionKernelMetadata:
    """Caller-owned cache/workspace data consumed by selected attention layers."""

    positions: torch.Tensor
    slot_mapping: torch.Tensor
    cos_sin_cache: torch.Tensor
    swa_cache: torch.Tensor
    block_size: int
    flash_kind: Literal["prefill", "decode"]
    indices: torch.Tensor
    topk_length: torch.Tensor
    output: torch.Tensor
    kv_workspace: torch.Tensor | None = None
    tile_scheduler_metadata: Any | None = None
    attn_sink: torch.Tensor | None = None
    softmax_scale: float | None = None
    padded_heads: int | None = None
    prepare_flash: Callable[[], None] | None = None
    runtime_layout: Any | None = None
    compressor_operation: Callable[..., Any] | None = None
    compressor_metadata: Any | None = None
    indexer_operation: Callable[..., Any] | None = None
    indexer_metadata: Any | None = None


@dataclass
class AttentionAdapters:
    fused_linear: Callable[[torch.Tensor, nn.Parameter], torch.Tensor]
    q_linear: Callable[[torch.Tensor, nn.Parameter], torch.Tensor]
    norm: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    kv_insert: Callable[..., torch.Tensor]
    flash: FlashMLAAdapter
    o_project: Callable[..., torch.Tensor]
    bf16_linear: Callable[[torch.Tensor, nn.Parameter], torch.Tensor] = F.linear
    fp32_linear: Callable[[torch.Tensor, nn.Parameter], torch.Tensor] = (
        lambda value, weight: F.linear(value.float(), weight.float())
    )
    compressor: CompressorKernelAdapter = field(
        default_factory=CompressorKernelAdapter
    )
    indexer: IndexerKernelAdapter = field(default_factory=IndexerKernelAdapter)


def _default_attention_adapters() -> AttentionAdapters:
    fused = DeploymentBlockFP8Adapter(cache_weight=True)
    q_proj = DeploymentBlockFP8Adapter(cache_weight=True)
    return AttentionAdapters(
        fused_linear=fused,
        q_linear=q_proj,
        bf16_linear=F.linear,
        fp32_linear=lambda value, weight: torch.mm(
            value.contiguous(), weight.T, out_dtype=torch.float32
        ),
        norm=FusedQKVRMSNormAdapter(),
        compressor=CompressorKernelAdapter(),
        indexer=IndexerKernelAdapter(),
        kv_insert=DS4KVInsertAdapter(KVCacheLayout.FP8_DS_MLA),
        flash=FlashMLAAdapter(),
        o_project=OProjectionAdapter(),
    )


class _RMSNormState(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

    def forward(self, value: torch.Tensor, eps: float) -> torch.Tensor:
        return F.rms_norm(value, (value.shape[-1],), self.weight, eps)


class _HyperConnectionState(nn.Module):
    def __init__(self, hidden_size: int, hc_mult: int, *, head: bool = False):
        super().__init__()
        rows = hc_mult if head else (2 + hc_mult) * hc_mult
        scale = 1 if head else 3
        self.hc_fn = nn.Parameter(
            torch.zeros(rows, hc_mult * hidden_size, dtype=torch.float32)
        )
        self.hc_base = nn.Parameter(torch.zeros(rows, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.ones(scale, dtype=torch.float32))


class _CompressorState(nn.Module):
    """Release-compatible masters for one DeepSeek-V4 compressor."""

    def __init__(self, config: DeepseekV4Config, compress_ratio: int, head_dim: int):
        super().__init__()
        coff = 2 if compress_ratio == 4 else 1
        width = coff * head_dim
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, width, dtype=torch.float32),
        )
        self.fused_wkv_wgate = nn.Parameter(
            torch.empty(2 * width, config.hidden_size, dtype=torch.bfloat16)
        )
        self.norm = _RMSNormState(head_dim)


class _IndexerState(nn.Module):
    """Release-compatible layer-2 indexer and nested compressor masters."""

    def __init__(self, config: DeepseekV4Config, compress_ratio: int):
        super().__init__()
        self.wq_b = nn.Parameter(
            torch.empty(
                config.index_n_heads * config.index_head_dim,
                config.q_lora_rank,
                dtype=torch.bfloat16,
            )
        )
        self.weights_proj = nn.Parameter(
            torch.empty(
                config.index_n_heads, config.hidden_size, dtype=torch.bfloat16
            )
        )
        self.compressor = _CompressorState(
            config, compress_ratio, config.index_head_dim
        )


class _AttentionState(nn.Module):
    """BF16 attention masters wired to explicit vLLM deployment kernels."""

    def __init__(
        self,
        config: DeepseekV4Config,
        *,
        layer_idx: int,
        selected_stages: frozenset[str],
        indexer_loss_coeff: float = 0.0,
        adapters: AttentionAdapters | None = None,
    ):
        super().__init__()
        h, qrank, dim = config.hidden_size, config.q_lora_rank, config.head_dim
        self.config = config
        self.layer_idx = layer_idx
        configured_ratio = (
            config.compress_ratios[layer_idx]
            if layer_idx < len(config.compress_ratios)
            else 0
        )
        self.compress_ratio = max(1, configured_ratio)
        self.selected_stages = selected_stages
        self.indexer_loss_coeff = indexer_loss_coeff
        self.fused_wqa_wkv = nn.Parameter(
            torch.empty(qrank + dim, h, dtype=torch.bfloat16)
        )
        self.q_norm = nn.Parameter(torch.ones(qrank, dtype=torch.bfloat16))
        self.kv_norm = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        self.wq_b = nn.Parameter(
            torch.empty(config.num_attention_heads * dim, qrank, dtype=torch.bfloat16)
        )
        # Official release shapes; o_project consumes both BF16 masters and
        # performs ephemeral FP8 packing.
        self.wo_a = nn.Parameter(
            torch.empty(
                config.o_groups * config.o_lora_rank,
                config.num_attention_heads * dim // config.o_groups,
                dtype=torch.bfloat16,
            )
        )
        self.wo_b = nn.Parameter(
            torch.empty(h, config.o_groups * config.o_lora_rank, dtype=torch.bfloat16)
        )
        self.attn_sink = nn.Parameter(
            torch.zeros(config.num_attention_heads, dtype=torch.float32),
            requires_grad=False,
        )
        self.compressor = (
            _CompressorState(config, self.compress_ratio, config.head_dim)
            if self.compress_ratio > 1
            else None
        )
        self.indexer = (
            _IndexerState(config, self.compress_ratio)
            if self.compress_ratio == 4
            else None
        )
        self.adapters = adapters or _default_attention_adapters()
        self._projection_streams: list[torch.cuda.Stream] | None = None
        self._projection_events: list[torch.cuda.Event] | None = None

    def _selected(self, stage: str) -> None:
        if stage not in self.selected_stages:
            raise NotImplementedError(f"layer-0 stage {stage!r} was not executed")

    def _input_projections(self, hidden_states: torch.Tensor):
        def fused_projection():
            return block_fp8_linear(
                self.adapters.fused_linear, hidden_states, self.fused_wqa_wkv
            )

        aux_fns: list[Callable[[], torch.Tensor] | None] = [None, None, None]
        if self.compressor is not None:
            aux_fns[0] = lambda: block_fp8_linear(
                self.adapters.fp32_linear,
                hidden_states,
                self.compressor.fused_wkv_wgate,
            )
        if self.indexer is not None:
            aux_fns[1] = lambda: self.adapters.bf16_linear(
                hidden_states, self.indexer.weights_proj
            )
            aux_fns[2] = lambda: block_fp8_linear(
                self.adapters.fp32_linear,
                hidden_states,
                self.indexer.compressor.fused_wkv_wgate,
            )
        if hidden_states.device.type != "cuda":
            return fused_projection(), tuple(
                function() if function is not None else None for function in aux_fns
            )
        if self._projection_streams is None:
            self._projection_streams = [torch.cuda.Stream() for _ in range(3)]
        if self._projection_events is None:
            self._projection_events = [torch.cuda.Event() for _ in range(4)]
        assert self._projection_events is not None
        from vllm.utils.multi_stream_utils import execute_in_parallel

        return execute_in_parallel(
            fused_projection,
            aux_fns,
            self._projection_events[0],
            self._projection_events[1:],
            self._projection_streams,
            enable=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: AttentionKernelMetadata | None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("layer-0 attention requires flat [tokens, hidden]")
        if metadata is None:
            raise NotImplementedError("layer-0 attention requires explicit metadata")
        self._selected("linear")
        qr_kv, projection_outputs = self._input_projections(hidden_states)
        compressor_kv_score, indexer_weights, indexer_kv_score = projection_outputs
        qr, kv = qr_kv.split([self.config.q_lora_rank, self.config.head_dim], dim=-1)
        qr, kv = fused_qkv_rms_norm(
            self.adapters.norm,
            qr,
            kv,
            self.q_norm,
            self.kv_norm,
            self.config.rms_norm_eps,
        )
        q = (
            block_fp8_linear(self.adapters.q_linear, qr, self.wq_b)
            .view(-1, self.config.num_attention_heads, self.config.head_dim)
            .clone(memory_format=torch.contiguous_format)
        )
        index_q = (
            self.adapters.bf16_linear(qr, self.indexer.wq_b).view(
                -1, self.config.index_n_heads, self.config.index_head_dim
            ).contiguous()
            if self.indexer is not None
            else None
        )
        indexer_topk = None
        if self.compressor is not None:
            if metadata.compressor_operation is None:
                raise NotImplementedError(
                    f"layer {self.layer_idx} compressor requires explicit runtime metadata"
                )
            assert compressor_kv_score is not None
            self.adapters.compressor(
                metadata.compressor_operation,
                compressor_kv_score.detach(),
                metadata.positions,
                self.compressor.ape.detach().float().contiguous(),
                self.compressor.norm.weight.detach(),
                compress_ratio=self.compress_ratio,
                head_dim=self.config.head_dim,
                metadata=metadata.compressor_metadata,
            )
        if self.indexer is not None:
            if metadata.indexer_operation is None:
                raise NotImplementedError(
                    f"layer {self.layer_idx} indexer requires explicit runtime metadata"
                )
            assert indexer_kv_score is not None and indexer_weights is not None
            self.adapters.compressor(
                metadata.compressor_operation,
                indexer_kv_score.detach(),
                metadata.positions,
                self.indexer.compressor.ape.detach().float().contiguous(),
                self.indexer.compressor.norm.weight.detach(),
                compress_ratio=self.compress_ratio,
                head_dim=self.config.index_head_dim,
                metadata=metadata.indexer_metadata,
            )
            assert index_q is not None
            index_result = self.adapters.indexer(
                metadata.indexer_operation,
                qr.detach(),
                index_q.detach(),
                indexer_weights.detach(),
                metadata.positions,
                compress_ratio=self.compress_ratio,
                topk=self.config.index_topk,
                metadata=metadata.indexer_metadata,
            )
            if isinstance(index_result, tuple):
                metadata.indices, metadata.topk_length = index_result
                indexer_topk = index_result[0]
            elif isinstance(index_result, torch.Tensor):
                metadata.indices = index_result
                indexer_topk = index_result

        self._selected("kv_flashmla")
        insert_cache = metadata.swa_cache
        if insert_cache.dtype == torch.uint8 and insert_cache.ndim == 3:
            insert_cache = insert_cache.view(insert_cache.shape[0], -1)
        scale = metadata.softmax_scale or self.config.head_dim**-0.5
        sink = (
            metadata.attn_sink
            if metadata.attn_sink is not None
            else self.attn_sink.float().contiguous()
        )
        if metadata.flash_kind == "prefill":
            if metadata.kv_workspace is None:
                raise NotImplementedError("FlashMLA prefill requires kv_workspace")
            def visible_attention(q_pre, kv_pre):
                q_visible = self.adapters.kv_insert(
                    q_pre,
                    kv_pre,
                    insert_cache,
                    metadata.slot_mapping,
                    metadata.positions,
                    metadata.cos_sin_cache,
                    eps=self.config.rms_norm_eps,
                    block_size=metadata.block_size,
                    padded_heads=(
                        metadata.padded_heads or self.config.num_attention_heads
                    ),
                ).contiguous()
                if metadata.prepare_flash is not None:
                    metadata.prepare_flash()
                flash_result = self.adapters.flash.sparse(
                    q_visible,
                    metadata.kv_workspace,
                    metadata.indices,
                    sm_scale=scale,
                    attn_sink=sink,
                    topk_length=metadata.topk_length,
                    out=metadata.output,
                )
                if not isinstance(flash_result, (tuple, list)) or len(flash_result) < 2:
                    raise RuntimeError(
                        "training FlashMLA prefill must return output and lse"
                    )
                # vLLM sparse prefill returns (output, max_logits, lse), while
                # Lite's differentiable DSA wrapper returns (output, lse).
                # Backward must retain the actual log-sum-exp from the same
                # visible kernel invocation, never max_logits.
                lse = flash_result[2] if len(flash_result) >= 3 else flash_result[1]
                return (
                    flash_result[0],
                    lse,
                    q_visible,
                    metadata.kv_workspace,
                    metadata.indices,
                    metadata.topk_length,
                )

            result = attention_core(
                visible_attention,
                q,
                kv,
                metadata.kv_workspace,
                metadata.indices,
                metadata.topk_length,
                sink,
                metadata.slot_mapping,
                metadata.positions,
                metadata.cos_sin_cache,
                softmax_scale=scale,
                eps=self.config.rms_norm_eps,
                rope_dim=self.config.qk_rope_head_dim,
                compressor_kv_score=compressor_kv_score,
                compressor_ape=(
                    self.compressor.ape if self.compressor is not None else None
                ),
                compressor_norm=(
                    self.compressor.norm.weight
                    if self.compressor is not None
                    else None
                ),
                compressor_ratio=self.compress_ratio,
            )
            if self.indexer is not None and torch.is_grad_enabled():
                assert indexer_topk is not None
                assert index_q is not None
                assert indexer_kv_score is not None and indexer_weights is not None
                assert compressor_kv_score is not None and self.compressor is not None
                result = attach_indexer_aux_loss(
                    result,
                    q,
                    index_q,
                    indexer_kv_score,
                    indexer_weights,
                    compressor_kv_score,
                    self.indexer.compressor.ape,
                    self.indexer.compressor.norm.weight,
                    self.compressor.ape,
                    self.compressor.norm.weight,
                    metadata.positions,
                    metadata.cos_sin_cache,
                    indexer_topk,
                    ratio=self.compress_ratio,
                    rope_dim=self.config.qk_rope_head_dim,
                    eps=self.config.rms_norm_eps,
                    softmax_scale=scale,
                    loss_coeff=self.indexer_loss_coeff,
                )
        else:
            if metadata.tile_scheduler_metadata is None:
                raise NotImplementedError("FlashMLA decode requires scheduler metadata")
            q = self.adapters.kv_insert(
                q,
                kv,
                insert_cache,
                metadata.slot_mapping,
                metadata.positions,
                metadata.cos_sin_cache,
                eps=self.config.rms_norm_eps,
                block_size=metadata.block_size,
                padded_heads=metadata.padded_heads or self.config.num_attention_heads,
            ).contiguous()
            if metadata.prepare_flash is not None:
                metadata.prepare_flash()
            result = self.adapters.flash.paged(
                q.unsqueeze(1),
                metadata.swa_cache.unsqueeze(-2),
                tile_scheduler_metadata=metadata.tile_scheduler_metadata,
                indices=metadata.indices,
                topk_length=metadata.topk_length,
                softmax_scale=scale,
                attn_sink=sink,
                out=metadata.output.unsqueeze(1),
            )
        if isinstance(result, (tuple, list)):
            result = result[0]
        result = result.reshape(hidden_states.shape[0], -1, self.config.head_dim)
        result = result[:, : self.config.num_attention_heads, :]

        self._selected("o_proj")
        heads_per_group = self.config.num_attention_heads // self.config.o_groups
        nope_dim = self.config.head_dim - self.config.qk_rope_head_dim
        output = o_projection(
            lambda o, wa, wb: self.adapters.o_project(
                o,
                metadata.positions,
                metadata.cos_sin_cache,
                wa,
                wb,
                n_groups=self.config.o_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=self.config.qk_rope_head_dim,
                o_lora_rank=self.config.o_lora_rank,
            ),
            result,
            self.wo_a,
            self.wo_b,
            positions=metadata.positions,
            cos_sin_cache=metadata.cos_sin_cache,
            n_groups=self.config.o_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=self.config.qk_rope_head_dim,
            o_lora_rank=self.config.o_lora_rank,
        )
        return output


class DeepseekV4Layer(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
        ps=None,
        layer_idx: int = 0,
        *,
        selected_layer_ids: frozenset[int] = frozenset(),
        selected_module_names: frozenset[str] = frozenset(),
        use_deepep: bool = False,
        indexer_loss_coeff: float = 0.0,
        attention_adapters: AttentionAdapters | None = None,
    ):
        super().__init__()
        selected = layer_idx in selected_layer_ids
        self.config = config
        self.layer_idx = layer_idx
        self.selected_stages = selected_module_names if selected else frozenset()
        self.input_layernorm = _RMSNormState(config.hidden_size)
        self.post_attention_layernorm = _RMSNormState(config.hidden_size)
        self.self_attn = _AttentionState(
            config,
            layer_idx=layer_idx,
            selected_stages=self.selected_stages,
            indexer_loss_coeff=indexer_loss_coeff,
            adapters=attention_adapters,
        )
        self.mlp = DeepseekV4MoE(
            config,
            ps,
            layer_idx=layer_idx,
            selected_stages=self.selected_stages,
            use_deepep=use_deepep,
        )
        self.attn_hc = _HyperConnectionState(config.hidden_size, config.hc_mult)
        self.ffn_hc = _HyperConnectionState(config.hidden_size, config.hc_mult)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_metadata: (
            AttentionKernelMetadata | dict[int, AttentionKernelMetadata] | None
        ) = None,
        moe_metadata: MoEKernelMetadata | dict[int, MoEKernelMetadata] | None = None,
        input_ids: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if "mhc" not in self.selected_stages:
            raise NotImplementedError("layer-0 stage 'mhc' was not executed")
        attn_fn = self.attn_hc.hc_fn.float().contiguous()
        attn_base = self.attn_hc.hc_base.float().contiguous()
        attn_scale = self.attn_hc.hc_scale.float().contiguous()
        fn_broadcast = (
            attn_fn.view(-1, self.config.hc_mult, self.config.hidden_size)
            .sum(dim=1)
            .contiguous()
        )
        if residual is None:
            adapter = MHCTileLangAdapter(MHCKernel.PRE_BROADCAST)
            residual, post_mix, res_mix, hidden_states = mhc_pre_broadcast(
                lambda hidden, fn, scale, base, norm_weight: adapter(
                    hidden,
                    fn,
                    scale,
                    base,
                    self.config.rms_norm_eps,
                    self.config.hc_eps,
                    self.config.hc_eps,
                    2.0,
                    self.config.hc_sinkhorn_iters,
                    norm_weight=norm_weight,
                    norm_eps=self.config.rms_norm_eps,
                    fn_broadcast=fn.view(
                        -1, self.config.hc_mult, self.config.hidden_size
                    ).sum(dim=1).contiguous(),
                ),
                hidden_states,
                attn_fn,
                attn_scale,
                attn_base,
                self.input_layernorm.weight,
                mult=self.config.hc_mult,
                iters=self.config.hc_sinkhorn_iters,
                eps=self.config.hc_eps,
                norm_eps=self.config.rms_norm_eps,
            )
        else:
            if post_mix is None or res_mix is None:
                raise ValueError("cross-layer MHC state requires post_mix and res_mix")
            adapter = MHCTileLangAdapter(MHCKernel.POST_PRE)
            residual, post_mix, res_mix, hidden_states = mhc_post_pre(
                lambda hidden, residual_, post, comb, fn, scale, base, norm_weight: adapter(
                    hidden,
                    residual_,
                    post,
                    comb,
                    fn,
                    scale,
                    base,
                    self.config.rms_norm_eps,
                    self.config.hc_eps,
                    self.config.hc_eps,
                    2.0,
                    self.config.hc_sinkhorn_iters,
                    n_splits=1,
                    tile_n=1,
                    norm_weight=norm_weight,
                    norm_eps=self.config.rms_norm_eps,
                ),
                hidden_states,
                residual,
                post_mix,
                res_mix,
                attn_fn,
                attn_scale,
                attn_base,
                self.input_layernorm.weight,
                mult=self.config.hc_mult,
                iters=self.config.hc_sinkhorn_iters,
                eps=self.config.hc_eps,
                norm_eps=self.config.rms_norm_eps,
            )
        hidden_states = self.self_attn(hidden_states, metadata=attention_metadata)
        adapter = MHCTileLangAdapter(MHCKernel.POST_PRE)
        residual, post_mix, res_mix, hidden_states = mhc_post_pre(
            lambda hidden, residual_, post, comb, fn, scale, base, norm_weight: adapter(
                hidden,
                residual_,
                post,
                comb,
                fn,
                scale,
                base,
                self.config.rms_norm_eps,
                self.config.hc_eps,
                self.config.hc_eps,
                2.0,
                self.config.hc_sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=norm_weight,
                norm_eps=self.config.rms_norm_eps,
            ),
            hidden_states,
            residual,
            post_mix,
            res_mix,
            self.ffn_hc.hc_fn.float().contiguous(),
            self.ffn_hc.hc_scale.float().contiguous(),
            self.ffn_hc.hc_base.float().contiguous(),
            self.post_attention_layernorm.weight,
            mult=self.config.hc_mult,
            iters=self.config.hc_sinkhorn_iters,
            eps=self.config.hc_eps,
            norm_eps=self.config.rms_norm_eps,
        )
        hidden_states = self.mlp(
            hidden_states, input_ids=input_ids, metadata=moe_metadata
        )
        return hidden_states, residual, post_mix, res_mix


class _LMHeadState(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.col = nn.Module()
        self.col.linear = nn.Linear(
            hidden_size, vocab_size, bias=False, dtype=torch.bfloat16
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.col.linear(value)


class DeepseekV4Model(nn.Module):
    """Layer-0 candidate; every production stage must execute explicitly."""

    def __init__(
        self,
        config: DeepseekV4Config,
        train_config=None,
        ps=None,
        *,
        vpp_chunk_id: int | None = None,
        use_thd: bool = False,
        hf_path: str = "",
        attention_backend_override: str | None = None,
        mtp_enable: bool = False,
        mtp_enable_train: bool = False,
        mtp_detach_encoder: bool = False,
        use_deepep: bool = False,
        indexer_loss_coeff: float = 0.0,
        selected_layer_ids: tuple[int, ...] = (),
        selected_module_names: tuple[str, ...] = (),
        attention_adapters: AttentionAdapters | None = None,
    ):
        super().__init__()
        del use_thd, hf_path, attention_backend_override, mtp_enable_train
        del mtp_detach_encoder
        if vpp_chunk_id is not None:
            raise NotImplementedError("The vLLM skeleton does not support VPP chunks.")
        if mtp_enable:
            raise NotImplementedError("The vLLM skeleton does not support MTP.")
        self.config = config
        self.train_config = train_config
        self.ps = ps
        self.layer_indices = list(range(config.num_hidden_layers))
        selected_ids = frozenset(selected_layer_ids)
        self.selected_layer_ids = tuple(sorted(selected_ids))
        aliases = {
            "attn": ("mhc", "linear", "kv_flashmla", "o_proj"),
            "moe": ("router_moe", "deepep"),
        }
        selected_names = frozenset(
            stage
            for name in selected_module_names
            for stage in aliases.get(name, (name,))
        )
        self.prefix_audit = bool(selected_names)
        self.embed_tokens = nn.Module()
        self.embed_tokens.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=torch.bfloat16
        )
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): DeepseekV4Layer(
                    config,
                    ps,
                    layer_idx=layer_idx,
                    selected_layer_ids=selected_ids,
                    selected_module_names=selected_names,
                    use_deepep=use_deepep,
                    indexer_loss_coeff=indexer_loss_coeff,
                    attention_adapters=attention_adapters,
                )
                for layer_idx in self.layer_indices
            }
        )
        self.norm = _RMSNormState(config.hidden_size)
        self.hc_head = _HyperConnectionState(config.hidden_size, config.hc_mult, head=True)
        self.lm_head = _LMHeadState(config.hidden_size, config.vocab_size)
        self.mtp = nn.ModuleList()  # First skeleton iteration intentionally disables MTP.
        self._input_tensor: torch.Tensor | None = None
        self._shared_projection_streams: list[torch.cuda.Stream] | None = None

    def set_input_tensor(self, input_tensor) -> None:
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("DeepseekV4Model expects one input tensor.")
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_metadata: AttentionKernelMetadata | None = None,
        moe_metadata: MoEKernelMetadata | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        calculate_entropy: bool = False,
        **unused,
    ) -> dict[str, torch.Tensor]:
        del unused
        if hidden_states is None:
            if input_ids is None:
                hidden_states = self._input_tensor
            else:
                if os.getenv("MLITE_VALIDATE_INDICES") == "1" and input_ids.numel():
                    minimum, maximum = torch.aminmax(input_ids)
                    if int(minimum.item()) < 0 or int(maximum.item()) >= self.config.vocab_size:
                        raise ValueError(
                            "input token IDs are outside embedding vocabulary: "
                            f"min={int(minimum.item())}, max={int(maximum.item())}, "
                            f"vocab={self.config.vocab_size}"
                        )
                hidden_states = self.embed_tokens.embedding(input_ids)
        if hidden_states is None:
            raise ValueError("input_ids or hidden_states is required.")
        if hidden_states.ndim != 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if hidden_states.device.type == "cuda":
            if self._shared_projection_streams is None:
                self._shared_projection_streams = [
                    torch.cuda.Stream() for _ in range(3)
                ]
            for layer in self.layers.values():
                layer.self_attn._projection_streams = self._shared_projection_streams
        if not self.selected_layer_ids:
            raise NotImplementedError(
                "DeepSeek-V4 vLLM forward requires a non-empty contiguous layer prefix."
            )
        residual = post_mix = res_mix = None
        for layer_idx in self.selected_layer_ids:
            layer = self.layers[str(layer_idx)]
            layer_attention_metadata = (
                attention_metadata.get(layer_idx)
                if isinstance(attention_metadata, dict)
                else attention_metadata
            )
            layer_moe_metadata = (
                moe_metadata.get(layer_idx)
                if isinstance(moe_metadata, dict)
                else moe_metadata
            )
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                attention_metadata=layer_attention_metadata,
                moe_metadata=layer_moe_metadata,
                input_ids=input_ids,
                residual=residual,
                post_mix=post_mix,
                res_mix=res_mix,
            )
        if labels is not None and len(self.selected_layer_ids) != self.config.num_hidden_layers:
            raise ValueError("training loss requires the complete decoder layer set")
        if (
            labels is None
            and len(self.selected_layer_ids) != self.config.num_hidden_layers
        ):
            # Layer-prefix audit mode deliberately stops at the native decoder
            # boundary.  Running the head here would compare a different graph
            # from the corresponding vLLM layer hook.
            return {
                "hidden_states": hidden_states,
                "residual": residual,
                "post_mix": post_mix,
                "res_mix": res_mix,
            }
        post_adapter = MHCTileLangAdapter(MHCKernel.POST)
        hidden_states = mhc_post(
            lambda *args: post_adapter(*args),
            hidden_states,
            residual,
            post_mix,
            res_mix,
        )
        head_adapter = MHCTileLangAdapter(MHCKernel.HEAD)
        hidden_states = mhc_head(
            lambda *args: head_adapter(
                *args, self.config.rms_norm_eps, self.config.hc_eps
            ),
            hidden_states,
            self.hc_head.hc_fn.float().contiguous(),
            self.hc_head.hc_scale.float().contiguous(),
            self.hc_head.hc_base.float().contiguous(),
            eps=self.config.hc_eps,
        )
        hidden_states = self.norm(hidden_states, self.config.rms_norm_eps)
        logits = self.lm_head(hidden_states)
        result = {
            "hidden_states": hidden_states,
            "logits": logits,
        }
        if labels is not None:
            temperature_value = float(
                temperature.detach().float().item()
                if isinstance(temperature, torch.Tensor)
                else temperature
            )
            if temperature_value <= 0:
                raise ValueError("temperature must be positive")
            flat_labels = labels.reshape(-1).long()
            if flat_labels.numel() != logits.shape[0]:
                raise ValueError("labels must contain one target per visible token")
            if os.getenv("MLITE_VALIDATE_INDICES") == "1" and flat_labels.numel():
                minimum, maximum = torch.aminmax(flat_labels)
                if int(minimum.item()) < 0 or int(maximum.item()) >= logits.shape[-1]:
                    raise ValueError(
                        "language-model targets are outside logits vocabulary: "
                        f"min={int(minimum.item())}, max={int(maximum.item())}, "
                        f"vocab={logits.shape[-1]}"
                    )
            scaled_logits = logits.float() / temperature_value
            token_log_probs = F.log_softmax(scaled_logits, dim=-1)
            token_loss = F.nll_loss(token_log_probs, flat_labels, reduction="none")
            result["log_probs"] = -token_loss
            if calculate_entropy:
                result["entropy"] = -(
                    token_log_probs.exp() * token_log_probs
                ).sum(dim=-1)
            mask = (
                torch.ones_like(token_loss)
                if loss_mask is None
                else loss_mask.reshape(-1).to(token_loss.dtype)
            )
            denominator = mask.sum()
            if not bool(denominator > 0):
                raise ValueError("loss_mask must select at least one token")
            result["loss"] = (token_loss * mask).sum() / denominator
        return result


__all__ = [
    "AttentionAdapters",
    "AttentionKernelMetadata",
    "DeepseekV4Layer",
    "DeepseekV4Model",
]
