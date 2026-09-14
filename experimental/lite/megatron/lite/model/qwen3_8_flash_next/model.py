# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native text decoder composed from MLite GDN/MoE and Qwen HC/QSA/PLE.

Layer order and GDN normalization follow NVIDIA-NeMo/Automodel #3690,
5cfe13b160eb7e23ac5a4868bbf611707cdf98fb (fetched 2026-09-13).
"""

from copy import copy

import torch
from megatron.lite.model.qwen3_5.lite.model import MoELayer
from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet
from megatron.lite.primitive.parallel import ColumnParallelLinear
from megatron.lite.primitive.utils.packed_seq import PackedSeqParams
from torch import nn
from torch.nn import functional as F

from .cp import ContiguousGDNHeadTransport
from .engram import (
    PRIMES,
    Qwen3_8_FlashNextEngramTableConfig,
    Qwen3_8_FlashNextNGramEmbedding,
    Qwen3_8_FlashNextPLELayer,
)
from .math import Qwen3_8_FlashNextHyperConnection as HyperConnection
from .qsa import Qwen3_8_FlashNextQSAAttention


class Qwen38GatedDeltaNet(ContiguousGDNHeadTransport, GatedDeltaNet):
    def __init__(self, config, ps):
        fields = (
            'hidden_size',
            'linear_num_key_heads',
            'linear_key_head_dim',
            'linear_num_value_heads',
            'linear_value_head_dim',
            'linear_conv_kernel_dim',
            'rms_norm_eps',
        )
        super().__init__(**{k: getattr(config, k) for k in fields}, ps=ps)
        if ps.cp_size > 1 and (
            config.linear_num_key_heads % ps.cp_size
            or config.linear_num_value_heads % ps.cp_size
        ):
            raise ValueError('CP_GDN_HEAD_OWNERSHIP')
        # HC already supplies normalized inputs. Preserve the shared recurrence,
        # convolution and projection layout, with no second input RMSNorm.
        self.in_proj = ColumnParallelLinear(config.hidden_size, self.in_proj_dim, ps)
        self.norm = nn.Module()
        self.norm.register_parameter('weight', nn.Parameter(torch.ones(self.dv)))
        self.eps = config.rms_norm_eps

    def _apply_gated_norm(self, x, gate):
        values = x.float().reshape(-1, x.shape[-1])
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return (
            values
            * self.norm.weight.float()
            * gate.float().reshape_as(values).sigmoid()
        ).to(x.dtype)


class Qwen38Layer(nn.Module):
    def __init__(
        self,
        config,
        ps,
        layer_idx,
        *,
        ngram_primes=None,
        fuse_wgrad_accumulation=False,
        ple_owner_sharding=False
    ):
        super().__init__()
        c = config
        self.linear_attn = (
            Qwen38GatedDeltaNet(c, ps)
            if c.layer_types[layer_idx] == 'linear_attention'
            else None
        )
        self.self_attn = (
            Qwen3_8_FlashNextQSAAttention(c) if self.linear_attn is None else None
        )
        self.mlp = MoELayer(
            c,
            ps,
            use_deepep=False,
            router_bias_rate=0.0,
            fp8=False,
            moe_act_recompute=False,
            router_dtype=torch.float32,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        )
        self.attn_hyper_connection = HyperConnection(
            c.hidden_size, c.hc_count, c.hc_lowrank, c.rms_norm_eps
        )
        self.mlp_hyper_connection = HyperConnection(
            c.hidden_size, c.hc_count, c.hc_lowrank, c.rms_norm_eps
        )
        self.ple = None
        if layer_idx + 1 in c.ple_layer_ids:
            if c.ple_embed_dim % 16:
                raise ValueError('PLE_HEAD_WIDTH')
            primes = PRIMES if ngram_primes is None else tuple(ngram_primes)
            rows = ((sum(primes) + 127) // 128) * 128
            table = Qwen3_8_FlashNextEngramTableConfig(
                rows, c.ple_embed_dim // 16
            ).build(
                process_group=ps.ep_group if ple_owner_sharding else None,
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
            )
            embedding = Qwen3_8_FlashNextNGramEmbedding(
                table, eos_token_id=c.eos_token_id, ngram_heads_vocab_sizes=primes
            )
            self.ple = Qwen3_8_FlashNextPLELayer(
                embedding,
                hidden_size=c.hidden_size,
                hc_count=c.hc_count,
                ple_embed_dim=c.ple_embed_dim,
                rms_norm_eps=c.rms_norm_eps,
                conv_kernel_size=c.ple_conv_kernel_size,
            )

    def forward(self, hidden, input_ids, angles, cu_seqlens=None, *, cp_context=None):
        if self.ple is not None:
            hidden = hidden + self.ple(
                hidden,
                input_ids,
                cu_seqlens=cu_seqlens if cp_context is None else None,
                cp_context=cp_context,
            )
        branch, residual = self.attn_hyper_connection.mix(hidden)
        if self.linear_attn is not None:
            packed = (
                None
                if cu_seqlens is None
                else PackedSeqParams.from_cu_seqlens(
                    cu_seqlens, int(cu_seqlens.diff().max())
                )
            )
            branch = self.linear_attn(
                branch.transpose(0, 1), packed_seq_params=packed
            ).transpose(0, 1)
        else:
            branch = self.self_attn(
                branch,
                angles,
                cu_seqlens=cu_seqlens if cp_context is None else None,
                cp_context=cp_context,
            )
        hidden = self.attn_hyper_connection.combine(branch, residual)
        branch, residual = self.mlp_hyper_connection.mix(hidden)
        kwargs = {}
        if cp_context is not None:
            context = cp_context
            kwargs = dict(
                token_mask=(
                    ~context.global_padding_mask[
                        :, context.local_sequence_start : context.local_sequence_end
                    ]
                ).reshape(-1),
                token_group=context.group,
                # Native DDP averages both dense and expert gradients over DP×CP.
                aux_loss_scale=context.size,
            )
        return self.mlp_hyper_connection.combine(self.mlp(branch, **kwargs), residual)


class Qwen38Model(nn.Module):
    def __init__(
        self,
        config,
        ps,
        *,
        ngram_primes=None,
        fuse_wgrad_accumulation=False,
        ple_owner_sharding=False
    ):
        super().__init__()
        if any(getattr(ps, k) != 1 for k in ('etp_size', 'pp_size')):
            raise NotImplementedError('QWEN38_MODEL_PARALLEL_NOT_VALIDATED')
        if ps.cp_size > 1:
            if ps.ep_size > 1:
                from .cp_ep import validate_cp_ep_contract

                validate_cp_ep_contract(ps, ple_owner_sharding=ple_owner_sharding)
            elif ps.tp_size != 1 or ps.dp_size != 1:
                raise NotImplementedError('QWEN38_CP_COMBINATION_NOT_VALIDATED')
            if ps.cp_group is None:
                raise ValueError('QWEN38_CP_GROUP_REQUIRED')
        if ps.tp_size > 1 and ps.ep_size > 1:
            raise NotImplementedError('QWEN38_TP_EP_COMBINATION_NOT_VALIDATED')
        if config.tie_word_embeddings or not config.norm_topk_prob:
            raise ValueError('QWEN38_RELEASE_TIED_OR_ROUTER_CONTRACT')
        if ple_owner_sharding and (
            ps.ep_size < 2 or ps.ep_group is None or ps.tp_size != 1
        ):
            raise ValueError('PLE_OWNER_REQUIRES_EP_WITH_TP_CP_ONE')
        self.config, self.ps = config, ps
        # Replicated consumers see complete projection outputs. Preserve EP/EDP
        # groups; dense projections below own the actual TP group.
        consumer_ps = copy(ps)
        consumer_ps.tp_size, consumer_ps.tp_rank, consumer_ps.tp_group = 1, 0, None
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen38Layer(
                    config,
                    consumer_ps,
                    i,
                    ngram_primes=ngram_primes,
                    ple_owner_sharding=ple_owner_sharding,
                    fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.hyper_connection_mixer = HyperConnection(
            config.hidden_size,
            config.hc_count,
            config.hc_lowrank,
            config.rms_norm_eps,
            write=False,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if ps.tp_size > 1:
            from .tp import parallelize_projections

            parallelize_projections(self, ps)

    def forward(
        self,
        input_ids,
        *,
        labels=None,
        cu_seqlens=None,
        position_ids=None,
        cp_context=None,
        loss_token_count=None
    ):
        c = self.config
        if self.ps.cp_size > 1:
            if (
                cp_context is None
                or cp_context.size != self.ps.cp_size
                or cp_context.rank != self.ps.cp_rank
                or input_ids.shape[1] != cp_context.local_sequence_length
                or cu_seqlens is None
                or position_ids is None
            ):
                raise ValueError('QWEN38_CP_FORWARD_METADATA')
            if labels is not None and loss_token_count is None:
                raise ValueError('QWEN38_CP_LOSS_POPULATION')
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).expand_as(input_ids)
            if cu_seqlens is not None:
                position_ids = (
                    position_ids
                    - cu_seqlens[
                        torch.bucketize(position_ids, cu_seqlens[1:], right=True)
                    ]
                )
        rotary_dim = int(c.head_dim * c.partial_rotary_factor)
        inv_freq = c.rope_theta ** (
            -torch.arange(0, rotary_dim, 2, device=input_ids.device).float()
            / rotary_dim
        )
        half_angles = position_ids.float().unsqueeze(-1) * inv_freq
        angles = torch.cat((half_angles, half_angles), -1).unsqueeze(-2)
        hidden = self.embed_tokens(input_ids).repeat(1, 1, c.hc_count)
        for layer in self.layers:
            if cp_context is None:
                hidden = layer(hidden, input_ids, angles, cu_seqlens)
            else:
                hidden = layer(
                    hidden, input_ids, angles, cu_seqlens, cp_context=cp_context
                )
        hidden, _ = self.hyper_connection_mixer.mix(hidden)
        logits = self.lm_head(hidden)
        output = {'logits': logits, 'hidden_states': hidden}
        if labels is not None:
            losses = F.cross_entropy(
                logits.float().flatten(0, 1), labels.flatten(), reduction='none'
            ).reshape_as(labels)
            valid = labels != -100
            denominator = valid.sum() if loss_token_count is None else loss_token_count
            loss = losses.sum() / denominator.clamp_min(1)
            if cp_context is not None:
                loss = loss * cp_context.size
            output.update(loss=loss, log_probs=-losses)
        return output
