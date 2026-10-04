# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""V4.1 shifted hyper-connections in batch/sequence/HC/hidden layout.

The paired return is part of the layer boundary: callers must checkpoint and
transport both tensors. Sublayers are injected so attention ownership remains
in the model's attention implementation, rather than in generic mHC primitives.
"""

from megatron.lite.primitive.modules.attention import mhc as primitives
from torch import nn

HCMixes = primitives.HCMixes
RMSNorm = primitives.RMSNorm
contract_hc = primitives.contract_hc
expand_hc = primitives.expand_hc
mix_residual = primitives.mix_residual


class DeepseekV41Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        copies,
        attention,
        ffn,
        *,
        norm_eps=1e-6,
        hc_eps=1e-6,
        iterations=20
    ):
        super().__init__()
        self.attn = attention
        self.ffn = ffn
        self.attn_norm = RMSNorm(hidden_size, norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, norm_eps)
        self.attn_mixes = HCMixes(hidden_size, copies, norm_eps, hc_eps, iterations)
        self.ffn_mixes = HCMixes(hidden_size, copies, norm_eps, hc_eps, iterations)

    def forward(
        self,
        hidden,
        pre_mix,
        state,
        *,
        attention_kwargs=None,
        ffn_kwargs=None,
        previous_post=None
    ):
        """Return hidden, shifted pre-mix and caller-owned CSA2 state.

        State is an explicit graph input/output, never a module cache. Callers
        must preserve it alongside both HC tensors across layer boundaries.
        """
        if self.attn_mixes.deployment_math and hidden.is_cuda:
            from megatron.lite.primitive.modules import deployment_math

            hidden, attn_pre, attn_post, attn_comb, x = deployment_math.mhc_joint(
                hidden, pre_mix, self.attn_norm.weight, self.attn_mixes, previous_post
            )
            x, state = self.attn(
                x, state, **({} if attention_kwargs is None else attention_kwargs)
            )
            hidden, ffn_pre, ffn_post, ffn_comb, ffn_input = deployment_math.mhc_joint(
                hidden,
                attn_pre,
                self.ffn_norm.weight,
                self.ffn_mixes,
                (x, hidden, attn_post, attn_comb),
            )
            x = self.ffn(ffn_input, **({} if ffn_kwargs is None else ffn_kwargs))
            pending = (x, hidden, ffn_post, ffn_comb)
            return deployment_math.mhc_post(*pending), ffn_pre, state, pending
        attn_pre, attn_post, attn_comb = self.attn_mixes(hidden)
        x = self.attn_norm(contract_hc(hidden, pre_mix))
        kwargs = {} if attention_kwargs is None else attention_kwargs
        x, state = self.attn(x, state, **kwargs)
        hidden = mix_residual(x, hidden, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = self.ffn_mixes(hidden)
        x = self.ffn_norm(contract_hc(hidden, attn_pre))
        x = self.ffn(x, **({} if ffn_kwargs is None else ffn_kwargs))
        return mix_residual(x, hidden, ffn_post, ffn_comb), ffn_pre, state
