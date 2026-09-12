# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""V4.1 shifted hyper-connections in batch/sequence/HC/hidden layout.

The paired return is part of the layer boundary: callers must checkpoint and
transport both tensors. Sublayers are injected so attention ownership remains
in the model's attention implementation, rather than in generic mHC primitives.
"""

from megatron.lite.primitive.modules.hyper_connection import (
    HCMixes,
    RMSNorm,
    contract_hc,
    expand_hc,
    mix_residual,
)
from torch import nn


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

    def forward(self, hidden, pre_mix, *, attention_kwargs=None, ffn_kwargs=None):
        hidden, pre_mix, _ = self._forward(
            hidden, pre_mix, attention_kwargs, ffn_kwargs, None, False
        )
        return hidden, pre_mix

    def forward_with_state(
        self, hidden, pre_mix, state, *, attention_kwargs=None, ffn_kwargs=None
    ):
        """Return hidden, shifted pre-mix and caller-owned CSA2 state.

        State is an explicit graph input/output, never a module cache. Callers
        must preserve it alongside both HC tensors across layer boundaries.
        """
        return self._forward(hidden, pre_mix, attention_kwargs, ffn_kwargs, state, True)

    def _forward(self, hidden, pre_mix, attention_kwargs, ffn_kwargs, state, stateful):
        attn_pre, attn_post, attn_comb = self.attn_mixes(hidden)
        x = self.attn_norm(contract_hc(hidden, pre_mix))
        kwargs = {} if attention_kwargs is None else attention_kwargs
        if stateful:
            x, state = self.attn(x, state, **kwargs)
        else:
            x = self.attn(x, **kwargs)
        hidden = mix_residual(x, hidden, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = self.ffn_mixes(hidden)
        x = self.ffn_norm(contract_hc(hidden, attn_pre))
        x = self.ffn(x, **({} if ffn_kwargs is None else ffn_kwargs))
        return mix_residual(x, hidden, ffn_post, ffn_comb), ffn_pre, state
