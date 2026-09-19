# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The V4.1 hyper-connection boundary is shifted by one sublayer.

Attention consumes the pre-mix it was handed by the previous block; the FFN
consumes the attention mix computed inside this block; the block hands the FFN
mix to the next one. Reusing a block's own coefficients for its own attention is
the documented wrong wiring -- it type-checks, runs, and trains to a different
model, so these tests assert on what each sublayer actually receives.
"""

import pytest
import torch
from torch import nn

HIDDEN, COPIES, TOKENS = 8, 2, 3


class _Spy(nn.Module):
    """Records what it was handed and returns something order-dependent."""

    def __init__(self, gain):
        super().__init__()
        self.gain = gain
        self.seen = []

    def forward(self, x, *args, **kwargs):
        self.seen.append(x.detach().clone())
        # The stateful entry hands state in and expects it threaded back out.
        return (x * self.gain, args[0]) if args else x * self.gain


@pytest.fixture
def block(v41_core_te):
    from megatron.lite.model.deepseek_v41.lite.block import DeepseekV41Block

    torch.manual_seed(7)
    module = DeepseekV41Block(HIDDEN, COPIES, _Spy(2.0), _Spy(3.0))
    with torch.no_grad():
        for mixes in (module.attn_mixes, module.ffn_mixes):
            mixes.fn.normal_(0.0, 0.1)
            mixes.base.normal_(0.0, 0.1)
    return module


def _mhc():
    from megatron.lite.primitive.modules.attention.mhc import contract_hc, mix_residual

    return contract_hc, mix_residual


def _inputs():
    generator = torch.Generator().manual_seed(21)
    hidden = torch.randn(1, TOKENS, COPIES, HIDDEN, generator=generator)
    pre_mix = torch.randn(1, TOKENS, COPIES, generator=generator)
    return hidden, pre_mix


def test_attention_consumes_the_incoming_pre_mix_not_its_own_coefficients(block):
    hidden, pre_mix = _inputs()
    attn_pre = block.attn_mixes(hidden)[0]
    block(hidden, pre_mix)

    handed = block.attn.seen[0]
    assert torch.allclose(handed, block.attn_norm(_mhc()[0](hidden, pre_mix)))
    # The documented wrong wiring: attention folding in the coefficients this
    # block just computed for itself.
    assert not torch.allclose(handed, block.attn_norm(_mhc()[0](hidden, attn_pre)))


def test_ffn_consumes_this_blocks_attention_mix(block):
    hidden, pre_mix = _inputs()
    attn_pre, attn_post, attn_comb = block.attn_mixes(hidden)
    block(hidden, pre_mix)

    after_attention = _mhc()[1](
        block.attn.seen[0] * block.attn.gain, hidden, attn_post, attn_comb
    )
    handed = block.ffn.seen[0]
    assert torch.allclose(handed, block.ffn_norm(_mhc()[0](after_attention, attn_pre)))
    # Not the incoming pre-mix, and not its own ffn coefficients.
    assert not torch.allclose(
        handed, block.ffn_norm(_mhc()[0](after_attention, pre_mix))
    )
    ffn_pre = block.ffn_mixes(after_attention)[0]
    assert not torch.allclose(
        handed, block.ffn_norm(_mhc()[0](after_attention, ffn_pre))
    )


def test_block_hands_the_ffn_mix_to_the_next_block(block):
    hidden, pre_mix = _inputs()
    attn_pre, attn_post, attn_comb = block.attn_mixes(hidden)
    _, emitted = block(hidden, pre_mix)

    after_attention = _mhc()[1](
        block.attn.seen[0] * block.attn.gain, hidden, attn_post, attn_comb
    )
    assert torch.allclose(emitted, block.ffn_mixes(after_attention)[0])
    # Emitting the attention mix instead would shift the boundary back by one.
    assert not torch.allclose(emitted, attn_pre)


def test_stateful_entry_returns_state_as_an_explicit_third_value(block):
    hidden, pre_mix = _inputs()
    sentinel = object()
    out_hidden, out_pre, state = block.forward_with_state(hidden, pre_mix, sentinel)
    assert out_hidden.shape == hidden.shape
    assert out_pre.shape == pre_mix.shape
    # State is threaded through, never cached on the module.
    assert state is sentinel
    assert not hasattr(block, "_state")
