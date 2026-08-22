from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from megatron.lite.model.deepseek_v4.vllm.primitive.moe.module import DeepseekV4MoE
from megatron.lite.primitive.modules.experts import swiglu_with_probs
from megatron.lite.primitive.modules.mlp import SwiGLUMLP


class _VisibleLinear(nn.Module):
    def forward(self, value, weight):
        return F.linear(value, weight)


def test_shared_experts_block_fp8_bridges_cover_bf16_master_gradients() -> None:
    torch.manual_seed(31)
    hidden_size = 8
    shared_intermediate = 8
    moe = DeepseekV4MoE.__new__(DeepseekV4MoE)
    nn.Module.__init__(moe)
    moe.shared_experts = SwiGLUMLP(hidden_size, shared_intermediate).to(
        dtype=torch.bfloat16
    )
    moe.config = SimpleNamespace(swiglu_limit=10.0)
    moe.shared_gate_up_fp8 = _VisibleLinear()
    moe.shared_down_fp8 = _VisibleLinear()
    hidden = (torch.randn(5, hidden_size, dtype=torch.bfloat16) * 8).requires_grad_(
        True
    )
    grad_output = torch.randn_like(hidden)

    output = moe._shared_expert_forward(hidden)
    gate_up_visible = F.linear(hidden, moe.shared_experts.gate_up.weight)
    expected_visible = F.linear(
        swiglu_with_probs(gate_up_visible, None, moe.config.swiglu_limit),
        moe.shared_experts.down.weight,
    )
    torch.testing.assert_close(output, expected_visible, rtol=0, atol=0)
    output.backward(grad_output)
    actual_grads = (
        hidden.grad,
        moe.shared_experts.gate_up.weight.grad,
        moe.shared_experts.down.weight.grad,
    )

    ref_hidden = hidden.detach().float().requires_grad_(True)
    ref_gate_up = (
        moe.shared_experts.gate_up.weight.detach().float().requires_grad_(True)
    )
    ref_down = moe.shared_experts.down.weight.detach().float().requires_grad_(True)
    gate_up = F.linear(ref_hidden, ref_gate_up)
    activated = swiglu_with_probs(gate_up, None, moe.config.swiglu_limit)
    reference = F.linear(activated, ref_down)
    expected_grads = torch.autograd.grad(
        reference,
        (ref_hidden, ref_gate_up, ref_down),
        grad_output.float(),
    )
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected, rtol=5e-2, atol=5e-2
        )
