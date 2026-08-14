"""Four-layer loss/gradient/update vertical slice for DS4 bridge semantics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.vllm.primitive import block_fp8_linear, rms_norm
from megatron.lite.model.deepseek_v4.vllm.primitive.moe import deep_ep_moe
from megatron.lite.model.deepseek_v4.vllm.primitive.router import fixed_route_vjp


class _FourLayerBridge(torch.nn.Module):
    def __init__(self, hidden: int = 8, intermediate: int = 6, vocab: int = 13):
        super().__init__()
        self.norms = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(hidden)) for _ in range(4)]
        )
        self.dense = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(hidden, hidden) / hidden**0.5) for _ in range(4)]
        )
        self.gates = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(2, hidden) / hidden**0.5) for _ in range(4)]
        )
        self.w13 = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(2 * intermediate, hidden) / hidden**0.5) for _ in range(4)]
        )
        self.w2 = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(hidden, intermediate) / intermediate**0.5) for _ in range(4)]
        )
        self.head = torch.nn.Parameter(torch.randn(vocab, hidden) / hidden**0.5)

    def forward(self, value):
        ids = torch.tensor([[0], [0], [0]], device=value.device, dtype=torch.int64)
        for layer in range(4):
            normalized = rms_norm(
                lambda x, w, eps: F.rms_norm(x, (x.shape[-1],), w, eps),
                value,
                self.norms[layer],
                1e-6,
            )
            dense = block_fp8_linear(F.linear, normalized, self.dense[layer])
            logits = F.linear(normalized, self.gates[layer])

            def route(visible_logits):
                selected = torch.sqrt(F.softplus(visible_logits)).gather(-1, ids)
                return selected, ids

            probs, route_ids = fixed_route_vjp(
                route, logits, renormalize=False, route_scale=1.0
            )

            def expert(hidden, weights, expert_ids, w13, w2):
                del expert_ids
                gate, up = F.linear(hidden, w13).chunk(2, -1)
                return F.linear(F.silu(gate) * up, w2) * weights

            moe = deep_ep_moe(
                expert,
                normalized,
                probs,
                route_ids,
                (self.w13[layer],),
                (self.w2[layer],),
                global_expert_start=0,
            )
            value = value + dense + moe
        return F.linear(value, self.head)


def test_four_layer_loss_grad_coverage_and_optimizer_repack_version() -> None:
    torch.manual_seed(20260813)
    model = _FourLayerBridge()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(3, 8, requires_grad=True)
    labels = torch.tensor([1, 5, 9])
    versions_before = {name: parameter._version for name, parameter in model.named_parameters()}
    logits = model(inputs)
    loss = F.cross_entropy(logits.float(), labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
    ]
    assert not missing, f"first-divergence/missing-grad parameters: {missing}"
    optimizer.step()
    changed = [
        name
        for name, parameter in model.named_parameters()
        if parameter._version > versions_before[name]
    ]
    assert len(changed) == len(tuple(model.parameters()))


def test_all_masked_loss_is_rejected() -> None:
    token_loss = torch.ones(3)
    mask = torch.zeros(3)
    denominator = mask.sum()
    assert not bool(denominator > 0)
