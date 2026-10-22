# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native MTP keeps policy-head and backbone gradient decisions independent."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, process_mtp_loss


class OutputLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(20, dtype=torch.float32).reshape(5, 4) / 20)

    def forward(self, hidden, weight=None, **kwargs):
        return F.linear(hidden, self.weight if weight is None else weight), None


def loss(labels, logits):
    return F.cross_entropy(
        logits.transpose(0, 1).reshape(-1, 5), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("detach_output", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_output_weight_detach_and_microbatch_normalization(tied, detach_output, per_token):
    config = SimpleNamespace(
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        calculate_per_token_loss=per_token,
        mtp_detach_heads=False,
        mtp_detach_output_weight=detach_output,
        attention_cp_layout="zigzag",
    )
    layer = OutputLayer()
    shared_weight = layer.weight if tied else None
    inputs = torch.arange(24, dtype=torch.float32).reshape(6, 1, 4).div(24).requires_grad_()
    labels = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    output = process_mtp_loss(
        inputs,
        labels,
        mask,
        layer,
        shared_weight,
        False,
        loss,
        config,
        loss_normalization_factor=2.0,
    )
    (output.sum() * 0).backward()
    actual_hidden_grad = inputs.grad.clone()
    assert torch.count_nonzero(actual_hidden_grad[3:]) > 0
    assert torch.count_nonzero(actual_hidden_grad[:3]) == 0
    assert (layer.weight.grad is None) == detach_output
    # Independent reference: labels/mask shift by one additional token for MTP.
    reference = inputs.detach()[3:].clone().requires_grad_()
    shifted_labels = torch.tensor([[2, 3, 0]])
    shifted_mask = torch.tensor([[1.0, 1.0, 0.0]])
    reference_loss = (
        loss(shifted_labels, F.linear(reference, layer.weight.detach())) * shifted_mask
    ).sum()
    coefficient = (
        0.1 * 2 * (mask.sum() / shifted_mask.sum() if per_token else 1 / shifted_mask.sum())
    )
    (reference_loss * coefficient).backward()
    torch.testing.assert_close(actual_hidden_grad[3:], reference.grad)


def test_mtp_embedding_roll_uses_runtime_cp_group(monkeypatch):
    from megatron.core.transformer import multi_token_prediction as mtp

    runtime_group = SimpleNamespace(size=lambda: 1)
    seen = []

    def roll(value, *, cp_group=None, **kwargs):
        seen.append(cp_group)
        return value, None

    monkeypatch.setattr(mtp, "roll_tensor", roll)
    layer = SimpleNamespace(
        cp_group=object(),
        config=SimpleNamespace(
            sequence_parallel=False, attention_cp_layout="zigzag", mtp_detach_heads=True
        ),
    )
    result = mtp.MultiTokenPredictionLayer._get_embeddings(
        layer,
        torch.ones(1, 3, dtype=torch.long),
        torch.arange(3).unsqueeze(0),
        lambda **kwargs: torch.ones(3, 1, 4, requires_grad=True),
        torch.ones(3, 1, 4, requires_grad=True),
        packed_seq_params=SimpleNamespace(cp_group=runtime_group, local_cp_size=1),
    )
    assert seen == [runtime_group, runtime_group]
    assert not result[-2].requires_grad
