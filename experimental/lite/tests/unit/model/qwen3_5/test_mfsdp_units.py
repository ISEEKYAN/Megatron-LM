# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext

import torch

from megatron.lite.model.qwen3_5.lite.model import (
    Qwen35Layer,
    SharedExpert,
    _masked_token_loss,
)
from megatron.lite.model.qwen3_5.lite.protocol import _mfsdp_unit_modules
from megatron.lite.primitive.parallel import VocabParallelEmbedding, VocabParallelOutput


def test_qwen35_mfsdp_shards_all_compute_weight_boundaries():
    assert _mfsdp_unit_modules() == (
        Qwen35Layer,
        VocabParallelEmbedding,
        VocabParallelOutput,
    )


def test_qwen35_masked_token_loss_exposes_numerator_count_and_mean():
    token_loss = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    loss_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    loss, loss_sum, num_tokens = _masked_token_loss(token_loss, loss_mask)

    torch.testing.assert_close(loss_sum, torch.tensor(13.0))
    torch.testing.assert_close(num_tokens, torch.tensor(3, dtype=torch.int64))
    torch.testing.assert_close(loss, torch.tensor(13.0 / 3.0))


def test_qwen35_shared_expert_backward_waits_for_routed_stream(monkeypatch):
    current_stream = object()

    class _SharedStream:
        waited_for = None

        def wait_stream(self, stream):
            self.waited_for = stream

    shared_stream = _SharedStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current_stream)
    value = torch.tensor(2.0, requires_grad=True)

    SharedExpert._BackwardStreamWait.apply(value, shared_stream).backward()

    assert shared_stream.waited_for is current_stream


def test_qwen35_shared_expert_stages_fc1_and_fc2_around_routed_path(monkeypatch):
    class _Stream:
        def __init__(self):
            self.waited_for = []

        def wait_stream(self, stream):
            self.waited_for.append(stream)

    routed_stream = _Stream()
    shared_stream = _Stream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: routed_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        SharedExpert, "_get_stream", staticmethod(lambda: shared_stream)
    )

    shared = SharedExpert.__new__(SharedExpert)
    torch.nn.Module.__init__(shared)
    shared.shared_gate = torch.nn.Linear(2, 1, bias=False)
    shared.gate_up = torch.nn.Linear(2, 4, bias=False)
    shared.down = torch.nn.Linear(2, 2, bias=False)
    shared.tp_group = None
    shared.use_mcore_overlap_graph = True
    shared._cached_fc1_input = None
    shared._cached_fc2_input = None
    shared._cached_fc2_output = None
    shared._cached_output = None
    shared._cached_gate = None

    value = torch.tensor([[1.0, -2.0]], requires_grad=True)
    routed_dispatch = value * 2.0
    routed_expert_output = value * 3.0
    shared.pre_forward(value)
    shared.wait_current_stream()
    shared.linear_fc1_forward_and_act(routed_dispatch)
    shared.wait_current_stream()
    shared.linear_fc2_forward(routed_expert_output)
    shared.post_forward_comm()
    output = shared.get_output()
    output.sum().backward()

    assert routed_stream in shared_stream.waited_for
    assert shared_stream in routed_stream.waited_for
    assert shared._cached_fc1_input is None
    assert shared._cached_fc2_input is None
    assert shared._cached_fc2_output is None
    assert shared._cached_output is None
    assert shared._cached_gate is None
