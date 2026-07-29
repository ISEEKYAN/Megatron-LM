# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch


def test_experts_accept_te_dbias_placeholders_when_bias_is_disabled(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    class FakeStore:
        context = SimpleNamespace(empty=lambda: False)

        @staticmethod
        def delay_wgrad_compute():
            return True

        @staticmethod
        def pop():
            grads = [torch.ones(2, 2), torch.ones(2, 2)]
            return (None, [torch.empty(0), torch.empty(0)], None), [None, None, grads]

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self, num_gemms, *_args, bias, **_kwargs):
            super().__init__()
            self.use_bias = bias
            self.wgrad_store = FakeStore()
            for idx in range(num_gemms):
                self.register_parameter(
                    f"weight{idx}", torch.nn.Parameter(torch.zeros(2, 2))
                )

    monkeypatch.setattr(
        experts_module.te, "GroupedLinear", FakeGroupedLinear, raising=False
    )
    config = SimpleNamespace(
        num_experts=2,
        hidden_size=2,
        moe_intermediate_size=2,
        swiglu_limit=0.0,
    )
    ps = SimpleNamespace(ep_size=1, etp_size=1, tp_size=1, etp_group=None)
    experts = experts_module.Experts(config, ps, delay_wgrad_compute=True)

    grads = experts.pop_delayed_weight_grads()

    assert len(grads) == 4
    assert all(parameter.grad is None for parameter in experts.parameters())


def test_dispatch_local_backward_accumulates_duplicate_token_rows_and_weights(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )

    chunk = SimpleNamespace(
        row_id_map=torch.tensor([0, 0, 2]),
        prob_flat_indices=torch.tensor([1, 3, 5]),
        recv_hidden_shape=torch.Size([3, 2]),
        recv_hidden_dtype=torch.float32,
        recv_probs_shape=torch.Size([3, 2]),
        recv_probs_dtype=torch.float32,
    )
    grad_dispatched = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    grad_probs = torch.tensor([0.25, 0.5, 0.75])

    grad_hidden, grad_recv_probs = _dispatch_local_backward(
        chunk, grad_dispatched, grad_probs
    )

    torch.testing.assert_close(
        grad_hidden, torch.tensor([[4.0, 6.0], [0.0, 0.0], [5.0, 6.0]])
    )
    torch.testing.assert_close(
        grad_recv_probs, torch.tensor([[0.0, 0.25], [0.0, 0.5], [0.0, 0.75]])
    )


def test_dispatch_local_backward_materializes_zero_probability_gradient(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )

    chunk = SimpleNamespace(
        row_id_map=torch.empty(0, dtype=torch.long),
        prob_flat_indices=torch.empty(0, dtype=torch.long),
        recv_hidden_shape=torch.Size([0, 4]),
        recv_hidden_dtype=torch.bfloat16,
        recv_probs_shape=torch.Size([0, 2]),
        recv_probs_dtype=torch.float32,
    )

    grad_hidden, grad_probs = _dispatch_local_backward(
        chunk, torch.empty(0, 4, dtype=torch.bfloat16), None
    )

    assert grad_hidden.shape == (0, 4)
    assert grad_probs.shape == (0, 2)


def test_forward_trace_pipelines_next_dispatch_before_current_expert(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkOverlapOperator,
    )

    trace = []

    class FakeEvent:
        def record(self, stream):
            trace.append(f"event.record.{stream.name}")

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, _event):
            trace.append(f"{self.name}.wait")

    class FakeDispatcher:
        def __init__(self, chunk):
            self.chunk = chunk
            self._local_tpe_list = [2]

        def submit_deepep_dispatch(self, hidden, _scores, _indices):
            trace.append(f"dispatch.submit.{self.chunk}")
            return {
                "hidden": hidden,
                "recv_hidden": None,
                "recv_indices": None,
                "recv_probs": None,
                "recv_per_expert": None,
            }

        def finish_deepep_dispatch(self, state):
            trace.append(f"dispatch.finish.{self.chunk}")
            return state["hidden"], torch.tensor([2]), torch.ones(2)

        def prepare_deepep_combine(self, expert_output):
            trace.append(f"combine.prepare.{self.chunk}")
            return expert_output, f"handle-{self.chunk}"

        def submit_deepep_combine_prepared(self, output, _handle):
            trace.append(f"combine.submit.{self.chunk}")
            return {"output": output}

        def finish_deepep_combine(self, state):
            trace.append(f"combine.finish.{self.chunk}")
            return state["output"]

    class FakeExperts:
        def __call__(self, hidden, *_args, **_kwargs):
            chunk = int(hidden[0, 0].item() // 2)
            trace.append(f"expert.{chunk}")
            return hidden + 1

    compute = FakeStream("compute")
    comm = FakeStream("comm")
    caller = FakeStream("caller")
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: caller)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())

    operator = object.__new__(EPChunkOverlapOperator)
    operator.router = lambda hidden: (
        torch.ones(hidden.size(0), 1),
        torch.zeros(hidden.size(0), 1, dtype=torch.long),
    )
    operator.experts = FakeExperts()
    operator._streams = lambda _device: (compute, comm)
    dispatchers = [FakeDispatcher(idx) for idx in range(3)]
    operator._forward_dispatcher = lambda idx: dispatchers[idx]

    inputs = torch.arange(6, dtype=torch.float32).view(6, 1)
    output = operator._forward_output_async(
        inputs,
        [(0, 2), (2, 4), (4, 6)],
        inputs.shape,
        inputs.dtype,
        disable_expert_act_recompute=False,
    )

    torch.testing.assert_close(output, inputs + 1)
    operations = [item for item in trace if not item.startswith(("event.", "caller."))]
    assert operations == [
        "comm.wait",
        "dispatch.submit.0",
        "comm.wait",
        "dispatch.submit.1",
        "dispatch.finish.0",
        "expert.0",
        "combine.prepare.0",
        "comm.wait",
        "combine.submit.0",
        "comm.wait",
        "dispatch.submit.2",
        "dispatch.finish.1",
        "expert.1",
        "combine.prepare.1",
        "comm.wait",
        "combine.submit.1",
        "dispatch.finish.2",
        "expert.2",
        "combine.prepare.2",
        "comm.wait",
        "combine.submit.2",
        "combine.finish.0",
        "combine.finish.1",
        "combine.finish.2",
    ]


def test_core_contains_token_wise_forward_and_backward_deepep_pipeline():
    source_path = (
        Path(__file__).parents[3]
        / "megatron"
        / "lite"
        / "primitive"
        / "modules"
        / "moe_ep_chunk_overlap.py"
    )
    tree = ast.parse(source_path.read_text())
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert {
        "submit_deepep_dispatch",
        "finish_deepep_dispatch",
        "submit_deepep_combine_prepared",
        "finish_deepep_combine",
        "submit_deepep_combine_backward",
        "finish_deepep_combine_backward",
        "submit_deepep_dispatch_backward",
        "finish_deepep_dispatch_backward",
    } <= calls

def test_expert_chunk_merge_round_trip_groups_each_expert_once(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _merge_expert_chunks,
        _split_expert_chunks,
    )

    counts = [[2, 1, 0], [1, 2, 1]]
    chunks = [
        torch.tensor([[0], [1], [2]]),
        torch.tensor([[10], [11], [12], [13]]),
    ]

    merged = _merge_expert_chunks(chunks, counts)

    assert merged.flatten().tolist() == [0, 1, 10, 2, 11, 12, 13]
    restored = _split_expert_chunks(merged, counts)
    torch.testing.assert_close(restored[0], chunks[0])
    torch.testing.assert_close(restored[1], chunks[1])
