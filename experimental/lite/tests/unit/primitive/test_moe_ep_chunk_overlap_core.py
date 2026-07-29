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
        def __init__(self, linear):
            self.linear = linear
            self.pending = 1
            self.context = SimpleNamespace(empty=lambda: self.pending == 0)

        @staticmethod
        def delay_wgrad_compute():
            return True

        def pop(self):
            self.pending -= 1
            grads = [
                getattr(self.linear, f"weight{idx}").main_grad
                for idx in range(self.linear.num_gemms)
            ]
            return (None, [torch.empty(0), torch.empty(0)], None), [None, None, grads]

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self, num_gemms, *_args, bias, **_kwargs):
            super().__init__()
            self.num_gemms = num_gemms
            self.use_bias = bias
            for idx in range(num_gemms):
                param = torch.nn.Parameter(torch.zeros(2, 2))
                param.main_grad = torch.ones(2, 2)
                self.register_parameter(f"weight{idx}", param)
            self.wgrad_store = FakeStore(self)

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

    experts.flush_delayed_weight_grads(num_contexts=1)

    assert all(parameter.grad is None for parameter in experts.parameters())


def test_delayed_expert_wgrads_reuse_distopt_main_grad(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    grouped_linear_kwargs = []

    class FakeStore:
        context = SimpleNamespace(empty=lambda: False)

        @staticmethod
        def delay_wgrad_compute():
            return True

        @staticmethod
        def pop():
            return (None, [None, None], None), [None, None, []]

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self, num_gemms, *_args, bias, **kwargs):
            super().__init__()
            grouped_linear_kwargs.append(kwargs)
            self.use_bias = bias
            self.wgrad_store = FakeStore()
            for idx in range(num_gemms):
                param = torch.nn.Parameter(torch.zeros(2, 2))
                param.main_grad = torch.zeros_like(param)
                self.register_parameter(f"weight{idx}", param)

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
    experts_module.Experts(config, ps, delay_wgrad_compute=True)

    assert all(
        kwargs["fuse_wgrad_accumulation"] for kwargs in grouped_linear_kwargs
    )


def test_dispatch_local_backward_accumulates_duplicate_token_rows_and_weights(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )

    recv_hidden = torch.full((3, 2), 9.0)
    recv_probs = torch.full((3, 2), 9.0)
    chunk = SimpleNamespace(
        row_id_map=torch.tensor([0, 0, 2]),
        prob_flat_indices=torch.tensor([1, 3, 5]),
        recv_hidden_scratch=recv_hidden,
        recv_probs_scratch=recv_probs,
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
    assert grad_hidden.data_ptr() == recv_hidden.data_ptr()
    assert grad_recv_probs.data_ptr() == recv_probs.data_ptr()


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
        recv_hidden_scratch=torch.empty(0, 4, dtype=torch.bfloat16),
        recv_probs_scratch=torch.empty(0, 2, dtype=torch.float32),
    )

    grad_hidden, grad_probs = _dispatch_local_backward(
        chunk, torch.empty(0, 4, dtype=torch.bfloat16), None
    )

    assert grad_hidden.shape == (0, 4)
    assert grad_probs.shape == (0, 2)


def test_dispatch_scratch_is_released_with_comm_stream_ownership(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _release_dispatch_scratch,
    )

    stream = object()

    class Scratch:
        def __init__(self):
            self.stream = None

        def record_stream(self, value):
            self.stream = value

    hidden = Scratch()
    probs = Scratch()
    chunk = SimpleNamespace(
        recv_hidden_scratch=hidden,
        recv_probs_scratch=probs,
    )

    _release_dispatch_scratch(chunk, stream)

    assert hidden.stream is stream
    assert probs.stream is stream
    assert chunk.recv_hidden_scratch is None
    assert chunk.recv_probs_scratch is None


def test_accumulate_reuses_first_chunk_gradient_storage(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import _accumulate

    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    first = torch.ones(2, 2)
    second = torch.full((2, 2), 2.0)
    accum = [None]

    _accumulate(accum, (parameter,), (first,))
    storage = accum[0].data_ptr()
    _accumulate(accum, (parameter,), (second,))

    assert accum[0].data_ptr() == storage
    torch.testing.assert_close(accum[0], torch.full((2, 2), 3.0))


def test_delayed_wgrad_excludes_expert_params_from_autograd_targets(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _expert_grad_inputs,
    )

    dispatched = torch.ones(2, 2, requires_grad=True)
    probs = torch.ones(2, requires_grad=True)
    params = tuple(torch.nn.Parameter(torch.zeros(1)) for _ in range(2))
    inputs = _expert_grad_inputs(dispatched, probs)

    assert inputs == (dispatched, probs)
    assert all(all(param is not item for item in inputs) for param in params)


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
