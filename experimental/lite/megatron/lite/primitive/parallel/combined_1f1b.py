# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Combined-1F1B driver for batch-level expert-parallel overlap.

The driver owns only the adjacent-microbatch phase order. Models expose a
fine-grained plan whose implementation keeps model composition out of this
parallel primitive.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import torch


class Combined1F1BPlan(Protocol):
    """Fine-grained model plan consumed by the combined-1F1B driver."""

    def forward(self) -> dict[str, Any]: ...

    def combined_forward_backward(
        self, backward_plan: Combined1F1BPlan
    ) -> dict[str, Any]: ...

    def backward(self) -> None: ...


_COMM_STREAM = None


def _get_comm_stream():
    global _COMM_STREAM
    if _COMM_STREAM is None:
        _COMM_STREAM = torch.cuda.Stream(device=torch.cuda.current_device())
    return _COMM_STREAM


class _ScheduleContext:
    """Per-microbatch dependency event with a process-wide ordered comm stream."""

    def __init__(self, *, use_cuda: bool):
        self.use_cuda = use_cuda
        self.event = torch.cuda.Event() if use_cuda else None
        self.compute_stream = torch.cuda.current_stream() if use_cuda else None
        self.comm_stream = _get_comm_stream() if use_cuda else None

    def record_current(self) -> None:
        if self.use_cuda:
            self.event.record(torch.cuda.current_stream())

    def wait_current(self) -> None:
        if self.use_cuda:
            self.event.wait(torch.cuda.current_stream())

    @contextmanager
    def acquire(self, stream_name: str, nvtx_name: str):
        if not self.use_cuda:
            with nullcontext():
                yield
            return
        stream = self.compute_stream if stream_name == "compute" else self.comm_stream
        self.event.wait(stream)
        with torch.cuda.stream(stream):
            with torch.cuda.nvtx.range(nvtx_name):
                yield
        self.event.record(stream)


def _as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


def _from_tuple(value):
    return value[0] if len(value) == 1 else value


class Combined1F1BNode:
    """Autograd boundary for one fine-grained compute or communication node."""

    def __init__(
        self, forward_fn: Callable, *, context: _ScheduleContext, stream: str, name: str
    ):
        self.forward_fn = forward_fn
        self.context = context
        self.stream = stream
        self.name = name
        self.inputs: tuple[torch.Tensor, ...] | None = None
        self.outputs: tuple[torch.Tensor, ...] | None = None

    def forward(self, inputs):
        raw_inputs = _as_tuple(inputs)
        detached = tuple(
            value.detach().requires_grad_(value.requires_grad) for value in raw_inputs
        )
        with self.context.acquire(
            self.stream, f"combined_1f1b.forward.{self.name}"
        ):
            outputs = _as_tuple(self.forward_fn(*detached))
        if not all(isinstance(value, torch.Tensor) for value in outputs):
            raise TypeError(f"{self.name} must return only tensors")
        self.inputs = detached
        self.outputs = outputs
        return _from_tuple(outputs)

    def backward(self, output_grads):
        if self.inputs is None or self.outputs is None:
            raise RuntimeError(f"{self.name} backward called before forward")
        grads = _as_tuple(output_grads)
        if len(grads) != len(self.outputs):
            raise RuntimeError(
                f"{self.name} got {len(grads)} grads for {len(self.outputs)} outputs"
            )
        tensors = []
        grad_tensors = []
        for output, grad in zip(self.outputs, grads, strict=True):
            if output.requires_grad and grad is not None:
                tensors.append(output)
                grad_tensors.append(grad)
        with self.context.acquire(
            self.stream, f"combined_1f1b.backward.{self.name}"
        ):
            if tensors:
                torch.autograd.backward(tensors, grad_tensors)
        input_grads = tuple(value.grad for value in self.inputs)
        self.inputs = None
        self.outputs = None
        return _from_tuple(input_grads)


class Combined1F1BLayerPlan:
    """Four-node Megatron ordering for one MoE transformer layer."""

    def __init__(self, callables: Sequence[Callable], *, context: _ScheduleContext):
        if len(callables) != 4:
            raise ValueError("combined-1F1B layers require exactly four callables")
        names = ("pre_dispatch", "dispatch", "experts", "combine")
        streams = ("compute", "comm", "compute", "comm")
        self.nodes = tuple(
            Combined1F1BNode(fn, context=context, stream=stream, name=name)
            for fn, stream, name in zip(callables, streams, names, strict=True)
        )

    def forward(self, inputs):
        for node in self.nodes:
            inputs = node.forward(inputs)
        return inputs

    def backward(self, grads):
        for node in reversed(self.nodes):
            grads = node.backward(grads)
        return grads

    @staticmethod
    def combined(
        forward_plan: Combined1F1BLayerPlan,
        backward_plan: Combined1F1BLayerPlan,
        forward_inputs,
        backward_grads,
    ):
        f_pre, f_dispatch, f_experts, f_combine = forward_plan.nodes
        b_pre, b_dispatch, b_experts, b_combine = backward_plan.nodes

        backward_grads = b_combine.backward(backward_grads)
        forward_inputs = f_pre.forward(forward_inputs)
        backward_grads = b_experts.backward(backward_grads)
        forward_inputs = f_dispatch.forward(forward_inputs)
        backward_grads = b_dispatch.backward(backward_grads)
        forward_inputs = f_experts.forward(forward_inputs)
        forward_inputs = f_combine.forward(forward_inputs)
        backward_grads = b_pre.backward(backward_grads)
        return forward_inputs, backward_grads


class Combined1F1BModelPlan:
    """Model-agnostic fine-grained plan used by the adjacent-microbatch driver."""

    def __init__(
        self,
        *,
        preprocess: Callable[[], torch.Tensor],
        layer_callables: Sequence[Sequence[Callable]],
        postprocess: Callable[[torch.Tensor], dict[str, Any]],
        num_microbatches: int,
        external_loss: (
            Callable[[dict[str, Any]], tuple[torch.Tensor, dict]] | None
        ) = None,
        use_cuda: bool,
    ):
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.num_microbatches = num_microbatches
        self.external_loss = external_loss
        self.context = _ScheduleContext(use_cuda=use_cuda)
        self.layers = [
            Combined1F1BLayerPlan(callables, context=self.context)
            for callables in layer_callables
        ]
        self.preprocess_output = None
        self.postprocess_input = None
        self.output: dict[str, Any] | None = None
        self.loss = None

    def _start_forward(self):
        self.preprocess_output = self.preprocess()
        self.context.record_current()
        return self.preprocess_output

    def _finish_forward(self, hidden):
        self.context.wait_current()
        self.postprocess_input = hidden.detach().requires_grad_(hidden.requires_grad)
        self.output = self.postprocess(self.postprocess_input)
        if self.external_loss is None:
            self.loss = self.output.get("loss")
            if self.loss is None:
                raise ValueError("combined-1F1B postprocess output has no loss")
        else:
            self.loss, metrics = self.external_loss(self.output)
            self.output["loss"] = self.loss
            self.output["_loss_fn_metrics"] = metrics
        self.context.record_current()
        return self.output

    def _start_backward(self):
        if self.loss is None or self.postprocess_input is None:
            raise RuntimeError("combined-1F1B backward called before forward")
        self.context.wait_current()
        (self.loss / self.num_microbatches).backward()
        grad = self.postprocess_input.grad
        if grad is None:
            raise RuntimeError(
                "combined-1F1B postprocess produced no hidden-state gradient"
            )
        self.context.record_current()
        return grad

    def _finish_backward(self, grad) -> None:
        if self.preprocess_output is None:
            raise RuntimeError("combined-1F1B backward has no preprocess output")
        self.context.wait_current()
        if self.preprocess_output.requires_grad and grad is not None:
            torch.autograd.backward(self.preprocess_output, grad)
        self.context.record_current()
        if self.output is not None:
            for key, value in self.output.items():
                if isinstance(value, torch.Tensor):
                    self.output[key] = value.detach()
        self.loss = None
        self.postprocess_input = None
        self.preprocess_output = None

    def forward(self) -> dict[str, Any]:
        hidden = self._start_forward()
        for layer in self.layers:
            hidden = layer.forward(hidden)
        return self._finish_forward(hidden)

    def combined_forward_backward(
        self, backward_plan: Combined1F1BModelPlan
    ) -> dict[str, Any]:
        forward_hidden = self._start_forward()
        backward_grad = backward_plan._start_backward()
        overlap = min(len(self.layers), len(backward_plan.layers))
        for index in range(overlap):
            forward_hidden, backward_grad = Combined1F1BLayerPlan.combined(
                self.layers[index],
                backward_plan.layers[-1 - index],
                forward_hidden,
                backward_grad,
            )
        for layer in backward_plan.layers[: len(backward_plan.layers) - overlap][::-1]:
            backward_grad = layer.backward(backward_grad)
        for layer in self.layers[overlap:]:
            forward_hidden = layer.forward(forward_hidden)
        output = self._finish_forward(forward_hidden)
        backward_plan._finish_backward(backward_grad)
        return output

    def backward(self) -> None:
        grad = self._start_backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        self._finish_backward(grad)


@dataclass(frozen=True)
class Combined1F1BConfig:
    """Closed first profile for the 8-GPU all-to-all proxy."""

    num_microbatches: int
    ep_size: int
    pp_size: int
    use_deepep: bool
    recompute: tuple[str, ...]
    moe_permute_fusion: bool

    def validate(self) -> None:
        if self.num_microbatches < 2:
            raise ValueError("combined-1F1B requires at least 2 microbatches")
        if self.ep_size <= 1:
            raise ValueError("combined-1F1B requires EP > 1")
        if self.pp_size != 1:
            raise ValueError("the first combined-1F1B profile requires PP=1")
        if self.use_deepep:
            raise ValueError(
                "the first combined-1F1B profile uses alltoall, not DeepEP"
            )
        if self.recompute:
            raise ValueError("combined-1F1B does not support recompute")
        if self.moe_permute_fusion:
            raise ValueError("the first combined-1F1B profile disables permute fusion")


def build_combined_1f1b_trace(
    num_microbatches: int,
) -> list[tuple[int | None, int | None]]:
    """Return ``(forward_mb, backward_mb)`` for every schedule phase."""
    if num_microbatches < 1:
        raise ValueError("num_microbatches must be positive")
    trace: list[tuple[int | None, int | None]] = [(0, None)]
    trace.extend((i, i - 1) for i in range(1, num_microbatches))
    trace.append((None, num_microbatches - 1))
    return trace


def run_combined_1f1b(
    plans: Sequence[Combined1F1BPlan],
    *,
    before_last_backward: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Run Megatron-style adjacent-microbatch combined-1F1B phases."""
    if len(plans) < 2:
        raise ValueError("combined-1F1B requires at least 2 microbatches")

    outputs = []
    for forward_mb, backward_mb in build_combined_1f1b_trace(len(plans)):
        if backward_mb is None:
            outputs.append(plans[forward_mb].forward())
        elif forward_mb is None:
            if before_last_backward is not None:
                before_last_backward()
            plans[backward_mb].backward()
        else:
            outputs.append(
                plans[forward_mb].combined_forward_backward(plans[backward_mb])
            )
    return outputs


__all__ = [
    "Combined1F1BConfig",
    "Combined1F1BLayerPlan",
    "Combined1F1BModelPlan",
    "Combined1F1BNode",
    "Combined1F1BPlan",
    "build_combined_1f1b_trace",
    "run_combined_1f1b",
]
