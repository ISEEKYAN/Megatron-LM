# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One-microbatch ownership of Shared-source compressed KV tensors."""
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

_REPLAY: ContextVar = ContextVar("checkpoint_replay", default=None)


def replayed(produce: Callable[[], Any]) -> Any:
    """``produce()`` once per checkpointed segment: the first forward keeps the value and
    the recompute replay takes it back in call order instead of producing it again.

    For values whose backward needs only the segment's own saved inputs, so the replay
    can skip the producer (its kernels and collectives) and return the same bits.
    Outside a checkpoint this is ``produce()``.
    """
    memo = _REPLAY.get()
    if memo is None:
        return produce()
    values, recording = memo
    if recording:
        values.append(produce())
        return values[-1]
    return values.popleft()


class KVSourceState:
    """Keep local owner outputs attached to their original autograd graph."""

    def __init__(self) -> None:
        self.active = False
        self.values = {}
        self._logical_grad_mode = None

    @property
    def grad_mode(self) -> bool:
        """Logical segment mode stays fixed through a checkpoint replay."""
        return (
            torch.is_grad_enabled() if self._logical_grad_mode is None else self._logical_grad_mode
        )

    @contextmanager
    def microbatch(self) -> Iterator["KVSourceState"]:
        if self.active:
            raise RuntimeError("A KV source microbatch is already active")
        self.active = True
        try:
            yield self
        finally:
            self.values.clear()
            self.active = False

    def resolve(self, layer: int, owner: int, geometry: tuple, produce: Callable[[], Any]) -> Any:
        if not self.active:
            raise RuntimeError("KV reuse requires an active microbatch scope")
        key = (owner, geometry)
        if layer == owner:
            self.values[key] = produce()
        elif key not in self.values:
            raise RuntimeError("KV owner has not produced this layout and grad mode")
        return self.values[key]

    def checkpoint(self, function: Callable, *args: torch.Tensor, checkpoint_fn=None) -> Any:
        """Pass shared source tensors across native Core checkpoint boundaries."""
        if not self.active:
            raise RuntimeError("Checkpointed KV reuse requires an active microbatch")
        if not all(isinstance(arg, torch.Tensor) for arg in args):
            raise TypeError("Shared-source checkpoint arguments must be tensors")
        if checkpoint_fn is None:
            from megatron.core.tensor_parallel.random import checkpoint

            checkpoint_fn = checkpoint
        leaves, input_spec = tree_flatten(self.values)
        slots = [i for i, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
        saved_sources = tuple(leaves[i] for i in slots)
        for slot in slots:
            leaves[slot] = None
        argument_count = len(args)
        logical_mode = self.grad_mode
        metadata = []
        memo = deque()

        def segment(*saved):
            local_leaves = list(leaves)
            for slot, tensor in zip(slots, saved[argument_count:]):
                local_leaves[slot] = tensor
            inputs = tree_unflatten(local_leaves, input_spec)
            old_values, old_active, old_mode = self.values, self.active, self._logical_grad_mode
            self.values, self.active, self._logical_grad_mode = dict(inputs), True, logical_mode
            token = _REPLAY.set((memo, not metadata))
            try:
                result = function(*saved[:argument_count])
                published = {
                    key: value
                    for key, value in self.values.items()
                    if key not in inputs or value is not inputs[key]
                }
                values, spec = tree_flatten((result, published))
                tensor_slots = [
                    i for i, value in enumerate(values) if isinstance(value, torch.Tensor)
                ]
                if metadata:
                    if metadata[0][1:] != (spec, tensor_slots):
                        raise RuntimeError(
                            "Shared-source checkpoint changed its source publication structure"
                        )
                else:
                    metadata.append(
                        (
                            [
                                None if isinstance(value, torch.Tensor) else value
                                for value in values
                            ],
                            spec,
                            tensor_slots,
                        )
                    )
                return tuple(values[i] for i in tensor_slots)
            finally:
                _REPLAY.reset(token)
                self.values, self.active, self._logical_grad_mode = old_values, old_active, old_mode

        output = checkpoint_fn(segment, False, *args, *saved_sources)
        if isinstance(output, torch.Tensor):
            output = (output,)
        values, spec, tensor_slots = metadata[0]
        values = list(values)
        for slot, tensor in zip(tensor_slots, output):
            values[slot] = tensor
        result, published = tree_unflatten(values, spec)
        self.values.update(published)
        return result
