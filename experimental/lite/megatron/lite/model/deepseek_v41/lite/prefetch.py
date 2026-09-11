# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch row prefetch with tagged, delayed gradient return to the row owner."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GradientTag:
    step: int
    version: int
    microbatch: object


def _record_event(device):
    if device.type != 'cuda':
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    return event


def _wait_event(event, device):
    if event is not None:
        torch.cuda.current_stream(device).wait_event(event)


class _PrefetchedTable(nn.Module):
    """A microbatch-scoped provider accepted by Engram and its FP8 projection."""

    def __init__(self, batch, microbatch):
        super().__init__()
        self.batch, self.microbatch = batch, microbatch

    def lookup_fp8(self, ids):
        batch = self.batch
        batch._check_live()
        expected = batch.ids[self.microbatch]
        _wait_event(batch.ready_event, expected.device)
        if ids.dtype != expected.dtype or ids.device != expected.device or not torch.equal(ids, expected):
            raise ValueError('Cached microbatch IDs do not match the prefetched request')
        batch.used.add(self.microbatch)
        rows = batch.rows[self.microbatch]
        if expected.is_cuda:
            consumer = torch.cuda.current_stream(expected.device)
            expected.record_stream(consumer)
            for tensor in rows:
                if tensor is not None:
                    tensor.record_stream(consumer)
        return rows

    def forward(self, ids):
        values, scales, master = self.lookup_fp8(ids)
        decoded = values.float() * scales.float().repeat_interleave(32, -1)
        if master is not None:
            decoded = master + (decoded - master).detach()
        return decoded.to(self.batch.state.table.output_dtype)


class EngramPrefetch:
    def __init__(self, state):
        self.state = state

    def start(self, step, microbatches, *, stream=None):
        """Fetch all local-batch IDs before any stage microbatch executes.

        An optional CUDA stream waits for the producer stream; consumers wait
        on a recorded ready event. CPU follows the same lifecycle synchronously.
        Every distributed rank must supply the same microbatch schedule, including
        explicit empty requests, and invoke flush in the same collective order.
        """
        if not microbatches:
            raise ValueError('Require an explicit microbatch schedule, including empty batches')
        device = self.state.table.weight.device
        for ids in microbatches.values():
            if ids.dtype != torch.int64 or ids.device != device:
                raise ValueError('Prefetch IDs must be resident int64 tensors')
        if stream is not None:
            if device.type != 'cuda':
                raise ValueError('CUDA prefetch stream requires CUDA table storage')
            stream.wait_stream(torch.cuda.current_stream(device))
        self.state.begin(step)
        try:
            with torch.enable_grad(), (torch.cuda.stream(stream) if stream is not None else nullcontext()):
                return PrefetchedBatch(self.state, step, microbatches)
        except Exception:
            self.state.active_step, self.state._ready = None, False
            raise


class PrefetchedBatch:
    def __init__(self, state, step, microbatches):
        self.state, self.step, self.version = state, step, state.version
        self.closed = False
        for ids in microbatches.values():
            if ids.is_cuda:
                ids.record_stream(torch.cuda.current_stream(ids.device))
        self.ids = {key: ids.detach().clone() for key, ids in microbatches.items()}
        self.rows, self.returns, self.return_events, self.used = {}, {}, {}, set()
        concatenated = torch.cat([ids.reshape(-1) for ids in self.ids.values()])
        values, scales, self.source_master = state.table.lookup_fp8(concatenated)
        offset = 0
        for key, ids in self.ids.items():
            end = offset + ids.numel()
            master = None
            if self.source_master is not None:
                master = self.source_master[offset:end].detach().clone().reshape(*ids.shape, values.shape[-1])
                master.requires_grad_()
                tag = GradientTag(step, self.version, key)
                master.register_hook(lambda gradient, tag=tag: self.return_gradient(tag, gradient))
            self.rows[key] = (
                values[offset:end].reshape(*ids.shape, values.shape[-1]),
                scales[offset:end].reshape(*ids.shape, scales.shape[-1]), master)
            offset = end
        self.ready_event = _record_event(values.device)

    def _check_live(self):
        if self.closed:
            raise RuntimeError('Prefetched batch is closed')
        if self.state.active_step != self.step or self.state.version != self.version:
            raise RuntimeError('Stale prefetch step/version tag')

    def view(self, microbatch):
        self._check_live()
        if microbatch not in self.ids:
            raise ValueError('Unknown microbatch tag')
        return _PrefetchedTable(self, microbatch)

    def return_gradient(self, tag, gradient):
        self._check_live()
        if tag.step != self.step or tag.version != self.version or tag.microbatch not in self.ids:
            raise RuntimeError('Gradient tag does not match the prefetched generation')
        if tag.microbatch in self.returns:
            raise RuntimeError('duplicate gradient return for microbatch')
        master = self.rows[tag.microbatch][2]
        if master is None or gradient.dtype != torch.float32 or gradient.shape != master.shape or gradient.device != master.device:
            raise ValueError('Gradient return must match the resident FP32 microbatch rows')
        self.returns[tag.microbatch] = gradient.detach().clone()
        self.return_events[tag.microbatch] = _record_event(gradient.device)
        # Returning None leaves the leaf gradient unchanged for autograd.

    def flush(self):
        """Run after backbone backward; return once through the original route."""
        self._check_live()
        trainable = self.source_master is not None
        completed = self.returns.keys() if trainable else self.used
        missing = self.ids.keys() - completed
        if missing:
            raise RuntimeError(f'missing microbatch returns/consumers: {list(missing)}')
        device = self.state.table.weight.device
        _wait_event(self.ready_event, device)
        gradient = None
        if trainable:
            for event in self.return_events.values():
                _wait_event(event, device)
            combined = torch.cat([self.returns[key].reshape(-1, self.source_master.shape[-1]) for key in self.ids])
            gradient, = torch.autograd.grad(self.source_master, self.state.table.master, combined)
        self.state.accept_gradient(self.step, gradient)
        self.closed = True
        self.rows.clear()
        self.returns.clear()
        self.return_events.clear()
        self.ids.clear()
        self.used.clear()
        self.source_master = None
