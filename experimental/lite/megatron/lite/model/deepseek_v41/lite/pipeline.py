# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paired PP payload lifetime; the C4 protocol owns scheduling and layer binding."""

from dataclasses import dataclass

import torch

PAYLOAD_FIELDS = (
    'h',
    'p',
    'ced_h',
    'ced_p',
    'kv',
    'kv_values',
    'kv_scale',
    'index_k',
    'index_scale',
    'topk',
    'positions',
    'latent',
    'candidates',
)


@dataclass(frozen=True)
class PipelineTag:
    step: int
    microbatch: int
    chunk: int
    generation: int
    kv_owner: int = 20
    index_owner: int = 20

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in self.as_tuple()[:4]):
            raise ValueError('Require nonnegative integer pipeline tags')
        if any(type(v) is not int or not -1 <= v < 40 for v in self.as_tuple()[4:]):
            raise ValueError('Invalid pipeline source owner')

    def as_tuple(self):
        return (
            self.step,
            self.microbatch,
            self.chunk,
            self.generation,
            self.kv_owner,
            self.index_owner,
        )


@dataclass(frozen=True)
class PairedPayload:
    h: torch.Tensor
    p: torch.Tensor
    ced_h: torch.Tensor | None = None
    ced_p: torch.Tensor | None = None
    kv: torch.Tensor | None = None
    kv_values: torch.Tensor | None = None
    kv_scale: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    index_scale: torch.Tensor | None = None
    topk: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    latent: torch.Tensor | None = None
    candidates: torch.Tensor | None = None

    def __post_init__(self):
        if not isinstance(self.h, torch.Tensor) or not isinstance(self.p, torch.Tensor):
            raise ValueError('Require a current HC pair')
        if (self.ced_h is None) != (self.ced_p is None):
            raise ValueError('CED h/p must travel as a pair')
        for h, p in ((self.h, self.p), (self.ced_h, self.ced_p)):
            if h is not None and (
                p is None
                or h.ndim != 4
                or p.shape != h.shape[:-1]
                or not h.is_floating_point()
                or not p.is_floating_point()
            ):
                raise ValueError(
                    'HC pair requires h[batch,sequence,hc,hidden] and p[batch,sequence,hc]'
                )
        for name in (
            'kv_values',
            'kv_scale',
            'index_k',
            'index_scale',
            'topk',
            'positions',
            'candidates',
        ):
            value = getattr(self, name)
            if value is not None and value.requires_grad:
                raise ValueError(
                    'Published bytes, integer positions and frozen indexers have no gradient'
                )
        for name in ('topk', 'positions'):
            value = getattr(self, name)
            if value is not None and value.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    'Pipeline selection and position metadata must be integer tensors'
                )

    def tensors(self):
        return tuple(getattr(self, name) for name in PAYLOAD_FIELDS)

    def differentiable(self):
        return {
            name: value
            for name, value in zip(PAYLOAD_FIELDS, self.tensors())
            if value is not None and value.requires_grad
        }

    @classmethod
    def from_tensors(cls, tensors):
        if len(tensors) != len(PAYLOAD_FIELDS):
            raise ValueError('Pipeline payload field count differs')
        return cls(**dict(zip(PAYLOAD_FIELDS, tensors)))


class PipelineLedger:
    """Keep each generation live through recompute and all consumer returns.

    A return names a consumer explicitly and supplies every differentiable field
    (zero for an unused path). Indexers/Top-K never enter autograd. Backward runs
    once after the exact expected consumer set has returned; no optimizer is
    registered here, so C4/F2 retain canonical parameter ownership.
    """

    def __init__(self):
        self._live = {}
        self._latest = {}
        self._finished_step = -1

    def publish(self, tag, payload, *, consumers):
        identity = (tag.step, tag.microbatch, tag.chunk)
        if (
            tag.step <= self._finished_step
            or tag.generation <= self._latest.get(identity, -1)
            or any(t.as_tuple()[:3] == identity for t in self._live)
        ):
            raise RuntimeError('Duplicate or still-live pipeline generation')
        consumers = tuple(consumers)
        if not consumers or len(set(consumers)) != len(consumers):
            raise ValueError('Require distinct expected pipeline consumers')
        self._latest[identity] = tag.generation
        self._live[tag] = (payload, set(consumers), set(), {})

    def _entry(self, tag):
        if tag not in self._live:
            raise RuntimeError('Unknown, stale or released pipeline generation')
        return self._live[tag]

    def read(self, tag):
        return self._entry(tag)[0]

    def return_gradients(self, tag, consumer, gradients):
        payload, expected, returned, total = self._entry(tag)
        if consumer not in expected or consumer in returned:
            raise RuntimeError('Unexpected or duplicate pipeline gradient return')
        fields = payload.differentiable()
        if gradients.keys() != fields.keys():
            raise ValueError('Gradient fields differ from the floating owner paths')
        for name, tensor in fields.items():
            gradient = gradients[name]
            if (
                gradient.shape != tensor.shape
                or gradient.dtype != tensor.dtype
                or gradient.device != tensor.device
            ):
                raise ValueError('Pipeline gradient shape/dtype/device differs')
        for name, gradient in gradients.items():
            if name not in total:
                total[name] = gradient.detach().clone()
            else:
                total[name].add_(gradient.detach())
        returned.add(consumer)

    def backward(self, tag):
        payload, expected, returned, total = self._entry(tag)
        if returned != expected:
            raise RuntimeError('Cannot release pipeline state with missing returns')
        fields = payload.differentiable()
        if fields:
            torch.autograd.backward(
                tuple(fields.values()), tuple(total[name] for name in fields)
            )
        del self._live[tag]

    def assert_quiescent(self):
        if self._live:
            raise RuntimeError('Pipeline generations are still live')

    def finish_step(self, step):
        """Bound generation bookkeeping after the scheduler drains a step."""
        if type(step) is not int or step <= self._finished_step:
            raise RuntimeError('Stale pipeline step retirement')
        if any(tag.step <= step for tag in self._live):
            raise RuntimeError('Pipeline generations are still live')
        self._finished_step = step
        self._latest = {
            identity: generation
            for identity, generation in self._latest.items()
            if identity[0] > step
        }


class PackedPipelineAdapter:
    """Sequence-preserving paired transport for the common PP scheduler.

    Each packed sequence has its own ledger and wire tag; no sequence dimension
    is flattened into another sample's CSA2 state. Frozen source index keys and
    candidates are read-only shadows. Only floating owner paths return gradients.
    """

    def __init__(self, model, parallel_state, *, recompute=False):
        self.model, self.ps = model, parallel_state
        self.recompute = recompute
        self.step = 0
        self._active = False
        self._records = {}
        self._ledgers = {}

    def begin(self, count, *, forward_only=False):
        if self._active or self._records:
            raise RuntimeError('Pipeline step is still live')
        if count < 1:
            raise ValueError('Require at least one pipeline microbatch')
        self._active = True
        self._count = count
        self._forward_only = forward_only

    @staticmethod
    def owners(boundary):
        layer = boundary - 1
        if layer < 2:
            return (-1, -1)
        kv = 20 if layer >= 20 else 14 if layer >= 14 else 8 if layer >= 8 else 2
        index = 20 + ((layer - 20) // 4) * 4 if layer >= 20 else kv
        return kv, index

    def _tag(self, boundary, microbatch):
        return PipelineTag(self.step, microbatch, boundary, 0, *self.owners(boundary))

    def _transfer(self, tensors, boundary, microbatch, sequence, phase, *, send):
        from megatron.lite.primitive.parallel import tensor_payload

        tag = (*self._tag(boundary, microbatch).as_tuple(), sequence, phase)
        forward = phase != 1
        peer = self.ps.pp_next_rank if send == forward else self.ps.pp_prev_rank
        kwargs = dict(
            peer=peer,
            group=self.ps.pp_group,
            device=next(self.model.parameters()).device,
        )
        if send:
            return tensor_payload.send_tensor_payload(tensors, tag, **kwargs)
        return tensor_payload.recv_tensor_payload(tag, **kwargs)

    def forward(self, batch, microbatch):
        from .protocol import packed_pipeline_forward_step

        if (
            not self._active
            or microbatch in self._records
            or not 0 <= microbatch < self._count
        ):
            raise RuntimeError('Unknown or duplicate pipeline microbatch')
        start, end = self.model.local_layer_range
        state = None
        if self.ps.pp_rank:
            (lengths,) = self._transfer(None, start, microbatch, 0, 2, send=False)
            if not torch.equal(lengths, batch.seq_lens):
                raise ValueError(
                    'Packed pipeline sequence boundaries differ across stages'
                )
            state = tuple(
                (
                    PairedPayload.from_tensors(
                        self._transfer(None, start, microbatch, sequence, 0, send=False)
                    ),
                    self.owners(start),
                )
                for sequence in range(len(batch.seq_lens))
            )

        def run():
            return packed_pipeline_forward_step(
                self.model, batch, start=start, end=end, state=state
            )

        if self._forward_only:
            with torch.no_grad():
                out = run()
        elif self.recompute:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(run, use_reentrant=False, preserve_rng_state=True)
        else:
            out = run()
        outputs = out['packed_pipeline_state']
        self._records[microbatch] = (state, len(outputs))
        if self.ps.pp_rank < self.ps.pp_size - 1:
            self._transfer((batch.seq_lens,), end, microbatch, 0, 2, send=True)
            for sequence, (payload, owners) in enumerate(outputs):
                if owners != self.owners(end):
                    raise ValueError('Pipeline range changed canonical source owners')
                if not self._forward_only:
                    ledger = self._ledgers.setdefault(sequence, PipelineLedger())
                    ledger.publish(
                        self._tag(end, microbatch),
                        payload,
                        consumers=(self.ps.pp_next_rank,),
                    )
                self._transfer(
                    payload.tensors(), end, microbatch, sequence, 0, send=True
                )
        # Graph ownership is retained by ledger or by final-stage loss, not metrics.
        return {
            key: value for key, value in out.items() if key != 'packed_pipeline_state'
        }

    def backward(self, microbatch, loss):
        if not self._active or self._forward_only or microbatch not in self._records:
            raise RuntimeError('Unknown, stale or released pipeline microbatch')
        state, sequences = self._records[microbatch]
        start, end = self.model.local_layer_range
        if self.ps.pp_rank == self.ps.pp_size - 1:
            if loss is None:
                raise ValueError('Final pipeline stage requires a scalar loss')
            loss.backward()
        else:
            for sequence in range(sequences):
                values = self._transfer(None, end, microbatch, sequence, 1, send=False)
                ledger = self._ledgers[sequence]
                tag = self._tag(end, microbatch)
                ledger.return_gradients(
                    tag,
                    self.ps.pp_next_rank,
                    {
                        name: value
                        for name, value in zip(PAYLOAD_FIELDS, values)
                        if value is not None
                    },
                )
                ledger.backward(tag)
        if state is not None:
            for sequence, (payload, _) in enumerate(state):
                gradients = {
                    name: torch.zeros_like(value) if value.grad is None else value.grad
                    for name, value in payload.differentiable().items()
                }
                self._transfer(
                    tuple(gradients.get(name) for name in PAYLOAD_FIELDS),
                    start,
                    microbatch,
                    sequence,
                    1,
                    send=True,
                )
        del self._records[microbatch]

    def finish(self):
        if not self._active:
            raise RuntimeError('No active pipeline step')
        if self._forward_only:
            self._records.clear()
        if self._records:
            raise RuntimeError('Pipeline microbatches are still live')
        for ledger in self._ledgers.values():
            ledger.assert_quiescent()
            ledger.finish_step(self.step)
        self._active = False
        self.step += 1
