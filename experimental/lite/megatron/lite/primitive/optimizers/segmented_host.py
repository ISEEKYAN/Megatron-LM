# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded owner updates with CPU masters/moments and two-pass validation.

The first pass validates every candidate without publishing any weight or
moment. A global vote precedes a deterministic second pass that recomputes
and publishes owners one at a time. Original gradients stay immutable. A
numerical rejection returns False with no publication; an unexpected device
failure during publication poisons the optimizer and must terminate the job,
just as a device failure during the original transaction's copy_ phase does.
"""

import torch

from .headwise_muon import _parameters


class SegmentedHostUpdate:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.masters = {}
        self.busy = self.broken = False
        self.last_metrics = {}

    def require_idle(self):
        if self.busy or self.broken:
            raise RuntimeError('Segmented host optimizer is busy or publication failed')

    def _masters(self):
        for parameter in _parameters(self.optimizer):
            if id(parameter) not in self.masters:
                self.masters[id(parameter)] = parameter.detach().to('cpu')
        if any(value.device.type != 'cpu' for value in self.masters.values()):
            raise RuntimeError('FP32 authoritative masters must remain on host')

    @torch.no_grad()
    def prepare_model_transfer(self, device):
        self.require_idle()
        if device != 'cpu':
            return
        self._masters()
        # Reuse the authoritative storage, avoiding a second CPU model cache.
        # Completed training windows do not retain gradients into rollout.
        for parameter in _parameters(self.optimizer):
            parameter.grad = parameter.main_grad = None
            parameter.data = self.masters[id(parameter)]

    @torch.no_grad()
    def load_state(self, state):
        self.require_idle()
        # Model checkpoint has already restored numerical weights. No duplicate
        # master bank is serialized inside the optimizer checkpoint.
        self.masters.clear()
        self._masters()
        parameters = _parameters(self.optimizer)
        device_storage = [p.data for p in parameters]
        try:
            # torch Optimizer.load_state_dict normally casts moments onto p.device.
            # Restore against the same owners' CPU storage, then reattach caches.
            for p in parameters:
                p.data = self.masters[id(p)]
            for backend, saved in zip(self.optimizer.optimizers, state['optimizers']):
                backend.load_state_dict(saved)
        finally:
            for p, data in zip(parameters, device_storage, strict=True):
                p.data = data
        self.optimizer.offload_state_to_cpu()

    def _segments(self):
        for backend in self.optimizer.optimizers:
            for group in backend.param_groups:
                for parameter in group['params']:
                    yield backend, group, parameter

    @torch.no_grad()
    def _prepare(self, backend, group, parameter, gradient, coefficient):
        # A FP32 execution cache remains resident for native training. Reload
        # only this owner from the authoritative CPU master for update math.
        parameter.copy_(self.masters[id(parameter)])
        scaled = None if gradient is None else gradient * coefficient
        state = backend.state.get(parameter, {})
        device_state = {
            key: (
                value.to('cpu' if key == 'step' else parameter.device, copy=True)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in state.items()
        }
        if isinstance(backend, torch.optim.AdamW):
            if scaled is None:
                return None
            candidate = torch.nn.Parameter(parameter.detach().clone())
            private = torch.optim.AdamW([{**group, 'params': [candidate]}])
            private.state[candidate] = device_state
            candidate.grad = scaled
            private.step()
            return candidate.detach(), private.state[candidate], None
        if scaled is None and backend._label != 'Sinkhorn':
            return None
        previous_groups, previous_grad = backend.param_groups, parameter.grad
        previous_main = parameter.main_grad
        previous_state = backend.state.get(parameter)
        try:
            backend.param_groups = [{**group, 'params': [parameter]}]
            backend.state[parameter] = device_state
            parameter.grad = parameter.main_grad = scaled
            if not backend.prepare_step():
                return False
            value = backend.candidates()[0][1]
            momentum = backend._prepared[0][2]
            return value, {backend._momentum_key: momentum}, None
        finally:
            backend.discard_step()
            backend.param_groups = previous_groups
            parameter.grad, parameter.main_grad = previous_grad, previous_main
            if previous_state is None:
                backend.state.pop(parameter, None)
            else:
                backend.state[parameter] = previous_state

    def _table(self, parameter, value):
        from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8

        for table in self.optimizer.tables:
            if table.master is parameter:
                weight, scale = quantize_block_fp8(value, (1, 32), scale_format='e8m0')
                return table, weight, scale
        return None

    @staticmethod
    def _valid(candidate):
        if candidate is False:
            return False
        if candidate is None:
            return True
        value, state, storage = candidate
        values = [value, *[v for v in state.values() if isinstance(v, torch.Tensor)]]
        if storage is not None:
            values.extend(v.float() for v in storage[1:])
        return all(bool(torch.isfinite(v).all()) for v in values)

    @torch.no_grad()
    def step(self):
        self.require_idle()
        self._masters()
        optimizer = self.optimizer
        parameters = _parameters(optimizer)
        gradients = [
            p.main_grad if p.main_grad is not None else p.grad for p in parameters
        ]
        if any(
            g.dtype != torch.float32 or g.is_sparse for g in gradients if g is not None
        ):
            raise ValueError('Expected native dense FP32 main_grad')
        norm = optimizer._grad_norm(parameters, gradients)
        if not optimizer._all_finite(bool(torch.isfinite(norm))):
            return False, float(norm), None
        coefficient = (
            min(1.0, optimizer.config.clip_grad / (float(norm) + 1e-6))
            if optimizer.config.clip_grad
            else 1.0
        )
        by_id = {id(p): g for p, g in zip(parameters, gradients, strict=True)}
        self.busy = True
        committing = False
        count = 0
        try:
            valid = True
            # Finish every owner, including row collectives, before global vote.
            # Stages may own different numbers of parameters.
            for backend, group, parameter in self._segments():
                candidate = self._prepare(
                    backend, group, parameter, by_id[id(parameter)], coefficient
                )
                if candidate is not None and candidate is not False:
                    candidate = (*candidate[:2], self._table(parameter, candidate[0]))
                valid = self._valid(candidate) and valid
                del candidate
                count += 1
            if not optimizer._all_finite(valid):
                return False, float(norm), None
            committing = True
            for backend, group, parameter in self._segments():
                candidate = self._prepare(
                    backend, group, parameter, by_id[id(parameter)], coefficient
                )
                if candidate is None:
                    continue
                if not self._valid(candidate):
                    raise RuntimeError(
                        'Validated segmented update changed during publication'
                    )
                value, state, _ = candidate
                storage = self._table(parameter, value)
                # Synchronous copies publish bits to CPU masters and GPU caches.
                host = self.masters[id(parameter)]
                host.copy_(value)
                parameter.copy_(value)
                backend.state[parameter] = {
                    key: (
                        tensor.to('cpu') if isinstance(tensor, torch.Tensor) else tensor
                    )
                    for key, tensor in state.items()
                }
                if storage is not None:
                    table, weight, scale = storage
                    table.weight.copy_(weight)
                    table.scale.copy_(scale)
                del candidate, value, state, storage
            self.last_metrics = {
                'segments': count,
                'validation_passes': 1,
                'publication_passes': 1,
                'host_master_bytes': sum(
                    v.numel() * v.element_size() for v in self.masters.values()
                ),
                'host_state_bytes': sum(
                    v.numel() * v.element_size()
                    for b in optimizer.optimizers
                    for s in b.state.values()
                    for v in s.values()
                    if isinstance(v, torch.Tensor)
                ),
            }
            return True, float(norm), None
        except Exception:
            self.broken = committing
            raise
        finally:
            self.busy = False
