# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Candidate publication shared by explicit matrix optimizer compositions."""
import torch


def _idle(self, action):
    if self._prepared is not None:
        raise RuntimeError(f'Cannot {action} a prepared {self._label} step')


def candidates(self):
    if self._prepared is None:
        raise RuntimeError(f'No prepared {self._label} step')
    return tuple((p, value) for p, value, _ in self._prepared)


@torch.no_grad()
def commit_step(self):
    for (p, value), (_, _, momentum) in zip(self.candidates(), self._prepared):
        p.copy_(value)
        self.state[p][self._momentum_key] = momentum
    self.discard_step()


def discard_step(self):
    self._prepared = None


def step(self, closure=None):
    if closure is not None:
        raise ValueError(f'{self._label} requires explicit accumulated gradients')
    if not self.prepare_step():
        return False
    self.commit_step()
    return True


def state_dict(self):
    self._idle('checkpoint')
    return torch.optim.Optimizer.state_dict(self)


def _validate_momentum(self, saved, shape):
    if saved is None:
        return
    momentum = saved.get(self._momentum_key)
    if (
        set(saved) != {self._momentum_key}
        or not isinstance(momentum, torch.Tensor)
        or momentum.dtype != torch.float32
        or momentum.shape != shape
        or not torch.isfinite(momentum).all()
    ):
        raise ValueError(
            f'{self._label} checkpoint requires matching finite FP32 momentum'
        )
