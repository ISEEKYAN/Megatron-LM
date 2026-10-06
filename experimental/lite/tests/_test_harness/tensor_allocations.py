# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


class AllocationPeak(TorchDispatchMode):
    """Observe live ATen output storage bytes, excluding pre-existing inputs."""

    def __init__(self, inputs):
        self.existing = {x.untyped_storage()._cdata for x in inputs}
        self.live = {}
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for tensor in tree_leaves(result):
            if not isinstance(tensor, torch.Tensor):
                continue
            storage = tensor.untyped_storage()
            key = storage._cdata
            if key not in self.existing:
                self.live[key] = (weakref.ref(storage), storage.nbytes())
        self.live = {
            key: value for key, value in self.live.items() if value[0]() is not None
        }
        self.peak = max(self.peak, sum(size for _, size in self.live.values()))
        return result
