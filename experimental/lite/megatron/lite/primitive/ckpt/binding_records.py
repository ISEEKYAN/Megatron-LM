# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit object ownership, independent of release key and codec policy."""

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class TensorBinding:
    name: str
    owner: nn.Module
    attribute: str
    sources: tuple[str, ...]
    row_key: str | None = None

    @property
    def tensor(self):
        return getattr(self.owner, self.attribute)


def validate_parameter_bindings(model, bindings):
    """Require one live binding per parameter, including frozen/QAT masters."""
    names = [binding.name for binding in bindings]
    ids = [id(binding.tensor) for binding in bindings]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate binding name or parameter owner")
    if set(ids) != {id(parameter) for parameter in model.parameters()}:
        raise ValueError("Every parameter must have exactly one binding")
    if any(not binding.sources for binding in bindings):
        raise ValueError("Each binding needs source keys")
    return bindings
