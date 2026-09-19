# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Object ownership records for checkpoint and optimizer inventories."""
from collections import namedtuple
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class TensorBinding:
    release_key: str
    owner: nn.Module
    attribute: str | None
    role: str
    head_count: int | None = None
    encoding: str | None = None

    @property
    def tensor(self):
        return None if self.attribute is None else getattr(self.owner, self.attribute)


class DeferredModule(nn.Module):
    """Placeholder for an inactive archival subtree."""

    def __init__(self, scope):
        super().__init__()
        self.scope = scope

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f'{self.scope} execution is not implemented in text-only mode'
        )


Rule = namedtuple(
    'Rule', 'attributes role encoding shape key', defaults=(None, None, None)
)


from fnmatch import fnmatchcase


def validate_parameter_bindings(self):
    ids = [id(b.tensor) for b in self.parameter_bindings()]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in self.parameters()}:
        raise ValueError('Every parameter must have exactly one binding')


def initialize_bindings(self, layer_range, count):
    start, end = (0, count) if layer_range is None else layer_range
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= count:
        raise ValueError('Invalid local pipeline stage interval')
    self.local_layer_range = (start, end)
    self._input_tensor = None
    self.tensor_bindings = {}
    self.archival_bindings = {}
    self.archival_store = None
    return start, end


def _bind(self, key, owner, attribute, role, head_count=None, encoding=None):
    if key in self.tensor_bindings:
        raise ValueError(f'duplicate binding: {key}')
    self.tensor_bindings[key] = TensorBinding(
        key, owner, attribute, role, head_count, encoding
    )
    sibling = self._scale_binding(key, owner, attribute, role, encoding)
    if sibling is not None:
        self.tensor_bindings[sibling.release_key] = sibling


def bind_rules(self, rules, extra_rules):
    for path, owner in self.named_modules():
        entries = [
            rule for pattern, rule in rules.items() if fnmatchcase(path, pattern)
        ]
        entries.extend(extra_rules(path, owner))
        for attributes, role, encoding, shape, key in entries:
            for attribute in attributes.split():
                tensor = getattr(owner, attribute, None)
                if tensor is None:
                    continue
                name = (key or '{module}.{a}').format(
                    module=path,
                    a=attribute,
                    parent=path.rsplit('.', 1)[0],
                    grandparent=path.rsplit('.', 2)[0],
                )
                self._bind(
                    name,
                    owner,
                    attribute,
                    role,
                    None if shape is None else shape[0],
                    encoding,
                )
