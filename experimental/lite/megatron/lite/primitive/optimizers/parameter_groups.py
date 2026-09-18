# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build optimizer groups from explicit object roles and logical matrix layouts."""

import math
from operator import attrgetter


class OwnedParameterGroups:
    def __init__(self, model, rules, lr):
        self.rules, self.lr = rules, lr
        if not math.isfinite(lr) or lr < 0:
            raise ValueError('Invalid base learning rate')
        model.validate_parameter_bindings()
        self.bindings = {id(b.tensor): b for b in model.parameter_bindings()}
        registered = list(model.named_parameters(remove_duplicate=False))
        if len(registered) != len({id(p) for _, p in registered}):
            raise ValueError('Unexpected parameter alias in module tree')
        self.groups, self.seen = [], set()

    def add(
        self,
        p,
        role,
        *,
        shape=None,
        heads=None,
        partitions=None,
        vector=False,
        **policy
    ):
        if id(p) in self.seen:
            raise ValueError('Duplicate optimizer owner')
        self.seen.add(id(p))
        b = self.bindings.get(id(p))
        if b is None or b.role != role:
            raise ValueError('Unknown or mismatched parameter owner role')
        if b.head_count != heads:
            raise ValueError('Unresolved or incorrect logical head count')
        if not p.requires_grad:
            return
        algorithm, matrix_decay, vector_decay, multiplier = self.rules[role]
        # Selection follows logical matrix rank, never a release-key prefix.
        if vector:
            algorithm = 'adamw'
        if algorithm in ('muon', 'sinkhorn') and p.ndim != 2:
            raise ValueError(
                'Matrix optimizer requires the declared two-dimensional owner'
            )
        if shape is not None and math.prod(shape) != p.numel():
            raise ValueError('Logical matrix shape disagrees with actual owner')
        if (
            shape is not None
            and len(shape) == 3
            and tuple(p.shape) != (shape[0] * shape[1], shape[2])
        ):
            raise ValueError('Head layout disagrees with physical matrix axes')
        decay = vector_decay if vector else matrix_decay
        self.groups.append(
            dict(
                params=[p],
                algorithm=algorithm,
                owner_key=b.release_key,
                matrix_shape=tuple(p.shape) if shape is None else tuple(shape),
                matrix_partitions=partitions,
                lr=self.lr * policy.get('multiplier', multiplier),
                weight_decay=policy.get('decay', decay),
            )
        )

    def route(self, owner, paths, role, **policy):
        for path in paths.split():
            self.add(attrgetter(path)(owner), role, **policy)

    def visual_linear(self, module, role, *, multiplier=1, partitions=None):
        shape = (module.out_features, module.in_features)
        if (
            math.prod(shape) != module.weight.numel()
            or tuple(module.weight.shape) != shape
        ):
            raise ValueError('Visual linear shape disagrees with physical owner')
        if module.bias is not None and tuple(module.bias.shape) != (shape[0],):
            raise ValueError('Visual bias shape disagrees with physical owner')
        if partitions is not None and (
            any(any(type(d) is not int or d < 1 for d in part) for part in partitions)
            or sum(math.prod(part) for part in partitions) != module.weight.numel()
            or any(part[-1] != shape[-1] for part in partitions)
        ):
            raise ValueError('Logical partitions disagree with visual matrix shape')
        self.add(
            module.weight,
            role,
            shape=shape,
            multiplier=multiplier,
            partitions=partitions,
        )
        if module.bias is not None:
            self.add(module.bias, role, multiplier=multiplier, decay=0, vector=True)

    def finish(self):
        if self.seen != set(self.bindings):
            raise ValueError('Unknown parameter owner; no catch-all optimizer route')
        return self.groups
