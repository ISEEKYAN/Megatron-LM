# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generic owner policy without an architecture-specific router or optimizer."""

from types import SimpleNamespace

import torch
from megatron.lite.primitive.optimizers.partitioned_mixed import (
    PartitionedMixedOptimizer,
)
from megatron.lite.primitive.optimizers.vision_config import OptimizerConfig


def test_partitioned_optimizer_accumulates_and_publishes_after_update():
    model = torch.nn.Linear(2, 2, bias=False)
    published = []
    router = SimpleNamespace(update_bias=lambda stats: published.append(stats))
    config = OptimizerConfig(0.01, 1, 'unused')
    optimizer = PartitionedMixedOptimizer(
        model,
        config,
        group_builder=lambda: [{'params': [model.weight], 'algorithm': 'adamw'}],
        owners=lambda: ([], [router], [], None),
        stats_factory=lambda counts, total: SimpleNamespace(
            counts=counts, total_tokens=total
        ),
    )
    stats = SimpleNamespace(
        counts=torch.tensor([2.0, 3.0]), total_tokens=torch.tensor(5.0)
    )
    optimizer.accumulate_modality_loads([[stats, stats]])
    assert not published, 'LOAD_PUBLICATION_REQUIRES_STEP'
    model(torch.ones(1, 2)).sum().backward()
    assert optimizer.step()[0]
    assert len(published) == 1
    assert torch.equal(published[0].counts, stats.counts * 2)
    assert optimizer._modality_loads == [None]
    optimizer.zero_grad()
    optimizer.accumulate_modality_loads([[stats]])
    model(torch.ones(1, 2)).sum().backward()
    model.weight.main_grad.fill_(float('nan'))
    assert not optimizer.step()[0]
    assert len(published) == 1, 'FAILED_UPDATE_DOES_NOT_PUBLISH'
