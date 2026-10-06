# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize('cp_size', [1, 2])
def test_cp_experts_use_ddp_and_cannot_reach_expert_finalize(
    v41_core_te, monkeypatch, cp_size
):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.ckpt.binding_records import TensorBinding
    from megatron.lite.primitive.optimizers.headwise_muon import MixedOptimizer
    from megatron.lite.primitive.parallel.owned_ddp import wrap_owned_ddp
    from megatron.lite.primitive.parallel.state import ParallelState
    from megatron.lite.runtime.contracts import ParallelConfig

    dp, dp_cp, row_group = object(), object(), object()
    ps = ParallelState(
        ep_size=1,
        cp_size=cp_size,
        dp_size=2,
        dp_cp_size=2 * cp_size,
        dp_group=dp,
        dp_cp_group=dp_cp,
    )
    model = torch.nn.Module()
    model.expert = torch.nn.Parameter(torch.ones(2))
    model.row = torch.nn.Parameter(torch.ones(2))
    model.parameter_bindings = lambda: [
        TensorBinding('expert', model, 'expert', 'expert')
    ]
    seen = []

    def ddp(chunk, *, process_group, **kwargs):
        seen.append(process_group)
        assert 'expert' not in chunk._ddp_params_and_buffers_to_ignore
        assert chunk._ddp_params_and_buffers_to_ignore == ['row']
        return chunk

    monkeypatch.setattr(torch.nn.parallel, 'DistributedDataParallel', ddp)
    wrap_owned_ddp(
        model,
        ps,
        optimizing=True,
        external_device=None,
        row_tables=[SimpleNamespace(master=model.row, buffers=lambda: ())],
        shard_group=row_group,
    )
    assert seen == [dp_cp if cp_size > 1 else dp]
    model.expert.grad = model.expert.main_grad = torch.tensor([6.0, 10.0])
    model.row.grad = model.row.main_grad = torch.tensor([8.0, 16.0])
    expert_finalize = Mock(
        side_effect=AssertionError('EP=1 reached expert-only reduction')
    )
    optimizer = SimpleNamespace(
        ps=ps, row_parameters=[model.row], finalize_expert_grads=expert_finalize
    )
    MixedOptimizer.finalize_grads(optimizer)
    expert_finalize.assert_not_called()
    assert torch.equal(model.expert.grad, torch.tensor([6.0, 10.0]))
    assert torch.equal(model.row.grad, torch.tensor([8.0, 16.0]) / (2 * cp_size))

    # If CP+EP becomes supported, this test must demand new normalization proof.
    monkeypatch.setattr(
        protocol,
        'init_parallel',
        Mock(side_effect=AssertionError('CP+EP escaped guard')),
    )
    with pytest.raises(
        NotImplementedError, match='CP_AND_EP_NOT_SIMULTANEOUSLY_SUPPORTED'
    ):
        protocol.build_model(
            object(), impl_cfg=protocol.ImplConfig(parallel=ParallelConfig(cp=2, ep=2))
        )
