# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts: logical documents, scoped FIFO replay, and packed reduction."""

import pytest
import torch
from megatron.lite.primitive.modules.paired_stream import packed_forward
from megatron.lite.primitive.modules.router_replay import (
    PackedRouterReplay,
    RouterReplay,
)
from megatron.lite.primitive.modules.router_replay import RouterReplayAction as Action
from megatron.lite.primitive.ops.packed_objective import packed_objective


@pytest.fixture(autouse=True)
def clean_replay():
    RouterReplay.clear_global_router_replay_instances()
    yield
    RouterReplay.clear_global_router_replay_instances()


def test_objective_mask_temperature_entropy_and_denominator():
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    labels = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    out = packed_objective(
        logits,
        labels,
        mask,
        temperature=2.0,
        denominator=4.0,
        calculate_entropy=True,
        loss_scale=3.0,
    )
    expected = torch.log(torch.tensor(2.0))
    assert torch.equal(out['loss'], expected * 1.5)
    assert torch.equal(out['log_probs'], -expected.expand_as(labels))
    assert torch.equal(out['entropy'], expected.expand_as(labels))
    out['loss'].backward()
    assert torch.equal(
        logits.grad, torch.tensor([[[-0.1875, 0.1875], [0.0, 0.0], [-0.1875, 0.1875]]])
    )
    assert packed_objective(logits.detach(), labels, mask * 0)['loss'] == 0
    inference = packed_objective(torch.tensor([[[2.0, -2.0]]]), temperature=2.0)
    assert torch.equal(inference['logits'], torch.tensor([[[1.0, -1.0]]]))


@pytest.mark.parametrize(
    'key,value',
    [
        ('temperature', 0.0),
        ('temperature', float('nan')),
        ('denominator', 0.0),
        ('denominator', float('inf')),
    ],
)
def test_objective_rejects_bad_scalars(key, value):
    with pytest.raises(ValueError):
        packed_objective(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long), **{key: value}
        )


def test_objective_rejects_bad_shapes():
    with pytest.raises(ValueError):
        packed_objective(torch.zeros(1, 2, 3), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError):
        packed_objective(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long), torch.ones(2)
        )


@pytest.mark.parametrize('recompute', [False, True])
def test_fifo_two_unequal_documents_and_recompute(recompute):
    router = RouterReplay()
    cu = torch.tensor([0, 2, 5])
    x = torch.arange(1.0, 6.0).reshape(5, 1).requires_grad_()
    native = torch.tensor([[0], [1], [0], [1], [0]])

    def forward(value):
        indices = router.select_indices((value.long() - 1) % 2)
        return value * (indices + 1)

    router.router_replay_action = Action.RECORD
    recorded = packed_forward(forward, x, cu, axes=0, routers=[router])
    assert torch.equal(router.recorded_topk_idx, native)
    assert torch.equal(recorded, x * (native + 1))
    first, second = 1 - native, torch.ones_like(native)
    mask = torch.tensor([True, False, True, True, False])
    RouterReplay.set_replay_data([first], mask)
    RouterReplay.set_replay_data([second])
    router.router_replay_action = Action.REPLAY_BACKWARD
    out = packed_forward(forward, x, cu, axes=0, routers=[router], recompute=recompute)
    chosen = torch.where(mask[:, None], first, native)
    assert torch.equal(out, x * (chosen + 1))
    assert len(router.replay_backward_list) == 1
    # Recompute must bind the first invocation even after the forward target moves.
    router.target_topk_idx = torch.full_like(native, 99)
    out.sum().backward()
    assert torch.equal(x.grad, (chosen + 1).float())
    assert len(router.replay_backward_list) == 1
    next_out = packed_forward(forward, x.detach(), cu, axes=0, routers=[router])
    assert torch.equal(next_out, x.detach() * 2)
    assert router.replay_backward_list == []
    assert router.router_replay_action == Action.REPLAY_BACKWARD
    assert torch.equal(router.target_topk_idx, torch.full_like(native, 99))


def test_forward_mask_and_exception_restore_without_consuming_fifo():
    router = RouterReplay()
    target = torch.ones(5, 1, dtype=torch.long)
    RouterReplay.set_replay_data([target])
    router.router_replay_action = Action.REPLAY_FORWARD
    value = torch.zeros(5, 1, dtype=torch.long)
    out = packed_forward(
        router.select_indices, value, torch.tensor([0, 2, 5]), axes=0, routers=[router]
    )
    assert torch.equal(out, target)
    assert len(router.replay_backward_list) == 1
    router.router_replay_action = Action.REPLAY_BACKWARD
    saved = dict(vars(router))

    def broken(x):
        router.select_indices(x)
        raise RuntimeError('injected failure')

    with pytest.raises(RuntimeError, match='injected failure'):
        packed_forward(broken, value, torch.tensor([0, 2, 5]), axes=0, routers=[router])
    assert vars(router).keys() == saved.keys()
    for key in saved:
        assert vars(router)[key] is saved[key]
    assert len(router.replay_backward_list) == 1


@pytest.mark.parametrize(
    'action', [Action.RECORD, Action.REPLAY_FORWARD, Action.REPLAY_BACKWARD]
)
def test_missing_router_is_an_error(action):
    router = RouterReplay()
    RouterReplay.set_replay_data([torch.zeros(5, 1, dtype=torch.long)])
    router.router_replay_action = action
    with pytest.raises(RuntimeError, match='router/token'):
        packed_forward(
            lambda x: x,
            torch.zeros(5, 1),
            torch.tensor([0, 2, 5]),
            axes=0,
            routers=[router],
        )


def test_missing_tokens_and_bad_targets_are_errors():
    router = RouterReplay()
    router.router_replay_action = Action.RECORD
    with pytest.raises(RuntimeError, match='router/token'):
        packed_forward(
            lambda x: router.select_indices(x[:1]),
            torch.zeros(5, 1),
            torch.tensor([0, 2, 5]),
            axes=0,
            routers=[router],
        )
    with pytest.raises(ValueError, match='partition'):
        packed_forward(lambda x: x, torch.zeros(5, 1), torch.tensor([0, 2, 4]), axes=0)
    replay = PackedRouterReplay(5, [router])
    with pytest.raises(RuntimeError, match='incomplete'):
        replay.finish()
    RouterReplay.set_replay_data([torch.zeros(4, 1, dtype=torch.long)])
    router.router_replay_action = Action.REPLAY_BACKWARD
    with pytest.raises(ValueError, match='token buffer'):
        PackedRouterReplay(5, [router])
    assert len(router.replay_backward_list) == 1
