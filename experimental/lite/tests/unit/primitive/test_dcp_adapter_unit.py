# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from megatron.lite.primitive.parallel import dcp as dcp_adapter
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.loss import LossContext

pytestmark = pytest.mark.mlite


class _Group:
    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _Scheduler:
    kwargs = None

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs

    def get_groups_and_subsamples(self, sample_id_seqlens):
        assert sample_id_seqlens == [(0, 2), (1, 3), (2, 1), (3, 4)]
        return [[[0], [0], [1], [1]], [[2], [2], [3], [3]]]


def _batch(tokens: list[int], seq_lens: list[int], source: list[int]):
    tensor = torch.tensor(tokens)
    return (
        PackedBatch(
            input_ids=tensor,
            labels=tensor.clone(),
            seq_lens=torch.tensor(seq_lens),
            loss_mask=torch.ones_like(tensor, dtype=torch.float32),
        ),
        LossContext(temperature=2.0, source_batch=torch.tensor(source)),
    )


def test_schedule_uses_mcore_policy_and_preserves_sample_identity(monkeypatch):
    def _gather(output, local, group):
        assert group.size() == 4
        output[:] = [local, [], [1, 3], []]

    monkeypatch.setattr(torch.distributed, "all_gather_object", _gather)
    scheduled, count, batch_size = dcp_adapter.schedule(
        iter(
            [
                _batch(
                    [10, 11, 20, 21, 22, 30, 40, 41, 42, 43],
                    [2, 3, 1, 4],
                    [100, 200, 300, 400],
                )
            ]
        ),
        num_microbatches=1,
        dp_size=2,
        cp_size=2,
        dcp_group=_Group(4),
        max_seqlen_per_dp_cp_rank=4,
        scheduler_cls=_Scheduler,
    )

    assert count == 2
    assert batch_size == 4
    assert _Scheduler.kwargs == {
        "max_seqlen_per_dp_cp_rank": 4,
        "cp_size": 2,
        "dp_size": 2,
        "microbatch_group_size_per_vp_stage": None,
        "min_cp_size": 1,
    }
    assert [batch.input_ids.tolist() for batch, _ in scheduled] == [[10, 11], [30]]
    assert [context.source_batch.tolist() for _, context in scheduled] == [[100], [300]]
    assert all(context.temperature == 2.0 for _, context in scheduled)
    assert [batch.extras["_mlite_dcp_local_cp_size"] for batch, _ in scheduled] == [
        2,
        2,
    ]


def test_schedule_does_not_accept_router_replay_payload(monkeypatch):
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda *args, **kwargs: None
    )
    batch, context = _batch([1], [1], [1])
    batch.r3_replay_mask = torch.ones(1, dtype=torch.bool)

    with pytest.raises(NotImplementedError, match="router replay"):
        dcp_adapter.schedule(
            iter([(batch, context)]),
            num_microbatches=1,
            dp_size=1,
            cp_size=2,
            dcp_group=_Group(2),
            max_seqlen_per_dp_cp_rank=1,
            scheduler_cls=_Scheduler,
        )


def test_schedule_accepts_even_non_power_of_two_topology(monkeypatch):
    class _SixRankScheduler:
        def __init__(self, **_kwargs):
            pass

        def get_groups_and_subsamples(self, sample_id_seqlens):
            assert sample_id_seqlens == [(idx, 1) for idx in range(6)]
            return [[[idx] for idx in range(6)]]

    def _gather(output, local, group):
        assert group.size() == 6
        output[:] = [[idx] for idx in range(6)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", _gather)
    scheduled, count, batch_size = dcp_adapter.schedule(
        iter([_batch(list(range(6)), [1] * 6, list(range(6)))]),
        num_microbatches=1,
        dp_size=3,
        cp_size=2,
        dcp_group=_Group(6),
        max_seqlen_per_dp_cp_rank=1,
        scheduler_cls=_SixRankScheduler,
    )

    assert count == 1
    assert batch_size == 6
    assert scheduled[0][0].input_ids.tolist() == [0]


def test_bind_group_restores_static_parallel_state(monkeypatch):
    dynamic_group = _Group(2, rank=1)
    static_group = object()
    state = SimpleNamespace(cp_size=4, cp_rank=3, cp_group=static_group)

    with dcp_adapter.bind_group(state, {2: dynamic_group}, 2):
        assert (state.cp_size, state.cp_rank, state.cp_group) == (2, 1, dynamic_group)

    assert (state.cp_size, state.cp_rank, state.cp_group) == (4, 3, static_group)


def test_initialize_groups_uses_global_collective_order(monkeypatch):
    from megatron.core import parallel_state as mpu

    calls = []
    subgroup = _Group(2)

    def _create(rank, ranks, pg_options, min_cp_size):
        calls.append((rank, ranks, pg_options, min_cp_size))
        return {2: subgroup} if rank in ranks else {}

    monkeypatch.setattr(mpu, "create_dynamic_dp_cp_groups", _create, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    full_group = _Group(4)
    groups = dcp_adapter.initialize_groups(
        SimpleNamespace(dp_size=2, dp_cp_group=full_group),
        SimpleNamespace(tp=2, cp=2, pp=1, min_dynamic_context_parallel_size=1),
    )

    assert [call[1] for call in calls] == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert groups == {2: subgroup, 4: full_group}


def test_prepare_runtime_binds_forward_and_loss_then_restores(monkeypatch):
    from megatron.lite.primitive.parallel import dcp as dcp_module

    dynamic_group = _Group(2, rank=1)
    static_group = object()
    state = SimpleNamespace(
        dp_size=1, cp_size=4, cp_rank=3, cp_group=static_group, dp_cp_group=_Group(4)
    )
    batch, context = _batch([1], [1], [1])
    batch.extras.update(
        {
            "_mlite_dcp_local_cp_size": 2,
            "_mlite_dcp_sample_ids": [0],
            "_mlite_dcp_group_leader": True,
        }
    )
    monkeypatch.setattr(
        dcp_module, "schedule", lambda *_args, **_kwargs: ([(batch, context)], 1, 1)
    )
    seen = []
    _, _, forward, loss, _finish, microbatch_context, pre_forward_scale = (
        dcp_adapter.prepare_runtime(
            iter([]),
            num_microbatches=1,
            input_num_microbatches=1,
            parallel_state=state,
            config=SimpleNamespace(
                max_seqlen_per_dp_cp_rank=1, min_dynamic_context_parallel_size=1
            ),
            groups={2: dynamic_group},
            forward_step=lambda _model, _batch: (
                seen.append(("forward", state.cp_size, state.cp_rank))
                or {"log_probs": torch.tensor([1.0])}
            ),
            loss_fn=lambda output, _batch, _context: (
                seen.append(("loss", state.cp_size, state.cp_rank))
                or (output["log_probs"].sum(), {})
            ),
        )
    )

    with microbatch_context(batch):
        assert pre_forward_scale(batch) == 2
        output = forward(None, batch)
        loss(output, batch, context)

    assert seen == [("forward", 2, 1), ("loss", 2, 1)]
    assert (state.cp_size, state.cp_rank, state.cp_group) == (4, 3, static_group)


def test_collect_records_restores_original_sample_order(monkeypatch):
    rank_two = [
        {
            "sample_ids": [1, 3],
            "model_output": {"log_probs": [torch.tensor([20.0]), torch.tensor([40.0])]},
            "loss": torch.tensor(3.0),
            "metrics": {"tokens": 2},
        }
    ]

    def _gather(output, local, group):
        output[:] = [local, [], rank_two, []]

    monkeypatch.setattr(torch.distributed, "all_gather_object", _gather)
    records = dcp_adapter.collect_records(
        [
            {
                "_mlite_dcp_sample_ids": [0, 2],
                "_mlite_dcp_group_leader": True,
                "model_output": {
                    "log_probs": torch.nested.as_nested_tensor(
                        [torch.tensor([10.0]), torch.tensor([30.0])],
                        layout=torch.jagged,
                    )
                },
                "loss": torch.tensor(1.0),
                "metrics": {"tokens": 2},
            }
        ],
        batch_size=4,
        dcp_group=_Group(4),
    )

    assert len(records) == 2
    assert [row.item() for row in records[0]["model_output"]["log_probs"].unbind()] == [
        10.0,
        20.0,
        30.0,
        40.0,
    ]
    assert [record["loss"].item() for record in records] == [1.0, 3.0]
    assert [record["metrics"] for record in records] == [{"tokens": 2}, {"tokens": 2}]


@pytest.mark.parametrize("minimum", [0, 3])
def test_schedule_rejects_invalid_minimum_cp_size(minimum):
    with pytest.raises(ValueError, match="positive power of two"):
        dcp_adapter.schedule(
            iter([]),
            num_microbatches=1,
            dp_size=1,
            cp_size=4,
            dcp_group=_Group(4),
            max_seqlen_per_dp_cp_rank=8,
            min_cp_size=minimum,
            scheduler_cls=_Scheduler,
        )
