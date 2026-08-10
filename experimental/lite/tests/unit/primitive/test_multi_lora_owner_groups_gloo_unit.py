# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU Gloo contracts for multi-LoRA dist-opt bank owner groups."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.distributed_test_utils import (
    gather_owner_factor_records_or_raise,
    select_lora_bank_owner_group,
)
from megatron.lite.primitive.parallel import init_parallel


def _init_gloo(rank: int, world: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )


def _tp2_ep2_factor_lane_worker(rank: int, world: int, init_file: str) -> None:
    _init_gloo(rank, world, init_file)
    try:
        ps = init_parallel(SimpleNamespace(tp=2, ep=2, etp=1, cp=1, pp=1))
        # Attention and FC banks select their distinct owner groups.
        expected_dp = (0, 2) if rank % 2 == 0 else (1, 3)
        expected_ep = (0, 1) if rank < 2 else (2, 3)
        attention_group = select_lora_bank_owner_group(ps, is_expert_bank=False)
        fc_group = select_lora_bank_owner_group(ps, is_expert_bank=True)
        assert attention_group is ps.dp_group
        assert fc_group is ps.ep_dp_group
        assert tuple(dist.get_process_group_ranks(attention_group)) == expected_dp
        assert tuple(dist.get_process_group_ranks(fc_group)) == expected_dp
        assert tuple(dist.get_process_group_ranks(ps.ep_group)) == expected_ep

        # This mirrors the oracle's fixed collection: gather factor records
        # within the bank's owner lane, never over WORLD then cross-checking
        # records from the other legal TP lane.  Each lane has one optimizer
        # owner and one legal no-shard member.
        def validate_records(values) -> None:
            assert len(values) == 1
            assert values[0]["factor"] == 0.5

        records = gather_owner_factor_records_or_raise(
            attention_group,
            lambda: {"rank": rank, "factor": 0.5} if rank < 2 else None,
            validate_records,
        )
        assert records == [{"rank": min(expected_dp), "factor": 0.5}]
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_tp2_ep2_factor_records_do_not_mix_dp_lanes_gloo(tmp_path):
    torch.multiprocessing.spawn(
        _tp2_ep2_factor_lane_worker,
        args=(4, str(tmp_path / "tp2_ep2_factor_lanes")),
        nprocs=4,
        join=True,
    )


def _malformed_factor_worker(rank: int, world: int, init_file: str) -> None:
    _init_gloo(rank, world, init_file)
    try:
        ps = init_parallel(SimpleNamespace(tp=2, ep=2, etp=1, cp=1, pp=1))
        owner_group = select_lora_bank_owner_group(ps, is_expert_bank=False)

        def build_record():
            if rank == 0:
                raise AssertionError("malformed local factor metadata")
            return {"rank": rank, "factor": 0.5}

        try:
            gather_owner_factor_records_or_raise(
                owner_group, build_record, lambda values: None
            )
        except RuntimeError as error:
            result = str(error)
        else:
            result = None
        expected = (
            "owner-group factor validation failed: RuntimeError: AssertionError: "
            "malformed local factor metadata"
        )
        assert result == expected
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_malformed_owner_factor_record_fails_the_whole_world_without_hanging_gloo(
    tmp_path,
):
    torch.multiprocessing.spawn(
        _malformed_factor_worker,
        args=(4, str(tmp_path / "malformed_factor_lane")),
        nprocs=4,
        join=True,
    )


def _owner_boundary_worker(
    rank: int, world: int, init_file: str, tp: int, ep: int
) -> None:
    _init_gloo(rank, world, init_file)
    try:
        ps = init_parallel(SimpleNamespace(tp=tp, ep=ep, etp=1, cp=1, pp=1))
        attention_group = select_lora_bank_owner_group(ps, is_expert_bank=False)
        fc_group = select_lora_bank_owner_group(ps, is_expert_bank=True)
        assert attention_group is ps.dp_group
        assert fc_group is ps.ep_dp_group
        expected_attention = 1 if (tp, ep) == (2, 1) else 2
        expected_fc = 2 if (tp, ep) == (2, 1) else 1
        assert dist.get_world_size(attention_group) == expected_attention
        assert dist.get_world_size(fc_group) == expected_fc

        def validate_attention(values) -> None:
            assert len(values) == expected_attention

        def validate_fc(values) -> None:
            assert len(values) == expected_fc

        attention_records = gather_owner_factor_records_or_raise(
            attention_group, lambda: {"rank": rank, "factor": 0.5}, validate_attention
        )
        fc_records = gather_owner_factor_records_or_raise(
            fc_group, lambda: {"rank": rank, "factor": 0.5}, validate_fc
        )
        assert all(record["factor"] == 0.5 for record in attention_records)
        assert all(record["factor"] == 0.5 for record in fc_records)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize(("tp", "ep"), [(2, 1), (1, 2)])
def test_attention_and_fc_owner_groups_cover_tp_ep_boundaries_gloo(tmp_path, tp, ep):
    torch.multiprocessing.spawn(
        _owner_boundary_worker,
        args=(2, str(tmp_path / f"tp{tp}_ep{ep}_owner_boundary"), tp, ep),
        nprocs=2,
        join=True,
    )
