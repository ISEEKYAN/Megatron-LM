# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU Gloo contracts for multi-LoRA dist-opt bank owner groups."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import lora_distributed_test_utils as lora_dist_utils
import pytest
import torch
import torch.distributed as dist
from megatron.lite.primitive.modules.multi_lora_bank import MultiLoraTrainingState
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
        attention_group = lora_dist_utils.select_lora_bank_owner_group(
            ps, is_expert_bank=False
        )
        fc_group = lora_dist_utils.select_lora_bank_owner_group(ps, is_expert_bank=True)
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

        records = lora_dist_utils.gather_owner_factor_records_or_raise(
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
        owner_group = lora_dist_utils.select_lora_bank_owner_group(
            ps, is_expert_bank=False
        )

        def build_record():
            if rank == 0:
                raise AssertionError("malformed local factor metadata")
            return {"rank": rank, "factor": 0.5}

        try:
            lora_dist_utils.gather_owner_factor_records_or_raise(
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
        attention_group = lora_dist_utils.select_lora_bank_owner_group(
            ps, is_expert_bank=False
        )
        fc_group = lora_dist_utils.select_lora_bank_owner_group(ps, is_expert_bank=True)
        assert attention_group is ps.dp_group
        assert fc_group is ps.ep_dp_group
        expected_attention = 1 if (tp, ep) == (2, 1) else 2
        expected_fc = 2 if (tp, ep) == (2, 1) else 1
        assert dist.get_world_size(attention_group) == expected_attention
        assert dist.get_world_size(fc_group) == expected_fc
        if (tp, ep) == (2, 1):
            fc_parameter = torch.nn.Parameter(torch.ones(1))
            fc_parameter.allreduce = False
            attention_parameter = torch.nn.Parameter(torch.ones(1))
            attention_parameter.allreduce = True
            fc_name = MultiLoraTrainingState.parameter_name(
                "layers.0.moe.experts._fc1_weight_0", "a"
            )
            assert fc_name.startswith("bank_")
            assert (
                lora_dist_utils.select_lora_bank_owner_group(
                    ps, is_expert_bank=not fc_parameter.allreduce
                )
                is ps.ep_dp_group
            )
            assert (
                lora_dist_utils.select_lora_bank_owner_group(
                    ps, is_expert_bank=not attention_parameter.allreduce
                )
                is ps.dp_group
            )

        def validate_attention(values) -> None:
            assert len(values) == expected_attention

        def validate_fc(values) -> None:
            assert len(values) == expected_fc

        attention_records = lora_dist_utils.gather_owner_factor_records_or_raise(
            attention_group, lambda: {"rank": rank, "factor": 0.5}, validate_attention
        )
        fc_records = lora_dist_utils.gather_owner_factor_records_or_raise(
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


def test_canonical_lora_bank_names_stabilize_rank_local_insertion_order():
    """Every rank must enter mixed FC/attention collectives in one order."""
    rank_zero_order = {
        "layers.0.attn.qkv.linear.weight": object(),
        "layers.0.mlp.experts.linear_fc1.weight": object(),
        "layers.0.attn.proj.linear.weight": object(),
    }
    rank_two_order = {
        "layers.0.mlp.experts.linear_fc1.weight": object(),
        "layers.0.attn.proj.linear.weight": object(),
        "layers.0.attn.qkv.linear.weight": object(),
    }
    expected = (
        "layers.0.attn.proj.linear.weight",
        "layers.0.attn.qkv.linear.weight",
        "layers.0.mlp.experts.linear_fc1.weight",
    )
    assert lora_dist_utils.canonical_lora_bank_names(rank_zero_order) == expected
    assert lora_dist_utils.canonical_lora_bank_names(rank_two_order) == expected


def _fixed_router_ep_coverage_worker(rank: int, world: int, init_file: str) -> None:
    _init_gloo(rank, world, init_file)
    try:
        smoke_path = (
            Path(__file__).parents[2]
            / "smoke/primitive/test_multi_lora_ep2_production_gpu.py"
        )
        spec = importlib.util.spec_from_file_location(
            "multi_lora_ep2_smoke", smoke_path
        )
        assert spec is not None and spec.loader is not None
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
        flight_dir = Path(init_file).parent / "phase_a_flight_records"
        flight_dir.mkdir(exist_ok=True)
        smoke._write_phase_a_flight_record(flight_dir, export_complete=True)
        smoke._write_phase_a_flight_record(flight_dir, reference_enter=True)
        smoke._write_phase_a_flight_record(flight_dir, reference_done=True)
        smoke._write_phase_a_flight_record(flight_dir, preflight_done=True)
        smoke._write_phase_a_flight_record(flight_dir, oracle_done=True)
        flight_record = smoke._write_phase_a_flight_record(
            flight_dir, phase_a_complete=True
        )
        assert flight_record == {
            "export_complete": True,
            "oracle_done": True,
            "phase_a_complete": True,
            "preflight_done": True,
            "reference_done": True,
            "reference_enter": True,
        }
        assert (flight_dir / f"phase_a_rank_{rank:05d}.json").is_file()
        try:
            smoke._run_phase_a_stage(
                flight_dir,
                "reference",
                lambda: (_ for _ in ()).throw(AssertionError("reference failure")),
            )
        except AssertionError:
            pass
        failed_record = json.loads(
            (flight_dir / f"phase_a_rank_{rank:05d}.json").read_text()
        )
        assert failed_record["current_stage"] == "reference"
        assert failed_record["stage_error"] == "AssertionError: reference failure"
        assert "reference failure" in failed_record["traceback"]
        smoke._run_phase_a_stage(flight_dir, "reference", lambda: None)
        repaired_record = json.loads(
            (flight_dir / f"phase_a_rank_{rank:05d}.json").read_text()
        )
        assert "stage_error" not in repaired_record
        assert "traceback" not in repaired_record
        assert repaired_record["reference_done"] is True
        dist.barrier(group=dist.group.WORLD)
        assert not list(flight_dir.glob(".phase_a_rank_*.tmp"))
        rows = torch.zeros(2, 8)
        _scores, production_experts = smoke._fixed_local_router(rows)
        _scores, phase_a_experts = smoke._phase_a_balanced_router(rows)
        production_routes = tuple(int(value) for value in production_experts.flatten())
        phase_a_routes = tuple(int(value) for value in phase_a_experts.flatten())
        local_experts = (0, 1) if rank % 2 == 0 else (2, 3)
        record = {
            "rank": rank,
            "production_routes": production_routes,
            "production_receives_tokens": any(
                expert in local_experts for expert in production_routes
            ),
            "phase_a_routes": phase_a_routes,
            "phase_a_receives_tokens": any(
                expert in local_experts for expert in phase_a_routes
            ),
        }
        records = [None] * world
        dist.all_gather_object(records, record, group=dist.group.WORLD)
        assert all(item["production_routes"] == (0, 0) for item in records)
        assert [item["production_receives_tokens"] for item in records] == [
            True,
            False,
            True,
            False,
        ]
        assert all(item["phase_a_routes"] == (0, 2) for item in records)
        assert all(item["phase_a_receives_tokens"] for item in records)
        dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_production_and_phase_a_routers_cover_their_tp2_ep2_contracts_gloo(tmp_path):
    torch.multiprocessing.spawn(
        _fixed_router_ep_coverage_worker,
        args=(4, str(tmp_path / "fixed_router_ep_coverage")),
        nprocs=4,
        join=True,
    )


def _assert_two_owner_records(values) -> None:
    assert len(values) == 2
    assert all(value["factor"] == 0.5 for value in values)


def _mixed_bank_collective_order_worker(rank: int, world: int, init_file: str) -> None:
    _init_gloo(rank, world, init_file)
    try:
        ps = init_parallel(SimpleNamespace(tp=2, ep=2, etp=1, cp=1, pp=1))
        ordered = (
            "layers.0.mlp.experts.linear_fc1.weight",
            "layers.0.attn.qkv.linear.weight",
        )
        local_banks = {
            name: torch.tensor(float(rank + 1))
            for name in (ordered if rank % 2 == 0 else tuple(reversed(ordered)))
        }
        kinds = {ordered[0]: "fc", ordered[1]: "attention"}
        records = lora_dist_utils.preflight_lora_bank_collective_order(
            local_banks,
            lambda name: (
                kinds[name],
                tuple(local_banks[name].shape),
                str(local_banks[name].dtype),
            ),
        )
        assert records == (
            (ordered[0], "fc", (), "torch.float32"),
            (ordered[1], "attention", (), "torch.float32"),
        )

        actual = {}
        for name, _kind, _shape, _dtype in records:
            value = local_banks[name].clone()
            if kinds[name] == "fc":
                dist.all_reduce(value, group=ps.ep_group)
                dist.all_reduce(value, group=ps.ep_dp_group)
                assert value.item() == 10.0
            else:
                dist.all_reduce(value, group=ps.dp_group)
                assert value.item() == (4.0 if rank % 2 == 0 else 6.0)
            actual[name] = value.item()
        gathered = [None] * world
        dist.all_gather_object(gathered, actual, group=dist.group.WORLD)
        assert all(record[ordered[0]] == 10.0 for record in gathered)
        assert [record[ordered[1]] for record in gathered] == [4.0, 6.0, 4.0, 6.0]
        lora_dist_utils.gather_owner_factor_records_or_raise(
            ps.dp_group,
            lambda: {"rank": rank, "factor": 0.5},
            _assert_two_owner_records,
        )
        dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_mixed_fc_attention_collectives_have_canonical_world4_order_gloo(tmp_path):
    torch.multiprocessing.spawn(
        _mixed_bank_collective_order_worker,
        args=(4, str(tmp_path / "mixed_fc_attention_collective_order")),
        nprocs=4,
        join=True,
    )


def _mixed_bank_preflight_mismatch_worker(
    rank: int, world: int, init_file: str
) -> None:
    _init_gloo(rank, world, init_file)
    try:
        banks = {"layers.0.mlp.experts.linear_fc1.weight": torch.ones(2, 3)}

        def describe_bank(name):
            assert name in banks
            return (
                "attention" if rank == 0 else "fc",
                tuple(banks[name].shape),
                str(banks[name].dtype),
            )

        try:
            lora_dist_utils.preflight_lora_bank_collective_order(banks, describe_bank)
        except RuntimeError as error:
            result = str(error)
        else:
            result = None
        results = [None] * world
        dist.all_gather_object(results, result, group=dist.group.WORLD)
        assert (
            result
            == "mixed LoRA bank collective preflight failed: records differ across WORLD"
        )
        assert results == [result] * world
        dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_mixed_bank_preflight_mismatch_fails_whole_world_without_hanging_gloo(tmp_path):
    torch.multiprocessing.spawn(
        _mixed_bank_preflight_mismatch_worker,
        args=(4, str(tmp_path / "mixed_bank_preflight_mismatch")),
        nprocs=4,
        join=True,
    )


def _invalid_descriptor_worker(rank: int, world: int, init_file: str) -> None:
    _init_gloo(rank, world, init_file)
    try:
        banks = {"layers.0.attn.qkv.linear.weight": torch.ones(2, 3)}
        try:
            lora_dist_utils.preflight_lora_bank_collective_order(
                banks, lambda _name: ("invalid", (2, 3), "torch.float32")
            )
        except RuntimeError as error:
            result = str(error)
        else:
            result = None
        results = [None] * world
        dist.all_gather_object(results, result, group=dist.group.WORLD)
        expected = (
            "mixed LoRA bank collective preflight failed: local record error: "
            "AssertionError: invalid LoRA bank kind: invalid"
        )
        assert results == [expected] * world
        dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_invalid_preflight_descriptor_fails_whole_world_without_hanging_gloo(tmp_path):
    torch.multiprocessing.spawn(
        _invalid_descriptor_worker,
        args=(4, str(tmp_path / "invalid_preflight_descriptor")),
        nprocs=4,
        join=True,
    )


def _descriptor_builder_error_worker(rank: int, world: int, init_file: str) -> None:
    _init_gloo(rank, world, init_file)
    try:
        banks = {"layers.0.attn.qkv.linear.weight": torch.ones(2, 3)}

        def build_descriptor(_name):
            if rank == 0:
                raise AssertionError("malformed descriptor")
            return ("attention", (2, 3), "torch.float32")

        try:
            lora_dist_utils.preflight_lora_bank_collective_order(
                banks, build_descriptor
            )
        except RuntimeError as error:
            result = str(error)
        else:
            result = None
        results = [None] * world
        dist.all_gather_object(results, result, group=dist.group.WORLD)
        expected = (
            "mixed LoRA bank collective preflight failed: local record error: "
            "AssertionError: malformed descriptor"
        )
        assert results == [expected] * world
        dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_descriptor_builder_error_fails_whole_world_without_hanging_gloo(tmp_path):
    torch.multiprocessing.spawn(
        _descriptor_builder_error_worker,
        args=(4, str(tmp_path / "descriptor_builder_error")),
        nprocs=4,
        join=True,
    )


def _fc_bank_dtype_drift_worker(rank: int, world: int, init_file: str) -> None:
    _init_gloo(rank, world, init_file)
    try:
        banks = {"layers.0.mlp.experts.linear_fc1.weight": torch.ones(2, 3)}

        def build_descriptor(_name):
            bank_dtype = torch.float32 if rank == 0 else torch.bfloat16
            return lora_dist_utils.build_lora_collective_descriptor(
                "fc", banks[_name], bank_dtype
            )

        try:
            lora_dist_utils.preflight_lora_bank_collective_order(
                banks, build_descriptor
            )
        except RuntimeError as error:
            result = str(error)
        else:
            result = None
        results = [None] * world
        dist.all_gather_object(results, result, group=dist.group.WORLD)
        expected = (
            "mixed LoRA bank collective preflight failed: local record error: "
            "AssertionError: FC LoRA bank dtype must be torch.bfloat16"
        )
        assert results == [expected] * world
        dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_fc_bank_dtype_drift_fails_whole_world_in_preflight_gloo(tmp_path):
    torch.multiprocessing.spawn(
        _fc_bank_dtype_drift_worker,
        args=(4, str(tmp_path / "fc_bank_dtype_drift")),
        nprocs=4,
        join=True,
    )


def _fc_contribution_dtype_drift_worker(rank: int, world: int, init_file: str) -> None:
    _init_gloo(rank, world, init_file)
    try:
        banks = {"layers.0.mlp.experts.linear_fc1.weight": torch.ones(2, 3)}

        def build_descriptor(_name):
            reduction_tensor = banks[_name].double() if rank == 0 else banks[_name]
            return lora_dist_utils.build_lora_collective_descriptor(
                "fc", reduction_tensor, torch.bfloat16
            )

        try:
            lora_dist_utils.preflight_lora_bank_collective_order(
                banks, build_descriptor
            )
        except RuntimeError as error:
            result = str(error)
        else:
            result = None
        results = [None] * world
        dist.all_gather_object(results, result, group=dist.group.WORLD)
        expected = (
            "mixed LoRA bank collective preflight failed: local record error: "
            "AssertionError: LoRA reduction tensor dtype must be torch.float32"
        )
        assert results == [expected] * world
        dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_fc_contribution_dtype_drift_fails_whole_world_in_preflight_gloo(tmp_path):
    torch.multiprocessing.spawn(
        _fc_contribution_dtype_drift_worker,
        args=(4, str(tmp_path / "fc_contribution_dtype_drift")),
        nprocs=4,
        join=True,
    )
