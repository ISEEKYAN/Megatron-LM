# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit coverage for the chunk-wise CUDA graph slot planner.

The FIFO allocator contract is ported from NVIDIA Megatron-LM PR #5258. Capture/replay
order is derived from MLite's own ``pipeline.py`` 1F1B and interleaved VPP schedules,
not from MCore schedule ordering.
"""

from __future__ import annotations

import pytest

from megatron.lite.primitive.parallel.chunk_cuda_graphs import (
    ChunkCudaGraphRuntimeSlots,
    build_chunk_cuda_graph_slot_plan,
    build_chunk_cuda_graph_slot_plan_from_mlite_1f1b,
    build_chunk_cuda_graph_slot_plan_from_mlite_pipeline,
    build_chunk_cuda_graph_slot_plan_from_mlite_vpp,
    build_chunk_cuda_graph_slot_plan_from_schedule,
    build_mlite_1f1b_signed_order,
    build_mlite_vpp_signed_order,
    build_mlite_vpp_signed_order_per_microbatch,
    get_cuda_graph_schedule_stage_order_from_counts,
    get_probe_num_microbatches_for_dynamic_slots,
    get_required_num_microbatch_slots_per_chunk,
    validate_cuda_graph_vpp_layout,
)

pytestmark = pytest.mark.mlite


def test_pp1_needs_single_slot():
    order = build_mlite_1f1b_signed_order(pp_size=1, pp_rank=0, num_microbatches=4)
    assert get_required_num_microbatch_slots_per_chunk(order, num_model_chunks=1) == (1,)
    assert get_probe_num_microbatches_for_dynamic_slots(pipeline_parallel_size=1) == 1


@pytest.mark.parametrize(
    "pp_size,pp_rank,expected_slots",
    [
        (4, 0, 4),
        (4, 1, 3),
        (4, 2, 2),
        (4, 3, 1),
    ],
)
def test_1f1b_slot_count_matches_outstanding_forwards(pp_size, pp_rank, expected_slots):
    order = build_mlite_1f1b_signed_order(pp_size, pp_rank, num_microbatches=8)
    slots = get_required_num_microbatch_slots_per_chunk(order, num_model_chunks=1)
    assert slots == (expected_slots,)


def test_slot_plan_drains_and_reuses_fifo():
    order = build_mlite_1f1b_signed_order(pp_size=4, pp_rank=0, num_microbatches=8)
    plan = build_chunk_cuda_graph_slot_plan(order, num_model_chunks=1)
    assert plan.num_slots_per_chunk == (4,)
    fwd_slots = [s for s, op in zip(plan.slot_ids, plan.op_types) if op == "forward"]
    bwd_slots = [s for s, op in zip(plan.slot_ids, plan.op_types) if op == "backward"]
    assert len(fwd_slots) == len(bwd_slots) == 8
    assert set(fwd_slots) <= {0, 1, 2, 3}
    assert fwd_slots == bwd_slots


def test_mlite_1f1b_planner_matches_signed_order():
    plan = build_chunk_cuda_graph_slot_plan_from_mlite_1f1b(
        pp_size=4, pp_rank=0, num_microbatches=8
    )
    expected_order = build_mlite_1f1b_signed_order(4, 0, 8)
    assert plan.order == expected_order
    assert plan.num_slots_per_chunk == (4,)


def test_runtime_slots_forward_backward_fifo():
    slots = ChunkCudaGraphRuntimeSlots(num_slots=2)
    a = slots.forward(0)
    b = slots.forward(1)
    assert {a, b} == {0, 1}
    released = slots.backward(0)
    assert released == a
    c = slots.forward(2)
    assert c == a


def test_runtime_slots_reject_double_forward_and_orphan_backward():
    slots = ChunkCudaGraphRuntimeSlots(num_slots=1)
    slots.forward(0)
    with pytest.raises(AssertionError):
        slots.forward(0)
    with pytest.raises(AssertionError):
        slots.backward(7)


def test_runtime_slots_exhaustion_is_fail_loud():
    slots = ChunkCudaGraphRuntimeSlots(num_slots=1)
    slots.forward(0)
    with pytest.raises(AssertionError):
        slots.forward(1)


def test_schedule_table_plan_preserves_virtual_microbatch_ids():
    num_model_chunks = 1
    schedule_table = [(mb, 0) for mb in range(6)]
    num_warmup = 1
    plan = build_chunk_cuda_graph_slot_plan_from_schedule(
        num_warmup_microbatches=num_warmup,
        num_model_chunks=num_model_chunks,
        schedule_table=schedule_table,
    )
    assert all(s is not None for s in plan.forward_slot_by_virtual_microbatch)
    assert all(s is not None for s in plan.backward_slot_by_virtual_microbatch)
    for mb in range(6):
        assert plan.get_forward_slot(0, mb) is not None
        assert plan.get_backward_slot(0, mb) is not None


def test_stage_order_counts_partition():
    stages = get_cuda_graph_schedule_stage_order_from_counts(
        num_warmup_microbatches=2, num_scheduled_microbatches=6
    )
    assert stages.count("warmup") == 2
    assert stages.count("cooldown") == 2
    assert stages.count("steady") == (6 - 2) * 2


@pytest.mark.parametrize(
    "pp_size,pp_rank,num_chunks",
    [
        (2, 0, 2),
        (2, 1, 2),
        (4, 2, 3),
    ],
)
def test_vpp_per_microbatch_order_matches_interleaved_schedule(pp_size, pp_rank, num_chunks):
    per_mb = build_mlite_vpp_signed_order_per_microbatch(pp_size, pp_rank, num_chunks)
    assert len(per_mb) == 2 * num_chunks
    assert all(entry > 0 for entry in per_mb[:num_chunks])
    assert all(entry < 0 for entry in per_mb[num_chunks:])
    slots = get_required_num_microbatch_slots_per_chunk(per_mb, num_model_chunks=num_chunks)
    assert all(s >= 1 for s in slots)


def test_vpp_planner_drains_outstanding_for_multiple_microbatches():
    pp_size, pp_rank, num_chunks, num_mb = 2, 0, 2, 4
    order = build_mlite_vpp_signed_order(pp_size, pp_rank, num_chunks, num_mb)
    plan = build_chunk_cuda_graph_slot_plan_from_mlite_vpp(
        pp_size, pp_rank, num_chunks, num_mb
    )
    assert plan.order == order
    assert len(plan.slot_ids) == len(order)
    assert all(op in {"forward", "backward"} for op in plan.op_types)
    assert plan.num_slots_per_chunk == get_required_num_microbatch_slots_per_chunk(
        order, num_model_chunks=num_chunks
    )


def test_pipeline_planner_dispatches_1f1b_and_vpp():
    pp_plan = build_chunk_cuda_graph_slot_plan_from_mlite_pipeline(
        pp_size=4, pp_rank=1, num_microbatches=6, num_model_chunks=1
    )
    assert pp_plan.num_slots_per_chunk == (3,)

    vpp_plan = build_chunk_cuda_graph_slot_plan_from_mlite_pipeline(
        pp_size=2, pp_rank=0, num_microbatches=3, num_model_chunks=2
    )
    assert len(vpp_plan.num_slots_per_chunk) == 2


def test_vpp_layout_boundary_reports_not_implemented():
    with pytest.raises(NotImplementedError, match="VPP / interleaved pipeline layout"):
        validate_cuda_graph_vpp_layout(vpp=2)
    with pytest.raises(NotImplementedError, match="VPP / interleaved pipeline layout"):
        validate_cuda_graph_vpp_layout(vpp_chunk_id=1)
    with pytest.raises(NotImplementedError, match="VPP / interleaved pipeline layout"):
        build_chunk_cuda_graph_slot_plan_from_mlite_pipeline(
            pp_size=2,
            pp_rank=0,
            num_microbatches=4,
            num_model_chunks=1,
            vpp_layout=2,
        )
