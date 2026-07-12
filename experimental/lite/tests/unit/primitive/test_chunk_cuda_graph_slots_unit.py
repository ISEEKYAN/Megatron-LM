# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit coverage for the chunk-wise CUDA graph slot planner.

The planner is ported verbatim from NVIDIA Megatron-LM PR #5258
(``chunk_cuda_graphs.py``). These tests pin the scheduling contract that the
TE-backed capture/replay controller relies on: a per-chunk FIFO of live
microbatch slots derived from a real 1F1B/VPP forward/backward order, so that
``(chunk, slot)`` keys index the captured graph pairs deterministically.
"""

from __future__ import annotations

import pytest

from megatron.lite.primitive.parallel.chunk_cuda_graphs import (
    ChunkCudaGraphRuntimeSlots,
    build_chunk_cuda_graph_slot_plan,
    build_chunk_cuda_graph_slot_plan_from_schedule,
    get_cuda_graph_schedule_stage_order_from_counts,
    get_probe_num_microbatches_for_dynamic_slots,
    get_required_num_microbatch_slots_per_chunk,
)

pytestmark = pytest.mark.mlite


def _one_f_one_b_order(pp_size: int, pp_rank: int, num_microbatches: int):
    """Reconstruct the signed forward(+1)/backward(-1) order MLite's 1F1B runs.

    Mirrors ``primitive/parallel/pipeline.py::_1f1b_schedule`` slot lifetime:
    ``num_warmup`` outstanding forwards, then interleaved 1F1B, then cooldown
    backwards. Chunk id is always 1 (single local chunk per PP rank; VPP>1 is
    unsupported in MLite).
    """
    num_warmup = min(pp_size - pp_rank - 1, num_microbatches)
    num_steady = num_microbatches - num_warmup
    order = [1] * num_warmup
    for _ in range(num_steady):
        order.append(1)
        order.append(-1)
    order.extend([-1] * num_warmup)
    return order


def test_pp1_needs_single_slot():
    # PP=1: forward then backward per microbatch, never two outstanding.
    order = _one_f_one_b_order(pp_size=1, pp_rank=0, num_microbatches=4)
    assert get_required_num_microbatch_slots_per_chunk(order, num_model_chunks=1) == (1,)
    assert get_probe_num_microbatches_for_dynamic_slots(pipeline_parallel_size=1) == 1


@pytest.mark.parametrize(
    "pp_size,pp_rank,expected_slots",
    [
        (4, 0, 4),  # first stage holds the most outstanding forwards
        (4, 1, 3),
        (4, 2, 2),
        (4, 3, 1),  # last stage is pure 1F1B
    ],
)
def test_1f1b_slot_count_matches_outstanding_forwards(pp_size, pp_rank, expected_slots):
    order = _one_f_one_b_order(pp_size, pp_rank, num_microbatches=8)
    slots = get_required_num_microbatch_slots_per_chunk(order, num_model_chunks=1)
    assert slots == (expected_slots,)


def test_slot_plan_drains_and_reuses_fifo():
    order = _one_f_one_b_order(pp_size=4, pp_rank=0, num_microbatches=8)
    plan = build_chunk_cuda_graph_slot_plan(order, num_model_chunks=1)
    # Every forward entry got a slot; every backward released one; nothing leaked.
    assert plan.num_slots_per_chunk == (4,)
    fwd_slots = [s for s, op in zip(plan.slot_ids, plan.op_types) if op == "forward"]
    bwd_slots = [s for s, op in zip(plan.slot_ids, plan.op_types) if op == "backward"]
    assert len(fwd_slots) == len(bwd_slots) == 8
    assert set(fwd_slots) <= {0, 1, 2, 3}
    # FIFO: the k-th backward releases the slot the k-th forward reserved.
    assert fwd_slots == bwd_slots


def test_runtime_slots_forward_backward_fifo():
    slots = ChunkCudaGraphRuntimeSlots(num_slots=2)
    a = slots.forward(0)
    b = slots.forward(1)
    assert {a, b} == {0, 1}  # both slots now live
    # backward of mb0 releases exactly mb0's slot back to the free pool.
    released = slots.backward(0)
    assert released == a
    # Next forward must reuse the only free slot (the one just released).
    c = slots.forward(2)
    assert c == a


def test_runtime_slots_reject_double_forward_and_orphan_backward():
    slots = ChunkCudaGraphRuntimeSlots(num_slots=1)
    slots.forward(0)
    with pytest.raises(AssertionError):
        slots.forward(0)  # forward twice before backward
    with pytest.raises(AssertionError):
        slots.backward(7)  # backward for a microbatch that never ran forward


def test_runtime_slots_exhaustion_is_fail_loud():
    slots = ChunkCudaGraphRuntimeSlots(num_slots=1)
    slots.forward(0)
    with pytest.raises(AssertionError):
        slots.forward(1)  # no free slot -> explicit error, never silent


def test_schedule_table_plan_preserves_virtual_microbatch_ids():
    # PP=2 style: two warmup forwards, schedule table of (microbatch_id, chunk_id).
    num_model_chunks = 1
    schedule_table = [(mb, 0) for mb in range(6)]
    num_warmup = 1
    plan = build_chunk_cuda_graph_slot_plan_from_schedule(
        num_warmup_microbatches=num_warmup,
        num_model_chunks=num_model_chunks,
        schedule_table=schedule_table,
    )
    # Each virtual microbatch has a forward and a backward slot recorded.
    assert all(s is not None for s in plan.forward_slot_by_virtual_microbatch)
    assert all(s is not None for s in plan.backward_slot_by_virtual_microbatch)
    # Lookup helpers agree with the recorded per-(chunk, microbatch) tables.
    for mb in range(6):
        assert plan.get_forward_slot(0, mb) is not None
        assert plan.get_backward_slot(0, mb) is not None


def test_stage_order_counts_partition():
    # 2 warmup, 6 scheduled -> 2 warmup, 4*2 steady halves, 2 cooldown.
    stages = get_cuda_graph_schedule_stage_order_from_counts(
        num_warmup_microbatches=2, num_scheduled_microbatches=6
    )
    assert stages.count("warmup") == 2
    assert stages.count("cooldown") == 2
    assert stages.count("steady") == (6 - 2) * 2
