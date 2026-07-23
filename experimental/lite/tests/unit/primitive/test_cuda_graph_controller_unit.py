# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit coverage for the chunk-wise CUDA Graph controller spine.

Exercises the pure-CPU architecture: explicit assembly, structural
qualification (enabled/partial/not-applicable), the fixed-capacity THD replay
signature (#4359 contract), the fail-loud fused-RoPE / MoE gates (AC#2), and
the schedule-derived slot plan. The TE ``make_graphed_callables`` capture is
GPU-only and covered by the 8-card proxy smoke, not here.
"""

from __future__ import annotations

import types

import pytest
import torch

from megatron.lite.primitive.cuda_graph import (
    CoverageEntry,
    CudaGraphController,
    CudaGraphDebugMode,
    CudaGraphError,
    ExclusionCode,
    assert_fused_rope_thd,
    build_replay_signature,
    inspect_chunk_moe_dispatch,
)
from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

pytestmark = pytest.mark.mlite


def _packed(token_capacity=16, max_sequences=3, with_max_seqlen=True):
    cu = torch.tensor([0, 5, 9, token_capacity], dtype=torch.int32)  # max_sequences+1 entries
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=8 if with_max_seqlen else None,
        max_seqlen_kv=8 if with_max_seqlen else None,
        total_tokens=token_capacity,
    )


# ---- replay signature (fixed-capacity THD contract, #4359) -------------------


def test_signature_is_derived_from_static_thd_contract():
    hs = torch.zeros(16, 32)
    sig = build_replay_signature(hs, _packed(token_capacity=16, max_sequences=3), cp_size=2)
    assert sig.token_capacity == 16
    assert sig.max_sequences == 3
    assert sig.cp_size == 2
    assert "cu_seqlens_q" in sig.tensor_fields
    assert dict(sig.static_metadata)["max_seqlen_q"] == 8


def test_same_shape_different_values_replay_interchangeable():
    # Values inside the fixed cu_seqlens buffer may change; the signature holds.
    hs = torch.zeros(16, 32)
    p1 = _packed(token_capacity=16, max_sequences=3)
    p2 = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 2, 11, 16], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 2, 11, 16], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 2, 11, 16], dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 2, 11, 16], dtype=torch.int32),
        max_seqlen_q=8,
        max_seqlen_kv=8,
        total_tokens=16,
    )
    s1 = build_replay_signature(hs, p1, cp_size=1)
    s2 = build_replay_signature(hs, p2, cp_size=1)
    assert s1.matches(s2)


def test_changed_token_capacity_breaks_signature():
    s1 = build_replay_signature(torch.zeros(16, 32), _packed(16, 3), cp_size=1)
    s2 = build_replay_signature(torch.zeros(24, 32), _packed(24, 3), cp_size=1)
    assert not s1.matches(s2)


def test_changed_dtype_breaks_signature():
    s1 = build_replay_signature(torch.zeros(16, 32, dtype=torch.bfloat16), _packed(16, 3), cp_size=1)
    s2 = build_replay_signature(torch.zeros(16, 32, dtype=torch.float32), _packed(16, 3), cp_size=1)
    assert not s1.matches(s2)


# ---- fail-loud gates ---------------------------------------------------------


def test_missing_max_seqlen_is_fatal_not_silent():
    hs = torch.zeros(16, 32)
    with pytest.raises(CudaGraphError) as ei:
        build_replay_signature(hs, _packed(with_max_seqlen=False), cp_size=1)
    assert ExclusionCode.UNFUSED_ROPE_HOST_SYNC.value in str(ei.value)


def test_assert_fused_rope_rejects_host_sync_path():
    with pytest.raises(CudaGraphError):
        assert_fused_rope_thd(_packed(with_max_seqlen=False))
    # Present metadata passes silently.
    assert_fused_rope_thd(_packed(with_max_seqlen=True))


def test_dropless_moe_dispatcher_excludes_chunk():
    dynamic = types.SimpleNamespace(is_moe_dispatcher=True, cuda_graph_safe=False)
    chunk = types.SimpleNamespace(modules=lambda: [dynamic])
    reason = inspect_chunk_moe_dispatch(chunk)
    assert reason is not None
    assert reason.code is ExclusionCode.DYNAMIC_MOE_ROUTING


def test_static_capacity_moe_dispatcher_is_graph_safe():
    static = types.SimpleNamespace(is_moe_dispatcher=True, cuda_graph_safe=True)
    chunk = types.SimpleNamespace(modules=lambda: [static])
    assert inspect_chunk_moe_dispatch(chunk) is None


# ---- controller qualification ------------------------------------------------


def _chunk(graph_safe_moe: bool):
    disp = types.SimpleNamespace(is_moe_dispatcher=True, cuda_graph_safe=graph_safe_moe)
    return types.SimpleNamespace(modules=lambda: [disp])


def test_qualify_enabled_when_all_chunks_graph_safe():
    ctrl = CudaGraphController(
        chunks=[_chunk(graph_safe_moe=True)],
        num_warmup_microbatches=0,
        num_microbatches=4,
    )
    status = ctrl.qualify()
    assert status.state == "enabled"
    assert status.implementation == "te_chunk_wise"
    assert status.captured and isinstance(status.captured[0], CoverageEntry)
    assert status.captured[0].region == "transformer_block"
    assert not status.excluded


def test_qualify_not_applicable_when_moe_dynamic_without_attention():
    ctrl = CudaGraphController(
        chunks=[_chunk(graph_safe_moe=False)],
        num_warmup_microbatches=0,
        num_microbatches=4,
    )
    status = ctrl.qualify()
    # Dropless MoE + no discoverable attention region → nothing to capture.
    assert status.state == "not-applicable"
    assert status.excluded[0].code is ExclusionCode.DYNAMIC_MOE_ROUTING
    assert not status.captured


def test_qualify_partial_when_moe_dynamic_with_attention():
    """bayan 2026-07-21 attn-only: MoE excluded, attention still partial-captured."""
    disp = types.SimpleNamespace(is_moe_dispatcher=True, cuda_graph_safe=False)
    attn = object()
    chunk = types.SimpleNamespace(modules=lambda: [disp], full_attn=attn)
    ctrl = CudaGraphController(
        chunks=[chunk],
        num_warmup_microbatches=0,
        num_microbatches=4,
    )
    status = ctrl.qualify()
    assert status.state == "partial"
    assert status.implementation == "te_attn_partial"
    assert status.captured[0].region == "attention"
    assert status.excluded[0].code is ExclusionCode.DYNAMIC_MOE_ROUTING


def test_resolve_attention_callable_walks_layers():
    from megatron.lite.primitive.cuda_graph import resolve_attention_callable

    attn = object()
    layer = types.SimpleNamespace(full_attn=attn)
    chunk = types.SimpleNamespace(layers=[layer])
    assert resolve_attention_callable(chunk) is attn
    assert resolve_attention_callable(types.SimpleNamespace()) is None


def test_qualify_off_is_diagnostic_not_applicable():
    ctrl = CudaGraphController(
        chunks=[_chunk(graph_safe_moe=True)],
        num_warmup_microbatches=0,
        num_microbatches=4,
        debug=CudaGraphDebugMode.OFF,
    )
    status = ctrl.qualify()
    assert status.state == "not-applicable"
    assert status.excluded[0].code is ExclusionCode.DEBUG_DISABLED


def test_pp1_controller_uses_single_slot():
    ctrl = CudaGraphController(
        chunks=[_chunk(graph_safe_moe=True)],
        num_warmup_microbatches=0,  # PP=1
        num_microbatches=8,
    )
    assert ctrl.num_slots == 1
    plan = ctrl.build_slot_plan()
    assert plan.num_slots_per_chunk == (1,)


def test_pp_warmup_controller_needs_multiple_slots():
    # PP=4 first stage: 3 warmup forwards outstanding -> 4 live slots.
    ctrl = CudaGraphController(
        chunks=[_chunk(graph_safe_moe=True)],
        num_warmup_microbatches=3,
        num_microbatches=8,
    )
    assert ctrl.num_slots == 4


def test_vpp_gt_1_is_rejected():
    with pytest.raises(CudaGraphError):
        CudaGraphController(
            chunks=[_chunk(graph_safe_moe=True)],
            num_warmup_microbatches=0,
            num_microbatches=4,
            num_model_chunks=2,
        )


# ---- #4359 decompose / reconstruct round-trip -------------------------------


def test_decompose_reconstruct_round_trip_preserves_contract():
    from megatron.lite.primitive.cuda_graph import (
        decompose_packed_seq_params,
        reconstruct_packed_seq_params,
    )

    src = _packed(token_capacity=16, max_sequences=3)
    tensor_kwargs, static_meta = decompose_packed_seq_params(src)
    # cu_seqlens tensors are threaded as graph inputs; metadata stays static.
    assert "cu_seqlens_q" in tensor_kwargs
    assert static_meta["max_seqlen_q"] == 8
    assert static_meta["qkv_format"] == "thd"

    rebuilt = reconstruct_packed_seq_params(tensor_kwargs, static_meta)
    assert rebuilt.max_seqlen_q == src.max_seqlen_q
    assert rebuilt.max_seqlen_kv == src.max_seqlen_kv
    assert torch.equal(rebuilt.cu_seqlens_q, src.cu_seqlens_q)
    # The reconstructed object replays against the same signature as the source.
    hs = torch.zeros(16, 32)
    assert build_replay_signature(hs, src, cp_size=1).matches(
        build_replay_signature(hs, rebuilt, cp_size=1)
    )
