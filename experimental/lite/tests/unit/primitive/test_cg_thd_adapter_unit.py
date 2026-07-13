# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit tests for the THD static-input adapter (CUDA-graph capture contract).

Covers the #4359 decompose/reconstruct round-trip, the fixed-capacity
cu_seqlens + padding-mask buffers with fail-loud over-capacity, and the
fused-RoPE / GQA-fallback gate (#5672). CPU-only: no capture, no eager numerics.
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.cg_thd_adapter import (
    UNFUSED_ROPE_HOST_SYNC,
    StaticThdInputBuffers,
    ThdRopeSafety,
    ThdStaticInputError,
    allocate_static_thd_buffers,
    assert_fused_rope_thd,
    classify_thd_rope,
    decompose_packed_seq_params,
    reconstruct_packed_seq_params,
    reject_gqa_python_int_fallback,
)
from megatron.lite.primitive.utils.packed_seq import PackedSeqParams


def _psp(
    *,
    cu_padded,
    cu_unpadded=None,
    max_seqlen_q=8,
    max_seqlen_kv=8,
    cp_group="cp-group-handle",
):
    padded = torch.tensor(cu_padded, dtype=torch.int32)
    unpadded = (
        torch.tensor(cu_unpadded, dtype=torch.int32) if cu_unpadded is not None else padded
    )
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=unpadded,
        cu_seqlens_kv=unpadded,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        cp_group=cp_group,
    )


# ---- #4359 decompose / reconstruct ------------------------------------------


def test_decompose_reconstruct_round_trip_preserves_contract():
    src = _psp(cu_padded=[0, 4, 8], cp_group="pg-42")
    tensor_kwargs, static_meta = decompose_packed_seq_params(src)

    # cu_seqlens tensors are threaded as graph inputs.
    assert set(tensor_kwargs) >= {
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
    }
    for value in tensor_kwargs.values():
        assert isinstance(value, torch.Tensor)
    # Static metadata is non-tensor and reconstructed in-callable.
    assert static_meta["max_seqlen_q"] == 8
    assert static_meta["max_seqlen_kv"] == 8
    assert static_meta["qkv_format"] == "thd"
    assert static_meta["cp_group"] == "pg-42"

    rebuilt = reconstruct_packed_seq_params(tensor_kwargs, static_meta)
    assert rebuilt.max_seqlen_q == 8
    assert rebuilt.max_seqlen_kv == 8
    assert rebuilt.cp_group == "pg-42"
    assert torch.equal(rebuilt.cu_seqlens_q_padded, src.cu_seqlens_q_padded)
    assert torch.equal(rebuilt.cu_seqlens_kv, src.cu_seqlens_kv)


def test_reconstruct_takes_no_host_read_of_metadata():
    # Metadata that would otherwise require a host read comes from static_meta.
    src = _psp(cu_padded=[0, 4, 8])
    tensor_kwargs, static_meta = decompose_packed_seq_params(src)
    rebuilt = reconstruct_packed_seq_params(tensor_kwargs, static_meta)
    assert rebuilt.max_seqlen_q is not None
    assert rebuilt.max_seqlen_kv is not None


# ---- fused-RoPE / GQA-fallback gate (#5672) ---------------------------------


def test_assert_fused_rope_rejects_missing_metadata():
    bad = _psp(cu_padded=[0, 4, 8], max_seqlen_q=None, max_seqlen_kv=None)
    with pytest.raises(ThdStaticInputError, match="max_seqlen"):
        assert_fused_rope_thd(bad)


def test_assert_fused_rope_accepts_fused_metadata():
    good = _psp(cu_padded=[0, 4, 8])
    assert_fused_rope_thd(good)  # does not raise


def test_reject_gqa_python_int_fallback_is_the_same_gate():
    assert reject_gqa_python_int_fallback is assert_fused_rope_thd


def test_classify_thd_rope_reports_instead_of_raising():
    # Non-raising classifier so the controller can narrow the boundary and
    # report `partial`, never a silent eager retry.
    unsafe = classify_thd_rope(
        _psp(cu_padded=[0, 4, 8], max_seqlen_q=None, max_seqlen_kv=None)
    )
    assert isinstance(unsafe, ThdRopeSafety)
    assert not unsafe  # __bool__ is False
    assert unsafe.fused_rope_available is False
    assert unsafe.exclusion_reason == UNFUSED_ROPE_HOST_SYNC

    safe = classify_thd_rope(_psp(cu_padded=[0, 4, 8]))
    assert safe  # __bool__ is True
    assert safe.exclusion_reason is None


# ---- fixed-capacity allocation ----------------------------------------------


def test_allocate_static_thd_buffers_shapes_and_dtypes():
    buffers = allocate_static_thd_buffers(token_capacity=16, max_sequences=4)
    assert isinstance(buffers, StaticThdInputBuffers)
    assert buffers.buffer_length == 5
    assert buffers.padding_mask.shape == (16,)
    assert buffers.padding_mask.dtype == torch.bool
    for name in (
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
    ):
        assert buffers.cu_seqlens[name].shape == (5,)
        assert buffers.cu_seqlens[name].dtype == torch.int32


def test_allocate_without_kv_drops_kv_buffers():
    buffers = allocate_static_thd_buffers(
        token_capacity=16, max_sequences=4, include_kv=False
    )
    assert "cu_seqlens_kv" not in buffers.cu_seqlens
    assert "cu_seqlens_q" in buffers.cu_seqlens


@pytest.mark.parametrize("token_capacity,max_sequences", [(0, 4), (16, 0), (-1, 4)])
def test_allocate_rejects_non_positive_capacity(token_capacity, max_sequences):
    with pytest.raises(ThdStaticInputError):
        allocate_static_thd_buffers(
            token_capacity=token_capacity, max_sequences=max_sequences
        )


# ---- copy_in: fixed-address values + padding mask ---------------------------


def test_copy_in_writes_values_and_masks_only_max_align_tail():
    # MLite convention: unpadded == padded, so only the max-alignment tail
    # (tokens >= total_padded) is masked.
    buffers = allocate_static_thd_buffers(token_capacity=12, max_sequences=4)
    q_buf = buffers.cu_seqlens["cu_seqlens_q_padded"]
    addr_before = q_buf.data_ptr()

    buffers.copy_in(_psp(cu_padded=[0, 4, 8]))

    # Address stays fixed (in-place copy), values updated, tail filled.
    assert buffers.cu_seqlens["cu_seqlens_q_padded"].data_ptr() == addr_before
    assert buffers.cu_seqlens["cu_seqlens_q_padded"].tolist() == [0, 4, 8, 8, 8]
    assert buffers.num_real_sequences == 2
    assert buffers.real_tokens == 8
    # First 8 tokens real, last 4 (max-align tail) masked.
    assert buffers.padding_mask.tolist() == [True] * 8 + [False] * 4


def test_copy_in_masks_per_sequence_padding_when_unpadded_known():
    # Distinct unpadded lengths [3, 2] within padded slots [4, 4].
    buffers = allocate_static_thd_buffers(token_capacity=12, max_sequences=4)
    buffers.copy_in(_psp(cu_padded=[0, 4, 8], cu_unpadded=[0, 3, 5]))
    expected = [
        True, True, True, False,   # seq0: 3 real + 1 pad
        True, True, False, False,  # seq1: 2 real + 2 pad
        False, False, False, False,  # max-align tail
    ]
    assert buffers.padding_mask.tolist() == expected
    assert buffers.real_tokens == 8  # padded packed tokens


def test_copy_in_rejects_too_many_sequences():
    buffers = allocate_static_thd_buffers(token_capacity=64, max_sequences=2)
    with pytest.raises(ThdStaticInputError, match="sequences"):
        buffers.copy_in(_psp(cu_padded=[0, 4, 8, 12]))  # 3 sequences > 2


def test_copy_in_rejects_too_many_tokens():
    buffers = allocate_static_thd_buffers(token_capacity=6, max_sequences=4)
    with pytest.raises(ThdStaticInputError, match="token capacity"):
        buffers.copy_in(_psp(cu_padded=[0, 4, 8]))  # 8 tokens > 6


def test_copy_in_rejects_missing_fused_rope_metadata():
    buffers = allocate_static_thd_buffers(token_capacity=12, max_sequences=4)
    with pytest.raises(ThdStaticInputError, match="max_seqlen"):
        buffers.copy_in(_psp(cu_padded=[0, 4, 8], max_seqlen_q=None, max_seqlen_kv=None))


def test_copy_in_is_reusable_across_replays():
    buffers = allocate_static_thd_buffers(token_capacity=12, max_sequences=4)
    buffers.copy_in(_psp(cu_padded=[0, 4, 8]))
    first_ptr = buffers.cu_seqlens["cu_seqlens_q_padded"].data_ptr()
    # A different-length microbatch replays into the same buffers.
    buffers.copy_in(_psp(cu_padded=[0, 4]))
    assert buffers.cu_seqlens["cu_seqlens_q_padded"].data_ptr() == first_ptr
    assert buffers.cu_seqlens["cu_seqlens_q_padded"].tolist() == [0, 4, 4, 4, 4]
    assert buffers.num_real_sequences == 1
    assert buffers.padding_mask.tolist() == [True] * 4 + [False] * 8


# ---- dummy tail sequence -----------------------------------------------------


def test_dummy_tail_sequence_spans_capacity_but_stays_masked():
    buffers = allocate_static_thd_buffers(token_capacity=12, max_sequences=4)
    buffers.copy_in(_psp(cu_padded=[0, 4, 8]), dummy_tail_sequence=True)
    # padded buffer gets one dummy sequence spanning [8, 12).
    assert buffers.cu_seqlens["cu_seqlens_q_padded"].tolist() == [0, 4, 8, 12, 12]
    # unpadded (non-_padded) buffers keep zero-length tail dummies.
    assert buffers.cu_seqlens["cu_seqlens_q"].tolist() == [0, 4, 8, 8, 8]
    # Tail tokens are still excluded from the padding mask (loss/router).
    assert buffers.padding_mask.tolist() == [True] * 8 + [False] * 4


def test_dummy_tail_needs_spare_sequence_slot():
    buffers = allocate_static_thd_buffers(token_capacity=12, max_sequences=2)
    with pytest.raises(ThdStaticInputError, match="sequences"):
        buffers.copy_in(_psp(cu_padded=[0, 4, 8]), dummy_tail_sequence=True)


# ---- to_packed_seq_params ----------------------------------------------------


def test_to_packed_seq_params_points_at_fixed_buffers():
    buffers = allocate_static_thd_buffers(token_capacity=12, max_sequences=4)
    buffers.copy_in(_psp(cu_padded=[0, 4, 8]))
    _, static_meta = decompose_packed_seq_params(_psp(cu_padded=[0, 4, 8]))
    rebuilt = buffers.to_packed_seq_params(static_meta)
    # Reconstructed object shares the persistent buffer storage.
    assert rebuilt.cu_seqlens_q_padded.data_ptr() == (
        buffers.cu_seqlens["cu_seqlens_q_padded"].data_ptr()
    )
    assert rebuilt.max_seqlen_q == 8
