# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.lite.model.protocol_utils import pack_thd_forward_kwargs
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.parallel.thd import pack_nested_thd, thd_pack_meta
from megatron.lite.runtime.contracts.data import PackedBatch


def test_thd_sequence_alignment_preserves_true_lengths_and_aligns_physical_layout():
    ids = torch.nested.as_nested_tensor([torch.arange(520)], layout=torch.jagged)

    packed = pack_nested_thd(
        ids,
        cp_size=2,
        split_cp=False,
        sequence_alignment=512,
    )

    assert packed.input_ids.shape == (1, 1024)
    assert packed.lengths.tolist() == [520]
    assert packed.padded_lengths.tolist() == [1024]
    assert packed.packed_seq_params.cu_seqlens_q.tolist() == [0, 520]
    assert packed.packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 1024]


def test_thd_pack_meta_uses_lcm_of_parallel_and_requested_alignment():
    meta = thd_pack_meta(
        torch.tensor([520, 1024], dtype=torch.int32),
        tp_size=3,
        cp_size=2,
        sequence_alignment=512,
    )

    # lcm(tp * 2 * cp, sequence_alignment) = lcm(12, 512) = 1536.
    assert meta.padded_lengths.tolist() == [1536, 1536]
    assert meta.cu_seqlens_padded.tolist() == [0, 1536, 3072]


def test_protocol_packing_keeps_true_offsets_while_cp_splitting_aligned_storage():
    class Model:
        ps = ParallelState(cp_size=2, cp_rank=0)

    batch = PackedBatch(
        input_ids=torch.arange(520, dtype=torch.long),
        labels=None,
        seq_lens=torch.tensor([520], dtype=torch.int32),
    )

    kwargs = pack_thd_forward_kwargs(Model(), batch, sequence_alignment=512)

    assert kwargs["input_ids"].shape == (1, 512)
    params = kwargs["packed_seq_params"]
    assert params.cu_seqlens_q.tolist() == [0, 520]
    assert params.cu_seqlens_q_padded.tolist() == [0, 1024]
    assert params.local_cp_size == 2
