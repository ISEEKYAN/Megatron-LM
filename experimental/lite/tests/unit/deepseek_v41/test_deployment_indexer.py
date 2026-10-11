# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FP32 index ranking matches native original packets and padded batches."""
import os

import pytest
import torch


def test_indexer_provider_cpu_fails_without_initializing_cuda(v41_core_te):
    from megatron.lite.primitive.modules import deployment_math as dm

    before = torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match="requires CUDA"):
        dm.indexer_scores(
            torch.zeros(1, 1, 32, 128),
            torch.zeros(1, 514, 128),
            torch.ones(1, 1, 32),
            torch.ones(1, 1, 1, dtype=torch.int32),
        )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        dm.indexer_select_positions(
            torch.zeros(1, 1, 514), torch.ones(1, 1, 1, dtype=torch.int32), 512
        )
    assert torch.cuda.is_initialized() == before


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("batch", [1, 2])
def test_native_indexer_original_packet_rounding_and_batch_padding(v41_core_te, batch):
    if os.environ.get("MEGATRON_LITE_REQUIRE_CUDA_TESTS") == "1":
        assert torch.cuda.is_available(), "Index score evidence requires CUDA"
    elif not torch.cuda.is_available():
        pytest.skip("CUDA native index score")
    from megatron.lite.primitive.modules import deployment_math as dm
    from megatron.lite.primitive.quantization.mxfp4 import quantize_index
    from vllm import _custom_ops as ops
    from vllm.model_executor.determinism.batch_invariant import (
        enable_batch_invariant_mode,
    )
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    enable_batch_invariant_mode()
    torch.manual_seed(1031)
    raw_q = torch.randn(batch, 7, 32, 128, device="cuda", dtype=torch.bfloat16)
    raw_k = torch.randn(batch, 640, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch, 7, 32, device="cuda", dtype=torch.bfloat16) / 64
    lengths = torch.tensor(
        [0, 1, 511, 512, 513, 514, 640], device="cuda", dtype=torch.int32
    ).reshape(7, 1)
    q, k = quantize_index(raw_q), quantize_index(raw_k)
    actual = dm.indexer_scores(q.decoded, k.decoded, weights, lengths)
    expected = []
    for index in range(batch):
        original = fp8_fp4_mqa_logits(
            (
                q.packed[index],
                q.scale[index].contiguous().view(torch.int32).squeeze(-1),
            ),
            (
                k.packed[index],
                k.scale[index].contiguous().view(torch.int32).squeeze(-1),
            ),
            weights[index].float(),
            torch.zeros(7, device="cuda", dtype=torch.int32),
            lengths.flatten(),
            clean_logits=True,
        )
        assert torch.equal(actual[index], original)
        expected.append(original)
    for candidate_mask in (
        None,
        torch.arange(640, device="cuda").remainder(2).bool().expand(batch, 7, 640),
    ):
        selected = dm.indexer_select_positions(
            actual, lengths, 512, candidates=candidate_mask
        )
        for index in range(batch):
            ref = expected[index]
            if candidate_mask is not None:
                ref = ref.masked_fill(~candidate_mask[index], -torch.inf)
            output = torch.empty((7, 512), device="cuda", dtype=torch.int32)
            ops.top_k_per_row_prefill(
                ref,
                torch.zeros(7, device="cuda", dtype=torch.int32),
                lengths.flatten(),
                output,
                7,
                ref.stride(0),
                ref.stride(1),
                512,
            )
            valid = (output >= 0) & (
                ref.gather(1, output.clamp(0, 639).long()) > -torch.inf
            )
            output = torch.where(valid, output, 640).sort(-1).values
            output = torch.where(output < 640, output, -1)
            assert torch.equal(selected[index], output)
    for rows in (1, 8, 239, 384, 4206):
        for zero_padding in (False, True):
            # First actual row is unchanged; other rows are explicit padding.
            pq = q.decoded[0, 5:6].expand(1, rows, 32, 128).contiguous().clone()
            pw = weights[0, 5:6].expand(1, rows, 32).contiguous().clone()
            pl = torch.full((rows, 1), 514, dtype=torch.int32, device="cuda")
            if zero_padding:
                pq[:, 1:] = 0
                pw[:, 1:] = 0
                pl[1:] = 0
            scores = dm.indexer_scores(pq, k.decoded[:1], pw, pl)
            assert torch.equal(scores[0, 0], actual[0, 5])
            selected = dm.indexer_select_positions(scores, pl, 512)
            target = dm.indexer_select_positions(actual[:1, 5:6], lengths[5:6], 512)
            assert torch.equal(selected[0, 0], target[0, 0])
            if zero_padding:
                assert (selected[:, 1:] == -1).all()
