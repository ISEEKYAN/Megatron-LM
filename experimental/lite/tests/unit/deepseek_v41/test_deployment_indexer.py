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
    from vllm.model_executor.determinism import batch_invariant
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    batch_invariant.enable_batch_invariant_mode()
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


@pytest.fixture
def native_blas_backend():
    # Upstream BI disables split-K and selects cuBLASLt. This tiny BF16
    # projection must own that prerequisite even in a combined test process.
    previous = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library("cublaslt")
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(previous)


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("deployment_math", [False, True])
def test_csa_index_query_rope_fp4_midpoint(
    v41_core_te, native_blas_backend, deployment_math
):
    """A one-component FP4 midpoint must match native index-Q arithmetic.

    Feed a small numerical counterexample through the production CSA index-Q
    branch, including its real codec. Stop at that boundary: scoring and
    attention are independent contracts covered by the tests above.
    """
    from types import SimpleNamespace

    from megatron.lite.primitive.modules.attention import csa

    if os.environ.get("MEGATRON_LITE_REQUIRE_CUDA_TESTS") == "1":
        assert torch.cuda.is_available(), "Index RoPE evidence requires CUDA"
    elif not torch.cuda.is_available():
        pytest.skip("CUDA index RoPE")

    class CapturedIndexQuery(Exception):
        pass

    class FixedQuery(torch.nn.Module):
        def forward(self, query):
            result = query.new_zeros(*query.shape[:-1], 32, 128)
            result[..., 31, 76] = -0.294921875
            result[..., 31, 77] = 0.5546875
            return result.flatten(-2)

    captured = []

    def capture_index(query, *, enabled):
        captured.append(csa.fake_quant_index(query, enabled=enabled))
        raise CapturedIndexQuery

    config = csa.CrossLayerAttentionConfig(
        dim=4,
        heads=1,
        q_rank=4,
        o_rank=4,
        groups=1,
        linear_fp8=False,
        main_qat=False,
        swa_fp8=False,
    )
    model = csa.CompressedSparseAttention(
        config,
        layer_idx=1,
        ps=None,
        compress_ratio=2,
        kv_owner=0,
        index_owner=1,
        candidate_mode="none",
        codecs={
            "main": csa.fake_quant_main_kv,
            "index": capture_index,
            "swa": csa.mxfp8.fake_quant_swa,
        },
    ).to(device="cuda", dtype=torch.bfloat16)
    model.indexer.wq_b = FixedQuery()
    model.deployment_math = deployment_math
    query = torch.zeros(1, 1, 4, device="cuda", dtype=torch.bfloat16)
    context = SimpleNamespace(start=2670, gather=lambda _: query.new_zeros(1, 2672, 4))
    state = csa.AttentionState(
        kv_owner=0,
        main_kv=query.new_zeros(1, 1336, 512),
        index_k=query.new_zeros(1, 1336, 128),
    )
    with torch.no_grad(), pytest.raises(CapturedIndexQuery):
        model._forward_cross_layer(query, state, cp_context=context)
    assert len(captured) == 1
    # Native fused_indexer_q uses ordered FP32 multiply-adds, then BF16 and
    # FP4 rounding. The ordinary complex-RoPE recipe retains its old result.
    assert captured[0][0, 0, 31, 77].item() == (0.75 if deployment_math else 0.5)
