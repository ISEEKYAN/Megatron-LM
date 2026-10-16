# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native BF16 index ranking matches original packets and padded batches."""

import os

import pytest
import torch


@pytest.mark.parametrize("rows", [0, 1, 63, 64, 65, 6464])
def test_private_cache_pages_isolate_requests_and_preserve_partial_pages(
    v41_core_te, rows
):
    from megatron.lite.primitive.modules.deployment_math.cache import cache_geometry

    table, slots, lengths = cache_geometry(2, rows, 64, device="cpu")
    assert table.dtype == torch.int32 and slots.dtype == torch.int64
    assert lengths.tolist() == [rows, rows]
    assert slots.shape == (2, rows)
    if rows:
        assert torch.equal(slots[0], torch.arange(rows))
        assert torch.equal(slots[1], torch.arange(rows) + table.shape[1] * 64)
        assert slots.unique().numel() == 2 * rows
        pages = slots.div(64, rounding_mode="floor")
        assert set(pages[0].tolist()) <= set(table[0].tolist())
        assert set(pages[1].tolist()) <= set(table[1].tolist())
        assert not set(table[0].tolist()) & set(table[1].tolist())


def test_private_cache_cpu_rejects_native_execution_without_cuda(v41_core_te):
    from megatron.lite.primitive.modules.deployment_math import cache

    before = torch.cuda.is_initialized()
    values = torch.ones(1, 2, 512, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA BF16"):
        cache.compressed_cache(values, 2, None, lambda x: x)
    with pytest.raises(ValueError, match="CUDA BF16"):
        cache.index_cache(values[..., :128], torch.ones(128), 1e-20, 2, None)
    assert torch.cuda.is_initialized() == before


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
                q.packed[index].view(torch.int8),
                q.scale[index].contiguous().view(torch.int32).squeeze(-1),
            ),
            (
                k.packed[index].view(torch.int8),
                k.scale[index].contiguous().view(torch.int32).squeeze(-1),
            ),
            weights[index].bfloat16(),
            torch.zeros(7, device="cuda", dtype=torch.int32),
            lengths.flatten(),
            max_seqlen_k=640,
        )
        assert original.dtype == torch.bfloat16
        valid = torch.arange(640, device="cuda") < lengths
        original = original.float().masked_fill(~valid, -torch.inf)
        assert torch.equal(actual[index], original)
        assert torch.isneginf(actual[index][~valid]).all()
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
            reference = torch.full((7, 512), -1, dtype=torch.int32)
            # The generic selector resolves ties by temporary atomic slot;
            # use an independent CPU oracle for the explicit local-index rule.
            for row, (values, limit) in enumerate(
                zip(ref.cpu().tolist(), lengths.flatten().cpu().tolist())
            ):
                valid = [
                    position
                    for position in range(limit)
                    if values[position] > -float("inf")
                ]
                order = sorted(
                    valid,
                    key=lambda position: (values[position], position),
                    reverse=True,
                )[:512]
                reference[row, : len(order)] = torch.tensor(
                    sorted(order), dtype=torch.int32
                )
            assert torch.equal(selected[index].cpu(), reference)
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


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("deployment_math", [False, True])
def test_csa_index_query_rope_fp4_midpoint(v41_core_te, deployment_math):
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

    # Isolate the index-Q rotation/codec boundary. These unrelated query/KV
    # projections are not under test; fixed tensors avoid a tiny-BF16 BLAS
    # fallback inheriting process-wide split-K settings from other tests.
    class FixedProjection(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.width = width

        def forward(self, value):
            return value.new_zeros(*value.shape[:-1], self.width)

    model.wq_a = FixedProjection(config.q_rank)
    model.wq_b = FixedProjection(config.heads * config.head_dim)
    model.wkv = FixedProjection(config.head_dim)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA native BF16 MQA')
def test_indexer_batched_requests_one_launch_and_serial_bitwise(monkeypatch):
    from megatron.lite.primitive.modules import deployment_math as dm
    from vllm.utils import deep_gemm

    torch.manual_seed(1943)
    q = torch.randn(3, 7, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 139, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3, 7, 32, device="cuda", dtype=torch.float32)
    lengths = torch.randint(1, 140, (3, 7, 1), device="cuda", dtype=torch.int32)
    expected = torch.cat(
        [
            dm.indexer_scores(
                q[b : b + 1], k[b : b + 1], w[b : b + 1], lengths[b : b + 1]
            )
            for b in range(3)
        ]
    )
    calls = []
    native = deep_gemm.fp8_fp4_mqa_logits

    def counted(*args, **kwargs):
        calls.append(args[0][0].shape[0])
        return native(*args, **kwargs)

    monkeypatch.setattr(deep_gemm, 'fp8_fp4_mqa_logits', counted)
    actual = dm.indexer_scores(q, k, w, lengths)
    assert calls == [21]
    assert torch.equal(actual, expected)
    assert actual.shape == (3, 7, 139)
    assert actual.numel() == 3 * 7 * 139
    selected = dm.indexer_select_positions(actual, lengths, 512)
    wanted = torch.cat(
        [
            dm.indexer_select_positions(
                dm.indexer_scores(
                    q[b : b + 1], k[b : b + 1], w[b : b + 1], lengths[b : b + 1]
                ),
                lengths[b : b + 1],
                512,
            )
            for b in range(3)
        ]
    )
    assert torch.equal(selected, wanted)


def test_fp32_diagnostic_keeps_weight_midpoint_and_masks_causal_tail():
    """The diagnostic must retain a coefficient that BF16 would round away."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / 'examples/verl/scripts/ds41_w4a8_proxy/indexer_diagnostic.py'
    )
    spec = importlib.util.spec_from_file_location('ds41_indexer_diagnostic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    query = torch.full((1, 130, 32, 128), 1 / 16, dtype=torch.bfloat16)
    keys = torch.full((1, 2, 128), 1 / 16, dtype=torch.bfloat16)
    weights = torch.full((1, 130, 32), 1 + 2**-8, dtype=torch.float32)
    lengths = (torch.arange(130, dtype=torch.int32) % 2).unsqueeze(-1)
    actual = module.fp32_score_reference(query, keys, weights, lengths)
    assert actual.dtype == torch.float32
    assert actual[0, 1, 0].item() == 16.0625
    assert actual[0, 1, 0].bfloat16().item() == 16.0
    assert torch.isneginf(actual[0, 0]).all() and torch.isneginf(actual[0, 1, 1])
    assert actual[0, 129, 0].item() == 16.0625
    assert torch.isneginf(actual[0, 128]).all() and torch.isneginf(actual[0, 129, 1])


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize('use_candidates', [False, True])
def test_indexer_selection_crosses_batch_dispatch_boundary(
    v41_core_te, monkeypatch, use_candidates
):
    """Shared selection and actual attention outputs survive the row boundary."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA Top-K')
    from megatron.lite.primitive.modules import deployment_math as dm
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels import batch_invariant as bi

    rows, columns, top_k = 12290, 640, 512
    lengths = torch.tensor([640, 0, 1, 511, 512, 513], dtype=torch.int32)
    lengths = lengths.repeat((rows + 5) // 6)[:rows]
    scores = torch.ones((1, rows, columns), device="cuda", dtype=torch.float32)
    ends = lengths.cuda().reshape(1, rows, 1)
    scores.masked_fill_(torch.arange(columns, device="cuda") >= ends, -torch.inf)
    candidates = None
    if use_candidates:
        candidates = torch.arange(columns, device="cuda") % 2 == 0
        candidates = candidates.expand_as(scores)
    calls = []
    native = bi.top_k_per_row_prefill

    def counted(*args):
        calls.append(args[4])
        return native(*args)

    def reject_legacy(*args, **kwargs):
        raise AssertionError('The training BI path used the legacy dispatcher')

    monkeypatch.setattr(bi, 'top_k_per_row_prefill', counted)
    monkeypatch.setattr(ops, 'top_k_per_row_prefill', reject_legacy)
    selected = dm.indexer_select_positions(scores, ends, top_k, candidates=candidates)
    assert calls == [rows]
    expected = torch.full((rows, top_k), -1, dtype=torch.int32)
    for length in lengths.unique().tolist():
        eligible = list(range(0, length, 2 if use_candidates else 1))[-top_k:]
        reference = torch.full((top_k,), -1, dtype=torch.int32)
        reference[: len(eligible)] = torch.tensor(eligible, dtype=torch.int32)
        expected[lengths == length] = reference
    assert torch.equal(selected[0].cpu(), expected)
    assert torch.equal(selected[0, 0], selected[0, 12288])

    # Exercise the inference post-selection owner and the real downstream
    # FlashMLA reduction; an equal score or set alone is insufficient.
    from vllm.models.deepseek_v41.attention import _sort_valid_topk_indices_
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    inference_scores = scores[0]
    if candidates is not None:
        inference_scores = inference_scores.masked_fill(~candidates[0], -torch.inf)
    inference_positions = torch.empty_like(selected[0])
    bi.top_k_per_row_prefill(
        inference_scores,
        torch.zeros(rows, device="cuda", dtype=torch.int32),
        ends.flatten(),
        inference_positions,
        rows,
        inference_scores.stride(0),
        inference_scores.stride(1),
        top_k,
    )
    _sort_valid_topk_indices_(inference_positions)
    assert calls == [rows, rows]
    assert torch.equal(inference_positions, selected[0])

    torch.manual_seed(1947)
    heads, dim = 32, 512
    query = torch.randn(1, 1, heads, dim, device="cuda", dtype=torch.bfloat16)
    query = query.expand(1, 2, heads, dim).contiguous()
    kv = torch.randn(1, columns, dim, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    mask = torch.ones(1, 2, columns, device="cuda", dtype=torch.bool)
    chosen = selected[:, [0, 12288]]
    kwargs = dict(main_size=columns, window_size=0, width=top_k, scale=dim**-0.5)
    actual = dm.sparse_attention(query, kv, sink, mask, chosen, **kwargs)
    single_positions = dm.indexer_select_positions(
        scores[:, :1],
        ends[:, :1],
        top_k,
        candidates=None if candidates is None else candidates[:, :1],
    )
    serial = dm.sparse_attention(
        query[:, :1], kv, sink, mask[:, :1], single_positions, **kwargs
    )
    padded_query = torch.nn.functional.pad(query[0], (0, 0, 0, 64 - heads))
    output = torch.empty_like(padded_query)
    native_positions = inference_positions[[0, 12288]]
    flash_mla_sparse_fwd(
        q=padded_query,
        kv=kv[0].unsqueeze(1).contiguous(),
        indices=native_positions.unsqueeze(1),
        sm_scale=dim**-0.5,
        attn_sink=torch.nn.functional.pad(sink, (0, 64 - heads)),
        topk_length=(native_positions >= 0).sum(-1).to(torch.int32),
        out=output,
    )
    assert torch.equal(actual[0], output[:, :heads])
    assert torch.equal(actual[:, :1], serial)
    assert torch.equal(actual[:, 0], actual[:, 1])
