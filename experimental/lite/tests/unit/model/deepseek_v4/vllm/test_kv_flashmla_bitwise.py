from __future__ import annotations

import importlib.util
from unittest.mock import Mock

import pytest
import torch

from megatron.lite.primitive.kernels import vllm_ds4
from megatron.lite.primitive.kernels.vllm_ds4 import (
    CompressorKernelAdapter,
    DS4KVInsertAdapter,
    FlashMLAAdapter,
    FusedQKVRMSNormAdapter,
    IndexerKernelAdapter,
)


def test_qkv_norm_cpu_contract_calls_official_symbol(monkeypatch) -> None:
    q, kv = torch.randn(2, 12).split((8, 4), dim=-1)
    assert not q.is_contiguous() and not kv.is_contiguous()
    assert q.stride(-1) == kv.stride(-1) == 1
    qw, kw = torch.ones(8), torch.ones(4)
    expected = (q + 1, kv + 1)
    kernel = Mock(return_value=expected)
    monkeypatch.setattr(vllm_ds4, "_symbol", lambda module, name: kernel)
    actual = FusedQKVRMSNormAdapter()(q, kv, qw, kw, 1e-6)
    assert all(got is want for got, want in zip(actual, expected, strict=True))
    kernel.assert_called_once_with(q, kv, qw, kw, 1e-6)


def test_kv_insert_cpu_contract_calls_exact_custom_op(monkeypatch) -> None:
    kernel = Mock()
    monkeypatch.setattr(vllm_ds4, "_op", lambda namespace, name: kernel)
    q = torch.zeros(2, 3, 8, dtype=torch.bfloat16)
    kv = torch.zeros(2, 8, dtype=torch.bfloat16)
    cache = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    result = DS4KVInsertAdapter("plain_bf16")(
        q,
        kv,
        cache,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.zeros(16, 16, dtype=torch.float32),
        eps=1e-6,
        block_size=4,
    )
    assert result is q
    assert kernel.call_count == 1
    assert (
        kernel.call_args.args[-2:]
        == (1e-6, 4)
    )


def test_sparse_and_paged_flashmla_cpu_contracts(monkeypatch) -> None:
    sparse = Mock(return_value=(torch.tensor(1), torch.tensor(2)))
    paged = Mock(return_value=(torch.tensor(3), torch.tensor(4)))

    def lookup(_module, name):
        return {"flash_mla_sparse_fwd": sparse, "flash_mla_with_kvcache": paged}[name]

    monkeypatch.setattr(vllm_ds4, "_symbol", lookup)
    adapter = FlashMLAAdapter()
    q = torch.zeros(1, 4, 8)
    indices = torch.zeros(1, 1, 4, dtype=torch.int32)
    adapter.sparse(q, torch.zeros(8, 1, 8), indices, sm_scale=0.5)
    sparse.assert_called_once()

    q_decode = torch.zeros(1, 1, 4, 8)
    out = torch.empty_like(q_decode)
    adapter.paged(
        q_decode,
        torch.zeros(1, 4, 1, 8),
        tile_scheduler_metadata=object(),
        indices=indices,
        topk_length=torch.ones(1, dtype=torch.int32),
        softmax_scale=0.5,
        attn_sink=torch.zeros(4),
        out=out,
    )
    paged.assert_called_once()
    assert paged.call_args.kwargs["tile_scheduler_metadata"] is not None
    assert paged.call_args.kwargs["out"] is out


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA for compressor/indexer kernel bitwise validation",
)
def test_compressor_and_indexer_kernel_boundaries_are_bitwise() -> None:
    torch.manual_seed(29)
    kv_score = torch.randn(2, 512, dtype=torch.float32, device="cuda")
    positions = torch.arange(2, dtype=torch.int64, device="cuda")
    ape = torch.randn(4, 256, dtype=torch.float32, device="cuda")
    norm = torch.randn(128, dtype=torch.bfloat16, device="cuda")

    def compressor_kernel(**kwargs):
        return kwargs["kv_score"].square()

    reference_compressor = compressor_kernel(kv_score=kv_score)
    candidate_compressor = CompressorKernelAdapter()(
        compressor_kernel,
        kv_score,
        positions,
        ape,
        norm,
        compress_ratio=4,
        head_dim=128,
        metadata=object(),
    )
    torch.testing.assert_close(
        candidate_compressor, reference_compressor, rtol=0, atol=0
    )

    qr = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda")
    index_q = torch.randn(2, 2, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(2, 2, dtype=torch.bfloat16, device="cuda")

    def indexer_kernel(**kwargs):
        return (kwargs["index_q"].float() * kwargs["index_weights"][..., None]).sum(
            dim=-1
        )

    reference_indexer = indexer_kernel(
        index_q=index_q, index_weights=weights
    )
    candidate_indexer = IndexerKernelAdapter()(
        indexer_kernel,
        qr,
        index_q,
        weights,
        positions,
        compress_ratio=4,
        topk=2,
        metadata=object(),
    )
    torch.testing.assert_close(candidate_indexer, reference_indexer, rtol=0, atol=0)


def _production_ready() -> bool:
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("vllm") is not None
        and importlib.util.find_spec("flash_mla") is not None
    )


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _production_ready(),
    reason="requires CUDA plus official vLLM and FlashMLA compiled dependencies",
)
def test_official_fused_qkv_norm_is_bitwise_through_adapter() -> None:
    from vllm.models.common.ops import fused_q_kv_rmsnorm

    torch.manual_seed(19)
    q = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(3, 576, dtype=torch.bfloat16, device="cuda")
    qw = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    kw = torch.randn(576, dtype=torch.bfloat16, device="cuda")
    reference = fused_q_kv_rmsnorm(
        q.clone(), kv.clone(), qw.clone(), kw.clone(), 1e-6
    )
    candidate = FusedQKVRMSNormAdapter()(
        q.clone(), kv.clone(), qw.clone(), kw.clone(), 1e-6
    )
    for actual, expected in zip(candidate, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _production_ready(),
    reason="requires CUDA plus official vLLM DS4 cache-insert compiled kernels",
)
def test_official_bf16_kv_insert_is_bitwise_through_adapter() -> None:
    import vllm._C_stable_libtorch  # noqa: F401

    torch.manual_seed(21)
    q = torch.randn(2, 64, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(2, 512, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    positions = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    cos = torch.ones(8, 64, dtype=torch.float32, device="cuda")
    reference_q, candidate_q = q.clone(), q.clone()
    reference_cache = torch.zeros(1, 64, 512, dtype=torch.bfloat16, device="cuda")
    candidate_cache = reference_cache.clone()
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
        reference_q,
        kv.clone(),
        reference_cache,
        slots,
        positions,
        cos,
        1e-6,
        64,
    )
    candidate = DS4KVInsertAdapter("plain_bf16")(
        candidate_q,
        kv.clone(),
        candidate_cache,
        slots,
        positions,
        cos,
        eps=1e-6,
        block_size=64,
    )
    torch.testing.assert_close(candidate, reference_q, rtol=0, atol=0)
    torch.testing.assert_close(candidate_cache, reference_cache, rtol=0, atol=0)


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _production_ready(),
    reason="requires CUDA plus official vLLM DS4 FP8 cache-insert kernels",
)
def test_official_fp8_ds_mla_kv_quant_insert_is_bitwise() -> None:
    import vllm._C_stable_libtorch  # noqa: F401

    torch.manual_seed(22)
    tokens, heads, head_dim, block_size = 2, 64, 512, 256
    token_bytes = 584
    q = torch.randn(
        tokens, heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    kv = torch.randn(tokens, head_dim, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    positions = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    cos = torch.ones(8, 64, dtype=torch.float32, device="cuda")
    reference_cache = torch.zeros(
        1, block_size * token_bytes, dtype=torch.uint8, device="cuda"
    )
    candidate_cache = reference_cache.clone()

    reference = (
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q.clone(),
            kv.clone(),
            reference_cache,
            slots,
            positions,
            cos,
            heads,
            1e-6,
            block_size,
        )
    )
    candidate = DS4KVInsertAdapter("fp8_ds_mla")(
        q.clone(),
        kv.clone(),
        candidate_cache,
        slots,
        positions,
        cos,
        eps=1e-6,
        block_size=block_size,
        padded_heads=heads,
    )
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    assert torch.equal(candidate_cache, reference_cache)


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _production_ready(),
    reason="requires CUDA plus official vLLM and FlashMLA compiled dependencies",
)
def test_official_flashmla_sparse_is_bitwise_through_adapter() -> None:
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    torch.manual_seed(23)
    q = torch.randn(1, 64, 576, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(8, 1, 576, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(8, (1, 1, 128), dtype=torch.int32, device="cuda")
    lengths = torch.full((1,), 128, dtype=torch.int32, device="cuda")
    sink = torch.zeros(64, dtype=torch.float32, device="cuda")
    out_ref = torch.empty(1, 64, 512, dtype=torch.bfloat16, device="cuda")
    out_candidate = torch.empty_like(out_ref)

    reference = flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=1.0,
        attn_sink=sink,
        topk_length=lengths,
        out=out_ref,
    )
    candidate = FlashMLAAdapter().sparse(
        q,
        kv,
        indices,
        sm_scale=1.0,
        attn_sink=sink,
        topk_length=lengths,
        out=out_candidate,
    )
    for actual, expected in zip(candidate, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
