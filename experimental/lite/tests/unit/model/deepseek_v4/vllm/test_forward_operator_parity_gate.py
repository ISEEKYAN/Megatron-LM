from __future__ import annotations

import importlib.util
import math

import pytest
import torch

from vllm_forward_parity import ParityTrace, assert_trace_equal


_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
_GPU_REASON = "requires CUDA and official vLLM compiled forward kernels"


def _require_gpu_reference() -> None:
    if not torch.cuda.is_available() or not _VLLM_AVAILABLE:
        pytest.skip(_GPU_REASON)


def _as_tuple(value) -> tuple[torch.Tensor, ...]:
    return value if isinstance(value, tuple) else (value,)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("tokens", [1, 4, 32])
@pytest.mark.parametrize("kernel", ["pre", "pre_broadcast", "post", "head"])
def test_mhc_each_official_forward_op_is_bitwise(
    transformer_engine_import_stub, kernel: str, tokens: int
) -> None:
    _require_gpu_reference()
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v4.vllm.primitive.dense import (
        _MHC_ENTRIES,
        mhc_kernel,
    )

    torch.manual_seed(9101 + tokens)
    mult, hidden = 4, 128
    if kernel == "post":
        args = (
            torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda"),
            torch.randn(tokens, mult, hidden, dtype=torch.bfloat16, device="cuda"),
            torch.randn(tokens, mult, 1, dtype=torch.float32, device="cuda"),
            torch.randn(tokens, mult, mult, dtype=torch.float32, device="cuda"),
        )
        kwargs = {}
    elif kernel == "head":
        args = (
            torch.randn(tokens, mult, hidden, dtype=torch.bfloat16, device="cuda"),
            torch.randn(mult, mult * hidden, dtype=torch.float32, device="cuda"),
            torch.randn(1, dtype=torch.float32, device="cuda"),
            torch.randn(mult, dtype=torch.float32, device="cuda"),
            1e-6,
            1e-6,
        )
        kwargs = {}
    else:
        residual_shape = (
            (tokens, hidden)
            if kernel == "pre_broadcast"
            else (tokens, mult, hidden)
        )
        fn = torch.randn(
            (2 + mult) * mult,
            mult * hidden,
            dtype=torch.float32,
            device="cuda",
        )
        args = (
            torch.randn(*residual_shape, dtype=torch.bfloat16, device="cuda"),
            fn,
            torch.randn(3, dtype=torch.float32, device="cuda"),
            torch.randn((2 + mult) * mult, dtype=torch.float32, device="cuda"),
            1e-6,
            1e-6,
            1e-6,
            2.0,
            2,
        )
        kwargs = {
            "norm_weight": torch.randn(hidden, dtype=torch.bfloat16, device="cuda"),
            "norm_eps": 1e-6,
        }
        if kernel == "pre_broadcast":
            kwargs["fn_broadcast"] = (
                fn.view(-1, mult, hidden).sum(dim=1).contiguous()
            )

    official = _as_tuple(_MHC_ENTRIES[kernel](*args, **kwargs))
    candidate = _as_tuple(mhc_kernel(kernel, *args, **kwargs))
    reference_trace, candidate_trace = ParityTrace(), ParityTrace()
    for index, (expected, actual) in enumerate(
        zip(official, candidate, strict=True)
    ):
        reference_trace.add(f"mhc.{kernel}", **{f"output{index}": expected})
        candidate_trace.add(f"mhc.{kernel}", **{f"output{index}": actual})
    assert_trace_equal(reference_trace, candidate_trace)


def _direct_moe_trace(
    hidden: torch.Tensor,
    counts: tuple[int, ...],
    w13: tuple[torch.Tensor, ...],
    w2: tuple[torch.Tensor, ...],
    limit: float,
    *,
    candidate_helpers: bool,
) -> ParityTrace:
    import vllm.envs as envs
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
        fused_silu_mul_per_token_group_quant_fp8,
        per_token_group_quant_fp8,
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.utils.deep_gemm import (
        DeepGemmQuantScaleFMT,
        m_grouped_fp8_gemm_nt_contiguous,
        per_block_cast_to_fp8,
    )
    from megatron.lite.model.deepseek_v4.vllm.primitive.block_fp8 import (
        pack_grouped_block_fp8_weight,
    )
    from megatron.lite.model.deepseek_v4.vllm.primitive.moe.grouped import (
        _pad_expert_rows,
        _vllm_quantize_contiguous_input,
        _vllm_silu_mul_quant,
    )

    def official_input_quant(value):
        scale_format = DeepGemmQuantScaleFMT.from_oracle()
        if scale_format == DeepGemmQuantScaleFMT.UE8M0:
            return per_token_group_quant_fp8_packed_for_deepgemm(
                value, 128, use_ue8m0=True
            )
        return per_token_group_quant_fp8(
            value,
            128,
            eps=1e-10,
            dtype=torch.float8_e4m3fn,
            column_major_scales=True,
            tma_aligned_scales=bool(
                envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES
            ),
            use_ue8m0=(
                scale_format == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
            ),
        )

    def official_weight_pack(weights):
        quantized = [
            per_block_cast_to_fp8(
                weight.detach(), block_size=[128, 128], use_ue8m0=False
            )
            for weight in weights
        ]
        qweight, scale = deepgemm_post_process_fp8_weight_block(
            wq=torch.stack([item[0] for item in quantized]),
            ws=torch.stack([item[1] for item in quantized]),
            quant_block_shape=(128, 128),
            use_e8m0=True,
        )
        return qweight, scale

    padded, layout = _pad_expert_rows(hidden, counts)
    m_indices = layout.m_indices
    input_q, input_scale = (
        _vllm_quantize_contiguous_input(padded)
        if candidate_helpers
        else official_input_quant(padded)
    )
    if candidate_helpers:
        packed_w13 = pack_grouped_block_fp8_weight(w13)
        w13_q, w13_scale = packed_w13.qweight, packed_w13.scales
    else:
        w13_q, w13_scale = official_weight_pack(w13)
    gate_up = hidden.new_empty((padded.shape[0], w13[0].shape[0]))
    m_grouped_fp8_gemm_nt_contiguous(
        (input_q, input_scale),
        (w13_q, w13_scale),
        gate_up,
        m_indices,
    )
    activated_q = torch.empty(
        padded.shape[0],
        w2[0].shape[1],
        device=hidden.device,
        dtype=torch.float8_e4m3fn,
    )
    scale_format = DeepGemmQuantScaleFMT.from_oracle()
    if candidate_helpers:
        activated_q, activated_scale = _vllm_silu_mul_quant(
            gate_up,
            output=activated_q,
            swiglu_limit=limit,
        )
    else:
        activated_q, activated_scale = fused_silu_mul_per_token_group_quant_fp8(
            gate_up,
            output_q=activated_q,
            use_ue8m0=(scale_format == DeepGemmQuantScaleFMT.UE8M0),
            round_scale=(scale_format != DeepGemmQuantScaleFMT.FLOAT32),
            clamp_limit=limit,
            masked_m=None,
            group_size=128,
        )
    if candidate_helpers:
        packed_w2 = pack_grouped_block_fp8_weight(w2)
        w2_q, w2_scale = packed_w2.qweight, packed_w2.scales
    else:
        w2_q, w2_scale = official_weight_pack(w2)
    output = hidden.new_empty((padded.shape[0], w2[0].shape[0]))
    m_grouped_fp8_gemm_nt_contiguous(
        (activated_q, activated_scale),
        (w2_q, w2_scale),
        output,
        m_indices,
    )
    if layout.valid_rows is not None:
        output = output.index_select(0, layout.valid_rows)

    trace = ParityTrace()
    trace.add("moe.input_quant", q=input_q, scale=input_scale)
    trace.add(
        "moe.w13_quant",
        q=w13_q,
        scale=w13_scale,
    )
    trace.add("moe.gate_up", intermediate=gate_up)
    trace.add(
        "moe.activation_quant",
        q=activated_q,
        scale=activated_scale,
    )
    trace.add("moe.w2_quant", q=w2_q, scale=w2_scale)
    trace.add("moe.output", hidden=output)
    return trace


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("tokens", [1, 4, 32])
def test_moe_quant_scale_intermediate_and_output_are_bitwise(tokens: int) -> None:
    _require_gpu_reference()
    from vllm.config import VllmConfig, set_current_vllm_config
    import vllm.utils.deep_gemm as deep_gemm
    from megatron.lite.model.deepseek_v4.vllm.primitive.moe.grouped import (
        _vllm_grouped_forward,
    )

    deep_gemm._lazy_init()
    deep_gemm.DeepGemmQuantScaleFMT.init_oracle_cache()
    torch.manual_seed(9201 + tokens)
    hidden = torch.randn(tokens, 128, dtype=torch.bfloat16, device="cuda")
    counts = (tokens,)
    w13 = (
        torch.randn(256, 128, dtype=torch.bfloat16, device="cuda"),
    )
    w2 = (
        torch.randn(128, 128, dtype=torch.bfloat16, device="cuda"),
    )
    with set_current_vllm_config(VllmConfig()), torch.no_grad():
        reference = _direct_moe_trace(
            hidden, counts, w13, w2, 10.0, candidate_helpers=False
        )
        output = _vllm_grouped_forward(hidden, counts, 10.0, w13, w2)
        candidate = _direct_moe_trace(
            hidden, counts, w13, w2, 10.0, candidate_helpers=True
        )
        candidate_entries = list(candidate.entries)
        candidate = ParityTrace()
        for entry in candidate_entries[:-1]:
            candidate.add(entry.op, **{entry.tensor: entry.value})
        candidate.add("moe.output", hidden=output)
    assert_trace_equal(reference, candidate)


@pytest.mark.gpus(1)
def test_sparse_attention_official_forward_is_bitwise_at_2k_6k_edges() -> None:
    _require_gpu_reference()
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd
    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.backward import (
        visible_sparse_attention,
    )

    torch.manual_seed(9301)
    positions = torch.tensor(
        [2046, 2047, 2048, 6142, 6143, 6144],
        dtype=torch.int64,
        device="cuda",
    )
    rows, heads, dim, topk = positions.numel(), 64, 576, 128
    q = (
        torch.randn(rows, heads, dim, device="cuda") / math.sqrt(dim)
    ).bfloat16()
    kv = (
        torch.randn(6145, dim, device="cuda") / math.sqrt(dim)
    ).bfloat16()
    indices = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    for row, position in enumerate(positions.tolist()):
        indices[row] = torch.arange(
            position + 1 - topk,
            position + 1,
            dtype=torch.int32,
            device="cuda",
        )
    lengths = torch.full((rows,), topk, dtype=torch.int32, device="cuda")
    sink = torch.zeros(heads, dtype=torch.float32, device="cuda")
    scale = dim**-0.5

    def official(_q: torch.Tensor, _kv: torch.Tensor):
        return flash_mla_sparse_fwd(
            _q,
            _kv.unsqueeze(1),
            indices.unsqueeze(1),
            scale,
            d_v=512,
            attn_sink=sink,
            topk_length=lengths,
        )

    with torch.no_grad():
        expected = official(q, kv)[0]
        actual = visible_sparse_attention(
            official,
            q,
            kv,
            indices,
            lengths,
            sink,
            softmax_scale=scale,
        )
    reference, candidate = ParityTrace(), ParityTrace()
    reference.add("indexer.attention", output=expected)
    candidate.add("indexer.attention", output=actual)
    assert_trace_equal(reference, candidate)


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_indexer_official_topk_is_bitwise_at_2k_6k_edges(
    transformer_engine_import_stub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_gpu_reference()
    transformer_engine_import_stub()
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        fused_indexer_q_rope_quant,
    )
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    import vllm.utils.deep_gemm as deep_gemm_module
    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.runtime import (
        official_indexer_topk,
    )

    torch.manual_seed(9401)
    ratio, topk, groups = 4, 512, 1537
    positions = torch.tensor(
        [2047, 2048, 2049, 6143, 6144, 6145],
        dtype=torch.int64,
        device="cuda",
    )
    index_q = torch.randn(
        positions.numel(), 64, 128, dtype=torch.bfloat16, device="cuda"
    )
    index_weights = torch.randn(
        positions.numel(), 64, dtype=torch.bfloat16, device="cuda"
    )
    index_k = torch.randn(groups, 128, dtype=torch.bfloat16, device="cuda")
    cos_sin = torch.randn(6146, 64, dtype=torch.float32, device="cuda")
    cu = torch.tensor([0, positions.numel()], dtype=torch.int32, device="cuda")
    cu_compressed = torch.tensor([0, groups], dtype=torch.int32, device="cuda")

    candidate_logits = []

    def capture_logits(*args, **kwargs):
        value = fp8_fp4_mqa_logits(*args, **kwargs)
        candidate_logits.append(value.clone())
        return value

    monkeypatch.setattr(
        deep_gemm_module,
        "fp8_fp4_mqa_logits",
        capture_logits,
    )
    actual = official_indexer_topk(
        index_q,
        index_weights,
        index_k,
        positions,
        cos_sin,
        cu,
        cu_compressed,
        global_start=0,
        ratio=ratio,
        topk=topk,
    )
    q_quant, weights = fused_indexer_q_rope_quant(
        positions,
        index_q,
        cos_sin,
        index_weights,
        index_q.shape[-1] ** -0.5,
        index_q.shape[1] ** -0.5,
        use_fp4=False,
    )
    k_quant, k_scale = per_token_group_quant_fp8(
        index_k.contiguous(),
        group_size=128,
        use_ue8m0=True,
    )
    k_scale = k_scale.view(torch.float32).squeeze(-1)
    padded_k_rows = ((k_quant.shape[0] + 127) // 128) * 128
    if padded_k_rows != k_quant.shape[0]:
        k_quant = torch.cat(
            (
                k_quant,
                k_quant.new_zeros(
                    (padded_k_rows - k_quant.shape[0], k_quant.shape[1])
                ),
            )
        )
        k_scale = torch.cat(
            (k_scale, k_scale.new_ones(padded_k_rows - k_scale.shape[0]))
        )
    row_starts = torch.zeros(positions.numel(), dtype=torch.int32, device="cuda")
    row_ends = torch.div(positions + 1, ratio, rounding_mode="floor").to(
        torch.int32
    )
    logits = fp8_fp4_mqa_logits(
        (q_quant, None),
        (k_quant, k_scale),
        weights,
        row_starts,
        row_ends,
        clean_logits=False,
    )
    assert len(candidate_logits) == 1
    assert torch.equal(candidate_logits[0], logits)
    expected = torch.full(
        (positions.numel(), topk),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    ops.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        expected,
        positions.numel(),
        logits.stride(0),
        logits.stride(1),
        topk,
    )
    expected_repeat = torch.full_like(expected, -1)
    ops.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        expected_repeat,
        positions.numel(),
        logits.stride(0),
        logits.stride(1),
        topk,
    )
    assert torch.equal(expected_repeat, expected)
    reference, candidate = ParityTrace(), ParityTrace()
    reference.add("indexer.topk", indices=expected)
    candidate.add("indexer.topk", indices=actual)
    assert_trace_equal(reference, candidate)
