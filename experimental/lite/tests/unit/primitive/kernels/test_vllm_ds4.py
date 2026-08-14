# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.mlite

from megatron.lite.primitive.kernels import vllm_ds4 as adapters


def _patch_symbol(monkeypatch, expected_name, result=None):
    call = Mock(return_value=result)

    def load(_module, name):
        assert name == expected_name
        return call

    monkeypatch.setattr(adapters, "_symbol", load)
    return call


def _patch_op(monkeypatch, expected_namespace, expected_name, result=None):
    call = Mock(return_value=result)

    def load(namespace, name):
        assert (namespace, name) == (expected_namespace, expected_name)
        return call

    monkeypatch.setattr(adapters, "_op", load)
    return call


def test_adapters_are_parameter_free_and_import_without_vllm():
    instances = [
        adapters.MHCTileLangAdapter("post"),
        adapters.FusedQKVRMSNormAdapter(),
        adapters.DS4KVInsertAdapter("plain_bf16"),
        adapters.FlashMLAAdapter(),
        adapters.GateLinearAdapter(),
        adapters.DS4TopKAdapter(),
        adapters.HashRouteAdapter(),
        adapters.FusedExpertsAdapter(),
        adapters.SharedExpertsAdapter(),
        adapters.OProjectionAdapter(),
    ]
    assert all(list(module.parameters()) == [] for module in instances)


def test_o_projection_uses_official_grouped_fp8_entry(monkeypatch):
    o = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
    positions = torch.zeros(1, dtype=torch.int64)
    cache = torch.zeros(2, 128, dtype=torch.bfloat16)
    wo_a = torch.nn.Parameter(torch.zeros(256, 128, dtype=torch.bfloat16))
    wo_b = torch.nn.Parameter(torch.zeros(128, 256, dtype=torch.bfloat16))
    expected = torch.ones(1, 128, dtype=torch.bfloat16)

    cast = Mock(
        side_effect=[
            (
                torch.empty_like(wo_a, dtype=torch.float8_e4m3fn),
                torch.ones(2, 1, dtype=torch.float32),
            ),
            (
                torch.empty_like(wo_b, dtype=torch.float8_e4m3fn),
                torch.ones(1, 2, dtype=torch.float32),
            ),
        ]
    )
    processed_a = (
        torch.empty(2, 128, 128, dtype=torch.float8_e4m3fn),
        torch.ones(2, 1, 1, dtype=torch.float32),
    )
    processed_b = (
        torch.empty_like(wo_b, dtype=torch.float8_e4m3fn),
        torch.ones(1, 2, dtype=torch.float32),
    )
    post = Mock(side_effect=[processed_a, processed_b])
    recipe = Mock(return_value=((1, 128, 128), False))
    official = Mock(return_value=expected)

    symbols = {
        "per_block_cast_to_fp8": cast,
        "deepgemm_post_process_fp8_weight_block": post,
        "deep_gemm_fp8_o_proj": official,
        "compute_fp8_einsum_recipe": recipe,
    }
    monkeypatch.setattr(adapters, "_symbol", lambda _module, name: symbols[name])

    actual = adapters.OProjectionAdapter()(
        o,
        positions,
        cache,
        wo_a,
        wo_b,
        n_groups=2,
        heads_per_group=1,
        nope_dim=64,
        rope_dim=64,
        o_lora_rank=128,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(NotImplementedError, match="inference-only"):
        actual.sum().backward()
    assert official.call_count == 1
    kwargs = official.call_args.kwargs
    assert kwargs["n_groups"] == 2
    assert kwargs["heads_per_group"] == 1
    assert kwargs["einsum_recipe"] == (1, 128, 128)
    assert post.call_args_list[0].kwargs["is_bmm"] is True
    assert post.call_args_list[0].kwargs["bmm_batch_size"] == 2


def test_missing_lazy_dependency_fails_closed(monkeypatch):
    def fail(_module):
        raise ImportError("missing")

    monkeypatch.setattr(adapters.importlib, "import_module", fail)
    adapter = adapters.FusedQKVRMSNormAdapter()
    q = torch.zeros(2, 4)
    with pytest.raises(NotImplementedError, match="unavailable"):
        adapter(q, q, torch.ones(4), torch.ones(4), 1e-6)


@pytest.mark.parametrize(
    ("kind", "args", "entry"),
    [
        (
            "pre",
            (
                torch.zeros(2, 2, 4, dtype=torch.bfloat16),
                torch.zeros(8, 8),
                torch.zeros(3),
                torch.zeros(8),
                1e-6,
                1e-6,
                1e-6,
                2.0,
                5,
            ),
            "mhc_pre_tilelang",
        ),
        (
            "pre_broadcast",
            (
                torch.zeros(2, 4, dtype=torch.bfloat16),
                torch.zeros(8, 8),
                torch.zeros(3),
                torch.zeros(8),
                1e-6,
                1e-6,
                1e-6,
                2.0,
                5,
            ),
            "mhc_pre_broadcast_tilelang",
        ),
        (
            "post",
            (
                torch.zeros(2, 4, dtype=torch.bfloat16),
                torch.zeros(2, 2, 4, dtype=torch.bfloat16),
                torch.zeros(2, 2, 1),
                torch.zeros(2, 2, 2),
            ),
            "mhc_post_tilelang",
        ),
        (
            "post_pre",
            (
                torch.zeros(2, 4, dtype=torch.bfloat16),
                torch.zeros(2, 2, 4, dtype=torch.bfloat16),
                torch.zeros(2, 2, 1),
                torch.zeros(2, 2, 2),
                torch.zeros(8, 8),
                torch.zeros(3),
                torch.zeros(8),
                1e-6,
                1e-6,
                1e-6,
                2.0,
                5,
            ),
            "mhc_fused_post_pre_tilelang",
        ),
        (
            "head",
            (
                torch.zeros(2, 2, 4, dtype=torch.bfloat16),
                torch.zeros(2, 8),
                torch.zeros(1),
                torch.zeros(2),
                1e-6,
                1e-6,
            ),
            "hc_head_fused_kernel_tilelang",
        ),
    ],
)
def test_mhc_adapters_pass_exact_arguments(monkeypatch, kind, args, entry):
    sentinel = object()
    call = _patch_symbol(monkeypatch, entry, sentinel)
    assert adapters.MHCTileLangAdapter(kind)(*args) is sentinel
    call.assert_called_once_with(*args)


def test_mhc_validation_and_backward_guard(monkeypatch):
    adapter = adapters.MHCTileLangAdapter("post")
    valid = (
        torch.zeros(2, 4, dtype=torch.bfloat16),
        torch.zeros(2, 2, 4, dtype=torch.bfloat16),
        torch.zeros(2, 2, 1),
        torch.zeros(2, 2, 2),
    )
    with pytest.raises(TypeError, match="dtype"):
        adapter(valid[0].float(), *valid[1:])
    _patch_symbol(monkeypatch, "mhc_post_tilelang", valid[0] + 1)
    output = adapter(valid[0].requires_grad_(), *valid[1:])
    with pytest.raises(NotImplementedError, match="inference-only"):
        output.sum().backward()


def test_fused_qkv_norm_passes_exact_arguments(monkeypatch):
    q, kv = torch.zeros(3, 4), torch.zeros(3, 6)
    qw, kw = torch.ones(4), torch.ones(6)
    expected = (q.clone(), kv.clone())
    call = _patch_symbol(monkeypatch, "fused_q_kv_rmsnorm", expected)
    actual = adapters.FusedQKVRMSNormAdapter()(q, kv, qw, kw, 1e-5)
    assert len(actual) == len(expected)
    assert all(a is e for a, e in zip(actual, expected, strict=True))
    call.assert_called_once_with(q, kv, qw, kw, 1e-5)


def test_fused_qkv_norm_rejects_shape_and_backward(monkeypatch):
    adapter = adapters.FusedQKVRMSNormAdapter()
    with pytest.raises(ValueError, match="token"):
        adapter(torch.zeros(2, 4), torch.zeros(3, 4), torch.ones(4), torch.ones(4), 1e-6)
    q = torch.zeros(2, 4, requires_grad=True)
    _patch_symbol(monkeypatch, "fused_q_kv_rmsnorm", (q + 1, torch.ones_like(q)))
    output, _ = adapter(q, torch.zeros(2, 4), torch.ones(4), torch.ones(4), 1e-6)
    with pytest.raises(NotImplementedError, match="inference-only"):
        output.sum().backward()


def _kv_inputs(cache):
    return dict(
        q=torch.zeros(2, 3, 4, dtype=torch.bfloat16),
        kv=torch.zeros(2, 4, dtype=torch.bfloat16),
        cache=cache,
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        positions=torch.tensor([0, 1], dtype=torch.int64),
        cos_sin_cache=torch.zeros(16, 8, dtype=torch.float32),
        eps=1e-6,
        block_size=2,
    )


def test_kv_insert_bf16_uses_only_real_op_name(monkeypatch):
    inputs = _kv_inputs(torch.zeros(2, 2, 4, dtype=torch.bfloat16))
    call = _patch_op(
        monkeypatch,
        "_C",
        "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert",
    )
    result = adapters.DS4KVInsertAdapter("plain_bf16")(**inputs)
    assert result is inputs["q"]
    call.assert_called_once()


def test_kv_insert_fp8_ds_mla_out_passes_output(monkeypatch):
    inputs = _kv_inputs(torch.zeros(4, 8, dtype=torch.uint8))
    q_out = torch.empty(2, 4, 4, dtype=torch.bfloat16)
    call = _patch_op(
        monkeypatch,
        "_C",
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out",
    )
    result = adapters.DS4KVInsertAdapter("fp8_ds_mla")(
        **inputs, padded_heads=4, q_out=q_out
    )
    assert result is q_out
    assert call.call_args.args[2] is q_out


def test_kv_insert_fp8_plain_requires_scales_and_exact_layout():
    inputs = _kv_inputs(torch.zeros(2, 2, 4, dtype=torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="kv_scale"):
        adapters.DS4KVInsertAdapter("plain_fp8_e4m3")(
            **inputs, q_out=torch.empty_like(inputs["q"], dtype=torch.float8_e4m3fn)
        )
    with pytest.raises(ValueError):
        adapters.DS4KVInsertAdapter("unknown")


def test_kv_insert_allows_forward_then_rejects_backward(monkeypatch):
    inputs = _kv_inputs(torch.zeros(2, 2, 4, dtype=torch.bfloat16))
    inputs["q"].requires_grad_()
    call = _patch_op(
        monkeypatch,
        "_C",
        "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert",
    )
    output = adapters.DS4KVInsertAdapter("plain_bf16")(**inputs)
    call.assert_called_once()
    assert torch.equal(output, inputs["q"])
    with pytest.raises(NotImplementedError, match="inference-only"):
        output.sum().backward()


def test_flashmla_sparse_is_context_free_and_exact(monkeypatch):
    q = torch.zeros(2, 4, 8)
    kv = torch.zeros(16, 1, 8)
    indices = torch.zeros(2, 1, 4, dtype=torch.int32)
    out = torch.empty_like(q)
    call = _patch_symbol(monkeypatch, "flash_mla_sparse_fwd", "ok")
    assert (
        adapters.FlashMLAAdapter().sparse(
            q, kv, indices, sm_scale=0.5, out=out
        )
        == "ok"
    )
    call.assert_called_once_with(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=0.5,
        attn_sink=None,
        topk_length=None,
        out=out,
    )


def test_flashmla_paged_requires_explicit_scheduler_metadata():
    q = torch.zeros(2, 1, 64, 8)
    cache = torch.zeros(4, 2, 1, 8)
    with pytest.raises(NotImplementedError, match="tile_scheduler_metadata"):
        adapters.FlashMLAAdapter().paged(
            q,
            cache,
            tile_scheduler_metadata=None,
            indices=torch.zeros(2, 1, 4, dtype=torch.int32),
            topk_length=torch.ones(2, dtype=torch.int32),
            softmax_scale=0.5,
            attn_sink=torch.zeros(64),
            out=torch.empty_like(q),
        )


def test_flashmla_backward_is_rejected_at_output(monkeypatch):
    q = torch.zeros(2, 4, 8, requires_grad=True)
    _patch_symbol(monkeypatch, "flash_mla_sparse_fwd", q + 1)
    output = adapters.FlashMLAAdapter().sparse(
        q,
        torch.zeros(16, 1, 8),
        torch.zeros(2, 1, 4, dtype=torch.int32),
        sm_scale=0.5,
    )
    with pytest.raises(NotImplementedError, match="inference-only"):
        output.sum().backward()


def test_gate_linear_explicit_callable_and_validation():
    gate = Mock(return_value=(torch.ones(2, 6), None))
    hidden = torch.zeros(2, 4)
    assert adapters.GateLinearAdapter()(gate, hidden).shape == (2, 6)
    gate.assert_called_once_with(hidden)
    output = adapters.GateLinearAdapter()(gate, hidden.requires_grad_())
    with pytest.raises(NotImplementedError, match="inference-only"):
        output.sum().backward()


def test_dsv4_topk_passes_exact_arguments(monkeypatch):
    logits = torch.zeros(2, 256)
    bias = torch.zeros(256)
    expected = (torch.zeros(2, 6), torch.zeros(2, 6, dtype=torch.int64))
    call = _patch_symbol(monkeypatch, "dsv4_topk", expected)
    actual = adapters.DS4TopKAdapter()(
        logits,
        bias,
        indices_dtype=torch.int64,
        routed_scaling_factor=2.5,
    )
    assert len(actual) == len(expected)
    assert all(a is e for a, e in zip(actual, expected, strict=True))
    call.assert_called_once_with(logits, bias, torch.int64, 2.5)


def test_dsv4_topk_rejects_non_kernel_shape():
    with pytest.raises(ValueError, match="256/384"):
        adapters.DS4TopKAdapter()(
            torch.zeros(2, 32),
            torch.zeros(32),
            indices_dtype=torch.int32,
            routed_scaling_factor=1.0,
        )


def test_dsv4_topk_allows_forward_then_rejects_backward(monkeypatch):
    logits = torch.zeros(2, 256, requires_grad=True)
    weights = torch.ones(2, 6)
    ids = torch.zeros(2, 6, dtype=torch.int32)
    call = _patch_symbol(monkeypatch, "dsv4_topk", (weights, ids))
    output, output_ids = adapters.DS4TopKAdapter()(
        logits,
        torch.zeros(256),
        indices_dtype=torch.int32,
        routed_scaling_factor=1.0,
    )
    call.assert_called_once()
    assert torch.equal(output, weights)
    assert torch.equal(output_ids, ids)
    with pytest.raises(NotImplementedError, match="inference-only"):
        output.sum().backward()


def test_hash_route_uses_actual_custom_op_signature(monkeypatch):
    call = _patch_symbol(monkeypatch, "topk_hash_softplus_sqrt")
    logits = torch.zeros(2, 8)
    tokens = torch.tensor([1, 2], dtype=torch.int64)
    table = torch.zeros(16, 2, dtype=torch.int64)
    weights, ids = adapters.HashRouteAdapter()(
        logits, tokens, table, topk=2, routed_scaling_factor=1.25
    )
    assert weights.shape == ids.shape == (2, 2)
    assert call.call_args.args[3] is logits
    assert call.call_args.args[7] is tokens
    assert call.call_args.args[8] is table


def test_hash_route_rejects_implicit_dtype_conversion():
    with pytest.raises(TypeError, match="dtypes"):
        adapters.HashRouteAdapter()(
            torch.zeros(2, 8),
            torch.tensor([1, 2], dtype=torch.int32),
            torch.zeros(16, 2, dtype=torch.int64),
            topk=2,
        )


def test_hash_route_allows_forward_then_rejects_backward(monkeypatch):
    call = _patch_symbol(monkeypatch, "topk_hash_softplus_sqrt")
    logits = torch.zeros(2, 8, requires_grad=True)
    weights, _ = adapters.HashRouteAdapter()(
        logits,
        torch.tensor([1, 2], dtype=torch.int64),
        torch.zeros(16, 2, dtype=torch.int64),
        topk=2,
    )
    call.assert_called_once()
    with pytest.raises(NotImplementedError, match="inference-only"):
        weights.sum().backward()


def test_local_fused_experts_uses_explicit_weights(monkeypatch):
    x = torch.zeros(3, 4)
    w1 = torch.zeros(2, 8, 4)
    w2 = torch.zeros(2, 4, 4)
    tw = torch.ones(3, 1)
    ti = torch.zeros(3, 1, dtype=torch.int32)
    call = _patch_symbol(monkeypatch, "fused_experts", x.clone())
    result = adapters.FusedExpertsAdapter()(
        x, w1, w2, tw, ti, activation="silu"
    )
    assert result.shape == x.shape
    call.assert_called_once_with(
        x,
        w1,
        w2,
        tw,
        ti,
        activation="silu",
        apply_router_weight_on_input=False,
        global_num_experts=-1,
        expert_map=None,
    )


def test_local_fused_experts_allows_forward_then_rejects_backward(monkeypatch):
    x = torch.zeros(3, 4, requires_grad=True)
    call = _patch_symbol(monkeypatch, "fused_experts", x + 1)
    output = adapters.FusedExpertsAdapter()(
        x,
        torch.zeros(2, 8, 4),
        torch.zeros(2, 4, 4),
        torch.ones(3, 1),
        torch.zeros(3, 1, dtype=torch.int32),
        activation="silu",
    )
    call.assert_called_once()
    assert torch.equal(output, x + 1)
    with pytest.raises(NotImplementedError, match="inference-only"):
        output.sum().backward()


def test_shared_expert_is_explicit_and_backward_closed():
    shared = Mock(return_value=(torch.ones(2, 4), None))
    x = torch.zeros(2, 4)
    assert adapters.SharedExpertsAdapter()(shared, x).shape == x.shape
    output = adapters.SharedExpertsAdapter()(shared, x.requires_grad_())
    with pytest.raises(NotImplementedError, match="inference-only"):
        output.sum().backward()


class _FakeDeepEP:
    def __init__(self):
        self.dispatch = Mock(return_value=("ht",))
        self.combine = Mock(return_value=("combined",))
        self.low_latency_dispatch = Mock(return_value=("ll",))
        self.low_latency_combine = Mock(return_value=("ll-combined",))


def test_deepep_ht_explicit_handle_group_and_passthrough():
    handle = _FakeDeepEP()
    group = SimpleNamespace(rank=0, world_size=2)
    adapter = adapters.DeepEPAdapter(handle, group, "high_throughput")
    x = torch.zeros(2, 4)
    ids = torch.zeros(2, 1, dtype=torch.int64)
    assert adapter.dispatch(x, ids, handle=None) == ("ht",)
    handle.dispatch.assert_called_once_with(x=x, topk_idx=ids, handle=None)
    assert adapter.combine(x, "dispatch-handle") == ("combined",)


def test_deepep_ll_requires_explicit_metadata_and_handle():
    adapter = adapters.DeepEPAdapter(
        _FakeDeepEP(), SimpleNamespace(rank=0, world_size=2), "low_latency"
    )
    x = torch.zeros(2, 4)
    ids = torch.zeros(2, 1, dtype=torch.int64)
    with pytest.raises(ValueError, match="missing"):
        adapter.dispatch(x, ids)
    with pytest.raises(ValueError, match="topk"):
        adapter.combine(x, object())
    with pytest.raises(ValueError, match="handle"):
        adapter.combine(
            x, None, topk_ids=ids, topk_weights=torch.ones(2, 1)
        )


def test_deepep_backward_guard():
    handle = _FakeDeepEP()
    handle.dispatch = Mock(return_value=torch.ones(2, 4))
    adapter = adapters.DeepEPAdapter(
        handle, SimpleNamespace(rank=0, world_size=2), "high_throughput"
    )
    output = adapter.dispatch(
        torch.zeros(2, 4, requires_grad=True),
        torch.zeros(2, 1, dtype=torch.int64),
    )
    with pytest.raises(NotImplementedError, match="inference-only"):
        output.sum().backward()


def _real_vllm_available():
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("vllm") is not None
        and importlib.util.find_spec("flash_mla") is not None
    )


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _real_vllm_available(),
    reason="requires CUDA plus matching vLLM and FlashMLA compiled dependencies",
)
def test_gpu_real_entry_parity_contract():
    """Real-environment gate: individual kernels need checkpoint-specific legal shapes.

    This test is intentionally a skip outside the production image rather than
    substituting torch numerics for any bitwise parity assertion.
    """
    pytest.skip(
        "requires production DS4 legal-shape fixtures and metadata workspace; "
        "run the vLLM parity suite in the DS4 image"
    )


@pytest.mark.gpus(2)
@pytest.mark.skipif(
    importlib.util.find_spec("deep_ep") is None or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices and DeepEP",
)
def test_deepep_two_rank_dispatch_combine_skeleton():
    pytest.skip(
        "requires a torchrun-created two-rank process group and DeepEP Buffer; "
        "the adapter deliberately does not create distributed global state"
    )
