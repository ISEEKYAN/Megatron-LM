# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for the independent dense multi-LoRA primitive."""

# isort: off
import json
from pathlib import Path
from types import SimpleNamespace

import megatron.lite.primitive.modules.multi_lora_reference as multi_lora_reference
import pytest
import torch
import torch.nn as nn
from megatron.lite.model.qwen3_moe.lite import multi_lora
from megatron.lite.primitive.ckpt.hf_weights import VLLM_LORA_NAME_PREFIX
from megatron.lite.primitive.modules import multi_lora_kernel
from megatron.lite.primitive.modules.lora import LoraSpec
from megatron.lite.primitive.modules.multi_lora import BatchedLoraDelta
from megatron.lite.primitive.modules.multi_lora_bank import (
    DenseLoraBank,
    NamedLoraBankRegistry,
    export_named_lora_adapter_state,
    load_named_lora_adapter,
    save_named_lora_adapter,
)

# isort: on


def _inputs(dtype=torch.float64):
    x = torch.tensor(
        [[1.0, 2.0], [-1.0, 3.0], [2.0, -2.0]], dtype=dtype, requires_grad=True
    )
    a_bank = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 1.0], [-1.0, 1.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    b_bank = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 1.0], [2.0, 0.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    indices = torch.tensor([0, 1, 1])
    return x, a_bank, b_bank, indices


def test_bgmv_mixed_precision_promotion_covers_all_scratch_dot_boundaries():
    """FP32 scratch must not leave an unpromoted BF16 ``tl.dot`` boundary."""
    source = Path(multi_lora_kernel.multi_lora_bgmv.__file__).read_text()

    assert source.count("_promote_mixed_dot_operands(") == 4
    for kernel_name in (
        "_bgmv_shrink_kernel",
        "_bgmv_expand_kernel",
        "_bgmv_shrink_transpose_kernel",
    ):
        kernel_source = source.split(f"def {kernel_name}", maxsplit=1)[1].split(
            "@triton.autotune", maxsplit=1
        )[0]
        assert "_promote_mixed_dot_operands(x, w)" in kernel_source


def _bf16_ground_truth_result(name, actual, ground_truth, eager):
    """Allow at most one final-output ULP from the FP32 mathematical oracle."""
    actual32 = actual.float()
    ground32, eager32 = ground_truth.float(), eager.float()
    error = (actual32 - ground32).abs()
    ground_bf16 = ground_truth.detach().to(torch.bfloat16)
    ulp = (
        torch.nextafter(ground_bf16, torch.full_like(ground_bf16, float("inf")))
        .float()
        .sub(ground32)
        .abs()
        .clamp_min(torch.finfo(torch.bfloat16).tiny)
    )
    ground_ulp_error = error / ulp
    ground_p99_ulp = torch.quantile(ground_ulp_error, 0.99)
    ground_max_ulp = ground_ulp_error.max()
    eager_p99 = torch.quantile((actual32 - eager32).abs(), 0.99)
    report = (
        f"{name}: fp32_ground_truth_max_ulp={ground_max_ulp.item():.9g} "
        f"p99_ulp={ground_p99_ulp.item():.9g}; "
        f"eager_bf16_p99_diagnostic={eager_p99.item():.9g}"
    )
    return report, ground_max_ulp <= 1


def test_batched_lora_delta_matches_dense_reference_for_sorted_slots():
    x, a_bank, b_bank, indices = _inputs()

    actual = BatchedLoraDelta.apply(x, a_bank, b_bank, indices, 0.25)
    expected = multi_lora_reference.dense_lora_delta_reference(
        x, a_bank, b_bank, indices, 0.25
    )

    torch.testing.assert_close(actual, expected)


def test_batched_lora_delta_backward_matches_reference_for_repeated_slots():
    x, a_bank, b_bank, _ = _inputs()
    indices = torch.tensor([0, 1, 1])
    grad_out = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.0, 4.0]], dtype=x.dtype)

    actual = BatchedLoraDelta.apply(x, a_bank, b_bank, indices, 0.75)
    actual.backward(grad_out)
    actual_grads = tuple(t.grad.detach().clone() for t in (x, a_bank, b_bank))

    ref_x, ref_a, ref_b, _ = _inputs()
    expected = multi_lora_reference.dense_lora_delta_reference(
        ref_x, ref_a, ref_b, indices, 0.75
    )
    expected.backward(grad_out)

    for actual_grad, reference_tensor in zip(
        actual_grads, (ref_x, ref_a, ref_b), strict=True
    ):
        torch.testing.assert_close(actual_grad, reference_tensor.grad)


def test_fp32_backward_reference_matches_float64_autograd_truth():
    """Validate FP32-single-cast backward against random float64 autograd truth."""
    torch.manual_seed(23)
    # Values originate as FP32, then become float64 exactly.  Float64 autograd
    # supplies an independent high-precision derivative truth for those exact
    # inputs; it does not duplicate the oracle's FP32 contraction code.
    x = torch.randn(5, 3, dtype=torch.float32).double().requires_grad_(True)
    a_bank = torch.randn(3, 2, 3, dtype=torch.float32).double().requires_grad_(True)
    b_bank = torch.randn(3, 4, 2, dtype=torch.float32).double().requires_grad_(True)
    indices = torch.tensor([0, 0, 1, 2, 2])
    grad_output = torch.randn(5, 4, dtype=torch.float32).double()
    output = multi_lora_reference.dense_lora_delta_reference(
        x, a_bank, b_bank, indices, 0.75
    )
    output.backward(grad_output)
    expected = tuple(value.grad.detach() for value in (x, a_bank, b_bank))
    actual = multi_lora_reference.dense_lora_backward_reference_fp32_single_cast(
        x.detach(), grad_output, a_bank.detach(), b_bank.detach(), indices, 0.75
    )
    for observed, ground_truth in zip(actual, expected, strict=True):
        # The oracle performs contractions and reductions in FP32, so its
        # intermediate rounding is part of the intended implementation.  It
        # then widens that FP32 result back to the caller's float64 dtype.  Check
        # this exact widening property, then compare both sides as FP32 with
        # torch's standard FP32 tolerance rather than fitting a threshold to
        # this particular sample.  The BF16 contract remains a separate final
        # output single-cast check in the CUDA parity test.
        torch.testing.assert_close(observed, observed.float().double(), rtol=0, atol=0)
        torch.testing.assert_close(observed.float(), ground_truth.float())


def test_dense_lora_reference_passes_float64_gradcheck():
    x, a_bank, b_bank, indices = _inputs()
    assert torch.autograd.gradcheck(
        lambda x_, a_, b_: multi_lora_reference.dense_lora_delta_reference(
            x_, a_, b_, indices, 0.75
        ),
        (x, a_bank, b_bank),
    )


def test_batched_lora_delta_passes_gradcheck():
    x, a_bank, b_bank, _ = _inputs()
    indices = torch.tensor([0, 1, 1])

    assert torch.autograd.gradcheck(
        lambda x_, a_, b_: BatchedLoraDelta.apply(x_, a_, b_, indices, 0.5),
        (x, a_bank, b_bank),
    )


def test_dense_lora_bank_delegates_to_operator_without_copying_tensors():
    x, a_bank, b_bank, indices = _inputs(dtype=torch.float32)
    bank = DenseLoraBank(a_bank, b_bank)

    assert bank.a_bank is a_bank
    assert bank.b_bank is b_bank
    torch.testing.assert_close(
        bank.delta(x, indices, scale=2.0),
        BatchedLoraDelta.apply(x, a_bank, b_bank, indices, 2.0),
    )


@pytest.mark.parametrize(
    ("out_features", "rank", "expected"),
    [(128, 32, False), (256, 8, False), (256, 32, True)],
)
def test_bgmv_fused_selection_has_one_production_predicate(
    out_features, rank, expected
):
    """The public predicate is exactly the production launch decision."""
    assert multi_lora_kernel.use_fused_bgmv(out_features, rank) is expected


@pytest.mark.skipif(
    not torch.cuda.is_available() or not multi_lora_kernel._TRITON_AVAILABLE,
    reason="requires CUDA Triton BGMV",
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(torch.float32, 2e-3, 2e-3), (torch.bfloat16, 8e-2, 5e-2)],
)
@pytest.mark.parametrize(("out_features", "rank"), [(128, 8), (256, 8), (256, 32)])
def test_cuda_triton_bgmv_training_matches_independent_oracle(
    monkeypatch, out_features, rank, dtype, rtol, atol
):
    """Exercise shrink and production fused launch regimes with full backward."""
    torch.manual_seed(7)
    tokens, slots, in_features = 5, 4, 16
    indices = torch.tensor([0, 0, 1, 2, 2], device="cuda", dtype=torch.int64)
    values = [
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for shape in (
            (tokens, in_features),
            (slots, rank, in_features),
            (slots, out_features, rank),
        )
    ]
    x, a_bank, b_bank = values
    backward_calls, saved_hidden_dtypes, backward_hidden_dtypes = 0, [], []
    internal_scratch_dtypes = []
    original_empty = multi_lora_kernel.multi_lora_bgmv.torch.empty
    original_forward = multi_lora_kernel.multi_lora_bgmv.bgmv_fwd
    original_backward = multi_lora_kernel.multi_lora_bgmv.bgmv_bwd

    def captured_empty(*size, **kwargs):
        if size == (tokens, rank):
            internal_scratch_dtypes.append(kwargs["dtype"])
        return original_empty(*size, **kwargs)

    def captured_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        saved_hidden_dtypes.append(result[1].dtype)
        return result

    def counted_backward(*args, **kwargs):
        nonlocal backward_calls
        backward_calls += 1
        backward_hidden_dtypes.append(kwargs["hidden"].dtype)
        return original_backward(*args, **kwargs)

    monkeypatch.setattr(
        multi_lora_kernel.multi_lora_bgmv.torch, "empty", captured_empty
    )
    monkeypatch.setattr(multi_lora_kernel.multi_lora_bgmv, "bgmv_fwd", captured_forward)
    monkeypatch.setattr(multi_lora_kernel.multi_lora_bgmv, "bgmv_bwd", counted_backward)
    actual = BatchedLoraDelta.apply(x, a_bank, b_bank, indices, 0.75)
    reference_values = [value.detach().clone().requires_grad_(True) for value in values]
    reference = multi_lora_reference.dense_lora_delta_reference(
        *reference_values, indices, 0.75
    )
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    grad_out = torch.randn_like(actual)
    actual.backward(grad_out)
    reference.backward(grad_out)
    assert backward_calls == 1
    assert saved_hidden_dtypes == [torch.float32]
    assert backward_hidden_dtypes == [torch.float32]
    assert internal_scratch_dtypes == [torch.float32, torch.float32]
    assert actual.dtype is dtype
    assert [(value.grad.dtype, value.grad.shape) for value in values] == [
        (dtype, x.shape),
        (dtype, a_bank.shape),
        (dtype, b_bank.shape),
    ]
    if dtype is torch.bfloat16:
        fp32_single_cast = (
            multi_lora_reference.dense_lora_backward_reference_fp32_single_cast(
                reference_values[0],
                grad_out,
                reference_values[1],
                reference_values[2],
                indices,
                0.75,
            )
        )
        reports, failures = [], []
    for name, actual_value, reference_value, fp32_value in zip(
        ("grad_input", "grad_A", "grad_B"),
        values,
        reference_values,
        fp32_single_cast if dtype is torch.bfloat16 else (None,) * 3,
        strict=True,
    ):
        if dtype is torch.bfloat16:
            report, passed = _bf16_ground_truth_result(
                name, actual_value.grad, fp32_value, reference_value.grad
            )
            reports.append(report)
            if not passed:
                failures.append(f"{name}: FP32-ground-truth max exceeds one BF16 ULP")
        else:
            torch.testing.assert_close(
                actual_value.grad, reference_value.grad, rtol=rtol, atol=atol
            )
    if dtype is torch.bfloat16:
        # Eager BF16 is intentionally diagnostic-only: its reduction order is
        # backend dependent, while the gate above specifies the math contract.
        print("\n".join(reports))
        assert not failures, "\n".join([*reports, *failures])
    torch.testing.assert_close(a_bank.grad[3], torch.zeros_like(a_bank.grad[3]))
    torch.testing.assert_close(b_bank.grad[3], torch.zeros_like(b_bank.grad[3]))


def _named_registry():
    return NamedLoraBankRegistry(
        banks={
            "model.layers.0.q_proj": DenseLoraBank(
                torch.arange(12, dtype=torch.float32).view(2, 2, 3),
                torch.arange(16, dtype=torch.float32).view(2, 4, 2),
            ),
            "model.layers.0.o_proj": DenseLoraBank(
                torch.arange(12, 24, dtype=torch.float32).view(2, 2, 3),
                torch.arange(16, 32, dtype=torch.float32).view(2, 4, 2),
            ),
        },
        names={"alpha": 0, "bravo": 1},
        rank=2,
        alpha=4,
        base_model_identity={"name": "tiny-qwen", "revision": "abc123"},
        lora_spec=LoraSpec(enabled=True, rank=2, alpha=4, dropout=0.125),
    )


def test_named_registry_exports_only_explicit_slot_as_two_dimensional_peft_state():
    registry = _named_registry()

    state = registry.export_state("bravo")

    assert set(state) == {
        f"{VLLM_LORA_NAME_PREFIX}model.layers.0.q_proj.lora_A.weight",
        f"{VLLM_LORA_NAME_PREFIX}model.layers.0.q_proj.lora_B.weight",
        f"{VLLM_LORA_NAME_PREFIX}model.layers.0.o_proj.lora_A.weight",
        f"{VLLM_LORA_NAME_PREFIX}model.layers.0.o_proj.lora_B.weight",
    }
    torch.testing.assert_close(
        state[f"{VLLM_LORA_NAME_PREFIX}model.layers.0.q_proj.lora_A.weight"],
        registry.banks["model.layers.0.q_proj"].a_bank[1],
    )
    assert (
        state[f"{VLLM_LORA_NAME_PREFIX}model.layers.0.q_proj.lora_A.weight"].ndim == 2
    )
    assert registry.manifest("bravo")["slot"] == 1
    assert registry.manifest("bravo")["name"] == "bravo"


def test_named_registry_export_reuses_hf_mapping_and_scaling_contract():
    """Named multi-LoRA export must take the production HF export route."""
    registry = NamedLoraBankRegistry(
        banks={
            "layers.0.moe.experts.fc1.weight0": DenseLoraBank(
                torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
                torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]),
            )
        },
        names={"alpha": 0},
        rank=2,
        alpha=4,
        base_model_identity={},
        lora_spec=LoraSpec(enabled=True, rank=2, alpha=4, use_rslora=True),
    )

    class Spec:
        num_experts = 1

        @staticmethod
        def tp_spec(name):
            assert name == "layers.0.moe.experts.fc1.weight0"
            return None

        @staticmethod
        def is_expert(name):
            return "experts" in name

        @staticmethod
        def native_to_hf(name, tensor):
            assert name == "layers.0.moe.experts.fc1.weight0"
            gate, up = tensor.chunk(2, dim=0)
            return [
                ("model.layers.0.mlp.experts.0.gate_proj.weight", gate),
                ("model.layers.0.mlp.experts.0.up_proj.weight", up),
            ]

    ps = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        etp_size=1,
        etp_rank=0,
        etp_group=None,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
    )
    state = export_named_lora_adapter_state(registry, "alpha", Spec(), ps)
    prefix = VLLM_LORA_NAME_PREFIX + "model.layers.0.mlp.experts.0"
    assert set(state) == {
        f"{prefix}.gate_proj.lora_A.weight",
        f"{prefix}.gate_proj.lora_B.weight",
        f"{prefix}.up_proj.lora_A.weight",
        f"{prefix}.up_proj.lora_B.weight",
    }
    # vLLM FusedMoE uses alpha/rank, while training chose rsLoRA alpha/sqrt(rank).
    torch.testing.assert_close(
        state[f"{prefix}.gate_proj.lora_B.weight"],
        registry.banks["layers.0.moe.experts.fc1.weight0"].b_bank[0, :2] * 2**0.5,
    )


def test_named_registry_export_gathers_tensor_parallel_output_before_mapping(
    monkeypatch,
):
    """A bank's local TP B rows must never be exported as a partial adapter."""
    registry = NamedLoraBankRegistry(
        banks={
            "layers.0.attn.qkv.linear.weight": DenseLoraBank(
                torch.ones(1, 2, 2), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
            )
        },
        names={"alpha": 0},
        rank=2,
        alpha=2,
        base_model_identity={},
    )

    class Spec:
        num_experts = 0
        tp_spec = staticmethod(lambda name: (0, 0))
        is_expert = staticmethod(lambda name: False)
        native_to_hf = staticmethod(
            lambda name, tensor: [("model.layers.0.self_attn.q_proj.weight", tensor)]
        )

    import megatron.lite.primitive.ckpt.hf_weights as hf_weights

    monkeypatch.setattr(
        hf_weights,
        "allgather_concat",
        lambda tensor, world_size, group, dim: torch.cat(
            (tensor, tensor + 10), dim=dim
        ),
    )
    ps = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        tp_group="tp",
        etp_size=1,
        etp_rank=0,
        etp_group=None,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
    )
    state = export_named_lora_adapter_state(registry, "alpha", Spec(), ps)
    key = f"{VLLM_LORA_NAME_PREFIX}model.layers.0.self_attn.q_proj.lora_B.weight"
    torch.testing.assert_close(
        state[key], torch.tensor([[1.0, 2.0], [3.0, 4.0], [11.0, 12.0], [13.0, 14.0]])
    )


def test_named_registry_rejects_duplicate_or_unknown_slots():
    bank = DenseLoraBank(torch.ones(2, 1, 1), torch.ones(2, 1, 1))
    with pytest.raises(ValueError, match="duplicate"):
        NamedLoraBankRegistry(
            banks={"q_proj": bank},
            names={"alpha": 0, "bravo": 0},
            rank=1,
            alpha=1,
            base_model_identity={},
        )
    with pytest.raises(KeyError, match="Unknown"):
        _named_registry().export_state("not-registered")


def test_named_registry_roundtrip_writes_only_named_slot_and_validates_manifest(
    tmp_path,
):
    source = _named_registry()
    save_named_lora_adapter(source, "bravo", tmp_path)
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["r"] == source.lora_spec.rank
    assert config["lora_alpha"] == source.lora_spec.alpha
    assert config["lora_dropout"] == source.lora_spec.dropout
    assert config["target_modules"] == ["o_proj", "q_proj"]
    destination = _named_registry()
    for bank in destination.banks.values():
        bank.a_bank.zero_()
        bank.b_bank.zero_()

    load_named_lora_adapter(destination, "bravo", tmp_path)

    for surface, source_bank in source.banks.items():
        target_bank = destination.banks[surface]
        torch.testing.assert_close(target_bank.a_bank[1], source_bank.a_bank[1])
        torch.testing.assert_close(target_bank.b_bank[1], source_bank.b_bank[1])
        torch.testing.assert_close(
            target_bank.a_bank[0], torch.zeros_like(target_bank.a_bank[0])
        )
        torch.testing.assert_close(
            target_bank.b_bank[0], torch.zeros_like(target_bank.b_bank[0])
        )

    manifest = source.manifest("bravo")
    manifest["slot"] = 0
    with pytest.raises(ValueError, match="slot"):
        destination.load_state("bravo", source.export_state("bravo"), manifest)


def test_named_registry_import_is_no_grad_and_rejects_shape_mismatch():
    source = _named_registry()
    destination = _named_registry()
    for bank in destination.banks.values():
        bank.a_bank.requires_grad_(True)
        bank.b_bank.requires_grad_(True)

    state = source.export_state("alpha")
    destination.load_state("alpha", state, source.manifest("alpha"))

    for bank in destination.banks.values():
        assert bank.a_bank.grad_fn is None
        assert bank.b_bank.grad_fn is None

    bad_state = dict(state)
    bad_state[f"{VLLM_LORA_NAME_PREFIX}model.layers.0.q_proj.lora_A.weight"] = (
        torch.ones(1, 3)
    )
    with pytest.raises(ValueError, match="shape"):
        destination.load_state("alpha", bad_state, source.manifest("alpha"))


@pytest.mark.parametrize(
    ("bad_indices", "message"),
    [
        (torch.tensor([0, 2, 1]), "out of range"),
        (torch.tensor([-1, 0, 1]), "out of range"),
        (torch.tensor([1, 0, 1]), "monotonically non-decreasing"),
        (torch.tensor([0, 1, 1], dtype=torch.int32), "torch.int64"),
        (torch.tensor([[0, 1, 0]]), "one-dimensional"),
    ],
)
def test_batched_lora_delta_rejects_non_prototype_indices(bad_indices, message):
    x, a_bank, b_bank, _ = _inputs()

    with pytest.raises((IndexError, ValueError), match=message):
        BatchedLoraDelta.apply(x, a_bank, b_bank, bad_indices, 1.0)
    with pytest.raises((IndexError, ValueError), match=message):
        multi_lora_reference.dense_lora_delta_reference(
            x, a_bank, b_bank, bad_indices, 1.0
        )


def test_qwen_moe_sidecar_sorts_then_restores_unsorted_slots():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    a_bank = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    b_bank = torch.tensor([[[2.0]], [[3.0]]])
    slots = torch.tensor([1, 0, 1], dtype=torch.int64)
    bank = DenseLoraBank(a_bank, b_bank)

    actual = multi_lora.apply_dense_lora_delta(bank, x, slots, scale=1.0)
    expected_rows = [
        b_bank[slot] @ (a_bank[slot] @ row) for row, slot in zip(x, slots, strict=True)
    ]
    expected = torch.stack(expected_rows)

    torch.testing.assert_close(actual, expected)


def test_ep_gradient_sync_is_identity_in_forward_and_all_reduces_in_backward(
    monkeypatch,
):
    calls = []
    param = torch.tensor([2.0], requires_grad=True)
    group = object()

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda actual: 2)

    def fake_all_reduce(grad, *, op, group):
        calls.append((grad.clone(), op, group))
        grad.add_(3.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    output = multi_lora._AllReduceGradient.apply(param, group)
    torch.testing.assert_close(output, param)
    output.sum().backward()

    assert len(calls) == 1
    assert calls[0][2] is group
    torch.testing.assert_close(param.grad, torch.tensor([4.0]))


@pytest.mark.parametrize(
    ("etp_size", "use_deepep", "message"), [(2, False, "ETP"), (1, True, "DeepEP")]
)
def test_qwen_moe_sidecar_rejects_unverified_parallel_modes(
    transformer_engine_import_stub, etp_size, use_deepep, message
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite.model import MoELayer

    layer = MoELayer.__new__(MoELayer)
    nn.Module.__init__(layer)
    layer.ps = SimpleNamespace(etp_size=etp_size)
    layer._use_deepep_requested = use_deepep
    sidecar = multi_lora.MoELoraSidecar(
        fc1=DenseLoraBank(torch.ones(1, 1, 2), torch.ones(1, 2, 1)),
        fc2=DenseLoraBank(torch.ones(1, 1, 1), torch.ones(1, 2, 1)),
        lora_indices=torch.zeros(1, dtype=torch.int64),
        scale=1.0,
    )

    with pytest.raises(RuntimeError, match=message):
        layer(torch.ones(1, 2), multi_lora_sidecar=sidecar)
