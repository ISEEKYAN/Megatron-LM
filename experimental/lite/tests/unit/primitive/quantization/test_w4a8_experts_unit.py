# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""W4A8 routed-expert primitive: rollout numerics contract, STE and opt-in surface.

The rollout contract is vLLM's ``DeepGemmFP4Experts`` path. The activation
quantizer is pinned against independent transcriptions of the two vLLM kernels
that produce the A8 operand (``per_token_group_quant.cu`` packed-register and
generic kernels), compared byte for byte.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4
from megatron.lite.primitive.quantization.qat import QATSpec, apply_qat_to_chunks
from megatron.lite.primitive.quantization.w4a8_experts import (
    dequantize_fp8_act,
    quantize_fp8_act,
    w4a8_expert_mlp,
    w4a8_grouped_gemm,
)

pytestmark = pytest.mark.mlite

LITE_ROOT = Path(__file__).resolve().parents[4]
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# --- vLLM reference transcriptions (csrc/.../fp8/per_token_group_quant.cu) ---


def _vllm_packed_register_kernel(x: torch.Tensor):
    """``per_token_group_quant_8bit_packed_register_kernel`` (group 128, UE8M0).

    ``local_absmax = eps``; ``y_s = fmaxf(absmax / 448, 1e-10f)``; scale byte =
    exponent field + (mantissa != 0); quantize ``x * (1 / 2^e)`` saturated to
    +-448 and converted to E4M3 round-to-nearest-even.
    """
    out_codes, out_bytes = [], []
    for row in x.float():
        codes_row, bytes_row = [], []
        for group in row.reshape(-1, 128):
            absmax = torch.maximum(group.abs().max(), torch.tensor(1e-10))
            y_s = torch.maximum(absmax / 448.0, torch.tensor(1e-10))
            bits = int(y_s.view(torch.int32))
            exp_byte = ((bits >> 23) & 0xFF) + (1 if bits & 0x7FFFFF else 0)
            inv = 1.0 / torch.tensor(exp_byte << 23, dtype=torch.int32).view(
                torch.float32
            )
            q = (group * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            codes_row.append(q.view(torch.uint8))
            bytes_row.append(exp_byte)
        out_codes.append(torch.cat(codes_row))
        out_bytes.append(bytes_row)
    return torch.stack(out_codes), torch.tensor(out_bytes, dtype=torch.int32)


def _vllm_generic_kernel_ue8m0(x: torch.Tensor):
    """``ComputeGroupScale<SCALE_UE8M0=true>`` + ``QuantizeGroup``.

    ``y_s = absmax / 448``; ``y_s = exp2f(ceilf(log2f(fmaxf(|y_s|, 1e-10f))))``;
    ``q = fminf(fmaxf(x / y_s, -448), 448)`` cast to E4M3.
    """
    groups = x.float().reshape(x.shape[0], -1, 128)
    absmax = groups.abs().amax(-1).clamp_min(1e-10)
    y_s = torch.exp2(torch.ceil(torch.log2((absmax / 448.0).clamp_min(1e-10))))
    q = (groups / y_s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(x.shape).view(torch.uint8), y_s


def _scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    return (scale.view(torch.int32) >> 23) & 0xFF


def _activation_cases() -> torch.Tensor:
    torch.manual_seed(0)
    rows = [
        torch.randn(512),
        torch.randn(512) * 1e-3,
        torch.randn(512) * 300.0,
        torch.zeros(512),
        torch.full((512,), 1e-8),  # amax/448 below the 1e-10 scale floor
        torch.full((512,), 448.0 * 4),  # amax / 448 exactly a power of two
        torch.full((512,), 7.0 * 2.0**-20),  # 448 * 2^-26, exact power-of-two scale
        torch.full((512,), 450.0),  # just above 448: scale 2, not 1
        torch.linspace(-3.0e4, 3.0e4, 512),
    ]
    return torch.stack(rows).to(torch.bfloat16)


def test_fp8_act_codes_and_scales_match_vllm_packed_kernel_bytewise():
    x = _activation_cases()
    codes, scale = quantize_fp8_act(x)
    ref_codes, ref_bytes = _vllm_packed_register_kernel(x)

    assert codes.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert torch.equal(codes.view(torch.uint8), ref_codes)
    assert torch.equal(_scale_bytes(scale), ref_bytes)


def test_fp8_act_matches_vllm_generic_ue8m0_kernel_on_bf16_inputs():
    """BF16 inputs never land within one ulp of a power-of-two scale, so the
    float ``ceil(log2)`` kernel and the bit-math kernel agree there."""
    torch.manual_seed(1)
    x = torch.cat([_activation_cases(), torch.randn(64, 512).to(torch.bfloat16)])
    codes, scale = quantize_fp8_act(x)
    ref_codes, ref_scale = _vllm_generic_kernel_ue8m0(x)

    assert torch.equal(codes.view(torch.uint8), ref_codes)
    assert torch.equal(scale, ref_scale)


def test_fp8_act_scale_edges_are_locked():
    x = _activation_cases()
    _, scale = quantize_fp8_act(x)

    assert torch.all(scale[3] == 2.0**-33)  # zero row: scale floor 1e-10 -> 2^-33
    assert torch.all(scale[4] == 2.0**-33)
    assert torch.all(scale[5] == 4.0)  # 448*4 maps to exactly 2^2, not 2^3
    assert torch.all(scale[6] == 2.0**-26)
    assert torch.all(scale[7] == 2.0)
    assert torch.all(scale == torch.exp2(torch.log2(scale).round()))


def test_fp8_act_floor_is_vllm_not_deepgemm_test_util():
    """DeepGEMM's own test cast floors amax at 1e-4; the rollout (vLLM) floors at 1e-10."""
    x = torch.full((1, 128), 1e-8, dtype=torch.bfloat16)
    codes, scale = quantize_fp8_act(x)
    deepgemm_util_scale = 2.0 ** math.ceil(math.log2(1e-4 / 448.0))

    assert scale.item() == 2.0**-33
    assert scale.item() != deepgemm_util_scale
    assert torch.equal(dequantize_fp8_act(codes, scale), codes.float() * scale)


def test_fp8_act_dequantization_is_exact_in_bf16():
    torch.manual_seed(2)
    x = (torch.randn(8, 256) * 5).to(torch.bfloat16)
    x_hat = dequantize_fp8_act(*quantize_fp8_act(x))
    assert torch.equal(x_hat.to(torch.bfloat16).float(), x_hat)
    assert not torch.equal(x_hat, x.float())  # quantization is not the identity


@pytest.mark.parametrize("shape", [(4, 100), (4,), (2, 2, 128)])
def test_fp8_act_rejects_non_group_shapes(shape):
    with pytest.raises(ValueError, match="divisible by 128"):
        quantize_fp8_act(torch.zeros(shape, dtype=torch.bfloat16))


# --- grouped GEMM ----------------------------------------------------------


def _decode_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    raw = packed.view(torch.uint8).long()
    nibbles = torch.stack([raw & 0xF, raw >> 4], dim=-1).flatten(-2)
    table = torch.tensor(_E2M1 + tuple(-v for v in _E2M1))
    exponent = scale.view(torch.uint8).float() - 127.0
    return table[nibbles] * torch.exp2(exponent).repeat_interleave(32, dim=-1)


def _problem(seed=3, m_splits=(5, 0, 11), n=64, k=256):
    torch.manual_seed(seed)
    x = torch.randn(sum(m_splits), k).to(torch.bfloat16)
    weights = [(torch.randn(n, k) * 0.05).to(torch.bfloat16) for _ in m_splits]
    return x, weights, list(m_splits)


def test_grouped_gemm_reference_multiplies_the_rollout_operands():
    x, weights, m_splits = _problem()
    out = w4a8_grouped_gemm(x, weights, m_splits, backend="reference")

    x_codes, x_scale = quantize_fp8_act(x)
    x_hat = x_codes.float() * x_scale.repeat_interleave(128, dim=-1)
    start, expected = 0, []
    for count, weight in zip(m_splits, weights):
        w_hat = _decode_mxfp4(*quantize_mxfp4(weight))
        expected.append(x_hat[start : start + count] @ w_hat.t())
        start += count
    expected = torch.cat(expected).to(torch.bfloat16)

    assert out.dtype == torch.bfloat16
    assert torch.equal(out, expected)
    bf16_out = torch.cat(
        [
            x[s : s + c].float() @ w.float().t()
            for s, c, w in zip((0, 5, 5), m_splits, weights)
        ]
    ).to(torch.bfloat16)
    assert not torch.equal(out, bf16_out)


def _load_qat_exporter():
    path = LITE_ROOT / "examples" / "verl" / "verl_mlite" / "qat_export.py"
    spec = importlib.util.spec_from_file_location("_w4a8_test_qat_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forward_weight_bytes_equal_the_exported_rollout_weight():
    x, weights, m_splits = _problem()
    masters = [weight.clone().requires_grad_() for weight in weights]
    out = w4a8_grouped_gemm(x, masters, m_splits, backend="reference")
    _, _, w_codes, w_scale = out.grad_fn.saved_tensors

    exporter = _load_qat_exporter()
    config = {"enable": True, "apply_modelopt_fake_quant": False, "mode": "mxfp4"}
    for expert, weight in enumerate(weights):
        name = f"model.layers.0.mlp.experts.{expert}.down_proj.weight"
        exported = dict(exporter.export_qat_weights(iter([(name, weight)]), config))
        assert torch.equal(w_codes[expert].view(torch.uint8), exported[name])
        assert torch.equal(
            w_scale[expert].view(torch.uint8),
            exported[name.removesuffix(".weight") + ".weight_scale"],
        )


def test_grouped_gemm_backward_is_ste_on_dequantized_operands():
    x, weights, m_splits = _problem()
    x = x.requires_grad_()
    weights = [w.requires_grad_() for w in weights]
    torch.manual_seed(4)
    grad_out = torch.randn(x.shape[0], weights[0].shape[0]).to(torch.bfloat16)
    w4a8_grouped_gemm(x, weights, m_splits, backend="reference").backward(grad_out)

    x_codes, x_scale = quantize_fp8_act(x.detach())
    x_hat = dequantize_fp8_act(x_codes, x_scale).to(torch.bfloat16)
    start = 0
    for count, weight in zip(m_splits, weights):
        w_hat = _decode_mxfp4(*quantize_mxfp4(weight.detach())).to(torch.bfloat16)
        grad = grad_out[start : start + count]
        rows = slice(start, start + count)
        assert torch.equal(x.grad[rows], grad @ w_hat)
        assert torch.equal(weight.grad, grad.t() @ x_hat[rows])
        assert weight.grad.dtype == torch.bfloat16
        if count:
            # STE through the quantized operands, not plain BF16 gradients.
            assert not torch.equal(x.grad[rows], grad @ weight.detach())
            assert not torch.equal(weight.grad, grad.t() @ x.detach()[rows])
        start += count
    assert torch.count_nonzero(weights[1].grad) == 0  # expert with no tokens


def test_grouped_gemm_validates_inputs():
    x, weights, m_splits = _problem()
    with pytest.raises(TypeError, match="BF16"):
        w4a8_grouped_gemm(x.float(), weights, m_splits)
    with pytest.raises(ValueError, match="m_splits"):
        w4a8_grouped_gemm(x, weights, [5, 11])
    with pytest.raises(ValueError, match="Unknown W4A8 backend"):
        w4a8_grouped_gemm(x, weights, m_splits, backend="bf16")


def test_deep_gemm_backend_fails_closed_without_kernel(monkeypatch):
    """Both DeepGEMM sources unimportable -> error, never another GEMM.

    The import failure is injected, so this runs whether or not DeepGEMM is
    installed in the test environment.
    """
    from megatron.lite.primitive.quantization import w4a8_experts

    tried = []

    def unavailable(name):
        tried.append(name)
        raise ImportError(name)

    monkeypatch.setattr(
        w4a8_experts, "importlib", SimpleNamespace(import_module=unavailable)
    )
    x, weights, m_splits = _problem()
    with pytest.raises(RuntimeError, match="require DeepGEMM"):
        w4a8_grouped_gemm(x, weights, m_splits, backend="deep_gemm")
    assert tried == ["deep_gemm", "vllm.third_party.deep_gemm"]


# --- expert MLP ---------------------------------------------------------------


def test_expert_mlp_follows_rollout_operation_order():
    torch.manual_seed(5)
    m_splits = [3, 6]
    x = torch.randn(9, 128).to(torch.bfloat16)
    fc1 = [(torch.randn(256, 128) * 0.1).to(torch.bfloat16) for _ in m_splits]
    fc2 = [(torch.randn(128, 128) * 0.1).to(torch.bfloat16) for _ in m_splits]
    probs = torch.rand(9, 1)

    out = w4a8_expert_mlp(x, fc1, fc2, m_splits, probs, backend="reference")

    fc1_out = w4a8_grouped_gemm(x, fc1, m_splits, backend="reference")
    gate, up = fc1_out.float().chunk(2, dim=-1)
    hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    fc2_out = w4a8_grouped_gemm(hidden, fc2, m_splits, backend="reference")
    expected = (fc2_out.float() * probs).to(torch.bfloat16)
    assert torch.equal(out, expected)

    # Applying the router weight before FC2 (the BF16 path) is a different contract.
    weighted_hidden = (torch.nn.functional.silu(gate) * up * probs).to(torch.bfloat16)
    probs_first = w4a8_grouped_gemm(weighted_hidden, fc2, m_splits, backend="reference")
    assert not torch.equal(out, probs_first)


def test_expert_mlp_swiglu_limit_clamps_gate_and_up():
    torch.manual_seed(6)
    x = (torch.randn(4, 128) * 4).to(torch.bfloat16)
    fc1 = [torch.randn(256, 128).to(torch.bfloat16)]
    fc2 = [(torch.randn(128, 128) * 0.1).to(torch.bfloat16)]
    limited = w4a8_expert_mlp(x, fc1, fc2, [4], None, 1.0, backend="reference")

    gate, up = w4a8_grouped_gemm(x, fc1, [4], backend="reference").float().chunk(2, -1)
    hidden = (torch.nn.functional.silu(gate.clamp(max=1.0)) * up.clamp(-1.0, 1.0)).to(
        torch.bfloat16
    )
    expected = w4a8_grouped_gemm(hidden, fc2, [4], backend="reference")
    assert torch.equal(limited, expected)
    assert not torch.equal(limited, w4a8_expert_mlp(x, fc1, fc2, [4], None))


def test_expert_mlp_gradients_reach_input_weights_and_probs():
    torch.manual_seed(7)
    x = torch.randn(6, 128).to(torch.bfloat16).requires_grad_()
    fc1 = [(torch.randn(256, 128) * 0.1).to(torch.bfloat16).requires_grad_()]
    fc2 = [(torch.randn(128, 128) * 0.1).to(torch.bfloat16).requires_grad_()]
    probs = torch.rand(6, 1, requires_grad=True)
    w4a8_expert_mlp(x, fc1, fc2, [6], probs).float().square().sum().backward()

    for tensor in (x, fc1[0], fc2[0], probs):
        assert tensor.grad is not None and torch.count_nonzero(tensor.grad) > 0


# --- opt-in surface -------------------------------------------------------------


def test_qat_spec_accepts_only_mxfp4_a8():
    spec = QATSpec(enabled=True, format="mxfp4", activation_bits=8)
    assert spec.activation_bits == 8
    for kwargs in (
        {"format": "fp8", "activation_bits": 8},
        {"format": "int8", "activation_bits": 8},
        {"format": "mxfp4", "activation_bits": 4},
    ):
        with pytest.raises(ValueError, match="activation quantization"):
            QATSpec(enabled=True, **kwargs)


def test_apply_qat_rejects_activation_bits_unless_experts_are_wired():
    spec = QATSpec(enabled=True, format="mxfp4", activation_bits=8)
    with pytest.raises(ValueError, match="wires W4A8 routed experts"):
        apply_qat_to_chunks([torch.nn.Linear(32, 32)], spec)

    stats = apply_qat_to_chunks([torch.nn.Linear(32, 32)], spec, w4a8_experts=True)
    assert stats["quantized_modules"] == 1  # dense linears stay weight-only MXFP4


def test_quantization_package_does_not_import_w4a8_module():
    code = (
        "import sys\n"
        "import megatron.lite.primitive.quantization as q\n"
        "from megatron.lite.primitive.quantization.qat import QATSpec, apply_qat_to_chunks\n"
        "QATSpec(enabled=True, format='mxfp4', activation_bits=8)\n"
        "assert 'megatron.lite.primitive.quantization.w4a8_experts' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=LITE_ROOT, check=True)


def _require_deep_gemm():
    from megatron.lite.primitive.quantization import w4a8_experts

    try:
        w4a8_experts._import_deep_gemm()
    except RuntimeError:
        pytest.skip("DeepGEMM (installed or vLLM-vendored) is not available")


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_deep_gemm_matches_reference_on_identical_operands():
    """Same quantized operands through DeepGEMM and the FP32 reference GEMM.

    Bitwise: E4M3 x E2M1 products carry at most 6 significant bits, so the
    FP32 partial sums are exact here and the BF16 outputs coincide (measured on
    GB200 with vLLM 0979892's DeepGEMM). Includes an all-zero MXFP4 block
    (scale byte 0), which crashed the kernel when passed as a 2^-127 denormal.
    """
    _require_deep_gemm()
    x, weights, m_splits = _problem(m_splits=(130, 0, 7, 300), n=256, k=512)
    weights[0][:, :32] = 0.0
    x, weights = x.cuda(), [w.cuda() for w in weights]
    out = w4a8_grouped_gemm(x, weights, m_splits, backend="deep_gemm")
    ref = w4a8_grouped_gemm(x, weights, m_splits, backend="reference")

    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    assert torch.equal(out.cpu(), ref.cpu())


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_expert_mlp_matches_vllm_deepgemm_fp4_experts_bitwise():
    """Training W4A8 expert MLP vs the rollout's ``DeepGemmFP4Experts.apply``.

    The rollout side is vLLM's own code end to end: A1 quantization as its
    prepare step does it, MXFP4 weight-scale packing as its weight loading does
    it, then permute, FC1, fused SiLU+A2 quantization, FC2 and the FP32 top-k
    gather. With top-k 1 there is no cross-expert sum, so the outputs must be
    bitwise equal (top-k > 1 differs by the documented BF16 combine).
    """
    _require_deep_gemm()
    vllm_moe = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe"
    )
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
        compute_aligned_M_and_alignment,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        _pack_deepgemm_mxfp4_scales,
    )
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
    from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout

    experts, hidden, inter, tokens = 4, 512, 256, 300
    torch.manual_seed(9)
    x = torch.randn(tokens, hidden).to(torch.bfloat16).cuda()
    fc1 = [
        (torch.randn(2 * inter, hidden) * 0.05).to(torch.bfloat16).cuda()
        for _ in range(experts)
    ]
    fc2 = [
        (torch.randn(hidden, inter) * 0.05).to(torch.bfloat16).cuda()
        for _ in range(experts)
    ]
    fc1[3][:, :32] = 0.0
    topk_ids = torch.tensor([0, 1, 3]).repeat(tokens)[:tokens, None].int().cuda()
    topk_weights = torch.rand(tokens, 1).cuda()

    # Rollout: vLLM DeepGemmFP4Experts.apply on the same BF16 tensors.
    packed1 = [quantize_mxfp4(w) for w in fc1]
    packed2 = [quantize_mxfp4(w) for w in fc2]
    w1 = torch.stack([codes for codes, _ in packed1])
    w2 = torch.stack([codes for codes, _ in packed2])
    w1_scale, w2_scale = _pack_deepgemm_mxfp4_scales(
        w1,
        w2,
        torch.stack([s for _, s in packed1]),
        torch.stack([s for _, s in packed2]),
    )
    a1q, a1q_scale = moe_kernel_quantize_input(
        x, None, torch.float8_e4m3fn, False, [128, 128]
    )
    m_sum, _ = compute_aligned_M_and_alignment(
        M=tokens,
        num_topk=1,
        local_num_experts=experts,
        alignment=get_mk_alignment_for_contiguous_layout()[0],
        expert_tokens_meta=None,
    )
    impl = vllm_moe.DeepGemmFP4Experts
    rollout_self = SimpleNamespace(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        gemm1_clamp_limit=None,
        _ACT_BLOCK_K=impl._ACT_BLOCK_K,
        _WEIGHT_BLOCK_K=impl._WEIGHT_BLOCK_K,
        adjust_N_for_activation=impl.adjust_N_for_activation,
    )
    rollout_self._act_mul_quant = impl._act_mul_quant.__get__(rollout_self)
    rollout = torch.zeros_like(x)
    impl.apply(
        rollout_self,
        output=rollout,
        hidden_states=a1q,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=experts,
        expert_map=None,
        a1q_scale=a1q_scale,
        a2_scale=None,
        workspace13=torch.zeros(m_sum, 2 * inter, dtype=torch.bfloat16, device="cuda"),
        workspace2=torch.zeros(m_sum, 2 * inter, dtype=torch.bfloat16, device="cuda"),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    # Training: expert-sorted tokens through the W4A8 expert MLP, scattered back.
    order = torch.argsort(topk_ids[:, 0], stable=True)
    m_splits = torch.bincount(topk_ids[:, 0], minlength=experts).tolist()
    rows = w4a8_expert_mlp(x[order], fc1, fc2, m_splits, topk_weights[order])
    training = torch.zeros_like(x).index_copy_(0, order, rows)

    assert m_splits[2] == 0
    assert torch.equal(training.cpu(), rollout.cpu())


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_fp8_act_matches_installed_vllm_kernel():
    """Byte-compare against the installed vLLM per-token-group quant op."""
    fp8_utils = pytest.importorskip(
        "vllm.model_executor.layers.quantization.utils.fp8_utils"
    )
    torch.manual_seed(8)
    x = torch.cat([_activation_cases(), torch.randn(64, 512).to(torch.bfloat16)])
    x = x.cuda()
    codes, scale = quantize_fp8_act(x)
    ref_codes, ref_scale = fp8_utils.per_token_group_quant_fp8(x, 128, use_ue8m0=True)

    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert torch.equal(scale, ref_scale.contiguous())
