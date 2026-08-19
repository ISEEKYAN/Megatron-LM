from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.lite.model import (
    DeepseekV4Layer as LiteLayer,
    DeepseekV4Model as LiteModel,
)
from megatron.lite.model.deepseek_v4.vllm import runtime_metadata
from megatron.lite.model.deepseek_v4.vllm.model import DeepseekV4Layer, DeepseekV4Model
from megatron.lite.model.deepseek_v4.vllm.moe import _gate_linear
from megatron.lite.model.deepseek_v4.vllm.primitive._recompute import (
    visible_functional_vjp,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.linear import visible_linear
from megatron.lite.model.deepseek_v4.vllm.primitive.router import fixed_route_vjp
from megatron.lite.primitive.quantization import deployment_block_fp8


def test_vllm_path_reuses_lite_model_and_external_abi_owners() -> None:
    assert issubclass(DeepseekV4Model, LiteModel)
    assert issubclass(DeepseekV4Layer, LiteLayer)
    assert runtime_metadata.AttentionKernelMetadata.__module__.startswith(
        "vllm.models.deepseek_v4"
    )
    assert deployment_block_fp8.DeploymentBlockFP8Adapter.__module__.startswith(
        "vllm.models.deepseek_v4"
    )


def test_model_path_has_no_runtime_mode_switches() -> None:
    root = Path(__file__).parents[6] / "megatron/lite/model/deepseek_v4/vllm"
    source = "\n".join(path.read_text() for path in root.rglob("*.py"))
    assert 'os.getenv("VLLM_BATCH_INVARIANT"' not in source
    assert "inference_only" not in source
    assert "dsa_indexer_loss_coeff" not in source


def test_visible_linear_keeps_visible_value_and_bf16_master_vjp() -> None:
    torch.manual_seed(1)
    value = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    visible = lambda x, w: F.linear(x, w) + 7
    output = visible_linear(visible, value, weight)
    assert torch.equal(output, visible(value, weight))
    grad = torch.randn_like(output)
    output.backward(grad)
    expected = grad @ weight.detach(), grad.T @ value.detach()
    torch.testing.assert_close(value.grad, expected[0])
    torch.testing.assert_close(weight.grad, expected[1])


def test_generic_visible_boundary_recomputes_functional_backward() -> None:
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    scale = torch.tensor([3.0, 4.0], requires_grad=True)
    output = visible_functional_vjp(
        lambda a, b: a + b,
        lambda a, b: a * b,
        (x, scale),
    )
    assert torch.equal(output, x + scale)
    output.sum().backward()
    assert torch.equal(x.grad, scale.detach())
    assert torch.equal(scale.grad, x.detach())


def test_fixed_route_backward_uses_visible_active_set() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0]], requires_grad=True)

    def visible(value):
        ids = torch.tensor([[2, 0]])
        scores = torch.sqrt(F.softplus(value.float())).gather(1, ids)
        return scores, ids

    weights, ids = fixed_route_vjp(
        visible, logits, renormalize=False, route_scale=1.0
    )
    assert ids.tolist() == [[2, 0]]
    weights.sum().backward()
    assert logits.grad[0, 1] == 0
    assert logits.grad[0, 0] != 0 and logits.grad[0, 2] != 0


def test_gate_linear_matches_official_training_shape_fallback() -> None:
    if not torch.cuda.is_available():
        pytest.skip("out_dtype mm is a CUDA contract")
    value = torch.randn(17, 8, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(12, 8, dtype=torch.bfloat16, device="cuda")
    assert torch.equal(
        _gate_linear(value, weight),
        torch.mm(value, weight.T, out_dtype=torch.float32),
    )


@pytest.mark.parametrize("name", ["build_prefill_batch", "from_hf"])
def test_metadata_builder_surface_is_owned_by_vllm(name: str) -> None:
    assert callable(getattr(runtime_metadata.DS4PrefillMetadataBuilder, name))
