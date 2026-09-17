"""Native dispatcher must preserve vLLM ReLU² routing arithmetic."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _config():
    return SimpleNamespace(
        n_routed_experts=4,
        hidden_size=128,
        moe_intermediate_size=256,
        mlp_bias=False,
        mlp_hidden_act="relu2",
        moe_latent_size=None,
        _experts_implementation="eager",
    )


def _inputs(device, tokens):
    torch.manual_seed(73)
    cfg = _config()
    up = torch.randn(4, 256, 128, device=device, dtype=torch.bfloat16) * 0.03
    down = torch.randn(4, 128, 256, device=device, dtype=torch.bfloat16) * 0.03
    x = torch.randn(tokens, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    ids = torch.stack((torch.arange(tokens) % 4, (torch.arange(tokens) + 2) % 4), 1)
    weights = torch.rand(tokens, 2, device=device)
    weights = weights / weights.sum(1, keepdim=True)
    return x, up, down, ids.to(device).int(), weights


def _real_inputs(device, tokens):
    from megatron.lite.model.nemotron_h.config import NemotronHConfig
    from safetensors import safe_open

    root = Path(os.environ["NEMOTRON_TEST_MODEL"])
    config = NemotronHConfig.from_hf(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = "backbone.layers.1.mixer.experts."
    loaded = {}
    for filename in sorted(
        {value for key, value in index.items() if key.startswith(prefix)}
    ):
        with safe_open(root / filename, framework="pt", device=str(device)) as handle:
            for key, value in index.items():
                if value == filename and key.startswith(prefix):
                    loaded[key] = handle.get_tensor(key)
    up, down = (
        torch.stack(
            [
                loaded[f"{prefix}{expert}.{projection}.weight"]
                for expert in range(config.n_routed_experts)
            ]
        )
        for projection in ("up_proj", "down_proj")
    )
    torch.manual_seed(73)
    x = torch.randn(tokens, config.hidden_size, device=device, dtype=torch.bfloat16)
    ids = (
        (
            torch.arange(tokens, device=device)[:, None] * 7
            + torch.arange(config.num_experts_per_tok, device=device)[None] * 19
        )
        .remainder(config.n_routed_experts)
        .int()
    )
    weights = torch.rand(ids.shape, device=device)
    weights /= weights.sum(1, keepdim=True)
    return config, (x, up, down, ids, weights)


def _check(rank, size, mode="synthetic"):
    from megatron.lite.model.nemotron_h.experts import RoutedExperts, routed_experts
    from megatron.lite.primitive.parallel import ParallelState

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    # Deliberately unequal token counts: native all-to-all must not need padding.
    if mode == "real":
        config, (x, up, down, ids, weights) = _real_inputs(device, 17 + rank * 2)
    else:
        config = _config()
        x, up, down, ids, weights = _inputs(device, 17 + rank * 2)
        if mode == "empty_owner":
            ids = ids.remainder(2)
    ps = ParallelState(ep_size=size, ep_rank=rank)
    if size > 1:
        ps.ep_group = torch.distributed.group.WORLD
    module = RoutedExperts(config, ps, device=device)
    with torch.no_grad():
        module.up_proj.copy_(up.chunk(size)[rank])
        module.down_proj.copy_(down.chunk(size)[rank])
    x.requires_grad_()
    weights.requires_grad_()
    output = module(x, ids, weights)
    with torch.no_grad():
        expected = routed_experts(x, up, down, weights, ids)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.float().square().sum().backward()
    for tensor in (x, weights, module.up_proj, module.down_proj):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.gpus(1)
def test_native_dispatch_relu2_matches_vllm_local():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    _check(0, 1)


@pytest.mark.gpus(1)
def test_expert_vjp_matches_hf_with_fixed_upstream_gradient():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from megatron.lite.model.nemotron_h.experts import routed_experts
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

    x, up, down, ids, weights = _inputs("cuda", 19)
    reference = NemotronHExperts(_config()).to(device="cuda", dtype=torch.bfloat16)
    reference.load_state_dict({"up_proj": up, "down_proj": down}, strict=True)
    candidate_inputs = [
        t.detach().clone().requires_grad_() for t in (x, up, down, weights)
    ]
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    output = routed_experts(*candidate_inputs, ids)
    ref_output = reference(ref_x, ids.long(), ref_weights)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    ref_output.backward(gradient.to(ref_output.dtype))
    for candidate, expected in zip(
        candidate_inputs,
        (ref_x, reference.up_proj, reference.down_proj, ref_weights),
        strict=True,
    ):
        torch.testing.assert_close(candidate.grad, expected.grad, rtol=0, atol=0)


@pytest.mark.gpus(1)
def test_composed_moe_matches_frozen_hf_alignment():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from megatron.lite.model.nemotron_h.experts import MoE
    from megatron.lite.primitive.parallel import ParallelState
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMoE

    from verl.models.transformers.nemotron_h_alignment import (
        install_linear_forward,
        install_transformers_moe_forward,
    )

    config = _config()
    config.num_local_experts = config.n_routed_experts
    config.num_experts_per_tok = 2
    config.n_group = config.topk_group = 1
    config.norm_topk_prob = True
    config.routed_scaling_factor = 2.5
    config.n_shared_experts = 1
    config.moe_shared_expert_intermediate_size = 256
    torch.manual_seed(74)
    reference = NemotronHMoE(config).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for param in reference.parameters():
            param.normal_(std=0.03)
    install_linear_forward(reference)
    install_transformers_moe_forward(reference)
    candidate = MoE(config, ParallelState(), device="cuda")
    candidate.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        torch.testing.assert_close(candidate(x), reference(x), rtol=0, atol=0)


def _worker(rank, init_file, mode):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        _check(rank, 2, mode)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.gpus(2)
@pytest.mark.parametrize("mode", ["synthetic", "empty_owner", "real"])
def test_native_ep2_relu2_unequal_tokens_matches_unsharded(tmp_path, mode):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices required")
    if mode == "real" and "NEMOTRON_TEST_MODEL" not in os.environ:
        pytest.skip("NEMOTRON_TEST_MODEL required")
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "init"), mode), nprocs=2)
