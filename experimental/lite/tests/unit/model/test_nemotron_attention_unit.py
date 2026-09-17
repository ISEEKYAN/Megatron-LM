"""Packed boundaries and native CP must preserve the frozen attention arithmetic."""

import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.gpus(1)
def test_packed_attention_matches_frozen_oracle_and_separate_requests():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from megatron.lite.model.nemotron_h.attention import packed_attention
    from megatron.lite.model.nemotron_h.mamba import SSMMeta

    from verl.models.transformers.nemotron_h_alignment import (
        _sequence_boundaries,
        attention_forward,
    )

    torch.manual_seed(81)
    q = torch.randn(36, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(36, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    meta = SSMMeta((0, 17, 36))
    output = packed_attention(q, k, v, meta, scale=32**-0.5)
    token = _sequence_boundaries.set(meta.boundaries)
    try:
        expected = attention_forward(
            *(x.transpose(0, 1)[None] for x in (q, k, v)), scale=32**-0.5
        ).squeeze(0)
    finally:
        _sequence_boundaries.reset(token)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    separate = torch.cat(
        [
            packed_attention(
                q[a:b], k[a:b], v[a:b], SSMMeta((0, b - a)), scale=32**-0.5
            )
            for a, b in ((0, 17), (17, 36))
        ]
    )
    torch.testing.assert_close(output, separate, rtol=0, atol=0)
    output.float().square().sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))


def _cp_worker(rank, init_file):
    from megatron.lite.model.nemotron_h.attention import Attention
    from megatron.lite.model.nemotron_h.config import NemotronHConfig
    from megatron.lite.model.nemotron_h.mamba import SSMMeta
    from megatron.lite.primitive.parallel import ParallelState
    from safetensors import safe_open

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        device = torch.device("cuda", rank)
        root = Path(os.environ["NEMOTRON_TEST_MODEL"])
        config = NemotronHConfig.from_hf(root)
        layer = config.layers_block_type.index("full_attention")
        prefix = f"backbone.layers.{layer}.mixer."
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        state = {}
        for filename in {
            value for key, value in index.items() if key.startswith(prefix)
        }:
            with safe_open(
                root / filename, framework="pt", device=str(device)
            ) as handle:
                for key, value in index.items():
                    if value == filename and key.startswith(prefix):
                        state[key.removeprefix(prefix)] = handle.get_tensor(key)
        reference = Attention(config, ParallelState(), device=device)
        candidate = Attention(
            config,
            ParallelState(
                cp_group=torch.distributed.group.WORLD,
                cp_size=2,
                cp_rank=rank,
            ),
            device=device,
        )
        reference.load_state_dict(state, strict=True)
        candidate.load_state_dict(state, strict=True)
        torch.manual_seed(82)
        x = torch.randn(36, config.hidden_size, device=device, dtype=torch.bfloat16)
        x.requires_grad_()
        local_x = x.detach().chunk(2)[rank].clone().requires_grad_()
        meta = SSMMeta((0, 17, 36))
        expected = reference(x, meta)
        actual = candidate(local_x, meta)
        torch.testing.assert_close(actual, expected.chunk(2)[rank], rtol=0, atol=0)
        expected.float().square().sum().backward()
        actual.float().square().sum().backward()
        torch.testing.assert_close(local_x.grad, x.grad.chunk(2)[rank], rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.gpus(2)
def test_real_weight_attention_cp2_matches_cp1(tmp_path):
    if torch.cuda.device_count() < 2 or "NEMOTRON_TEST_MODEL" not in os.environ:
        pytest.skip("Two GPUs and NEMOTRON_TEST_MODEL required")
    torch.multiprocessing.spawn(_cp_worker, args=(str(tmp_path / "init"),), nprocs=2)


@pytest.mark.gpus(1)
def test_residual_norm_matches_frozen_forward_and_vjp():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from megatron.lite.model.nemotron_h.functional import RMSNorm

    from verl.models.transformers.nemotron_h_alignment import residual_rms

    torch.manual_seed(83)
    norm = RMSNorm(2688, 1e-5, device="cuda")
    inputs = [
        torch.randn(17, 2688, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(2)
    ]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    weight = norm.weight.detach().clone().requires_grad_()
    actual = norm(*inputs)
    expected = residual_rms(*refs, weight, 1e-5)
    gradients = tuple(torch.randn_like(x) for x in actual)
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.autograd.backward(actual, gradients)
    torch.autograd.backward(expected, gradients)
    for x, y in zip((*inputs, norm.weight), (*refs, weight), strict=True):
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
