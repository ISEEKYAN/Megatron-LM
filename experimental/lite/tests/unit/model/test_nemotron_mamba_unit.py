"""Request boundaries, native VJP ownership, and real two-rank CP exchange."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.lite.model.nemotron_h.functional import visible_forward
from megatron.lite.model.nemotron_h.mamba import SSMMeta, exchange_sequence_channels


def test_chunks_restart_at_every_request_not_packed_offset():
    meta = SSMMeta((0, 127, 256, 259))
    assert meta.chunks() == ((0, 127, 255, 256, 259), (0, 2, 3), (0, 1, 1, 2))
    with pytest.raises(ValueError, match="token count"):
        meta.validate_tokens(258)


@pytest.mark.parametrize("boundaries", [(1, 4), (0,), (0, 2, 2), (0, 3, 1)])
def test_invalid_request_boundaries_fail_closed(boundaries):
    with pytest.raises(ValueError):
        SSMMeta(boundaries)


def test_visible_value_and_native_gradient_with_frozen_input():
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    weight = torch.tensor([4.0, 5.0])
    native = lambda a, b: a.square() * b
    visible = lambda a, b: native(a, b) + 0.125
    result = visible_forward(visible, native, x, weight)
    assert torch.equal(result, visible(x, weight))
    upstream = torch.tensor([0.5, -2.0])
    assert torch.equal(
        torch.autograd.grad(result, x, upstream)[0],
        torch.autograd.grad(native(x, weight), x, upstream)[0],
    )


def test_parameter_mutation_before_backward_is_rejected():
    x = torch.tensor([2.0], requires_grad=True)
    result = visible_forward(torch.square, torch.square, x)
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()


@pytest.mark.parametrize("requires_grad", [False, True])
def test_scoring_does_not_call_native_backward_reference(requires_grad):
    x = torch.tensor([2.0], requires_grad=requires_grad)

    def native(_):
        raise AssertionError("Scoring must not build backward intermediates")

    with torch.no_grad():
        assert torch.equal(visible_forward(torch.square, native, x), x.square())


def _cp_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        full = torch.arange(8 * 4 * 2, dtype=torch.float64).reshape(8, 4, 2)
        local = full.chunk(2, dim=0)[rank].clone().requires_grad_()
        headwise = exchange_sequence_channels(local, dist.group.WORLD)
        assert torch.equal(headwise, full.chunk(2, dim=1)[rank])
        # Squaring makes a missing inverse/gradient exchange observable.
        result = exchange_sequence_channels(
            headwise.square(), dist.group.WORLD, reverse=True
        )
        assert torch.equal(result, local.square())
        result.sum().backward()
        assert torch.equal(local.grad, 2 * local)
    finally:
        dist.destroy_process_group()


def test_cp2_exchange_and_backward_match_unsharded_computation(tmp_path):
    mp.spawn(_cp_worker, args=(f"file://{tmp_path}/rendezvous",), nprocs=2, join=True)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("lengths", [(17, 19), (127, 129)])
def test_packed_conv_matches_independent_requests_and_native_vjp(lengths):
    from megatron.lite.model.nemotron_h.mamba import packed_conv
    from transformers.models.nemotron_h.modeling_nemotron_h import causal_conv1d_fn

    causal_conv1d_fn = getattr(causal_conv1d_fn, "__wrapped__", causal_conv1d_fn)
    torch.manual_seed(42)
    first, second = lengths
    meta = SSMMeta((0, first, first + second))
    x = torch.randn(
        first + second, 6144, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(
        6144, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    bias = torch.randn(6144, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = packed_conv(x, weight, bias, meta)
    independent = torch.cat(
        [
            packed_conv(x[a:b], weight, bias, SSMMeta((0, b - a)))
            for a, b in zip(meta.boundaries, meta.boundaries[1:])
        ]
    )
    assert torch.equal(output, independent)
    native = torch.cat(
        [
            causal_conv1d_fn(x[a:b].T.unsqueeze(0), weight, bias, activation="silu")
            .squeeze(0)
            .T
            for a, b in zip(meta.boundaries, meta.boundaries[1:])
        ]
    )
    upstream = torch.randn_like(output)
    actual = torch.autograd.grad(output, (x, weight, bias), upstream)
    expected = torch.autograd.grad(native, (x, weight, bias), upstream)
    for a, b in zip(actual, expected, strict=True):
        assert torch.isfinite(a).all()
        assert torch.equal(a, b)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("lengths", [(17, 19), (127, 129)])
def test_packed_ssd_matches_independent_requests_and_native_vjp(lengths):
    from megatron.lite.model.nemotron_h.mamba import packed_scan
    from transformers.models.nemotron_h.modeling_nemotron_h import mamba2_chunk_scan

    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    native_scan = getattr(mamba2_chunk_scan, "__wrapped__", mamba2_chunk_scan)
    init_batch_invariance()
    torch.manual_seed(42)
    first, second = lengths
    tokens = first + second
    meta = SSMMeta((0, first, tokens))

    def rand(*shape):
        return torch.randn(
            *shape, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

    x, dt = rand(tokens, 64, 64), rand(tokens, 64)
    A = (-torch.arange(1, 65, device="cuda", dtype=torch.float32)).requires_grad_()
    B, C = rand(tokens, 8, 128), rand(tokens, 8, 128)
    D = torch.ones(64, device="cuda", requires_grad=True)
    bias = torch.full((64,), -4.0, device="cuda", requires_grad=True)
    inputs = (x, dt, A, B, C, D, bias)
    output = packed_scan(*inputs, meta)
    independent, native = [], []
    for a, b in zip(meta.boundaries, meta.boundaries[1:]):
        independent.append(
            packed_scan(
                x[a:b], dt[a:b], A, B[a:b], C[a:b], D, bias, SSMMeta((0, b - a))
            )
        )
        native.append(
            native_scan(
                x[a:b].unsqueeze(0),
                dt[a:b].unsqueeze(0),
                A,
                B[a:b].unsqueeze(0),
                C[a:b].unsqueeze(0),
                chunk_size=128,
                D=D,
                dt_bias=bias,
                dt_softplus=True,
            ).squeeze(0)
        )
    assert torch.equal(output, torch.cat(independent))
    upstream = torch.randn_like(output)
    actual = torch.autograd.grad(output, inputs, upstream)
    expected = torch.autograd.grad(torch.cat(native), inputs, upstream)
    for a, b in zip(actual, expected, strict=True):
        assert torch.isfinite(a).all()
        assert torch.equal(a, b)


def _ssd_cp_worker(rank, rendezvous):
    from megatron.lite.model.nemotron_h.mamba import packed_scan
    from megatron.lite.primitive.parallel.cp import get_parameter_local_cp_headwise

    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        init_batch_invariance()
        torch.manual_seed(42)

        def rand(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        full = (rand(256, 64, 64), rand(256, 64), rand(256, 8, 128), rand(256, 8, 128))
        A = -torch.arange(1, 65, device="cuda", dtype=torch.float32)
        D = torch.ones(64, device="cuda")
        bias = torch.full((64,), -4.0, device="cuda")
        meta = SSMMeta((0, 127, 256))  # Request2 crosses the rank boundary at128.
        reference_inputs = tuple(t.clone().requires_grad_() for t in full)
        fx, fdt, fB, fC = reference_inputs
        reference = packed_scan(fx, fdt, A, fB, fC, D, bias, meta)

        local_inputs = tuple(
            t.chunk(2, dim=0)[rank].clone().requires_grad_() for t in full
        )
        x, dt, B, C = (
            exchange_sequence_channels(t, dist.group.WORLD) for t in local_inputs
        )
        local_A, local_D, local_bias = (
            get_parameter_local_cp_headwise(t, 0, 2, rank) for t in (A, D, bias)
        )
        y = packed_scan(x, dt, local_A, B, C, local_D, local_bias, meta)
        y = exchange_sequence_channels(y, dist.group.WORLD, reverse=True)
        assert torch.equal(y, reference.chunk(2, dim=0)[rank])
        gradient = rand(*reference.shape)
        actual = torch.autograd.grad(y, local_inputs, gradient.chunk(2, dim=0)[rank])
        expected = torch.autograd.grad(reference, reference_inputs, gradient)
        for a, b in zip(actual, expected, strict=True):
            assert torch.isfinite(a).all()
            assert torch.equal(a, b.chunk(2, dim=0)[rank])
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(2)
def test_ssd_cp2_matches_cp1_forward_and_input_gradients(tmp_path):
    mp.spawn(
        _ssd_cp_worker,
        args=(f"file://{tmp_path}/nccl-rendezvous",),
        nprocs=2,
        join=True,
    )


@pytest.mark.gpus(1)
def test_full_mamba_mixer_matches_frozen_hf_alignment_with_real_weights():
    import json
    import os
    from pathlib import Path

    import nemotron_h_reference as oracle
    from megatron.lite.model.nemotron_h.config import NemotronHConfig
    from megatron.lite.model.nemotron_h.mamba import MambaMixer
    from megatron.lite.primitive.parallel.state import ParallelState
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.nemotron_h import modeling_nemotron_h as hf

    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    if "NEMOTRON_TEST_MODEL" not in os.environ:
        pytest.skip("Set NEMOTRON_TEST_MODEL to the frozen BF16 checkpoint")
    root = Path(os.environ["NEMOTRON_TEST_MODEL"])
    config = NemotronHConfig.from_hf(str(root))
    init_batch_invariance()
    torch.manual_seed(42)
    model = MambaMixer(config, ParallelState(), device="cuda")
    hf_config = AutoConfig.from_pretrained(root, local_files_only=True)
    hf_config.time_step_limit = (0.0, float("inf"))
    reference = (
        hf.NemotronHMamba2Mixer(hf_config, layer_idx=0)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = "backbone.layers.0.mixer."
    weights = {}
    for file in sorted({file for key, file in index.items() if key.startswith(prefix)}):
        with safe_open(root / file, framework="pt", device="cpu") as reader:
            for key, filename in index.items():
                if filename == file and key.startswith(prefix):
                    weights[key.removeprefix(prefix)] = reader.get_tensor(key)
    model.load_state_dict(weights, strict=True)
    reference.load_state_dict(weights, strict=True)
    old_conv, old_scan = hf.causal_conv1d_fn, hf.mamba2_chunk_scan
    token = oracle._sequence_boundaries.set((0, 17, 36))
    try:
        oracle.install_linear_forward(reference)
        oracle.install_transformers_gated_rms_forward(reference)
        oracle.install_transformers_conv_forward()
        oracle.install_transformers_mamba_forward(reference)
        x = torch.randn(
            36,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        xr = x.detach().clone().requires_grad_()
        output = model(x, SSMMeta((0, 17, 36)))
        expected = reference(xr.unsqueeze(0)).squeeze(0)
        assert torch.equal(output, expected)
        grad = torch.randn_like(output)
        actual_grads = torch.autograd.grad(output, (x, *model.parameters()), grad)
        expected_grads = torch.autograd.grad(
            expected, (xr, *reference.parameters()), grad
        )
        # HF declaration order differs; compare by names, not positional parameters.
        actual_grads = dict(
            zip(("input", *dict(model.named_parameters())), actual_grads)
        )
        expected_grads = dict(
            zip(("input", *dict(reference.named_parameters())), expected_grads)
        )
        assert actual_grads.keys() == expected_grads.keys()
        for name, actual in actual_grads.items():
            assert torch.isfinite(actual).all(), name
            assert torch.equal(actual, expected_grads[name]), name
    finally:
        oracle._sequence_boundaries.reset(token)
        hf.causal_conv1d_fn, hf.mamba2_chunk_scan = old_conv, old_scan


def _mixer_cp_worker(rank, rendezvous, model_path):
    import json
    from pathlib import Path

    from megatron.lite.model.nemotron_h.config import NemotronHConfig
    from megatron.lite.model.nemotron_h.mamba import MambaMixer
    from megatron.lite.primitive.parallel.state import ParallelState
    from safetensors import safe_open

    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        init_batch_invariance()
        torch.manual_seed(42)
        root = Path(model_path)
        config = NemotronHConfig.from_hf(str(root))
        parallel = ParallelState(cp_group=dist.group.WORLD, cp_size=2, cp_rank=rank)
        candidate = MambaMixer(config, parallel, device="cuda")
        reference = MambaMixer(config, ParallelState(), device="cuda")
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        prefix = "backbone.layers.0.mixer."
        weights = {}
        for file in sorted(
            {file for key, file in index.items() if key.startswith(prefix)}
        ):
            with safe_open(root / file, framework="pt", device="cpu") as reader:
                for key, filename in index.items():
                    if filename == file and key.startswith(prefix):
                        weights[key.removeprefix(prefix)] = reader.get_tensor(key)
        candidate.load_state_dict(weights, strict=True)
        reference.load_state_dict(weights, strict=True)
        x = torch.randn(
            36,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        local_x = x.detach().chunk(2, dim=0)[rank].clone().requires_grad_()
        meta = SSMMeta((0, 17, 36))
        expected = reference(x, meta)
        actual = candidate(local_x, meta)
        assert torch.equal(actual, expected.chunk(2, dim=0)[rank])
        gradient = torch.randn_like(expected)
        actual_grad = torch.autograd.grad(
            actual, local_x, gradient.chunk(2, dim=0)[rank]
        )[0]
        expected_grad = torch.autograd.grad(expected, x, gradient)[0]
        assert torch.isfinite(actual_grad).all()
        assert torch.equal(actual_grad, expected_grad.chunk(2, dim=0)[rank])
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(2)
def test_full_mamba_cp2_matches_cp1_with_real_weights(tmp_path):
    import os

    if "NEMOTRON_TEST_MODEL" not in os.environ:
        pytest.skip("Set NEMOTRON_TEST_MODEL to the frozen BF16 checkpoint")
    mp.spawn(
        _mixer_cp_worker,
        args=(f"file://{tmp_path}/mixer-cp", os.environ["NEMOTRON_TEST_MODEL"]),
        nprocs=2,
        join=True,
    )
