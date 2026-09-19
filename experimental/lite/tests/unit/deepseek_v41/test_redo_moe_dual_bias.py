# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dual routing biases move only at step time, from merged replica statistics.

V4.1 keeps one bias per modality and selects between them per token. The update
is an explicit call: a forward or a recompute that nudged the bias would make
activation checkpointing change the model, and a bias updated from local counts
would diverge across replicas. Both are silent, so both are pinned here.
"""

import pytest
import torch


@pytest.fixture
def moe(v41_core_te):
    from megatron.lite.model.deepseek_v41.lite import moe as modality_moe

    return modality_moe


def _router(moe, experts=4, rate=0.5, dp_size=1):
    class _Config:
        n_routed_experts = experts
        num_experts_per_tok = 2
        hidden_size = 8
        norm_topk_prob = True
        topk_method = "noaux_tc"
        n_group = 1
        topk_group = 1
        routed_scaling_factor = 1.0

    from megatron.lite.primitive.parallel.state import ParallelState

    torch.manual_seed(3)
    return moe.ModalityRouter(_Config(), ParallelState(dp_size=dp_size), bias_rate=rate)


# --- replica-scope statistics ----------------------------------------------


@pytest.mark.parametrize(
    'indices, image_mask, counts, totals',
    [
        (
            [[0, 1], [1, 2], [2, 3]],
            [False, False, False],
            [[1, 2, 2, 1], [0, 0, 0, 0]],
            [3, 0],
        ),
        ([[0, 0], [3, 3]], [False, True], [[2, 0, 0, 0], [0, 0, 0, 2]], [1, 1]),
    ],
)
def test_both_modalities_are_counted(moe, indices, image_mask, counts, totals):
    # A rank whose batch happens to be text-only must still emit a row for the
    # image modality; skipping it desynchronises the collective on other ranks.
    indices = torch.tensor(indices)
    image_mask = torch.tensor(image_mask)
    load = moe.reduce_modality_load(indices, image_mask, num_experts=4)
    assert load.counts.shape == (2, 4)
    assert load.counts.dtype == torch.int64
    assert torch.equal(load.counts, torch.tensor(counts))
    assert torch.equal(load.total_tokens, torch.tensor(totals))


# --- the bias pair ----------------------------------------------------------


def test_the_two_biases_are_separate_buffers_not_parameters(moe):
    router = _router(moe)
    assert torch.equal(router.bias, torch.zeros(4))
    assert torch.equal(router.bias_vl, torch.zeros(4))
    assert router.bias is not router.bias_vl
    names = {name for name, _ in router.named_parameters()}
    assert "bias" not in names and "bias_vl" not in names


def test_a_dtype_cast_does_not_round_the_biases(moe):
    router = _router(moe)
    with torch.no_grad():
        router.bias.add_(0.123456789)
    kept = router.bias.clone()
    router.bfloat16()
    assert router.bias.dtype == torch.float32
    assert torch.equal(router.bias, kept)


def test_forward_never_mutates_the_biases(moe):
    # Recompute replays forward; if forward nudged the bias, activation
    # checkpointing would silently train a different model.
    router = _router(moe)
    with torch.no_grad():
        router.bias.add_(0.25)
        router.bias_vl.add_(-0.25)
    before, before_vl = router.bias.clone(), router.bias_vl.clone()
    x = torch.randn(5, 8)
    router(x, image_mask=torch.tensor([True, False, True, False, True]))
    router(x, image_mask=torch.zeros(5, dtype=torch.bool))
    assert torch.equal(router.bias, before)
    assert torch.equal(router.bias_vl, before_vl)


def test_routing_selects_the_bias_belonging_to_each_token_modality(moe):
    # The gap a single shared bias would leave: both buffers still exist and
    # neither is mutated, yet every token routes as if it were text.
    router = _router(moe, experts=4)
    with torch.no_grad():
        router.bias.copy_(torch.tensor([50.0, 0.0, 0.0, 0.0]))
        router.bias_vl.copy_(torch.tensor([0.0, 0.0, 0.0, 50.0]))
    x = torch.randn(6, 8)
    image_mask = torch.tensor([False, True, False, True, False, True])
    _, indices, _ = router(x, image_mask=image_mask)
    text_rows = indices[~image_mask]
    image_rows = indices[image_mask]
    assert (text_rows == 0).any(dim=-1).all(), "text tokens ignored self.bias"
    assert (image_rows == 3).any(dim=-1).all(), "image tokens ignored self.bias_vl"
    # And the two modalities must not collapse onto the same expert set.
    assert not (text_rows == 3).all()
    assert not (image_rows == 0).all()


@pytest.mark.parametrize('world_size, dp_size', [(2, 1), (1, 2)])
def test_update_bias_requires_an_initialized_process_group_for_multiple_ranks(
    moe, monkeypatch, world_size, dp_size
):
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    # Without a group the merge cannot happen, so updating from local counts
    # would quietly diverge the replicas instead of failing.
    router = _router(moe, dp_size=dp_size)
    stats = moe.ModalityLoad(
        torch.tensor([[4, 0, 0, 0], [0, 0, 0, 4]]), torch.tensor([2, 2])
    )
    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    with pytest.raises(RuntimeError, match="process group"):
        router.update_bias(stats)


def test_update_bias_uses_local_counts_without_a_group(moe, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert not torch.distributed.is_initialized()
    router = _router(moe, rate=0.5)
    stats = moe.ModalityLoad(
        torch.tensor([[8, 0, 0, 0], [0, 0, 0, 8]]), torch.tensor([4, 4])
    )
    router.update_bias(stats)
    torch.testing.assert_close(router.bias, torch.tensor([-0.5, 0.5, 0.5, 0.5]))
    torch.testing.assert_close(router.bias_vl, torch.tensor([0.5, 0.5, 0.5, -0.5]))


def test_multiple_ranks_still_reduce_counts_before_bias_update(
    moe, monkeypatch, tmp_path
):
    monkeypatch.setenv('WORLD_SIZE', '2')
    router = _router(moe, rate=0.5)
    torch.distributed.init_process_group(
        backend='gloo',
        init_method=f"file://{tmp_path / 'multi-store'}",
        world_size=1,
        rank=0,
    )
    try:
        monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
        calls = []

        def reduce_counts(counts, *, group):
            assert group is torch.distributed.group.WORLD
            calls.append(counts.clone())
            counts.add_(torch.tensor([0.0, 0.0, 0.0, 16.0]))

        monkeypatch.setattr(torch.distributed, 'all_reduce', reduce_counts)
        stats = moe.ModalityLoad(
            torch.tensor([[8, 0, 0, 0], [8, 0, 0, 0]]), torch.tensor([4, 4])
        )
        router.update_bias(stats)
        assert len(calls) == 2
        torch.testing.assert_close(router.bias, torch.tensor([-0.5, 0.5, 0.5, -0.5]))
        torch.testing.assert_close(router.bias_vl, router.bias)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize('use_deepep', [False, True])
def test_dispatch_option_reaches_model_and_preserves_local_moe(
    moe, monkeypatch, use_deepep
):
    from test_redo_parity import release_config
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.modules import dispatcher

    original, selected = dispatcher.TokenDispatcher, []

    def capture(*args, **kwargs):
        selected.append(kwargs['use_deepep'])
        return original(*args, **kwargs)

    monkeypatch.setattr(dispatcher, 'TokenDispatcher', capture)
    impl = protocol.ImplConfig(device='cpu', quantized=False, use_deepep=use_deepep)
    model = protocol.build_model(release_config(), impl_cfg=impl).chunks[0]
    assert selected == [use_deepep] * 40
    assert protocol.ImplConfig().use_deepep is False
    torch.manual_seed(417)
    module = model.layers[0].ffn.float()
    x = torch.randn(5, 32, requires_grad=True)
    weights, indices, _ = module.gate(x)
    # Independent gather/scatter oracle: route first, then apply weighted SwiGLU
    # before the down projection, avoiding a different FP32 reduction order.
    expected = torch.zeros_like(x)
    for slot in range(indices.shape[1]):
        for expert_id, expert in enumerate(module.experts):
            rows = torch.where(indices[:, slot] == expert_id)[0]
            gate = torch.nn.functional.linear(x[rows], expert.w1.weight).clamp(max=10)
            up = torch.nn.functional.linear(x[rows], expert.w3.weight).clamp(-10, 10)
            hidden = torch.nn.functional.silu(gate) * up * weights[rows, slot, None]
            expected = expected.index_add(
                0, rows, torch.nn.functional.linear(hidden, expert.w2.weight)
            )
    expected = expected + module.shared_experts(x)
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    wanted = torch.autograd.grad(expected.sum(), x)[0]
    actual.sum().backward()
    torch.testing.assert_close(x.grad, wanted, rtol=1e-6, atol=1e-7)


def test_requested_deepep_never_silently_falls_back(moe, monkeypatch):
    from megatron.lite.primitive.modules import dispatcher
    from megatron.lite.primitive.parallel.state import ParallelState

    monkeypatch.setattr(dispatcher, 'deep_ep', None)
    with pytest.raises(RuntimeError, match='V4.1_DEEPEP_UNAVAILABLE'):
        moe.DeepseekV41MoE(
            _router(moe),
            [torch.nn.Identity()] * 4,
            ps=ParallelState(ep_size=2),
            use_deepep=True,
        )
