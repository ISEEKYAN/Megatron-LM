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
def moe(transformer_engine_import_stub):
    import megatron.core.fp8_utils  # noqa: F401

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import modality_moe

    return modality_moe


def _router(moe, experts=4, rate=0.5):
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
    return moe.ModalityRouter(_Config(), ParallelState(), bias_rate=rate)


# --- replica-scope statistics ----------------------------------------------


def test_both_modalities_are_counted_even_when_one_is_locally_empty(moe):
    # A rank whose batch happens to be text-only must still emit a row for the
    # image modality; skipping it desynchronises the collective on other ranks.
    indices = torch.tensor([[0, 1], [1, 2], [2, 3]])
    image_mask = torch.zeros(3, dtype=torch.bool)
    load = moe.reduce_modality_load(indices, image_mask, num_experts=4)
    assert load.counts.shape == (2, 4)
    assert torch.equal(load.counts[0], torch.tensor([1, 2, 2, 1]))
    assert torch.equal(load.counts[1], torch.zeros(4, dtype=torch.int64))
    assert torch.equal(load.total_tokens, torch.tensor([3, 0]))


def test_counts_are_split_by_modality(moe):
    indices = torch.tensor([[0, 0], [3, 3]])
    image_mask = torch.tensor([False, True])
    load = moe.reduce_modality_load(indices, image_mask, num_experts=4)
    assert torch.equal(load.counts[0], torch.tensor([2, 0, 0, 0]))
    assert torch.equal(load.counts[1], torch.tensor([0, 0, 0, 2]))
    assert torch.equal(load.total_tokens, torch.tensor([1, 1]))


def test_statistics_are_detached_from_the_graph(moe):
    indices = torch.tensor([[0, 1]])
    load = moe.reduce_modality_load(indices, torch.zeros(1, dtype=torch.bool), 4)
    assert not load.counts.requires_grad
    assert load.counts.dtype == torch.int64


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


def test_update_bias_requires_an_initialized_process_group_for_multiple_ranks(
    moe, monkeypatch
):
    monkeypatch.setenv("WORLD_SIZE", "2")
    # Without a group the merge cannot happen, so updating from local counts
    # would quietly diverge the replicas instead of failing.
    router = _router(moe)
    stats = moe.ModalityLoad(
        torch.tensor([[4, 0, 0, 0], [0, 0, 0, 4]]), torch.tensor([2, 2])
    )
    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    with pytest.raises(RuntimeError, match="process group"):
        router.update_bias(stats)


def test_update_bias_moves_each_modality_from_its_own_counts(moe, tmp_path):
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'store'}", world_size=1, rank=0
    )
    try:
        router = _router(moe, rate=0.5)
        stats = moe.ModalityLoad(
            torch.tensor([[8, 0, 0, 0], [0, 0, 0, 8]]), torch.tensor([4, 4])
        )
        router.update_bias(stats)
        # Overloaded experts are pushed down, starved ones up, by bias_rate.
        assert router.bias[0] < router.bias[1]
        assert router.bias_vl[3] < router.bias_vl[0]
        # Each modality reads only its own row.
        assert torch.equal(router.bias[1], router.bias[2])
        assert torch.equal(router.bias_vl[0], router.bias_vl[1])
    finally:
        torch.distributed.destroy_process_group()


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


def test_parallel_topology_cannot_silently_use_local_counts(moe, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    # Construct through the same router config, but declare two DP ranks.
    from types import SimpleNamespace

    from megatron.lite.primitive.parallel.state import ParallelState

    config = SimpleNamespace(
        n_routed_experts=4,
        num_experts_per_tok=2,
        hidden_size=8,
        norm_topk_prob=True,
        topk_method='noaux_tc',
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    router = moe.ModalityRouter(config, ParallelState(dp_size=2))
    stats = moe.ModalityLoad(
        torch.tensor([[8, 0, 0, 0], [0, 0, 0, 8]]), torch.tensor([4, 4])
    )
    with pytest.raises(RuntimeError, match='process group'):
        router.update_bias(stats)


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
