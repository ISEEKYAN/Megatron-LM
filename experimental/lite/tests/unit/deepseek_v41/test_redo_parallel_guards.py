# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unsupported combinations fail in order before model or process-group setup."""

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite.protocol import (
    ImplConfig,
    _validate_parallel,
    build_model,
)
from megatron.lite.runtime.contracts import ParallelConfig


def _build(**kwargs):
    return build_model(object(), impl_cfg=ImplConfig(**kwargs))


# --- the coarse dimension check, which runs before dependent guards ----------------


@pytest.mark.parametrize(
    "parallel, named",
    [
        (ParallelConfig(tp=2), "tp"),
        (ParallelConfig(vpp=2), "vpp"),
        (ParallelConfig(pp=4), "pp"),
        (ParallelConfig(etp=2), "etp"),
        (ParallelConfig(pp_layout=[[0], [1]]), "pp_layout"),
    ],
)
def test_unsupported_dimensions_are_named_individually(parallel, named):
    with pytest.raises(NotImplementedError) as caught:
        _build(parallel=parallel)
    message = str(caught.value)
    assert message.startswith("V4.1_UNSUPPORTED_PARALLELISM:")
    # The offending knob is named, not just "unsupported": an operator reading
    # the log must be able to tell which setting to change.
    assert named in message.split(";")[0]


def test_several_unsupported_dimensions_are_reported_together():
    with pytest.raises(NotImplementedError) as caught:
        _build(parallel=ParallelConfig(tp=2, vpp=2))
    head = str(caught.value).split(";")[0]
    assert "tp" in head and "vpp" in head


# --- dependent guards --------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, key",
    [
        (dict(text_only=False), "V4.1_PP_TEXT_ONLY"),
        (dict(external_vision_device="cuda:0"), "V4.1_PP_TEXT_ONLY"),
        (dict(pipeline_split_layer=10), "V4.1_PP_CSA2_PAYLOAD_UNSUPPORTED"),
        (dict(optimizer="muon"), "V4.1_PP_OPTIMIZER_UNSUPPORTED"),
    ],
)
def test_pp2_rejections_name_their_own_cause(kwargs, key):
    with pytest.raises(NotImplementedError) as caught:
        _build(parallel=ParallelConfig(pp=2), **kwargs)
    assert str(caught.value).startswith(key), str(caught.value)


@pytest.mark.parametrize("extra", [dict(ep=2), dict(cp=2)])
def test_pp2_rejects_being_combined_with_ep_or_cp(extra):
    with pytest.raises(NotImplementedError) as caught:
        _build(parallel=ParallelConfig(pp=2, **extra))
    assert str(caught.value).startswith("V4.1_PP_COMBINATION_UNSUPPORTED")


def test_cp_and_ep_cannot_be_combined():
    with pytest.raises(
        NotImplementedError, match="CP_AND_EP_NOT_SIMULTANEOUSLY_SUPPORTED"
    ):
        _build(parallel=ParallelConfig(cp=2, ep=2))


def test_split_layer_20_is_the_one_accepted_cut(monkeypatch):
    # The topology table puts every KV owner (2/8/14/20) on the same stage as
    # its readers only when the cut falls on layer 20; the guard encodes that.
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    parallel = ParallelConfig(pp=2)
    assert _validate_parallel(ImplConfig(pipeline_split_layer=20), parallel) is None
    for cut in (2, 8, 14, 19, 21, 39):
        with pytest.raises(
            NotImplementedError, match='V4.1_PP_CSA2_PAYLOAD_UNSUPPORTED'
        ):
            _validate_parallel(ImplConfig(pipeline_split_layer=cut), parallel)
    # With PP off the cut is irrelevant and must not be rejected.
    assert (
        _validate_parallel(ImplConfig(pipeline_split_layer=10), ParallelConfig(pp=1))
        is None
    )


def test_guards_run_before_any_distributed_setup(monkeypatch):
    # A guard that only fires after init_parallel would hang the ranks that do
    # reach it and never reject on the ranks that do not.
    def explode(*args, **kwargs):
        raise AssertionError("distributed setup ran before the guards rejected")

    monkeypatch.setattr(torch.distributed, "is_initialized", explode)
    with pytest.raises(NotImplementedError):
        _build(parallel=ParallelConfig(tp=2))
    with pytest.raises(NotImplementedError):
        _build(parallel=ParallelConfig(pp=2), text_only=False)


@pytest.mark.parametrize(
    'parallel, initialized, world, message',
    [
        (ParallelConfig(pp=2), False, 1, 'V4.1_PP_WORLD'),
        (ParallelConfig(ep=0), False, 1, 'EP size must be a positive integer'),
        (ParallelConfig(cp=0), False, 1, 'CP size must be a positive integer'),
        (ParallelConfig(cp=2), False, 2, 'CP requires an initialized CP-only world'),
        (ParallelConfig(cp=2), True, 4, 'CP requires an initialized CP-only world'),
        (
            ParallelConfig(ep=2),
            False,
            2,
            'EP requires an initialized distributed world',
        ),
        (ParallelConfig(ep=2), True, 1, 'EP requires an initialized distributed world'),
        (ParallelConfig(ep=2), True, 3, 'EP requires an initialized distributed world'),
    ],
)
def test_remaining_table_guards_through_build_model(
    monkeypatch, parallel, initialized, world, message
):
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: initialized)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: world)
    with pytest.raises(ValueError, match=message):
        _build(parallel=parallel)


@pytest.mark.parametrize('rank', range(4))
def test_ep_replicas_use_independent_expert_dp_groups(monkeypatch, v41_core_te, rank):
    from megatron.lite.model.deepseek_v41.lite import model, optimizer_groups
    from megatron.lite.primitive.modules import native_fp32_linear

    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 4)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: rank)
    monkeypatch.setattr(
        torch.distributed, 'new_group', lambda ranks, **kw: tuple(ranks)
    )

    def inspect_groups(*args, parallel_state, **kwargs):
        # Real init_parallel computes these; only network creation is replaced.
        assert parallel_state.ep_group == ((0, 1) if rank < 2 else (2, 3))
        assert parallel_state.ep_dp_group == ((0, 2) if rank % 2 == 0 else (1, 3))
        assert parallel_state.dp_group == (0, 1, 2, 3)
        assert parallel_state.expert_dp_size == 2
        chunk = torch.nn.Module()
        chunk.engram_hash = None
        chunk.parameter_bindings = lambda: []
        return chunk

    def inspect_optimizer(*args, dp_group, ps, **kwargs):
        # Follow the real protocol -> V41Optimizer -> MixedOptimizer wiring.
        # Router counts use dense DP (all ranks), never expert-DP replicas.
        assert dp_group is ps.dp_group
        assert dp_group == tuple(range(torch.distributed.get_world_size()))
        assert dp_group != ps.ep_dp_group
        raise RuntimeError('optimizer count scope verified before allocation')

    monkeypatch.setattr(model, 'DeepseekV41Model', inspect_groups)
    monkeypatch.setattr(optimizer_groups, 'MixedOptimizer', inspect_optimizer)
    monkeypatch.setattr(
        native_fp32_linear, 'configure_residual_projections', lambda *args: None
    )
    with pytest.raises(RuntimeError, match='optimizer count scope verified'):
        _build(
            device='cpu',
            parallel=ParallelConfig(ep=2),
            optimizer='muon',
            optimizer_config=optimizer_groups.OptimizerConfig(
                lr=0.001, ns_steps=5, coefficient_type='quintic'
            ),
        )
