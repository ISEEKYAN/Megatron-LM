# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks for resuming a DistributedOptimizer checkpoint under a changed lr."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron.core.optimizer")

from megatron.lite.primitive.ckpt import distckpt  # noqa: E402

OLD_LR, NEW_LR = 3.0e-5, 1.0e-4


def _group(lr: float, **overrides) -> dict:
    group = {
        "lr": lr,
        "max_lr": lr,
        "min_lr": 0.0,
        "wd_mult": 1.0,
        "lr_mult": 1.0,
        "is_expert_parallel": False,
        "is_decoupled_lr": False,
        "weight_decay": 0.1,
    }
    group.update(overrides)
    return group


def _fake_dist_opt(groups: list[dict]) -> SimpleNamespace:
    # Shape that _iter_distributed_optimizers recognises as a DistributedOptimizer.
    return SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=groups),
        sharded_state_dict=lambda *args, **kwargs: {},
        gbuf_ranges=[],
        buffers=[],
    )


def _saved_state(groups: list[dict]) -> dict:
    return {"optimizer": {"param_groups": groups}}


def test_rewrites_saved_lr_identity_to_current_groups():
    current = [_group(NEW_LR), _group(NEW_LR, is_expert_parallel=True)]
    saved = [_group(OLD_LR, is_expert_parallel=True), _group(OLD_LR)]

    changes = distckpt._rewrite_saved_lr_identity(current, saved)

    assert [idx for idx, _, _ in changes] == [0, 1]
    assert changes[0][1] == {"max_lr": OLD_LR, "min_lr": 0.0, "lr": OLD_LR}
    assert changes[0][2] == {"max_lr": NEW_LR, "min_lr": 0.0, "lr": NEW_LR}
    for group in saved:
        assert group["max_lr"] == group["lr"] == NEW_LR
        # Non-lr hyperparameters keep coming from the checkpoint.
        assert group["weight_decay"] == 0.1


def test_noop_when_every_current_group_has_an_exact_saved_match():
    # Groups that legitimately differ only by max_lr still load without any rewrite.
    current = [_group(OLD_LR), _group(NEW_LR)]
    saved = [_group(NEW_LR), _group(OLD_LR)]

    assert distckpt._rewrite_saved_lr_identity(current, saved) == []
    assert [group["max_lr"] for group in saved] == [NEW_LR, OLD_LR]


def test_refuses_when_current_groups_differ_only_by_lr():
    current = [_group(NEW_LR), _group(2 * NEW_LR)]
    saved = [_group(OLD_LR), _group(2 * OLD_LR)]

    with pytest.raises(ValueError, match="current param groups differ only by"):
        distckpt._rewrite_saved_lr_identity(current, saved)
    assert [group["max_lr"] for group in saved] == [OLD_LR, 2 * OLD_LR]


def test_refuses_when_saved_groups_differ_only_by_lr():
    current = [_group(NEW_LR)]
    saved = [_group(OLD_LR), _group(2 * OLD_LR)]

    with pytest.raises(ValueError, match="saved param groups differ only by"):
        distckpt._rewrite_saved_lr_identity(current, saved)


def test_refuses_when_a_non_lr_identifier_differs():
    current = [_group(NEW_LR, wd_mult=0.0)]
    saved = [_group(OLD_LR)]

    with pytest.raises(ValueError, match="no saved param group matches"):
        distckpt._rewrite_saved_lr_identity(current, saved)


def test_apply_lr_change_walks_chained_optimizers_by_sorted_state_key():
    dense = _fake_dist_opt([_group(NEW_LR)])
    expert = _fake_dist_opt([_group(2 * NEW_LR, is_expert_parallel=True)])
    chained = SimpleNamespace(chained_optimizers=[dense, expert])
    state = {
        1: _saved_state([_group(2 * OLD_LR, is_expert_parallel=True)]),
        0: _saved_state([_group(OLD_LR)]),
    }

    distckpt._apply_lr_change(chained, state)

    assert state[0]["optimizer"]["param_groups"][0]["max_lr"] == NEW_LR
    assert state[1]["optimizer"]["param_groups"][0]["max_lr"] == 2 * NEW_LR


def test_apply_lr_change_single_chained_optimizer_uses_unwrapped_state():
    dist_opt = _fake_dist_opt([_group(NEW_LR)])
    state = _saved_state([_group(OLD_LR)])

    distckpt._apply_lr_change(SimpleNamespace(chained_optimizers=[dist_opt]), state)

    assert state["optimizer"]["param_groups"][0]["lr"] == NEW_LR


def test_apply_lr_change_rejects_non_distributed_optimizer():
    plain = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[_group(NEW_LR)]))

    with pytest.raises(ValueError, match="DistributedOptimizer only"):
        distckpt._apply_lr_change(plain, _saved_state([_group(OLD_LR)]))


@pytest.mark.parametrize("use_dcp", [True, False])
def test_load_training_checkpoint_rejects_allow_lr_change_without_dist_opt(tmp_path, use_dcp):
    import torch.nn as nn

    from megatron.lite.primitive.ckpt import load_training_checkpoint

    with pytest.raises(ValueError, match="only supported for dist_opt checkpoints"):
        load_training_checkpoint(
            nn.Linear(2, 2), None, str(tmp_path), use_dcp=use_dcp, allow_lr_change=True
        )
