# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Where the LoRA rollout-sync mode is actually decided.

``LoraSpec.rollout_sync`` declares a default, but no production code reads it:
the rollout export path resolves the mode from the raw engine config, because
hydra hands nested sections over as ``DictConfig`` rather than ``dict``. For a
while the two disagreed -- the dataclass said ``merge``, the resolver said
``adapter`` -- and the test guarding the dataclass stayed green while every run
took the other path.

So these tests are attached to the resolver. An assertion only guards an
invariant if it sits where that invariant is still visible.
"""

from __future__ import annotations

import pytest
from verl_mlite.engine.mlite_engine import MegatronLiteEngine

resolve = MegatronLiteEngine._lora_rollout_sync_is_merge


def test_default_is_adapter_only_when_the_key_is_absent():
    """The default that runs. Omitting the key must not silently mean merge."""
    assert resolve({"enabled": True, "rank": 8}) is False


def test_declared_default_matches_the_resolved_default():
    """The two surfaces must not drift apart again.

    This is the regression that motivated the file: a dataclass default of
    ``merge`` next to a resolver default of ``adapter``, with no test able to
    see both at once.
    """
    from megatron.lite.primitive.modules.lora import LoraSpec

    declared = LoraSpec().rollout_sync
    resolved_is_merge = resolve({"enabled": True, "rank": 8})
    assert (declared == "merge") is resolved_is_merge


@pytest.mark.parametrize("mode,expect_merge", [("adapter", False), ("merge", True)])
def test_explicit_mode_is_honoured(mode, expect_merge):
    assert resolve({"enabled": True, "rank": 8, "rollout_sync": mode}) is expect_merge


def test_unknown_mode_fails_loud():
    """A typo must not fall back to a working-looking path."""
    with pytest.raises(ValueError, match="rollout_sync"):
        resolve({"enabled": True, "rank": 8, "rollout_sync": "adaptor"})


@pytest.mark.parametrize("init", ["olora", "pissa", "OLoRA", "PiSSA"])
def test_residual_base_inits_force_merge_even_when_adapter_is_requested(init):
    """These inits subtract the adapter's starting delta from the base weight.

    The rollout base is then not the pretrained weight, so an adapter-only sync
    would apply the delta to the wrong operand -- a numerically wrong policy
    rather than a slow one. The override is silent-safe because it errs toward
    the lossy-but-correct path.
    """
    assert (
        resolve({"enabled": True, "rank": 8, "rollout_sync": "adapter", "init": init})
        is True
    )


def test_ordinary_init_does_not_trigger_the_override():
    """Guard against the residual-base check swallowing the normal case."""
    assert (
        resolve(
            {"enabled": True, "rank": 8, "rollout_sync": "adapter", "init": "default"}
        )
        is False
    )
