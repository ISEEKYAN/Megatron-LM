# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""VERL_MLITE_ROLLOUT_ENV reaches only the rollout server env, on top of the platform's own vars."""

import sys
import types

from verl_mlite import compat


def test_rollout_env_overrides(monkeypatch):
    class Platform:
        def rollout_env_vars(self):
            return {"NCCL_CUMEM_ENABLE": "0", "VLLM_BATCH_INVARIANT": "1"}

    platform = Platform()
    module = types.ModuleType("verl.plugin.platform.platform_manager")
    module.get_platform = lambda: platform
    monkeypatch.setitem(sys.modules, "verl.plugin.platform.platform_manager", module)
    monkeypatch.setenv("VERL_MLITE_ROLLOUT_ENV", 'VLLM_BATCH_INVARIANT=0')
    assert compat._patch_verl_rollout_env_overrides()
    assert compat._patch_verl_rollout_env_overrides()  # idempotent
    assert platform.rollout_env_vars() == {"NCCL_CUMEM_ENABLE": "0", "VLLM_BATCH_INVARIANT": "0"}
    monkeypatch.delenv("VERL_MLITE_ROLLOUT_ENV")
    assert not compat._patch_verl_rollout_env_overrides()
