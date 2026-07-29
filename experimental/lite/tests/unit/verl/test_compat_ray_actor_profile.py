# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from verl_mlite import compat


class _ActorClass:
    def options(self, **kwargs):
        return kwargs


def test_ray_actor_profile_adds_torchrun_process_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
    actor = compat._RayActorClassProfile(_ActorClass(), {"PROFILE": "enabled"})

    options = actor.options(
        runtime_env={
            "env_vars": {"CALLER": "preserved"},
            "working_dir": "/workspace",
        },
        name="worker",
    )

    assert options == {
        "runtime_env": {
            "env_vars": {
                "CALLER": "preserved",
                "OMP_NUM_THREADS": "1",
                "PROFILE": "enabled",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            },
            "working_dir": "/workspace",
        },
        "name": "worker",
    }


def test_ray_actor_profile_preserves_explicit_torchrun_process_values(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    actor = compat._RayActorClassProfile(_ActorClass(), {})

    options = actor.options()

    assert options["runtime_env"]["env_vars"] == {
        "OMP_NUM_THREADS": "3",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
    }
