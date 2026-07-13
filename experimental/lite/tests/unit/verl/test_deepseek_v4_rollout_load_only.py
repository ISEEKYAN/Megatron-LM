# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit coverage for the 8-GPU load-only GPU-residency probe used to measure the
sibling vLLM TP-rank leak onto GPU 0 (see docs/ds4-resync-gpu0-sibling-residency.md).

Only the pure ``_parse_compute_apps`` accounting is exercised here; it is split from
the ``nvidia-smi`` subprocess call precisely so it is testable without a GPU."""

from pathlib import Path

from deepseek_v4_rollout_load_only import _parse_compute_apps, build_llm_kwargs


def test_parse_compute_apps_attributes_siblings_to_gpu0() -> None:
    # gpu_uuid,pid,used_gpu_memory (noheader,nounits) — reproduces the OOM shape:
    # one home rank (20 GiB) + seven siblings (2.51 GiB) all resident on GPU 0.
    uuid_to_index = {f"GPU-{i}": i for i in range(2)}
    csv_text = "\n".join(
        [
            "GPU-0, 3536515, 20604",  # home vLLM rank on GPU0
            "GPU-0, 3536516, 2570",   # sibling
            "GPU-0, 3536517, 2570",   # sibling
            "GPU-1, 3536518, 20604",  # home rank on GPU1 (its own device)
        ]
    )
    residency = _parse_compute_apps(csv_text, uuid_to_index)
    gpu0 = residency[0]
    assert len(gpu0) == 3
    assert sum(mib for _, mib in gpu0) == 20604 + 2570 + 2570
    # sibling processes (not the home rank) are the reclaimable leak
    siblings = [pid for pid, mib in gpu0 if mib < 3000]
    assert siblings == [3536516, 3536517]
    assert len(residency[1]) == 1


def test_parse_compute_apps_skips_unknown_and_malformed() -> None:
    uuid_to_index = {"GPU-KNOWN": 0}
    csv_text = "\n".join(
        [
            "GPU-KNOWN, 100, 512",
            "GPU-UNKNOWN, 200, 512",  # uuid not in map -> skipped
            "garbage line",           # wrong field count -> skipped
            "GPU-KNOWN, 101, [N/A]",  # non-numeric memory -> counted as 0
        ]
    )
    residency = _parse_compute_apps(csv_text, uuid_to_index)
    assert set(residency) == {0}
    assert residency[0] == [(100, 512), (101, 0)]


def _base_env() -> dict[str, str]:
    return {"CHECKPOINT_DIR": "/models/DeepSeek-V4-Flash", "ROLLOUT_TP": "8"}


def test_build_llm_kwargs_default_omits_executor_backend() -> None:
    # No MLITE_VLLM_DISTRIBUTED_EXECUTOR_BACKEND -> vLLM keeps its default (multiproc).
    kwargs = build_llm_kwargs(
        _base_env(), checkpoint_dir=Path("/ckpt"), rollout_tp=8, sync_probe=False
    )
    assert "distributed_executor_backend" not in kwargs
    assert kwargs["model"] == "/ckpt"
    assert kwargs["tensor_parallel_size"] == 8
    assert "worker_extension_cls" not in kwargs


def test_build_llm_kwargs_passes_ray_backend_through() -> None:
    # C1 arm: the ray executor is a pure env flip over the baseline multiproc arm.
    env = _base_env() | {"MLITE_VLLM_DISTRIBUTED_EXECUTOR_BACKEND": "ray"}
    kwargs = build_llm_kwargs(
        env, checkpoint_dir=Path("/ckpt"), rollout_tp=8, sync_probe=False
    )
    assert kwargs["distributed_executor_backend"] == "ray"


def test_build_llm_kwargs_blank_backend_is_omitted() -> None:
    # A stray/empty export must not force an invalid empty backend onto vLLM.
    env = _base_env() | {"MLITE_VLLM_DISTRIBUTED_EXECUTOR_BACKEND": "  "}
    kwargs = build_llm_kwargs(
        env, checkpoint_dir=Path("/ckpt"), rollout_tp=8, sync_probe=False
    )
    assert "distributed_executor_backend" not in kwargs


def test_build_llm_kwargs_sync_probe_adds_worker_extension() -> None:
    kwargs = build_llm_kwargs(
        _base_env(), checkpoint_dir=Path("/ckpt"), rollout_tp=8, sync_probe=True
    )
    assert kwargs["worker_extension_cls"].endswith("VllmCheckpointWorkerExtension")
