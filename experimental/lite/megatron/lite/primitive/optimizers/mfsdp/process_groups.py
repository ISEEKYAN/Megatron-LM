# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compatibility helpers for the self-contained M-FSDP implementation.

Process-group and device-mesh construction now lives in ``impl.fully_shard``;
these names remain as explicit errors for callers of the old internal API.
"""


def install_mfsdp_mesh_patch() -> None:
    """No-op retained for source compatibility; no namespace patch is needed."""


def build_mfsdp_pg_collection(*_args, **_kwargs):
    raise RuntimeError(
        "M-FSDP process groups are constructed by impl.fully_shard_model; "
        "ProcessGroupCollection is no longer used."
    )


__all__ = ["build_mfsdp_pg_collection", "install_mfsdp_mesh_patch"]
