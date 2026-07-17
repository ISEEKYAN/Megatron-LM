# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Schema adapters shared by VERL datasets without importing VERL itself."""

from __future__ import annotations

from typing import Any


def normalize_rule_reward_fields(
    example: dict[str, Any], *, label_key: str, default_data_source: str
) -> dict[str, Any]:
    """Adapt compact ``prompt``/``label`` rows to VERL's rule-reward schema."""
    reward_model = example.get("reward_model")
    if example.get("data_source") and isinstance(reward_model, dict) and reward_model.get(
        "ground_truth"
    ) is not None:
        return {}
    if label_key not in example or example[label_key] is None:
        raise KeyError(
            f"dataset row lacks both VERL reward fields and label_key={label_key!r}"
        )
    return {
        "data_source": example.get("data_source") or default_data_source,
        "reward_model": {"style": "rule", "ground_truth": example[label_key]},
    }


__all__ = ["normalize_rule_reward_fields"]
