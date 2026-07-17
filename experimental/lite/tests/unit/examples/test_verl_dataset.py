from __future__ import annotations

import pytest

from verl_mlite.dataset_schema import normalize_rule_reward_fields


def test_normalize_compact_rule_reward_row() -> None:
    assert normalize_rule_reward_fields(
        {"label": "34"}, label_key="label", default_data_source="math_dapo"
    ) == {
        "data_source": "math_dapo",
        "reward_model": {"style": "rule", "ground_truth": "34"},
    }


def test_preserve_native_verl_reward_row() -> None:
    assert (
        normalize_rule_reward_fields(
            {
                "data_source": "aime2024",
                "reward_model": {"style": "rule", "ground_truth": "7"},
            },
            label_key="label",
            default_data_source="math_dapo",
        )
        == {}
    )


def test_reject_row_without_reward_or_label() -> None:
    with pytest.raises(KeyError, match="label_key='label'"):
        normalize_rule_reward_fields(
            {}, label_key="label", default_data_source="math_dapo"
        )
