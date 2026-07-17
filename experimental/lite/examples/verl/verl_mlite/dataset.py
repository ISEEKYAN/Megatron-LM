# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""VERL dataset adapters owned by the MLite application integration."""

from __future__ import annotations

from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset

from verl_mlite.dataset_schema import normalize_rule_reward_fields


class ChatTemplateRLHFDataset(RLHFDataset):
    """Apply an explicit chat template before VERL filters dataset prompts."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: Any,
        config: Any,
        processor: Any = None,
        max_samples: int = -1,
    ) -> None:
        chat_template = config.get("chat_template")
        if not isinstance(chat_template, str) or not chat_template.strip():
            raise ValueError("ChatTemplateRLHFDataset requires data.chat_template")
        tokenizer.chat_template = chat_template
        if processor is not None:
            processor.chat_template = chat_template
        super().__init__(
            data_files,
            tokenizer,
            config,
            processor=processor,
            max_samples=max_samples,
        )
        label_key = config.get("label_key")
        default_data_source = config.get("default_data_source")
        if label_key is not None or default_data_source is not None:
            if not isinstance(label_key, str) or not label_key:
                raise ValueError("data.label_key must be a non-empty string")
            if not isinstance(default_data_source, str) or not default_data_source:
                raise ValueError("data.default_data_source must be a non-empty string")
            self.dataframe = self.dataframe.map(
                lambda example: normalize_rule_reward_fields(
                    example,
                    label_key=label_key,
                    default_data_source=default_data_source,
                ),
                desc="Normalizing VERL rule-reward fields",
            )


__all__ = ["ChatTemplateRLHFDataset"]
