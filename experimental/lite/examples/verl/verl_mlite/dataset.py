# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""VERL dataset adapters owned by the MLite application integration."""

from __future__ import annotations

from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset


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


__all__ = ["ChatTemplateRLHFDataset"]
