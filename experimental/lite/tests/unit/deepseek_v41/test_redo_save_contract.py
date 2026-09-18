# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HF save rejects unsupported resync requests before writing any checkpoint."""
from unittest.mock import Mock

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import protocol


@pytest.mark.parametrize(
    'options',
    [
        {'target': 'vllm'},
        {'resync_config': {'format': 'fp8'}},
        {'target': 'vllm', 'resync_config': {'format': 'fp8'}},
    ],
)
def test_save_rejects_resync_before_writing(monkeypatch, tmp_path, options):
    writer = Mock()
    monkeypatch.setattr(protocol, 'save_model', writer)
    with pytest.raises(NotImplementedError, match='V4.1_HF_SAVE_RESYNC_UNSUPPORTED'):
        protocol.save_hf_weights([object()], tmp_path / 'save', None, None, **options)
    writer.assert_not_called()


def test_save_preserves_supported_export_options(monkeypatch, tmp_path):
    writer = Mock()
    monkeypatch.setattr(protocol, 'save_model', writer)
    model = object()
    protocol.save_hf_weights(
        [model],
        tmp_path,
        None,
        None,
        target=None,
        resync_config=None,
        export_dtype=torch.float32,
        buffer_max_size_bytes=4096,
    )
    writer.assert_called_once_with(
        model, tmp_path, export_dtype=torch.float32, buffer_max_size_bytes=4096
    )
