# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HF save/export reject resync requests before touching checkpoint tensors."""
from unittest.mock import Mock

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import checkpoint, protocol


@pytest.mark.parametrize(
    'options',
    [
        {'target': 'vllm'},
        {'resync_config': {'format': 'fp8'}},
        {'target': 'vllm', 'resync_config': {'format': 'fp8'}},
    ],
)
def test_save_and_export_reject_resync(monkeypatch, tmp_path, options):
    writer = Mock()
    monkeypatch.setattr(protocol, 'save_model', writer)
    exporter = Mock(side_effect=AssertionError('Resync reached tensor exporter'))
    monkeypatch.setattr(checkpoint, 'export_checkpoint', exporter)
    with pytest.raises(NotImplementedError, match='V4.1_HF_SAVE_RESYNC_UNSUPPORTED'):
        protocol.save_hf_weights([object()], tmp_path / 'save', None, None, **options)
    with pytest.raises(NotImplementedError, match='V4.1_HF_SAVE_RESYNC_UNSUPPORTED'):
        list(protocol.export_hf_weights([object()], None, None, **options))
    writer.assert_not_called()
    exporter.assert_not_called()


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
