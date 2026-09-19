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


@pytest.mark.parametrize(
    'options, code',
    [
        ({'include_mtp_only': True}, 'MTP_ONLY'),
        ({'include_local_prefixes': ['layers.0.']}, 'LOCAL_PREFIXES'),
        ({'include_local_prefixes': []}, 'LOCAL_PREFIXES'),
    ],
)
def test_export_rejects_unsupported_selection_before_reading(
    monkeypatch, options, code
):
    exporter = Mock(return_value=iter([('weight', torch.ones(1))]))
    monkeypatch.setattr(checkpoint, 'export_checkpoint', exporter)
    with pytest.raises(NotImplementedError, match=f'V4.1_HF_EXPORT_{code}_UNSUPPORTED'):
        list(protocol.export_hf_weights([object()], None, None, **options))
    exporter.assert_not_called()


def test_export_default_selection_preserves_tensors(monkeypatch):
    weight = torch.tensor([2.0, 3.0])
    exporter = Mock(return_value=iter([('weight', weight)]))
    monkeypatch.setattr(checkpoint, 'export_checkpoint', exporter)
    model = object()
    result = list(
        protocol.export_hf_weights(
            [model], None, None, include_mtp_only=False, include_local_prefixes=None
        )
    )
    assert len(result) == 1 and result[0][0] == 'weight'
    assert torch.equal(result[0][1], weight)
    exporter.assert_called_once_with(model)
