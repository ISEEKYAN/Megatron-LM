# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real VERL padding can run without the optional FlashAttention CUDA package."""
import builtins
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.optional


def _block_flash_attention_import(monkeypatch, missing_dependency='flash_attn'):
    original_import = builtins.__import__

    def without_flash_attention(name, *args, **kwargs):
        if name == 'flash_attn.bert_padding':
            raise ModuleNotFoundError(name=missing_dependency)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_flash_attention)


def test_padding_fallback_real_verl_roundtrip_and_gradient(monkeypatch):
    pytest.importorskip(
        "verl.utils.attention_padding_utils", reason="requires VERL native padding fallback"
    )
    from tensordict import TensorDict
    from verl.utils import attention_utils as attention
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    _block_flash_attention_import(monkeypatch)
    mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    values = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2).requires_grad_()
    unpacked, indices, offsets, maximum, lengths = attention.unpad_input(values, mask)
    assert torch.equal(unpacked, values[mask.bool()])
    assert offsets.tolist() == [0, 4, 8] and maximum == 4 and lengths.tolist() == [4, 4]
    restored = attention.pad_input(unpacked, indices, 2, 5)
    assert torch.equal(restored[mask.bool()], values[mask.bool()])
    restored.sum().backward()
    assert torch.equal(values.grad, mask[..., None].expand_as(values).float())
    ids = torch.tensor([[0, 11, 12, 21, 22], [13, 14, 15, 23, 0]])
    batch = TensorDict(
        dict(
            input_ids=ids,
            attention_mask=mask,
            response_mask=torch.tensor([[1, 1], [1, 0]]),
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
            prompts=ids[:, :3],
            responses=ids[:, 3:],
        ),
        batch_size=[2],
    )
    packed = left_right_2_no_padding(batch)
    assert packed['input_ids'].values().tolist() == [11, 12, 21, 22, 13, 14, 15, 23]
    scores = no_padding_2_padding(torch.arange(8, dtype=torch.float32), packed)
    assert torch.equal(
        scores.view(torch.int32), torch.tensor([[1.0, 2.0], [6.0, 0.0]]).view(torch.int32)
    )


def test_padding_keeps_installed_native_provider(monkeypatch):
    from verl.utils import attention_utils as attention

    sentinel = (object(),) * 4
    provider = SimpleNamespace(
        **dict(zip(('index_first_axis', 'pad_input', 'rearrange', 'unpad_input'), sentinel))
    )
    original_import = builtins.__import__

    def with_flash_attention(name, *args, **kwargs):
        if name == 'flash_attn.bert_padding':
            return provider
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', with_flash_attention)
    assert attention._get_attention_functions() == sentinel


@pytest.mark.parametrize('missing_dependency', ['flash_attn', 'flash_attn_2_cuda'])
def test_padding_uses_native_fallback_when_provider_cannot_import(monkeypatch, missing_dependency):
    pytest.importorskip(
        "verl.utils.attention_padding_utils", reason="requires VERL native padding fallback"
    )
    from verl.utils import attention_padding_utils, attention_utils

    _block_flash_attention_import(monkeypatch, missing_dependency)
    expected = tuple(
        getattr(attention_padding_utils, name)
        for name in ('index_first_axis', 'pad_input', 'rearrange', 'unpad_input')
    )
    assert attention_utils._get_attention_functions() == expected
