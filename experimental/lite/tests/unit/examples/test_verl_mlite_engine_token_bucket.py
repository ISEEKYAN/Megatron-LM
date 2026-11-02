# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.optional]


@pytest.fixture
def engine(monkeypatch):
    engine_type = pytest.importorskip("verl_mlite.engine.mlite_engine").MegatronLiteEngine
    value = engine_type.__new__(engine_type)
    value.engine_config = SimpleNamespace(tp=1, cp=8)
    return value


def test_bucket_off_keeps_the_batch(engine, monkeypatch):
    monkeypatch.delenv("MLITE_PACK_BUCKET_TOKENS", raising=False)
    tokens, mask, lens = torch.arange(30), torch.ones(30), torch.tensor([10, 20])
    out = engine._pad_to_token_bucket(tokens, mask, lens, None, None)
    assert out[2].tolist() == [10, 20] and engine._bucket_pad_row is False


def test_bucket_appends_one_masked_sequence(engine, monkeypatch):
    monkeypatch.setenv("MLITE_PACK_BUCKET_TOKENS", "64")
    tokens, mask, lens = torch.arange(30), torch.ones(30), torch.tensor([10, 20])
    tokens, mask, lens = engine._pad_to_token_bucket(tokens, mask, lens, None, None)
    aligned = int(((lens + 7) // 8 * 8).sum())
    assert aligned % 64 == 0 and lens[:2].tolist() == [10, 20]
    assert tokens.numel() == int(lens.sum()) and mask[30:].sum() == 0
    assert engine._bucket_pad_row is True


def test_bucket_must_align_to_cp(engine, monkeypatch):
    monkeypatch.setenv("MLITE_PACK_BUCKET_TOKENS", "60")
    with pytest.raises(ValueError):
        engine._pad_to_token_bucket(torch.arange(4), None, torch.tensor([4]), None, None)
