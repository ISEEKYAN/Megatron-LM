# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TASK-1.1.12.8 · 复现 DS4 THD packing 不一致崩(seq_len 206 > packed tensor 128)。

``_nested_from_packed_tensor(tensor, seq_lens)`` 逐 seq narrow 一个 1-D packed
tensor。当某 seq_len 超过 packed tensor 剩余长度(seq_lens 与 input_ids.numel()
不一致)时,``tensor.narrow(0, offset, length)`` 直接抛 RuntimeError
"...exceeds dimension size..."——这就是真实训练里 206>128 的崩点。

本测只【复现】(证明该不一致态必崩),不修。root(为何真实训练出现此不一致)见 step-2。
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.model.deepseek_v4.lite.protocol import _nested_from_packed_tensor


def test_reproduce_seqlen_exceeds_packed_tensor_206_over_128():
    """packed tensor 只有 128 token,但 seq_lens 声明 206 -> narrow 越界崩(复现)。"""
    packed = torch.arange(128, dtype=torch.long)          # 128 个 token
    seq_lens = torch.tensor([206], dtype=torch.long)       # 声明 206(> 128)
    with pytest.raises(RuntimeError) as ei:
        _nested_from_packed_tensor(packed, seq_lens)
    msg = str(ei.value).lower()
    assert "exceeds" in msg or "size" in msg or "range" in msg, str(ei.value)


def test_reproduce_multi_seq_second_overruns():
    """多 seq: 第二个 seq 使累计越过 tensor 尾 -> 复现同类崩。"""
    packed = torch.arange(128, dtype=torch.long)
    seq_lens = torch.tensor([100, 128], dtype=torch.long)  # 100+128=228 > 128
    with pytest.raises(RuntimeError):
        _nested_from_packed_tensor(packed, seq_lens)


def test_sum_mismatch_underrun_raises_valueerror():
    """seq_lens 和 < numel(每段都放得下但没铺满)-> offset!=numel 的 ValueError。

    区分两类不一致: 越界=RuntimeError(narrow); 欠铺=ValueError(sum 校验)。
    """
    packed = torch.arange(128, dtype=torch.long)
    seq_lens = torch.tensor([100], dtype=torch.long)       # 100 < 128
    with pytest.raises(ValueError, match="sizes sum to 100"):
        _nested_from_packed_tensor(packed, seq_lens)


def test_consistent_packing_succeeds():
    """一致态(sum(seq_lens)==numel) -> 正常返回 nested tensor(对照,证崩因是不一致非函数本身坏)。"""
    packed = torch.arange(128, dtype=torch.long)
    seq_lens = torch.tensor([50, 78], dtype=torch.long)    # 50+78=128 ==
    out = _nested_from_packed_tensor(packed, seq_lens)
    assert out is not None
    assert out.size(0) == 2                                 # 2 段
