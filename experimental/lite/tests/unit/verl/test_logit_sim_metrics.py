# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU 单测: logit-similarity 核心比对能区分 fp8 噪声(PASS) vs 对位错(FAIL)。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# verl_mlite 在 examples/verl 下
_VERL_EX = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(_VERL_EX) not in sys.path:
    sys.path.insert(0, str(_VERL_EX))

from verl_mlite.logit_sim_metrics import compute_logit_similarity  # noqa: E402


def _mask(B=4, T=16):
    m = torch.ones(B, T)
    m[:, :4] = 0  # 前 4 位当 prompt(不计)
    return m


def test_pass_under_fp8_level_noise():
    """两侧 logprob 仅差 fp8 量级噪声 -> PASS。"""
    torch.manual_seed(0)
    actor = torch.randn(4, 16) * 2 - 3  # 典型 logprob 尺度(负)
    rollout = actor + torch.randn(4, 16) * 0.02  # ~fp8 级扰动
    res = compute_logit_similarity(rollout, actor, _mask())
    assert res.verdict == "PASS", res.as_dict()
    assert res.cosine_mean >= 0.98 and res.pearson >= 0.99 and res.rel_l2 <= 0.05


def test_fail_on_block_scale_mispairing_garbage():
    """对位错 = 权重乱 = logprob 与 actor 无关(随机) -> FAIL。"""
    torch.manual_seed(1)
    actor = torch.randn(4, 16) * 2 - 3
    rollout = torch.randn(4, 16) * 2 - 3  # 独立随机(模拟对位错的灾难性发散)
    res = compute_logit_similarity(rollout, actor, _mask())
    assert res.verdict == "FAIL", res.as_dict()


def test_empty_mask_is_empty_not_pass():
    actor = torch.randn(4, 16)
    res = compute_logit_similarity(actor, actor.clone(), torch.zeros(4, 16))
    assert res.verdict == "EMPTY"


def test_identical_is_pass():
    a = torch.randn(2, 8) * 2 - 3
    res = compute_logit_similarity(a.clone(), a, torch.ones(2, 8))
    assert res.verdict == "PASS" and res.rel_l2 < 1e-6
