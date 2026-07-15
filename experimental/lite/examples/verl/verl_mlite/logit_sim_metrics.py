# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TASK-1.1.12 拆桥验证 step-3 · logit-similarity 核心比对(纯函数,可 CPU 单测)。

比对同一批 token 上两侧逐 token logprob:
  - rollout 侧 = vLLM(fp8 resync 后权重)每 token logprob
  - actor 侧   = mlite(bf16 actor 权重)compute_log_prob 每 token logprob
block-scale 逐块对位对 -> 两侧高度相似(仅 fp8 量级差);对位错 -> 发散。

这里只做数学;取数/落盘/编排在 logit_sim_probe.py。纯 torch,无框架依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch

# 判据(fp8 量级容差,对位错时差一个数量级,阈值放得宽也能区分):
COSINE_PASS = 0.98          # 逐样本 logprob 向量 cosine 均值
REL_L2_PASS = 0.05          # 相对 L2 误差(响应 token 上) <=5%
CORR_PASS = 0.99            # Pearson 相关


@dataclass
class LogitSimResult:
    n_tokens: int
    cosine_mean: float
    rel_l2: float
    pearson: float
    max_abs_diff: float
    mean_abs_diff: float
    verdict: str            # "PASS" / "FAIL" / "EMPTY"

    def as_dict(self) -> dict:
        return asdict(self)


def _masked_select(logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B,T] logprob + [B,T] {0,1} mask -> 1-D flat 有效 token logprob。"""
    return logprobs.float()[mask.bool()]


def compute_logit_similarity(
    rollout_logprobs: torch.Tensor,
    actor_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
) -> LogitSimResult:
    """两侧 [B,T] 逐 token logprob(响应位) + [B,T] mask -> 相似度指标 + 判决。

    logprob 空间比对(= softmax 后 logits);对位错=权重乱=logprob 灾难性发散。
    """
    assert rollout_logprobs.shape == actor_logprobs.shape == response_mask.shape, (
        rollout_logprobs.shape, actor_logprobs.shape, response_mask.shape
    )
    r = _masked_select(rollout_logprobs, response_mask)
    a = _masked_select(actor_logprobs, response_mask)
    n = int(r.numel())
    if n == 0:
        return LogitSimResult(0, 0.0, float("inf"), 0.0, float("inf"), float("inf"), "EMPTY")

    # 逐样本(整个响应序列作为一个向量)cosine 更稳,这里用全体 flat 的单向量 cosine
    cos = torch.nn.functional.cosine_similarity(r, a, dim=0).item()
    rel_l2 = (torch.linalg.vector_norm(r - a) / torch.linalg.vector_norm(a).clamp_min(1e-12)).item()
    # Pearson
    rc, ac = r - r.mean(), a - a.mean()
    denom = (torch.linalg.vector_norm(rc) * torch.linalg.vector_norm(ac)).clamp_min(1e-12)
    pearson = (rc @ ac / denom).item()
    max_abs = (r - a).abs().max().item()
    mean_abs = (r - a).abs().mean().item()

    ok = (cos >= COSINE_PASS) and (rel_l2 <= REL_L2_PASS) and (pearson >= CORR_PASS)
    return LogitSimResult(
        n_tokens=n, cosine_mean=cos, rel_l2=rel_l2, pearson=pearson,
        max_abs_diff=max_abs, mean_abs_diff=mean_abs,
        verdict="PASS" if ok else "FAIL",
    )
