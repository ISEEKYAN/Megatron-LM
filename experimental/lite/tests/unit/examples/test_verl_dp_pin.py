# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A session key in SamplingParams.extra_args pins every turn to one DP rank (prefix-cache reuse)."""

import asyncio
import sys
import types
from types import SimpleNamespace

from verl_mlite import compat


def _fake_vllm(monkeypatch, dp_size):
    seen = []

    class AsyncLLM:
        vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=dp_size))

        async def generate(self, prompt, sampling_params, request_id, *, data_parallel_rank=None):
            seen.append(data_parallel_rank)
            yield request_id

    module = types.ModuleType("vllm.v1.engine.async_llm")
    module.AsyncLLM = AsyncLLM
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", module)
    monkeypatch.setattr(compat, "_vllm_importable", lambda: True)
    monkeypatch.setenv("VERL_MLITE_DP_PIN", "1")
    assert compat._patch_vllm_dp_pin_by_session()
    return AsyncLLM(), seen


def _run(engine, extra_args, **kwargs):
    async def go():
        params = SimpleNamespace(extra_args=extra_args)
        return [o async for o in engine.generate("p", params, "r", **kwargs)]

    return asyncio.run(go())


def test_same_session_same_rank_and_explicit_rank_wins(monkeypatch):
    engine, seen = _fake_vllm(monkeypatch, dp_size=4)
    for _ in range(3):
        _run(engine, {"verl_dp_pin": "session-a"})
    assert len(set(seen)) == 1 and 0 <= seen[0] < 4
    ranks = set()
    for i in range(64):
        _run(engine, {"verl_dp_pin": f"s{i}"})
        ranks.add(seen[-1])
    assert ranks == {0, 1, 2, 3}
    _run(engine, {"verl_dp_pin": "session-a"}, data_parallel_rank=2)
    assert seen[-1] == 2
    _run(engine, None)
    assert seen[-1] is None
