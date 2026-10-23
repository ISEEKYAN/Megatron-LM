# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SWE-bench Pro agent loop for the DS4.1 verl recipe (legacy trainer, token in/out).

One verl sample = one uni-agent ``swe_bench_pro`` episode: the uni-agent Task owns
the sandbox (E2B), the ReAct agent and the reward (FAIL_TO_PASS + PASS_TO_PASS on
the final repo state); this loop only replaces the agent's HTTP chat client with an
in-process client that drives verl's ``LLMServerClient`` with token ids:

* the first turn renders system+user+tool schemas with the V4.1 encoder;
* each sampled turn is kept as sampled (``response_mask`` 1, rollout log-probs);
* tool results are rendered after the committed turns (``response_mask`` 0);
* every turn of a session goes to the same rollout replica (request id) and the
  same DP rank inside it (``extra_args.verl_dp_pin``, see compat), so multi-turn
  prompts hit that rank's prefix cache.

Env (all optional except the task config):
  DS41_SWE_TASK_CONFIG   uni-agent Task Config YAML (required)
  DS41_SWE_THINKING      1 = V4.1 thinking mode, 0 = chat mode (default 0)
  DS41_SWE_REASONING_EFFORT  V4.1 effort (thinking mode only)
  DS41_SWE_TRACE_DIR     per-episode JSON trace (turn timings, token counts)
  DS41_SWE_SESSION_TIMEOUT  wall deadline of one episode incl. eval (default 3600 s)
  DS41_SWE_IGNORE_EOS    1 = memory check only: every turn fills the remaining response budget
  DS41_SWE_LEASE_RENEW_S  min seconds between sandbox lease renewals before tool calls (default 60)

Tool calls are guarded like uni-agent's tool errors: an unexpected tool exception is
returned to the model as a tool error while the sandbox is alive, and fails the episode
as ``SandboxLost`` (counted as ``infra``) once it is gone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput

from .ds41_codec import DS41Codec

logger = logging.getLogger(__name__)

_SESSIONS: dict[str, "_TokenSession"] = {}
_INPROC = "inproc://"


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))


def _canon(message: dict[str, Any]) -> dict[str, Any]:
    out = {"role": message["role"], "content": message.get("content") or ""}
    if message.get("tool_calls"):
        out["tool_calls"] = [
            {"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in message["tool_calls"]
        ]
    return out


class _ContextFull(Exception):
    pass


class SandboxLost(RuntimeError):
    """The episode's sandbox is gone; the episode fails as an infrastructure failure."""


_INFRA_MARKERS = (
    "SandboxLost",
    "EvalInfraError",
    "SandboxChannelError",
    "RemoteProtocolError",
    "TimeoutException",
    "ConnectError",
    "E2BSandbox not started",
)


def failure_kind(error: str | None) -> str | None:
    """``infra`` (sandbox/eval infrastructure), ``timeout`` (episode deadline) or ``other``."""
    if error is None:
        return None
    if any(m in error for m in _INFRA_MARKERS):
        return "infra"
    if error.startswith("TimeoutError"):
        return "timeout"
    return "other"


async def _renew_lease(sandbox: Any, min_interval: float) -> None:
    lease = getattr(sandbox, "lease_seconds", None)
    handle = getattr(sandbox, "_sandbox", None)
    if not lease or handle is None:
        return
    now = time.monotonic()
    if now - getattr(sandbox, "_ds41_renewed", -1e9) < min_interval:
        return
    sandbox._ds41_renewed = now
    try:
        await asyncio.wait_for(handle.set_timeout(int(lease)), timeout=30)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 -- the keepalive task still renews
        logger.warning("sandbox %s lease renewal failed: %r", getattr(sandbox, "sandbox_id", "?"), exc)


def _install_tool_guard(min_interval: float) -> None:
    from uni_agent.tools import base

    if getattr(base.Toolbox, "_ds41_guard", False):
        return
    original = base.Toolbox.call

    async def call(self, name, args=None, *, timeout=None):
        sandbox = getattr(self._tools.get(name), "sandbox", None)
        if sandbox is not None:
            await _renew_lease(sandbox, min_interval)
        try:
            return await original(self, name, args, timeout=timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {str(exc)[:2000]}"
            if sandbox is not None and not await sandbox.is_alive():
                raise SandboxLost(f"sandbox gone during {name!r}: {detail}") from exc
            return base.ToolResult(text=f"Error: {detail}", status="error")

    base.Toolbox.call = call
    base.Toolbox._ds41_guard = True


class _TokenSession:
    """Token stream of one episode; the agent's chat client calls :meth:`query`."""

    def __init__(self, loop: "DS41SWEAgentLoop", sampling_params: dict[str, Any], key: str):
        self.loop = loop
        self.codec = loop.codec
        self.sampling_params = sampling_params
        self.key = key
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.committed: list[dict[str, Any]] = []
        self.turns: list[dict[str, Any]] = []
        self.truncated = False

    def _room(self) -> int:
        return self.loop.response_length - len(self.response_ids)

    def _extend(self, ids: list[int], mask: int, logprobs: list[float] | None = None) -> None:
        self.response_ids += ids
        self.response_mask += [mask] * len(ids)
        self.response_logprobs += logprobs if logprobs is not None else [0.0] * len(ids)

    async def query(self, messages: list[dict[str, Any]], tools: list[dict]) -> tuple[str, list[dict], dict]:
        t_in = time.time()
        if not self.committed:
            self.prompt_ids = self.codec.initial(messages, tools)
            if len(self.prompt_ids) > self.loop.prompt_length:
                raise ValueError(f"initial prompt {len(self.prompt_ids)} > prompt_length {self.loop.prompt_length}")
            appended_tokens = 0
        else:
            n = len(self.committed)
            if [_canon(m) for m in messages[:n]] != [_canon(m) for m in self.committed]:
                raise RuntimeError("agent rewrote committed history; token stream would diverge")
            appended = messages[n:]
            ids = []
            if self.response_ids[-1:] != [self.codec.eos_id]:
                ids.append(self.codec.eos_id)  # turn stopped at the length cap
            ids += self.codec.append(self.committed, appended, tools)
            if len(ids) >= self._room():
                self.truncated = True
                self.committed = list(messages)
                return "", [], {"prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "length"}
            self._extend(ids, 0)
            appended_tokens = len(ids)
        context = self.prompt_ids + self.response_ids
        params = dict(self.sampling_params)
        params["max_tokens"] = self._room()
        params["extra_args"] = {"verl_dp_pin": self.key}
        if self.loop.ignore_eos:
            params["ignore_eos"] = True
        t0 = time.time()
        out = await self.loop.server_manager.generate(request_id=self.key, prompt_ids=context, sampling_params=params)
        t1 = time.time()
        ids = list(out.token_ids)
        logprobs = list(out.log_probs) if out.log_probs is not None else None
        self._extend(ids, 1, logprobs)
        stopped_by_eos = bool(ids) and ids[-1] == self.codec.eos_id
        content, calls = self.codec.parse(ids, tools) if ids else ("", [])
        finish = "length" if not stopped_by_eos and not calls else ("tool_calls" if calls else "stop")
        if out.stop_reason == "aborted":
            finish = "length"
        assistant = {"role": "assistant", "content": content}
        if calls:
            assistant["tool_calls"] = calls
        self.committed = list(messages) + [assistant]
        self.turns.append(
            {"t_in": t_in, "t_gen0": t0, "t_gen1": t1, "ctx": len(context), "out": len(ids), "obs": appended_tokens}
        )
        info = {"prompt_tokens": len(context), "completion_tokens": len(ids), "finish_reason": finish}
        return content, calls, info


class _InprocChatModel:
    """Drop-in for uni-agent's ``OpenAICompatibleChatModel`` bound to a token session."""

    def __init__(self, base_url: str, *, tools_schemas: list[dict] | None = None, **kwargs: Any):
        self._session = _SESSIONS[base_url[len(_INPROC) :]]
        self.tools_schemas = tools_schemas or []

    async def query(self, messages, *, sampling_params=None):
        return await self._session.query(messages, self.tools_schemas)

    async def aclose(self) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


def _install_inproc_model() -> None:
    import uni_agent.agents.react.agent as react

    if getattr(react, "_ds41_inproc", False):
        return
    original = react.OpenAICompatibleChatModel

    def factory(base_url: str, **kwargs: Any):
        if base_url.startswith(_INPROC):
            return _InprocChatModel(base_url, **kwargs)
        return original(base_url, **kwargs)

    react.OpenAICompatibleChatModel = factory
    react._ds41_inproc = True


class DS41SWEAgentLoop(AgentLoopBase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        from uni_agent.tasks import TaskConfigResolver

        _install_inproc_model()
        _install_tool_guard(float(os.environ.get("DS41_SWE_LEASE_RENEW_S", "60")))
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        thinking = os.environ.get("DS41_SWE_THINKING", "0") == "1"
        effort = os.environ.get("DS41_SWE_REASONING_EFFORT") or None
        if effort and effort.isdigit():
            effort = int(effort)
        self.codec = DS41Codec(self.tokenizer, thinking=thinking, reasoning_effort=effort)
        self.resolver = TaskConfigResolver.from_file(os.environ["DS41_SWE_TASK_CONFIG"])
        self.session_timeout = float(os.environ.get("DS41_SWE_SESSION_TIMEOUT", "3600"))
        self.ignore_eos = os.environ.get("DS41_SWE_IGNORE_EOS") == "1"
        trace_dir = os.environ.get("DS41_SWE_TRACE_DIR")
        self.trace_dir = Path(trace_dir) if trace_dir else None

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        from uni_agent.tasks import get_task

        extra = _plain(kwargs["extra_info"])
        sample = dict(extra["tools_kwargs"]["task"])
        sample["prompt"] = _plain(kwargs["raw_prompt"])
        key = uuid.uuid4().hex
        task_cfg = self.resolver.resolve(sample, runtime_model={"base_url": _INPROC + key, "api_key": "EMPTY"})
        session = _TokenSession(self, sampling_params, key)
        _SESSIONS[key] = session
        t0 = time.time()
        reward, error, info = 0.0, None, {}
        try:
            result = await asyncio.wait_for(get_task(task_cfg).run(), timeout=self.session_timeout)
            reward = float(result.reward)
            info = result.extra_info or {}
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- infra failure: keep the step, drop the sample from the loss
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
            logger.warning("swe episode %s failed: %s", task_cfg.get("metadata", {}).get("instance_id"), error)
        finally:
            _SESSIONS.pop(key, None)
        wall = time.time() - t0
        kind = failure_kind(error)
        gen_s = sum(t["t_gen1"] - t["t_gen0"] for t in session.turns)
        response_mask = session.response_mask if error is None else [0] * len(session.response_ids)
        if not session.prompt_ids:
            session.prompt_ids = self.codec.initial(task_cfg["prompt"], None)[: self.prompt_length]
        if not session.response_ids:
            session._extend([self.codec.eos_id], 0)
            response_mask = [0]
        record = {
            "instance_id": task_cfg.get("metadata", {}).get("instance_id"),
            "reward": reward,
            "error": error,
            "failure_kind": kind,
            "truncated": session.truncated,
            "turns": session.turns,
            "wall_s": wall,
            "t_start": t0,
            "prompt_tokens": len(session.prompt_ids),
            "response_tokens": len(session.response_ids),
            "generated_tokens": int(sum(session.response_mask)),
            "eval_s": info.get("eval_execution_time"),
        }
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            with open(self.trace_dir / f"{os.getpid()}.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
        return AgentLoopOutput(
            prompt_ids=session.prompt_ids,
            response_ids=session.response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=session.response_logprobs[: self.response_length],
            reward_score=reward,
            num_turns=2 * len(session.turns) + 1,
            metrics=AgentLoopMetrics(generate_sequences=gen_s, tool_calls=max(0.0, wall - gen_s)),
            extra_fields={
                "swe_failed": error is not None,
                "swe_infra_failed": kind == "infra",
                "swe_truncated": session.truncated,
                "swe_turns": len(session.turns),
                "swe_wall_s": wall,
                "swe_gen_s": gen_s,
            },
        )
