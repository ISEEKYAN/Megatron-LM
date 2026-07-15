# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TASK-1.1.12 拆桥验证 step-3 · post-resync in-process logit-similarity 探针。

env-gate ``MLITE_LOGIT_SIM_PROBE=1``。在 ``ActorRolloutRefWorker.update_weights``
(resync 完成)后一次性触发:
  1. 建固定短 prompt 批(短 prompt + 短 max_tokens 避 THD packing bug);
  2. ``worker.rollout.generate_sequences`` -> responses + rollout_log_probs (vLLM fp8);
  3. ``worker.compute_log_prob`` 喂上一步产出 -> old_log_probs (mlite bf16 同 token);
  4. compute_logit_similarity 比 rollout vs actor logprob -> 判决 + 落盘;
  5. 探针后(可选)干净退出,不进训练步 -> 天然避开 packing bug。

契约点(首跑核实,失败 loud 不静默):
  [C1] worker.tokenizer / worker.rollout / worker.compute_log_prob 存在;
  [C2] generate_sequences 输入 prompt DataProto 键: input_ids/attention_mask/position_ids;
  [C3] 产出键: responses, rollout_log_probs (需 calculate_log_probs), response_mask/attention_mask;
  [C4] compute_log_prob 产出 old_log_probs [B,R]。
"""

from __future__ import annotations

import json
import os

import torch

from verl_mlite.logit_sim_metrics import compute_logit_similarity

_PROBE_DONE = False

# 固定短 prompt(确定性;短以避 THD packing 206>128)。
_PROBE_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "The sun rises in the",
    "A triangle has three",
    "The first president of the United States was",
    "Roses are red, violets are",
]
_PROBE_MAX_NEW_TOKENS = 16     # 短响应,避 packing
_PROBE_PROMPT_PAD = 32         # 左 pad 到定长(总 <=48 << 128)


def _log(msg: str) -> None:
    print(f"VERL_MLITE_LOGIT_SIM_PROBE {msg}", flush=True)


def _build_prompt_dataproto(worker):
    """[C1][C2] 建 prompt DataProto(input_ids 左 pad + mask + position_ids)。"""
    from verl.protocol import DataProto

    tok = worker.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids_list = [tok(p, add_special_tokens=False)["input_ids"] for p in _PROBE_PROMPTS]
    L = _PROBE_PROMPT_PAD
    input_ids, attn = [], []
    for ids in ids_list:
        ids = ids[-L:]
        padn = L - len(ids)
        input_ids.append([pad_id] * padn + ids)          # 左 pad
        attn.append([0] * padn + [1] * len(ids))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attn, dtype=torch.long)
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0) * attention_mask
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    dp = DataProto.from_single_dict(batch)
    dp.meta_info.update({
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": pad_id,
        "do_sample": False,                # greedy: 两侧同 token 可比
        "response_length": _PROBE_MAX_NEW_TOKENS,
        "calculate_log_probs": True,       # [C3] 让 vLLM 返回 rollout_log_probs
        "validate": True,
    })
    return dp


def _extract(gen, lp_out):
    """[C3][C4] 取 rollout_log_probs / old_log_probs / response_mask -> [B,T] 对齐。"""
    def _get(d, k):
        b = getattr(d, "batch", d)
        return b[k] if k in b.keys() else None

    rollout = _get(gen, "rollout_log_probs")
    actor = _get(lp_out, "old_log_probs")
    if actor is None:
        actor = _get(lp_out, "log_probs")
    rmask = _get(gen, "response_mask")
    if rmask is None:
        responses = _get(gen, "responses")
        attn = _get(gen, "attention_mask")
        R = responses.shape[1]
        rmask = attn[:, -R:] if attn is not None else torch.ones_like(responses)
    if rollout is None or actor is None:
        raise RuntimeError(
            f"[C3/C4] missing logprob tensors: rollout={rollout is not None} "
            f"actor={actor is not None} (keys gen={list(getattr(gen,'batch',gen).keys())})"
        )
    return rollout.float().cpu(), actor.float().cpu(), rmask.float().cpu()


def run_logit_sim_probe(worker) -> None:
    global _PROBE_DONE
    if _PROBE_DONE:
        return
    _PROBE_DONE = True
    try:
        from verl.utils.vllm.vllm_dsv4_fp8_utils import is_deepseek_v4_model  # DS4-only 保险
    except Exception:
        is_deepseek_v4_model = None

    try:
        _log("start (post-resync)")
        prompts = _build_prompt_dataproto(worker)
        gen = worker.rollout.generate_sequences(prompts)          # vLLM fp8
        lp_out = worker.compute_log_prob(gen)                      # mlite bf16 同 token
        rollout, actor, rmask = _extract(gen, lp_out)
        res = compute_logit_similarity(rollout, actor, rmask)
        _log(f"RESULT {json.dumps(res.as_dict())}")
        _log(f"VERDICT {res.verdict} "
             f"(cosine={res.cosine_mean:.4f} rel_l2={res.rel_l2:.4f} pearson={res.pearson:.4f} "
             f"n_tokens={res.n_tokens})")
        out_dir = os.environ.get("RUN_ROOT") or os.environ.get("MLITE_LOGIT_SIM_OUT") or "."
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        except Exception:
            rank = 0
        path = os.path.join(out_dir, f"logit_sim_probe_rank{rank}.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({
                "result": res.as_dict(),
                "rollout_logprobs": rollout[rmask.bool()].tolist()[:512],
                "actor_logprobs": actor[rmask.bool()].tolist()[:512],
            }) + "\n")
        _log(f"dumped {path}")
    except Exception as e:  # 失败 loud,不静默伪绿
        import traceback
        _log(f"ERROR {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if os.environ.get("MLITE_LOGIT_SIM_PROBE_EXIT", "1") == "1":
            _log("clean-exit after probe (avoids downstream packing bug)")
            os._exit(0)
