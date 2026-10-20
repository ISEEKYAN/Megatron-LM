# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Env-gated evidence dumps for the train/inference two-step RL gate (C8).

Enabled only when ``VERL_MLITE_C8_DIR`` is set. Writes per-process files:

* ``logprob/``: every actor micro-batch's token log-probs next to the rollout
  log-probs and masks, tagged with grad mode (old-logprob ``no_grad`` path vs the
  grad-enabled training forward) and the actor's optimizer-step count;
* ``optim/``: per optimizer step, finite/non-zero gradient coverage and how many
  model parameters changed (bitwise) with the update;
* ``sync/``: per rollout weight sync, a fingerprint of the payload the worker
  received and of the model state after the layerwise reload (and, on the first
  GPU of each node, the payload itself for an offline cold-load comparison).
"""

import json
import os
import socket
import time

import torch

_STATE = {"optim_steps": 0, "seq": 0, "sync": 0, "suppress": False, "packab": 0}


def enabled():
    return bool(os.environ.get("VERL_MLITE_C8_DIR"))


def _dir(sub):
    path = os.path.join(os.environ["VERL_MLITE_C8_DIR"], sub)
    os.makedirs(path, exist_ok=True)
    return path


def _tag():
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    return f"{socket.gethostname()}_r{rank}_p{os.getpid()}"


def _cpu(t):
    if t is None or not isinstance(t, torch.Tensor):
        return t
    if getattr(t, "is_nested", False):
        return {"values": t.values().detach().cpu(), "offsets": t.offsets().cpu()}
    return t.detach().cpu()


def dump_logprobs(model_output, micro_batch, forward_only):
    """Record one actor micro-batch (called from the engine's runtime loss hook)."""
    if not enabled() or _STATE["suppress"]:
        return
    from verl.workers.utils.padding import no_padding_2_padding

    lp = model_output["log_probs"]
    record = {
        "optim_step": _STATE["optim_steps"],
        "forward_only": bool(forward_only),
        "grad_enabled": torch.is_grad_enabled(),
        "requires_grad": bool(lp.requires_grad),
        "dtype": str(lp.dtype),
        "log_probs_padded": _cpu(no_padding_2_padding(lp, micro_batch)),
        "input_ids": _cpu(micro_batch["input_ids"]),
        "n_docs": len(micro_batch),
        "n_tokens": int(micro_batch["input_ids"].offsets()[-1]),
    }
    for key in ("rollout_log_probs", "response_mask", "old_log_probs", "advantages"):
        if key in micro_batch.keys():
            record[key] = _cpu(micro_batch[key])
    _STATE["seq"] += 1
    torch.save(record, os.path.join(_dir("logprob"), f"{_tag()}_{_STATE['seq']:05d}.pt"))


def _param_sums(params):
    from verl_mlite.c8_fingerprint import checksum

    return [tuple(checksum(p)[3:]) for p in params]


def _pinned_stats():
    try:
        st = torch.cuda.host_memory_stats()
    except Exception:
        return None
    return {
        k: v
        for k, v in st.items()
        if k.split(".")[0] in ("allocated_bytes", "reserved_bytes")
        and k.endswith(("current", "peak"))
    }


def _host_bytes(optimizer):
    """Bytes of CPU tensors held by the optimizer (params/master, grads, state), by kind and dtype."""
    out, seen = {}, set()

    def add(kind, t):
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu" or id(t) in seen:
            return
        seen.add(id(t))
        key = f"{kind}:{str(t.dtype)[6:]}{':pinned' if t.is_pinned() else ''}"
        out[key] = out.get(key, 0) + t.untyped_storage().nbytes()

    stack = [optimizer]
    while stack:
        opt = stack.pop()
        if opt is None:
            continue
        chained = getattr(opt, "chained_optimizers", None)
        if chained:
            stack.extend(chained)
            continue
        try:
            inner = getattr(opt, "optimizer", None)
        except Exception:
            inner = None
        if inner is not None and inner is not opt:
            stack.append(inner)
        stack.extend(getattr(opt, "sub_optimizers", []) or [])
        for group in getattr(opt, "param_groups", []) or []:
            for p in group.get("params", []):
                add("param", p)
                add("grad", getattr(p, "grad", None))
        for st in (getattr(opt, "state", {}) or {}).values():
            if isinstance(st, dict):
                for v in st.values():
                    add("state", v)
    return out


def optimizer_step(handle, step_fn):
    """Wrap one optimizer step with gradient-coverage and parameter-change evidence."""
    if not enabled():
        return step_fn()
    chunks = handle._model if isinstance(handle._model, (list, tuple)) else [handle._model]
    params = [p for chunk in chunks for p in chunk.parameters() if p.requires_grad]
    finite = nonzero = with_grad = 0
    sq = 0.0
    for p in params:
        g = getattr(p, "main_grad", None)
        if g is None:
            g = p.grad
        if g is None:
            continue
        with_grad += 1
        gf = g.float()
        finite += bool(torch.isfinite(gf).all())
        nz = bool((gf != 0).any())
        nonzero += nz
        sq += float(gf.pow(2).sum())
    before = _param_sums(params)
    result = step_fn()
    torch.cuda.synchronize()
    after = _param_sums(params)
    changed = sum(a != b for a, b in zip(before, after))
    _STATE["optim_steps"] += 1
    record = {
        "optim_step": _STATE["optim_steps"],
        "params": len(params),
        "with_grad": with_grad,
        "grad_all_finite": finite,
        "grad_nonzero": nonzero,
        "local_grad_sq": sq,
        "params_changed": changed,
        "host_optimizer_bytes": _host_bytes(handle._optimizer),
        "trainable_numel": sum(p.numel() for p in params),
        "pinned_host_allocator": _pinned_stats(),
        "non_bf16_params": (
            sorted(
                f"{n}:{str(p.dtype)[6:]}"
                for chunk in chunks
                for n, p in chunk.named_parameters()
                if p.requires_grad and p.dtype != torch.bfloat16
            )
            if _STATE["optim_steps"] == 1
            else None
        ),
        "step_result": [
            (
                float(x)
                if isinstance(x, (int, float)) or torch.is_tensor(x) and x.numel() == 1
                else str(x)
            )
            for x in (result if isinstance(result, tuple) else (result,))
        ],
        "ts": time.time(),
    }
    with open(os.path.join(_dir("optim"), f"{_tag()}.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    return result


def _fp(t):
    from verl_mlite.c8_fingerprint import checksum

    return checksum(t)


def _dp_rank():
    try:
        from vllm.distributed.parallel_state import get_dp_group

        return get_dp_group().rank_in_group
    except Exception:
        return -1


class SyncRecorder:
    """Rollout-worker side of one weight sync."""

    def __init__(self):
        _STATE["sync"] += 1
        self.version = _STATE["sync"]
        self.payload = {}
        self.dp = _dp_rank()
        self.save = self.dp == 0 and os.environ.get("VERL_MLITE_C8_SAVE_PAYLOAD") == "1"
        self.saved = {}

    def add(self, weights):
        for name, tensor in weights:
            self.payload[name] = _fp(tensor)
            if self.save:
                self.saved[name] = tensor.detach().to("cpu", copy=True)

    def finish(self, model):
        from verl_mlite.c8_fingerprint import fingerprint_model

        tag = f"{socket.gethostname()}_dp{self.dp}_v{self.version}"
        out = _dir("sync")
        with open(os.path.join(out, f"{tag}.payload.json"), "w") as f:
            json.dump(self.payload, f)
        with open(os.path.join(out, f"{tag}.state.json"), "w") as f:
            json.dump(fingerprint_model(model), f)
        if self.saved:
            from safetensors.torch import save_file

            root = os.environ.get("VERL_MLITE_C8_PAYLOAD_DIR", os.environ["VERL_MLITE_C8_DIR"])
            os.makedirs(os.path.join(root, f"payload_v{self.version}"), exist_ok=True)
            path = os.path.join(root, f"payload_v{self.version}", "payload.safetensors")
            save_file(self.saved, path)
            self.saved = {}


def _seq_lens(data):
    return (data["input_ids"].offsets()[1:] - data["input_ids"].offsets()[:-1]).tolist()


def pack_ab(engine, data, loss_function, result):
    """Packing A/C check on the old-logprob path (``Y_C8_PACK_AB=1``).

    Recomputes the same batch under (a) one sequence per micro-batch, (b) the smallest legal
    dynamic budget ``ceil(max_len/sp)`` and (c) the recipe budget with the document order
    reversed, and compares every token's log-prob bits with the production pass. Every rank
    runs the recomputes (CP ranks share the collectives); only ranks holding outputs compare.
    """
    if not enabled() or os.environ.get("Y_C8_PACK_AB") != "1" or _STATE["suppress"]:
        return
    from verl.utils import tensordict_utils as tu

    base = (result or {}).get("model_output", {}).get("log_probs")
    n = len(data)
    sp = tu.get_non_tensor_data(data=data, key="sp_size", default=1)
    budget = int(tu.get_non_tensor_data(data=data, key="max_token_len_per_gpu", default=0))
    lens = _seq_lens(data)
    variants = {
        "one_seq": (list(range(n)), None),
        "min_budget": (list(range(n)), -(-max(lens) // sp)),
        "reversed_order": (list(range(n))[::-1], budget),
    }
    rec = {
        "optim_step": _STATE["optim_steps"],
        "n_docs": n,
        "seq_lens": lens,
        "sp": sp,
        "budget_per_gpu": budget,
        "variants": {},
    }
    base_rows = None if base is None else base.unbind()
    _STATE["suppress"] = True
    try:
        for name, (perm, b) in variants.items():
            d2 = tu.index_select_tensor_dict(data, perm)
            if b is None:
                tu.assign_non_tensor(d2, use_dynamic_bsz=False, micro_batch_size_per_gpu=1)
            else:
                tu.assign_non_tensor(d2, max_token_len_per_gpu=b)
            groups = []
            _STATE["groups"] = groups
            out = engine.forward_backward_batch(d2, loss_function, forward_only=True)
            if base_rows is None:
                continue
            rows = out["model_output"]["log_probs"].unbind()
            mism, maxd, tok = 0, 0.0, 0
            for i, j in enumerate(perm):
                a, c = base_rows[j], rows[i]
                tok += a.numel()
                if a.shape != c.shape or a.dtype != c.dtype:
                    mism += a.numel()
                    continue
                ne = a.contiguous().view(torch.int32).ne(c.contiguous().view(torch.int32))
                mism += int(ne.sum())
                if ne.any():
                    maxd = max(maxd, float((a.float() - c.float()).abs().max()))
            rec["variants"][name] = {
                "budget_per_gpu": b,
                "tokens": tok,
                "mismatch": mism,
                "max_abs_diff": maxd,
                "micro_batches": groups,
            }
    finally:
        _STATE["suppress"] = False
        _STATE.pop("groups", None)
    if base_rows is None:
        return
    _STATE["packab"] += 1
    with open(os.path.join(_dir("packab"), f"{_tag()}_{_STATE['packab']:03d}.json"), "w") as f:
        json.dump(rec, f)


def note_micro_batches(micro_batches):
    """Record (docs, tokens) per micro-batch for the active pack_ab variant."""
    groups = _STATE.get("groups")
    if groups is not None:
        groups.extend((len(mb), int(mb["input_ids"].offsets()[-1])) for mb in micro_batches)
