# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared harness for the two Scale-Out memory experiments (arXiv:2606.02437 §5):

  * Memory-capacity law (Fig 21a): recall accuracy vs capacity C = mem_tokens / trainable_params,
    swept over (rank, n_facts). Expect a single sigmoid collapse with the knee in [1e-3, 1e-2].
  * Module ablation (Fig 21c): recall-per-param over target_modules {mlp, attn, all}.
    Expect MLP > Attn ~= All on per-param efficiency.

Synthetic key->value fact-memorization (no external data, fully seeded): each fact is a
(prompt, answer) pair whose answer is random => unmemorizable except by storing it in the
adapter. The frozen real Qwen3-30B-A3B prior is essential (small high-leverage updates).
Reuses the rsLoRA harness's mlite runtime + THD PackedBatch + fsdp2 + EP plumbing.

Run: torchrun --nproc_per_node=<tp*ep> memorize.py [args]
"""
from __future__ import annotations

import argparse
import json
import random

import torch

from megatron.lite.runtime import RuntimeConfig, create_runtime
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch

_TARGETS = {
    "mlp": ("fc1", "fc2"),
    "attn": ("qkv", "proj"),
    "all": ("qkv", "proj", "fc1", "fc2"),
}
_ADJ = ["azure", "crimson", "golden", "silent", "rapid", "hollow", "lunar", "iron", "jade", "swift"]
_NOUN = ["falcon", "harbor", "cipher", "meadow", "lantern", "quartz", "willow", "ember", "vortex", "delta"]
_COLOR = ["violet", "amber", "teal", "scarlet", "indigo", "olive", "maroon", "cyan", "coral", "slate"]


def _make_facts(n: int, seed: int):
    """N facts with UNIQUE keys; answer = one random color word (short, ~1-2 tokens) so the
    adapter can actually memorize the key->value map in a feasible #epochs. Capacity is about
    bits/param: a short answer just shifts the tokens/param constant, the law is unchanged.
    The key is unique+high-entropy (the thing that must be stored); the answer is a small
    closed set (10 colors) keyed by it -> genuine recall test, fast to fit at low C."""
    rng = random.Random(seed)
    facts = []
    seen = set()
    while len(facts) < n:
        key = f"{rng.choice(_ADJ)}-{rng.choice(_NOUN)}-{rng.randint(0, 999999):06d}"
        if key in seen:
            continue
        seen.add(key)
        facts.append((f"Fact {key}: the code is", f" {rng.randint(0, 999):03d}"))  # ~3 tok, 0.1% guess
    return facts


def _pack_facts(facts, tok, seq: int, device: str) -> tuple[PackedBatch, int]:
    """THD PackedBatch over facts; loss_mask=0 on prompt tokens, 1 on answer tokens (memorize the
    answer only). Each fact padded/truncated to `seq`. Returns (batch, total answer tokens)."""
    eos = tok.eos_token_id
    ids_all, mask_all = [], []
    ans_tokens = 0
    for prompt, answer in facts:
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(answer, add_special_tokens=False)["input_ids"] + ([eos] if eos is not None else [])
        toks = (p + a)[:seq]
        mask = ([0.0] * len(p) + [1.0] * len(a))[:seq]
        if len(toks) < seq:  # right-pad to uniform length (masked out)
            pad = seq - len(toks)
            toks = toks + [eos if eos is not None else 0] * pad
            mask = mask + [0.0] * pad
        ids_all.extend(toks)
        mask_all.extend(mask)
        ans_tokens += int(sum(mask))
    n = len(facts)
    ids = torch.tensor(ids_all, dtype=torch.long, device=device)
    return (
        PackedBatch(
            input_ids=ids,
            labels=ids.clone(),  # roll_labels does the next-token shift
            seq_lens=torch.full((n,), seq, dtype=torch.int64, device=device),
            loss_mask=torch.tensor(mask_all, dtype=torch.float32, device=device),
        ),
        ans_tokens,
    )


def _global_trainable_numel(model, ep: int) -> int:
    """Honest global trainable-param count: attention LoRA is REPLICATED across EP ranks (count
    once), expert LoRA is EP-distinct (x ep). Names: *_lora.* under attn vs experts."""
    attn_local = expert_local = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "fc1_lora" in name or "fc2_lora" in name or ".experts" in name:
            expert_local += p.numel()
        else:
            attn_local += p.numel()
    return attn_local + expert_local * max(1, ep)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-path", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--alpha", type=int, default=None, help="default = 2*rank (standard scale 2)")
    p.add_argument("--targets", default="all", choices=list(_TARGETS))
    p.add_argument("--n-facts", type=int, required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--micro-bs", type=int, default=16, help="facts per microbatch")
    p.add_argument("--eval-n", type=int, default=512, help="recall sampled on min(n_facts, eval_n)")
    p.add_argument("--recall-thresh", type=float, default=0.6931, help="per-fact mean answer NLL <= this = recalled (prob>=0.5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--etp", type=int, default=1)
    p.add_argument("--model-name", default="qwen3_moe", choices=["qwen3_moe", "qwen2"],
                   help="qwen2 = cheap dense proxy for the capacity law / module ablation")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda"
    alpha = args.alpha if args.alpha is not None else 2 * args.rank
    backend_cfg = MegatronLiteConfig(
        model_name=args.model_name,
        impl="lite",
        hf_path=args.hf_path,
        parallel=ParallelConfig(tp=args.tp, etp=args.etp, ep=args.ep, pp=1, cp=1),
        optimizer=OptimizerConfig(lr=args.lr, min_lr=args.lr, weight_decay=0.0),
        load_hf_weights=True,
        impl_cfg={
            "lora": {"rank": args.rank, "alpha": alpha, "target_modules": _TARGETS[args.targets]},
            "optimizer": "fsdp2",
            "deterministic": True,
        },
    )
    rt = create_runtime(RuntimeConfig(backend="mlite", hf_path=args.hf_path, backend_cfg=backend_cfg))
    handle = rt.build_model()
    print(f"[stage] model built (targets={args.targets} rank={args.rank} n={args.n_facts})", flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_path)
    facts = _make_facts(args.n_facts, args.seed)
    train_pb, mem_tokens = _pack_facts(facts, tok, args.seq_len, device)

    def _microbatches(pb):
        n = int(pb.seq_lens.shape[0])
        s = args.seq_len
        out = []
        for i in range(0, n, args.micro_bs):
            j = min(i + args.micro_bs, n)
            out.append(
                PackedBatch(
                    input_ids=pb.input_ids[i * s : j * s],
                    labels=pb.labels[i * s : j * s],
                    seq_lens=pb.seq_lens[i:j],
                    loss_mask=pb.loss_mask[i * s : j * s],
                )
            )
        return out

    train_micro = _microbatches(train_pb)
    n_micro = len(train_micro)

    def _iter():
        i = 0
        while True:
            yield train_micro[i % n_micro]
            i += 1

    data_iter = _iter()

    def _result_loss(res) -> float:
        # Post-#68 contract: with loss_fn=None the loss lives in model_output.loss;
        # metrics only carries loss_fn rows (same fix as the rslora sweep, 4cfceb0f7).
        loss = getattr(res.model_output, "loss", None)
        if loss is not None:
            return float(loss.item() if hasattr(loss, "item") else loss)
        return float(res.metrics.get("loss", float("nan")))

    # Mini-batch SGD (num_microbatches=1 per step): decouples per-step cost from N so large fact
    # sets (needed to reach the high-capacity collapse regime) stay affordable. `steps` is the
    # total microbatch budget; epochs over the fact set = steps * micro_bs / n_facts.
    losses = []
    every = max(1, args.steps // 10)
    with rt.train_mode(handle):
        for step in range(args.steps):
            rt.zero_grad(handle)
            res = rt.forward_backward(handle, data_iter, loss_fn=None, num_microbatches=1)
            rt.optimizer_step(handle)
            rt.lr_scheduler_step(handle)
            losses.append(_result_loss(res))
            if step % every == 0 or step == args.steps - 1:
                print(f"[train] step {step}/{args.steps} loss={losses[-1]:.4f}", flush=True)

    # ---- recall eval: per-fact mean answer NLL (== metrics["loss"] with 1 fact/forward) ----
    eval_facts = facts[: min(args.eval_n, args.n_facts)]
    recalled = 0
    nll_sum = 0.0
    nll_n = nan_n = 0
    with rt.eval_mode(handle):
        for f in eval_facts:
            pb1, _ = _pack_facts([f], tok, args.seq_len, device)
            res = rt.forward_backward(handle, iter([pb1]), loss_fn=None, num_microbatches=1, forward_only=True)
            nll = _result_loss(res)
            if nll == nll:
                nll_sum += nll
                nll_n += 1
                if nll <= args.recall_thresh:
                    recalled += 1
            else:
                nan_n += 1
    recall_acc = recalled / max(1, len(eval_facts))
    mean_eval_nll = nll_sum / nll_n if nll_n else float("nan")
    print(f"[eval] recall={recall_acc:.3f} mean_nll={mean_eval_nll:.4f} nan={nan_n}/{len(eval_facts)}", flush=True)

    import torch.distributed as dist

    trainable = _global_trainable_numel(handle._model, args.ep)
    if not dist.is_initialized() or dist.get_rank() == 0:
        rec = {
            "targets": args.targets,
            "rank": args.rank,
            "alpha": alpha,
            "n_facts": args.n_facts,
            "trainable_numel": trainable,
            "mem_tokens": mem_tokens,
            "capacity": mem_tokens / max(1, trainable),
            "recall_acc": recall_acc,
            "mean_eval_nll": mean_eval_nll,
            "eval_nan": nan_n,
            "eval_n": len(eval_facts),
            "final_train_loss": losses[-1] if losses else None,
            "min_train_loss": min([x for x in losses if x == x], default=None),
        }
        with open(args.out, "w") as fh:
            json.dump(rec, fh)
        mt = rec["min_train_loss"]
        print(
            f"DONE targets={args.targets} rank={args.rank} n={args.n_facts} "
            f"params={trainable} C={rec['capacity']:.2e} recall={recall_acc:.3f} "
            f"min_train={'none' if mt is None else f'{mt:.4f}'}",
            flush=True,
        )


if __name__ == "__main__":
    import sys
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
