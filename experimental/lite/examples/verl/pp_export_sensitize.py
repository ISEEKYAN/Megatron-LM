# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""8-GPU pp>1 HF-export sensitization harness (peak + timing + fingerprint).

The streaming pp>1 export path in
``megatron.lite.primitive.ckpt.hf_weights.export_hf_weights`` only runs when
``pp_size > 1``.  Existing GPU harnesses (resync_tp4 / hopper_resync) run at
pp=1/TP4 and never reach that branch, so they cannot sensitize the change.

This standalone harness builds a *tiny random* model (no checkpoint required —
same tiny configs the save/load/export smoke uses) with a pp=2 topology across
8 ranks, then fully consumes ``export_hf_weights`` and records, on rank 0:

  * peak CUDA memory during the export  (arm 1: DS4 streaming-peak MEMLOG)
  * export wall-clock time              (arm 2: qwen3.5 old/new A/B timing)
  * a per-tensor fingerprint            (arm 2: old/new bitwise equivalence)

The same file, run against the legacy ``all_gather_object`` checkout (OLD) and
the streaming checkout (NEW), yields the A/B: NEW must not regress wall-clock by
>5% and the fingerprints must match exactly.

Run: ``torchrun --standalone --nproc_per_node=8 -m examples.verl.pp_export_sensitize \
        --model deepseek_v4 --tag new --out /path/out.json``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig


# ── tiny model registry (verbatim from tests/smoke/.../test_save_load_export_smoke.py)
# Each builder imports its env-specific deps lazily so a wrong-env run reports a
# clean skip instead of a hard error.
_TP1_ONLY = {"glm5", "deepseek_v4"}


def _build_qwen3_5():
    import fla  # noqa: F401  (GatedDeltaNet / FLA stack)
    import transformer_engine.pytorch as te  # noqa: F401

    from megatron.lite.model.qwen3_5.config import Qwen35Config
    from megatron.lite.model.qwen3_5.lite import protocol

    cfg = Qwen35Config(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["full_attention", "linear_attention"],
        partial_rotary_factor=1.0,
        max_position_embeddings=4096,
    )
    return cfg, protocol


def _build_deepseek_v4():
    import cudnn  # noqa: F401  (fused DSA stack)
    import transformer_engine.pytorch as te  # noqa: F401

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite import protocol

    cfg = DeepseekV4Config(
        vocab_size=64,
        hidden_size=128,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=64,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        max_position_embeddings=4096,
        compress_ratios=[4, 4],
        sliding_window=128,
        num_hash_layers=2,
        hc_mult=2,
        index_head_dim=64,
        index_n_heads=8,
        index_topk=512,
        num_nextn_predict_layers=1,
        rms_norm_eps=1e-6,
    )
    return cfg, protocol


_BUILDERS = {
    "qwen3_5": _build_qwen3_5,
    "deepseek_v4": _build_deepseek_v4,
}


def _topology(model_name: str) -> ParallelConfig:
    forced = os.environ.get("MLITE_FORCE_TOPO")
    if forced:
        tp, ep, etp, pp, cp = (int(x) for x in forced.split(","))
        return ParallelConfig(tp=tp, ep=ep, etp=etp, pp=pp, cp=cp)
    if model_name in _TP1_ONLY:  # CSA/DSA are TP=1 only: tp1·ep2·pp2·cp1·dp4 = 8
        return ParallelConfig(tp=1, ep=2, etp=1, pp=2, cp=1)
    # tp2·ep2·pp2·cp1·dp2 = 8
    return ParallelConfig(tp=2, ep=2, etp=1, pp=2, cp=1)


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        optimizer="adam",
        lr=1.0e-3,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        clip_grad=1.0,
        offload_fraction=0.0,
    )


def _build_handle(model_name: str):
    cfg, protocol = _BUILDERS[model_name]()
    torch.manual_seed(4242)
    torch.cuda.manual_seed_all(4242)
    parallel = _topology(model_name)
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer="dist_opt",
        optimizer_config=_optimizer_config(),
        use_deepep=False,
        deterministic=True,
    )
    bundle = protocol.build_model(cfg, impl_cfg=impl_cfg)
    return bundle, cfg, protocol, parallel


def _fingerprint(weights: dict[str, torch.Tensor]) -> tuple[str, int]:
    """Order-independent fingerprint of an exported HF state dict.

    Uses (name, shape, dtype, float64 sum) per tensor — bitwise-exact export on
    both A/B legs produces identical sums, so a fingerprint mismatch flags a
    correctness regression between OLD and NEW.
    """
    h = hashlib.sha256()
    for name in sorted(weights):
        t = weights[name]
        s = t.detach().to("cpu")
        val = s.double().sum().item() if s.is_floating_point() else int(s.long().sum().item())
        h.update(f"{name}|{tuple(s.shape)}|{s.dtype}|{val!r}\n".encode())
    return h.hexdigest(), len(weights)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(_BUILDERS))
    ap.add_argument("--tag", default="new", help="A/B label: new|old")
    ap.add_argument("--out", required=True, help="rank-0 JSON result path")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    result: dict = {
        "model": args.model,
        "tag": args.tag,
        "world_size": world,
        "status": "ok",
    }

    if world != 8:
        result.update(status="skip", reason=f"needs 8 GPUs, got {world}")
        _write(rank, args.out, result)
        return

    try:
        bundle, cfg, protocol, parallel = _build_handle(args.model)
    except Exception as exc:  # env / dep missing → clean skip, fail-loud reason
        result.update(status="skip", reason=f"{type(exc).__name__}: {exc}")
        _write(rank, args.out, result)
        return

    chunks = bundle.chunks
    ps = bundle.parallel_state
    result["topology"] = dict(
        tp=parallel.tp, ep=parallel.ep, etp=parallel.etp, pp=parallel.pp, cp=parallel.cp
    )

    def _one_export() -> tuple[dict, float, float]:
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        w = dict(
            protocol.export_hf_weights(chunks, cfg, ps, rank0_only=True, cpu=True)
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return w, dt, torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Warm up (discard first passes: NCCL/CUDA context + collective warmup would
    # otherwise dominate a tiny-model wall and drown the mechanism's real cost),
    # then take the median of several timed reps so a single cold-jitter sample
    # cannot found a spurious speed verdict.
    n_warmup = int(os.environ.get("MLITE_EXPORT_WARMUP", "2"))
    n_reps = int(os.environ.get("MLITE_EXPORT_REPS", "7"))
    for _ in range(n_warmup):
        _one_export()
    walls: list[float] = []
    peaks: list[float] = []
    weights: dict = {}
    for _ in range(n_reps):
        weights, dt, pk = _one_export()
        walls.append(dt)
        peaks.append(pk)
    walls_sorted = sorted(walls)
    wall = walls_sorted[len(walls_sorted) // 2]  # median
    peak_mib = max(peaks)
    result["wall_min_s"] = round(min(walls), 5)
    result["wall_all_s"] = [round(x, 5) for x in walls]
    result["reps"] = n_reps

    # PP-completeness: rank 0 must carry every decoder layer (all stages gathered).
    layers_ok = True
    if rank == 0:
        num_layers = int(getattr(cfg, "num_hidden_layers"))
        keys = set(weights)

        def _has(i: int) -> bool:
            p = f"layers.{i}."
            return any(k.startswith(p) or f".{p}" in k for k in keys)

        missing = [i for i in range(num_layers) if not _has(i)]
        layers_ok = not missing
        result["missing_layers"] = missing

    fp, n = _fingerprint(weights) if rank == 0 else ("", 0)
    result.update(
        wall_s=round(wall, 5),
        peak_mib=round(peak_mib, 4),
        n_tensors=n,
        fingerprint=fp,
        layers_ok=layers_ok,
    )
    _write(rank, args.out, result)
    dist.barrier()


def _write(rank: int, out: str, result: dict) -> None:
    line = json.dumps(result, sort_keys=True)
    print(f"PP_EXPORT_SENSITIZE rank={rank} {line}", flush=True)
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
