# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 GatedDeltaNet gdn_cp_mode parity proxy harness.

Compares the two CP execution strategies of the native Qwen3.5 GatedDeltaNet
primitive against a CP-off full-sequence reference and against each other:

    gdn_cp_mode = "replicated"  (default; all-gather full seq, compute, slice)
    gdn_cp_mode = "sharded"     (FLA cp_context ring; zigzag<->contiguous swap)

Matrix (per AC #1):
    * baseline = CP off (cp_size == 1), full sequence, single rank compute.
    * {replicated, sharded} x {CP2, CP4 (world dependent)}.
    * per-tensor diff report of the two apples-to-apples signals -- the forward
      OUTPUT and the INPUT-activation grad -- both sliced to this rank's zigzag
      shard of the CP-off reference (max_abs / max_rel), plus a direct
      replicated-vs-sharded cross diff.

    NOTE on parameter grads: we deliberately do NOT diff parameter grads here.
    This harness runs a bare GatedDeltaNet without a CP grad all-reduce, so each
    CP rank only accumulates the parameter grad for its own sequence shard;
    that is not directly comparable to the CP-off full-sequence parameter grad.
    The forward output and input-activation grad ARE per-token quantities and
    slice cleanly, so they are the correct parity signals for locating a CP
    gather/split precision defect.

The proxy uses REAL GDN head dims (dk=dv=128, conv_kernel=4) because tiny
dim=4 probes are non-representative (see K-0125 / dead_ends/cp.md); only the
head COUNT and sequence length are truncated to keep it an 8-GPU proxy.

Run under torchrun, e.g.:
    torchrun --standalone --nproc_per_node=4 \
        experimental/lite/tests/smoke/primitive/gdn_cp_mode_parity_report.py
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist


# Real Qwen3.5 GDN per-head geometry (proxy truncates head COUNT + seq only).
PROXY_GDN_KWARGS = dict(
    hidden_size=512,
    linear_num_key_heads=2,
    linear_key_head_dim=128,
    linear_num_value_heads=4,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    rms_norm_eps=1e-6,
)
PROXY_SEED = 20260714
INPUT_SEED = 4242


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _build_gdn(ps, cp_mode: str, device, dtype):
    from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet

    torch.manual_seed(PROXY_SEED)
    mod = GatedDeltaNet(ps=ps, cp_mode=cp_mode, deterministic=False, **PROXY_GDN_KWARGS)
    return mod.to(device=device, dtype=dtype)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    a = actual.detach().float()
    e = expected.detach().float()
    diff = (a - e).abs()
    max_abs = diff.max()
    scale = torch.maximum(a.abs().max(), e.abs().max()).clamp_min(1e-6)
    return float(max_abs.item()), float((max_abs / scale).item())


def _run_mode(ps, cp_mode: str, full_x: torch.Tensor, device, dtype, rank, world):
    """Run one CP mode; return (local_out, local_input_grad, module)."""
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp

    mod = _build_gdn(ps, cp_mode, device, dtype)
    local_x = (
        zigzag_slice_for_cp(full_x, rank, world, seq_dim=1)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    out = mod(local_x)
    out.float().sum().backward()
    return out, local_x.grad, mod


def _run_reference(full_x: torch.Tensor, device, dtype):
    """CP-off baseline: single-rank full sequence, cp_size == 1."""
    from megatron.lite.primitive.parallel.state import ParallelState

    ps = ParallelState()  # cp_size defaults to 1
    mod = _build_gdn(ps, "replicated", device, dtype)  # mode irrelevant at cp=1
    ref_x = full_x.detach().clone().requires_grad_(True)
    out = mod(ref_x)
    out.float().sum().backward()
    return out, ref_x.grad


def main() -> int:
    if not torch.cuda.is_available():
        _log(0, json.dumps({"status": "skip", "reason": "CUDA required"}))
        return 0
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        _log(0, json.dumps({"status": "skip", "reason": "requires torchrun"}))
        return 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import ParallelState

    # CP sizes to sweep: 2 and (if world allows) 4, always <= world.
    cp_sizes = [c for c in (2, 4) if c <= world]
    if not cp_sizes:
        _log(0, json.dumps({"status": "skip", "reason": f"world={world} < 2"}))
        return 0

    # Fixed global sequence divisible by 2*max(cp) for zigzag.
    seq = 256
    max_cp = max(cp_sizes)
    if seq % (2 * max_cp) != 0:
        _log(0, json.dumps({"status": "skip", "reason": f"seq {seq} not div 2*{max_cp}"}))
        return 0

    hidden = PROXY_GDN_KWARGS["hidden_size"]
    torch.manual_seed(INPUT_SEED)
    full_x = torch.randn(1, seq, hidden, device=device, dtype=dtype)

    # CP-off reference (full sequence, single rank). Every rank computes it so
    # each can slice the reference to its own zigzag shard for local diffs.
    ref_out, ref_in_grad = _run_reference(full_x, device, dtype)

    report: dict[str, object] = {
        "status": "ok",
        "world_size": world,
        "seq_len": seq,
        "proxy_gdn": PROXY_GDN_KWARGS,
        "results": [],
    }

    for cp in cp_sizes:
        # Build a CP subgroup over the FIRST `cp` ranks so every configuration
        # runs on a well-defined group. Ranks outside the group skip this cp.
        group_ranks = list(range(cp))
        cp_group = dist.new_group(ranks=group_ranks)
        in_group = rank in group_ranks
        for cp_mode in ("replicated", "sharded"):
            if not in_group:
                dist.barrier()
                continue
            cp_rank = rank  # ranks 0..cp-1 map directly
            ps = ParallelState(cp_group=cp_group, cp_size=cp, cp_rank=cp_rank)
            entry: dict[str, object] = {"cp_mode": cp_mode, "cp_size": cp, "cp_rank": cp_rank}
            try:
                out, in_grad, _mod = _run_mode(ps, cp_mode, full_x, device, dtype, cp_rank, cp)
            except NotImplementedError as exc:
                entry["error"] = f"NotImplementedError: {exc}"
                report["results"].append(entry)  # type: ignore[attr-defined]
                dist.barrier()
                continue
            except Exception as exc:  # noqa: BLE001 - surfaces backend gaps in report
                entry["error"] = f"{type(exc).__name__}: {exc}"
                report["results"].append(entry)  # type: ignore[attr-defined]
                dist.barrier()
                continue

            exp_out = zigzag_slice_for_cp(ref_out, cp_rank, cp, seq_dim=1)
            exp_in_grad = zigzag_slice_for_cp(ref_in_grad, cp_rank, cp, seq_dim=1)
            o_abs, o_rel = _diff_stats(out, exp_out)
            g_abs, g_rel = _diff_stats(in_grad, exp_in_grad)
            entry["out_max_abs"] = o_abs
            entry["out_max_rel"] = o_rel
            entry["in_grad_max_abs"] = g_abs
            entry["in_grad_max_rel"] = g_rel
            # cache local outputs for cross-mode diff
            entry["_out"] = out.detach()
            entry["_in_grad"] = in_grad.detach()
            report["results"].append(entry)  # type: ignore[attr-defined]
            dist.barrier()

    # Cross diff replicated vs sharded (per cp, on group ranks only).
    cross: list[dict[str, object]] = []
    results = report["results"]  # type: ignore[assignment]
    for cp in cp_sizes:
        rep = next((r for r in results if r.get("cp_size") == cp and r["cp_mode"] == "replicated" and "_out" in r), None)  # type: ignore[union-attr,index]
        sha = next((r for r in results if r.get("cp_size") == cp and r["cp_mode"] == "sharded" and "_out" in r), None)  # type: ignore[union-attr,index]
        if rep is not None and sha is not None:
            c_abs, c_rel = _diff_stats(rep["_out"], sha["_out"])  # type: ignore[index]
            cg_abs, cg_rel = _diff_stats(rep["_in_grad"], sha["_in_grad"])  # type: ignore[index]
            cross.append({
                "cp_size": cp,
                "replicated_vs_sharded_out_max_abs": c_abs,
                "replicated_vs_sharded_out_max_rel": c_rel,
                "replicated_vs_sharded_in_grad_max_abs": cg_abs,
                "replicated_vs_sharded_in_grad_max_rel": cg_rel,
            })
    report["cross_mode"] = cross

    # strip cached tensors before serialization
    for r in results:  # type: ignore[assignment]
        r.pop("_out", None)  # type: ignore[union-attr]
        r.pop("_in_grad", None)  # type: ignore[union-attr]

    if rank == 0:
        print("GDN_CP_MODE_PARITY_REPORT_BEGIN", flush=True)
        print(json.dumps(report, indent=2), flush=True)
        print("GDN_CP_MODE_PARITY_REPORT_END", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
