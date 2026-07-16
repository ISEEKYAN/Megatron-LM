# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 GatedDeltaNet CP-mode segmented profiler (chunkwise vs headwise).

Companion to ``gdn_cp_mode_parity.py`` (numerical faithfulness) and
``gdn_cp_peakmem_bench.py`` (peak memory / total speed). This harness answers a
different question raised by the DAPO run: **why is chunkwise slower than
replicated/headwise in the real RL regime** (measured ``update_actor`` chunkwise
~159 ms vs replicated ~111 ms) **while the single-long-sequence 128k micro-bench
shows chunkwise as the fastest / lowest-memory mode?** The suspicion is that the
per-layer CP plumbing (all-to-all, packing-aware reshuffle, ring cp_context
build) is a *fixed per-layer overhead* that dominates when the packed batch is
made of many short sequences (the DAPO/THD regime) rather than one long one.

To locate the cost we time the GDN forward **segment by segment** without
touching production code: the relevant ``GatedDeltaNet`` methods are wrapped with
CUDA-event timers (opt-in, harness-only monkeypatch). The segments are exactly
the ones named in the task:

- ``a2a_cp2hp`` / ``a2a_hp2cp`` -- headwise all-to-all in / out.
- ``ctx_build``               -- chunkwise FLA ring ``cp_context`` build (contains
                                 a boundary all-to-all; rebuilt every packed fwd).
- ``reshuffle``               -- chunkwise zigzag<->contiguous chunk reshuffle
                                 (summed over the pre- and post-recurrence calls).
- ``recurrence``              -- ``chunk_gated_delta_rule`` (for chunkwise this
                                 includes the cross-rank ring P2P inside the kernel).
- ``conv``                    -- causal conv1d.
- ``replicate_gather`` / ``replicate_slice`` -- replicated all-gather / slice.
- ``other``                   -- ``fwd_total`` minus the sum of the above
                                 (in_proj / o_proj / split / gated-norm / elementwise).

Two regimes are profiled to expose the contradiction directly:

- ``packed_multi``: a THD packed batch of **many variable-length short sequences**
  (the DAPO packing regime). This is where per-layer plumbing is expected to bite.
- ``single_long`` : a THD packed batch of **one long sequence** (the 128k-style
  regime), where the recurrence should dominate and plumbing amortises away.

For each (mode, regime) we emit, over ``ITERS`` timed iterations after ``WARMUP``:

  GDN_PROFILE mode=X regime=R cp=8 tokens=T nseq=N seg=<name> fwd_ms=<median>
  GDN_PROFILE_TOTAL mode=X regime=R cp=8 tokens=T nseq=N fwd_ms=... bwd_ms=... seg_sum_ms=...
  GDN_PROFILE_DEV mode=X regime=R cp=8 bucket=<comm|recurrence|conv|other> dev_ms=... (fwd+bwd)

The per-segment CUDA-event timing is forward-only (backward does not re-enter the
Python wrappers). The ``GDN_PROFILE_DEV`` lines add a device-time breakdown over a
full **fwd+bwd** pass via ``torch.profiler``, bucketed by op name, so the backward
comm/recurrence split is visible too. Only rank 0's numbers are used for the table;
CP-collective segments block on peers, so rank 0's wall time already includes the
communication wait.

Run under: torchrun --nproc_per_node={2,4,8} gdn_cp_profile.py

Tunable via env:
  PROFILE_MODES     (default chunkwise,headwise,replicated)
  PROFILE_REGIMES   (default packed_multi,single_long)
  PROFILE_MULTI_LENS(default 2048,512,1024,256,1536,384,768,128,896,640,1152,256)
  PROFILE_LONG_LEN  (default 32768)
  PROFILE_WARMUP    (default 3)
  PROFILE_ITERS     (default 8)
  PROFILE_DEV       (default 1 -- also run the torch.profiler device breakdown)
"""
from __future__ import annotations

import os
import statistics
import sys
import traceback

import torch
import torch.distributed as dist

# Reuse the parity harness geometry + module builders verbatim (real Qwen3.5 GDN
# head config, weight randomization) -- same import trick as the peakmem bench so
# this works both in-tree and flat next to the parity file in the Slurm run dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_cp_mode_parity import (  # noqa: E402
    BATCH,
    HIDDEN,
    DTYPE,
    SEED,
    _make_gdn,
    _randomize_and_broadcast,
)

MODES = tuple(m.strip() for m in os.environ.get(
    "PROFILE_MODES", "chunkwise,headwise,replicated").split(",") if m.strip())
REGIMES = tuple(r.strip() for r in os.environ.get(
    "PROFILE_REGIMES", "packed_multi,single_long").split(",") if r.strip())
MULTI_LENS = [int(v) for v in os.environ.get(
    "PROFILE_MULTI_LENS",
    "2048,512,1024,256,1536,384,768,128,896,640,1152,256").split(",") if v.strip()]
LONG_LEN = int(os.environ.get("PROFILE_LONG_LEN", "32768"))
WARMUP = int(os.environ.get("PROFILE_WARMUP", "3"))
ITERS = int(os.environ.get("PROFILE_ITERS", "8"))
DEV_BREAKDOWN = os.environ.get("PROFILE_DEV", "1") == "1"

# ------------------------------------------------------------------ segment timing
# Harness-only monkeypatch: wrap the named GatedDeltaNet methods with CUDA-event
# timers. Production code is not touched; wrappers are transparent unless _PROF["on"].
_PROF = {"on": False, "pending": []}

# method name on GatedDeltaNet -> segment label emitted in the table
_SEG_METHODS = {
    "_headwise_cp2hp": "a2a_cp2hp",
    "_headwise_hp2cp": "a2a_hp2cp",
    "_build_chunkwise_cp_context": "ctx_build",
    "_chunkwise_reshuffle": "reshuffle",
    "_causal_conv1d": "conv",
    "_gated_delta_rule": "recurrence",
    "_replicate_cp_qkvzba": "replicate_gather",
    "_slice_replicated_output": "replicate_slice",
}


def _wrap(seg, fn):
    def wrapper(*args, **kwargs):
        if not _PROF["on"]:
            return fn(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        _PROF["pending"].append((seg, start, end))
        return out

    return wrapper


def _install_segment_timers():
    from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet

    for name, seg in _SEG_METHODS.items():
        fn = getattr(GatedDeltaNet, name)
        setattr(GatedDeltaNet, name, _wrap(seg, fn))


def _drain_segments():
    """After a synced forward, reduce pending (seg,start,end) events to per-seg ms."""
    per_seg = {}
    for seg, start, end in _PROF["pending"]:
        per_seg[seg] = per_seg.get(seg, 0.0) + start.elapsed_time(end)
    _PROF["pending"].clear()
    return per_seg


# ------------------------------------------------------------------ inputs
def _psp(cu, max_seqlen, cp_size, cp_rank, cp_group):
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    extra = {}
    if cp_size > 1:
        extra["local_cp_size"] = cp_size
        extra["cp_group"] = cp_group
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        cp_rank=cp_rank,
        **extra,
    )


def _round_up(n, m):
    return ((n + m - 1) // m) * m


def _regime_lens(regime, cp_size):
    """Per-sequence lengths for a regime, each divisible by 2*cp and by the FLA chunk 64.

    Chunkwise requires every per-seq length % cp == 0 (chunk boundaries land on CP
    shards); the reshuffle needs % (2*cp); FLA needs % 64. Rounding each length up to
    a multiple of lcm(2*cp, 64) satisfies all three for cp in {2,4,8}.
    """
    align = _round_up(2 * cp_size, 1)
    unit = align
    while unit % 64 != 0:
        unit += align
    if regime == "packed_multi":
        raw = MULTI_LENS
    elif regime == "single_long":
        raw = [LONG_LEN]
    else:
        raise ValueError(f"unknown regime {regime!r}")
    return [max(unit, _round_up(v, unit)) for v in raw]


def _build_input(regime, cp_size, cp_rank, cp_group, device):
    from megatron.lite.primitive.parallel.thd import split_packed_to_cp_local

    lens = _regime_lens(regime, cp_size)
    lens_t = torch.tensor(lens, dtype=torch.int32, device=device)
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
    torch.cumsum(lens_t, dim=0, out=cu[1:])
    total = int(cu[-1].item())
    max_seqlen = int(lens_t.max().item())

    torch.manual_seed(SEED + 7)
    full_x = torch.randn(total, BATCH, HIDDEN, device=device, dtype=DTYPE)
    dist.broadcast(full_x, src=0)
    local_x = split_packed_to_cp_local(
        full_x, cu_seqlens_padded=cu, cp_size=cp_size, cp_rank=cp_rank, dim=0
    ).contiguous()
    del full_x
    psp = _psp(cu, max_seqlen, cp_size, cp_rank, cp_group)
    cot = torch.randn_like(local_x)
    return local_x, psp, total, len(lens), cot


def _run_mode(module, x_base, psp, cot, do_backward):
    module.zero_grad(set_to_none=True)
    x = x_base.clone().requires_grad_(do_backward)
    out = module(x, packed_seq_params=psp)
    if do_backward:
        out.backward(cot)
    return out, x


# ------------------------------------------------------------------ device breakdown
_DEV_BUCKETS = (
    ("comm", ("nccl", "alltoall", "all_to_all", "allgather", "all_gather",
              "broadcast", "c10d", "reducescatter", "reduce_scatter")),
    ("recurrence", ("chunk_gated_delta", "chunk_delta", "chunk_fwd", "chunk_bwd",
                    "chunk_scan", "fused_recurrent", "delta_rule", "solve_tril",
                    "wy_fast", "cumsum", "fla")),
    ("conv", ("conv",)),
)


def _bucket_of(name):
    low = name.lower()
    for bucket, keys in _DEV_BUCKETS:
        if any(k in low for k in keys):
            return bucket
    return "other"


def _device_breakdown(module, x_base, psp, cot, device):
    """Full fwd+bwd device-time breakdown by op-name bucket (backward included)."""
    from torch.profiler import ProfilerActivity, profile

    # warm once so kernels are compiled/cached before profiling
    out, x = _run_mode(module, x_base, psp, cot, True)
    del out, x
    torch.cuda.synchronize()
    dist.barrier()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            out, x = _run_mode(module, x_base, psp, cot, True)
            del out, x
        torch.cuda.synchronize()

    buckets = {"comm": 0.0, "recurrence": 0.0, "conv": 0.0, "other": 0.0}
    for evt in prof.key_averages():
        dev_us = getattr(evt, "self_device_time_total", None)
        if dev_us is None:
            dev_us = getattr(evt, "self_cuda_time_total", 0.0)
        if dev_us:
            buckets[_bucket_of(evt.key)] += dev_us / 1e3 / 3.0  # us->ms, per-iter
    return buckets


# ------------------------------------------------------------------ per (mode,regime)
def profile_case(mode, regime, cp_size, cp_rank, cp_group, device, rank):
    x_base, psp, total, nseq, cot = _build_input(
        regime, cp_size, cp_rank, cp_group, device
    )
    module = _make_gdn(cp_size, cp_rank, cp_group, mode).to(device=device, dtype=DTYPE)
    _randomize_and_broadcast(module)

    # --- warmup (compiles FLA/tilelang kernels, stabilises allocator) ---
    for _ in range(WARMUP):
        out, x = _run_mode(module, x_base, psp, cot, True)
        del out, x
    torch.cuda.synchronize()
    dist.barrier()

    # --- per-segment forward timing (CUDA events, forward-only) ---
    seg_samples = {}
    fwd_totals = []
    _PROF["on"] = True
    for _ in range(ITERS):
        _PROF["pending"].clear()
        torch.cuda.synchronize()
        dist.barrier()
        whole_s = torch.cuda.Event(enable_timing=True)
        whole_e = torch.cuda.Event(enable_timing=True)
        whole_s.record()
        out, x = _run_mode(module, x_base, psp, cot, False)
        whole_e.record()
        torch.cuda.synchronize()
        fwd_totals.append(whole_s.elapsed_time(whole_e))
        for seg, ms in _drain_segments().items():
            seg_samples.setdefault(seg, []).append(ms)
        del out, x
    _PROF["on"] = False

    # --- full fwd+bwd wall totals (CUDA events) ---
    fb_fwd, fb_bwd = [], []
    for _ in range(ITERS):
        module.zero_grad(set_to_none=True)
        x = x_base.clone().requires_grad_(True)
        torch.cuda.synchronize()
        dist.barrier()
        s = torch.cuda.Event(enable_timing=True)
        m = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = module(x, packed_seq_params=psp)
        m.record()
        out.backward(cot)
        e.record()
        torch.cuda.synchronize()
        dist.barrier()
        fb_fwd.append(s.elapsed_time(m))
        fb_bwd.append(m.elapsed_time(e))
        del out, x

    seg_med = {seg: statistics.median(v) for seg, v in seg_samples.items()}
    res = dict(
        total=total,
        nseq=nseq,
        seg=seg_med,
        fwd_total=statistics.median(fwd_totals),
        fb_fwd=statistics.median(fb_fwd),
        fb_bwd=statistics.median(fb_bwd),
    )

    dev = None
    if DEV_BREAKDOWN:
        try:
            dev = _device_breakdown(module, x_base, psp, cot, device)
        except Exception as exc:  # noqa: BLE001 - profiler is best-effort
            if rank == 0:
                print(
                    f"GDN_PROFILE_DEV_SKIP mode={mode} regime={regime} :: "
                    f"{type(exc).__name__}: {str(exc)[:120]}",
                    flush=True,
                )
    res["dev"] = dev

    del module, x_base, cot
    torch.cuda.empty_cache()
    dist.barrier()
    return res


def _emit(rank, mode, regime, cp_size, res):
    if rank != 0:
        return
    total, nseq = res["total"], res["nseq"]
    seg = res["seg"]
    seg_sum = sum(seg.values())
    other = max(0.0, res["fwd_total"] - seg_sum)
    ordered = ["a2a_cp2hp", "a2a_hp2cp", "ctx_build", "reshuffle", "conv",
               "recurrence", "replicate_gather", "replicate_slice"]
    for name in ordered:
        if name in seg:
            print(
                f"GDN_PROFILE mode={mode} regime={regime} cp={cp_size} "
                f"tokens={total} nseq={nseq} seg={name} fwd_ms={seg[name]:.3f}",
                flush=True,
            )
    print(
        f"GDN_PROFILE mode={mode} regime={regime} cp={cp_size} "
        f"tokens={total} nseq={nseq} seg=other fwd_ms={other:.3f}",
        flush=True,
    )
    print(
        f"GDN_PROFILE_TOTAL mode={mode} regime={regime} cp={cp_size} "
        f"tokens={total} nseq={nseq} fwd_ms={res['fb_fwd']:.3f} "
        f"bwd_ms={res['fb_bwd']:.3f} seg_sum_ms={seg_sum:.3f} "
        f"fwd_probe_ms={res['fwd_total']:.3f}",
        flush=True,
    )
    if res.get("dev"):
        d = res["dev"]
        print(
            f"GDN_PROFILE_DEV mode={mode} regime={regime} cp={cp_size} "
            f"comm_ms={d['comm']:.3f} recurrence_ms={d['recurrence']:.3f} "
            f"conv_ms={d['conv']:.3f} other_ms={d['other']:.3f} (fwd+bwd)",
            flush=True,
        )


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    cp_size = world
    cp_rank = rank
    cp_group = dist.new_group(list(range(world)))

    _install_segment_timers()

    if rank == 0:
        import megatron.lite.primitive.modules.gated_delta_net as g

        print(
            f"GDN_PROFILE_ENV world={world} cp={cp_size} modes={','.join(MODES)} "
            f"regimes={','.join(REGIMES)} warmup={WARMUP} iters={ITERS} "
            f"dev={int(DEV_BREAKDOWN)} HAS_FLA={g._HAS_FLA}",
            flush=True,
        )

    for regime in REGIMES:
        for mode in MODES:
            torch.cuda.empty_cache()
            try:
                res = profile_case(
                    mode, regime, cp_size, cp_rank, cp_group, device, rank
                )
            except Exception as exc:  # noqa: BLE001
                if rank == 0:
                    print(
                        f"GDN_PROFILE_ERR mode={mode} regime={regime} :: "
                        f"{type(exc).__name__}: {str(exc)[:200]}",
                        flush=True,
                    )
                dist.barrier()
                continue
            _emit(rank, mode, regime, cp_size, res)
            dist.barrier()

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("GDN_PROFILE_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("GDN_PROFILE_ERROR", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
