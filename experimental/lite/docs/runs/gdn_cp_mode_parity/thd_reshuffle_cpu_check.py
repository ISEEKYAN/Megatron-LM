# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU gloo check: GDN chunkwise packing-aware THD reshuffle is bitwise-correct.

The restored ``chunkwise`` GDN CP mode reshuffles the Megatron zigzag CP layout to
contiguous-time chunks before the FLA recurrence and back afterwards. For packed THD
this reshuffle must be *packing-aware*: routing is derived from the **global**
``cu_seqlens`` so that per-sequence zigzag chunk boundaries are honoured even when a
sequence spans the contiguous CP-rank boundary.

The removed ``sharded`` copy instead sliced ``cu_seqlens // cp_size`` and swapped each
sequence independently, which corrupts exactly the boundary-spanning case and was the
root cause of the RL train/inference log-prob mismatch (~220x step-1 ppo_kl).

This check runs the real primitive under gloo and asserts, per CP rank:
  1. ``get_thd_context_parallel_rank_indices`` matches hand-computed indices (anchor).
  2. ``zigzag_to_contiguous_chunks(local_zigzag, cu_seqlens=cu)`` == the ground-truth
     contiguous span ``full[r*T/cp : (r+1)*T/cp]`` (bitwise).
  3. ``contiguous_to_zigzag_chunks`` inverts it back to the local zigzag shard (bitwise).

The per-head conv/recurrence is unchanged FLA GPU code, so a bitwise-correct reshuffle
here => the GPU chunkwise path is fed the same contiguous-time tokens upstream Megatron
feeds its FLA kernel. Numeric CP-off parity of the recurrence itself is the GPU gate.
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.lite.primitive.parallel.cp import (
    contiguous_to_zigzag_chunks,
    get_thd_context_parallel_rank_indices,
    zigzag_to_contiguous_chunks,
)

H = 8  # carried hidden width (arbitrary)

# (cp_size, cu_seqlens_padded, note). Per-seq lengths are divisible by 2*cp (zigzag req);
# the contiguous CP boundary (T/cp) deliberately falls *inside* a sequence in each case.
CASES = [
    (2, [0, 8, 12], "2 seqs len[8,4]; contiguous boundary 6 splits seq0"),
    (4, [0, 16, 24, 32], "3 seqs len[16,8,8]; BATCH>1, multiple boundary spans"),
    (2, [0, 16, 24], "padded seq0=16 + seq1=8; boundary 12 inside seq0"),
]

# Hand-computed indices for the anchor case (cp=2, cu=[0,8,12]); see module docstring
# of the fix for the derivation. Anchors the reference independently of the a2a wiring.
ANCHOR_CU = [0, 8, 12]
ANCHOR = {
    "zigzag": {0: [0, 1, 6, 7, 8, 11], 1: [2, 3, 4, 5, 9, 10]},
    "contiguous": {0: [0, 1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10, 11]},
}


def _check_anchor(cp_rank: int) -> None:
    cu = torch.tensor(ANCHOR_CU, dtype=torch.long)
    for layout in ("zigzag", "contiguous"):
        got = get_thd_context_parallel_rank_indices(cu, 2, cp_rank, layout).tolist()
        exp = ANCHOR[layout][cp_rank]
        assert got == exp, f"anchor {layout} rank{cp_rank}: {got} != {exp}"


def worker(rank, world, cu_list, port, results):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world)
    )
    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)))

    cu = torch.tensor(cu_list, dtype=torch.long)
    total = int(cu[-1])
    part = total // world

    # Anchor the index function (only the cp=2 case matches the hand table).
    if cu_list == ANCHOR_CU and world == 2:
        _check_anchor(rank)

    torch.manual_seed(1234)
    full = torch.randn(total, H)  # identical on every rank (same seed)

    # This rank's zigzag shard (the module's CP-local forward input).
    zz_idx = get_thd_context_parallel_rank_indices(cu, world, rank, "zigzag")
    local_zigzag = full.index_select(0, zz_idx).contiguous()

    # zigzag -> contiguous: expect this rank's contiguous-time span, bitwise.
    got_contig = zigzag_to_contiguous_chunks(local_zigzag, group, seq_dim=0, cu_seqlens=cu)
    expect_contig = full[rank * part : (rank + 1) * part].contiguous()
    fwd = (got_contig - expect_contig).abs().max().item()

    # contiguous -> zigzag: round-trips back to the local zigzag shard, bitwise.
    back = contiguous_to_zigzag_chunks(got_contig, group, seq_dim=0, cu_seqlens=cu)
    rt = (back - local_zigzag).abs().max().item()

    ok = fwd == 0.0 and rt == 0.0
    results.append((rank, ok, fwd, rt))
    dist.barrier()
    dist.destroy_process_group()


def main() -> int:
    import multiprocessing

    all_ok = True
    for i, (cp, cu_list, note) in enumerate(CASES):
        mgr = multiprocessing.Manager()
        results = mgr.list()
        port = 29610 + i
        mp.spawn(worker, args=(cp, cu_list, port, results), nprocs=cp, join=True)
        case_ok = len(results) == cp and all(r[1] for r in results)
        maxfwd = max((r[2] for r in results), default=float("nan"))
        maxrt = max((r[3] for r in results), default=float("nan"))
        all_ok = all_ok and case_ok
        print(
            f"THD_RESHUFFLE cp={cp} cu={cu_list} fwd_max={maxfwd:.3e} rt_max={maxrt:.3e} "
            f"{'PASS' if case_ok else 'FAIL'}  # {note}",
            flush=True,
        )
    print(f"THD_RESHUFFLE_DONE {'ALL_PASS' if all_ok else 'FAIL'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
