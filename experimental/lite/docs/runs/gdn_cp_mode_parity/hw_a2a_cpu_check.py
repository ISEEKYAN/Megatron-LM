"""CPU gloo check: GDN headwise cp2hp/hp2cp redistribution is a correct, lossless
round-trip against the ground-truth full-sequence layout. Validates the only new
distributed plumbing without TE/GPU. Heads are independent, so a correct
redistribution => headwise output == full-seq (replicated/CP-off) output bitwise.
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.lite.primitive.parallel.cp import (
    all_to_all_hidden_shards,
    build_headwise_section_perm,
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
    zigzag_split_for_cp,
)

SECTIONS = [256, 256, 512, 512, 4, 4]   # q,k,v,z,beta,alpha  (H=1544)
H = sum(SECTIONS)
VPC_SECTION = 512                        # value output width
SEQ = 64
CP = 4


def _cp2hp(qkvzba, cp_group, cp_size, cp_rank):
    perm = build_headwise_section_perm(SECTIONS, cp_size, qkvzba.device)
    q = qkvzba.index_select(-1, perm)
    hpc = H // cp_size
    send = [q[..., k * hpc:(k + 1) * hpc].contiguous() for k in range(cp_size)]
    recv = all_to_all_hidden_shards(send, cp_group)
    return zigzag_reconstruct_from_cp_parts(recv, seq_dim=1).contiguous()


def _hp2cp(out, cp_group, cp_size):
    send = [zigzag_slice_for_cp(out, j, cp_size, seq_dim=1).contiguous() for j in range(cp_size)]
    recv = all_to_all_hidden_shards(send, cp_group)
    return torch.cat(recv, dim=-1).contiguous()


def worker(rank, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29591",
                      RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    g = dist.new_group(list(range(world)))
    torch.manual_seed(1234)
    full = torch.randn(1, SEQ, H)              # identical on every rank (same seed)

    # --- cp2hp: feed zigzag shard, expect full contiguous seq for this rank's head shard.
    local = zigzag_split_for_cp(full, rank, world, seq_dim=1)
    hp = _cp2hp(local, g, world, rank)          # [1, SEQ, H/cp]

    # ground truth: permute full, take contiguous hidden shard `rank`, in contiguous time.
    perm = build_headwise_section_perm(SECTIONS, world, full.device)
    full_perm = full.index_select(-1, perm)
    hpc = H // world
    expect = full_perm[..., rank * hpc:(rank + 1) * hpc].contiguous()
    a = (hp - expect).abs().max().item()

    # --- hp2cp: round-trip a per-rank value block back to zigzag shards + gathered heads.
    torch.manual_seed(99)
    vfull = torch.randn(1, SEQ, VPC_SECTION)    # full-seq value output, identical across ranks
    vpc = VPC_SECTION // world
    my_block = vfull[..., rank * vpc:(rank + 1) * vpc].contiguous()  # this rank owns block `rank`
    back = _hp2cp(my_block, g, world)           # [1, SEQ/cp (zigzag), VPC]
    expect_back = zigzag_slice_for_cp(vfull, rank, world, seq_dim=1).contiguous()
    b = (back - expect_back).abs().max().item()

    if rank == 0:
        print(f"HW_A2A cp2hp_max_abs={a:.3e} hp2cp_max_abs={b:.3e} "
              f"{'PASS' if a == 0.0 and b == 0.0 else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(CP,), nprocs=CP, join=True)
