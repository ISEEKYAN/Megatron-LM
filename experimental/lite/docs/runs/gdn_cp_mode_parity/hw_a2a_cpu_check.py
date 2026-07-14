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
from megatron.lite.primitive.parallel.thd import (
    reconstruct_packed_from_cp_parts,
    split_packed_to_cp_local,
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

    # --- packed THD round-trip (RL layout): per-sample zigzag over a packed batch.
    cu = torch.tensor([0, 32, 64], dtype=torch.long)  # two samples, len 32 (=%8) each
    T = int(cu[-1])
    torch.manual_seed(7)
    packed_full = torch.randn(1, T, H)                 # full packed qkvzba, identical across ranks
    packed_perm = packed_full.index_select(-1, perm)
    # this rank's zigzag shard of the packed sequence (module's forward input under CP)
    local_p = split_packed_to_cp_local(packed_full[0], cu_seqlens_padded=cu, cp_size=world,
                                        cp_rank=rank, dim=0).unsqueeze(0)
    lp = local_p.index_select(-1, perm)
    send_p = [lp[..., k * hpc:(k + 1) * hpc].contiguous() for k in range(world)]
    recv_p = all_to_all_hidden_shards(send_p, g)
    parts = [p[0].contiguous() for p in recv_p]
    hp_p = reconstruct_packed_from_cp_parts(parts, cu_seqlens_padded=cu, cp_size=world,
                                            dim=0).unsqueeze(0)
    expect_p = packed_perm[..., rank * hpc:(rank + 1) * hpc].contiguous()
    c = (hp_p - expect_p).abs().max().item()

    # packed hp2cp inverse on a full-seq per-rank value block
    torch.manual_seed(55)
    vfull_p = torch.randn(1, T, VPC_SECTION)
    my_block_p = vfull_p[..., rank * vpc:(rank + 1) * vpc].contiguous()
    send_pb = [split_packed_to_cp_local(my_block_p[0], cu_seqlens_padded=cu, cp_size=world,
                                        cp_rank=j, dim=0).unsqueeze(0).contiguous()
               for j in range(world)]
    recv_pb = all_to_all_hidden_shards(send_pb, g)
    back_p = torch.cat(recv_pb, dim=-1).contiguous()
    expect_back_p = split_packed_to_cp_local(vfull_p[0], cu_seqlens_padded=cu, cp_size=world,
                                             cp_rank=rank, dim=0).unsqueeze(0).contiguous()
    d = (back_p - expect_back_p).abs().max().item()

    if rank == 0:
        ok = a == 0.0 and b == 0.0 and c == 0.0 and d == 0.0
        print(f"HW_A2A dense[cp2hp={a:.3e} hp2cp={b:.3e}] "
              f"packed[cp2hp={c:.3e} hp2cp={d:.3e}] "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(CP,), nprocs=CP, join=True)
