# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Two-rank GLM5 DSA CP correctness, latency, and activation-memory validation."""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.lite.primitive.modules.attention.dsa import (
    DynamicSparseAttention,
    _packed_cp_layout,
    build_rope_cache,
)
from megatron.lite.primitive.parallel.cp import zigzag_position_ids_for_cp, zigzag_slice_for_cp


def _module(*, cp_size: int, cp_rank: int, cp_group, cp_mode: str) -> DynamicSparseAttention:
    torch.manual_seed(20260720)
    return DynamicSparseAttention(
        hidden_size=128,
        num_attention_heads=64,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        rms_norm_eps=1.0e-5,
        indexer_loss_coeff=0.0,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_group=cp_group,
        cp_mode=cp_mode,
    ).cuda().to(torch.bfloat16)


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0))


def _step(module, x, cos, sin, position_ids, packed=None):
    module.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    out = module(
        x,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        packed_seq_params=packed,
    )
    out.float().square().mean().backward()
    return out.detach(), x.grad.detach()


def _measure(module, x, cos, sin, position_ids, *, warmup: int, steps: int):
    for _ in range(warmup):
        _step(module, x, cos, sin, position_ids)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    started = time.perf_counter()
    for _ in range(steps):
        _step(module, x, cos, sin, position_ids)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / steps
    peak_delta_mb = (torch.cuda.max_memory_allocated() - baseline) / (1024**2)
    return {"step_ms": elapsed_ms, "activation_peak_mb": peak_delta_mb}


def _dense_case(seq: int, rank: int, world: int, warmup: int, steps: int):
    device = torch.device("cuda", rank)
    torch.manual_seed(123 + seq)
    full_x = torch.randn(1, seq, 128, device=device, dtype=torch.bfloat16)
    local_x = zigzag_slice_for_cp(full_x, rank, world, seq_dim=1).contiguous()
    cos, sin = build_rope_cache(dim=64, max_position_embeddings=seq, rope_theta=1_000_000.0, device=device)
    full_pos = torch.arange(seq, device=device).unsqueeze(0)
    local_pos = zigzag_position_ids_for_cp(seq, rank, world, device).unsqueeze(0)

    reference = _module(cp_size=1, cp_rank=0, cp_group=None, cp_mode="native")
    ref_out, ref_grad = _step(reference, full_x, cos, sin, full_pos)
    expected_out = zigzag_slice_for_cp(ref_out, rank, world, seq_dim=1)
    expected_grad = zigzag_slice_for_cp(ref_grad, rank, world, seq_dim=1)
    ref_metrics = _measure(reference, full_x, cos, sin, full_pos, warmup=warmup, steps=steps)
    del reference

    metrics = {"cp1": ref_metrics}
    for mode in ("native", "legacy_gather_all"):
        module = _module(cp_size=world, cp_rank=rank, cp_group=dist.group.WORLD, cp_mode=mode)
        out, grad = _step(module, local_x, cos, sin, local_pos)
        mode_metrics = _measure(module, local_x, cos, sin, local_pos, warmup=warmup, steps=steps)
        mode_metrics.update(output_cosine=_cosine(out, expected_out), input_grad_cosine=_cosine(grad, expected_grad))
        metrics[mode] = mode_metrics
        del module
    return metrics


def _thd_case(rank: int, world: int):
    device = torch.device("cuda", rank)
    lengths = [512, 520, 528]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
    total = int(cu[-1])
    torch.manual_seed(818)
    full_x = torch.randn(1, total, 128, device=device, dtype=torch.bfloat16)
    local_rows, _ = _packed_cp_layout(cu, cp_size=world, cp_rank=rank, device=device)
    local_x = full_x.index_select(1, local_rows)
    full_pos = torch.cat([torch.arange(length, device=device) for length in lengths]).unsqueeze(0)
    local_pos = full_pos.index_select(1, local_rows)
    packed = SimpleNamespace(
        cu_seqlens_q=cu,
        cu_seqlens_q_padded=cu,
        max_seqlen_q=max(lengths),
    )
    cos, sin = build_rope_cache(dim=64, max_position_embeddings=max(lengths), rope_theta=1_000_000.0, device=device)

    reference = _module(cp_size=1, cp_rank=0, cp_group=None, cp_mode="native")
    ref_out, ref_grad = _step(reference, full_x, cos, sin, full_pos, packed)
    native = _module(cp_size=world, cp_rank=rank, cp_group=dist.group.WORLD, cp_mode="native")
    out, grad = _step(native, local_x, cos, sin, local_pos, packed)
    result = {
        "output_cosine": _cosine(out, ref_out.index_select(1, local_rows)),
        "input_grad_cosine": _cosine(grad, ref_grad.index_select(1, local_rows)),
        "per_sample_loss": [],
    }
    for sample, (start, end) in enumerate(zip(cu[:-1], cu[1:], strict=True)):
        selected = (local_rows >= start) & (local_rows < end)
        local_sum = out[:, selected].float().square().sum()
        dist.all_reduce(local_sum)
        cp_loss = local_sum / ((int(end - start)) * out.shape[-1])
        ref_loss = ref_out[:, int(start) : int(end)].float().square().mean()
        result["per_sample_loss"].append(
            {"sample": sample, "cp2": float(cp_loss), "cp1": float(ref_loss), "abs_diff": float((cp_loss - ref_loss).abs())}
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="512,1024,2048")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"This validation requires exactly two ranks, got {world}.")

    result = {"world_size": world, "rank": rank, "dense": {}, "thd": _thd_case(rank, world)}
    for seq in (int(value) for value in args.seq_lens.split(",")):
        if seq % (2 * world):
            raise ValueError(f"seq={seq} must be divisible by {2 * world}")
        result["dense"][str(seq)] = _dense_case(seq, rank, world, args.warmup, args.steps)
    gathered = [None] * world
    dist.all_gather_object(gathered, result)
    if rank == 0:
        for rank_result in gathered:
            if rank_result["thd"]["output_cosine"] < 0.9999:
                raise AssertionError(f"THD output cosine below 0.9999: {rank_result['thd']}")
            if rank_result["thd"]["input_grad_cosine"] < 0.9999:
                raise AssertionError(f"THD input-grad cosine below 0.9999: {rank_result['thd']}")
            for seq, dense in rank_result["dense"].items():
                if dense["native"]["output_cosine"] < 0.9999:
                    raise AssertionError(f"dense seq={seq} output cosine below 0.9999: {dense}")
                if dense["native"]["input_grad_cosine"] < 0.9999:
                    raise AssertionError(f"dense seq={seq} input-grad cosine below 0.9999: {dense}")
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump({"ranks": gathered}, stream, indent=2, sort_keys=True)
        print("GLM5_DSA_CP_VALIDATION", json.dumps({"output": args.output, "ranks": gathered}, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
