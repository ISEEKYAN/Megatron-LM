# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Matched single-layer performance gate for DeepEP ChunkedEP.

Run with ``torchrun`` on one EP group. The baseline and candidate share exact
weights and inputs. Both use DeepEP; the only behavioral difference is token
chunking. The backward measurements use full recomputation in both arms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.qwen3_moe.lite.model import MoELayer
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.runtime.contracts.config import ParallelConfig
from torch.utils.checkpoint import checkpoint

MODES = ("forward", "backward", "fused_forward_backward")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--tokens-per-gpu", type=int, default=16384)
    parser.add_argument("--chunks", type=int, choices=(1, 2), default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _all_reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _global_grad_norm(module: torch.nn.Module, device: torch.device) -> float:
    squared = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().double().square().sum()
    dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    return math.sqrt(float(squared.item()))


def _full_recompute_output(
    module: torch.nn.Module, hidden: torch.Tensor, *, native_chunked_recompute: bool
) -> torch.Tensor:
    if native_chunked_recompute:
        return module(hidden)
    return checkpoint(module, hidden, use_reentrant=True)


def _time_call(call: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _run_once(
    module: torch.nn.Module,
    base_hidden: torch.Tensor,
    *,
    mode: str,
    native_chunked_recompute: bool,
) -> tuple[float, float]:
    module.zero_grad(set_to_none=True)
    hidden = base_hidden.detach().clone().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats(hidden.device)

    if mode == "forward":

        def call() -> None:
            output = _full_recompute_output(
                module, hidden, native_chunked_recompute=native_chunked_recompute
            )
            output.detach()

    elif mode == "backward":
        output = _full_recompute_output(
            module, hidden, native_chunked_recompute=native_chunked_recompute
        )
        loss = output.float().square().mean()
        torch.cuda.synchronize(hidden.device)

        def call() -> None:
            loss.backward()

    elif mode == "fused_forward_backward":

        def call() -> None:
            output = _full_recompute_output(
                module, hidden, native_chunked_recompute=native_chunked_recompute
            )
            output.float().square().mean().backward()

    else:
        raise ValueError(f"unknown mode: {mode}")

    elapsed_ms = _time_call(call)
    peak_gb = torch.cuda.max_memory_allocated(hidden.device) / 1e9
    module.zero_grad(set_to_none=True)
    return elapsed_ms, peak_gb


def _measure_pair(
    baseline: torch.nn.Module,
    candidate: torch.nn.Module,
    hidden: torch.Tensor,
    *,
    mode: str,
    warmup: int,
    repeats: int,
) -> dict:
    samples: dict[str, list[float]] = {"baseline": [], "chunked": []}
    peak_gb: dict[str, float] = {"baseline": 0.0, "chunked": 0.0}
    arms = {"baseline": (baseline, False), "chunked": (candidate, True)}
    for iteration in range(warmup + repeats):
        order = (
            ("baseline", "chunked") if iteration % 2 == 0 else ("chunked", "baseline")
        )
        for name in order:
            module, native_chunked_recompute = arms[name]
            elapsed_ms, arm_peak_gb = _run_once(
                module,
                hidden,
                mode=mode,
                native_chunked_recompute=native_chunked_recompute,
            )
            if iteration >= warmup:
                samples[name].append(elapsed_ms)
                peak_gb[name] = max(peak_gb[name], arm_peak_gb)

    baseline_median = statistics.median(samples["baseline"])
    chunked_median = statistics.median(samples["chunked"])
    paired_speedups = [
        base / chunked
        for base, chunked in zip(samples["baseline"], samples["chunked"], strict=True)
    ]
    speedup = baseline_median / chunked_median
    wins = sum(
        chunked < base for base, chunked in zip(samples["baseline"], samples["chunked"])
    )
    stable_wins_required = math.ceil(repeats * 0.8)
    passed = speedup > 1.0 and wins >= stable_wins_required
    return {
        "baseline_ms": samples["baseline"],
        "chunked_ms": samples["chunked"],
        "baseline_median_ms": baseline_median,
        "chunked_median_ms": chunked_median,
        "speedup": speedup,
        "paired_speedup_median": statistics.median(paired_speedups),
        "chunked_wins": wins,
        "stable_wins_required": stable_wins_required,
        "baseline_peak_gb": peak_gb["baseline"],
        "chunked_peak_gb": peak_gb["chunked"],
        "passed": passed,
    }


def _parity(
    baseline: torch.nn.Module, candidate: torch.nn.Module, hidden: torch.Tensor
) -> dict:
    outputs = []
    input_grads = []
    param_grads = []
    losses = []
    grad_norms = []
    for module, native_chunked_recompute in ((baseline, False), (candidate, True)):
        module.zero_grad(set_to_none=True)
        x = hidden.detach().clone().requires_grad_(True)
        output = _full_recompute_output(
            module, x, native_chunked_recompute=native_chunked_recompute
        )
        loss = output.float().square().mean()
        loss.backward()
        outputs.append(output.detach())
        input_grads.append(x.grad.detach())
        param_grads.append(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in module.named_parameters()
                if parameter.grad is not None
            }
        )
        losses.append(float(loss.detach()))
        grad_norms.append(_global_grad_norm(module, hidden.device))
        module.zero_grad(set_to_none=True)

    output_max_abs = float((outputs[1] - outputs[0]).abs().max())
    input_grad_max_abs = float((input_grads[1] - input_grads[0]).abs().max())
    param_grad_max_abs = 0.0
    for name, expected in param_grads[0].items():
        actual = param_grads[1][name]
        param_grad_max_abs = max(
            param_grad_max_abs, float((actual - expected).abs().max())
        )
    loss_abs = abs(losses[1] - losses[0])
    grad_norm_abs = abs(grad_norms[1] - grad_norms[0])
    parity = {
        "loss_abs": _all_reduce_max(loss_abs, hidden.device),
        "grad_norm_abs": _all_reduce_max(grad_norm_abs, hidden.device),
        "output_max_abs": _all_reduce_max(output_max_abs, hidden.device),
        "input_grad_max_abs": _all_reduce_max(input_grad_max_abs, hidden.device),
        "param_grad_max_abs": _all_reduce_max(param_grad_max_abs, hidden.device),
    }
    parity["passed"] = parity["loss_abs"] <= 1e-4 and parity["grad_norm_abs"] <= 1e-4
    return parity


def main() -> int:
    args = _parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = Qwen3MoEConfig.from_hf(
        args.hf_path,
        num_hidden_layers=1,
        num_experts=world_size,
        num_experts_per_tok=world_size,
        layer_types=["full_attention"],
    )
    parallel = init_parallel(
        ParallelConfig(tp=1, etp=1, ep=world_size, pp=1, vpp=1, cp=1)
    )
    baseline = MoELayer(config, parallel, use_deepep=True).to(torch.bfloat16).cuda()
    candidate = MoELayer(
        config,
        parallel,
        num_chunks_ep_a2a_overlap=args.chunks,
        use_deepep=True,
        layer_idx=0,
    ).to(torch.bfloat16).cuda()
    candidate.load_state_dict(baseline.state_dict())
    assert baseline.dispatcher.use_deepep
    assert candidate.ep_chunk_overlap.dispatcher.use_deepep

    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    hidden = torch.randn(
        args.tokens_per_gpu,
        config.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    parity = _parity(baseline, candidate, hidden)
    metrics = {
        mode: _measure_pair(
            baseline,
            candidate,
            hidden,
            mode=mode,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for mode in MODES
    }
    result = {
        "world_size": world_size,
        "tokens_per_gpu": args.tokens_per_gpu,
        "hidden_size": config.hidden_size,
        "num_experts": config.num_experts,
        "topk": config.num_experts_per_tok,
        "chunks": args.chunks,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "use_deepep": True,
        "full_recompute": True,
        "parity": parity,
        "metrics": metrics,
        "passed": parity["passed"] and all(item["passed"] for item in metrics.values()),
    }
    if rank == 0:
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print("CHUNKED_EP_LAYER_PERF " + rendered, flush=True)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
