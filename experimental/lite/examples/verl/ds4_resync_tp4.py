# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Full-checkpoint DeepSeek-V4 TP4 resync parity validation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def math_prompts() -> list[str]:
    cases = (
        "17 * 19",
        "144 / 12",
        "2^10",
        "gcd(84, 126)",
        "lcm(12, 18)",
        "the next prime after 97",
        "the sum of the first 20 integers",
        "15% of 240",
        "3/4 + 5/8",
        "7/9 - 2/3",
        "sqrt(2025)",
        "the area of a circle of radius 3 in terms of pi",
        "the perimeter of a 5 by 8 rectangle",
        "x if 3x + 7 = 31",
        "x if x^2 = 169 and x is positive",
        "the mean of 4, 7, 9, 10",
        "the median of 2, 3, 8, 11, 15",
        "the remainder of 1234 divided by 9",
        "the number of diagonals in a hexagon",
        "6 choose 2",
        "the derivative of x^3 at x=2",
        "the integral of 2x from 0 to 3",
        "log base 2 of 128",
        "the slope through (1,2) and (5,10)",
        "the hypotenuse of a right triangle with legs 5 and 12",
        "the missing angle of a triangle with angles 35 and 65 degrees",
        "the sum of interior angles of an octagon",
        "0.125 as a fraction",
        "the reciprocal of 7/11",
        "the solution count of x^2+1=0 over reals",
        "the smallest positive multiple of both 8 and 14",
        "the value of 5!",
        "the sum 1+3+5+7+9",
        "the cube root of 1728",
        "the probability of heads on one fair coin toss",
        "the distance between -7 and 5 on the number line",
    )
    return [f"Solve briefly: what is {case}?" for case in cases]


def percentile(values: torch.Tensor, quantile: float) -> float:
    if values.ndim != 1 or not values.numel():
        raise ValueError("percentile requires a non-empty flat tensor")
    rank = max(1, min(values.numel(), int(quantile * values.numel() + 0.999999)))
    return float(torch.kthvalue(values, rank).values)


def compare_distributions(
    reference: list[dict[str, torch.Tensor]],
    candidate: list[dict[str, torch.Tensor]],
) -> dict[str, float | int]:
    if len(reference) != len(candidate):
        raise ValueError("prompt result counts differ")
    deltas, kls, selected = [], [], []
    token_count = 0
    for left, right in zip(reference, candidate, strict=True):
        ref = left["logprobs"].float()
        cand = right["logprobs"].float()
        ids = left["token_ids"].long()
        if ref.shape != cand.shape or ids.shape != ref.shape[:-1]:
            raise ValueError("token-aligned distribution shapes differ")
        if not torch.equal(ids, right["token_ids"].long()):
            raise ValueError("tokenized prompts differ between arms")
        delta = cand - ref
        deltas.append(delta.abs().flatten())
        kls.append((ref.exp() * (ref - cand)).sum(dim=-1).flatten())
        selected.append(delta.gather(-1, ids.unsqueeze(-1)).abs().flatten())
        token_count += ids.numel()
    all_delta = torch.cat(deltas)
    all_kl = torch.cat(kls)
    all_selected = torch.cat(selected)
    return {
        "prompt_count": len(reference),
        "token_count": token_count,
        "max_abs": float(all_delta.max()),
        "p99_abs": percentile(all_delta, 0.99),
        "max_kl": float(all_kl.max()),
        "p99_kl": percentile(all_kl, 0.99),
        "max_selected_token_logprob_delta": float(all_selected.max()),
        "p99_selected_token_logprob_delta": percentile(all_selected, 0.99),
    }


def copy_checkpoint_metadata(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if path.is_dir():
            shutil.copytree(path, output / path.name, dirs_exist_ok=True)
        elif path.suffix != ".safetensors":
            shutil.copy2(path, output / path.name)


def model_vocab_size(config: Any, tokenizer: Any) -> int:
    size = int(config.vocab_size)
    if size < int(tokenizer.vocab_size):
        raise ValueError("model vocabulary is smaller than tokenizer vocabulary")
    return size


def _dense_logprobs(entries: list[Any], vocab_size: int) -> torch.Tensor:
    rows = []
    for entry in entries:
        row = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
        for token_id, value in entry.items():
            row[int(token_id)] = float(value.logprob)
        if not torch.isfinite(row).all():
            raise ValueError("vLLM did not return the complete prompt distribution")
        rows.append(row)
    return torch.stack(rows)


def collect(model: Path, output: Path) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    prompts = math_prompts()
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    vocab_size = model_vocab_size(config, tokenizer)
    llm = LLM(
        model=str(model),
        tensor_parallel_size=4,
        trust_remote_code=True,
        kv_cache_dtype="fp8",
        max_model_len=512,
        max_num_seqs=1,
        max_logprobs=vocab_size,
        gpu_memory_utilization=0.90,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=-1)
    rows = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        result = llm.generate(
            [{"prompt_token_ids": token_ids}], params, use_tqdm=False
        )[0]
        entries = result.prompt_logprobs
        if entries is None or len(entries) != len(token_ids):
            raise ValueError("vLLM prompt logprobs are not token-aligned")
        rows.append(
            {
                "token_ids": torch.tensor(token_ids[1:], dtype=torch.int32),
                "logprobs": _dense_logprobs(entries[1:], vocab_size).half(),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, output)
    print(f"DS4_TP4_COLLECT_COMPLETE={output}", flush=True)


def convert(source: Path, output: Path, device: str) -> None:
    """Materialize the MLite BF16 -> checkpoint-format arm one shard at a time."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    from examples.verl.ds4_checkpoint_roundtrip import _checkpoint_index, _scale_name
    from megatron.lite.model.deepseek_v4.lite.resync import (
        export_resync_weights,
        is_routed_expert,
    )
    from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8
    from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4

    config_dict = json.loads((source / "config.json").read_text())
    config = SimpleNamespace(**config_dict)
    quantization = config_dict["quantization_config"]
    block_shape = tuple(quantization.get("weight_block_size", (128, 128)))
    expert_dtype = config_dict.get("expert_dtype", "fp4")
    weight_map, shards = _checkpoint_index(source)
    names = set(weight_map)
    output.mkdir(parents=True, exist_ok=True)
    copy_checkpoint_metadata(source, output)
    target_device = torch.device(device)
    for shard_index, shard in enumerate(shards, 1):
        converted: dict[str, torch.Tensor] = {}
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            for name in handle.keys():
                if name.endswith(".scale"):
                    continue
                tensor = handle.get_tensor(name)
                scale_name = _scale_name(name) if name.endswith(".weight") else ""
                if scale_name not in names:
                    converted[name] = tensor
                    continue
                scale = handle.get_tensor(scale_name)
                if expert_dtype == "fp4" and is_routed_expert(name):
                    bf16 = dequantize_mxfp4(
                        tensor.to(target_device), scale.to(target_device)
                    ).to(torch.bfloat16)
                else:
                    bf16 = dequantize_block_fp8(
                        tensor.to(target_device), scale.to(target_device), block_shape
                    ).to(torch.bfloat16)
                for out_name, out_tensor in export_resync_weights(
                    [(name, bf16)], config
                ):
                    converted[out_name] = out_tensor.cpu()
                del bf16
        save_file(converted, output / shard, metadata=metadata)
        print(f"DS4_RESYNC_SHARD={shard_index}/{len(shards)}:{shard}", flush=True)
    print("DS4_RESYNC_CHECKPOINT_COMPLETE", flush=True)


def compare(reference: Path, candidate: Path, output: Path) -> None:
    report = compare_distributions(
        torch.load(reference, weights_only=True),
        torch.load(candidate, weights_only=True),
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    print("DS4_TP4_PARITY_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--model", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--source", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--device", default="cuda:0")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.model, args.output)
    elif args.command == "convert":
        convert(args.source, args.output, args.device)
    else:
        compare(args.reference, args.candidate, args.output)


if __name__ == "__main__":
    main()
