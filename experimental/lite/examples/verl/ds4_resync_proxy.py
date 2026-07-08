# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-GPU DeepSeek-V4 serialized-resync proxy validation.

The proxy builds a small random MLite model, serializes an official-format
mixed block-FP8/MXFP4 checkpoint, reloads it through MLite to BF16, serializes
it again, and compares direct vLLM load against native checkpoint reload.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def tiny_config() -> dict[str, Any]:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "torch_dtype": "bfloat16",
        "vocab_size": 1024,
        "hidden_size": 1024,
        "moe_intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_hash_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 128,
        "o_lora_rank": 128,
        "o_groups": 8,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "routed_scaling_factor": 1.0,
        "norm_topk_prob": True,
        "scoring_func": "sqrtsoftplus",
        "swiglu_limit": 0.0,
        "max_position_embeddings": 512,
        "rope_theta": 10000.0,
        "compress_rope_theta": 40000.0,
        "compress_ratios": [0, 0],
        "sliding_window": 64,
        "hc_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 4,
        "index_head_dim": 128,
        "index_n_heads": 8,
        "index_topk": 16,
        "num_nextn_predict_layers": 0,
        "rms_norm_eps": 1e-6,
        "expert_dtype": "fp4",
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }


def _proxy_parallel_config():
    from megatron.lite.runtime.contracts import ParallelConfig

    return ParallelConfig(tp=1, etp=1, ep=1, pp=1, vpp=1, cp=1)


def _proxy_impl_config():
    from megatron.lite.model.deepseek_v4.lite.protocol import ImplConfig

    return ImplConfig(
        parallel=_proxy_parallel_config(),
        optimizer=None,
        mtp_enable=False,
        mtp_enable_train=False,
        attention_backend_override="local",
    )


def _build_bundle(config_dict: dict[str, Any]):
    from megatron.lite.model.deepseek_v4.lite.protocol import build_model
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    config = DeepseekV4Config._from_hf_dict(config_dict)
    bundle = build_model(config, impl_cfg=_proxy_impl_config())
    return config, bundle


def _initialize_model(chunks: list[torch.nn.Module]) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260708)
    with torch.no_grad():
        for chunk in chunks:
            for name, parameter in chunk.named_parameters():
                if name.endswith("norm.weight"):
                    parameter.fill_(1.0)
                elif parameter.dtype.is_floating_point:
                    parameter.normal_(mean=0.0, std=0.01, generator=generator)


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def _save_checkpoint(path: Path, config, bundle) -> None:
    from megatron.lite.model.deepseek_v4.lite.protocol import save_hf_weights

    save_hf_weights(
        bundle.chunks,
        str(path),
        config,
        bundle.parallel_state,
        target="vllm_checkpoint",
        export_dtype="bfloat16",
    )


def _load_checkpoint(path: Path, config, bundle) -> None:
    from megatron.lite.model.deepseek_v4.lite.protocol import load_hf_weights

    for chunk in bundle.chunks:
        load_hf_weights(chunk, str(path), config, bundle.parallel_state)


def _fixed_token_sequences(vocab_size: int) -> list[list[int]]:
    return [
        [2 + ((row * 31 + column * 17) % (vocab_size - 2)) for column in range(12)]
        for row in range(8)
    ]


@torch.inference_mode()
def _mlite_logprobs(
    model: torch.nn.Module, sequences: list[list[int]]
) -> list[torch.Tensor]:
    rows = []
    for sequence in sequences:
        tokens = torch.tensor(sequence, dtype=torch.long, device="cuda").unsqueeze(0)
        logits = model(input_ids=tokens, enable_mtp=False)["logits"].float()
        rows.append(torch.log_softmax(logits[0, :-1], dim=-1).cpu())
    return rows


def _dense_logprob_row(entry: dict[int, Any], vocab_size: int) -> torch.Tensor:
    row = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
    for token_id, value in entry.items():
        row[int(token_id)] = float(
            value.logprob if hasattr(value, "logprob") else value
        )
    if not torch.isfinite(row).all():
        raise ValueError(
            f"vLLM returned only {torch.isfinite(row).sum().item()} / {vocab_size} logits"
        )
    return row


def _vllm_logprobs(
    llm, sequences: list[list[int]], vocab_size: int
) -> list[torch.Tensor]:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=-1)
    rows = []
    for sequence in sequences:
        output = llm.generate([{"prompt_token_ids": sequence}], params, use_tqdm=False)[
            0
        ]
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None or len(prompt_logprobs) != len(sequence):
            raise ValueError("vLLM prompt logprobs are not token-aligned")
        rows.append(
            torch.stack(
                [
                    _dense_logprob_row(prompt_logprobs[position], vocab_size)
                    for position in range(1, len(sequence))
                ]
            )
        )
    return rows


def _comparison(
    reference: list[torch.Tensor], candidate: list[torch.Tensor]
) -> dict[str, float]:
    ref = torch.cat(reference).float()
    cand = torch.cat(candidate).float()
    if ref.shape != cand.shape:
        raise ValueError(
            f"logprob shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}"
        )
    delta = cand - ref
    probability = ref.exp()
    kl = (probability * (ref - cand)).sum(dim=-1)
    return {
        "max_abs": float(delta.abs().max()),
        "p99_abs": float(torch.quantile(delta.abs().flatten(), 0.99)),
        "max_kl": float(kl.max()),
        "p99_kl": float(torch.quantile(kl, 0.99)),
    }


def _dequantized_checkpoint_diff(
    direct_path: Path, resync_path: Path
) -> dict[str, Any]:
    from safetensors.torch import load_file

    from megatron.lite.model.deepseek_v4.lite.checkpoint import (
        _dequantize_scaled_tensor,
    )

    direct = load_file(direct_path / "model.safetensors")
    resync = load_file(resync_path / "model.safetensors")
    if direct.keys() != resync.keys():
        raise ValueError("direct and resync checkpoint names differ")
    scale_mismatches = 0
    scale_values = 0
    worst_relative_l2 = 0.0
    worst_max_abs = 0.0
    worst_name = ""
    for name in sorted(direct):
        if name.endswith(".scale"):
            left = direct[name].view(torch.uint8)
            right = resync[name].view(torch.uint8)
            scale_mismatches += int((left != right).sum())
            scale_values += left.numel()
            continue
        left, right = direct[name], resync[name]
        if name.endswith(".weight"):
            scale_name = f"{name[:-7]}.scale"
            if scale_name in direct:
                target_shape = (
                    (*left.shape[:-1], left.shape[-1] * 2)
                    if left.dtype == torch.int8
                    else left.shape
                )
                left = _dequantize_scaled_tensor(
                    left, direct[scale_name], torch.Size(target_shape)
                )
                right = _dequantize_scaled_tensor(
                    right, resync[scale_name], torch.Size(target_shape)
                )
        if not left.dtype.is_floating_point:
            if not torch.equal(left, right):
                raise ValueError(f"non-floating checkpoint tensor changed: {name}")
            continue
        error = (right.float() - left.float()).abs()
        relative_l2 = float(
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(left.float()).clamp_min(1e-30)
        )
        max_abs = float(error.max()) if error.numel() else 0.0
        if relative_l2 > worst_relative_l2:
            worst_relative_l2, worst_max_abs, worst_name = relative_l2, max_abs, name
    return {
        "scale_mismatches": scale_mismatches,
        "scale_values": scale_values,
        "scale_mismatch_fraction": scale_mismatches / max(scale_values, 1),
        "worst_relative_l2": worst_relative_l2,
        "worst_max_abs": worst_max_abs,
        "worst_name": worst_name,
    }


def _fsync_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(output_dir: Path) -> None:
    from vllm import LLM

    output_dir.mkdir(parents=True, exist_ok=True)
    direct_path = output_dir / "direct-checkpoint"
    resync_path = output_dir / "resync-checkpoint"
    config_dict = tiny_config()
    _write_config(direct_path, config_dict)
    _write_config(resync_path, config_dict)

    config, source_bundle = _build_bundle(config_dict)
    _initialize_model(source_bundle.chunks)
    _save_checkpoint(direct_path, config, source_bundle)
    del source_bundle
    torch.cuda.empty_cache()

    config, resync_bundle = _build_bundle(config_dict)
    _load_checkpoint(direct_path, config, resync_bundle)
    sequences = _fixed_token_sequences(config.vocab_size)
    mlite_rows = _mlite_logprobs(resync_bundle.chunks[0], sequences)
    _save_checkpoint(resync_path, config, resync_bundle)
    del resync_bundle
    torch.cuda.empty_cache()

    checkpoint_diff = _dequantized_checkpoint_diff(direct_path, resync_path)
    dist.destroy_process_group()
    os.environ["VERL_VLLM_FP8_QUANT_ENABLED"] = "0"
    llm = LLM(
        model=str(direct_path),
        tokenizer=None,
        skip_tokenizer_init=True,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=1,
        max_logprobs=config.vocab_size,
        gpu_memory_utilization=0.5,
        kv_cache_dtype="fp8",
        worker_extension_cls=(
            "verl_mlite.rollout.vllm_worker.VllmCheckpointPathWorkerExtension"
        ),
    )
    direct_rows = _vllm_logprobs(llm, sequences, config.vocab_size)
    llm.collective_rpc(
        "reload_checkpoint_from_path", args=(str(resync_path),), timeout=None
    )
    resync_rows = _vllm_logprobs(llm, sequences, config.vocab_size)

    report = {
        "schema_version": 1,
        "checkpoint_diff": checkpoint_diff,
        "vllm_resync_vs_direct": _comparison(direct_rows, resync_rows),
        "mlite_vs_vllm_direct": _comparison(direct_rows, mlite_rows),
        "sequence_count": len(sequences),
        "predicted_token_count": sum(len(sequence) - 1 for sequence in sequences),
    }
    report_path = output_dir / "report.json"
    _fsync_json(report_path, report)
    marker_path = output_dir / "DS4_RESYNC_PROXY_COMPLETE"
    with marker_path.open("w") as marker:
        marker.write("complete\n")
        marker.flush()
        os.fsync(marker.fileno())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("DS4_RESYNC_PROXY_COMPLETE", flush=True)
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() != 1:
        raise ValueError("The proxy requires exactly one distributed rank")
    torch.cuda.set_device(0)
    run(args.output_dir)


if __name__ == "__main__":
    main()
