# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU bf16 HF weight round-trip validation for GLM5 DSA IndexShare."""

from __future__ import annotations

import argparse
import json
import tempfile
from types import SimpleNamespace

import torch
from safetensors import safe_open


def _config(*, index_share: bool):
    from megatron.lite.model.glm5.config import Glm5Config

    return Glm5Config(
        num_hidden_layers=4,
        hidden_size=128,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        vocab_size=32,
        max_position_embeddings=520,
        initializer_range=0.002,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=512,
        index_topk_freq=4 if index_share else 1,
        index_skip_topk_offset=3 if index_share else 0,
        indexer_types=(
            ["full", "full", "full", "shared"]
            if index_share
            else ["full", "full", "full", "full"]
        ),
        num_nextn_predict_layers=0,
        intermediate_size=32,
        moe_intermediate_size=16,
        first_k_dense_replace=1,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=3,
    )


def _train_config(ps):
    return SimpleNamespace(
        tp=ps.tp_size,
        ep=ps.ep_size,
        etp=ps.etp_size,
        pp=ps.pp_size,
        cp=ps.cp_size,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
        deterministic=True,
    )


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
    )


def _run_case(*, index_share: bool, device: torch.device) -> dict:
    from megatron.lite.model.glm5.lite.checkpoint import (
        load_hf_weights,
        save_hf_weights,
    )
    from megatron.lite.model.glm5.lite.model import Glm5Model
    from megatron.lite.primitive.parallel.state import ParallelState

    cfg = _config(index_share=index_share)
    ps = ParallelState()
    torch.manual_seed(20260720 + int(index_share))
    original = (
        Glm5Model(cfg, _train_config(ps), ps)
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    state = original.state_dict()
    source_indexer = "layers.2.self_attention.self_attention.indexer.wq_b.weight"
    shared_indexer = "layers.3.self_attention.self_attention.indexer.wq_b.weight"
    if source_indexer not in state:
        raise AssertionError(f"source layer indexer missing: {source_indexer}")
    if (shared_indexer in state) == index_share:
        raise AssertionError(
            "shared layer indexer ownership does not match the requested schedule: "
            f"index_share={index_share}, present={shared_indexer in state}"
        )

    with tempfile.TemporaryDirectory(prefix="glm5_gpu_roundtrip_") as hf_dir:
        save_hf_weights(original, hf_dir, cfg, ps, export_dtype=torch.bfloat16)
        with safe_open(
            f"{hf_dir}/model.safetensors", framework="pt", device="cpu"
        ) as handle:
            exported_names = set(handle.keys())
        hf_source_indexer = "model.layers.2.self_attn.indexer.wq_b.weight"
        hf_shared_indexer = "model.layers.3.self_attn.indexer.wq_b.weight"
        if hf_source_indexer not in exported_names:
            raise AssertionError(
                f"source indexer was not exported: {hf_source_indexer}"
            )
        if (hf_shared_indexer in exported_names) == index_share:
            raise AssertionError(
                "shared layer HF indexer ownership does not match the requested schedule: "
                f"index_share={index_share}, present={hf_shared_indexer in exported_names}"
            )

        reloaded = (
            Glm5Model(cfg, _train_config(ps), ps)
            .to(device=device, dtype=torch.bfloat16)
            .eval()
        )
        load_hf_weights(reloaded, hf_dir, cfg, ps)

        reloaded_state = reloaded.state_dict()
        if state.keys() != reloaded_state.keys():
            raise AssertionError(
                "state keys changed across HF round-trip: "
                f"missing={sorted(state.keys() - reloaded_state.keys())}, "
                f"unexpected={sorted(reloaded_state.keys() - state.keys())}"
            )
        mismatched = [
            name
            for name, tensor in state.items()
            if not torch.equal(tensor, reloaded_state[name])
        ]
        if mismatched:
            raise AssertionError(
                f"{len(mismatched)} tensors changed across HF round-trip: "
                f"{mismatched[:10]}"
            )

        torch.manual_seed(311)
        input_ids = torch.randint(
            0, cfg.vocab_size, (1, cfg.max_position_embeddings), device=device
        )
        with torch.no_grad():
            expected = original(input_ids=input_ids)["logits"]
            actual = reloaded(input_ids=input_ids)["logits"]
        logits_max_abs = float((actual.float() - expected.float()).abs().max())
        logits_cosine = _cosine(actual, expected)
        if logits_max_abs != 0.0 and logits_cosine < 0.9999:
            raise AssertionError(
                "logits changed across HF round-trip: "
                f"max_abs={logits_max_abs}, cosine={logits_cosine}"
            )

    return {
        "index_share": index_share,
        "schedule": [
            cfg.dsa_indexer_type(layer) for layer in range(cfg.num_hidden_layers)
        ],
        "bf16_tensor_count": sum(
            tensor.dtype == torch.bfloat16 for tensor in state.values()
        ),
        "state_tensor_count": len(state),
        "state_bitwise_equal": True,
        "source_indexer_exported": True,
        "shared_indexer_exported": hf_shared_indexer in exported_names,
        "logits_max_abs": logits_max_abs,
        "logits_cosine": logits_cosine,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GLM5 weight round-trip validation requires CUDA.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    results = [
        _run_case(index_share=False, device=device),
        _run_case(index_share=True, device=device),
    ]
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump({"cases": results}, stream, indent=2, sort_keys=True)
    print(
        "GLM5_GPU_WEIGHT_ROUNDTRIP",
        json.dumps({"output": args.output, "cases": results}, sort_keys=True),
    )


if __name__ == "__main__":
    main()
