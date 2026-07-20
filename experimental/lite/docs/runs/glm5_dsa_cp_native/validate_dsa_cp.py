# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM5 DSA CP correctness, latency, and activation-memory validation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from megatron.lite.primitive.parallel.cp import (
    contiguous_position_ids_for_cp,
    contiguous_slice_for_cp,
)

if TYPE_CHECKING:
    from megatron.lite.primitive.modules.attention.dsa import DynamicSparseAttention


def _module(
    *, cp_size: int, cp_rank: int, cp_group, cp_mode: str
) -> "DynamicSparseAttention":
    from megatron.lite.primitive.modules.attention.dsa import DynamicSparseAttention

    torch.manual_seed(20260720)
    return (
        DynamicSparseAttention(
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
        )
        .cuda()
        .to(torch.bfloat16)
    )


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
    )


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


def _hf_reference_logits(rank: int, world: int, seq: int) -> dict:
    """Compare real dense and packed Glm5Model CP paths to Transformers."""
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM,
    )

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import load_hf_weights
    from megatron.lite.model.glm5.lite.model import Glm5Model
    from megatron.lite.primitive.parallel.state import ParallelState

    cfg = Glm5Config(
        num_hidden_layers=4,
        hidden_size=128,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        vocab_size=32,
        max_position_embeddings=seq,
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
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=["full", "full", "full", "shared"],
        num_nextn_predict_layers=0,
        # Transformers grouped-mm requires bf16 matrix strides to be aligned
        # to 16 bytes; keep the toy GLM5 reference dimensions kernel-valid.
        intermediate_size=32,
        moe_intermediate_size=16,
        first_k_dense_replace=1,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=3,
    )
    hf_cfg = GlmMoeDsaConfig(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        moe_intermediate_size=cfg.moe_intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        vocab_size=cfg.vocab_size,
        n_shared_experts=cfg.n_shared_experts,
        n_routed_experts=cfg.n_routed_experts,
        routed_scaling_factor=cfg.routed_scaling_factor,
        kv_lora_rank=cfg.kv_lora_rank,
        q_lora_rank=cfg.q_lora_rank,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        index_topk=cfg.index_topk,
        index_head_dim=cfg.index_head_dim,
        index_n_heads=cfg.index_n_heads,
        index_topk_freq=cfg.index_topk_freq,
        index_skip_topk_offset=cfg.index_skip_topk_offset,
        indexer_types=cfg.indexer_types,
        n_group=cfg.n_group,
        topk_group=cfg.topk_group,
        num_experts_per_tok=cfg.num_experts_per_tok,
        first_k_dense_replace=cfg.first_k_dense_replace,
        norm_topk_prob=cfg.norm_topk_prob,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
    )
    device = torch.device("cuda", rank)
    torch.manual_seed(20260611)
    reference = (
        GlmMoeDsaForCausalLM(hf_cfg).to(device=device, dtype=torch.bfloat16).eval()
    )
    with tempfile.TemporaryDirectory(prefix=f"glm5_hf_rank{rank}_") as hf_dir:
        # ``state_dict()`` exposes Transformers' fused in-memory expert
        # representation.  Its checkpoint writer instead emits the real GLM5
        # safetensor schema: one gate/up/down triplet per routed expert.  Make
        # that schema the sole source of truth, then reload both implementations
        # from it rather than feeding MLite a Transformers-private layout.
        reference.save_pretrained(hf_dir, safe_serialization=True)
        reference = (
            GlmMoeDsaForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.bfloat16)
            .to(device=device)
            .eval()
        )
        ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)
        train_cfg = SimpleNamespace(
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
        native = (
            Glm5Model(cfg, train_cfg, ps).to(device=device, dtype=torch.bfloat16).eval()
        )
        load_hf_weights(native, hf_dir, cfg, ps)
        torch.manual_seed(311)
        full_ids = torch.randint(0, cfg.vocab_size, (1, seq), device=device)
        local_ids = contiguous_slice_for_cp(
            full_ids, rank, world, seq_dim=1
        ).contiguous()
        local_pos = contiguous_position_ids_for_cp(
            full_ids.shape[1], rank, world, device
        )
        cu = torch.tensor([0, full_ids.shape[1]], device=device, dtype=torch.int32)
        packed = SimpleNamespace(
            cu_seqlens_q=cu,
            cu_seqlens_q_padded=cu,
            max_seqlen_q=full_ids.shape[1],
            cp_layout="contiguous",
        )
        with torch.no_grad():
            expected = contiguous_slice_for_cp(
                reference(full_ids).logits, rank, world, seq_dim=1
            )
            dense_actual = native(input_ids=local_ids)["logits"]
            packed_actual = native(
                input_ids=local_ids,
                position_ids=local_pos,
                packed_seq_params=packed,
            )["logits"]
        shared_attention = native.layers[3].self_attention.self_attention
        return {
            "dense_logits_cosine": _cosine(dense_actual, expected),
            "packed_logits_cosine": _cosine(packed_actual, expected),
            "sequence_length": seq,
            "index_topk": cfg.index_topk,
            "sparse_fraction": cfg.index_topk / seq,
            "index_share": {
                "schedule": cfg.indexer_types,
                "shared_layer": 4,
                "source_layer": shared_attention.index_share_source_layer,
                "shared_layer_has_indexer": shared_attention.indexer is not None,
                "verified_by_dense_and_packed_forward": True,
            },
        }


def _dense_case(seq: int, rank: int, world: int, warmup: int, steps: int):
    from megatron.lite.primitive.modules.attention.dsa import build_rope_cache

    device = torch.device("cuda", rank)
    torch.manual_seed(123 + seq)
    full_x = torch.randn(1, seq, 128, device=device, dtype=torch.bfloat16)
    local_x = contiguous_slice_for_cp(full_x, rank, world, seq_dim=1).contiguous()
    cos, sin = build_rope_cache(
        dim=64, max_position_embeddings=seq, rope_theta=1_000_000.0, device=device
    )
    full_pos = torch.arange(seq, device=device).unsqueeze(0)
    local_pos = contiguous_position_ids_for_cp(seq, rank, world, device)

    reference = _module(cp_size=1, cp_rank=0, cp_group=None, cp_mode="native")
    ref_out, ref_grad = _step(reference, full_x, cos, sin, full_pos)
    expected_out = contiguous_slice_for_cp(ref_out, rank, world, seq_dim=1)
    expected_grad = contiguous_slice_for_cp(ref_grad, rank, world, seq_dim=1)
    ref_metrics = _measure(
        reference, full_x, cos, sin, full_pos, warmup=warmup, steps=steps
    )
    del reference

    metrics = {"cp1": ref_metrics}
    for mode in ("native", "legacy_gather_all"):
        module = _module(
            cp_size=world, cp_rank=rank, cp_group=dist.group.WORLD, cp_mode=mode
        )
        out, grad = _step(module, local_x, cos, sin, local_pos)
        mode_metrics = _measure(
            module, local_x, cos, sin, local_pos, warmup=warmup, steps=steps
        )
        mode_metrics.update(
            output_cosine=_cosine(out, expected_out),
            input_grad_cosine=_cosine(grad, expected_grad),
        )
        metrics[mode] = mode_metrics
        del module
    return metrics


def _thd_case(lengths: list[int], rank: int, world: int):
    from megatron.lite.primitive.modules.attention.dsa import (
        _packed_cp_layout,
        build_rope_cache,
    )

    device = torch.device("cuda", rank)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32
    )
    total = int(cu[-1])
    torch.manual_seed(818)
    full_x = torch.randn(1, total, 128, device=device, dtype=torch.bfloat16)
    local_rows, _ = _packed_cp_layout(
        cu,
        cp_size=world,
        cp_rank=rank,
        device=device,
    )
    local_x = full_x.index_select(1, local_rows)
    full_pos = torch.cat(
        [torch.arange(length, device=device) for length in lengths]
    ).unsqueeze(0)
    local_pos = full_pos.index_select(1, local_rows)
    packed = SimpleNamespace(
        cu_seqlens_q=cu,
        cu_seqlens_q_padded=cu,
        max_seqlen_q=max(lengths),
        cp_layout="contiguous",
    )
    cos, sin = build_rope_cache(
        dim=64,
        max_position_embeddings=max(lengths),
        rope_theta=1_000_000.0,
        device=device,
    )

    reference = _module(cp_size=1, cp_rank=0, cp_group=None, cp_mode="native")
    ref_out, ref_grad = _step(reference, full_x, cos, sin, full_pos, packed)
    native = _module(
        cp_size=world, cp_rank=rank, cp_group=dist.group.WORLD, cp_mode="native"
    )
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
            {
                "sample": sample,
                "cp": float(cp_loss),
                "cp1": float(ref_loss),
                "abs_diff": float((cp_loss - ref_loss).abs()),
            }
        )
    return result


def _assert_validation(gathered: list[dict]) -> None:
    for rank_result in gathered:
        if "hf_reference" in rank_result:
            hf_reference = rank_result["hf_reference"]
            for path in ("dense_logits_cosine", "packed_logits_cosine"):
                hf_cosine = hf_reference[path]
                if hf_cosine < 0.9999:
                    raise AssertionError(f"HF {path} below 0.9999: {hf_reference}")
            if hf_reference["index_topk"] >= hf_reference["sequence_length"]:
                raise AssertionError(
                    f"HF leg did not exercise sparse top-k: {hf_reference}"
                )
            index_share = hf_reference["index_share"]
            if index_share["schedule"] != ["full", "full", "full", "shared"]:
                raise AssertionError(f"Unexpected IndexShare schedule: {index_share}")
            if (
                index_share["source_layer"] != 3
                or index_share["shared_layer_has_indexer"]
            ):
                raise AssertionError(
                    f"IndexShare layer did not reuse source top-k: {index_share}"
                )
        if rank_result["thd"]["output_cosine"] < 0.9999:
            raise AssertionError(
                f"THD output cosine below 0.9999: {rank_result['thd']}"
            )
        if rank_result["thd"]["input_grad_cosine"] < 0.9999:
            raise AssertionError(
                f"THD input-grad cosine below 0.9999: {rank_result['thd']}"
            )
        for seq, dense in rank_result["dense"].items():
            native = dense["native"]
            if native["output_cosine"] < 0.9999:
                raise AssertionError(
                    f"dense seq={seq} output cosine below 0.9999: {dense}"
                )
            if native["input_grad_cosine"] < 0.9999:
                raise AssertionError(
                    f"dense seq={seq} input-grad cosine below 0.9999: {dense}"
                )
            if native["activation_peak_mb"] >= dense["cp1"]["activation_peak_mb"]:
                raise AssertionError(
                    "CP-native activation peak did not shrink "
                    f"for seq={seq}: cp{rank_result['world_size']}="
                    f"{native['activation_peak_mb']:.3f}MiB "
                    f"cp1={dense['cp1']['activation_peak_mb']:.3f}MiB"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="4096,4104,8192,8200")
    parser.add_argument("--thd-seq-lens", default="4096,4104")
    parser.add_argument("--hf-seq-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--only-thd", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world < 2:
        raise RuntimeError(f"This validation requires CP world size >=2, got {world}.")
    seq_lens = [int(value) for value in args.seq_lens.split(",")]
    thd_seq_lens = [int(value) for value in args.thd_seq_lens.split(",")]
    if any(
        seq <= 0 or seq % world
        for seq in [*seq_lens, sum(thd_seq_lens), args.hf_seq_len]
    ):
        raise ValueError(
            "dense/HF lengths and total packed THD tokens must be positive and "
            f"divisible by CP world={world}; got dense={seq_lens}, "
            f"packed={thd_seq_lens}, hf={args.hf_seq_len}"
        )
    if args.hf_seq_len <= 512:
        raise ValueError("HF sequence length must exceed index_topk=512.")

    result = {
        "world_size": world,
        "rank": rank,
        "dense": {},
        "thd": _thd_case(thd_seq_lens, rank, world),
    }
    if not args.only_thd:
        for seq in seq_lens:
            result["dense"][str(seq)] = _dense_case(
                seq, rank, world, args.warmup, args.steps
            )
        result["hf_reference"] = _hf_reference_logits(rank, world, args.hf_seq_len)
    gathered = [None] * world
    dist.all_gather_object(gathered, result)
    if rank == 0:
        _assert_validation(gathered)
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump({"ranks": gathered}, stream, indent=2, sort_keys=True)
        print(
            "GLM5_DSA_CP_VALIDATION",
            json.dumps({"output": args.output, "ranks": gathered}, sort_keys=True),
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
