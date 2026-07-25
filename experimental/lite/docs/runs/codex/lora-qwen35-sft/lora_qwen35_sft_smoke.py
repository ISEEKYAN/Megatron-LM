# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tiny Qwen3.5 LoRA SFT smoke through the real MLite dist_opt path."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist


def _tiny_qwen35_config():
    from megatron.lite.model.qwen3_5.config import Qwen35Config

    return Qwen35Config(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["full_attention", "linear_attention"],
        partial_rotary_factor=1.0,
        max_position_embeddings=4096,
    )


def _optimizer_config():
    from megatron.lite.runtime.contracts.config import OptimizerConfig

    return OptimizerConfig(
        optimizer="adam",
        lr=1.0e-3,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        clip_grad=1.0,
    )


def _packed_sft_batch(vocab_size: int, step: int):
    from megatron.lite.runtime.contracts.data import PackedBatch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260725 + step)
    input_ids = torch.randint(0, vocab_size, (256,), generator=generator, device="cuda")
    labels = input_ids.roll(-1)
    labels[-1] = -100
    return PackedBatch(
        input_ids=input_ids,
        labels=labels,
        seq_lens=torch.tensor([input_ids.numel()], dtype=torch.int64, device="cuda"),
    )


def _snapshot(chunks, *, trainable: bool) -> dict[str, torch.Tensor]:
    result = {}
    for chunk_idx, chunk in enumerate(chunks):
        for name, param in chunk.named_parameters():
            if param.requires_grad == trainable:
                result[f"{chunk_idx}.{name}"] = param.detach().cpu().clone()
    return result


def _changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> list[str]:
    assert before.keys() == after.keys()
    return [name for name in before if not torch.equal(before[name], after[name])]


def main() -> None:
    import wandb

    from megatron.core import parallel_state as mpu
    from megatron.lite.model.qwen3_5.lite import protocol
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
    from megatron.lite.runtime.contracts.config import ParallelConfig
    from megatron.lite.runtime.contracts.handle import ModelHandle

    assert torch.cuda.is_available(), "CUDA is required"
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    torch.manual_seed(20260725)
    torch.cuda.manual_seed_all(20260725)

    parallel = ParallelConfig(tp=1, pp=1, cp=1, ep=1, etp=1, vpp=1)
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer="dist_opt",
        optimizer_config=_optimizer_config(),
        use_deepep=False,
        deterministic=True,
        lora={
            "enabled": True,
            "rank": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        },
    )
    cfg = _tiny_qwen35_config()
    bundle = protocol.build_model(cfg, impl_cfg=impl_cfg)
    assert bundle.extras["optimizer_backend"] == "dist_opt"
    assert bundle.extras["lora_spec"].enabled is True
    assert bundle.extras["lora_stats"]["attached_modules"] > 0

    chunks = bundle.chunks
    trainable_names = [
        name for chunk in chunks for name, param in chunk.named_parameters() if param.requires_grad
    ]
    frozen_names = [
        name for chunk in chunks for name, param in chunk.named_parameters() if not param.requires_grad
    ]
    assert trainable_names and frozen_names
    assert all("adapter" in name.lower() or "lora" in name.lower() for name in trainable_names)

    handle = ModelHandle(
        model=chunks[0],
        optimizer=bundle.optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=parallel),
        _extras={
            **bundle.extras,
            "model_chunks": chunks,
            "forward_step": bundle.forward_step,
            "finalize_grads": bundle.finalize_grads,
            "protocol": protocol,
        },
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    adapter_before = _snapshot(chunks, trainable=True)
    base_before = _snapshot(chunks, trainable=False)

    run = wandb.init(
        entity="megatron-core-moe-dev",
        project="mlite-lora-sft-smoke",
        name=os.environ.get("WANDB_RUN_NAME", "qwen35-lora-dist-opt-smoke"),
        mode="online",
        config={
            "model": "qwen3_5_tiny",
            "optimizer_backend": "dist_opt",
            "lora_enabled": True,
            "lora_rank": 4,
            "steps": 3,
        },
    )
    assert run.url and run.url.startswith("http"), f"missing online W&B URL: {run.url!r}"

    losses = []
    grad_norms = []
    for step in range(3):
        runtime.zero_grad(handle)
        result = runtime.forward_backward(
            handle,
            iter([_packed_sft_batch(cfg.vocab_size, step)]),
            None,
            num_microbatches=1,
        )
        loss = float(result.model_output.loss.detach())
        updated, grad_norm, _num_zeros = runtime.optimizer_step(handle)
        assert updated
        assert torch.isfinite(torch.tensor(loss))
        assert torch.isfinite(torch.tensor(grad_norm))
        losses.append(loss)
        grad_norms.append(grad_norm)
        run.log({"train/loss": loss, "train/grad_norm": grad_norm}, step=step)

    adapter_after = _snapshot(chunks, trainable=True)
    base_after = _snapshot(chunks, trainable=False)
    changed_adapters = _changed(adapter_before, adapter_after)
    changed_base = _changed(base_before, base_after)
    assert changed_adapters, "dist_opt completed but no LoRA adapter parameter changed"
    assert not changed_base, f"frozen base parameters changed: {changed_base[:5]}"

    url = run.url
    run.summary["attached_modules"] = bundle.extras["lora_stats"]["attached_modules"]
    run.summary["changed_adapter_tensors"] = len(changed_adapters)
    run.finish()
    print(
        "LORA_QWEN35_DIST_OPT_SFT_PASSED "
        f"steps=3 attached={bundle.extras['lora_stats']['attached_modules']} "
        f"changed_adapters={len(changed_adapters)} base_changed=0 "
        f"losses={losses} grad_norms={grad_norms} wandb={url}",
        flush=True,
    )

    if mpu.is_initialized():
        mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
