# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build the deployment-aligned model with frozen base weights and shared LoRA owners.

Launch under torchrun with the matching vLLM build documented in LORA.md.
Training/optimizer steps use the normal runtime API, not a separate LoRA engine.
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.lite.runtime import create_runtime
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts import ParallelConfig
from megatron.lite.runtime.contracts.config import RuntimeConfig
from megatron.lite.runtime.contracts.data import PackedBatch


def train_and_export(runtime, handle, batch, steps, output):
    """Run ordinary runtime updates, then collectively export adapter factors only."""
    from safetensors.torch import save_file

    if steps < 1:
        raise ValueError('At least one training step is required')
    with runtime.train_mode(handle):
        for step in range(steps):
            runtime.zero_grad(handle)
            result = runtime.forward_backward(handle, iter([batch]), loss_fn=None)
            success, grad_norm, _ = runtime.optimizer_step(handle)
            if not success or not math.isfinite(float(grad_norm)) or grad_norm <= 0:
                raise RuntimeError('Expected an effective finite adapter update')
            if dist.get_rank() == 0:
                print(dict(step=step, grad_norm=float(grad_norm), metrics=result.metrics))
    factors = dict(runtime.export_weights(handle, target='lora'))
    if not factors or any('.lora_' not in name for name in factors):
        raise ValueError('Adapter export must contain only LoRA factors')
    if dist.get_rank() == 0:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        save_file(factors, output)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Native release-format model directory')
    parser.add_argument('--ep', type=int, default=8)
    parser.add_argument('--layers', type=int, nargs=2, required=True, metavar=('START', 'END'))
    parser.add_argument('--token-ids', required=True, help='JSON array of token IDs from real text')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--output', required=True, help='Adapter-only safetensors output')
    args = parser.parse_args()
    tokens = json.loads(Path(args.token_ids).read_text())
    if len(tokens) < 2 or any(type(i) is not int or i < 0 for i in tokens):
        parser.error('--token-ids must contain at least two nonnegative integer IDs')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    try:
        backend = MegatronLiteConfig(
            model_name='deepseek_v41',
            impl='vllm',
            hf_path=args.model,
            load_hf_weights=True,
            parallel=ParallelConfig(ep=args.ep),
            impl_cfg=dict(
                optimizer='adam',
                optimizer_config=dict(
                    lr=1e-5,
                    min_lr=1e-5,
                    adam_beta1=0.9,
                    adam_beta2=0.95,
                    adam_eps=1e-8,
                    weight_decay=0.0,
                    clip_grad=1.0,
                ),
                lora=dict(rank=32, alpha=32, layers=args.layers),
            ),
        )
        runtime = create_runtime(RuntimeConfig(backend='mlite', backend_cfg=backend))
        handle = runtime.build_model()
        ids = torch.tensor(tokens, device='cuda', dtype=torch.long)
        batch = PackedBatch(
            input_ids=ids,
            labels=ids,  # DS4.1 prepare_batch rolls labels and masks each document end.
            seq_lens=torch.tensor([ids.numel()], device=ids.device, dtype=torch.int32),
        )
        train_and_export(runtime, handle, batch, args.steps, args.output)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
