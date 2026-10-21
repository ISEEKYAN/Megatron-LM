# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build the deployment-aligned model with frozen base weights and shared LoRA owners.

Launch under torchrun with the matching vLLM build documented in LORA.md.
Training/optimizer steps use the normal runtime API, not a separate LoRA engine.
"""
import argparse
import os

import torch
import torch.distributed as dist

from megatron.lite.runtime import create_runtime
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts import ParallelConfig
from megatron.lite.runtime.contracts.config import RuntimeConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Native release-format model directory')
    parser.add_argument('--ep', type=int, default=8)
    parser.add_argument('--layers', type=int, nargs=2, required=True, metavar=('START', 'END'))
    args = parser.parse_args()
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
        if dist.get_rank() == 0:
            print('Built adapter-only training handle:', type(handle).__name__)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
