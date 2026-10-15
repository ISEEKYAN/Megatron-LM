# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU/CUDA RMS provider example with a live FP32 norm master."""
import argparse

import torch
from megatron.lite.primitive.modules import deployment_math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    torch.manual_seed(71)
    values = torch.randn(3, 128, device=args.device, dtype=torch.bfloat16)
    values.requires_grad_()
    weight = torch.nn.Parameter(torch.ones(128, device=args.device))
    if args.device == 'cuda':
        output = deployment_math.rms_norm(values, weight, (128,), 1e-6)
    else:
        # CPU reference example only; the CUDA provider has no CPU fallback.
        def reference(x, master):
            return torch.nn.functional.rms_norm(
                x, (128,), deployment_math.decoded_bf16_master(master), 1e-6
            )

        output = deployment_math.visible_forward(reference, reference, values, weight)
    output.float().square().mean().backward()
    assert output.dtype == torch.bfloat16
    assert weight.dtype == weight.grad.dtype == torch.float32
    assert torch.isfinite(weight.grad).all() and torch.count_nonzero(weight.grad)
    assert values.grad is not None and torch.isfinite(values.grad).all()
    print('RMS reference VJP and FP32 master: PASS')


if __name__ == '__main__':
    main()
