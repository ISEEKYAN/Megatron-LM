"""Exact format comparison with original pinned TileLang FP4 kernels; Slurm only."""

import argparse
import hashlib
import importlib.util
import os

import torch

from megatron.lite.primitive.quantization.ds41_index import quantize_index
from megatron.lite.primitive.quantization.ds41_kv import quantize_main_kv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-kernel", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU validation requires Slurm")
    with open(args.official_kernel, "rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == args.expected_sha256
    assert torch.cuda.is_available(), "GPU required; no skip or CPU fallback"
    spec = importlib.util.spec_from_file_location("official_ds41_kernel", args.official_kernel)
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    torch.manual_seed(1729)
    cases = {
        "random": torch.randn(32, 64, device="cuda", dtype=torch.bfloat16),
        "zero": torch.zeros(32, 64, device="cuda", dtype=torch.bfloat16),
        "saturation": torch.full((32, 64), 6.25, device="cuda", dtype=torch.bfloat16),
        "midpoints": torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.,
                                    -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5., -6.],
                                   device="cuda", dtype=torch.bfloat16).repeat(32, 4),
    }
    count = 0
    for name, x in cases.items():
        for codec, group, scale_dtype in (
            (quantize_main_kv, 16, torch.float8_e4m3fn),
            (quantize_index, 32, torch.float8_e8m0fnu),
        ):
            original = x.clone()
            packed, scale = kernel.fp4_act_quant(x, block_size=group, scale_dtype=scale_dtype)
            decoded = kernel.fp4_act_quant(x.clone(), block_size=group, scale_dtype=scale_dtype, inplace=True)
            actual = codec(x)
            torch.cuda.synchronize()
            assert torch.equal(x, original), "unexpected input mutation"
            assert torch.equal(packed.view(torch.uint8), actual.packed.view(torch.uint8)), (name, group, "codes")
            assert torch.equal(scale.view(torch.uint8), actual.scale.view(torch.uint8)), (name, group, "scales")
            assert torch.equal(decoded, actual.decoded), (name, group, "decoded")
            count += 1
    print(f"OFFICIAL_FP4_GPU_OK cases={count} job={os.environ['SLURM_JOB_ID']} device={torch.cuda.get_device_name()} torch={torch.__version__}")


if __name__ == "__main__":
    main()
