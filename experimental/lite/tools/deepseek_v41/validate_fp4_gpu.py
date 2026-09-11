"""Exact format comparison with original pinned TileLang FP4 kernels; Slurm only."""

import argparse
import hashlib
import importlib.util
import os

import torch
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
from megatron.lite.primitive.quantization.ds41_fp8 import (
    dynamic_fp8_linear,
    quantize_swa,
)
from megatron.lite.primitive.quantization.ds41_index import quantize_index
from megatron.lite.primitive.quantization.ds41_kv import quantize_main_kv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-kernel", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--include-fp8", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU validation requires Slurm")
    with open(args.official_kernel, "rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == args.expected_sha256
    assert torch.cuda.is_available(), "GPU required; no skip or CPU fallback"
    spec = importlib.util.spec_from_file_location(
        "official_ds41_kernel", args.official_kernel
    )
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    torch.manual_seed(1729)
    cases = {
        "random": torch.randn(32, 64, device="cuda", dtype=torch.bfloat16),
        "zero": torch.zeros(32, 64, device="cuda", dtype=torch.bfloat16),
        "saturation": torch.full((32, 64), 6.25, device="cuda", dtype=torch.bfloat16),
        "midpoints": torch.tensor(
            [
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                6.0,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -3.5,
                -5.0,
                -6.0,
            ],
            device="cuda",
            dtype=torch.bfloat16,
        ).repeat(32, 4),
    }
    count = 0
    for name, x in cases.items():
        for codec, group, scale_dtype in (
            (quantize_main_kv, 16, torch.float8_e4m3fn),
            (quantize_index, 32, torch.float8_e8m0fnu),
        ):
            original = x.clone()
            packed, scale = kernel.fp4_act_quant(
                x, block_size=group, scale_dtype=scale_dtype
            )
            decoded = kernel.fp4_act_quant(
                x.clone(), block_size=group, scale_dtype=scale_dtype, inplace=True
            )
            actual = codec(x)
            torch.cuda.synchronize()
            assert torch.equal(x, original), "unexpected input mutation"
            assert torch.equal(
                packed.view(torch.uint8), actual.packed.view(torch.uint8)
            ), (name, group, "codes")
            assert torch.equal(
                scale.view(torch.uint8), actual.scale.view(torch.uint8)
            ), (name, group, "scales")
            assert torch.equal(decoded, actual.decoded), (name, group, "decoded")
            count += 1
    print(
        f"OFFICIAL_FP4_GPU_OK cases={count} job={os.environ['SLURM_JOB_ID']} device={torch.cuda.get_device_name()} torch={torch.__version__}"
    )
    if args.include_fp8:
        for name, x in cases.items():
            encoded, scale = kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
            decoded = kernel.act_quant(
                x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True
            )
            actual = quantize_swa(x)
            assert torch.equal(
                encoded.view(torch.uint8), actual.values.view(torch.uint8)
            ), (name, "FP8 codes")
            assert torch.equal(
                scale.view(torch.uint8), actual.scale.view(torch.uint8)
            ), (name, "FP8 scales")
            assert torch.equal(decoded, actual.decoded), (name, "FP8 decoded")
        torch.set_default_dtype(torch.bfloat16)
        x = cases["random"].clone().requires_grad_()
        weight = torch.randn(
            128, 64, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        a, sa = kernel.act_quant(x.detach(), 32, "ue8m0", torch.float8_e8m0fnu)
        b, sb = quantize_block_fp8(weight.detach(), (32, 32), scale_format="e8m0")
        reference = kernel.fp8_gemm(a, sa, b, sb, torch.float8_e8m0fnu, block_size=32)
        result = dynamic_fp8_linear(x, weight)
        error = (result.float() - reference.float()).abs().max().item()
        assert torch.equal(result, reference), ("FP8 GEMM", error)
        grad = torch.randn_like(result)
        (result * grad).sum().backward()
        ref_x = (
            (a.float().reshape(32, 2, 32) * sa.float()[:, :, None])
            .reshape(32, 64)
            .bfloat16()
        )
        ref_w = (
            (b.float().reshape(4, 32, 2, 32) * sb.float()[:, None, :, None])
            .reshape(128, 64)
            .bfloat16()
        )
        assert torch.equal(
            x.grad, (grad.float() @ ref_w.float()).bfloat16()
        ), "input derivative"
        assert torch.equal(
            weight.grad, (grad.float().T @ ref_x.float()).bfloat16()
        ), "weight derivative"
        print(
            f"OFFICIAL_FP8_GPU_OK activation_cases={len(cases)} gemm_max_error={error} derivatives=2 job={os.environ['SLURM_JOB_ID']}"
        )


if __name__ == "__main__":
    main()
