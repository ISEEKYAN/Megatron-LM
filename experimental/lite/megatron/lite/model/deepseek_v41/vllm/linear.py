# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pinned BI linear forwards with the W9-b5 decoded-operand VJPs.

Only the forward must be batch-invariant and bitwise. Backward GEMMs follow the TE/Megatron
recipe: BF16 operands on tensor cores with FP32 accumulation (the decoded operands are exact
in BF16), and the weight gradient lands in the FP32 ``main_grad`` when DDP owns one.
"""
import torch

from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8, quantize_block_fp8


def _row_scales(scale, rows, inner):
    tiles, groups = (rows + 127) // 128, inner // 32
    ktiles = (groups + 3) // 4
    return (
        scale.view(tiles, ktiles, 32, 4, 4)
        .transpose(1, 3)
        .contiguous()
        .view(tiles * 128, ktiles * 4)[:rows, :groups]
    )


def tc_mm(a, b, out_dtype=None):
    """``a @ b`` with BF16 operands on tensor cores and FP32 accumulation; the result is
    BF16 unless ``out_dtype`` asks for the FP32 accumulator. CPU (derivative references
    only): the same rounded operands in an FP32 GEMM."""
    a, b = a.bfloat16(), b.bfloat16()
    if not a.is_cuda:
        out = a.float() @ b.float()
        return out.bfloat16() if out_dtype is None else out.to(out_dtype)
    return a @ b if out_dtype is None else torch.mm(a, b, out_dtype=out_dtype)


def accumulate_wgrad(owners, dw, like):
    """Weight gradient for the forward input ``like`` whose rows stack ``owners``.

    When every owner has an FP32 ``main_grad`` (DDP), ``dw`` (FP32; None means zero) is
    added there and DDP's placeholder gradient is returned, as Megatron's gradient-
    accumulation fusion does; otherwise ``dw`` itself (autograd casts it to ``like``'s
    dtype)."""
    if not owners or any(
        getattr(o, "main_grad", None) is None or o.main_grad.dtype != torch.float32 for o in owners
    ):
        return torch.zeros_like(like) if dw is None else dw
    from megatron.lite.primitive.parallel.linear import _dummy_wgrad

    start = 0
    for owner in owners:
        owner.grad_added_to_main_grad = True
        if dw is None:
            continue
        rows = owner.shape[0] if len(owners) > 1 else dw.shape[0]
        owner.main_grad.add_(dw[start : start + rows].reshape(owner.main_grad.shape))
        start += rows
    return _dummy_wgrad(like)


class _MXFP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, activation, row_perm=None, owners=None):
        if not x.is_cuda or not weight.is_cuda:
            raise RuntimeError("Deployment MXFP8 linear requires CUDA")
        ctx.owners, ctx.like = owners, weight
        from vllm.model_executor.kernels.linear.mxfp8.batch_invariant import _uses_tensor_core
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
            swizzle_mxfp8_scale,
        )

        flat = x.reshape(-1, x.shape[-1])
        ctx.input_shape, ctx.weight_dtype = x.shape, weight.dtype
        if not flat.shape[0]:
            ctx.row_perm = None
            w = weight.to(x.dtype)
            ctx.save_for_backward(flat, w if row_perm is None else w.index_select(0, row_perm))
            if row_perm is not None:
                ctx.row_perm = row_perm
            return x.new_empty((*x.shape[:-1], weight.shape[0]))
        tc = _uses_tensor_core(tuple(weight.shape))
        if activation is None:
            codes, scales = mxfp8_e4m3_quantize(flat, is_sf_swizzled_layout=tc)
        else:
            if not tc:
                raise ValueError("Swizzled prequant requires the tensor-core consumer")
            codes, scales = activation.data.reshape(flat.shape), activation.scale
        raw = _row_scales(scales, *flat.shape) if tc else scales
        decoded_x = (
            (
                codes.float().reshape(flat.shape[0], -1, 32)
                * torch.exp2(raw.float() - 127).unsqueeze(-1)
            )
            .reshape(flat.shape)
            .bfloat16()
            .to(x.dtype)
        )
        packed, weight_scales = quantize_block_fp8(weight, (32, 32), scale_format="e8m0")
        decoded_w = dequantize_block_fp8(packed, weight_scales, (32, 32)).to(x.dtype)
        row_scales = weight_scales.view(torch.uint8).repeat_interleave(32, dim=0)
        if row_perm is not None:
            # vLLM gathers whole rows of the loaded per-row-scale weight
            # (permute_on_load): quantize first, then permute codes and scales.
            packed = packed.view(torch.uint8).index_select(0, row_perm).view(packed.dtype)
            row_scales = row_scales.index_select(0, row_perm)
            decoded_w = decoded_w.index_select(0, row_perm)
        ctx.row_perm = row_perm
        if tc:
            row_scales = swizzle_mxfp8_scale(row_scales, *weight.shape)
            output = torch.ops.vllm.bi_pinned_mxfp8_mm(
                codes, packed.t(), scales, row_scales, x.dtype
            )
        else:
            from vllm.model_executor.kernels.batch_invariant import block32_scaled_mm

            output = block32_scaled_mm(codes, scales, packed, row_scales, x.dtype)
        ctx.save_for_backward(decoded_x, decoded_w)
        return output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        activation, weight = ctx.saved_tensors
        flat_grad = grad.reshape(-1, weight.shape[0])
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dx = tc_mm(flat_grad, weight).reshape(ctx.input_shape).to(activation.dtype)
            dw = tc_mm(flat_grad.T, activation, torch.float32)
            perm = getattr(ctx, "row_perm", None)
            if perm is not None:
                dw = torch.empty_like(dw).index_copy_(0, perm, dw)
        owners = getattr(ctx, "owners", None)
        if owners is None:
            return dx, dw.to(ctx.weight_dtype), None, None
        return dx, accumulate_wgrad(owners, dw, ctx.like), None, None, None


class _BF16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, owners=None):
        if not x.is_cuda or not weight.is_cuda:
            raise RuntimeError("Deployment BF16 linear requires CUDA")
        ctx.owners, ctx.like = owners, weight
        from vllm.model_executor.kernels.linear.cute_dsl.batch_invariant import bf16_fixed_mm

        compute = weight.bfloat16()
        ctx.save_for_backward(x, compute)
        if not x.reshape(-1, x.shape[-1]).shape[0]:
            return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=torch.float32)
        out = bf16_fixed_mm(x.reshape(-1, x.shape[-1]).bfloat16(), compute)
        return out.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        x, compute = ctx.saved_tensors
        flat_grad = grad.reshape(-1, compute.shape[0])
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dx = tc_mm(flat_grad, compute).reshape(x.shape).to(x.dtype)
            dw = tc_mm(flat_grad.T, x.reshape(-1, compute.shape[1]), torch.float32)
        return dx, accumulate_wgrad(ctx.owners, dw, ctx.like), None


def linear(x, weight, *, fp8, activation=None, row_perm=None):
    """BI GEMM ``x @ weight.T``. ``row_perm`` gathers output rows of the quantized weight.

    ``weight``: a tensor, or a tuple of parameters whose rows are stacked (each keeps its
    own FP32 ``main_grad``)."""
    owners = tuple(weight) if isinstance(weight, (tuple, list)) else (weight,)
    weight = owners[0] if len(owners) == 1 else torch.cat(owners)
    if fp8:
        if x.shape[-1] != weight.shape[-1] or any(n % 32 for n in weight.shape):
            raise ValueError("MXFP8 requires matching K and 32-aligned weights")
        return _MXFP8Linear.apply(x, weight, activation, row_perm, owners)
    if activation is not None or row_perm is not None:
        raise ValueError("BF16 linear takes neither MXFP8 prequant nor a row permutation")
    return _BF16Linear.apply(x, weight, owners)
