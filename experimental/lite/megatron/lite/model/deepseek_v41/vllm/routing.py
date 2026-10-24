# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Non-Mega gate selection with an owned fixed-slot score derivative."""
import torch

# vLLM deepseek_v41 common/mm_preprocess IMAGE_SENTINEL_BASE_ID: its router kernels route ids in
# [IMAGE_SENTINEL_LO, IMAGE_SENTINEL_LO + 5) with ffn.gate.bias_vl when the release has vision.
IMAGE_SENTINEL_LO = 129264


class _Route(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, bias, tokens, table, topk, factor, bias_vl=None):
        if not logits.is_cuda:
            raise RuntimeError('Deployment routing requires CUDA')
        from vllm import _custom_ops
        from vllm.model_executor.layers.fused_moe.router.dsv4_topk import (
            can_use_dsv4_topk,
            dsv4_topk,
        )

        ids = torch.empty((logits.shape[0], topk), device=logits.device, dtype=torch.int32)
        weights = torch.empty_like(ids, dtype=torch.float32)
        sentinel = 0 if bias_vl is None else IMAGE_SENTINEL_LO
        if logits.shape[0]:
            if table is None and can_use_dsv4_topk(logits, bias, topk, True, ids.dtype):
                weights, ids = dsv4_topk(
                    logits,
                    bias,
                    ids.dtype,
                    factor,
                    input_ids=tokens,
                    bias_vl=bias_vl,
                    image_sentinel_lo=sentinel,
                )
            else:
                _custom_ops.topk_hash_softplus_sqrt(
                    weights,
                    ids,
                    torch.empty_like(ids),
                    logits.contiguous(),
                    renormalize=True,
                    routed_scaling_factor=factor,
                    e_score_correction_bias=bias,
                    input_tokens=tokens,
                    hash_indices_table=table,
                    bias_vl=bias_vl,
                    image_sentinel_lo=sentinel,
                )
        ctx.save_for_backward(logits, ids)
        ctx.factor = factor
        ctx.mark_non_differentiable(ids)
        return weights, ids

    @staticmethod
    def backward(ctx, grad, ignored_ids):
        logits, ids = ctx.saved_tensors
        with torch.autocast(device_type=logits.device.type, enabled=False):
            selected = logits.float().gather(1, ids.long())
            scores = torch.nn.functional.softplus(selected).sqrt()
            total = scores.sum(-1, keepdim=True)
            ds = (
                ctx.factor
                / total
                * (grad.float() - (grad.float() * scores).sum(-1, keepdim=True) / total)
            )
            dz = ds * torch.sigmoid(selected) / (2 * scores)
            result = torch.zeros_like(logits).scatter_add_(1, ids.long(), dz.to(logits.dtype))
        return result, None, None, None, None, None, None


def route(logits, router, tokens):
    table = router.tid2eid if router.is_hash_layer else None
    bias_vl = getattr(router, "bias_vl", None)
    if (table is not None or bias_vl is not None) and tokens is None:
        raise ValueError("Hash and image-id routing require the actual packed input token IDs")
    return _Route.apply(
        logits,
        router.expert_bias,
        None if table is None and bias_vl is None else tokens.reshape(-1).contiguous(),
        table,
        router.topk,
        router.config.moe_router_topk_scaling_factor or 1.0,
        bias_vl,
    )
