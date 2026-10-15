# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment-visible routing arithmetic and owned gradients."""

import importlib

import torch

from .autograd import visible_forward


def router_topk(logits, bias, *, topk, scaling_factor):
    """CUDA selection order and FP32 normalization, with an owned score VJP.

    Bias only selects experts. Keep the selected slots fixed for backward, as
    in the reference router; never backpropagate through ranking or bias.
    """
    indices = torch.empty(
        (logits.shape[0], topk), device=logits.device, dtype=torch.int32
    )

    def visible(x):
        from vllm import _custom_ops as ops

        kernels = importlib.import_module(
            'vllm.model_executor.layers.fused_moe.router.dsv4_topk'
        )

        scores, correction = x.contiguous(), bias.contiguous()
        if kernels.can_use_dsv4_topk(scores, correction, topk, True, indices.dtype):
            weights, selected = kernels.dsv4_topk(
                scores, correction, indices.dtype, scaling_factor
            )
            indices.copy_(selected)
            return weights
        weights = torch.empty_like(indices, dtype=torch.float32)
        token_expert_indices = torch.empty_like(indices)
        ops.topk_hash_softplus_sqrt(
            weights,
            indices,
            token_expert_indices,
            scores,
            renormalize=True,
            routed_scaling_factor=scaling_factor,
            e_score_correction_bias=correction,
        )
        return weights

    def reference(x):
        scores = torch.nn.functional.softplus(x.float()).sqrt()
        selected = scores.gather(1, indices.long())
        return selected / selected.sum(-1, keepdim=True) * scaling_factor

    weights = visible_forward(visible, reference, logits)
    return weights, indices.long()
