# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadHyperConnectionHead(nn.Module):
    def __init__(self, hidden_size: int, hc_mult: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.eps = eps
        self.hc_fn = nn.Parameter(
            torch.empty(hc_mult, hc_mult * hidden_size, dtype=torch.float32)
        )
        self.hc_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.hc_fn)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x
        shape, dtype = x.shape, x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(xf, self.hc_fn.float()) * rsqrt
        pre = (
            torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float())
            + self.eps
        )
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
        return y.to(dtype)


from inspect import unwrap

import megatron.core.fusions.fused_mhc_kernels as core_mhc_kernels
import megatron.core.transformer.hyper_connection as core_hyper_connection


class RMSNorm(nn.RMSNorm):
    deployment_math = False
    deployment_reduction_width = None

    def forward(self, x):
        if self.deployment_math:
            from megatron.lite.primitive.modules import deployment_math

            if x.is_cuda:
                return deployment_math.qkv_rms_norm(
                    x,
                    self.weight,
                    self.normalized_shape,
                    self.eps,
                    reduction_width=self.deployment_reduction_width,
                )
            return F.rms_norm(
                x,
                self.normalized_shape,
                deployment_math.decoded_bf16_master(self.weight),
                self.eps,
            )
        return super().forward(x)


fused_h_aggregate = core_mhc_kernels.fused_h_aggregate
fused_h_post_bda = core_mhc_kernels.fused_h_post_bda
native_h_aggregate = core_hyper_connection.native_h_aggregate
native_h_post_bda = core_hyper_connection.native_h_post_bda
_sinkhorn_iterations = unwrap(core_hyper_connection._sinkhorn_iterations)


def contract_hc(hidden, pre_mix):
    op = fused_h_aggregate if hidden.is_cuda else unwrap(native_h_aggregate)
    return op(hidden.float(), pre_mix.float()).to(hidden.dtype)


def mix_residual(output, residual, post, comb):
    op = fused_h_post_bda if output.is_cuda else unwrap(native_h_post_bda)
    return op(comb, residual, post, output, None).to(output.dtype)


def expand_hc(tokens, copies):
    from megatron.lite.primitive.parallel import mhc as pipeline_mhc

    hidden = pipeline_mhc.expand_mhc_hidden_for_pipeline(tokens, hc_mult=copies)
    pre = tokens.new_zeros(*tokens.shape[:2], copies, dtype=torch.float32)
    pre[..., 0] = 1
    return hidden, pre


class HCMixes(nn.Module):
    def __init__(self, hidden_size, copies, norm_eps=1e-6, hc_eps=1e-6, iterations=20):
        super().__init__()
        if copies < 1 or iterations < 1:
            raise ValueError("HC copies and Sinkhorn iterations must be positive")
        self.copies = copies
        self.deployment_math = False
        self.broadcast_projection = False
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.iterations = iterations
        size = copies * (copies + 2)
        self.fn = nn.Parameter(
            torch.empty(size, copies * hidden_size, dtype=torch.float32)
        )
        self.base = nn.Parameter(torch.zeros(size, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        nn.init.xavier_uniform_(self.fn)

    def forward(self, hidden):
        if self.deployment_math and hidden.is_cuda:
            from megatron.lite.primitive.modules import deployment_math

            return deployment_math.mhc_coefficients(
                hidden,
                self.fn,
                self.scale,
                self.base,
                copies=self.copies,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
                iterations=self.iterations,
                broadcast=self.broadcast_projection,
            )
        return reference_mixes(
            hidden,
            self.fn,
            self.scale,
            self.base,
            self.copies,
            self.norm_eps,
            self.hc_eps,
            self.iterations,
        )


def reference_mixes(hidden, fn, scale, base, copies, norm_eps, hc_eps, iterations):
    flat = hidden.flatten(2).float()
    mixes = F.linear(flat, fn.float()) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + norm_eps
    )
    sizes = [copies, copies, copies**2]
    pre, post, comb = mixes.split(sizes, dim=-1)
    bp, bpost, bc = base.float().split(sizes)
    pre = torch.sigmoid(pre * scale[0] + bp) + hc_eps
    post = 2 * torch.sigmoid(post * scale[1] + bpost)
    comb = (comb * scale[2] + bc).reshape(*flat.shape[:-1], copies, copies)
    comb = _sinkhorn_iterations(comb, iterations, hc_eps)
    return pre, post, comb
