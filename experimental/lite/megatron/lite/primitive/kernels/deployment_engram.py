# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 Engram gate/residual arithmetic, matching the deployment kernel.

Kernel copied unchanged from vLLM 168a040, common/engram.py, blob
c415bbaa6e42626106e78cec32d70206bcdc88d8. The caller owns live operands and
output storage. No model/rollout objects or cached activation are consumed.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_kv_tokens"])
def _fused_engram_post_wkv_kernel(
    hidden_states,
    kv,
    q_weight,
    k_weight,
    token_mask,
    output,
    num_kv_tokens,
    hidden_stride_t,
    hidden_stride_h,
    hidden_stride_d,
    kv_stride_t,
    kv_stride_d,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_d,
    mask_stride,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    eps,
    clamp_value,
    DIM: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    program_idx = tl.program_id(0)
    token_idx = program_idx // HC_MULT
    hc_idx = program_idx % HC_MULT
    token_idx = token_idx.to(tl.int64)
    source_idx = token_idx
    source_valid = source_idx < num_kv_tokens

    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_valid = dim_offsets < DIM
    hidden = tl.load(
        hidden_states
        + token_idx * hidden_stride_t
        + hc_idx * hidden_stride_h
        + dim_offsets * hidden_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        kv + source_idx * kv_stride_t + (hc_idx * DIM + dim_offsets) * kv_stride_d,
        mask=source_valid & dim_valid,
        other=0.0,
    ).to(tl.float32)
    q = tl.load(
        q_weight + hc_idx * q_stride_h + dim_offsets * q_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_weight + hc_idx * k_stride_h + dim_offsets * k_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)

    hidden_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / DIM + eps)
    key_rms = tl.rsqrt(tl.sum(key * key, axis=0) / DIM + eps)
    dot = tl.sum(hidden * q * k * key, axis=0)
    dot *= hidden_rms * key_rms * tl.rsqrt(DIM * 1.0)
    gate_input = tl.sqrt(tl.maximum(tl.abs(dot), clamp_value))
    gate_input = tl.where(dot < 0.0, -gate_input, gate_input)
    gate = tl.sigmoid(gate_input)
    if HAS_MASK:
        active = tl.load(
            token_mask + source_idx * mask_stride, mask=source_valid, other=0
        )
        gate = tl.where(active, gate, 0.0)

    value = tl.load(
        kv + source_idx * kv_stride_t + (HC_MULT * DIM + dim_offsets) * kv_stride_d,
        mask=source_valid & dim_valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output
        + token_idx * output_stride_t
        + hc_idx * output_stride_h
        + dim_offsets * output_stride_d,
        hidden + gate * value,
        mask=dim_valid,
    )


def post_wkv(hidden, kv, query, key, *, eps, token_mask=None):
    """[T,C,D] residual + sigmoid(signed-sqrt(normalized dot))*value."""
    if hidden.ndim != 3:
        raise ValueError('Engram residual must be [T,C,D]')
    tokens, copies, dim = hidden.shape
    if kv.shape != (tokens, (copies + 1) * dim):
        raise ValueError('Engram KV shape differs from the residual')
    if query.shape != (copies, dim) or key.shape != query.shape:
        raise ValueError('Engram normalization weights must be [C,D]')
    operands = (hidden, kv, query, key)
    if any(x.device != hidden.device or x.dtype != torch.bfloat16 for x in operands):
        raise ValueError('Engram deployment operands must share BF16 CUDA storage')
    if not hidden.is_cuda:
        raise NotImplementedError('Engram deployment requires CUDA')
    if token_mask is not None and (
        token_mask.shape != (tokens,)
        or token_mask.device != hidden.device
        or token_mask.dtype != torch.bool
    ):
        raise ValueError('Engram token mask must be a CUDA bool token vector')
    output = torch.empty_like(hidden)
    if tokens == 0:
        return output
    block = triton.next_power_of_2(dim)
    _fused_engram_post_wkv_kernel[(tokens * copies,)](
        hidden,
        kv,
        query,
        key,
        token_mask if token_mask is not None else hidden,
        output,
        tokens,
        *hidden.stride(),
        *kv.stride(),
        *query.stride(),
        *key.stride(),
        token_mask.stride(0) if token_mask is not None else 0,
        *output.stride(),
        eps,
        1e-6,
        DIM=dim,
        HC_MULT=copies,
        BLOCK_SIZE=block,
        HAS_MASK=token_mask is not None,
        num_warps=8 if block >= 2048 else 4,
    )
    return output
