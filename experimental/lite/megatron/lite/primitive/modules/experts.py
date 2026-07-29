# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE expert compute: SwiGLU fusions, _AllReduceETP, and Experts."""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
import transformer_engine.pytorch as te  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.kernels.swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.lite.primitive.modules.lora import (
    LoraConfig,
    SharedGroupedLinearLoRA,
    normalize_lora_config,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.recompute import CheckpointWithoutOutput
from megatron.lite.primitive.utils import ensure_divisible

__all__ = ["Experts", "_AllReduceETP"]

_DELAYED_WGRAD_STAGING: dict[
    tuple[str, int | None, torch.dtype, int, int, str, int], torch.Tensor
] = {}


def _pack_delayed_wgrad_tensors(
    contexts: list[list[Any]],
    tensor_index: int,
    *,
    linear_index: int,
    kind: str,
) -> tuple[list[torch.Tensor], list[int]] | None:
    per_context = [context[tensor_index] for context in contexts]
    if not per_context or not all(
        isinstance(items, (list, tuple)) for items in per_context
    ):
        return None
    num_experts = len(per_context[0])
    if any(len(items) != num_experts for items in per_context):
        return None
    tensors = [tensor for items in per_context for tensor in items]
    if not tensors or not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return None
    first = tensors[0]
    if first.dim() != 2 or any(
        tensor.dim() != 2
        or tensor.size(1) != first.size(1)
        or tensor.dtype != first.dtype
        or tensor.device != first.device
        for tensor in tensors
    ):
        return None

    rows_per_expert = [
        sum(int(items[expert_idx].size(0)) for items in per_context)
        for expert_idx in range(num_experts)
    ]
    total_rows = sum(rows_per_expert)
    capacity_rows = ((max(total_rows, 1) + 1023) // 1024) * 1024
    stream_id = (
        int(torch.cuda.current_stream(first.device).cuda_stream)
        if first.is_cuda
        else 0
    )
    key = (
        first.device.type,
        first.device.index,
        first.dtype,
        stream_id,
        linear_index,
        kind,
        first.size(1),
    )
    staging = _DELAYED_WGRAD_STAGING.get(key)
    if staging is None or staging.size(0) < capacity_rows:
        staging = first.new_empty((capacity_rows, first.size(1)), requires_grad=False)
        _DELAYED_WGRAD_STAGING[key] = staging

    packed = []
    offset = 0
    with torch.no_grad():
        for expert_idx, rows in enumerate(rows_per_expert):
            expert_view = staging[offset : offset + rows]
            write_offset = 0
            for items in per_context:
                source = items[expert_idx]
                next_offset = write_offset + source.size(0)
                expert_view[write_offset:next_offset].copy_(source)
                write_offset = next_offset
            packed.append(expert_view)
            offset += rows
    return packed, rows_per_expert


@contextmanager
def _expert_nvtx_range(name: str):
    if os.environ.get("MEGATRON_LITE_EP_EXPERT_NVTX") != "1" or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def swiglu_with_probs(
    y: torch.Tensor, probs: torch.Tensor | None, swiglu_limit: float = 0.0
) -> torch.Tensor:
    """SwiGLU with optional expert probability scaling."""
    if swiglu_limit > 0:
        gate, up = y.chunk(2, dim=-1)
        up = torch.clamp(up.float(), min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate.float(), max=swiglu_limit)
        out = torch.nn.functional.silu(gate) * up
        if probs is not None:
            out = out * probs
        return out.to(dtype=y.dtype)
    if probs is not None:
        return weighted_bias_swiglu_impl(y, bias=None, weights=probs)
    return bias_swiglu_impl(y, bias=None)


class _AllReduceETP(torch.autograd.Function):
    """AllReduce with proper autograd: grad(AllReduce) = AllReduce."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class Experts(nn.Module):

    def __init__(
        self,
        config: Any,
        ps: ParallelState,
        *,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        delay_wgrad_compute: bool = False,
        lora_config: LoraConfig | dict | None = None,
    ):
        super().__init__()
        self.num_local_experts = ensure_divisible(config.num_experts, ps.ep_size)
        self.fp8 = fp8
        self.moe_act_recompute = moe_act_recompute
        self.etp_group = ps.etp_group if ps.etp_size > 1 else None
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 0.0) or 0.0)

        self.fc1 = te.GroupedLinear(
            self.num_local_experts,
            config.hidden_size,
            config.moe_intermediate_size * 2 // ps.etp_size,
            bias=False,
            params_dtype=torch.bfloat16,
            delay_wgrad_compute=delay_wgrad_compute,
            fuse_wgrad_accumulation=delay_wgrad_compute,
        )
        self.fc2 = te.GroupedLinear(
            self.num_local_experts,
            config.moe_intermediate_size // ps.etp_size,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            delay_wgrad_compute=delay_wgrad_compute,
            fuse_wgrad_accumulation=delay_wgrad_compute,
        )
        lora = normalize_lora_config(lora_config)
        self.fc1_lora: SharedGroupedLinearLoRA | None = None
        self.fc2_lora: SharedGroupedLinearLoRA | None = None
        if lora.enabled and lora.targets_module("linear_fc1"):
            self.fc1_lora = SharedGroupedLinearLoRA(
                self.num_local_experts,
                config.hidden_size,
                config.moe_intermediate_size * 2 // ps.etp_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
            )
        if lora.enabled and lora.targets_module("linear_fc2"):
            self.fc2_lora = SharedGroupedLinearLoRA(
                self.num_local_experts,
                config.moe_intermediate_size // ps.etp_size,
                config.hidden_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
            )
        if ps.tp_size > 1 and ps.ep_size == 1 and ps.etp_size == 1:
            tp_group = ps.tp_group
            for module in (self.fc1, self.fc2, self.fc1_lora, self.fc2_lora):
                if module is None:
                    continue
                for param in module.parameters():

                    def _ar(grad, g=tp_group):
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=g)
                        return grad

                    param.register_hook(_ar)

    def flush_delayed_weight_grads(self, *, num_contexts: int) -> None:
        """Fuse queued chunk wgrads into one grouped GEMM over reusable staging."""
        for linear_index, linear in enumerate((self.fc1, self.fc2)):
            store = linear.wgrad_store
            if not store.delay_wgrad_compute():
                raise RuntimeError("Expert delayed weight gradients are not enabled.")
            if linear.use_bias:
                raise RuntimeError("Chunked EP expert grouped linears must not use bias.")
            if num_contexts < 1:
                raise RuntimeError("Expert delayed wgrad requires at least one context.")
            if store.context is None:
                raise RuntimeError("Expert delayed weight-gradient queue is unavailable.")

            if num_contexts == 1:
                if store.context.empty():
                    raise RuntimeError("Expert delayed weight-gradient queue is empty.")
                (_, _grad_biases, _), tensors = store.pop()
                weight_grads = tensors[2]
            else:
                queued = []
                for _ in range(num_contexts):
                    if store.context.empty():
                        raise RuntimeError(
                            "Expert delayed weight-gradient queue is empty."
                        )
                    queued.append(store.context.get())
                contexts = [item[0] for item in queued]
                funcs = [item[1] for item in queued]
                packed_inputs = _pack_delayed_wgrad_tensors(
                    contexts, 0, linear_index=linear_index, kind="input"
                )
                packed_grads = _pack_delayed_wgrad_tensors(
                    contexts, 1, linear_index=linear_index, kind="grad_output"
                )
                can_fuse = (
                    packed_inputs is not None
                    and packed_grads is not None
                    and isinstance(funcs[0], functools.partial)
                    and all(
                        isinstance(func, functools.partial)
                        and func.func is funcs[0].func
                        and func.args == funcs[0].args
                        for func in funcs
                    )
                )
                if can_fuse:
                    input_tensors, input_splits = packed_inputs
                    grad_tensors, grad_splits = packed_grads
                    if input_splits != grad_splits:
                        raise RuntimeError(
                            "Expert delayed wgrad input and output splits differ."
                        )
                    weight_grads = contexts[0][2]
                    if any(
                        any(
                            grad.data_ptr() != weight_grads[idx].data_ptr()
                            for idx, grad in enumerate(context[2])
                        )
                        for context in contexts[1:]
                    ):
                        raise RuntimeError(
                            "Expert delayed wgrad contexts do not share main_grad."
                        )
                    keywords = dict(funcs[0].keywords or {})
                    keywords["m_splits"] = input_splits
                    fused_func = functools.partial(
                        funcs[0].func, *funcs[0].args, **keywords
                    )
                    fused_func(input_tensors, grad_tensors, weight_grads)
                else:
                    weight_grads = contexts[-1][2]
                    for tensors, func in queued:
                        func(*tensors)

            for idx, grad in enumerate(weight_grads):
                param = getattr(linear, f"weight{idx}")
                main_grad = getattr(param, "main_grad", None)
                if main_grad is None or grad.data_ptr() != main_grad.data_ptr():
                    raise RuntimeError(
                        "Expert delayed wgrad did not reuse its DistOpt main_grad buffer."
                    )
            if store.context is not None and not store.context.empty():
                raise RuntimeError("Expert delayed weight-gradient queue was not drained.")

    def forward(
        self,
        x: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor | None = None,
        tokens_per_expert_list: list[int] | None = None,
    ) -> torch.Tensor:
        m_splits = (
            tokens_per_expert.tolist()
            if tokens_per_expert_list is None
            else list(tokens_per_expert_list)
        )
        pad_mask = None
        if self.fp8:
            x, permuted_probs, m_splits, pad_mask = self._fp8_pad(x, permuted_probs, m_splits)

        etp_real_len = x.shape[0]
        if self.etp_group is not None:
            max_len = torch.tensor([etp_real_len], device=x.device, dtype=torch.int64)
            dist.all_reduce(max_len, op=dist.ReduceOp.MAX, group=self.etp_group)
            max_len = int(max_len.item())
            if etp_real_len < max_len:
                x = torch.cat(
                    [
                        x,
                        torch.zeros(
                            max_len - etp_real_len, x.shape[1], dtype=x.dtype, device=x.device
                        ),
                    ],
                    dim=0,
                )
                if permuted_probs is not None:
                    permuted_probs = torch.cat(
                        [
                            permuted_probs,
                            torch.zeros(
                                max_len - etp_real_len, dtype=permuted_probs.dtype, device=x.device
                            ),
                        ],
                        dim=0,
                    )
                m_splits = list(m_splits)
                m_splits[-1] += max_len - etp_real_len

        probs = permuted_probs.unsqueeze(-1) if permuted_probs is not None else None
        with _expert_nvtx_range("ep_experts.forward"):
            if self.moe_act_recompute and probs is not None:
                act_ckpt = CheckpointWithoutOutput(preserve_rng_state=True)
                fc1_out = self.fc1(x, m_splits)
                if self.fc1_lora is not None:
                    fc1_out = fc1_out + self.fc1_lora(x, m_splits)
                h = act_ckpt.checkpoint(swiglu_with_probs, fc1_out, probs, self.swiglu_limit)
                out = self.fc2(h, m_splits)
                if self.fc2_lora is not None:
                    out = out + self.fc2_lora(h, m_splits)
                act_ckpt.discard_output_and_register_recompute(out)
            else:
                fc1_out = self.fc1(x, m_splits)
                if self.fc1_lora is not None:
                    fc1_out = fc1_out + self.fc1_lora(x, m_splits)
                h = swiglu_with_probs(fc1_out, probs, self.swiglu_limit)
                out = self.fc2(h, m_splits)
                if self.fc2_lora is not None:
                    out = out + self.fc2_lora(h, m_splits)

        if self.etp_group is not None:
            out = _AllReduceETP.apply(out, self.etp_group)
            out = out[:etp_real_len]

        if pad_mask is not None:
            out = out[pad_mask]
        return out

    @staticmethod
    def _fp8_pad(x, permuted_probs, m_splits):
        padded = [(s + 15) // 16 * 16 for s in m_splits]
        if padded == m_splits:
            return x, permuted_probs, m_splits, None
        device, dtype = x.device, x.dtype
        total_padded = sum(padded)
        x_pad = torch.zeros(total_padded, x.size(1), device=device, dtype=dtype)
        mask = torch.zeros(total_padded, dtype=torch.bool, device=device)
        probs_pad = None
        if permuted_probs is not None:
            probs_pad = torch.zeros(total_padded, device=device, dtype=permuted_probs.dtype)
        src_off, dst_off = 0, 0
        for real, pad in zip(m_splits, padded, strict=True):
            x_pad[dst_off : dst_off + real] = x[src_off : src_off + real]
            mask[dst_off : dst_off + real] = True
            if probs_pad is not None:
                probs_pad[dst_off : dst_off + real] = permuted_probs[src_off : src_off + real]
            src_off += real
            dst_off += pad
        return x_pad, probs_pad, padded, mask
