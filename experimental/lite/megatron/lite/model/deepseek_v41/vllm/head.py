# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment head log-probs a block of rows at a time, GEMM included."""

import torch

from .linear import tc_mm


def log_softmax(logits):
    if not logits.is_cuda:
        return torch.log_softmax(logits.float(), -1)  # CPU derivative reference only.
    from vllm.model_executor.kernels.batch_invariant import native_log_softmax

    return native_log_softmax(logits.float(), -1)


# Rows per block: a 129280-wide FP32 block of 1024 rows is 0.5 GiB.
HEAD_ROWS = 1024
# Vocabulary rows of the weight rounded at a time in backward (8192 x 5120
# FP32 is 160 MiB), so backward holds no full-size copy of the weight.
HEAD_VOCAB_ROWS = 8192


def _compute_weight(hidden, weight):
    # CUDA: vLLM's batch-invariant BF16 GEMM (as bf16_fp32_linear). CPU: the
    # FP32 linear the model's non-CUDA head uses.
    return weight.bfloat16() if hidden.is_cuda else weight.float()


def _compute_weight_rows(hidden, weight, start, stop):
    """``_compute_weight`` for weight rows ``[start, stop)``."""
    rows = weight[start:stop]
    return rows.bfloat16() if hidden.is_cuda else rows.float()


def _project(hidden, compute):
    """Deployment logits: vLLM's default head_dtype (= model dtype) rounds the
    FP32-accumulated BI GEMM to BF16 (``linear_batch_invariant``)."""
    if not hidden.is_cuda:
        return torch.nn.functional.linear(hidden.float(), compute).bfloat16()
    from vllm.model_executor.kernels.linear.cute_dsl.batch_invariant import bf16_fixed_mm

    return bf16_fixed_mm(hidden.bfloat16(), compute).to(torch.bfloat16)


def head_logits(hidden, weight):
    """The whole BF16 deployment logits, same arithmetic as one head block."""
    if hidden.is_cuda:
        from .linear import linear

        return _RoundBF16.apply(linear(hidden, weight, fp8=False))
    return torch.nn.functional.linear(hidden.float(), weight.float()).bfloat16()


class _RoundBF16(torch.autograd.Function):
    """BF16 logits of the deployment head; identity (STE) derivative."""

    @staticmethod
    def forward(ctx, x):
        return x.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad):
        return grad.float()


def _probabilities(logits, labels, temperature, entropy):
    # vLLM raw logprobs ignore temperature; T == 1 keeps the bytes (x / 1.0 == x).
    logits = logits.float()
    distribution = log_softmax(logits if float(temperature) == 1.0 else logits / temperature)
    picked = distribution.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    spread = -(distribution.exp() * distribution).sum(-1) if entropy else None
    return picked, spread


class _HeadLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, temperature, entropy, rows):
        compute = _compute_weight(hidden, weight)
        picked = hidden.new_empty(hidden.shape[0], dtype=torch.float32)
        spread = picked.new_empty(picked.shape) if entropy else None
        for start in range(0, hidden.shape[0], rows):
            stop = start + rows
            block = _probabilities(
                _project(hidden[start:stop], compute), labels[start:stop], temperature, entropy
            )
            picked[start:stop] = block[0]
            if entropy:
                spread[start:stop] = block[1]
        # Save the weight itself, not its rounded copy: backward rounds it a
        # slice of vocabulary rows at a time.
        ctx.save_for_backward(hidden, weight, labels)
        ctx.options = (temperature, entropy, rows)
        return picked, spread

    @staticmethod
    def backward(ctx, grad_picked, grad_spread):
        hidden, weight, labels = ctx.saved_tensors
        temperature, entropy, rows = ctx.options
        need_hidden, need_weight = ctx.needs_input_grad[:2]
        dhidden = torch.zeros_like(hidden) if need_hidden else None
        # CUDA: BF16 operands on tensor cores with FP32 accumulation (backward need not be
        # batch-invariant); the weight gradient goes straight into an FP32 main_grad when
        # DDP owns one, else into the only weight-sized FP32 buffer of backward.
        tc = hidden.is_cuda
        main = getattr(weight, "main_grad", None)
        fused = tc and need_weight and main is not None and main.dtype == torch.float32
        dweight = (
            main
            if fused
            else torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
            if need_weight
            else None
        )
        if not (need_hidden or need_weight) or not hidden.shape[0]:
            return dhidden, _weight_grad(weight, dweight, fused), None, None, None, None
        vocab = weight.shape[0]
        slices = [
            (start, min(start + HEAD_VOCAB_ROWS, vocab))
            for start in range(0, vocab, HEAD_VOCAB_ROWS)
        ]
        # Backward recomputes the logits with torch FP32 GEMMs, not vLLM's
        # bf16_fixed_mm: called from the autograd engine thread it races with
        # the surrounding torch kernels (nondeterministic, at times non-finite
        # gradients; exact under CUDA_LAUNCH_BLOCKING=1). Backward need not be
        # bitwise; forward keeps the deployment GEMM.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for start in range(0, hidden.shape[0], rows):
                stop = start + rows
                block = hidden[start:stop].bfloat16() if tc else hidden[start:stop].float()
                logits = block.new_empty((block.shape[0], vocab), dtype=torch.float32)
                with torch.no_grad():
                    for v0, v1 in slices:
                        w = _compute_weight_rows(hidden, weight, v0, v1)
                        logits[:, v0:v1] = tc_mm(block, w.T) if tc else (block @ w.T).bfloat16()
                logits.requires_grad_()
                with torch.enable_grad():
                    outputs = _probabilities(logits, labels[start:stop], temperature, entropy)
                pairs = [
                    (output, grad[start:stop])
                    for output, grad in zip(outputs, (grad_picked, grad_spread))
                    if output is not None and grad is not None
                ]
                if not pairs:
                    continue
                (grad,) = torch.autograd.grad([o for o, _ in pairs], logits, [g for _, g in pairs])
                del logits, outputs, pairs
                # Same VJP as _BF16Linear, one block of rows and one slice of
                # vocabulary at a time, accumulating in place.
                dblock = block.new_zeros(block.shape, dtype=torch.float32) if need_hidden else None
                for v0, v1 in slices:
                    w = _compute_weight_rows(hidden, weight, v0, v1)
                    if not tc:
                        if need_hidden:
                            dblock.addmm_(grad[:, v0:v1], w)
                        if need_weight:
                            dweight[v0:v1].addmm_(grad[:, v0:v1].T, block)
                        continue
                    if need_hidden:
                        dblock += tc_mm(grad[:, v0:v1], w, torch.float32)
                    if need_weight:
                        dweight[v0:v1] += tc_mm(grad[:, v0:v1].T, block, torch.float32)
                if need_hidden:
                    dhidden[start:stop] = dblock.to(hidden.dtype)
        return dhidden, _weight_grad(weight, dweight, fused), None, None, None, None


def _weight_grad(weight, dweight, fused):
    if dweight is None:
        return None
    if fused:
        from megatron.lite.primitive.parallel.linear import _dummy_wgrad

        weight.grad_added_to_main_grad = True
        return _dummy_wgrad(weight)
    return dweight if weight.dtype == torch.float32 else dweight.to(weight.dtype)


def head_log_probs(hidden, weight, labels, temperature, *, entropy, rows=None):
    """Target log-probs and (optionally) entropy of ``hidden @ weight.T``.

    ``hidden`` is ``[rows, hidden]``, ``labels`` ``[rows]``. Forward is bitwise
    ``log_softmax(head_logits(hidden, weight).float() / temperature)`` gathered at the
    labels, because the GEMM and log-softmax are row-invariant (vLLM's
    batch-invariant kernels on CUDA). Neither the logits nor their gradient
    exist for more than ``rows`` rows at once: backward recomputes each block
    and accumulates the weight gradient across blocks.
    """
    if not hidden.is_cuda:
        raise RuntimeError("Deployment head requires CUDA; CPU is a private derivative reference")
    rows = HEAD_ROWS if rows is None else rows
    if rows <= 0:
        raise ValueError('Head block rows must be positive')
    return _HeadLogProbs.apply(hidden, weight, labels, temperature, entropy, rows)
