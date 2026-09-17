# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-rank HF reference for native Nemotron tests, never a training backend."""

from contextvars import ContextVar
from functools import wraps

import torch

_sequence_boundaries = ContextVar("nemotron_sequence_boundaries", default=None)


def _with_sequence_boundaries(forward):
    @wraps(forward)
    def wrapped(backbone, *args, **kwargs):
        positions = kwargs.get("position_ids")
        cu = kwargs.pop("cu_seqlens", None)
        kwargs.pop("cu_seqlens_cpu", None)
        boundaries = None
        if cu is not None:
            boundaries = tuple(cu.tolist())
        elif positions is not None:
            if positions.ndim != 2:
                raise ValueError("Expected two-dimensional position_ids")
            if positions.shape[0] == 1:
                values = positions[0].tolist()
                starts = [i for i, value in enumerate(values) if value == 0]
                if len(starts) > 1:
                    boundaries = (*starts, len(values))
        if boundaries is not None:
            ids = kwargs.get("input_ids", args[0] if args else None)
            inputs = ids if ids is not None else kwargs["inputs_embeds"]
            if (
                inputs.shape[0] != 1
                or boundaries[0] != 0
                or boundaries[-1] != inputs.shape[1]
                or any(a >= b for a, b in zip(boundaries, boundaries[1:], strict=False))
            ):
                raise ValueError("Invalid packed sequence boundaries")
            expected = torch.cat(
                [
                    torch.arange(b - a, device=inputs.device)
                    for a, b in zip(boundaries, boundaries[1:], strict=False)
                ]
            )[None]
            if positions is not None and not torch.equal(positions, expected):
                raise ValueError("Packed positions must restart at zero per sequence")
            kwargs["position_ids"] = expected
        token = _sequence_boundaries.set(boundaries)
        try:
            return forward(backbone, *args, **kwargs)
        finally:
            _sequence_boundaries.reset(token)

    return wrapped


def install_training_alignment(model):
    """Install the shared-norm BF16, cache-free trainer path exactly once.

    Initialize vLLM batch invariance before loading the HF model. Inference
    must use nemotron_shared_norms=True with the same weights and BF16 head.
    The underlying HF function replacements are process-wide.
    """
    from vllm import envs

    if not envs.VLLM_BATCH_INVARIANT:
        raise ValueError("Nemotron alignment requires VLLM_BATCH_INVARIANT=1")
    if model.config.model_type != "nemotron_h":
        raise ValueError("Expected a Nemotron-H model")
    if model.config._attn_implementation != "eager":
        raise ValueError("Aligned training requires the eager HF attention entry")
    if model.lm_head.weight.dtype != torch.bfloat16:
        raise ValueError("Aligned training currently requires a BF16 head")
    if model.is_gradient_checkpointing:
        raise ValueError("Aligned training does not yet support checkpointing")
    if getattr(model, "_nemotron_training_aligned", False):
        return
    install_transformers_mamba_forward(model)
    install_linear_forward(model)
    install_transformers_conv_forward()
    install_transformers_gated_rms_forward(model)
    install_transformers_residual_forward(model)
    install_transformers_moe_forward(model)
    install_transformers_attention_forward()
    install_transformers_lm_forward(model)
    model._nemotron_training_aligned = True


class _VisibleForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, native, *inputs):
        ctx.native = native
        ctx.save_for_backward(*inputs)
        return visible(*inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        with torch.enable_grad():
            inputs = tuple(
                x.detach().requires_grad_(required)
                for x, required in zip(
                    ctx.saved_tensors, ctx.needs_input_grad[2:], strict=False
                )
            )
            active = tuple(x for x in inputs if x.requires_grad)
            output = ctx.native(*inputs)
            grads = iter(torch.autograd.grad(output, active, grad_outputs))
        return None, None, *(next(grads) if x.requires_grad else None for x in inputs)


def visible_forward(visible, native, *inputs):
    """Run inference arithmetic in forward and the native mathematical VJP."""
    if not torch.is_grad_enabled() or not any(x.requires_grad for x in inputs):
        return visible(*inputs)
    return _VisibleForward.apply(visible, native, *inputs)


def token_logprobs(logits, token_ids):
    """Score with the inference kernel and input dtype; retain native logsoftmax VJP."""
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    def native(logits, token_ids):
        return logits.float().log_softmax(-1).gather(-1, token_ids.long())

    return visible_forward(compute_token_logprobs, native, logits, token_ids)


def install_transformers_lm_forward(model):
    """Preserve HF's causal-LM contract, returning the head's actual dtype."""
    from types import MethodType

    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.utils import can_return_tuple

    @can_return_tuple
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(
        module,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        logits_to_keep=0,
        shift_labels=None,
        temperature=1.0,
        **kwargs,
    ):
        if shift_labels is not None and temperature != 1.0:
            raise ValueError("Aligned logprobs currently require temperature=1")
        output = module.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = module.lm_head(output.last_hidden_state[:, indices, :])
        loss = None
        if labels is not None:
            loss = module.loss_function(
                logits.float(), labels, module.vocab_size, **kwargs
            )
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
        if shift_labels is not None:
            if shift_labels.shape != logits.shape[:-1]:
                raise ValueError("shift_labels must match the returned token positions")
            result["log_probs"] = token_logprobs(
                logits.reshape(-1, logits.shape[-1]), shift_labels.reshape(-1, 1)
            ).reshape_as(shift_labels)
        return result

    model.forward = MethodType(forward, model)


def install_transformers_rms_forward(model):
    """Match inference's fused rounding while retaining native RMS gradients."""
    from types import MethodType

    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHRMSNorm

    from vllm.model_executor.models.nemotron_h_alignment import rms_forward

    def forward(module, x, residual=None):
        if residual is not None:
            return residual_rms(x, residual, module.weight, module.variance_epsilon)

        def native(x, weight):
            y = x.float()
            y = y * torch.rsqrt(
                y.square().mean(-1, keepdim=True) + module.variance_epsilon
            )
            return weight * y.to(x.dtype)

        def visible(x, weight):
            return rms_forward(x, weight, module.variance_epsilon)

        return visible_forward(visible, native, x, module.weight)

    for module in model.modules():
        if isinstance(module, NemotronHRMSNorm):
            module.forward = MethodType(forward, module)


def install_transformers_gated_rms_forward(model):
    """Share the inference gated RMS kernel; keep the native training VJP."""
    from types import MethodType

    from transformers.models.zamba2.modeling_zamba2 import Zamba2RMSNormGated

    from vllm.model_executor.models.nemotron_h_alignment import gated_forward

    def make_forward(group_size, eps):
        def native(x, gate, weight):
            y = x.float() * torch.nn.functional.silu(gate.float())
            grouped = y.unflatten(-1, (-1, group_size))
            grouped = grouped * torch.rsqrt(
                grouped.square().mean(-1, keepdim=True) + eps
            )
            return weight * grouped.flatten(-2).to(x.dtype)

        def visible(x, gate, weight):
            return gated_forward(x, gate, weight, group_size, eps)

        def forward(module, x, gate=None):
            if gate is None:
                raise ValueError("Aligned Mamba gated RMS requires a gate")
            shape = x.shape
            x = x.reshape(-1, shape[-1])
            gate = gate.reshape_as(x)
            out = visible_forward(visible, native, x, gate, module.weight)
            return out.reshape(shape)

        return forward

    for module in model.modules():
        if isinstance(module, Zamba2RMSNormGated):
            module.forward = MethodType(
                make_forward(module.group_size, module.variance_epsilon), module
            )


def _native_residual_rms(eps):
    def native(x, residual, weight):
        y = x.float() + residual.float()
        residual_out = y.to(weight.dtype)
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
        return y.to(weight.dtype) * weight, residual_out

    return native


def residual_rms(x, residual, weight, eps):
    """Share residual normalization and retain the native mathematical VJP."""
    from vllm.model_executor.models.nemotron_h_alignment import rms_forward

    def visible(x, residual, weight):
        return rms_forward(x, weight, eps, residual)

    return visible_forward(visible, _native_residual_rms(eps), x, residual, weight)


def install_transformers_residual_forward(model):
    """Carry separate residuals through the cache-free HF training backbone."""
    if getattr(model.model, "_nemotron_residual_aligned", False):
        return
    from types import MethodType

    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.nemotron_h import modeling_nemotron_h as modeling

    install_transformers_rms_forward(model)

    def block_forward(layer, hidden, residual, masks, position_ids, **kwargs):
        if residual is None:
            residual = hidden
            hidden = layer.norm(hidden)
        else:
            hidden, residual = layer.norm(hidden, residual)
        if layer.block_type == "linear_attention":
            hidden = layer.mixer(
                hidden,
                cache_params=None,
                attention_mask=masks.get(layer.block_type),
            )
        elif layer.block_type == "full_attention":
            hidden, _ = layer.mixer(
                hidden_states=hidden,
                attention_mask=masks.get(layer.block_type),
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                **kwargs,
            )
        else:
            hidden = layer.mixer(hidden)
        return hidden, residual

    for layer in model.model.layers:
        layer.forward = MethodType(block_forward, layer)

    def forward(
        backbone,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
        **kwargs,
    ):
        if backbone.config._attn_implementation != "eager":
            raise ValueError("Aligned training requires the eager HF attention entry")
        if backbone.is_gradient_checkpointing:
            raise ValueError("Aligned training does not yet support checkpointing")
        if past_key_values is not None or use_cache:
            raise ValueError("Aligned training backbone is cache-free")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds")
        length = (input_ids if input_ids is not None else inputs_embeds).shape[1]
        if isinstance(attention_mask, dict):
            recurrent_mask = attention_mask.get("linear_attention")
            causal_mask = attention_mask.get("full_attention")
            if causal_mask is not None:
                _validate_causal_mask(causal_mask, length)
        else:
            recurrent_mask = attention_mask
        if recurrent_mask is not None and (
            recurrent_mask.ndim != 2 or not torch.all(recurrent_mask == 1)
        ):
            raise ValueError("Aligned training currently requires unpadded sequences")
        hidden = (
            backbone.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        )
        if position_ids is None:
            position_ids = torch.arange(hidden.shape[1], device=hidden.device)[None]
        masks = attention_mask
        if _sequence_boundaries.get() is not None:
            if attention_mask is not None:
                raise ValueError("Packed training uses sequence boundaries, not a mask")
            masks = {}
        elif not isinstance(masks, dict):
            arguments = dict(
                config=backbone.config,
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            masks = {
                "full_attention": modeling.create_causal_mask(**arguments),
                "linear_attention": modeling.create_recurrent_attention_mask(
                    **arguments
                ),
            }
        residual = None
        for layer in backbone.layers:
            hidden, residual = layer(hidden, residual, masks, position_ids, **kwargs)
        hidden, _ = backbone.norm_f(hidden, residual)
        return BaseModelOutputWithPast(last_hidden_state=hidden)

    model.model.forward = MethodType(_with_sequence_boundaries(forward), model.model)
    model.model._nemotron_residual_aligned = True


def mamba_scan(x, dt, A, B, C, D, dt_bias, *, chunk_size):
    """Canonical SSD forward for an unpadded training batch, without a cache."""
    from vllm.model_executor.layers.mamba.ops.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )

    batch, length, heads, head_dim = x.shape
    if length == 0:
        raise ValueError("Mamba training sequences must be nonempty")
    chunk_offsets = []
    last_chunks = []
    seq_ids = []
    boundaries = _sequence_boundaries.get()
    if boundaries is None:
        boundaries = tuple(range(0, (batch + 1) * length, length))
    for row, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
        starts = list(range(start, end, chunk_size))
        chunk_offsets.extend(starts)
        seq_ids.extend([row] * len(starts))
        last_chunks.append(len(chunk_offsets) - 1)
    chunk_offsets.append(batch * length)

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=x.device)

    output = torch.empty(
        (batch * length, heads, head_dim), dtype=x.dtype, device=x.device
    )
    mamba_chunk_scan_combined_varlen(
        x.reshape(batch * length, heads, head_dim),
        dt.reshape(batch * length, heads),
        A,
        B.flatten(0, 1),
        C.flatten(0, 1),
        chunk_size=chunk_size,
        cu_seqlens=i32(boundaries),
        cu_chunk_seqlens=i32(chunk_offsets),
        last_chunk_indices=i32(last_chunks),
        seq_idx=i32(seq_ids),
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=output,
        state_dtype=torch.float32,
    )
    return output.view_as(x)


def install_transformers_mamba_forward(model):
    """Use canonical SSD for cache-free training; retain native SSD backward.

    Returns the original function so the caller can restore it. Other model
    operations remain unchanged; this is an operator integration, not a claim
    that all trainer logprobs are aligned.
    """
    from transformers.models.nemotron_h import modeling_nemotron_h as modeling

    native_scan = getattr(
        modeling.mamba2_chunk_scan, "__wrapped__", modeling.mamba2_chunk_scan
    )
    # Nemotron Automodel uses time_step_limit for forward clipping;
    # time_step_min specifies initialization, not a forward clamp.
    for module in model.modules():
        if isinstance(module, modeling.NemotronHMamba2Mixer):
            module.time_step_limit = tuple(model.config.time_step_limit)

    @wraps(native_scan)
    def aligned_scan(x, dt, A, B, C, **kwargs):
        if (
            kwargs.get("return_final_states", False)
            or kwargs.get("initial_states") is not None
            or kwargs.get("z") is not None
            or not kwargs.get("dt_softplus", False)
            or tuple(kwargs.get("dt_limit", (0.0, float("inf")))) != (0.0, float("inf"))
        ):
            raise ValueError("Aligned SSD currently requires cache-free training")
        D, bias = kwargs["D"], kwargs["dt_bias"]
        boundaries = _sequence_boundaries.get()

        def visible(x, dt, A, B, C, D, bias):
            return mamba_scan(x, dt, A, B, C, D, bias, chunk_size=kwargs["chunk_size"])

        def native(x, dt, A, B, C, D, bias):
            if boundaries is not None:
                return torch.cat(
                    [
                        native_scan(
                            x[:, a:b],
                            dt[:, a:b],
                            A,
                            B[:, a:b],
                            C[:, a:b],
                            **{**kwargs, "D": D, "dt_bias": bias},
                        )
                        for a, b in zip(boundaries, boundaries[1:], strict=False)
                    ],
                    dim=1,
                )
            return native_scan(x, dt, A, B, C, **{**kwargs, "D": D, "dt_bias": bias})

        output = visible_forward(visible, native, x, dt, A, B, C, D, bias)
        return output

    modeling.mamba2_chunk_scan = aligned_scan
    return native_scan


def install_linear_forward(model):
    """Replace dense linear forward with the same BI GEMM used by vLLM."""
    from types import MethodType

    from vllm.model_executor.determinism.batch_invariant import linear_batch_invariant

    def forward(module, x):
        inputs = (x, module.weight)
        if module.bias is not None:
            inputs += (module.bias,)
        return visible_forward(
            linear_batch_invariant, torch.nn.functional.linear, *inputs
        )

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = MethodType(forward, module)


def router_linear(x, weight):
    """Use the captured router's BF16 GEMM with FP32 output, not FP32 operands."""

    def visible(x, weight):
        return torch.mm(x, weight.T, out_dtype=torch.float32)

    def native(x, weight):
        return torch.nn.functional.linear(x.float(), weight.float())

    return visible_forward(visible, native, x, weight)


class _OriginalForward(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.original = module.forward

    def forward(self, *args):
        return self.original(*args)


def install_transformers_moe_forward(model):
    """Reuse vLLM routing/expert kernels and HF's differentiable expert math."""
    from types import MethodType

    from transformers.models.nemotron_h import modeling_nemotron_h as modeling

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        grouped_topk,
    )

    def route(module, x):
        x = x.reshape(-1, module.hidden_dim)
        logits = router_linear(x, module.weight)
        with torch.no_grad():
            weights, ids = grouped_topk(
                x,
                logits,
                module.top_k,
                module.norm_topk_prob,
                module.num_group,
                module.topk_group,
                "sigmoid",
                1.0,
                module.e_score_correction_bias.float(),
            )

        def native(logits):
            selected = logits.sigmoid().gather(1, ids.long())
            if module.norm_topk_prob:
                selected = selected / selected.sum(-1, keepdim=True)
            return selected

        # Differentiate the actual selected experts, not a recomputed ordering.
        weights = visible_forward(lambda logits: weights, native, logits)
        return logits, weights, ids

    def wrap_experts(module):
        original = _OriginalForward(module)

        def forward(module, x, ids, weights):
            def visible(x, up, down, weights):
                return fused_experts(
                    x,
                    up,
                    down,
                    weights,
                    ids,
                    activation=MoEActivation.RELU2_NO_MUL,
                )

            def native(x, up, down, weights):
                return torch.func.functional_call(
                    original,
                    {"module.up_proj": up, "module.down_proj": down},
                    (x, ids.long(), weights),
                )

            return visible_forward(
                visible, native, x, module.up_proj, module.down_proj, weights
            )

        return forward

    def wrap_moe(scale):
        def combine(shared, routed):
            return shared + routed * scale

        compiled_combine = torch.compile(combine, fullgraph=True, dynamic=True)

        def forward(module, x):
            shape = x.shape
            _, weights, ids = module.gate(x)
            routed = module.fc1_latent_proj(x.reshape(-1, shape[-1]))
            routed = module.fc2_latent_proj(module.experts(routed, ids, weights)).view(
                shape
            )
            shared = module.shared_experts(x)
            return visible_forward(compiled_combine, combine, shared, routed)

        return forward

    for module in model.modules():
        if isinstance(module, modeling.NemotronHMoE):
            if getattr(module, "_nemotron_moe_aligned", False):
                continue
            if module.config.moe_latent_size is not None:
                raise ValueError(
                    "This alignment adapter requires non-latent Nemotron MoE"
                )
            module.gate.forward = MethodType(route, module.gate)
            module.experts.forward = MethodType(
                wrap_experts(module.experts), module.experts
            )
            module.forward = MethodType(
                wrap_moe(module.config.routed_scaling_factor), module
            )
            module._nemotron_moe_aligned = True


def install_transformers_conv_forward():
    """Reuse vLLM's four-tap causal convolution and native convolution VJP."""
    from transformers.models.nemotron_h import modeling_nemotron_h as modeling

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    native_conv = getattr(
        modeling.causal_conv1d_fn, "__wrapped__", modeling.causal_conv1d_fn
    )

    @wraps(native_conv)
    def aligned_conv(x, weight, bias=None, activation=None, **kwargs):
        boundaries = _sequence_boundaries.get()

        def visible(x, weight, *bias_arg):
            batch, channels, length = x.shape
            offsets = boundaries or tuple(range(0, (batch + 1) * length, length))
            num_sequences = len(offsets) - 1
            bias = bias_arg[0] if bias_arg else None
            packed = x.transpose(1, 2).reshape(-1, channels).T
            states = torch.zeros(
                num_sequences + 1,
                channels,
                weight.shape[-1] - 1,
                device=x.device,
                dtype=x.dtype,
            )
            cu = torch.tensor(offsets, device=x.device, dtype=torch.int32)
            slots = torch.arange(
                1, num_sequences + 1, device=x.device, dtype=torch.int32
            )
            result = causal_conv1d_fn(
                packed,
                weight,
                bias,
                states,
                cu,
                cache_indices=slots,
                has_initial_state=torch.zeros(
                    num_sequences, device=x.device, dtype=torch.bool
                ),
                activation=activation,
            )
            return result.T.reshape(batch, length, channels).transpose(1, 2)

        def native(x, weight, *bias_arg):
            if boundaries is not None:
                return torch.cat(
                    [
                        native_conv(
                            x[:, :, a:b],
                            weight,
                            bias_arg[0] if bias_arg else None,
                            activation=activation,
                            **kwargs,
                        )
                        for a, b in zip(boundaries, boundaries[1:], strict=False)
                    ],
                    dim=2,
                )
            return native_conv(
                x,
                weight,
                bias_arg[0] if bias_arg else None,
                activation=activation,
                **kwargs,
            )

        inputs = (x, weight) if bias is None else (x, weight, bias)
        output = visible_forward(visible, native, *inputs)
        return output

    modeling.causal_conv1d_fn = aligned_conv
    return native_conv


def attention_forward(q, k, v, *, scale):
    """Canonical FA2 for a cache-free, unpadded causal training batch."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    batch, heads, length, dim = q.shape
    if k.shape[0] != batch or k.shape[2] != length or v.shape != k.shape:
        raise ValueError("Aligned attention requires equal query/key sequence lengths")
    boundaries = _sequence_boundaries.get()
    offsets = boundaries or tuple(range(0, (batch + 1) * length, length))
    cu = torch.tensor(offsets, device=q.device, dtype=torch.int32)
    max_length = max(b - a for a, b in zip(offsets, offsets[1:], strict=False))

    def pack(x):
        return x.transpose(1, 2).reshape(batch * length, x.shape[1], dim).contiguous()

    output = flash_attn_varlen_func(
        q=pack(q),
        k=pack(k),
        v=pack(v),
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_length,
        max_seqlen_k=max_length,
        softmax_scale=scale,
        causal=True,
        fa_version=2,
        num_splits=1,
    )
    return output.view(batch, length, heads, dim)


def _validate_causal_mask(mask, length):
    if not mask.is_floating_point() or mask.shape[-2:] != (length, length):
        raise ValueError("Aligned attention requires a square additive causal mask")
    allowed = torch.ones(length, length, device=mask.device, dtype=torch.bool).tril()
    blocked = torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min)
    if not torch.all(torch.where(allowed, mask == 0, blocked)):
        raise ValueError("Aligned attention supports unpadded causal masks only")


def install_transformers_attention_forward():
    """Use FA2 forward and the existing HF causal-attention VJP."""
    from transformers.models.nemotron_h import modeling_nemotron_h as modeling

    original = getattr(
        modeling.eager_attention_forward,
        "__wrapped__",
        modeling.eager_attention_forward,
    )

    @wraps(original)
    def forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        if dropout:
            raise ValueError("Aligned attention requires zero dropout")
        length = query.shape[2]
        boundaries = _sequence_boundaries.get()
        if attention_mask is not None:
            _validate_causal_mask(attention_mask, length)

        def visible(q, k, v):
            return attention_forward(q, k, v, scale=scaling)

        def native(q, k, v):
            mask = attention_mask
            if mask is None:
                allowed = torch.ones(
                    length, length, device=q.device, dtype=torch.bool
                ).tril()
                if boundaries is not None:
                    sequence_ids = torch.cat(
                        [
                            torch.full((b - a,), i, device=q.device)
                            for i, (a, b) in enumerate(
                                zip(boundaries, boundaries[1:], strict=False)
                            )
                        ]
                    )
                    allowed &= sequence_ids[:, None] == sequence_ids[None, :]
                mask = torch.zeros_like(allowed, dtype=q.dtype).masked_fill(
                    ~allowed, float("-inf")
                )
            return original(
                module, q, k, v, mask, scaling=scaling, dropout=0.0, **kwargs
            )[0]

        output = visible_forward(visible, native, query, key, value)
        return output, None

    modeling.eager_attention_forward = forward
    return original
