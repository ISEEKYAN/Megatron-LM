# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kimi Delta Attention recurrence and training-backend dispatch."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

KDABackend = Literal["auto", "torch", "fla"]


def _validate_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor | None,
) -> None:
    if q.ndim != 4:
        raise ValueError("KDA q must have shape [batch, sequence, heads, head_dim]")
    if q.shape != k.shape or q.shape != v.shape or q.shape != gate_logits.shape:
        raise ValueError("KDA q, k, v, and gate_logits must have the same shape")
    if beta_logits.shape != q.shape[:-1]:
        raise ValueError("KDA beta_logits must have shape [batch, sequence, heads]")
    heads, head_dim = q.shape[-2:]
    if a_log.shape != (heads,):
        raise ValueError(f"KDA a_log must have shape [{heads}]")
    if dt_bias.shape != (heads, head_dim):
        raise ValueError(f"KDA dt_bias must have shape [{heads}, {head_dim}]")
    if initial_state is not None:
        expected_state = (q.shape[0], heads, head_dim, v.shape[-1])
        if initial_state.shape != expected_state:
            raise ValueError(f"KDA initial_state must have shape {expected_state}")
    if not -5.0 <= lower_bound < 0.0:
        raise ValueError("KDA lower_bound must be in the FLA safe range [-5, 0)")


def torch_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    beta_logits: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable bounded recurrence for tiny correctness references."""
    _validate_kda_inputs(
        q, k, v, gate_logits, beta_logits, a_log, dt_bias, lower_bound, initial_state
    )
    output_dtype = v.dtype
    batch, sequence, heads, head_dim = q.shape
    scale = head_dim**-0.5 if scale is None else scale
    q = F.normalize(q.float(), p=2, dim=-1)
    k = F.normalize(k.float(), p=2, dim=-1)
    v = v.float()
    gate = lower_bound * torch.sigmoid(
        a_log.float().exp().view(1, 1, heads, 1)
        * (gate_logits.float() + dt_bias.float().view(1, 1, heads, head_dim))
    )
    beta = torch.sigmoid(beta_logits.float())
    state = (
        q.new_zeros(batch, heads, head_dim, v.shape[-1])
        if initial_state is None
        else initial_state.float()
    )
    outputs = []
    for token in range(sequence):
        state = state * gate[:, token].exp().unsqueeze(-1)
        prediction = torch.einsum("bhd,bhdv->bhv", k[:, token], state)
        update = beta[:, token].unsqueeze(-1) * (v[:, token] - prediction)
        state = state + torch.einsum("bhd,bhv->bhdv", k[:, token], update)
        outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, token], state) * scale)
    return torch.stack(outputs, dim=1).to(output_dtype), state


def kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    beta_logits: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    backend: KDABackend = "auto",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Execute KDA through a tiny torch oracle or FLA's trainable chunk kernel."""
    _validate_kda_inputs(
        q, k, v, gate_logits, beta_logits, a_log, dt_bias, lower_bound, initial_state
    )
    if backend not in {"auto", "torch", "fla"}:
        raise ValueError(f"unsupported KDA backend: {backend!r}")
    selected = "fla" if backend == "auto" and q.is_cuda else backend
    if selected in {"auto", "torch"}:
        if cu_seqlens is not None:
            raise NotImplementedError(
                "the torch KDA oracle accepts equal-length batches only"
            )
        output, final_state = torch_recurrent_kda(
            q,
            k,
            v,
            gate_logits,
            beta_logits,
            a_log=a_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            initial_state=initial_state,
            scale=scale,
        )
        return output, final_state if output_final_state else None

    try:
        from fla.ops.kda import chunk_kda
    except ImportError as error:
        raise ImportError(
            "FLA KDA requires flash-linear-attention with fla.ops.kda.chunk_kda"
        ) from error

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate_logits,
        beta=torch.sigmoid(beta_logits),
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        A_log=a_log,
        # FLA consumes a flat [heads * head_dim] bias. Keep the public
        # primitive/checkpoint contract head-major and adapt at this boundary.
        dt_bias=dt_bias.reshape(-1),
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        transpose_state_layout=True,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["KDABackend", "kda", "torch_recurrent_kda"]
