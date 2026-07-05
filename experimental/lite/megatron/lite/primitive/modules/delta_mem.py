# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""δ-mem online-memory primitive: a writable delta-rule state that steers attention.

Reference implementation: declare-lab/delta-Mem ``deltamem/core/delta_impl.py``
(``DeltaMemAttention``), the released code for arXiv:2605.12357. The math below is
copied op-for-op from the reference's *released* configuration (``normalize_qk=True``,
``couple_lambda=True``, ``rankwise_gates=True``, ``state_update_mode="standard"``,
``memory_write_source="learned_hidden"``); the reference's archived variants
(partition routing, free λ, scalar gates, sentence granularity) are not ported.

Per position ``t`` (paper Eq. 4-12, code ``_memory_sequence_projections`` /
``_memory_update_coefficients`` / ``_memory_affine_scan_torch``):

- projections: ``q^m/k^m = L2norm(tanh(W x))`` per sub-state, ``v^m = W_v x``,
  gate ``β = σ(W_β x + b)`` with tied ``λ = 1 − β`` (per state dimension);
- read (BEFORE the position's write): ``r_t = S_{t-1} q^m_t``;
- write: ``S_t = Diag(λ) S_{t-1} + Diag(β) (v^m − S_{t-1} k^m) (k^m)ᵀ``;
- steer: ``Δq = (α/r)·W_q^Δ r_t`` added to the RAW query-projection output —
  before any per-head q-norm and before RoPE — and ``Δo = (α/r)·W_o^Δ r_t``
  added to the attention output projection (integration-layer contract).

Masked (padding) positions read zero and leave the state unchanged. With
``write_granularity="message"`` (SSW), hidden states are mean-pooled per message
id and written once per message while every token reads the *pre-chunk* state;
when no active message id exists in a chunk the reference silently falls back to
per-token writes — that fallback is replicated here on purpose.

Deliberate departures from the reference (plumbing only, never math):

- the state is an explicit input/output (``init_state`` → ``forward``), never a
  module attribute: mlite engines own state placement, lifetime, and snapshots;
- only the configured ``branches`` allocate Δ-heads (the reference also allocates
  zeroed, frozen k/v heads and stores them in its adapter files, so its
  *allocated* count exceeds its *trainable* count — ours are equal);
- the token-validity mask is a boolean ``[batch, seq]`` supplied by the caller
  (the reference derives it from HF attention-mask formats);
- torch sequential scan only, no env-var dispatch. The reference's Triton kernel
  is an execution detail (time loop inside the kernel, saved state history for a
  custom backward); no chunked/parallel form exists upstream to port.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_WRITE_GRANULARITIES = ("token", "message")
_GRANULARITY_ALIASES = {
    # reference config spellings
    "token": "token",
    "message_mean": "message",
    "message": "message",
}
_VALID_BRANCHES = ("q", "o")
_OUTPUT_INITS = ("zero", "base_slice_fixed")


def delta_mem_scaling(rank: int, alpha: float) -> float:
    """δ-mem steering scale ``α / r`` (reference ``delta_scaling``; α=16, r=8 ⇒ 2.0)."""
    return float(alpha) / float(rank)


@dataclass(frozen=True)
class DeltaMemConfig:
    rank: int = 0
    alpha: float = 16.0
    num_states: int = 1  # MSW sub-states N; 1 = TSW/SSW
    write_granularity: str = "token"  # "token" (TSW/MSW) | "message" (SSW)
    branches: tuple[str, ...] = ("q", "o")
    beta_bias_init: float = -1.5  # β₀ = σ(−1.5) = 0.1824255…
    output_init: str = "zero"  # reference config default; released models use "base_slice_fixed"
    base_slice_ref_width: int = 8
    online_gain: float = 0.05

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError(f"delta_mem rank must be >= 0, got {self.rank}.")
        if self.num_states < 1:
            raise ValueError(f"delta_mem num_states must be >= 1, got {self.num_states}.")
        if self.write_granularity not in _WRITE_GRANULARITIES:
            raise ValueError(
                f"delta_mem write_granularity must be one of {_WRITE_GRANULARITIES}, "
                f"got {self.write_granularity!r}."
            )
        for branch in self.branches:
            if branch not in _VALID_BRANCHES:
                raise ValueError(
                    f"delta_mem branches may contain only {_VALID_BRANCHES}, got {branch!r}."
                )
        if self.output_init not in _OUTPUT_INITS:
            raise ValueError(
                f"delta_mem output_init must be one of {_OUTPUT_INITS}, got {self.output_init!r}."
            )

    @property
    def enabled(self) -> bool:
        return self.rank > 0

    @property
    def scale(self) -> float:
        return delta_mem_scaling(self.rank, self.alpha)

    @property
    def state_dim(self) -> int:
        return self.rank * self.num_states


def normalize_delta_mem_config(
    config: DeltaMemConfig | dict[str, Any] | None,
) -> DeltaMemConfig:
    if config is None:
        return DeltaMemConfig()
    if isinstance(config, DeltaMemConfig):
        return config
    if not isinstance(config, dict):
        raise TypeError(
            f"delta_mem config must be DeltaMemConfig, dict, or None, got {type(config)!r}."
        )
    values = dict(config)
    enabled = values.pop("enabled", None)
    if enabled is False:
        values["rank"] = 0
    if "num_state_heads" in values and "num_states" not in values:  # reference field name
        values["num_states"] = values.pop("num_state_heads")
    else:
        values.pop("num_state_heads", None)
    if "memory_write_granularity" in values and "write_granularity" not in values:
        values["write_granularity"] = values.pop("memory_write_granularity")
    else:
        values.pop("memory_write_granularity", None)
    if "write_granularity" in values:
        raw = values["write_granularity"]
        if raw not in _GRANULARITY_ALIASES:
            raise ValueError(
                f"delta_mem write_granularity must be one of {sorted(_GRANULARITY_ALIASES)}, "
                f"got {raw!r}."
            )
        values["write_granularity"] = _GRANULARITY_ALIASES[raw]
    if "branches" in values and not isinstance(values["branches"], tuple):
        values["branches"] = tuple(values["branches"])
    return DeltaMemConfig(**values)


def apply_delta_mem_base_slice_init(model: nn.Module) -> dict[str, int]:
    """Apply the released-model ``base_slice_fixed`` Δ-head init to every
    ``DeltaMemory`` under ``model`` (post-load hook, like OLoRA-tail init).

    Pairs each adapter with its frozen base weights by the mlite attribute
    convention: the owning attention module exposes a fused ``qkv`` linear
    (query rows first), a ``proj`` linear, and ``q_size``.
    """
    count = 0
    for module in model.modules():
        adapter = getattr(module, "delta_mem", None)
        if not isinstance(adapter, DeltaMemory):
            continue
        if adapter.config.output_init != "base_slice_fixed":
            continue
        qkv = getattr(module, "qkv", None)
        proj = getattr(module, "proj", None)
        q_size = getattr(module, "q_size", None)
        if qkv is None or proj is None or q_size is None:
            raise ValueError(
                "delta_mem base_slice init: the module owning `delta_mem` must "
                "expose fused `qkv` (query rows first), `proj`, and `q_size`."
            )
        qkv_weight = qkv.weight if hasattr(qkv, "weight") else qkv.linear.weight
        proj_weight = proj.weight if hasattr(proj, "weight") else proj.linear.weight
        adapter.base_slice_init_(qkv_weight[:q_size], proj_weight)
        count += 1
    return {"delta_mem_base_slice_inits": count}


class DeltaMemory(nn.Module):
    """The δ-mem state module for one attention layer (all heads share one state).

    Owns the memory projections, the tied gate, and the active Δ-heads; produces
    ``(delta_q, delta_o, next_state)`` from ``(hidden_states, state)``. It does NOT
    own the base attention computation — the integration layer adds ``delta_q`` to
    the raw query-projection output (pre q-norm, pre RoPE) and ``delta_o`` to the
    output-projection result.

    ``query_out_features`` is the base query-projection output width (Δq target),
    ``output_out_features`` the attention-output width (Δo target). tp=1 single-GPU
    semantics; parallel placement is the integration layer's concern.
    """

    def __init__(
        self,
        hidden_size: int,
        query_out_features: int,
        output_out_features: int,
        config: DeltaMemConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        config = normalize_delta_mem_config(config)
        if not config.enabled:
            raise ValueError("DeltaMemory requires rank > 0; gate creation on config.enabled.")
        self.config = config
        self.rank = config.rank
        self.num_states = config.num_states
        self.state_dim = config.state_dim
        self.gate_dim = config.state_dim  # rankwise gates: one β per state dimension
        self.scale = config.scale
        self.hidden_size = hidden_size
        self.query_out_features = query_out_features
        self.output_out_features = output_out_features

        self.memory_q_proj = nn.Parameter(torch.empty(self.state_dim, hidden_size))
        self.memory_k_proj = nn.Parameter(torch.empty(self.state_dim, hidden_size))
        self.memory_v_proj = nn.Parameter(torch.empty(self.state_dim, hidden_size))
        self.beta_proj = nn.Parameter(torch.empty(self.gate_dim, hidden_size))
        self.beta_bias = nn.Parameter(torch.full((self.gate_dim,), config.beta_bias_init))
        if "q" in config.branches:
            self.delta_q_proj = nn.Parameter(torch.empty(query_out_features, self.state_dim))
        else:
            self.register_parameter("delta_q_proj", None)
        if "o" in config.branches:
            self.delta_o_proj = nn.Parameter(torch.empty(output_out_features, self.state_dim))
        else:
            self.register_parameter("delta_o_proj", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Reference ``reset_parameters``: kaiming(a=√5) on the memory trio, zero gate
        # weight (β₀ = σ(bias) exactly), Δ-heads zero until ``base_slice_init_`` for
        # the released ``base_slice_fixed`` mode.
        nn.init.kaiming_uniform_(self.memory_q_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.memory_k_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.memory_v_proj, a=math.sqrt(5))
        nn.init.zeros_(self.beta_proj)
        with torch.no_grad():
            self.beta_bias.fill_(self.config.beta_bias_init)
        for head in (self.delta_q_proj, self.delta_o_proj):
            if head is not None:
                nn.init.zeros_(head)

    @torch.no_grad()
    def base_slice_init_(
        self,
        query_weight: Optional[torch.Tensor] = None,
        output_weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Released-model Δ-head init (reference ``_init_delta_head``, ``base_slice_fixed``).

        First ``min(base_slice_ref_width, rank, in_features)`` base-weight columns,
        column-L2-normalized in float32 (eps 1e-6), scaled by ``online_gain``; the
        remaining Δ-head columns stay zero. Call post-load, like OLoRA-tail init.
        """
        if self.config.output_init != "base_slice_fixed":
            raise ValueError(
                f"base_slice_init_ requires output_init='base_slice_fixed', "
                f"got {self.config.output_init!r}."
            )
        for head, base_weight in ((self.delta_q_proj, query_weight), (self.delta_o_proj, output_weight)):
            if head is None:
                continue
            if base_weight is None:
                raise ValueError("base_slice_init_ needs the base weight for every active branch.")
            if base_weight.shape[0] != head.shape[0]:
                raise ValueError(
                    f"base weight rows {base_weight.shape[0]} != delta head rows {head.shape[0]}."
                )
            slice_width = min(self.config.base_slice_ref_width, self.rank, base_weight.shape[1])
            head.zero_()
            if slice_width == 0:
                continue
            base_slice = base_weight[:, :slice_width].detach().clone().float()
            base_slice = F.normalize(base_slice, dim=0, eps=1e-6)
            head[:, :slice_width].copy_((base_slice * self.config.online_gain).to(head.dtype))

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Zero state ``S_0``: ``[B, r, r]`` (N=1) or ``[B, N, r, r]`` (MSW), backbone dtype."""
        if self.num_states > 1:
            shape = (batch_size, self.num_states, self.rank, self.rank)
        else:
            shape = (batch_size, self.rank, self.rank)
        return torch.zeros(shape, device=device, dtype=dtype)

    def project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference ``_memory_sequence_projections`` (couple_lambda path)."""
        packed_gates = F.linear(hidden_states, self.beta_proj)
        packed_memory_weight = torch.cat(
            [self.memory_q_proj, self.memory_k_proj, self.memory_v_proj], dim=0
        )
        packed_memory = F.linear(hidden_states, packed_memory_weight)
        memory_q, memory_k, memory_v = torch.split(
            packed_memory, [self.state_dim, self.state_dim, self.state_dim], dim=-1
        )
        memory_q = self._normalize_memory_projection(memory_q)
        memory_k = self._normalize_memory_projection(memory_k)
        beta = torch.sigmoid(
            packed_gates
            + self.beta_bias.view(*([1] * (hidden_states.dim() - 1)), self.gate_dim)
        ).unsqueeze(-1)
        lam = 1.0 - beta
        return memory_q, memory_k, memory_v, beta, lam

    def _normalize_memory_projection(self, projected: torch.Tensor) -> torch.Tensor:
        # Reference ``_normalize_memory_projection``: tanh + L2 per sub-state.
        if self.num_states > 1:
            projected = projected.view(*projected.shape[:-1], self.num_states, self.rank)
            projected = torch.tanh(projected)
            projected = F.normalize(projected, dim=-1, eps=1e-6)
            return projected.reshape(*projected.shape[:-2], self.state_dim)
        projected = torch.tanh(projected)
        return F.normalize(projected, dim=-1, eps=1e-6)

    def _update_coefficients(
        self, beta_seq: torch.Tensor, lambda_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference ``_memory_update_coefficients``, ``standard`` mode with rankwise gates:
        keep = λ, erase = β, write = β."""
        beta_rows = beta_seq.squeeze(-1) if beta_seq.ndim == 4 else beta_seq
        lambda_rows = lambda_seq.squeeze(-1) if lambda_seq.ndim == 4 else lambda_seq
        if self.num_states > 1:
            beta_rows = beta_rows.view(
                beta_rows.size(0), beta_rows.size(1), self.num_states, self.rank
            )
            lambda_rows = lambda_rows.view(
                lambda_rows.size(0), lambda_rows.size(1), self.num_states, self.rank
            )
        return lambda_rows, beta_rows, beta_rows

    @staticmethod
    def affine_scan_torch(
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        memory_k_seq: torch.Tensor,
        memory_v_seq: torch.Tensor,
        keep_seq: torch.Tensor,
        erase_seq: torch.Tensor,
        write_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference ``_memory_affine_scan_torch``, verbatim: sequential read-before-write
        delta rule; masked positions read zero and keep the previous state."""
        batch_size, seq_len, _ = memory_q_seq.shape
        current_state = state
        read_steps: list[torch.Tensor] = []

        for token_idx in range(seq_len):
            q_t = memory_q_seq[:, token_idx, :]
            k_t = memory_k_seq[:, token_idx, :]
            v_t = memory_v_seq[:, token_idx, :]
            keep_t = keep_seq[:, token_idx, :].unsqueeze(-1)
            erase_t = erase_seq[:, token_idx, :].unsqueeze(-1)
            write_t = write_seq[:, token_idx, :].unsqueeze(-1)

            read_t = torch.einsum("bij,bj->bi", current_state, q_t)

            if token_mask is not None:
                valid = token_mask[:, token_idx].view(batch_size, 1)
                read_t = read_t * valid.to(dtype=read_t.dtype)

            pred_t = torch.einsum("bij,bj->bi", current_state, k_t)
            write_outer = v_t.unsqueeze(-1) * k_t.unsqueeze(1)
            pred_outer = pred_t.unsqueeze(-1) * k_t.unsqueeze(1)
            next_state = keep_t * current_state - erase_t * pred_outer + write_t * write_outer

            if token_mask is not None:
                valid_state = (
                    token_mask[:, token_idx].view(batch_size, 1, 1).to(dtype=next_state.dtype)
                )
                current_state = next_state * valid_state + current_state * (1.0 - valid_state)
            else:
                current_state = next_state

            read_steps.append(read_t)

        reads = torch.stack(read_steps, dim=1)
        return current_state, reads

    def scan(
        self,
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        memory_k_seq: torch.Tensor,
        memory_v_seq: torch.Tensor,
        beta_seq: torch.Tensor,
        lambda_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference ``_memory_affine_scan`` (single-head and multi-head-state paths)."""
        keep_seq, erase_seq, write_seq = self._update_coefficients(beta_seq, lambda_seq)
        if self.num_states > 1:
            batch_size, seq_len, _ = memory_q_seq.shape
            n = self.num_states
            q_for_scan = memory_q_seq.view(batch_size, seq_len, n, self.rank)
            k_for_scan = memory_k_seq.view(batch_size, seq_len, n, self.rank)
            v_for_scan = memory_v_seq.view(batch_size, seq_len, n, self.rank)
            state_for_scan = state.reshape(batch_size * n, self.rank, self.rank)
            q_for_scan = q_for_scan.permute(0, 2, 1, 3).reshape(batch_size * n, seq_len, self.rank)
            k_for_scan = k_for_scan.permute(0, 2, 1, 3).reshape(batch_size * n, seq_len, self.rank)
            v_for_scan = v_for_scan.permute(0, 2, 1, 3).reshape(batch_size * n, seq_len, self.rank)
            keep_for_scan = keep_seq.permute(0, 2, 1, 3).reshape(batch_size * n, seq_len, self.rank)
            erase_for_scan = erase_seq.permute(0, 2, 1, 3).reshape(batch_size * n, seq_len, self.rank)
            write_for_scan = write_seq.permute(0, 2, 1, 3).reshape(batch_size * n, seq_len, self.rank)
            token_mask_for_scan = None
            if token_mask is not None:
                token_mask_for_scan = (
                    token_mask.unsqueeze(1)
                    .expand(batch_size, n, seq_len)
                    .reshape(batch_size * n, seq_len)
                )
            final_state, reads = self.affine_scan_torch(
                state_for_scan,
                q_for_scan,
                k_for_scan,
                v_for_scan,
                keep_for_scan,
                erase_for_scan,
                write_for_scan,
                token_mask=token_mask_for_scan,
            )
            final_state = final_state.reshape(batch_size, n, self.rank, self.rank)
            reads = reads.reshape(batch_size, n, seq_len, self.rank)
            reads = reads.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.state_dim)
            return final_state, reads
        return self.affine_scan_torch(
            state,
            memory_q_seq,
            memory_k_seq,
            memory_v_seq,
            keep_seq,
            erase_seq,
            write_seq,
            token_mask=token_mask,
        )

    def token_reads(
        self,
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reference ``_token_state_reads``: read a frozen state, no writes."""
        if self.num_states > 1:
            head_q = memory_q_seq.view(
                memory_q_seq.size(0), memory_q_seq.size(1), self.num_states, self.rank
            )
            reads = torch.einsum("bhij,bthj->bthi", state, head_q)
            reads = reads.reshape(memory_q_seq.size(0), memory_q_seq.size(1), self.state_dim)
        else:
            reads = torch.einsum("bij,btj->bti", state, memory_q_seq)
        if token_mask is not None:
            reads = reads * token_mask.unsqueeze(-1).to(dtype=reads.dtype)
        return reads

    def message_write_means(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        message_ids: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Reference ``_message_write_inputs`` + ``_build_message_write_means``.

        Returns ``(message_hidden [B, M, d], message_mask [B, M])`` or ``(None, None)``
        when message ids are absent/mis-shaped or no active (≥0 ∧ unmasked) token
        exists — the caller then falls back to per-token writes (reference behavior).
        """
        if message_ids is None or message_ids.dim() != 2:
            return None, None
        if message_ids.size(0) != hidden_states.size(0) or message_ids.size(1) != hidden_states.size(1):
            return None, None
        message_ids = message_ids.to(device=hidden_states.device)
        active_mask = message_ids.ge(0)
        if token_mask is not None:
            active_mask = active_mask & token_mask
        if not active_mask.any():
            return None, None
        max_message_id = int(message_ids.masked_select(active_mask).max().item())
        num_messages_max = max_message_id + 1
        message_hidden = hidden_states.new_zeros(
            hidden_states.size(0), num_messages_max, hidden_states.size(-1)
        )
        message_mask = torch.zeros(
            hidden_states.size(0), num_messages_max, dtype=torch.bool, device=hidden_states.device
        )
        for batch_idx in range(hidden_states.size(0)):
            sample_message_ids = message_ids[batch_idx]
            sample_active_mask = active_mask[batch_idx]
            if not sample_active_mask.any():
                continue
            for message_id in (
                sample_message_ids.masked_select(sample_active_mask).unique(sorted=True).tolist()
            ):
                current_message_id = int(message_id)
                token_selector = sample_active_mask & sample_message_ids.eq(current_message_id)
                message_hidden[batch_idx, current_message_id] = hidden_states[
                    batch_idx, token_selector
                ].mean(dim=0)
                message_mask[batch_idx, current_message_id] = True
        return message_hidden, message_mask

    @staticmethod
    def masked_gate_mean(
        values: torch.Tensor, token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Reference ``_masked_gate_mean`` — the β̄ statistic the write-sparsity
        penalty consumes (trainer wiring, not used inside ``forward``)."""
        if token_mask is None:
            return values.mean()
        expanded_mask = token_mask.unsqueeze(-1).unsqueeze(-1)
        masked_values = values * expanded_mask.to(dtype=values.dtype)
        denom = expanded_mask.sum().clamp_min(1).to(dtype=values.dtype)
        return masked_values.sum() / denom

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        *,
        token_mask: Optional[torch.Tensor] = None,
        message_ids: Optional[torch.Tensor] = None,
        write_enabled: bool = True,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """One chunk: read → (optionally) write → steer.

        Returns ``(delta_q, delta_o, next_state)``; a Δ output is ``None`` for an
        inactive branch. ``write_enabled=False`` reads the frozen state (the
        reference's eval/read-pass mode). ``message_ids [B, T]`` (−1 = skip)
        activates SSW writes when ``write_granularity="message"``.
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"DeltaMemory expects hidden_states [batch, seq, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        memory_q_seq, memory_k_seq, memory_v_seq, beta_seq, lambda_seq = self.project(hidden_states)
        if write_enabled:
            state_before_write = state
            write_hidden = None
            write_mask = None
            if self.config.write_granularity == "message":
                write_hidden, write_mask = self.message_write_means(
                    hidden_states, token_mask, message_ids
                )
            if write_hidden is not None and write_mask is not None:
                write_q, write_k, write_v, write_beta, write_lambda = self.project(write_hidden)
                state, _ = self.scan(
                    state, write_q, write_k, write_v, write_beta, write_lambda,
                    token_mask=write_mask,
                )
                reads = self.token_reads(state_before_write, memory_q_seq, token_mask)
            else:
                state, reads = self.scan(
                    state, memory_q_seq, memory_k_seq, memory_v_seq, beta_seq, lambda_seq,
                    token_mask=token_mask,
                )
        else:
            reads = self.token_reads(state, memory_q_seq, token_mask)
        delta_q = None
        if self.delta_q_proj is not None:
            delta_q = F.linear(reads, self.delta_q_proj) * self.scale
        delta_o = None
        if self.delta_o_proj is not None:
            delta_o = F.linear(reads, self.delta_o_proj) * self.scale
        return delta_q, delta_o, state
