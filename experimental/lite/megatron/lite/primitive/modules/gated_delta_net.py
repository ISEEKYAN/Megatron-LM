# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen-style Gated DeltaNet primitive.

Context-parallel (CP) modes
---------------------------
- ``headwise`` (default under CP): head-parallel all-to-all. Each rank all-to-alls
  its zigzag sequence shard so it ends up holding the *full* sequence for ``1/cp`` of
  the heads (with the matching slices of conv1d / A_log / dt_bias), runs the ordinary
  full-sequence gated-delta recurrence, then all-to-alls back. Heads are independent,
  so this is numerically **bitwise-identical to CP-off** while sharding the per-head
  state/activation memory across ranks. This mirrors upstream Megatron
  ``linear_cp_mode='headwise'`` (``megatron/core/ssm/gated_delta_net.py``).
- ``replicated``: all-gather the sequence, run the full recurrence on every rank
  (every head replicated), zigzag-slice the output. Also bitwise-identical to CP-off,
  but every rank materialises the full sequence for *all* heads (worst memory).

The FLA chunkwise ``cp_context`` ring path (formerly ``sharded``) has been removed:
its cross-rank chunk-state accumulation order differs from the reference, which under
RL amplifies bf16 rounding into a large train/inference log-prob mismatch. Use
``headwise`` for a memory-efficient *and* numerically exact CP linear-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from megatron.lite.primitive.kernels.jit import jit_fuser
from megatron.lite.primitive.ops.gated_delta_rule import (
    l2norm,
    torch_chunk_gated_delta_rule,
)
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
)
from megatron.lite.primitive.parallel.cp import (
    all_to_all_hidden_shards,
    build_headwise_section_perm,
    get_parameter_local_cp_headwise,
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
)
from megatron.lite.primitive.parallel.thd import (
    reconstruct_packed_from_cp_parts,
    split_packed_to_cp_local,
)
from megatron.lite.primitive.utils import ensure_divisible


try:
    from fla.modules.convolution import (
        causal_conv1d as _fla_causal_conv1d,  # pyright: ignore[reportMissingImports]
    )
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,  # pyright: ignore[reportMissingImports]
    )

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False

_CONV_PAD_ALIGNMENT = 4096

_CP_MODES = {"headwise", "replicated"}


class GatedDeltaNet(nn.Module):
    """Native Gated DeltaNet with head-parallel / replicated CP support."""

    def __init__(
        self,
        *,
        hidden_size: int,
        linear_num_key_heads: int,
        linear_key_head_dim: int,
        linear_num_value_heads: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int,
        rms_norm_eps: float,
        ps: ParallelState,
        deterministic: bool = False,
        cp_mode: str = "headwise",
    ):
        super().__init__()
        if cp_mode not in _CP_MODES:
            raise ValueError(
                f"Unsupported GatedDeltaNet CP mode: {cp_mode!r}; expected one of {sorted(_CP_MODES)}."
            )
        self.ps = ps
        self.deterministic = bool(deterministic)
        self.cp_mode = cp_mode
        self.num_k_heads = linear_num_key_heads
        self.num_v_heads = linear_num_value_heads
        self.dk = linear_key_head_dim
        self.dv = linear_value_head_dim
        self.v_heads_per_k_head = ensure_divisible(self.num_v_heads, self.num_k_heads)
        self.num_k_heads_local = ensure_divisible(self.num_k_heads, ps.tp_size)
        self.num_v_heads_local = ensure_divisible(self.num_v_heads, ps.tp_size)
        self.qk_dim = self.num_k_heads * self.dk
        self.v_dim = self.num_v_heads * self.dv
        self.qk_dim_local = self.num_k_heads_local * self.dk
        self.v_dim_local = self.num_v_heads_local * self.dv
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_v_heads * 2

        self.in_proj = ColumnParallelLinear(
            hidden_size,
            self.in_proj_dim,
            ps,
            bias=False,
            normalization="RMSNorm",
            eps=rms_norm_eps,
            zero_centered_gamma=True,
        )
        conv_dim_local = self.qk_dim_local * 2 + self.v_dim_local
        self.conv_dim_local = conv_dim_local
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim_local,
            out_channels=conv_dim_local,
            kernel_size=linear_conv_kernel_dim,
            groups=conv_dim_local,
            bias=False,
            padding=linear_conv_kernel_dim - 1,
        )
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads_local, dtype=torch.float32)
        )
        self.A_log = nn.Parameter(
            torch.zeros(self.num_v_heads_local, dtype=torch.float32)
        )
        self.norm = te.RMSNorm(self.dv, eps=rms_norm_eps, zero_centered_gamma=True)
        self.o_proj = RowParallelLinear(self.v_dim, hidden_size, ps, bias=False)

    # The six ``qkvzba`` sections and the three conv channel sections, in hidden order.
    def _qkvzba_sections(self) -> list[int]:
        return [
            self.qk_dim_local,
            self.qk_dim_local,
            self.v_dim_local,
            self.v_dim_local,
            self.num_v_heads_local,
            self.num_v_heads_local,
        ]

    def _conv_sections(self) -> list[int]:
        return [self.qk_dim_local, self.qk_dim_local, self.v_dim_local]

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None, packed_seq_params=None
    ) -> torch.Tensor:
        del position_ids
        is_packed = packed_seq_params is not None
        qkvzba = self.in_proj(x).transpose(0, 1).contiguous()
        cu_seqlens = self._packed_cu_seqlens(packed_seq_params) if is_packed else None

        headwise = False
        replicated = False
        cp_div = 1
        if self.ps.cp_size > 1:
            if self.ps.cp_group is None:
                raise RuntimeError("CP>1 requires ParallelState.cp_group.")
            if self.cp_mode == "replicated":
                qkvzba, cu_seqlens = self._replicate_cp_qkvzba(qkvzba, cu_seqlens)
                replicated = True
            else:  # headwise
                if not is_packed and qkvzba.shape[0] > 1:
                    raise ValueError(
                        "GatedDeltaNet headwise CP with SBHD inputs currently requires "
                        "micro_batch_size == 1. Use packed THD input or micro_batch_size=1."
                    )
                qkvzba, cu_seqlens = self._headwise_cp2hp(qkvzba, cu_seqlens)
                headwise = True
                cp_div = self.ps.cp_size

        batch, seq_len = qkvzba.shape[:2]

        # Per-rank parameter slices (identity unless headwise).
        if headwise:
            conv_weight = get_parameter_local_cp_headwise(
                self.conv1d.weight,
                dim=0,
                cp_size=self.ps.cp_size,
                cp_rank=self.ps.cp_rank,
                split_sections=self._conv_sections(),
            )
            A_log = get_parameter_local_cp_headwise(
                self.A_log, dim=0, cp_size=self.ps.cp_size, cp_rank=self.ps.cp_rank
            )
            dt_bias = get_parameter_local_cp_headwise(
                self.dt_bias, dim=0, cp_size=self.ps.cp_size, cp_rank=self.ps.cp_rank
            )
        else:
            conv_weight, A_log, dt_bias = None, self.A_log, self.dt_bias

        query, key, value, gate, beta, alpha = self._split_proj(qkvzba, cp_div)
        qkv = torch.cat(
            [
                query.reshape(batch, seq_len, -1),
                key.reshape(batch, seq_len, -1),
                value.reshape(batch, seq_len, -1),
            ],
            dim=-1,
        )

        qkv = self._causal_conv1d(
            qkv, seq_len, cu_seqlens=cu_seqlens, conv_weight=conv_weight, cp_div=cp_div
        )
        query, key, value, gate, beta, alpha = self._prepare_qkv(
            qkv, gate, beta, alpha, batch, seq_len, cp_div
        )
        g, beta = self._compute_g_and_beta(A_log, dt_bias, alpha, beta)
        out, _ = self._gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )

        out = self._apply_gated_norm(out, gate)
        out = out.reshape(batch, seq_len, self.v_dim_local // cp_div)
        if self.ps.cp_size > 1:
            if replicated:
                out = self._slice_replicated_output(out, cu_seqlens)
            else:  # headwise
                out = self._headwise_hp2cp(out, cu_seqlens)
            batch, seq_len = out.shape[:2]
        out = out.transpose(0, 1).contiguous()
        return self.o_proj(out)

    # ------------------------------------------------------------------ CP: replicated
    def _all_gather_cp_tensor(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        if self.ps.cp_size <= 1:
            return [tensor]
        try:
            from torch.distributed.nn.functional import all_gather

            return list(all_gather(tensor, group=self.ps.cp_group))
        except Exception:
            parts = [torch.empty_like(tensor) for _ in range(self.ps.cp_size)]
            torch.distributed.all_gather(parts, tensor, group=self.ps.cp_group)
            return parts

    def _replicate_cp_qkvzba(
        self,
        qkvzba: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cu_seqlens is None:
            parts = self._all_gather_cp_tensor(qkvzba)
            return zigzag_reconstruct_from_cp_parts(parts, seq_dim=1), None
        if qkvzba.shape[0] != 1:
            raise ValueError("Packed THD GatedDeltaNet expects a single packed batch row.")
        parts = self._all_gather_cp_tensor(qkvzba[0].contiguous())
        full = reconstruct_packed_from_cp_parts(
            parts,
            cu_seqlens_padded=cu_seqlens,
            cp_size=self.ps.cp_size,
            dim=0,
        )
        return full.unsqueeze(0).contiguous(), cu_seqlens

    def _slice_replicated_output(
        self,
        out: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            return zigzag_slice_for_cp(out, self.ps.cp_rank, self.ps.cp_size, seq_dim=1)
        if out.shape[0] != 1:
            raise ValueError("Packed THD GatedDeltaNet expects a single packed batch row.")
        local = split_packed_to_cp_local(
            out[0].contiguous(),
            cu_seqlens_padded=cu_seqlens,
            cp_size=self.ps.cp_size,
            cp_rank=self.ps.cp_rank,
            dim=0,
        )
        return local.unsqueeze(0).contiguous()

    # ------------------------------------------------------------------ CP: headwise
    def _headwise_cp2hp(
        self,
        qkvzba: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Zigzag-sharded ``[b, s_local, h]`` -> contiguous full-seq ``[b, s_global, h/cp]``.

        Permutes the hidden dim section-wise so a plain hidden-scatter all-to-all leaves
        each rank with ``1/cp`` heads of every section, then reassembles the full
        sequence from the per-rank shards with the same (verified) reconstruction used by
        ``replicated``.
        """
        cp_size = self.ps.cp_size
        perm = build_headwise_section_perm(
            self._qkvzba_sections(), cp_size, qkvzba.device
        )
        qkvzba = qkvzba.index_select(-1, perm)
        h = qkvzba.shape[-1]
        hpc = h // cp_size
        send_parts = [qkvzba[..., k * hpc : (k + 1) * hpc].contiguous() for k in range(cp_size)]
        recv = all_to_all_hidden_shards(send_parts, self.ps.cp_group)
        if cu_seqlens is None:
            full = zigzag_reconstruct_from_cp_parts(recv, seq_dim=1)
            return full.contiguous(), None
        if qkvzba.shape[0] != 1:
            raise ValueError("Packed THD GatedDeltaNet expects a single packed batch row.")
        parts = [p[0].contiguous() for p in recv]
        full = reconstruct_packed_from_cp_parts(
            parts,
            cu_seqlens_padded=cu_seqlens,
            cp_size=cp_size,
            dim=0,
        )
        return full.unsqueeze(0).contiguous(), cu_seqlens

    def _headwise_hp2cp(
        self,
        out: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Inverse of :meth:`_headwise_cp2hp` for the value output.

        Full-seq ``[b, s_global, v/cp]`` (this rank owns value-head shard ``cp_rank``) ->
        zigzag-sharded ``[b, s_local, v]`` with all heads. The value section's per-rank
        blocks are contiguous, so concatenating the received hidden shards in rank order
        restores the natural head order (no inverse permutation required).
        """
        cp_size = self.ps.cp_size
        if cu_seqlens is None:
            send_parts = [
                zigzag_slice_for_cp(out, j, cp_size, seq_dim=1).contiguous()
                for j in range(cp_size)
            ]
            recv = all_to_all_hidden_shards(send_parts, self.ps.cp_group)
            return torch.cat(recv, dim=-1).contiguous()
        if out.shape[0] != 1:
            raise ValueError("Packed THD GatedDeltaNet expects a single packed batch row.")
        base = out[0].contiguous()
        send_parts = [
            split_packed_to_cp_local(
                base,
                cu_seqlens_padded=cu_seqlens,
                cp_size=cp_size,
                cp_rank=j,
                dim=0,
            ).contiguous()
            for j in range(cp_size)
        ]
        recv = all_to_all_hidden_shards(send_parts, self.ps.cp_group)
        return torch.cat(recv, dim=-1).unsqueeze(0).contiguous()

    # ------------------------------------------------------------------ compute
    def _causal_conv1d(
        self,
        qkv: torch.Tensor,
        seq_len: int,
        *,
        cu_seqlens: torch.Tensor | None,
        conv_weight: torch.Tensor | None,
        cp_div: int,
    ) -> torch.Tensor:
        # ``conv_weight`` is the (headwise) per-rank slice; None means use the full module.
        weight = self.conv1d.weight if conv_weight is None else conv_weight
        groups = self.conv_dim_local // cp_div
        if _HAS_FLA and (cu_seqlens is not None or not self.deterministic):
            orig_seq_len = qkv.shape[1]
            pad_n = -orig_seq_len % _CONV_PAD_ALIGNMENT
            conv_input = qkv
            conv_cu_seqlens = cu_seqlens
            if pad_n > 0:
                conv_input = F.pad(qkv, (0, 0, 0, pad_n))
                if conv_cu_seqlens is not None:
                    conv_cu_seqlens = conv_cu_seqlens.clone()
                    conv_cu_seqlens[-1] += pad_n
            qkv, _ = _fla_causal_conv1d(
                x=conv_input,
                weight=weight.squeeze(1),
                bias=None,
                activation="silu",
                cu_seqlens=conv_cu_seqlens,
            )
            if pad_n > 0:
                qkv = qkv[:, :orig_seq_len, :]
            return qkv
        if cu_seqlens is None:
            conv_out = F.conv1d(
                qkv.transpose(1, 2),
                weight=weight,
                bias=None,
                padding=self.conv1d.padding[0],
                groups=groups,
            )
            return F.silu(conv_out[:, :, :seq_len].transpose(1, 2))
        raise NotImplementedError("GatedDeltaNet packed THD requires FLA causal conv.")

    def _gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if _HAS_FLA and (cu_seqlens is not None or not self.deterministic):
            return _fla_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )
        return torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    @staticmethod
    def _packed_cu_seqlens(packed_seq_params) -> torch.Tensor:
        cu_seqlens = (
            packed_seq_params.cu_seqlens_q_padded
            if getattr(packed_seq_params, "cu_seqlens_q_padded", None) is not None
            else packed_seq_params.cu_seqlens_q
        )
        if cu_seqlens is None:
            raise ValueError(
                "packed_seq_params must carry cu_seqlens_q for CP GatedDeltaNet."
            )
        return cu_seqlens

    def _split_proj(self, qkvzba: torch.Tensor, cp_div: int):
        qk = self.qk_dim_local // cp_div
        vd = self.v_dim_local // cp_div
        nvh = self.num_v_heads_local // cp_div
        nkh = self.num_k_heads_local // cp_div
        q, k, v, z, b, a = qkvzba.split([qk, qk, vd, vd, nvh, nvh], dim=-1)
        batch, seq_len = qkvzba.shape[:2]
        return (
            q.reshape(batch, seq_len, nkh, self.dk),
            k.reshape(batch, seq_len, nkh, self.dk),
            v.reshape(batch, seq_len, nvh, self.dv),
            z.reshape(batch, seq_len, nvh, self.dv),
            b.reshape(batch, seq_len, nvh),
            a.reshape(batch, seq_len, nvh),
        )

    @jit_fuser
    def _prepare_qkv(
        self, qkv: torch.Tensor, gate, beta, alpha, batch: int, seq_len: int, cp_div: int
    ):
        qk = self.qk_dim_local // cp_div
        nkh = self.num_k_heads_local // cp_div
        query_key, value = qkv.split([2 * qk, self.v_dim_local // cp_div], dim=-1)
        query_key = query_key.reshape(batch, seq_len, 2 * nkh, self.dk)
        value = value.reshape(batch, seq_len, self.num_v_heads_local // cp_div, self.dv)
        query_key = self._l2norm(query_key.contiguous())
        query, key = query_key.split(nkh, dim=2)
        if self.v_heads_per_k_head > 1:
            query = query.repeat_interleave(self.v_heads_per_k_head, dim=2)
            key = key.repeat_interleave(self.v_heads_per_k_head, dim=2)
        return (
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            gate.contiguous(),
            beta.contiguous(),
            alpha.contiguous(),
        )

    @staticmethod
    @jit_fuser
    def _l2norm(x: torch.Tensor) -> torch.Tensor:
        return l2norm(x)

    @staticmethod
    @jit_fuser
    def _compute_g_and_beta(A_log, dt_bias, alpha, beta):
        g = -A_log.exp() * F.softplus(alpha.float() + dt_bias)
        return g, beta.sigmoid()

    @jit_fuser
    def _apply_gated_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.norm(x)
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * F.silu(gate.float())
        return y.to(x_dtype)


__all__ = ["GatedDeltaNet"]
