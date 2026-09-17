"""Packed Mamba primitives with explicit boundaries and mlite CP collectives."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from megatron.lite.model.nemotron_h.functional import visible_forward
from megatron.lite.primitive.parallel.cp import all_to_all_hidden_shards


@dataclass(frozen=True)
class SSMMeta:
    """Global packed request boundaries; every request starts from zero SSM state."""

    boundaries: tuple[int, ...]
    chunk_size: int = 128

    def __post_init__(self):
        if (
            self.chunk_size <= 0
            or len(self.boundaries) < 2
            or self.boundaries[0] != 0
            or any(a >= b for a, b in zip(self.boundaries, self.boundaries[1:]))
        ):
            raise ValueError("SSM boundaries must start at zero and strictly increase")

    def chunks(self):
        starts, last, sequence_ids = [], [], []
        for sequence, (start, end) in enumerate(
            zip(self.boundaries, self.boundaries[1:])
        ):
            offsets = list(range(start, end, self.chunk_size))
            starts.extend(offsets)
            last.append(len(starts) - 1)
            sequence_ids.extend([sequence] * len(offsets))
        return (*starts, self.boundaries[-1]), tuple(last), tuple(sequence_ids)

    def validate_tokens(self, tokens):
        if tokens != self.boundaries[-1]:
            raise ValueError("Packed SSM token count disagrees with request boundaries")


def exchange_sequence_channels(x, group, *, reverse=False):
    """[local T, all H, ...] <-> [global T, local H, ...], with native autograd."""
    size = dist.get_world_size(group) if group is not None else 1
    if size == 1:
        return x
    scatter, gather = (0, 1) if reverse else (1, 0)
    if x.shape[scatter] % size:
        raise ValueError("Mamba CP exchange requires a divisible scatter dimension")
    parts = list(x.chunk(size, dim=scatter))
    return torch.cat(all_to_all_hidden_shards(parts, group), dim=gather)


def packed_conv(x, weight, bias, meta: SSMMeta):
    """Causal SiLU convolution, x[T,C], weight[C,K]; no cross-request history."""
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        causal_conv1d_fn as native_conv,
    )

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    native_conv = getattr(native_conv, "__wrapped__", native_conv)
    meta.validate_tokens(x.shape[0])

    def visible(x, weight, *bias_arg):
        sequences = len(meta.boundaries) - 1
        states = x.new_zeros(sequences + 1, x.shape[1], weight.shape[1] - 1)
        return causal_conv1d_fn(
            x.T,
            weight,
            bias_arg[0] if bias_arg else None,
            states,
            torch.tensor(meta.boundaries, dtype=torch.int32, device=x.device),
            cache_indices=torch.arange(
                1, sequences + 1, dtype=torch.int32, device=x.device
            ),
            has_initial_state=torch.zeros(sequences, dtype=torch.bool, device=x.device),
            activation="silu",
        ).T

    def native(x, weight, *bias_arg):
        return torch.cat(
            [
                native_conv(
                    x[a:b].T.unsqueeze(0),
                    weight,
                    bias_arg[0] if bias_arg else None,
                    activation="silu",
                )
                .squeeze(0)
                .T
                for a, b in zip(meta.boundaries, meta.boundaries[1:])
            ]
        )

    inputs = (x, weight) if bias is None else (x, weight, bias)
    return visible_forward(visible, native, *inputs)


def packed_scan(x, dt, A, B, C, D, dt_bias, meta: SSMMeta):
    """Cache-free SSD, x[T,H,P], B/C[T,G,N]; inference forward and native VJP."""
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        mamba2_chunk_scan,
    )

    from vllm.model_executor.layers.mamba.ops.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )

    native_scan = getattr(mamba2_chunk_scan, "__wrapped__", mamba2_chunk_scan)
    meta.validate_tokens(x.shape[0])

    def visible(x, dt, A, B, C, D, dt_bias):
        chunks, last, sequence_ids = meta.chunks()

        def i32(values):
            return torch.tensor(values, dtype=torch.int32, device=x.device)

        output = torch.empty_like(x)
        mamba_chunk_scan_combined_varlen(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=meta.chunk_size,
            cu_seqlens=i32(meta.boundaries),
            cu_chunk_seqlens=i32(chunks),
            last_chunk_indices=i32(last),
            seq_idx=i32(sequence_ids),
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            out=output,
            state_dtype=torch.float32,
        )
        return output

    def native(x, dt, A, B, C, D, dt_bias):
        return torch.cat(
            [
                native_scan(
                    x[a:b].unsqueeze(0),
                    dt[a:b].unsqueeze(0),
                    A,
                    B[a:b].unsqueeze(0),
                    C[a:b].unsqueeze(0),
                    chunk_size=meta.chunk_size,
                    D=D,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                    dt_limit=(0.0, float("inf")),
                ).squeeze(0)
                for a, b in zip(meta.boundaries, meta.boundaries[1:])
            ]
        )

    return visible_forward(visible, native, x, dt, A, B, C, D, dt_bias)


class MambaMixer(torch.nn.Module):
    """Native TP1 Mamba2 mixer; weights retain HF names for lossless checkpoint IO."""

    def __init__(self, config, parallel_state, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        from megatron.lite.model.nemotron_h.functional import GatedRMSNorm

        if parallel_state.tp_size != 1:
            raise NotImplementedError("Nemotron alignment currently targets TP1")
        if config.mamba_hidden_act != "silu":
            raise ValueError("Mamba visible convolution requires SiLU")
        if config.n_groups % parallel_state.cp_size:
            raise NotImplementedError("Mamba CP requires whole SSM groups per rank")
        self.config, self.ps = config, parallel_state
        factory = dict(device=device, dtype=dtype)
        self.in_proj = torch.nn.Linear(
            config.hidden_size,
            config.mamba_in_proj_size,
            bias=config.use_bias,
            **factory,
        )
        self.conv1d = torch.nn.Conv1d(
            config.mamba_conv_dim,
            config.mamba_conv_dim,
            config.conv_kernel,
            groups=config.mamba_conv_dim,
            bias=config.use_conv_bias,
            **factory,
        )
        self.dt_bias = torch.nn.Parameter(
            torch.empty(config.mamba_num_heads, **factory)
        )
        self.A_log = torch.nn.Parameter(torch.empty(config.mamba_num_heads, **factory))
        self.D = torch.nn.Parameter(torch.empty(config.mamba_num_heads, **factory))
        self.norm = GatedRMSNorm(
            config.mamba_inner_size,
            config.mamba_inner_size // config.n_groups,
            config.layer_norm_epsilon,
            **factory,
        )
        self.out_proj = torch.nn.Linear(
            config.mamba_inner_size, config.hidden_size, bias=config.use_bias, **factory
        )

    def forward(self, hidden, meta: SSMMeta):
        from megatron.lite.model.nemotron_h.functional import linear
        from megatron.lite.primitive.parallel.cp import get_parameter_local_cp_headwise

        c, ps = self.config, self.ps
        meta.validate_tokens(hidden.shape[0] * ps.cp_size)
        if meta.chunk_size != c.chunk_size:
            raise ValueError("SSM metadata chunk size differs from model config")
        projected = linear(hidden, self.in_proj.weight, self.in_proj.bias)
        gate, xbc, dt = projected.split(
            (c.mamba_inner_size, c.mamba_conv_dim, c.mamba_num_heads), dim=-1
        )
        sections = [
            c.mamba_inner_size,
            c.n_groups * c.ssm_state_size,
            c.n_groups * c.ssm_state_size,
        ]
        xbc = torch.cat(
            [
                exchange_sequence_channels(t, ps.cp_group)
                for t in xbc.split(sections, dim=-1)
            ],
            dim=-1,
        )

        def local_parameter(p, split_sections=None):
            if p is None:
                return None
            return get_parameter_local_cp_headwise(
                p, 0, ps.cp_size, ps.cp_rank, split_sections=split_sections
            )

        conv = packed_conv(
            xbc,
            local_parameter(self.conv1d.weight[:, 0], sections),
            local_parameter(self.conv1d.bias, sections),
            meta,
        )
        x, B, C = conv.split([size // ps.cp_size for size in sections], dim=-1)
        tokens = conv.shape[0]
        local_heads, local_groups = (
            c.mamba_num_heads // ps.cp_size,
            c.n_groups // ps.cp_size,
        )
        dt = exchange_sequence_channels(dt, ps.cp_group)
        scanned = packed_scan(
            x.reshape(tokens, local_heads, c.mamba_head_dim),
            dt,
            -torch.exp(local_parameter(self.A_log).float()),
            B.reshape(tokens, local_groups, c.ssm_state_size),
            C.reshape(tokens, local_groups, c.ssm_state_size),
            local_parameter(self.D),
            local_parameter(self.dt_bias),
            meta,
        )
        scanned = exchange_sequence_channels(scanned, ps.cp_group, reverse=True)
        normalized = self.norm(
            scanned.reshape(hidden.shape[0], c.mamba_inner_size), gate
        )
        return linear(normalized, self.out_proj.weight, self.out_proj.bias)
