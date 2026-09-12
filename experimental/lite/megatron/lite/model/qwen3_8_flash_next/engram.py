# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

MULTIPLIERS = (23703573157769, 20109073645365, 8052911324071)
PRIMES = (
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
)


class Qwen3_8_FlashNextNGramEmbedding(nn.Module):
    def __init__(
        self,
        ngram_embedding,
        *,
        ngram_size=3,
        heads_per_ngram=8,
        eos_token_id=248044,
        layer_multipliers=MULTIPLIERS,
        ngram_heads_vocab_sizes=PRIMES,
        ngram_heads_offsets=None
    ):
        super().__init__()
        self.ngram_embedding, self.eos_token_id = ngram_embedding, eos_token_id
        if (
            ngram_size != 3
            or heads_per_ngram != 8
            or len(layer_multipliers) != 3
            or len(ngram_heads_vocab_sizes) != 16
        ):
            raise ValueError('HASH_RELEASE_LAYOUT')
        self.register_buffer('layer_multipliers', torch.tensor(layer_multipliers))
        primes = torch.tensor(ngram_heads_vocab_sizes)
        self.register_buffer('ngram_heads_vocab_sizes', primes)
        offsets = (
            primes.cumsum(0) - primes
            if ngram_heads_offsets is None
            else torch.tensor(ngram_heads_offsets)
        )
        self.register_buffer('ngram_heads_offsets', offsets)

    def hash_ids(self, ids, cu_seqlens=None):
        if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('HASH_IDS_INTEGER_BS')
        # Promote before history construction and every multiply/XOR operation.
        ids = ids.long()
        b, s = ids.shape
        pos = torch.arange(s, device=ids.device).expand(b, -1)
        eos = torch.where(ids == self.eos_token_id, pos, -1).cummax(1).values
        starts = F.pad(eos[:, :-1] + 1, (1, 0))
        if cu_seqlens is not None:
            if (
                b != 1
                or cu_seqlens.ndim != 1
                or cu_seqlens.numel() < 2
                or int(cu_seqlens[0]) != 0
                or int(cu_seqlens[-1]) != s
                or not bool((cu_seqlens.diff() > 0).all())
            ):
                raise ValueError('HASH_PACKED_BOUNDARIES')
            doc = torch.bucketize(pos, cu_seqlens[1:], right=True)
            starts = torch.maximum(starts, cu_seqlens[doc])
        history = []
        for shift in range(3):
            values = ids.gather(1, (pos - shift).clamp_min(0))
            history.append(
                torch.where(pos - shift >= starts, values, self.eos_token_id)
            )
        mixed = (
            history[0] * self.layer_multipliers[0]
            ^ history[1] * self.layer_multipliers[1]
        )
        tri = mixed ^ history[2] * self.layer_multipliers[2]
        return (
            torch.cat(
                [
                    mixed.unsqueeze(-1) % self.ngram_heads_vocab_sizes[:8],
                    tri.unsqueeze(-1) % self.ngram_heads_vocab_sizes[8:],
                ],
                -1,
            )
            + self.ngram_heads_offsets
        )

    def forward(self, input_ids, cu_seqlens=None):
        if not callable(self.ngram_embedding):
            raise RuntimeError('ROW_LOOKUP_GATHER_ROWS_REQUIRED')
        return self.ngram_embedding(self.hash_ids(input_ids, cu_seqlens)).flatten(-2)

    def _hash_input_ids(self, input_ids):
        return self.hash_ids(input_ids)


@dataclass(frozen=True)
class Qwen3_8_FlashNextEngramTableConfig:
    num_embeddings: int
    embedding_dim: int
    initializer_range: float = 0.02

    def build(self, *, process_group, device, dtype):
        return Qwen3_8_FlashNextOwnerShardedEmbedding(
            self, process_group=process_group, device=device, dtype=dtype
        )


class Qwen3_8_FlashNextOwnerShardedEmbedding(nn.Module):
    def __init__(self, config, *, process_group, device, dtype):
        super().__init__()
        try:
            from megatron.lite.primitive.modules.engram_lookup import RowLookup
        except ModuleNotFoundError as error:
            if error.name != 'megatron.lite.primitive.modules.engram_lookup':
                raise
            RowLookup = None

        size = (
            1
            if process_group is None
            else torch.distributed.get_world_size(process_group)
        )
        rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
        boundaries = [config.num_embeddings * r // size for r in range(size + 1)]
        self.lookup = (
            None if RowLookup is None else RowLookup(boundaries, process_group)
        )
        self.global_row_start, self.global_row_end = boundaries[rank : rank + 2]
        self.weight = nn.Parameter(
            torch.empty(
                self.global_row_end - self.global_row_start,
                config.embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.initializer_range = config.initializer_range
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.initializer_range)

    def forward(self, global_ids):
        gather = getattr(self.lookup, 'gather_rows', None)
        if not callable(gather):
            raise RuntimeError('ROW_LOOKUP_GATHER_ROWS_REQUIRED')
        return gather(self.weight, global_ids)


class Qwen3_8_FlashNextPLELayer(nn.Module):
    def __init__(
        self,
        ple_embedding,
        *,
        hidden_size,
        hc_count,
        ple_embed_dim,
        backend=None,
        dtype=torch.bfloat16,
        conv_kernel_size=4,
        rms_norm_eps=1e-6
    ):
        super().__init__()
        if backend is not None:
            raise ValueError('PLE_BACKEND_UNSUPPORTED')
        self.ple_embedding = ple_embedding
        self.hidden_size, self.hc_count, self.eps = hidden_size, hc_count, rms_norm_eps
        width = hidden_size * hc_count
        self.key_proj = nn.Linear(ple_embed_dim, width, bias=False, dtype=dtype)
        self.value_proj = nn.Linear(ple_embed_dim, hidden_size, bias=False, dtype=dtype)
        for name in ('norm_key', 'norm_query', 'norm_conv'):
            module = nn.Module()
            module.register_parameter(
                'weight', nn.Parameter(torch.zeros(width, dtype=dtype))
            )
            setattr(self, name, module)
        self.conv1d = nn.Conv1d(
            width,
            width,
            conv_kernel_size,
            groups=width,
            dilation=3,
            bias=False,
            dtype=dtype,
        )
        nn.init.zeros_(self.conv1d.weight)

    def _norm(self, x, module):
        x = x.float()
        return (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
            * (1 + module.weight.float().reshape(self.hc_count, self.hidden_size))
        )

    def forward(self, hidden_states, input_ids, *, cp_context=None, cu_seqlens=None):
        from .cp import qwen3_8_flash_next_cp_left_halo

        if cp_context is None:
            embeddings = self.ple_embedding(input_ids, cu_seqlens)
        else:
            global_ids = cp_context.global_input_ids.masked_fill(
                cp_context.global_padding_mask, self.ple_embedding.eos_token_id
            )
            cu = cp_context.global_cu_seqlens
            if cu is not None and int(cu[-1]) < cp_context.global_sequence_length:
                cu = torch.cat((cu, cu.new_tensor([cp_context.global_sequence_length])))
            ids = self.ple_embedding.hash_ids(global_ids, cu)
            embeddings = self.ple_embedding.ngram_embedding(
                ids[:, cp_context.local_sequence_start : cp_context.local_sequence_end]
            ).flatten(-2)
        embeddings = embeddings.to(self.key_proj.weight.dtype)
        key = self.key_proj(embeddings).unflatten(-1, (self.hc_count, self.hidden_size))
        query = hidden_states.unflatten(-1, (self.hc_count, self.hidden_size))
        dot = (self._norm(key, self.norm_key) * self._norm(query, self.norm_query)).sum(
            -1, keepdim=True
        ) / self.hidden_size**0.5
        gate = torch.sigmoid(dot.sign() * dot.abs().clamp_min(1e-6).sqrt())
        value = gate * self.value_proj(embeddings).unsqueeze(-2)
        normalized = (
            self._norm(value, self.norm_conv).flatten(-2).to(self.conv1d.weight.dtype)
        )
        history = 3 * (self.conv1d.kernel_size[0] - 1)
        if cp_context is not None:
            start, end = cp_context.local_sequence_start, cp_context.local_sequence_end
            mask = cp_context.global_padding_mask
            normalized = normalized.masked_fill(mask[:, start:end, None], 0)
            normalized = torch.cat(
                (
                    qwen3_8_flash_next_cp_left_halo(
                        normalized, cp_context, history=history
                    ),
                    normalized,
                ),
                1,
            )
            positions = torch.arange(mask.shape[1], device=normalized.device)
            # Padding splits independent token spans, including across CP ranks.
            starts = torch.where(mask, positions + 1, 0).cummax(1).values[:, start:end]
            positions = positions[start:end]
            cu = cp_context.global_cu_seqlens
            if cu is not None:
                starts = torch.maximum(
                    starts, cu[torch.bucketize(positions, cu[1:], right=True)]
                )
            convolution = normalized.new_zeros(hidden_states.shape)
            for i in range(self.conv1d.kernel_size[0]):
                valid = (positions - history + 3 * i >= starts).unsqueeze(-1)
                convolution = (
                    convolution
                    + normalized[:, 3 * i : 3 * i + hidden_states.shape[1]]
                    * self.conv1d.weight[:, 0, i]
                    * valid
                )
            convolution = F.silu(convolution)
        elif cu_seqlens is not None:
            convolution = torch.cat(
                [
                    F.silu(
                        self.conv1d(
                            F.pad(normalized[:, a:b].transpose(1, 2), (history, 0))
                        )
                    ).transpose(1, 2)
                    for a, b in zip(cu_seqlens.tolist(), cu_seqlens.tolist()[1:])
                ],
                1,
            )
        else:
            convolution = F.silu(
                self.conv1d(F.pad(normalized.transpose(1, 2), (history, 0)))
            ).transpose(1, 2)
        return (value.flatten(-2) + convolution).to(hidden_states.dtype)
