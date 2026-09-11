# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engram computation, with explicit table/projection providers.

The local table provider supports frozen FP8 and trainable FP32-master modes.
FP32 master representation is an approved port choice, not an official recipe.
Sharding and optimizer-step publication hooks belong to the distributed layer.
"""

import numpy as np
import torch
from torch import nn


def build_compressed_token_map(tokenizer):
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    mapping, keys = [], {}
    backend = tokenizer.backend_tokenizer
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        if key not in keys:
            keys[key] = len(keys)
        mapping.append(keys[key])
    return mapping, len(keys)


def hash_multipliers(layer_ids, max_ngram_size, vocab_size):
    if vocab_size < 1 or max_ngram_size < 2:
        raise ValueError("Require nonempty compressed vocabulary and ngram order >= 2")
    bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
    return torch.stack(
        [
            torch.from_numpy(
                np.random.default_rng(10007 * layer).integers(
                    0, bound, size=max_ngram_size, dtype=np.int64
                )
                * 2
                + 1
            )
            for layer in layer_ids
        ]
    )


def prime_buckets(layer_ids, max_ngram_size, heads, vocab_size):
    from sympy import nextprime

    seen, result = set(), []
    for _ in layer_ids:
        layer = []
        for _ in range(max_ngram_size - 1):
            current, sizes = vocab_size - 1, []
            for _ in range(heads):
                current = int(nextprime(current))
                while current in seen:
                    current = int(nextprime(current))
                seen.add(current)
                sizes.append(current)
            layer.append(sizes)
        result.append(layer)
    return torch.tensor(result, dtype=torch.int64)


class NgramHash(nn.Module):
    """Hash a full local batch; packed boundaries restart every lookback.

    No process-global cache: each call is independent of previous microbatches
    and safe to recompute. Distributed history transport belongs to the caller.
    """

    def __init__(self, token_map, pad_id, multipliers, primes):
        super().__init__()
        mapping = torch.as_tensor(token_map, dtype=torch.int64)
        if multipliers.ndim != 2 or primes.ndim != 3:
            raise ValueError(
                "Expected multipliers [layers,order] and primes [layers,order-1,heads]"
            )
        if primes.shape[:2] != (multipliers.shape[0], multipliers.shape[1] - 1):
            raise ValueError("Hash layout and multiplier shape mismatch")
        self.pad_id = int(mapping[pad_id])
        self.register_buffer("token_map", mapping, persistent=False)
        self.register_buffer(
            "multipliers", multipliers.to(torch.int64), persistent=False
        )
        self.register_buffer("primes", primes.to(torch.int64), persistent=False)
        flat = primes.flatten(1)
        self.register_buffer("offsets", flat.cumsum(-1) - flat, persistent=False)

    def forward(self, input_ids, token_mask=None, *, cu_seqlens=None):
        if input_ids.ndim != 2:
            raise ValueError("Expected input IDs [B,S]")
        b, length = input_ids.shape
        if token_mask is not None and (
            token_mask.shape != input_ids.shape or token_mask.dtype != torch.bool
        ):
            raise ValueError("Expected boolean token mask [B,S]")
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = compressed.masked_fill(~token_mask, -1)
        positions = torch.arange(length, device=input_ids.device).expand(b, -1)
        starts = torch.zeros_like(positions)
        if cu_seqlens is not None:
            if b != 1:
                raise ValueError("THD packing requires B=1")
            from megatron.lite.primitive.utils import packed_seq

            for begin, end in packed_seq.packed_sequence_ranges(cu_seqlens, length):
                starts[:, begin:end] = begin
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        history = []
        for shift in range(self.multipliers.shape[1]):
            source = compressed.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions - shift < starts) | (source == -1)
            history.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(history, -1).unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.multipliers.shape[1]):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, -1) + self.offsets


class EngramTable(nn.Module):
    """Resident FP8 rows with one construction-time trainability switch.

    Trainable mode uses a persistent FP32 master and identity STE on gathered
    rows. Call refresh_storage after an accepted optimizer step. Forward and
    recompute do not publish or mutate storage. No host offload is performed.
    """

    def __init__(self, weight, scale, *, trainable=False, output_dtype=torch.bfloat16):
        super().__init__()
        if (
            weight.ndim != 2
            or weight.shape[1] % 32
            or weight.dtype != torch.float8_e4m3fn
            or scale.dtype != torch.float8_e8m0fnu
            or scale.shape != (weight.shape[0], weight.shape[1] // 32)
            or scale.device != weight.device
        ):
            raise ValueError("Expected FP8 table [rows,D] and E8M0 row/block32 scales")
        self.output_dtype = output_dtype
        self.register_buffer("weight", weight.detach().clone())
        self.register_buffer("scale", scale.detach().clone())
        if trainable:
            master = weight.float() * scale.float().repeat_interleave(32, -1)
            self.master = nn.Parameter(master)
        else:
            self.register_parameter("master", None)

    def _apply(self, fn, recurse=True):
        self.output_dtype = fn(
            torch.empty(0, dtype=self.output_dtype, device=self.weight.device)
        ).dtype

        # A parent .bfloat16() must not widen FP8 storage or round the master.
        # Probe only the destination device/dtype, without a lossy round-trip.
        def preserve_dtype(tensor):
            probe = fn(torch.empty(0, dtype=tensor.dtype, device=tensor.device))
            if probe.dtype == tensor.dtype:
                return fn(tensor)
            return tensor.to(device=probe.device)

        return super()._apply(preserve_dtype, recurse=recurse)

    def forward(self, ids):
        # Byte indexing works for FP8 on CPU as well as CUDA; only fetched rows
        # are dequantized, so frozen execution never materializes a full master.
        rows = self.weight.view(torch.uint8)[ids].view(self.weight.dtype).float()
        scales = self.scale.view(torch.uint8)[ids].view(self.scale.dtype).float()
        decoded = rows * scales.repeat_interleave(32, -1)
        if self.master is not None:
            floating = self.master[ids]
            decoded = floating + (decoded - floating).detach()
        return decoded.to(self.output_dtype)

    @torch.no_grad()
    def refresh_storage(self):
        if self.master is None:
            return
        from megatron.lite.primitive.quantization import block_fp8

        weight, scale = block_fp8.quantize_block_fp8(
            self.master, (1, 32), scale_format="e8m0"
        )
        self.weight.copy_(weight)
        self.scale.copy_(scale)


class Engram(nn.Module):
    def __init__(self, hidden_size, copies, embedding, projection, *, eps=1e-6):
        super().__init__()
        self.dim = hidden_size
        self.copies = copies
        self.eps = eps
        self.embed = embedding
        self.wkv = projection
        self.q_weight = nn.Parameter(torch.ones(copies, hidden_size))
        self.k_weight = nn.Parameter(torch.ones(copies, hidden_size))

    def forward(self, hidden, hash_ids, token_mask=None):
        if hidden.ndim != 4 or hidden.shape[-2:] != (self.copies, self.dim):
            raise ValueError("Expected residual stream [B,S,HC,D]")
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.copies * self.dim, self.dim], -1)
        key = key.float().unflatten(-1, (self.copies, self.dim))
        h = hidden.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (
            (h * key * self.q_weight.float() * self.k_weight.float()).sum(-1)
            * rstd
            * self.dim**-0.5
        )
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
