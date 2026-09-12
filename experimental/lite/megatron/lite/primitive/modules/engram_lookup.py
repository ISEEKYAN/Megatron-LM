# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Resident row-sharded lookup preserving published FP8 and scale bytes.

Every group member must participate in forward and, for trainable tables,
backward, including members with no requests. Request counts are host metadata;
no table or fetched value is offloaded. Group=None explicitly means local.
"""

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from .moe import _AllToAll


class _Route:
    def __init__(self, group, order, local_ids, send_counts, recv_counts):
        self.group = group
        self.order = order
        self.local_ids = local_ids
        self.send_counts = send_counts
        self.recv_counts = recv_counts

    def exchange(self, tensor, send_counts, recv_counts):
        if self.group is None:
            return tensor
        return _AllToAll.apply(tensor, send_counts, recv_counts, self.group)

    def return_rows(self, rows):
        ordered = self.exchange(rows, self.recv_counts, self.send_counts)
        result = torch.empty_like(ordered)
        result[self.order] = ordered
        return result


class RowLookup:
    def __init__(self, boundaries, group=None):
        self.boundaries = tuple(boundaries)
        self.group = group
        self.size = 1 if group is None else dist.get_world_size(group)
        self.rank = 0 if group is None else dist.get_rank(group)
        if (
            len(self.boundaries) != self.size + 1
            or self.boundaries[0] != 0
            or any(a > b for a, b in zip(self.boundaries, self.boundaries[1:]))
        ):
            raise ValueError(
                "Require monotone row boundaries matching process group size"
            )

    def route(self, ids):
        if ids.dtype != torch.int64:
            raise ValueError("Row IDs must be int64")
        if self.group is not None and ids.device.type != 'cuda':
            raise ValueError("Distributed Engram lookup requires GPU-resident tensors")
        flat = ids.reshape(-1)
        invalid = ((flat < 0) | (flat >= self.boundaries[-1])).any().to(torch.int32)
        if self.group is not None:
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=self.group)
        if invalid.item():
            raise ValueError("Row ID outside logical table on a lookup group member")
        cuts = torch.tensor(self.boundaries[1:-1], device=ids.device, dtype=torch.int64)
        owners = torch.bucketize(flat, cuts, right=True)
        order = torch.argsort(owners, stable=True)
        counts = torch.bincount(owners, minlength=self.size)
        received = torch.empty_like(counts)
        if self.group is None:
            received.copy_(counts)
        else:
            dist.all_to_all_single(received, counts, group=self.group)
        # Only small per-rank split metadata crosses to the host.
        send_counts, recv_counts = counts.tolist(), received.tolist()
        route = _Route(self.group, order, None, send_counts, recv_counts)
        routed = route.exchange(flat[order], send_counts, recv_counts)
        route.local_ids = routed - self.boundaries[self.rank]
        return route

    def _validate_storage(self, values, scales, ids):
        rows = self.boundaries[self.rank + 1] - self.boundaries[self.rank]
        invalid = (
            values.ndim != 2
            or scales.ndim != 2
            or values.shape[0] != rows
            or scales.shape[0] != rows
            or values.device != ids.device
            or scales.device != ids.device
            or values.element_size() != 1
            or scales.element_size() != 1
            or ids.dtype != torch.int64
        )
        if self.group is not None:
            if ids.device.type != "cuda":
                raise ValueError(
                    "Distributed Engram lookup requires GPU-resident tensors"
                )
            flag = torch.tensor(int(invalid), device=ids.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.group)
            invalid = bool(flag.item())
        if invalid:
            raise ValueError(
                "Expected colocated resident byte-valued row and scale shards and int64 IDs"
            )

    def fetch(self, values, scales, ids, master=None):
        self._validate_storage(values, scales, ids)
        if self.group is not None:
            descriptor = torch.tensor(
                [values.shape[1], scales.shape[1], int(master is not None)],
                device=ids.device,
            )
            low, high = descriptor.clone(), descriptor.clone()
            dist.all_reduce(low, op=dist.ReduceOp.MIN, group=self.group)
            dist.all_reduce(high, op=dist.ReduceOp.MAX, group=self.group)
            if not torch.equal(low, high):
                raise ValueError("Lookup ranks disagree on row widths or trainability")
        if master is not None:
            invalid = (
                master.shape != values.shape
                or master.dtype != torch.float32
                or master.device != values.device
            )
            flag = torch.tensor(int(invalid), device=ids.device)
            if self.group is not None:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.group)
            if flag.item():
                raise ValueError(
                    "Require resident FP32 master matching local row shard"
                )
        route = self.route(ids)
        raw = route.return_rows(values.view(torch.uint8)[route.local_ids])
        scale = route.return_rows(scales.view(torch.uint8)[route.local_ids])
        shape = ids.shape
        raw = raw.view(values.dtype).reshape(*shape, values.shape[1])
        scale = scale.view(scales.dtype).reshape(*shape, scales.shape[1])
        floating = None
        if master is not None:
            floating = route.return_rows(master[route.local_ids]).reshape(
                *shape, values.shape[1]
            )
        return raw, scale, floating

    def raw_rows(self, values, scales, ids):
        raw, scale, _ = self.fetch(values, scales, ids)
        return raw, scale


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
        rows, scales, floating = self.lookup_fp8(ids)
        decoded = rows.float() * scales.float().repeat_interleave(32, -1)
        if floating is not None:
            decoded = floating + (decoded - floating).detach()
        return decoded.to(self.output_dtype)

    def lookup_fp8(self, ids):
        rows = self.weight.view(torch.uint8)[ids].view(self.weight.dtype)
        scales = self.scale.view(torch.uint8)[ids].view(self.scale.dtype)
        master = None if self.master is None else self.master[ids]
        return rows, scales, master

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


class ShardedEngramTable(EngramTable):
    """Engram provider with resident local rows and collective request routing.

    Replica gradient reduction and optimizer-state sharding are step operations,
    outside lookup. All ranks in the row group must execute backward together.
    """

    def __init__(
        self, weight, scale, lookup, *, trainable=False, output_dtype=torch.bfloat16
    ):
        super().__init__(weight, scale, trainable=trainable, output_dtype=output_dtype)
        expected = lookup.boundaries[lookup.rank + 1] - lookup.boundaries[lookup.rank]
        if weight.shape[0] != expected:
            raise ValueError("Table rows do not match lookup ownership interval")
        self.lookup = lookup

    def lookup_fp8(self, ids):
        return self.lookup.fetch(self.weight, self.scale, ids, self.master)


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
