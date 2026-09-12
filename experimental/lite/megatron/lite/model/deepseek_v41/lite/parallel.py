# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engram placement over existing global ranks, not another parallel axis."""


def _interval(size, parts, index):
    width, extra = divmod(size, parts)
    begin = index * width + min(index, extra)
    return begin, begin + width + (index < extra)


class EngramLayout:
    """Explicit [replica][row shard] global rank map.

    Callers derive this map from the existing dense/expert decomposition.
    Every replica contains the complete logical table. Columns identify equal
    row intervals; optimizer authority partitions that interval across replicas.
    Aliases (including PP shadows) resolve to one logical parameter name.
    This describes state placement; it does not perform optimizer collectives.
    """

    def __init__(self, rows, rank_groups, *, world_size, aliases=None):
        self.rank_groups = tuple(tuple(group) for group in rank_groups)
        if rows < 0 or not self.rank_groups or not self.rank_groups[0]:
            raise ValueError("Require nonnegative rows and nonempty rank groups")
        width = len(self.rank_groups[0])
        ranks = [rank for group in self.rank_groups for rank in group]
        if (
            any(len(group) != width for group in self.rank_groups)
            or len(set(ranks)) != len(ranks)
            or any(
                not isinstance(rank, int) or rank < 0 or rank >= world_size
                for rank in ranks
            )
        ):
            raise ValueError(
                "Rank map must be rectangular with unique existing global ranks"
            )
        self.rows = rows
        self.row_intervals = tuple(_interval(rows, width, i) for i in range(width))
        self.replica_groups = tuple(zip(*self.rank_groups))
        self.aliases = dict(aliases or {})
        for name in self.aliases:
            self.canonical_name(name)

    @property
    def boundaries(self):
        return (0,) + tuple(end for _, end in self.row_intervals)

    def canonical_name(self, name):
        seen = set()
        while name in self.aliases:
            if name in seen:
                raise ValueError("Cyclic parameter alias")
            seen.add(name)
            name = self.aliases[name]
        return name

    def coordinates(self, rank):
        for replica, group in enumerate(self.rank_groups):
            if rank in group:
                return replica, group.index(rank)
        raise ValueError("Rank does not participate in this Engram table")

    def owner(self, row, *, replica=0):
        if not 0 <= row < self.rows or not 0 <= replica < len(self.rank_groups):
            raise ValueError("Invalid logical row or replica")
        for shard, (begin, end) in enumerate(self.row_intervals):
            if begin <= row < end:
                return self.rank_groups[replica][shard]
        raise AssertionError("Uncovered row")

    def canonical_owner(self, rank):
        return self.rank_groups[0][self.coordinates(rank)[1]]

    def optimizer_interval(self, rank):
        replica, shard = self.coordinates(rank)
        begin, end = self.row_intervals[shard]
        local_begin, local_end = _interval(end - begin, len(self.rank_groups), replica)
        return begin + local_begin, begin + local_end

    def create_groups(self):
        """All WORLD ranks must call in the same order, including nonmembers.

        ProcessGroup ranks are sorted by PyTorch. Reject nonascending row maps
        here rather than silently permuting row ownership in lookup routing.
        Pure placement/inspection still permits arbitrary maps.
        """
        import torch.distributed as dist

        if any(
            tuple(sorted(group)) != tuple(group)
            for group in (*self.rank_groups, *self.replica_groups)
        ):
            raise ValueError(
                "Collective row and replica groups require ascending global ranks"
            )
        lookup = [dist.new_group(list(group)) for group in self.rank_groups]
        replicas = [dist.new_group(sorted(group)) for group in self.replica_groups]
        return lookup, replicas

    def create_optimizer_group(self):
        """All WORLD ranks call after create_groups in the same table order.

        Replica-owned momentum pieces partition the logical rows once; their
        statistics span all existing table ranks, not one physical replica.
        Reuse this group for tables with the same rank map.
        """
        import torch.distributed as dist

        return dist.new_group(
            sorted(rank for group in self.rank_groups for rank in group)
        )


class PackedContextLayout:
    """Full unpadded batch to the same padded contiguous CP layout as replay."""

    def __init__(self, batch, *, cp_size, cp_rank, cp_group=None, tp_size=1):
        import torch
        from megatron.lite.primitive.parallel.cp import contiguous_slice_for_cp
        from megatron.lite.primitive.parallel.thd import thd_pack_meta

        if cp_size < 1 or not 0 <= cp_rank < cp_size:
            raise ValueError('Invalid context parallel rank')
        if batch.input_ids.ndim != 1 or batch.total_tokens != batch.input_ids.numel():
            raise ValueError('Context input must be the full unpadded packed batch')
        if not batch.seq_lens.numel() or (batch.seq_lens <= 0).any():
            raise ValueError('Context sequences must be nonempty')
        self.batch, self.cp_size, self.cp_rank, self.cp_group = (
            batch,
            cp_size,
            cp_rank,
            cp_group,
        )
        self.meta = thd_pack_meta(
            batch.seq_lens,
            tp_size=tp_size,
            cp_size=cp_size,
            cp_group=cp_group,
            contiguous=True,
        )
        self.lengths = batch.seq_lens.tolist()
        self.boundaries = self.meta.cu_seqlens_padded.tolist()
        self.width = self.boundaries[-1] // cp_size
        indices = torch.full(
            (self.boundaries[-1],), -1, dtype=torch.long, device=batch.input_ids.device
        )
        offset = 0
        for start, length in zip(self.boundaries, self.lengths):
            indices[start : start + length] = torch.arange(
                offset, offset + length, device=indices.device
            )
            offset += length
        self.local_token_indices = contiguous_slice_for_cp(
            indices, cp_rank, cp_size, seq_dim=0
        )
        # This is the actual model input; padding remains explicitly distinguishable.
        self.local_input_ids = batch.input_ids[self.local_token_indices.clamp_min(0)]
        self.local_input_ids = self.local_input_ids.masked_fill(
            self.local_token_indices < 0, 0
        )

    def sequence(self, index):
        return ContextSequence(self, index)


class ContextSequence:
    def __init__(self, layout, index):
        import torch

        self.layout, self.index = layout, index
        self.length = layout.lengths[index]
        start = layout.boundaries[index]
        self.ranges = tuple(
            (
                max(0, min(self.length, rank * layout.width - start)),
                max(0, min(self.length, (rank + 1) * layout.width - start)),
            )
            for rank in range(layout.cp_size)
        )
        self.begin, self.end = self.ranges[layout.cp_rank]
        self.positions = torch.arange(
            self.begin, self.end, device=layout.batch.input_ids.device
        )
        self.local_offset = (
            max(start, layout.cp_rank * layout.width) - layout.cp_rank * layout.width
        )

    def input_ids(self):
        size = self.end - self.begin
        return self.layout.local_input_ids[
            self.local_offset : self.local_offset + size
        ][None]

    def slice(self, full):
        return full[:, self.begin : self.end]

    def gather(self, local, *, replicated_loss=False):
        """Gather variable valid-token pieces; KV backward sums consumer gradients.

        The final replicated logits use a mean in backward because every CP rank
        evaluates the same full objective. Intermediate KV uses the default sum.
        """
        import torch
        import torch.distributed as dist
        from torch.distributed.nn.functional import all_gather

        if local.shape[1] != self.end - self.begin:
            raise ValueError(
                'Context tensor does not match its global sequence interval'
            )
        if self.layout.cp_size == 1:
            return local
        if self.layout.cp_group is None:
            raise ValueError('Context gathering requires the actual process group')
        width = max(end - begin for begin, end in self.ranges)
        padded = torch.cat(
            (
                local,
                local.new_zeros(
                    (local.shape[0], width - local.shape[1], *local.shape[2:])
                ),
            ),
            dim=1,
        ).contiguous()
        if local.requires_grad:
            parts = all_gather(padded, group=self.layout.cp_group)
        else:
            parts = [torch.empty_like(padded) for _ in self.ranges]
            dist.all_gather(parts, padded, group=self.layout.cp_group)
        full = torch.cat(
            [part[:, : end - begin] for part, (begin, end) in zip(parts, self.ranges)],
            dim=1,
        )
        if replicated_loss and full.requires_grad:
            full = full.detach() + (full - full.detach()) / self.layout.cp_size
        return full


def context_parallel_forward(model, batch):
    import torch
    from torch.nn import functional as F

    from .block import contract_hc, expand_hc

    layout = PackedContextLayout(
        batch,
        cp_size=model.ps.cp_size,
        cp_rank=model.ps.cp_rank,
        cp_group=model.ps.cp_group,
        tp_size=model.ps.tp_size,
    )
    outputs = []
    for index in range(len(layout.lengths)):
        sequence = layout.sequence(index)
        ids = sequence.input_ids()
        embedding = model.embed(ids)
        if hasattr(model, 'residual_dtype'):
            embedding = embedding.to(model.residual_dtype)
        hidden, pre = expand_hc(embedding, model.hc_mult)
        hidden, pre = model._sequence(hidden, pre, input_ids=ids, cp_context=sequence)
        logits = F.linear(
            model.norm(contract_hc(hidden, pre)).float(), model.head.weight.float()
        )
        outputs.append(sequence.gather(logits, replicated_loss=True))
    return torch.cat(outputs, dim=1)[0]


def finalize_context_gradients(model):
    import torch.distributed as dist

    if model.ps.cp_size <= 1:
        return
    # Replicated dense owners accumulate disjoint-query losses. KV gather's
    # backward already returned remote consumer contributions to local tokens.
    for parameter in model.parameters():
        if parameter.requires_grad:
            gradient = getattr(parameter, 'main_grad', None)
            if gradient is None:
                gradient = parameter.grad
            import torch

            active = torch.tensor(int(gradient is not None), device=parameter.device)
            dist.all_reduce(active, op=dist.ReduceOp.MAX, group=model.ps.cp_group)
            if active.item():
                if gradient is None:
                    parameter.grad = torch.zeros_like(parameter)
                    gradient = parameter.grad
                    if hasattr(parameter, 'main_grad'):
                        parameter.main_grad = gradient
                dist.all_reduce(gradient, group=model.ps.cp_group)
