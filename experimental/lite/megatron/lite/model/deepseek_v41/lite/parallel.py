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

        if any(tuple(sorted(group)) != group for group in self.rank_groups):
            raise ValueError("Collective row groups require ascending global ranks")
        lookup = [dist.new_group(list(group)) for group in self.rank_groups]
        replicas = [dist.new_group(sorted(group)) for group in self.replica_groups]
        return lookup, replicas
