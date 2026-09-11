# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Resident row-sharded lookup preserving published FP8 and scale bytes.

Every group member must participate in forward and, for trainable tables,
backward, including members with no requests. Request counts are host metadata;
no table or fetched value is offloaded. Group=None explicitly means local.
"""

import torch
import torch.distributed as dist


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, master, route):
        ctx.route = route
        ctx.shape = master.shape
        return route.return_rows(master[route.local_ids])

    @staticmethod
    def backward(ctx, gradient):
        route = ctx.route
        ordered = gradient[route.order].contiguous()
        received = route.exchange(ordered, route.send_counts, route.recv_counts)
        result = received.new_zeros(ctx.shape)
        result.index_add_(0, route.local_ids, received)
        return result, None


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
        result = tensor.new_empty((sum(recv_counts), *tensor.shape[1:]))
        dist.all_to_all_single(result, tensor.contiguous(), recv_counts, send_counts, group=self.group)
        return result

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
        if (len(self.boundaries) != self.size + 1 or self.boundaries[0] != 0
                or any(a > b for a, b in zip(self.boundaries, self.boundaries[1:]))):
            raise ValueError("Require monotone row boundaries matching process group size")

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
        invalid = (values.ndim != 2 or scales.ndim != 2 or values.shape[0] != rows
                or scales.shape[0] != rows or values.device != ids.device
                or scales.device != ids.device or values.element_size() != 1
                or scales.element_size() != 1 or ids.dtype != torch.int64)
        if self.group is not None:
            if ids.device.type != "cuda":
                raise ValueError("Distributed Engram lookup requires GPU-resident tensors")
            flag = torch.tensor(int(invalid), device=ids.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.group)
            invalid = bool(flag.item())
        if invalid:
            raise ValueError("Expected colocated resident byte-valued row and scale shards and int64 IDs")

    def fetch(self, values, scales, ids, master=None):
        self._validate_storage(values, scales, ids)
        if self.group is not None:
            descriptor = torch.tensor(
                [values.shape[1], scales.shape[1], int(master is not None)], device=ids.device)
            low, high = descriptor.clone(), descriptor.clone()
            dist.all_reduce(low, op=dist.ReduceOp.MIN, group=self.group)
            dist.all_reduce(high, op=dist.ReduceOp.MAX, group=self.group)
            if not torch.equal(low, high):
                raise ValueError("Lookup ranks disagree on row widths or trainability")
        if master is not None:
            invalid = (master.shape != values.shape or master.dtype != torch.float32
                       or master.device != values.device)
            flag = torch.tensor(int(invalid), device=ids.device)
            if self.group is not None:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.group)
            if flag.item():
                raise ValueError("Require resident FP32 master matching local row shard")
        route = self.route(ids)
        raw = route.return_rows(values.view(torch.uint8)[route.local_ids])
        scale = route.return_rows(scales.view(torch.uint8)[route.local_ids])
        shape = ids.shape
        raw = raw.view(values.dtype).reshape(*shape, values.shape[1])
        scale = scale.view(scales.dtype).reshape(*shape, scales.shape[1])
        floating = None
        if master is not None:
            floating = _GatherRows.apply(master, route).reshape(*shape, values.shape[1])
        return raw, scale, floating

    def raw_rows(self, values, scales, ids):
        raw, scale, _ = self.fetch(values, scales, ids)
        return raw, scale
