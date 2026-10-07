# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Assemble an encoded local-stage stream, bounded by one borrowed payload."""
import torch
import torch.distributed as dist


def broadcast_stage_stream(weights, ps):
    """All PP ranks consume every stage; EP gathering belongs to local exporters.

    Source payloads must remain live until the consumer advances the iterator.
    CPU outputs use the process group's CUDA device for NCCL transport.
    """
    device = (
        torch.device('cuda', torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device('cpu')
    )
    for stage, source in enumerate(ps.pp_global_ranks):
        own = ps.pp_rank == stage
        iterator = iter(weights) if own else None
        while True:
            item = next(iterator, None) if own else None
            header = [
                None if item is None else (item[0], tuple(item[1].shape), item[1].dtype)
            ]
            dist.broadcast_object_list(
                header, src=source, group=ps.pp_group, device=device
            )
            if header[0] is None:
                break
            name, shape, dtype = header[0]
            value = (
                item[1].to(device).contiguous()
                if own
                else torch.empty(shape, dtype=dtype, device=device)
            )
            dist.broadcast(value, src=source, group=ps.pp_group)
            yield name, value
            del value, item
