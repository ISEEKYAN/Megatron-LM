# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded PP export of already encoded (name, Tensor) pairs.

The bucket/header/broadcast order follows hf_weights.export_hf_weights's PP
path. Source-only iterators may contain stage-local EP/row collectives; only
advance them during that stage's turn.
"""

import torch
import torch.distributed as dist


def stream_pipeline_tensors(weights, ps, *, buffer_max_size_bytes, cpu=False):
    if type(buffer_max_size_bytes) is not int or buffer_max_size_bytes <= 0:
        raise ValueError('Pipeline export requires a positive byte budget')
    if ps.pp_size <= 1:
        yield from weights
        return
    device = (
        torch.device('cuda', torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device('cpu')
    )
    own = iter(weights)
    for source_stage, source_rank in enumerate(ps.pp_global_ranks):
        is_source = source_stage == ps.pp_rank
        while True:
            bucket = []
            if is_source:
                size = 0
                for name, tensor in own:
                    # Encoded row iterators may reuse borrowed storage on advance.
                    tensor = tensor.to(device).contiguous().clone()
                    bucket.append((name, tensor))
                    size += tensor.numel() * tensor.element_size()
                    if size >= buffer_max_size_bytes:
                        break
                header = [[(n, tuple(t.shape), t.dtype) for n, t in bucket]]
            else:
                header = [None]
            dist.broadcast_object_list(
                header, src=source_rank, group=ps.pp_group, device=device
            )
            if not header[0]:
                break
            for index, (name, shape, dtype) in enumerate(header[0]):
                tensor = (
                    bucket[index][1]
                    if is_source
                    else torch.empty(shape, dtype=dtype, device=device)
                )
                # The encoded stream includes FP8 storage; Gloo has no FP8
                # scalar type. Broadcast its exact bytes, retaining the dtype
                # in the header, rather than casting numerical values.
                dist.broadcast(
                    tensor.reshape(-1).view(torch.uint8),
                    src=source_rank,
                    group=ps.pp_group,
                )
                yield name, tensor.cpu() if cpu else tensor
                del tensor
            del bucket
