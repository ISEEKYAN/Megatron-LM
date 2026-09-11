# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in EP agreement before a microbatch can omit backward or replay.

This checks a small, exact graph census, not token equality or arbitrary graph
isomorphism. Ranks may have different token counts. Opaque checkpoint/kernel
internals and failures before the boundary are outside this contract.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

_FIELDS = (
    'last_stage',
    'microbatch',
    'virtual_stage',
    'valid_root',
    'alltoall',
    'deepep_dispatch',
    'deepep_combine',
    'checkpoint',
    'activation_checkpoint',
)
_GRAPH_KINDS = {
    '_AllToAllBackward': 0,
    '_DeepEPDispatchBackward': 1,
    '_DeepEPCombineBackward': 2,
    'CheckpointFunctionBackward': 3,
    '_CheckpointWithoutOutputFnBackward': 4,
}


def _reachable_operations(root, ep_group):
    counts = [0] * len(_GRAPH_KINDS)
    pending = [root.grad_fn] if isinstance(root, torch.Tensor) else []
    seen = set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        name = type(node).__name__
        kind = _GRAPH_KINDS.get(name)
        if kind is not None:
            # Native AllToAll carries its process group in the autograd context.
            if name != '_AllToAllBackward' or node.group is ep_group:
                counts[kind] += 1
        pending.extend(parent for parent, _ in node.next_functions)
    return counts


def validate_ep_backward_contract(
    output: dict,
    ps,
    *,
    is_last_stage: bool,
    microbatch: int = 0,
    virtual_stage: int = 0,
) -> None:
    """All EP ranks must call before loss scaling, P2P, backward, or replay.

    Enable MEGATRON_LITE_VALIDATE_EP_BACKWARD=1 uniformly on the job. Disabled
    and EP=1 calls do no graph walk/communication. One enabled call costs one
    18-int64 MAX all-reduce (144-byte payload) and a host synchronization.
    Compare counts as integers directly; unlike routing payloads, no hash is
    necessary. This detects partial loss of visible communication nodes, even
    when the root still requires grad; it cannot see inside checkpoint replay.
    """
    if os.environ.get('MEGATRON_LITE_VALIDATE_EP_BACKWARD') != '1' or ps.ep_size <= 1:
        return
    root = output.get('loss' if is_last_stage else 'hidden_states')
    valid = (
        isinstance(root, torch.Tensor)
        and root.requires_grad
        and (not is_last_stage or root.numel() == 1)
    )
    values = [int(is_last_stage), microbatch, virtual_stage, int(valid)]
    values.extend(_reachable_operations(root, ps.ep_group))
    # Select by the group, not root.device: missing-root ranks must still join.
    device = (
        torch.device('cuda', torch.cuda.current_device())
        if dist.get_backend(ps.ep_group) == 'nccl'
        else torch.device('cpu')
    )
    bounds = torch.tensor(
        values + [-v for v in values], dtype=torch.int64, device=device
    )
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX, group=ps.ep_group)
    maximum, neg_minimum = bounds.cpu().view(2, -1).tolist()
    minimum = [-v for v in neg_minimum]
    differences = [
        f'{name}={lo}..{hi}'
        for name, lo, hi in zip(_FIELDS, minimum, maximum)
        if lo != hi
    ]
    if differences or minimum[3] != 1:
        detail = ', '.join(differences) or 'no valid differentiable training root'
        raise RuntimeError(
            f'EP backward contract failed before backward/replay: {detail}; '
            f'local_ep_rank={dist.get_rank(ps.ep_group)}, microbatch={microbatch}, '
            f'virtual_stage={virtual_stage}. Check loss/hidden usage on every EP rank. '
        )
