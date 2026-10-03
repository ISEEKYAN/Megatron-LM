# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Document execution over tensor trees with explicit token axes."""

import torch
from megatron.lite.primitive.modules.router_replay import PackedRouterReplay
from torch.utils._pytree import tree_flatten, tree_map


def packed_forward(
    sequence_forward,
    state,
    cu_seqlens,
    *,
    axes,
    output_axes=None,
    routers=(),
    recompute=False,
):
    """Call once per document, then concatenate.

    Axes mirror each tensor tree (or one integer applies to every leaf). Offsets
    describe the physical buffer: padded adapters must pass padded offsets.
    The callable owns fresh document state.
    """

    if isinstance(axes, int):
        axes = tree_map(lambda _: axes, state)
    lengths = tree_flatten(tree_map(lambda x, d: x.shape[d], state, axes))[0]
    if not lengths or len(set(lengths)) != 1:
        raise ValueError("State token axes must have equal lengths")
    total = lengths[0]
    cu = cu_seqlens.tolist()
    if (
        len(cu) < 2
        or cu[0] != 0
        or cu[-1] != total
        or any(b >= e for b, e in zip(cu, cu[1:]))
    ):
        raise ValueError("Document offsets must partition the token buffer")
    replay = PackedRouterReplay(lengths[0], routers)
    outputs = []
    for begin, end in zip(cu, cu[1:]):
        piece = tree_map(lambda x, d: x.narrow(d, begin, end - begin), state, axes)
        outputs.append(
            replay.run(sequence_forward, begin, end, piece, recompute=recompute)
        )
    replay.finish()
    dims = axes if output_axes is None else output_axes
    if isinstance(dims, int):
        dims = tree_map(lambda _: dims, outputs[0])
    return tree_map(lambda d, *xs: torch.cat(xs, dim=d), dims, *outputs)
