# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batched stable compaction of compressed and sliding-window KV slots."""
import torch
import triton
import triton.language as tl


@triton.jit
def _compact(
    Main,
    Mask,
    Out,
    Length,
    S: tl.constexpr,
    MAIN: tl.constexpr,
    WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    MM0: tl.constexpr,
    MM1: tl.constexpr,
    MM2: tl.constexpr,
    MA0: tl.constexpr,
    MA1: tl.constexpr,
    MA2: tl.constexpr,
    HAS_MAIN: tl.constexpr,
    TM: tl.constexpr,
    TW: tl.constexpr,
    TO: tl.constexpr,
):
    row = tl.program_id(0)
    batch, token = row // S, row % S
    out = tl.arange(0, TO)
    tl.store(Out + row * WIDTH + out, -1, out < WIDTH)
    tl.debug_barrier()
    count = tl.full((), 0, tl.int32)
    if HAS_MAIN:
        slot = tl.arange(0, TM)
        value = tl.load(
            Main + batch * MM0 + token * MM1 + slot * MM2, slot < TOPK, other=-1
        )
        valid = (slot < TOPK) & (value >= 0) & (value < MAIN)
        position = tl.cumsum(valid.to(tl.int32)) - 1
        count = tl.sum(valid.to(tl.int32))
        tl.store(Out + row * WIDTH + position, value, valid & (position < WIDTH))
    slot = tl.arange(0, TW)
    valid = tl.load(
        Mask + batch * MA0 + token * MA1 + slot * MA2, slot < WINDOW, other=False
    )
    position = tl.cumsum(valid.to(tl.int32)) - 1 + count
    total = count + tl.sum(valid.to(tl.int32))
    tl.device_assert(total <= WIDTH, "sparse row exceeds physical width")
    tl.store(Out + row * WIDTH + position, MAIN + slot, valid & (position < WIDTH))
    tl.store(Length + row, total)


def sparse_metadata(mask, main_indices, *, main_size, window_size, width):
    """Keep input main-slot order followed by ascending visible window slots."""
    if mask.ndim != 3 or mask.dtype != torch.bool or not mask.is_cuda:
        raise ValueError('Sparse mask must be CUDA BSN boolean')
    batch, rows, _ = mask.shape
    if not 0 <= window_size <= mask.shape[-1] or width <= 0:
        raise ValueError('Invalid sparse window or output width')
    has_main = main_indices is not None
    if has_main and (main_indices.shape[:2] != (batch, rows) or main_indices.ndim != 3):
        raise ValueError('Sparse main indices must be BST')
    topk = main_indices.shape[-1] if has_main else 0
    main = main_indices if has_main else mask
    indices = torch.empty((batch, rows, width), device=mask.device, dtype=torch.int32)
    lengths = torch.empty((batch, rows), device=mask.device, dtype=torch.int32)
    if batch * rows:
        _compact[(batch * rows,)](
            main,
            mask,
            indices,
            lengths,
            rows,
            main_size,
            window_size,
            width,
            topk,
            *(main.stride() if has_main else (0, 0, 0)),
            *mask.stride(),
            has_main,
            triton.next_power_of_2(max(topk, 1)),
            triton.next_power_of_2(max(window_size, 1)),
            triton.next_power_of_2(width),
            num_warps=4,
            debug=True,
        )
    return indices, lengths
