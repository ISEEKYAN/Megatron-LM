# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batched metadata and Lite/TE adjoints; no serving imports at module scope."""
import weakref
from collections import OrderedDict
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def _metadata(device, rows, groups):
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

    meta = LoRAKernelMeta.make(groups, rows * groups, device)
    mapping = torch.arange(groups, dtype=torch.int32, device=device).repeat_interleave(rows)
    meta.prepare_tensors(mapping)
    args = meta.meta_args(rows * groups, specialize_active_lora=True)
    ready = torch.cuda.Event() if device.type == 'cuda' else None
    if ready is not None:
        ready.record()
    return args, ready


def metadata(x, groups=1):
    if groups < 1 or x.shape[0] % groups:
        raise ValueError("Native LoRA groups require equally sized row partitions")
    args, ready = _metadata(x.device, x.shape[0] // groups, groups)
    if ready is not None:
        torch.cuda.current_stream(x.device).wait_event(ready)
    return args


@lru_cache(maxsize=32)
def _aligned_rows(counts, alignment, device):
    """One vectorized layout for every expert, including empty groups."""
    if alignment < 1 or any(n < 0 for n in counts):
        raise ValueError("Expert counts must be nonnegative and alignment positive")
    sizes = torch.tensor(counts, dtype=torch.int64, device=device)
    padded = (sizes + alignment - 1) // alignment * alignment
    starts = padded.cumsum(0) - padded
    raw_starts = sizes.cumsum(0) - sizes
    ids = torch.arange(len(counts), dtype=torch.int32, device=device).repeat_interleave(
        sizes, output_size=sum(counts)
    )
    positions = torch.arange(sum(counts), device=device) + (starts - raw_starts)[ids.long()]
    capacity = max(sum((n + alignment - 1) // alignment * alignment for n in counts), alignment)
    result = positions, ids, (starts + sizes).int(), capacity
    ready = torch.cuda.Event() if device.type == 'cuda' else None
    if ready is not None:
        ready.record()
    return result, ready


def aligned_rows(counts, alignment, device):
    result, ready = _aligned_rows(tuple(counts), alignment, torch.device(device))
    if ready is not None:
        torch.cuda.current_stream(device).wait_event(ready)
    return result


def owned_slots(routes, first_expert, local_experts):
    """Stable expert-major slots from one selection and one grouped sort."""
    local = routes.flatten() - first_expert
    positions = ((local >= 0) & (local < local_experts)).nonzero().flatten()
    owned = local[positions]
    positions = positions[owned.argsort(stable=True)]
    counts = tuple(torch.bincount(owned, minlength=local_experts).cpu().tolist())
    return positions, counts


def dense_mm(left, right):
    """Lite's TE GEMM on CUDA; CPU float accumulation is an explicit reference."""
    if not left.numel() or not right.numel():
        return left.new_zeros((left.shape[0], right.shape[1]), dtype=torch.float32)
    if not left.is_cuda:
        return left.float() @ right.float()
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    result = general_gemm(
        right.bfloat16().contiguous(),
        left.bfloat16().contiguous(),
        torch.float32,
        layout='NN',
        grad=True,
    )
    return result[0]


def grouped_mm(left, right, counts, *, wgrad=False):
    """One native TE GroupedLinear GEMM; expert iteration creates tensor views only."""
    dtype = torch.float32
    if not left.shape[0]:
        shape = (len(counts), left.shape[1], right.shape[1]) if wgrad else (0, right.shape[-1])
        return left.new_zeros(shape, dtype=dtype)
    if not left.is_cuda:
        if wgrad:
            return torch.stack(
                [a.float().T @ b.float() for a, b in zip(left.split(counts), right.split(counts))]
            )
        return torch.cat(
            [a.float() @ b.float() for a, b in zip(left.split(counts), right.unbind(0))]
        )
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    if wgrad:
        a, b = list(right.bfloat16().split(counts)), list(left.bfloat16().split(counts))
        result = left.new_zeros((len(counts), left.shape[1], right.shape[1]), dtype=dtype)
        outputs, layout, single = list(result.unbind(0)), 'NT', False
    else:
        a, b = list(right.bfloat16().unbind(0)), list(left.bfloat16().split(counts))
        result = left.new_empty((left.shape[0], right.shape[-1]), dtype=dtype)
        outputs, layout, single = [result], 'NN', True
    general_grouped_gemm(
        a,
        b,
        outputs,
        [None] * len(counts),
        dtype,
        layout=layout,
        m_splits=counts,
        single_output=single,
        grad=True,
    )
    if wgrad:
        empty = torch.tensor(counts, device=left.device) == 0
        result.masked_fill_(empty[:, None, None], 0)
    return result


_PACKED = OrderedDict()


def packed_weights(weights):
    """Batch codec once; only frozen owners can persist across forward calls.

    Core optimizer shard writes need not bump a model Tensor's version counter, so
    trainable weights deliberately bypass persistence. Frozen adapter-only bases
    are keyed by identity/storage/version; weak owners avoid retaining a model.
    """
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

    from .frozen import FrozenWeight

    if any(isinstance(w, FrozenWeight) for w in weights):
        if not all(isinstance(w, FrozenWeight) and w.kind == 'mxfp4' for w in weights):
            raise ValueError('Expert groups cannot mix native and decoded weights')
        return torch.stack([w.codes for w in weights]), torch.stack([w.scale for w in weights])
    key = tuple((id(w), w.data_ptr(), w._version, w.dtype, w.device) for w in weights)
    cacheable = not any(w.requires_grad for w in weights)
    cached = _PACKED.get(key) if cacheable else None
    if cached is not None and all(ref() is w for ref, w in zip(cached[0], weights)):
        _PACKED.move_to_end(key)
        if cached[2] is not None:
            torch.cuda.current_stream(weights[0].device).wait_event(cached[2])
        return cached[1]
    packed = quantize_mxfp4(torch.stack([w.detach() for w in weights]))
    if cacheable:
        # Remove superseded versions immediately rather than retaining stale codes.
        owner_ids = tuple(id(w) for w in weights)
        for old in list(_PACKED):
            if tuple(item[0] for item in old) == owner_ids:
                del _PACKED[old]
        ready = torch.cuda.Event() if weights[0].device.type == 'cuda' else None
        if ready is not None:
            ready.record()
        _PACKED[key] = tuple(weakref.ref(w) for w in weights), packed, ready
        if len(_PACKED) > 128:
            _PACKED.popitem(last=False)
    return packed
