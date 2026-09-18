# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Collect and unpack routes for contiguous pipeline-owned layer intervals."""
import torch
from megatron.lite.primitive.ckpt.hf_weights import allgather_concat
from megatron.lite.primitive.parallel.state import ParallelState


def router_replay_roots(chunk, *, model_name="Model"):
    """E stages retain global layer slots; absent/nonlocal slots contain None."""
    while hasattr(chunk, 'module'):
        chunk = chunk.module
    layers = chunk.layers
    start, end = getattr(chunk, 'local_layer_range', (0, len(layers)))
    if not 0 <= start < end <= len(layers):
        raise ValueError(
            'Invalid {model} replay layer interval'.replace("{model}", model_name)
        )
    if any((layer is not None) != (start <= i < end) for i, layer in enumerate(layers)):
        raise ValueError(
            '{model} replay requires contiguous stage-owned global layer slots'.replace(
                "{model}", model_name
            )
        )
    return list(layers[start:end])


def validate_router_replay(chunks, action, *, model_name="Model"):
    """Fail before collectives for scheduler interfaces not implemented by E yet."""
    from megatron.lite.primitive.parallel.thd import parallel_state_from_model

    if len(chunks) != 1:
        raise NotImplementedError(
            '{model} replay requires one local PP chunk; VPP is not wired'.replace(
                "{model}", model_name
            )
        )
    ps = parallel_state_from_model(chunks[0]) or ParallelState()
    router_replay_roots(chunks[0], model_name=model_name)
    if action == 'record' and ps.pp_size > 1:
        raise NotImplementedError(
            '{model} PP record requires E post-drain route collection; a collective inside stage forward would deadlock'.replace(
                "{model}", model_name
            )
        )


def unpack_recorded_routed_experts(
    model, batch, recorded, *, pipeline_drained=False, model_name="Model"
):
    """Invert local route packing. PP record needs E's post-drain scheduler hook."""
    import torch.distributed as dist
    from megatron.lite.primitive.parallel.thd import (
        parallel_state_from_model,
        thd_pack_meta,
    )

    ps = parallel_state_from_model(model) or ParallelState()
    if ps.pp_size > 1 and not pipeline_drained:
        validate_router_replay([model], 'record', model_name=model_name)
    if not recorded or any(row is None for row in recorded):
        raise RuntimeError(
            '{model} record did not visit every local router'.replace(
                "{model}", model_name
            )
        )
    full = torch.stack(recorded, dim=1)
    for size, group in ((ps.tp_size, ps.tp_group), (ps.cp_size, ps.cp_group)):
        if size > 1:
            if group is None:
                raise RuntimeError(
                    '{model} route gather requires the corresponding parallel group'.replace(
                        "{model}", model_name
                    )
                )
            full = allgather_concat(full, size, group, dim=0)
    if ps.pp_size > 1:
        if ps.pp_group is None:
            raise RuntimeError(
                '{model} PP route gather requires pp_group after pipeline drain'.replace(
                    "{model}", model_name
                )
            )
        widths = allgather_concat(
            torch.tensor([full.shape[1]], dtype=torch.long, device=full.device),
            ps.pp_size,
            ps.pp_group,
            dim=0,
        ).tolist()
        current = model
        while hasattr(current, 'module'):
            current = current.module
        expected_range = (sum(widths[: ps.pp_rank]), sum(widths[: ps.pp_rank + 1]))
        valid = torch.tensor(
            [getattr(current, 'local_layer_range', None) == expected_range],
            dtype=torch.int32,
            device=full.device,
        )
        dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=ps.pp_group)
        if not valid.item():
            raise ValueError(
                '{model} PP router counts disagree with global stage layer order'.replace(
                    "{model}", model_name
                )
            )
        padded = full.new_zeros(full.shape[0], max(widths), full.shape[2])
        padded[:, : full.shape[1]] = full
        parts = [torch.empty_like(padded) for _ in widths]
        dist.all_gather(parts, padded, group=ps.pp_group)
        full = torch.cat([part[:, :width] for part, width in zip(parts, widths)], dim=1)
    meta = thd_pack_meta(
        batch.seq_lens, tp_size=ps.tp_size, cp_size=ps.cp_size, contiguous=True
    )
    if full.shape[0] != int(meta.cu_seqlens_padded[-1]):
        raise ValueError(
            '{model} recorded rows differ from the shared THD token layout'.replace(
                "{model}", model_name
            )
        )
    rows = [
        full[int(start) : int(start) + int(length)]
        for start, length in zip(meta.cu_seqlens_padded[:-1], meta.lengths)
    ]
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)
