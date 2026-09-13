"""Explicit BF16 dgrad arithmetic, independent of tested TP tensors.

MCore 0cd11658f tensor_parallel/layers.py:557,569-571 returns a matmul
in the input dtype and all-reduces that tensor without promoting to FP32.
This is a bitwise reference contract, not a numerical tolerance.
"""

import torch


def bf16_ordered_sum(partials):
    values = [value.to(torch.bfloat16) for value in partials]
    assert values, 'TP_DGRAD_PARTIALS_REQUIRED'
    result = values[0].clone()
    for value in values[1:]:
        result = result + value
    return result


def independent_partials(dy, full_weight, size=2):
    assert dy.dtype == full_weight.dtype == torch.bfloat16, 'TP_DGRAD_DTYPE'
    return [
        grad.contiguous().matmul(weight)
        for grad, weight in zip(
            dy.chunk(size, -1), full_weight.chunk(size, 0), strict=True
        )
    ]


def explained_difference(actual, modeled, serial):
    if isinstance(actual, torch.Tensor):
        return torch.equal(
            actual.double() - serial.double(), modeled.double() - serial.double()
        )
    return actual - serial == modeled - serial


def assert_dgrad_reference(actual, modeled, serial):
    assert actual.dtype == modeled.dtype and torch.equal(
        actual, modeled
    ), 'TP_DGRAD_BF16_MODEL'
    assert explained_difference(actual, modeled, serial), 'TP_DGRAD_SERIAL_EXPLAINED'


def norm_reference_inputs(optimizer, model, full_gradients, rank, world):
    """Read only owner/layout metadata; every number comes from the full reference."""
    from megatron.lite.model.qwen3_8_flash_next.tp import (
        projection_shard,
        serial_parameter_name,
    )

    names = {id(param): name for name, param in model.named_parameters()}
    owners = {}
    for original_groups, main_groups in (
        (optimizer.model_float16_groups, optimizer.shard_fp32_from_float16_groups),
        (optimizer.model_fp32_groups, optimizer.shard_fp32_groups),
    ):
        for originals, mains in zip(original_groups, main_groups, strict=True):
            for original, main in zip(originals, mains, strict=True):
                owners[id(main)] = original
    tensors, layout = [], []
    for main in optimizer.get_parameters():
        original = owners[id(main)]
        name = names[id(original)]
        # Explicit logical ownership, independent of the tested norm filter.
        if rank != 0 and not (projection_shard(name) or '.experts.' in name):
            continue
        value = full_gradients[serial_parameter_name(name)]
        if projection_shard(name):
            value = value.chunk(world, 0)[rank]
        interval = optimizer._get_model_param_range_map(original)['param']
        value = value.reshape(-1)[interval.start : interval.end].contiguous()
        assert value.numel() == main.numel(), ('TP_NORM_OWNER_RANGE', name)
        assert value.dtype == torch.float32, ('TP_NORM_FP32_REFERENCE', name)
        tensors.append(value.to(main.device))
        layout.append(
            dict(
                name=name,
                start=interval.start,
                end=interval.end,
                shape=list(main.shape),
            )
        )
    return tensors, layout


def ordered_norm_reference(optimizer, model, full_gradients, rank, world):
    """Match MCore's local FP32 L2 -> square -> group SUM -> root exactly.

    Uses the upstream local kernel, but no tested norm, partial sum or gradient
    values. Only the complete single-process model's independent gradients and
    optimizer ownership metadata determine the reference inputs.
    """
    import math
    import torch.distributed as dist
    import megatron.core.optimizer.clip_grads as clip

    children = getattr(optimizer, 'chained_optimizers', [optimizer])
    grouped = [
        norm_reference_inputs(child, model, full_gradients, rank, world)
        for child in children
    ]
    trace = []

    def reduce_group(tensors, group, layout):
        if tensors:
            local, _ = clip.multi_tensor_applier(
                clip.l2_norm_impl,
                torch.zeros(1, dtype=torch.int, device='cuda'),
                [tensors],
                False,
            )
        else:
            local = torch.zeros(1, dtype=torch.float, device='cuda')
        square = local**2.0
        before = square.detach().cpu().clone()
        dist.all_reduce(square, op=dist.ReduceOp.SUM, group=group)
        trace.append(
            dict(
                layout=layout,
                local_l2=local.detach().cpu().clone(),
                local_square=before,
                global_square=square.detach().cpu().clone(),
                group=dist.get_process_group_ranks(group or dist.group.WORLD),
            )
        )
        return (
            square.pow(0.5)
            if clip.multi_tensor_scale_tensor_impl is not None
            else square.item() ** 0.5
        )

    if len(children) == 1:
        value = reduce_group(
            *[grouped[0][0], children[0].get_grad_stats_parallel_group(), grouped[0][1]]
        )
    elif optimizer.grads_states_parallel_group_is_shared():
        value = reduce_group(
            [g for gs, _ in grouped for g in gs],
            optimizer.get_grad_stats_parallel_group(),
            [entry for _, layout in grouped for entry in layout],
        )
    else:
        values = [
            reduce_group(gs, child.get_grad_stats_parallel_group(), layout)
            for child, (gs, layout) in zip(children, grouped, strict=True)
        ]
        value = math.sqrt(sum([x**2 for x in values]))
    return float(value), trace
