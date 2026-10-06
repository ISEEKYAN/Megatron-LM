# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Synchronize replicated owners while excluding explicitly sharded storage."""
import torch


def wrap_owned_ddp(model, ps, *, optimizing, external_device, row_tables, shard_group):
    execution_model = model
    if (ps.dp_size > 1 or ps.cp_size > 1) and optimizing:
        if external_device is not None:
            raise NotImplementedError(
                'DP external vision requires staged gradient synchronization'
            )
        from torch.nn.parallel import DistributedDataParallel

        # DDP synchronizes parameter initialization. Encoded row buffers need
        # byte collectives because NCCL does not accept their FP8 storage dtype.
        gradient_group = ps.dp_cp_group if ps.cp_size > 1 else ps.dp_group
        sharded = set()
        if shard_group is not None:
            for table in row_tables:
                sharded.update(id(tensor) for tensor in table.buffers())
                if table.master is not None:
                    sharded.add(id(table.master))
        model._ddp_params_and_buffers_to_ignore = [
            name
            for name, tensor in (*model.named_parameters(), *model.named_buffers())
            if id(tensor) in sharded
        ]
        with torch.no_grad():
            for buffer in model.buffers():
                if id(buffer) in sharded:
                    continue
                value = buffer.contiguous().reshape(-1).view(torch.uint8)
                torch.distributed.broadcast(value, src=0, group=gradient_group)
                buffer.copy_(value.view(buffer.dtype).reshape(buffer.shape))
        if ps.ep_size > 1:
            expert_ids = {
                id(b.tensor) for b in model.parameter_bindings() if b.role == "expert"
            }
            model._ddp_params_and_buffers_to_ignore += [
                name
                for name, parameter in model.named_parameters()
                if id(parameter) in expert_ids
            ]
            for parameter in model.parameters():
                if id(parameter) in expert_ids:
                    torch.distributed.broadcast(
                        parameter.data,
                        src=torch.distributed.get_global_rank(ps.ep_dp_group, 0),
                        group=ps.ep_dp_group,
                    )
        execution_model = DistributedDataParallel(
            model,
            process_group=gradient_group,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    return execution_model
