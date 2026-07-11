# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-Core utilities aligned with VERL's megatron_utils.

All functions here are ported from VERL and could theoretically be replaced
by ``from verl.utils.megatron_utils import ...`` if Megatron Lite ever depends on verl.

Sources:
  - verl/utils/megatron_utils.py  (offload/load, register_megatron_training_hooks)
  - verl/utils/megatron/optimizer.py  (get_megatron_last_lr)
"""

from __future__ import annotations

import gc
from collections.abc import MutableMapping
from typing import Any

import torch

# ======================================================================
# Rank utilities
# ======================================================================


def is_mp_src_rank_with_outputs() -> bool:
    """True on the rank that holds the final model output (loss).

    Only last PP stage, first TP rank, first CP rank has the output.
    VERL: MegatronEngine.is_mp_src_rank_with_outputs
    """
    from megatron.core import parallel_state as mpu

    return (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank()
        == mpu.get_pipeline_model_parallel_world_size() - 1
        and mpu.get_context_parallel_rank() == 0
    )


# ======================================================================
# Training hooks — register_megatron_training_hooks
# ======================================================================


def register_training_hooks(model_list: list, optimizer) -> None:
    """Register megatron training callbacks on model config.

    Ref: megatron/training/training.py (core_v0.15.0rc7, L2039-L2057)
    """
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.utils import get_model_config

    for one_model in model_list:
        config = get_model_config(one_model)
        if optimizer is not None:
            config.grad_scale_func = optimizer.scale_loss
        config.finalize_model_grads_func = finalize_model_grads

        optimizer_config = getattr(optimizer, "config", None)
        overlap_param_gather = getattr(optimizer_config, "overlap_param_gather", False)
        overlap_grad_reduce = getattr(one_model.ddp_config, "overlap_grad_reduce", False)
        align_grad_reduce = True
        align_param_gather = getattr(one_model.ddp_config, "align_param_gather", False)

        if isinstance(model_list[0], DDP) and overlap_grad_reduce:
            config.no_sync_func = [m.no_sync for m in model_list]
            if len(model_list) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if align_grad_reduce:
                config.grad_sync_func = [m.start_grad_sync for m in model_list]
                if len(model_list) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if overlap_param_gather and align_param_gather:
            config.param_sync_func = [m.start_param_sync for m in model_list]
            if len(model_list) == 1:
                config.param_sync_func = config.param_sync_func[0]


# ======================================================================
# Model offload / load — offload_megatron_model_to_cpu / load_megatron_model_to_gpu
# ======================================================================


def _is_megatron_ddp(model_chunk: Any) -> bool:
    try:
        from megatron.core.distributed import DistributedDataParallel as DDP
    except Exception:
        return False

    return (
        isinstance(model_chunk, DDP)
        and hasattr(model_chunk, "buffers")
        and hasattr(model_chunk, "expert_parallel_buffers")
        and hasattr(model_chunk, "module")
    )


def offload_model_to_cpu(model_list: list) -> None:
    """Offload DDP model to CPU via buffer-resize (zero-copy on GPU side)."""
    for model_chunk in model_list:
        if _is_megatron_ddp(model_chunk):
            all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in all_buffers:
                for buffer in buffers:
                    if buffer.param_data.storage().size() > 0:
                        buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                        buffer.param_data_size = buffer.param_data.storage().size()
                        buffer.param_data.storage().resize_(0)

                    if buffer.grad_data.storage().size() > 0:
                        buffer.grad_data_size = buffer.grad_data.storage().size()
                        buffer.grad_data.storage().resize_(0)

            for param in model_chunk.module.parameters():
                if not param.requires_grad and param.device.type != "cpu":
                    param.data = param.data.to("cpu", non_blocking=True)
        else:
            model_chunk.to("cpu")


def load_model_to_gpu(model_list: list, load_grad: bool = True) -> None:
    """Load DDP model back to GPU from pinned CPU copy."""
    for model_chunk in model_list:
        if _is_megatron_ddp(model_chunk):
            all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in all_buffers:
                for buffer in buffers:
                    if load_grad and hasattr(buffer, "grad_data_size"):
                        current_size = buffer.grad_data.storage().size()
                        if current_size == 0 or current_size == buffer.grad_data_size:
                            buffer.grad_data.storage().resize_(buffer.grad_data_size)
                            buffer.grad_data.zero_()
                        else:
                            buffer.grad_data.zero_()

                    if buffer.param_data.storage().size() == 0:
                        buffer.param_data.storage().resize_(buffer.param_data_size)
                        buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)

            for param in model_chunk.module.parameters():
                if not param.requires_grad and param.device.type == "cpu":
                    param.data = param.data.to("cuda", non_blocking=True)
        else:
            model_chunk.to("cuda")


# ======================================================================
# Optimizer offload / load — offload_megatron_optimizer / load_megatron_optimizer
# ======================================================================


def offload_optimizer(optimizer) -> None:
    """Recursively offload every tensor in a composed optimizer's runtime state."""
    if getattr(optimizer, "_mlite_offloaded_optimizer_state", None):
        return

    moved: dict[int, tuple[torch.Tensor, torch.device]] = {}
    roots = []
    for state in _iter_optimizer_state_mappings(optimizer):
        for key, value in list(state.items()):
            state[key], devices = _offload_tensor_tree(value, moved)
            if devices is not None:
                roots.append((state, key, devices))
    optimizer._mlite_offloaded_optimizer_state = roots

    gc.collect()
    torch.cuda.empty_cache()


def load_optimizer(optimizer) -> None:
    """Restore only state tensors moved by :func:`offload_optimizer`."""
    roots = getattr(optimizer, "_mlite_offloaded_optimizer_state", None)
    if not roots:
        return

    restored: dict[tuple[int, str], torch.Tensor] = {}
    for state, key, devices in roots:
        state[key] = _restore_tensor_tree(state[key], devices, restored)
    optimizer._mlite_offloaded_optimizer_state = []

    gc.collect()
    torch.cuda.empty_cache()


def _iter_optimizer_state_mappings(optimizer):
    """Yield unique torch-style state mappings from nested optimizer facades."""
    seen_nodes: set[int] = set()
    seen_states: set[int] = set()
    pending = [optimizer]
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node))

        state = getattr(node, "state", None)
        if isinstance(state, MutableMapping) and id(state) not in seen_states:
            seen_states.add(id(state))
            yield state

        for attribute in ("chained_optimizers", "sub_optimizers"):
            children = getattr(node, attribute, None)
            if children is not None:
                pending.extend(children)
        try:
            wrapped = getattr(node, "optimizer", None)
        except (AssertionError, AttributeError, RuntimeError, ValueError):
            wrapped = None
        if wrapped is not None:
            pending.append(wrapped)


def _offload_tensor_tree(value, moved):
    if isinstance(value, torch.Tensor):
        if value.device.type == "cpu":
            return value, None
        cached = moved.get(id(value))
        if cached is None:
            cached = (value.to("cpu", non_blocking=True), value.device)
            moved[id(value)] = cached
        return cached[0], ("tensor", cached[1])
    if isinstance(value, MutableMapping):
        devices = {}
        for key, item in list(value.items()):
            value[key], item_devices = _offload_tensor_tree(item, moved)
            if item_devices is not None:
                devices[key] = item_devices
        return value, ("mapping", devices) if devices else None
    if isinstance(value, list):
        devices = {}
        for index, item in enumerate(value):
            value[index], item_devices = _offload_tensor_tree(item, moved)
            if item_devices is not None:
                devices[index] = item_devices
        return value, ("list", devices) if devices else None
    if isinstance(value, tuple):
        items = []
        devices = {}
        for index, item in enumerate(value):
            item, item_devices = _offload_tensor_tree(item, moved)
            items.append(item)
            if item_devices is not None:
                devices[index] = item_devices
        rebuilt = type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
        return rebuilt, ("tuple", devices) if devices else None
    return value, None


def _restore_tensor_tree(value, devices, restored):
    kind, children = devices
    if kind == "tensor":
        cache_key = (id(value), str(children))
        if cache_key not in restored:
            restored[cache_key] = value.to(children, non_blocking=True)
        return restored[cache_key]
    if kind == "mapping":
        for key, item_devices in children.items():
            value[key] = _restore_tensor_tree(value[key], item_devices, restored)
        return value
    if kind == "list":
        for index, item_devices in children.items():
            value[index] = _restore_tensor_tree(value[index], item_devices, restored)
        return value
    if kind == "tuple":
        items = list(value)
        for index, item_devices in children.items():
            items[index] = _restore_tensor_tree(items[index], item_devices, restored)
        return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
    raise RuntimeError(f"Unknown optimizer state tree kind: {kind}")


# ======================================================================
# Checkpoint helpers
# ======================================================================


def build_sharded_state_dict(
    model_list: list, optimizer: Any = None, lr_scheduler: Any = None
) -> dict[str, Any]:
    """Build sharded state dict for model + optimizer + lr_scheduler.

    Uses ``model0.`` / ``model1.`` prefix for VPP (multiple model chunks).
    """
    sharded_state_dict: dict[str, Any] = {}

    for i, model_chunk in enumerate(model_list):
        prefix = f"model{i}." if len(model_list) > 1 else "model."
        chunk_sd = model_chunk.sharded_state_dict(prefix=prefix)
        sharded_state_dict.update(chunk_sd)

    if optimizer is not None:
        opt_sd = optimizer.sharded_state_dict(model_sharded_state_dict=sharded_state_dict)
        sharded_state_dict.update(opt_sd)

    if lr_scheduler is not None:
        sharded_state_dict["lr_scheduler"] = lr_scheduler.state_dict()

    return sharded_state_dict
