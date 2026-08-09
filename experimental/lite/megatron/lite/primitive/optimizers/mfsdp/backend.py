# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# isort: skip_file
"""Runtime backend contract for standalone M-FSDP."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard, _StridedShard


def gather_chunk_metadata(dtensor: DTensor) -> ChunkStorageMetadata:
    """Compute this rank's global offset from the actual uneven local sizes."""
    local_shape = list(dtensor.to_local().shape)
    cumulative_shape = list(local_shape)
    offsets = [0] * len(local_shape)
    placements = list(dtensor.placements)
    shard_order = getattr(dtensor.device_mesh, "_shard_order", None)
    if shard_order is None:
        strided = [
            i
            for i, placement in enumerate(placements)
            if isinstance(placement, _StridedShard)
        ]
        if len(strided) > 1:
            raise ValueError(
                "M-FSDP DCP supports at most one strided DTensor placement."
            )
        ordinary = [i for i in range(len(placements)) if i not in strided]
        shard_order = list(reversed(strided + ordinary))

    for mesh_dim in reversed(shard_order):
        placement = placements[mesh_dim]
        if isinstance(placement, Replicate):
            continue
        if not isinstance(placement, (Shard, _StridedShard)):
            raise ValueError(f"Unsupported DTensor placement: {placement!r}")
        shard_dim = int(placement.dim)
        group = dtensor.device_mesh.get_group(mesh_dim)
        shapes: list[list[int] | None] = [None] * dist.get_world_size(group)
        dist.all_gather_object(shapes, list(cumulative_shape), group=group)
        resolved = [shape for shape in shapes if shape is not None]
        rank = dist.get_rank(group)
        offsets[shard_dim] += sum(shape[shard_dim] for shape in resolved[:rank])
        cumulative_shape[shard_dim] = sum(shape[shard_dim] for shape in resolved)
    return ChunkStorageMetadata(offsets=tuple(offsets), sizes=tuple(local_shape))


def update_uneven_dtensor_chunk_metadata(dtensor: DTensor) -> None:
    chunk = gather_chunk_metadata(dtensor)

    def create_chunk_list():
        return [chunk]

    def create_write_items(fqn: str, tensor: DTensor) -> list[WriteItem]:
        local = tensor.to_local()
        if local.numel() == 0:
            return []
        return [
            WriteItem(
                type=WriteItemType.SHARD,
                index=MetadataIndex(fqn, chunk.offsets),
                tensor_data=TensorWriteData(
                    chunk=chunk,
                    properties=TensorProperties.create_from_tensor(local),
                    size=tensor.size(),
                ),
            )
        ]

    local = dtensor.to_local()
    local.__create_chunk_list__ = create_chunk_list
    local.__create_write_items__ = create_write_items


def preprocess_state_dict_for_uneven_dtensor(
    state_dict: dict[str, Any]
) -> dict[str, Any]:
    """Attach DCP planner closures to every nested DTensor, in stable key order."""
    for _path, value in sorted(_walk_state(state_dict), key=lambda item: item[0]):
        if isinstance(value, DTensor):
            update_uneven_dtensor_chunk_metadata(value)
    return state_dict


def _walk_state(
    value: Any, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_state(child, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_state(child, (*path, str(index)))
    else:
        yield path, value


@dataclass(frozen=True, slots=True)
class MegatronFSDPBackend:
    name: str = "mfsdp"
    runtime_backend: str = "megatron_fsdp"

    def zero_grad(self, optimizer: Any) -> None:
        optimizer.zero_grad()

    def finish_grad_sync(self, optimizer: Any) -> None:
        optimizer.finish_grad_sync()

    def clip_grad_norm(self, optimizer: Any):
        return optimizer.clip_grad_norm()

    def step(self, optimizer: Any):
        return optimizer.step()

    def state_dict(self, optimizer: Any) -> dict[str, Any]:
        return optimizer.state_dict()

    def dcp_state_dict(self, optimizer: Any, *, is_loading: bool) -> dict[str, Any]:
        state = optimizer.state_dict()
        return preprocess_state_dict_for_uneven_dtensor(state)

    def load_state_dict(self, optimizer: Any, state_dict: dict[str, Any]) -> None:
        optimizer.load_state_dict(state_dict)

    def sync_model_weights_to_main_weights(self, optimizer: Any) -> bool:
        for chunk in optimizer._model_chunks:
            chunk.param_sync.copy_full_parameters_to_shards()
        cpu_group = getattr(optimizer._inner_optimizer, "cpu_group", None)
        if cpu_group is not None:
            cpu_group.sync_master_params_from_model()
        return True

    def finalize_grads(
        self, finalize_fn, model_chunks: list[Any], optimizer: Any
    ) -> None:
        finalize_fn()


BACKEND = MegatronFSDPBackend()


__all__ = [
    "BACKEND",
    "MegatronFSDPBackend",
    "gather_chunk_metadata",
    "preprocess_state_dict_for_uneven_dtensor",
    "update_uneven_dtensor_chunk_metadata",
]
