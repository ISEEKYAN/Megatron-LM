"""Declarative fused checkpoint tensor layouts.

Model code declares semantic segments and their head geometry. This primitive
owns concatenation, splitting, tensor-parallel slicing, and quantized
packed/scale pairing so callers never infer layout from parameter names.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class WeightSegment:
    """One semantic row segment in a fused projection."""

    name: str
    heads: int
    head_dim: int
    replicated: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("weight segment name must not be empty")
        if self.heads <= 0:
            raise ValueError(f"{self.name!r} heads must be positive")
        if self.head_dim <= 0:
            raise ValueError(f"{self.name!r} head_dim must be positive")

    @property
    def rows(self) -> int:
        return self.heads * self.head_dim

    def local_rows(self, world_size: int) -> int:
        """Return local rows, validating head divisibility for sharded segments."""
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.replicated or world_size == 1:
            return self.rows
        if self.heads % world_size:
            raise ValueError(
                f"{self.name!r} heads={self.heads} must be divisible by TP={world_size}"
            )
        return self.heads // world_size * self.head_dim


@dataclass(frozen=True, slots=True)
class QuantizedWeight:
    """A packed tensor and its scale, kept together as one semantic value."""

    packed: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True, slots=True)
class FusedWeightLayout:
    """Ordered, fail-loud layout for a fused projection's output rows."""

    name: str
    segments: tuple[WeightSegment, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("fused weight layout name must not be empty")
        if not self.segments:
            raise ValueError(f"{self.name!r} requires at least one segment")
        names = tuple(segment.name for segment in self.segments)
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name!r} has duplicate segment names")

    @property
    def rows(self) -> int:
        return sum(segment.rows for segment in self.segments)

    def _ordered(self, tensors: Mapping[str, torch.Tensor]) -> list[torch.Tensor]:
        expected = {segment.name for segment in self.segments}
        actual = set(tensors)
        if actual != expected:
            raise ValueError(
                f"{self.name!r} segment set mismatch: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )
        ordered = []
        trailing_shape = None
        dtype = None
        device = None
        for segment in self.segments:
            tensor = tensors[segment.name]
            if tensor.ndim < 1:
                raise ValueError(
                    f"{self.name!r}.{segment.name} must have at least one dimension"
                )
            if tensor.size(0) != segment.rows:
                raise ValueError(
                    f"{self.name!r}.{segment.name} requires {segment.rows} rows "
                    f"from {segment.heads} heads x {segment.head_dim}, got {tensor.size(0)}"
                )
            current_trailing = tuple(tensor.shape[1:])
            if trailing_shape is None:
                trailing_shape = current_trailing
                dtype = tensor.dtype
                device = tensor.device
            elif current_trailing != trailing_shape:
                raise ValueError(
                    f"{self.name!r}.{segment.name} trailing shape {current_trailing} "
                    f"does not match {trailing_shape}"
                )
            elif tensor.dtype != dtype or tensor.device != device:
                raise ValueError(
                    f"{self.name!r}.{segment.name} dtype/device must match earlier segments"
                )
            ordered.append(tensor)
        return ordered

    def fuse(self, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Fuse semantic tensors in declared order after validating geometry."""
        return torch.cat(self._ordered(tensors), dim=0).contiguous()

    def fuse_ordered(self, tensors: Sequence[tuple[str, torch.Tensor]]) -> torch.Tensor:
        """Fuse an ordered source stream, rejecting swapped semantic segments."""
        expected = tuple(segment.name for segment in self.segments)
        actual = tuple(name for name, _ in tensors)
        if actual != expected:
            raise ValueError(
                f"{self.name!r} segment order mismatch: "
                f"expected={expected}, actual={actual}"
            )
        return self.fuse(dict(tensors))

    def split(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split one fused tensor using the same declaration used to fuse it."""
        if tensor.ndim < 1 or tensor.size(0) != self.rows:
            got = tensor.size(0) if tensor.ndim else 0
            raise ValueError(
                f"{self.name!r} requires {self.rows} fused rows, got {got}"
            )
        return dict(
            zip(
                (segment.name for segment in self.segments),
                tensor.split([segment.rows for segment in self.segments], dim=0),
                strict=True,
            )
        )

    def tp_shard(
        self, tensor: torch.Tensor, *, rank: int, world_size: int
    ) -> torch.Tensor:
        """Shard each head-based segment independently, preserving replicas."""
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be within a positive world_size")
        parts = self.split(tensor)
        local = {}
        local_segments = []
        for segment in self.segments:
            part = parts[segment.name]
            if segment.replicated or world_size == 1:
                local[segment.name] = part
                local_segments.append(segment)
                continue
            segment.local_rows(world_size)
            headed = part.reshape(segment.heads, segment.head_dim, *part.shape[1:])
            local[segment.name] = headed.chunk(world_size, dim=0)[rank].reshape(
                -1, *part.shape[1:]
            )
            local_segments.append(
                WeightSegment(
                    segment.name,
                    segment.heads // world_size,
                    segment.head_dim,
                )
            )
        return FusedWeightLayout(self.name, tuple(local_segments)).fuse(local)

    def fuse_quantized(
        self,
        tensors: Mapping[str, QuantizedWeight],
        *,
        materialize: Callable[[QuantizedWeight], torch.Tensor],
    ) -> torch.Tensor:
        """Validate packed/scale pairing, materialize, then fuse by declaration."""
        expected = {segment.name for segment in self.segments}
        actual = set(tensors)
        if actual != expected:
            raise ValueError(
                f"{self.name!r} quantized segment set mismatch: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )
        materialized = {}
        for segment in self.segments:
            pair = tensors[segment.name]
            for label, value in (("packed", pair.packed), ("scale", pair.scale)):
                if value.ndim < 1 or value.size(0) != segment.rows:
                    got = value.size(0) if value.ndim else 0
                    raise ValueError(
                        f"{self.name!r}.{segment.name} {label} requires "
                        f"{segment.rows} rows, got {got}"
                    )
            materialized[segment.name] = materialize(pair)
        return self.fuse(materialized)

    def fuse_quantized_ordered(
        self,
        tensors: Sequence[tuple[str, QuantizedWeight]],
        *,
        materialize: Callable[[QuantizedWeight], torch.Tensor],
    ) -> torch.Tensor:
        """Materialize an ordered quantized stream with fail-loud semantics."""
        expected = tuple(segment.name for segment in self.segments)
        actual = tuple(name for name, _ in tensors)
        if actual != expected:
            raise ValueError(
                f"{self.name!r} quantized segment order mismatch: "
                f"expected={expected}, actual={actual}"
            )
        return self.fuse_quantized(dict(tensors), materialize=materialize)

    def split_quantized(
        self, packed: torch.Tensor, scale: torch.Tensor
    ) -> dict[str, QuantizedWeight]:
        """Split packed weights and scales through identical semantic boundaries."""
        packed_parts = self.split(packed)
        scale_parts = self.split(scale)
        return {
            segment.name: QuantizedWeight(
                packed=packed_parts[segment.name], scale=scale_parts[segment.name]
            )
            for segment in self.segments
        }


__all__ = ["FusedWeightLayout", "QuantizedWeight", "WeightSegment"]
