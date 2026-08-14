# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Device-aware model construction and strict meta materialization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import torch
import torch.nn as nn

_T = TypeVar("_T", bound=nn.Module)
_PENDING_ATTR = "_mlite_pending_meta_tensors"


def build_module_on_device(
    factory: Callable[..., _T],
    *args: Any,
    use_meta: bool,
    dtype: torch.dtype,
    device: torch.device | str = "cuda",
    **kwargs: Any,
) -> _T:
    """Build a module eagerly or with parameter storage deferred on ``meta``.

    Modules used with the meta path must construct non-checkpoint buffers on an
    explicit real device. Any buffer that remains on meta is included in the
    strict materialization audit instead of silently becoming uninitialized.
    """

    if use_meta:
        with torch.device("meta"):
            module = factory(*args, **kwargs)
        # Transformer Engine constructors default to an explicit CUDA device,
        # which overrides PyTorch's ambient meta context. Large TE grouped
        # linears receive the context device at their call site; canonicalize
        # every residual real-device parameter here so the protocol contract is
        # all-meta before FSDP2 wrapping. Constructor-owned buffers keep their
        # initialized values and are moved normally during materialization.
        if any(param.device.type != "meta" for param in module.parameters()):
            preserved_buffers = {
                name: buffer.detach().clone()
                for name, buffer in module.named_buffers()
                if buffer.device.type != "meta"
            }
            module.to_empty(device="meta")
            _restore_named_buffers(module, preserved_buffers)
        module = module.to(dtype=dtype)
        unexpected = [
            name
            for name, param in module.named_parameters()
            if param.device.type != "meta"
        ]
        if unexpected:
            raise RuntimeError(
                "Meta-device model construction left parameters on a real device: "
                + ", ".join(unexpected)
            )
        return module
    module = factory(*args, **kwargs).to(dtype=dtype)
    target = torch.device(device)
    return module.cuda() if target.type == "cuda" else module.to(device=target)


def use_fsdp2_meta_init(impl_cfg: Any) -> bool:
    """Return whether the FSDP2-only default meta path is enabled."""

    return bool(
        getattr(impl_cfg, "optimizer", None) == "fsdp2"
        and getattr(impl_cfg, "fsdp2_meta_init", True)
    )


def transformer_engine_init_device() -> torch.device:
    """Match TE's explicit device to the active model-construction context."""

    default = torch.get_default_device()
    return default if default.type == "meta" else torch.device("cuda")


def materialize_meta_module(module: _T, *, device: torch.device | str = "cuda") -> _T:
    """Materialize meta tensors after FSDP2 wrapping and start a fill audit."""

    pending = {
        name
        for name, tensor in _named_parameters_and_buffers(module)
        if tensor.device.type == "meta"
    }
    if not pending:
        return module

    # ``Module.to_empty`` also replaces already-real buffers. Preserve those
    # small constructor-initialized values while materializing FSDP parameters.
    preserved_buffers = {
        name: buffer.detach().clone()
        for name, buffer in module.named_buffers()
        if buffer.device.type != "meta"
    }
    module.to_empty(device=device)
    with torch.no_grad():
        buffers = dict(module.named_buffers())
        for name, value in preserved_buffers.items():
            buffers[name].copy_(
                value.to(device=buffers[name].device, dtype=buffers[name].dtype)
            )
    setattr(module, _PENDING_ATTR, pending)
    return module


def record_materialized_tensor(module: nn.Module, name: str) -> None:
    """Mark one checkpoint destination as initialized."""

    pending = getattr(module, _PENDING_ATTR, None)
    if pending is not None:
        pending.discard(name)


def finalize_meta_materialization(module: nn.Module) -> None:
    """Reject any meta-created tensor not filled by checkpoint loading."""

    pending = getattr(module, _PENDING_ATTR, None)
    if pending is None:
        return
    if pending:
        names = ", ".join(sorted(pending))
        raise RuntimeError(
            "Meta-device materialization left tensors uninitialized by the checkpoint: "
            f"{names}. Add them to the checkpoint mapping or initialize them explicitly "
            "on a real device before finalizing model construction."
        )
    delattr(module, _PENDING_ATTR)


def _named_parameters_and_buffers(module: nn.Module):
    yield from module.named_parameters()
    yield from module.named_buffers()


def _restore_named_buffers(module: nn.Module, buffers: dict[str, torch.Tensor]) -> None:
    for name, value in buffers.items():
        parent_name, _, leaf_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, leaf_name, value)


__all__ = [
    "build_module_on_device",
    "finalize_meta_materialization",
    "materialize_meta_module",
    "record_materialized_tensor",
    "transformer_engine_init_device",
    "use_fsdp2_meta_init",
]
