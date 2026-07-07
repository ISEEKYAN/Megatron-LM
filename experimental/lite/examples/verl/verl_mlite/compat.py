# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small compatibility patches for dependency-version gaps in examples."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from collections.abc import Iterable
from functools import wraps
from pathlib import Path
from typing import Any

_BUCKETED_SENDER_MODULE = "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"


def _weight_sync_probe_enabled() -> bool:
    return os.getenv("MLITE_WEIGHT_SYNC_PROBE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _instrument_bucketed_weight_sender(sender_cls: type) -> bool:
    """Patch veRL's sender only while the opt-in sync probe is enabled."""
    if getattr(sender_cls, "_mlite_weight_sync_probe_patch", False):
        return False

    import torch
    import torch.distributed as dist
    from torch.utils._python_dispatch import TorchDispatchMode

    from megatron.lite.primitive.ckpt.weight_sync_probe import (
        get_weight_sync_probe,
        weight_sync_probe_session,
    )

    probe = get_weight_sync_probe()
    original_init_socket = sender_cls._init_socket
    original_async_send_weights = sender_cls.async_send_weights

    class _ProfiledSocket:
        def __init__(self, socket):
            self._socket = socket

        def __getattr__(self, name):
            return getattr(self._socket, name)

        def send_pyobj(self, *args, **kwargs):
            with probe.measure("handshake"):
                return self._socket.send_pyobj(*args, **kwargs)

        def recv(self, *args, **kwargs):
            with probe.measure("handshake"):
                return self._socket.recv(*args, **kwargs)

    class _H2DCopyMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.ops.aten.copy_.default and len(args) >= 2:
                dst, src = args[:2]
                if (
                    isinstance(dst, torch.Tensor)
                    and isinstance(src, torch.Tensor)
                    and dst.device.type == "cuda"
                    and src.device.type == "cpu"
                ):
                    with probe.measure("h2d", nbytes=src.nbytes, device=dst.device):
                        return func(*args, **kwargs)
            return func(*args, **kwargs)

    def profiled_init_socket(self, *args, **kwargs):
        result = original_init_socket(self, *args, **kwargs)
        self.socket = _ProfiledSocket(self.socket)
        return result

    async def profiled_async_send_weights(self, weights):
        backend = os.getenv("MLITE_WEIGHT_SYNC_PROBE_BACKEND", "unknown")
        original_all_gather_into_tensor = dist.all_gather_into_tensor

        def profiled_all_gather_into_tensor(output, tensor, *args, **kwargs):
            with probe.measure("mbridge_gather", nbytes=output.nbytes, device=tensor.device):
                return original_all_gather_into_tensor(output, tensor, *args, **kwargs)

        with weight_sync_probe_session(backend), _H2DCopyMode():
            dist.all_gather_into_tensor = profiled_all_gather_into_tensor
            try:
                return await original_async_send_weights(self, weights)
            finally:
                dist.all_gather_into_tensor = original_all_gather_into_tensor

    sender_cls._init_socket = profiled_init_socket
    sender_cls.async_send_weights = profiled_async_send_weights
    sender_cls._mlite_weight_sync_probe_patch = True
    return True


class _SenderPatchLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader):
        self._loader = loader

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        _instrument_bucketed_weight_sender(module.BucketedWeightSender)


class _SenderPatchFinder(importlib.abc.MetaPathFinder):
    _mlite_weight_sync_probe_finder = True

    def find_spec(self, fullname, path, target=None):
        if fullname != _BUCKETED_SENDER_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _SenderPatchLoader(spec.loader)
        return spec


def _patch_bucketed_weight_sender() -> bool:
    if not _weight_sync_probe_enabled():
        return False

    module = sys.modules.get(_BUCKETED_SENDER_MODULE)
    if module is not None:
        return _instrument_bucketed_weight_sender(module.BucketedWeightSender)
    if any(getattr(finder, "_mlite_weight_sync_probe_finder", False) for finder in sys.meta_path):
        return False
    sys.meta_path.insert(0, _SenderPatchFinder())
    return True


def _patch_transformers_rope_ignore_keys() -> None:
    try:
        import transformers.modeling_rope_utils as rope_utils
    except Exception:
        return

    for cls in vars(rope_utils).values():
        if not isinstance(cls, type):
            continue
        if getattr(cls, "_verl_mlite_rope_ignore_keys_patch", False):
            continue
        descriptor = vars(cls).get("_check_received_keys")
        if descriptor is None:
            continue

        is_staticmethod = isinstance(descriptor, staticmethod)
        is_classmethod = isinstance(descriptor, classmethod)
        original = descriptor.__func__ if is_staticmethod or is_classmethod else descriptor

        def build_wrapper(check_received_keys: Any) -> Any:
            @wraps(check_received_keys)
            def patched(*args: Any, **kwargs: Any) -> Any:
                ignore_keys = kwargs.get("ignore_keys")
                if isinstance(ignore_keys, list):
                    kwargs["ignore_keys"] = set(ignore_keys)
                elif ignore_keys is not None and not isinstance(ignore_keys, set):
                    if isinstance(ignore_keys, Iterable) and not isinstance(
                        ignore_keys, (str, bytes)
                    ):
                        kwargs["ignore_keys"] = set(ignore_keys)
                return check_received_keys(*args, **kwargs)

            return patched

        patched = build_wrapper(original)
        if is_staticmethod:
            cls._check_received_keys = staticmethod(patched)
        elif is_classmethod:
            cls._check_received_keys = classmethod(patched)
        else:
            cls._check_received_keys = patched
        cls._verl_mlite_rope_ignore_keys_patch = True


def apply_runtime_patches() -> None:
    _patch_transformers_rope_ignore_keys()
    _patch_bucketed_weight_sender()


def _load_verl_file(relative_path: str, module_name: str):
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError("No module named 'verl'")

    path = Path(next(iter(spec.submodule_search_locations))) / relative_path
    file_spec = importlib.util.spec_from_file_location(module_name, path)
    if file_spec is None or file_spec.loader is None:
        raise ImportError(f"Unable to load VERL module from {path}")

    module = importlib.util.module_from_spec(file_spec)
    sys.modules[module_name] = module
    file_spec.loader.exec_module(module)
    return module


def load_verl_engine_api():
    # Prefer the canonical package import so the MLite engine registers into the
    # SAME EngineRegistry that verl's trainers resolve against. Loading base.py as
    # a standalone module (below) creates a *duplicate* registry, which silently
    # drops the mlite backend ("Unknown backend: mlite"). The file-load path is
    # only a fallback for environments where verl isn't importable as a package.
    try:
        from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
        from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches
    except (ModuleNotFoundError, ImportError):
        base = _load_verl_file("workers/engine/base.py", "_verl_mlite_verl_engine_base")
        utils = _load_verl_file("workers/engine/utils.py", "_verl_mlite_verl_engine_utils")
        BaseEngine = base.BaseEngine
        BaseEngineCtx = base.BaseEngineCtx
        EngineRegistry = base.EngineRegistry
        postprocess_batch_func = utils.postprocess_batch_func
        prepare_micro_batches = utils.prepare_micro_batches

    return BaseEngine, BaseEngineCtx, EngineRegistry, postprocess_batch_func, prepare_micro_batches
