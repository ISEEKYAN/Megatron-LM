# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Raw-tensor dump hooks for the wgrad accumulation verdict experiment.

Env contract (all optional; dumping is off unless both DIR and PATTERNS set):
  MLITE_WGRAD_DUMP_DIR       output directory (created if missing)
  MLITE_WGRAD_DUMP_PATTERNS  comma-separated substrings matched against
                             module qualified names; only modules that
                             directly own a 2-D ``weight`` Parameter match
  MLITE_WGRAD_DUMP_STEP      training step to dump (default 0)
  MLITE_WGRAD_DUMP_RANK      rank that dumps (default 0)

For every matched module the context saves, per forward invocation index i:
  <name>.X.<i>.pt   module forward input (first tensor arg)
  <name>.dY.<i>.pt  gradient flowing into the module output
and after backward (context exit):
  <name>.<pname>.wgrad.pt  param.grad or param.main_grad
  <name>.<pname>.weight.pt the parameter value itself
plus modules_all.txt / modules_matched.txt for name discovery.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import torch


def _rank() -> int:
    for name in ("RANK", "SLURM_PROCID"):
        raw = os.environ.get(name)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                continue
    return 0


def _to_plain(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        tensor = to_local()
    return tensor.detach().to("cpu", copy=True)


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


@contextmanager
def wgrad_dump_context(model: torch.nn.Module, *, step: int):
    dump_dir = os.environ.get("MLITE_WGRAD_DUMP_DIR")
    patterns = [
        p.strip() for p in os.environ.get("MLITE_WGRAD_DUMP_PATTERNS", "").split(",") if p.strip()
    ]
    dump_step = int(os.environ.get("MLITE_WGRAD_DUMP_STEP", "0"))
    dump_rank = int(os.environ.get("MLITE_WGRAD_DUMP_RANK", "0"))
    if not dump_dir or not patterns or step != dump_step or _rank() != dump_rank:
        yield
        return

    out_dir = Path(dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matched: dict[str, torch.nn.Module] = {}
    all_names: list[str] = []
    for name, module in model.named_modules():
        all_names.append(f"{name}\t{type(module).__qualname__}")
        if not name or not any(p in name for p in patterns):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.nn.Parameter) or weight.dim() != 2:
            continue
        matched[name] = module
    (out_dir / "modules_all.txt").write_text("\n".join(all_names) + "\n", encoding="utf-8")
    (out_dir / "modules_matched.txt").write_text("\n".join(matched) + "\n", encoding="utf-8")

    def _save(tag: str, tensor: torch.Tensor) -> None:
        torch.save(_to_plain(tensor), out_dir / f"{tag}.pt")

    counters: dict[tuple[str, str], int] = {}

    def _next(name: str, kind: str) -> int:
        key = (name, kind)
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        return idx

    hooks = []
    tensor_hooks = []
    for name, module in matched.items():

        def _pre_hook(_module, args, _name=name):
            x = _first_tensor(args)
            if x is not None:
                _save(f"{_name}.X.{_next(_name, 'X')}", x)

        def _fwd_hook(_module, _args, output, _name=name):
            out = _first_tensor(output)
            if isinstance(out, torch.Tensor) and out.requires_grad:
                idx = _next(_name, "dY")

                def _grad_hook(grad, __name=_name, __idx=idx):
                    _save(f"{__name}.dY.{__idx}", grad)

                tensor_hooks.append(out.register_hook(_grad_hook))

        hooks.append(module.register_forward_pre_hook(_pre_hook))
        hooks.append(module.register_forward_hook(_fwd_hook))

    try:
        yield
    finally:
        for hook in tensor_hooks:
            hook.remove()
        for hook in hooks:
            hook.remove()
        for name, module in matched.items():
            for pname, param in module.named_parameters(recurse=False):
                grad = param.grad
                if grad is None:
                    grad = getattr(param, "main_grad", None)
                if grad is not None:
                    _save(f"{name}.{pname}.wgrad", grad)
                _save(f"{name}.{pname}.weight", param)
