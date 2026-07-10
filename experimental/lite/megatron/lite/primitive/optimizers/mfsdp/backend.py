# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime backend contract for standalone M-FSDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_STATE_FORMAT = "megatron_lite.mfsdp_optimizer.v1"
_STATE_KEYS = {"format", "optimizer", "main_params"}


def _main_parameters(optimizer: Any) -> list[torch.nn.Parameter]:
    params = getattr(getattr(optimizer, "_inner_optimizer", None), "params", None)
    if params is None:
        raise TypeError("M-FSDP optimizer does not expose persistent main parameters.")
    return list(params)


@dataclass(frozen=True, slots=True)
class MegatronFSDPBackend:
    name: str = "mfsdp"
    runtime_backend: str = "megatron_fsdp"

    def zero_grad(self, optimizer: Any) -> None:
        optimizer.zero_grad()

    def step(self, optimizer: Any):
        return optimizer.step()

    def state_dict(self, optimizer: Any) -> dict[str, Any]:
        return {
            "format": _STATE_FORMAT,
            "optimizer": optimizer.state_dict(),
            "main_params": [
                param.detach().cpu().clone() for param in _main_parameters(optimizer)
            ],
        }

    def load_state_dict(self, optimizer: Any, state_dict: dict[str, Any]) -> None:
        state_format = (
            state_dict.get("format") if isinstance(state_dict, dict) else None
        )
        if state_format is None:
            optimizer.load_state_dict(state_dict)
            optimizer._mfsdp_checkpoint_restored_main_params = False
            return
        if state_format != _STATE_FORMAT:
            raise RuntimeError(
                f"Unsupported M-FSDP optimizer checkpoint format: {state_format!r}"
            )
        keys = set(state_dict)
        if keys != _STATE_KEYS:
            raise RuntimeError(
                "M-FSDP optimizer checkpoint keys differ: "
                f"missing={sorted(_STATE_KEYS - keys)}, unexpected={sorted(keys - _STATE_KEYS)}"
            )

        params = _main_parameters(optimizer)
        saved_params = state_dict["main_params"]
        if len(saved_params) != len(params):
            raise RuntimeError(
                "M-FSDP optimizer main-parameter count differs: "
                f"checkpoint={len(saved_params)}, runtime={len(params)}"
            )
        optimizer.load_state_dict(state_dict["optimizer"])
        with torch.no_grad():
            for index, (param, saved) in enumerate(
                zip(params, saved_params, strict=True)
            ):
                if tuple(saved.shape) != tuple(param.shape):
                    raise RuntimeError(
                        f"M-FSDP main parameter {index} shape differs: "
                        f"checkpoint={tuple(saved.shape)}, runtime={tuple(param.shape)}"
                    )
                if saved.dtype != param.dtype:
                    raise RuntimeError(
                        f"M-FSDP main parameter {index} dtype differs: "
                        f"checkpoint={saved.dtype}, runtime={param.dtype}"
                    )
                param.copy_(saved.to(device=param.device))
        optimizer._mfsdp_checkpoint_restored_main_params = True

    def sync_model_weights_to_main_weights(self, optimizer: Any) -> bool:
        if bool(getattr(optimizer, "_mfsdp_checkpoint_restored_main_params", False)):
            optimizer._mfsdp_checkpoint_restored_main_params = False
            return True
        for chunk in optimizer._model_chunks:
            chunk.param_sync.copy_full_parameters_to_shards()
        return True


BACKEND = MegatronFSDPBackend()


__all__ = [
    "BACKEND",
    "MegatronFSDPBackend",
]
