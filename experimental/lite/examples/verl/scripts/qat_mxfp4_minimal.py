# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run one CPU MXFP4-QAT step and materialize a packed snapshot."""

from __future__ import annotations

import torch
import torch.nn as nn

from megatron.lite.primitive.quantization import (
    QATSpec,
    apply_qat_to_chunks,
    quantize_weight,
)


class TinyQwen3MoeBlock(nn.Module):
    """A small Qwen3-MoE-shaped block for the QAT construction contract."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_up = nn.Linear(32, 64, bias=False)
        self.down = nn.Linear(64, 32, bias=False)
        self.router = nn.Module()
        self.router.gate = nn.Linear(32, 4, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate_up(hidden)))


def main() -> None:
    torch.manual_seed(7)
    model = TinyQwen3MoeBlock()
    train_spec = QATSpec(
        enabled=True,
        format="mxfp4",
        group_size=32,
        ignore_patterns=("gate", "router"),
        export_mode="fake",
    )

    # Protocols perform this step before constructing their optimizer.
    stats = apply_qat_to_chunks([model], train_spec)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    master = model.gate_up.parametrizations.weight.original
    optimizer_owns_master = any(
        parameter is master
        for group in optimizer.param_groups
        for parameter in group["params"]
    )

    hidden = torch.randn(4, 32)
    loss = model(hidden).square().mean()
    loss.backward()
    optimizer.step()

    deploy_spec = QATSpec(
        enabled=True,
        format="mxfp4",
        group_size=32,
        export_mode="packed",
    )
    packed = quantize_weight(master.detach(), deploy_spec)

    print(f"quantized_modules={stats['quantized_modules']}")
    print(f"optimizer_owns_master={optimizer_owns_master}")
    print(f"loss={loss.item():.6f}")
    print(
        "packed="
        f"format:{packed['format']} "
        f"qweight:{tuple(packed['qweight'].shape)} "
        f"scale:{tuple(packed['scale'].shape)}"
    )


if __name__ == "__main__":
    main()
