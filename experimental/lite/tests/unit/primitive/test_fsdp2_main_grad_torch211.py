# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import multiprocessing
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

# isort: off
from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW
from megatron.lite.primitive.optimizers.fsdp2.optimizer import FSDP2Optimizer
from megatron.lite.primitive.optimizers.fsdp2 import wrap as fsdp2_wrap
from megatron.lite.primitive.optimizers.fsdp2.wrap import (
    FSDP2Config,
    build_fsdp2_process_group_mesh,
    wrap_fsdp2,
    wrap_fsdp2_module,
)
from megatron.lite.primitive.parallel.state import ParallelState

# isort: on

pytestmark = [pytest.mark.mlite, pytest.mark.distributed]
_TORCH_211_COMMIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"


def _get_main_grad(param: nn.Parameter) -> torch.Tensor | None:
    main_grad = getattr(param, "main_grad", None)
    return main_grad if isinstance(main_grad, torch.Tensor) else None


class TinyExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bd,edh->beh", x, self.weight).mean(dim=1)


class TinyTransformerLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(4, 4, bias=False)
        self.experts = TinyExperts()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dense(x) + self.experts(x))


class TinyQwen3MoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyTransformerLayer(), TinyTransformerLayer()])
        self.out = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


def _main_grad_worker(rank: int, world: int, port: int, results) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.manual_seed(1234)
        model = TinyQwen3MoE().to(dtype=torch.bfloat16)
        ps = ParallelState(dp_cp_size=world, expert_dp_size=world)
        dense_mesh = build_fsdp2_process_group_mesh(
            dist.group.WORLD, mesh_dim_name="dp", device_type="cpu"
        )
        expert_mesh = build_fsdp2_process_group_mesh(
            dist.group.WORLD, mesh_dim_name="expert_dp", device_type="cpu"
        )
        config = FSDP2Config(
            unit_modules=(TinyTransformerLayer,),
            device_type="cpu",
            param_dtype="bfloat16",
            reduce_dtype="float32",
            grad_dtype="float32",
            forward_prefetch_depth=0,
        )
        for layer in model.layers:
            wrap_fsdp2_module(layer.experts, ps, config, mesh=expert_mesh)
        expert_params = {
            param for layer in model.layers for param in layer.experts.parameters()
        }
        wrap_fsdp2(model, ps, config, mesh=dense_mesh, ignored_params=expert_params)

        groups = []
        lazy_init_impls = []
        for module in model.modules():
            getter = getattr(module, "_get_fsdp_state", None)
            if not callable(getter):
                continue
            group = getter()._fsdp_param_group
            if group is not None and id(group) not in {id(item) for item in groups}:
                groups.append(group)
                lazy_init_impls.append(
                    getattr(group.lazy_init, "__func__", group.lazy_init)
                )

        register = getattr(fsdp2_wrap, "register_fsdp2_main_grad_hooks", None)
        if callable(register):
            touched = register(model, torch.float32)
        else:
            # Run this exact carrier against 4e683aa08: its plural-state scan
            # returns zero on the production pin and leaves the real BF16 grad.
            touched = fsdp2_wrap.set_fsdp2_gradient_dtype(model, torch.float32)
        params = list(model.parameters())
        torch_optimizer = FP32AdamW(
            params, lr=1.0e-3, weight_decay=0.0, betas=(0.9, 0.95), eps=1.0e-8
        )
        optimizer = FSDP2Optimizer(torch_optimizer, params, clip_grad=1.0e9)
        x = torch.arange(12, dtype=torch.bfloat16).view(3, 4) / 8
        model(x).float().square().mean().backward()
        first_main_grads = [
            (
                main_grad.clone()
                if (main_grad := _get_main_grad(param)) is not None
                else None
            )
            for param in params
        ]
        model(x).float().square().mean().backward()

        landed_grads_cleared = all(param.grad is None for param in params)
        landed_dtypes = {
            param.grad.to_local().dtype for param in params if param.grad is not None
        }
        main_dtypes = {
            main_grad.dtype
            for param in params
            if (main_grad := _get_main_grad(param)) is not None
        }
        accumulated = all(
            first is not None and torch.equal(_get_main_grad(param), first * 2)
            for param, first in zip(params, first_main_grads, strict=True)
        )
        independent = True
        for param in params:
            state = torch_optimizer.state[param]
            main_grad = _get_main_grad(param)
            if main_grad is None:
                independent = False
                continue
            pointers = {
                tensor.untyped_storage().data_ptr()
                for tensor in (
                    param.to_local(),
                    main_grad.to_local(),
                    state["master_param"].to_local(),
                    state["exp_avg"].to_local(),
                    state["exp_avg_sq"].to_local(),
                )
            }
            independent = independent and len(pointers) == 5

        current_lazy_init_impls = [
            getattr(group.lazy_init, "__func__", group.lazy_init) for group in groups
        ]
        main_grad_bytes = sum(
            main_grad.to_local().numel() * 4
            for param in params
            if (main_grad := _get_main_grad(param)) is not None
        )
        bf16_param_bytes = sum(param.to_local().numel() * 2 for param in params)
        optimizer.step()
        optimizer_consumed = all(
            torch.equal(
                torch_optimizer.state[param]["exp_avg"].to_local(),
                _get_main_grad(param).to_local() * 0.1,
            )
            for param in params
            if _get_main_grad(param) is not None
        ) and all(_get_main_grad(param) is not None for param in params)
        optimizer.zero_grad()
        results.append(
            {
                "touched": touched,
                "group_count": len(groups),
                "landed_grads_cleared": landed_grads_cleared,
                "landed_dtypes": landed_dtypes,
                "main_dtypes": main_dtypes,
                "accumulated": accumulated,
                "independent": independent,
                "optimizer_consumed": optimizer_consumed,
                "lazy_init_unchanged": current_lazy_init_impls == lazy_init_impls,
                "cleared": all(_get_main_grad(param) is None for param in params),
                "main_grad_bytes": main_grad_bytes,
                "bf16_param_bytes": bf16_param_bytes,
            }
        )
    finally:
        dist.destroy_process_group()


def test_torch211_nested_moe_main_grad_contract() -> None:
    if (
        not torch.__version__.startswith("2.11.")
        or torch.version.git_version != _TORCH_211_COMMIT
    ):
        pytest.skip(f"requires production PyTorch 2.11 pin {_TORCH_211_COMMIT}")
    manager = multiprocessing.Manager()
    results = manager.list()
    torch.multiprocessing.spawn(
        _main_grad_worker, args=(2, 29711, results), nprocs=2, join=True
    )

    assert len(results) == 2
    for result in results:
        assert result["touched"] == result["group_count"] == 5, result
        assert result["landed_grads_cleared"], result
        assert result["landed_dtypes"] == set(), result
        assert result["main_dtypes"] == {torch.float32}, result
        assert result["accumulated"], result
        assert result["independent"], result
        assert result["optimizer_consumed"], result
        assert result["lazy_init_unchanged"], result
        assert result["cleared"], result
        assert result["main_grad_bytes"] == 2 * result["bf16_param_bytes"], result
