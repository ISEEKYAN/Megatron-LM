# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import copy
import importlib
import multiprocessing
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

# isort: off
from megatron.lite.primitive.optimizers.fsdp2 import (
    FSDP2Config,
    FSDP2Optimizer,
    all_reduce_scalar_,
    clip_grads_with_sharded_norm_,
    fsdp2_available,
)
from megatron.lite.primitive.optimizers.fsdp2.adamw import (
    FP32AdamW,
    build_adamw_optimizer,
    to_local_tensor,
)
from megatron.lite.primitive.optimizers.fsdp2.main_grad import get_param_grad
from megatron.lite.primitive.optimizers.fsdp2.wrap import build_fsdp2_shard_placement_fn
from megatron.lite.primitive.parallel.state import ParallelState

# isort: on

pytestmark = pytest.mark.mlite

fsdp2_wrap = importlib.import_module("megatron.lite.primitive.optimizers.fsdp2.wrap")
fsdp2_optimizer = importlib.import_module(
    "megatron.lite.primitive.optimizers.fsdp2.optimizer"
)
fsdp2_grad_clip = importlib.import_module(
    "megatron.lite.primitive.optimizers.fsdp2.grad_clip"
)


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ToyBlock()
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        return self.out(self.block(x))


class TwoBlockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = ToyBlock()
        self.block1 = ToyBlock()
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        return self.out(self.block1(self.block0(x)))


class NestedToyBlock(ToyBlock):
    def __init__(self):
        super().__init__()
        self.inner = ToyBlock()


class ToyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


class ToyMoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(4, 4, bias=False)
        self.experts = ToyExperts()

    def forward(self, x):
        return self.experts(self.dense(x))


def _assert_loss_and_gradients_bitwise_equal(
    actual_loss: torch.Tensor,
    expected_loss: torch.Tensor,
    actual_model: nn.Module,
    expected_model: nn.Module,
) -> None:
    assert torch.equal(actual_loss, expected_loss)
    actual_grads = dict(actual_model.named_parameters())
    expected_grads = dict(expected_model.named_parameters())
    assert actual_grads.keys() == expected_grads.keys()
    for name in actual_grads:
        assert torch.equal(actual_grads[name].grad, expected_grads[name].grad), name


def test_fsdp2_config_validates_empty_wrap_surface():
    with pytest.raises(ValueError, match="wrap_root=True"):
        FSDP2Config(wrap_root=False)


def test_fsdp2_pipeline_preserves_reshard_after_forward_setting():
    ps = SimpleNamespace(pp_size=4)

    assert (
        fsdp2_optimizer._fsdp2_unit_reshard_after_forward(
            ps, reshard_after_forward=True
        )
        is True
    )


def test_fsdp2_pipeline_wraps_dense_and_experts_with_reshard(monkeypatch):
    model = ToyMoEBlock()
    dense_calls = []
    expert_calls = []
    expert_mesh = SimpleNamespace(name="expert_dp")
    optimizer = object()
    ps = ParallelState(
        pp_size=2, ep_size=2, dp_cp_size=4, expert_dp_size=2, ep_dp_group=object()
    )

    def fake_wrap_expert(module, _ps, config, **kwargs):
        expert_calls.append((module, config, kwargs))
        return module

    def fake_wrap_dense(module, _ps, config, **kwargs):
        dense_calls.append((module, config, kwargs))
        return module

    monkeypatch.setattr(
        fsdp2_optimizer,
        "build_fsdp2_process_group_mesh",
        lambda *args, **kwargs: expert_mesh,
    )
    monkeypatch.setattr(fsdp2_optimizer, "wrap_fsdp2_module", fake_wrap_expert)
    monkeypatch.setattr(fsdp2_optimizer, "wrap_fsdp2", fake_wrap_dense)
    monkeypatch.setattr(
        fsdp2_optimizer, "register_fsdp2_main_grad_hooks", lambda *_args, **_kwargs: 1
    )
    monkeypatch.setattr(
        fsdp2_optimizer, "build_fsdp2_adamw", lambda *args, **kwargs: optimizer
    )

    result = fsdp2_optimizer.build_fsdp2_training_optimizer(
        [model],
        None,
        ps,
        unit_modules=(ToyMoEBlock,),
        expert_classifier=lambda name: ".experts." in f".{name}",
        reshard_after_forward=True,
        grad_dtype="bfloat16",
        use_fp32_master=False,
    )

    assert result is optimizer
    assert len(expert_calls) == 1
    expert_module, expert_config, expert_kwargs = expert_calls[0]
    assert expert_module is model.experts
    assert expert_config.reshard_after_forward is True
    assert expert_kwargs["reshard_after_forward"] is True
    assert expert_kwargs["mesh"] is expert_mesh

    assert len(dense_calls) == 1
    dense_module, dense_config, dense_kwargs = dense_calls[0]
    assert dense_module is model
    assert dense_config.reshard_after_forward is True
    assert dense_config.last_unit_reshard_after_forward is True
    assert dense_kwargs["ignored_params"] == set(model.experts.parameters())


@pytest.mark.parametrize("removed_value", [False, True])
def test_fsdp2_rejects_removed_use_fp32_shards_keyword(removed_value: bool):
    with pytest.raises(
        ValueError, match="use_fp32_shards has been removed; use fsdp2_param_dtype"
    ):
        fsdp2_optimizer.build_fsdp2_training_optimizer(
            [ToyModel()],
            None,
            ParallelState(),
            unit_modules=(ToyBlock,),
            use_fp32_shards=removed_value,
        )


@pytest.mark.parametrize(
    "opt",
    [
        SimpleNamespace(fsdp2_use_fp32_shards=False),
        {"fsdp2_use_fp32_shards": True},
        SimpleNamespace(override_optimizer_config={"fsdp2_use_fp32_shards": False}),
    ],
)
def test_fsdp2_rejects_removed_use_fp32_shards_config(opt):
    with pytest.raises(
        ValueError,
        match="fsdp2_use_fp32_shards has been removed; use fsdp2_param_dtype",
    ):
        fsdp2_optimizer.build_fsdp2_training_optimizer(
            [ToyModel()], opt, ParallelState(), unit_modules=(ToyBlock,)
        )


def test_fsdp2_default_mixed_precision_policy_keeps_bf16_params_fp32_reductions():
    policy = fsdp2_wrap._mixed_precision_policy_from_config(
        FSDP2Config(param_dtype="bfloat16", reduce_dtype="float32")
    )

    assert policy.param_dtype is torch.bfloat16
    assert policy.reduce_dtype is torch.float32


def test_fsdp2_config_defaults_to_fp32_grad_and_accepts_only_storage_dtypes():
    assert FSDP2Config().grad_dtype is torch.float32
    assert FSDP2Config(grad_dtype="bfloat16").grad_dtype is torch.bfloat16
    assert FSDP2Config(grad_dtype=torch.float32).grad_dtype is torch.float32
    with pytest.raises(ValueError, match="grad_dtype"):
        FSDP2Config(grad_dtype="float16")


def test_fsdp2_fp32_main_grad_captures_post_reduce_output_without_changing_landed_grad(
    monkeypatch,
):
    monkeypatch.setattr(fsdp2_wrap.torch, "__version__", "2.11.0")
    sharded_param = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    sharded_param.grad = torch.ones_like(sharded_param)
    fsdp_param = SimpleNamespace(
        sharded_param=sharded_param,
        sharded_size=torch.Size((2,)),
        contiguous_sharded_stride=(1,),
        padded_sharded_param_size=torch.Size((2,)),
    )
    param_group = SimpleNamespace(
        fsdp_params=[fsdp_param],
        _all_reduce_hook=None,
        gradient_divide_factor=None,
        force_sum_reduction_for_comms=False,
    )
    module = ToyModel()
    module._get_fsdp_state = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        _fsdp_param_group=param_group
    )
    module.set_all_reduce_hook = lambda hook: setattr(  # type: ignore[attr-defined]
        param_group, "_all_reduce_hook", hook
    )

    touched = fsdp2_wrap.register_fsdp2_main_grad_hooks(
        module, torch.float32, recurse=False
    )
    param_group._all_reduce_hook(torch.tensor([1.25, 2.5], dtype=torch.float32))

    assert touched == 1
    assert sharded_param.dtype is torch.bfloat16
    assert sharded_param.grad.dtype is torch.bfloat16
    assert sharded_param.main_grad.dtype is torch.float32
    assert torch.equal(sharded_param.main_grad, torch.tensor([1.25, 2.5]))
    assert (
        to_local_tensor(sharded_param.main_grad).untyped_storage().data_ptr()
        != sharded_param.grad.untyped_storage().data_ptr()
    )
    state = sharded_param._mlite_fsdp2_main_grad_state
    state._clear_torch_landed_grad(sharded_param)
    assert sharded_param.grad is None

    param_group._all_reduce_hook(torch.tensor([0.75, 0.5], dtype=torch.float32))
    assert torch.equal(sharded_param.main_grad, torch.tensor([2.0, 3.0]))


def test_fsdp2_fp32_main_grad_rejects_incomplete_flat_group(monkeypatch):
    monkeypatch.setattr(fsdp2_wrap.torch, "__version__", "2.11.0")
    sharded_param = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    fsdp_param = SimpleNamespace(
        sharded_param=sharded_param,
        sharded_size=torch.Size((2,)),
        contiguous_sharded_stride=(1,),
        padded_sharded_param_size=torch.Size((2,)),
    )
    param_group = SimpleNamespace(
        fsdp_params=[fsdp_param],
        _all_reduce_hook=None,
        gradient_divide_factor=None,
        force_sum_reduction_for_comms=False,
    )
    module = ToyModel()
    module._get_fsdp_state = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        _fsdp_param_group=param_group
    )
    module.set_all_reduce_hook = lambda hook: setattr(  # type: ignore[attr-defined]
        param_group, "_all_reduce_hook", hook
    )

    fsdp2_wrap.register_fsdp2_main_grad_hooks(module, torch.float32, recurse=False)

    with pytest.raises(RuntimeError, match="expected 2 elements"):
        param_group._all_reduce_hook(torch.ones(1, dtype=torch.float32))


def test_fsdp2_optimizer_consumes_and_clears_main_grad_independent_of_state():
    param = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    fsdp_param = SimpleNamespace(
        sharded_param=param,
        sharded_size=torch.Size((2,)),
        contiguous_sharded_stride=(1,),
        padded_sharded_param_size=torch.Size((2,)),
    )
    state = fsdp2_wrap._FSDP2MainGradState(
        SimpleNamespace(fsdp_params=[fsdp_param]), torch.float32
    )
    param.grad = torch.full_like(param, 100.0)
    state.capture(torch.tensor([0.25, -0.5], dtype=torch.float32))
    state._clear_torch_landed_grad(param)
    torch_optimizer = FP32AdamW(
        [param], lr=1.0e-3, weight_decay=0.0, betas=(0.9, 0.95), eps=1.0e-8
    )
    optimizer = FSDP2Optimizer(torch_optimizer, [param], clip_grad=1.0e9)
    master = torch_optimizer.state[param]["master_param"]
    exp_avg = torch_optimizer.state[param]["exp_avg"]
    exp_avg_sq = torch_optimizer.state[param]["exp_avg_sq"]
    storages = {
        tensor.untyped_storage().data_ptr()
        for tensor in (param, param.main_grad, master, exp_avg, exp_avg_sq)
    }

    assert len(storages) == 5
    assert param.grad is None
    assert torch.equal(get_param_grad(param), torch.tensor([0.25, -0.5]))
    assert optimizer.step()[0]
    assert torch.equal(exp_avg, torch.tensor([0.025, -0.05]))

    optimizer.zero_grad()
    assert get_param_grad(param) is None
    assert not hasattr(param, "main_grad")


def test_fsdp2_gradient_dtype_rejects_unsupported_dtype(monkeypatch):
    monkeypatch.setattr(fsdp2_wrap.torch, "__version__", "2.11.0")
    with pytest.raises(ValueError, match="grad_dtype"):
        fsdp2_wrap.register_fsdp2_main_grad_hooks(ToyModel(), torch.float16)


def _fsdp2_gradient_landing_worker(rank, world, grad_dtype, port, results):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed import init_device_mesh

        model = nn.Linear(4, 2, bias=False).to(dtype=torch.bfloat16)
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        policy = fsdp2_wrap._mixed_precision_policy_from_config(
            FSDP2Config(
                param_dtype="bfloat16", reduce_dtype="float32", device_type="cpu"
            )
        )
        fsdp2_wrap._load_fully_shard()(model, mesh=mesh, mp_policy=policy)
        param_group = model._get_fsdp_state()._fsdp_param_group
        lazy_init = param_group.lazy_init
        lazy_init_impl = getattr(lazy_init, "__func__", lazy_init)
        try:
            fsdp2_wrap.register_fsdp2_main_grad_hooks(model, grad_dtype)
        except RuntimeError as exc:
            results.append(("unsupported", str(exc)))
            return
        model(torch.ones(3, 4, dtype=torch.bfloat16)).float().square().mean().backward()
        param = next(model.parameters())
        current_lazy_init = param_group.lazy_init
        current_lazy_init_impl = getattr(
            current_lazy_init, "__func__", current_lazy_init
        )
        results.append(
            (
                param.dtype,
                param.grad.to_local().dtype,
                (
                    getattr(param, "main_grad", None).dtype
                    if hasattr(param, "main_grad")
                    else None
                ),
                current_lazy_init_impl is lazy_init_impl,
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("grad_dtype", "port"), [(torch.bfloat16, 29671), (torch.float32, 29672)]
)
def test_fsdp2_gradient_dtype_lands_real_gloo_grad(grad_dtype, port):
    if not fsdp2_available():
        pytest.skip("Installed PyTorch does not expose FSDP2 fully_shard.")
    manager = multiprocessing.Manager()
    results = manager.list()
    torch.multiprocessing.spawn(
        _fsdp2_gradient_landing_worker,
        args=(2, grad_dtype, port, results),
        nprocs=2,
        join=True,
    )
    if torch.__version__.startswith("2.11."):
        expected_main_grad = torch.float32 if grad_dtype is torch.float32 else None
        assert (
            list(results)
            == [(torch.bfloat16, torch.bfloat16, expected_main_grad, True)] * 2
        )
    else:
        assert (
            list(results)
            == [("unsupported", "FSDP2 main_grad hooks require pinned PyTorch 2.11.x.")]
            * 2
        )


def test_fsdp2_training_optimizer_threads_independent_grad_dtype(monkeypatch):
    model = ToyModel()
    seen_grad_dtypes = []
    monkeypatch.setattr(
        fsdp2_optimizer, "wrap_fsdp2", lambda model, *_args, **_kwargs: model
    )
    monkeypatch.setattr(
        fsdp2_optimizer,
        "register_fsdp2_main_grad_hooks",
        lambda module, dtype: seen_grad_dtypes.append((module, dtype)),
    )
    monkeypatch.setattr(
        fsdp2_optimizer, "build_fsdp2_adamw", lambda *_args, **_kwargs: object()
    )

    fsdp2_optimizer.build_fsdp2_training_optimizer(
        [model], None, ParallelState(), unit_modules=(ToyBlock,), grad_dtype="float32"
    )

    assert seen_grad_dtypes == [(model, torch.float32)]


@pytest.mark.parametrize(
    ("param_dtype", "grad_dtype"),
    [
        ("bfloat16", "bfloat16"),
        ("bfloat16", "float32"),
        ("float32", "bfloat16"),
        ("float32", "float32"),
    ],
)
def test_fsdp2_training_optimizer_applies_orthogonal_storage_dtypes(
    monkeypatch, param_dtype, grad_dtype
):
    model = ToyModel().to(dtype=torch.float32)
    seen_grad_dtypes = []
    monkeypatch.setattr(
        fsdp2_optimizer, "wrap_fsdp2", lambda model, *_args, **_kwargs: model
    )
    monkeypatch.setattr(
        fsdp2_optimizer,
        "register_fsdp2_main_grad_hooks",
        lambda _module, dtype: seen_grad_dtypes.append(dtype),
    )
    monkeypatch.setattr(
        fsdp2_optimizer, "build_fsdp2_adamw", lambda *_args, **_kwargs: object()
    )

    fsdp2_optimizer.build_fsdp2_training_optimizer(
        [model],
        SimpleNamespace(fsdp2_param_dtype=param_dtype, fsdp2_grad_dtype=grad_dtype),
        ParallelState(),
        unit_modules=(ToyBlock,),
    )

    expected_param_dtype = getattr(torch, param_dtype)
    expected_grad_dtype = getattr(torch, grad_dtype)
    assert {param.dtype for param in model.parameters()} == {expected_param_dtype}
    assert seen_grad_dtypes == [expected_grad_dtype]


@pytest.mark.parametrize("invalid", [None, "float16", "not-a-dtype"])
def test_fsdp2_training_optimizer_rejects_invalid_storage_dtype(monkeypatch, invalid):
    model = ToyModel()
    monkeypatch.setattr(
        fsdp2_optimizer, "wrap_fsdp2", lambda model, *_args, **_kwargs: model
    )
    with pytest.raises(ValueError, match="fsdp2_param_dtype"):
        fsdp2_optimizer.build_fsdp2_training_optimizer(
            [model],
            SimpleNamespace(fsdp2_param_dtype=invalid),
            ParallelState(),
            unit_modules=(ToyBlock,),
        )


def test_fsdp2_training_optimizer_rejects_non_fp32_reduce_dtype():
    with pytest.raises(ValueError, match="reduce_dtype must be float32"):
        fsdp2_optimizer.build_fsdp2_training_optimizer(
            [ToyModel()],
            None,
            ParallelState(),
            unit_modules=(ToyBlock,),
            reduce_dtype="bfloat16",
        )


def test_fsdp2_fp32_main_grad_requires_fp32_master(monkeypatch):
    monkeypatch.setattr(
        fsdp2_optimizer, "wrap_fsdp2", lambda model, *_args, **_kwargs: model
    )
    with pytest.raises(ValueError, match="requires fsdp2_use_fp32_master=True"):
        fsdp2_optimizer.build_fsdp2_training_optimizer(
            [ToyModel()],
            None,
            ParallelState(),
            unit_modules=(ToyBlock,),
            grad_dtype="float32",
            use_fp32_master=False,
        )


def test_removed_fp32_shards_path_preserves_loss_and_gradients_bitwise_cpu(monkeypatch):
    monkeypatch.setattr(
        fsdp2_optimizer, "wrap_fsdp2", lambda model, *_args, **_kwargs: model
    )
    monkeypatch.setattr(
        fsdp2_optimizer, "register_fsdp2_main_grad_hooks", lambda *_args, **_kwargs: 1
    )
    torch.manual_seed(1234)
    actual_model = ToyModel().to(dtype=torch.bfloat16)
    expected_model = copy.deepcopy(actual_model)
    opt = SimpleNamespace(
        optimizer="adam",
        lr=1.0e-3,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        clip_grad=1.0e9,
        offload_fraction=0.0,
    )
    actual_optimizer = fsdp2_optimizer.build_fsdp2_training_optimizer(
        [actual_model],
        opt,
        ParallelState(),
        unit_modules=(ToyBlock,),
        use_fp32_master=True,
        adamw_foreach=False,
    )
    # This is the old use_fp32_shards=False path: wrapping leaves BF16 model
    # parameters intact, then the optimizer owns the only FP32 parameter copy.
    expected_optimizer = fsdp2_optimizer.build_fsdp2_adamw(
        [expected_model],
        opt,
        ParallelState(),
        use_fp32_master=True,
        adamw_foreach=False,
    )
    actual_params = list(actual_model.parameters())
    actual_masters = actual_optimizer.optimizer.state_dict()["master_params"]
    assert all(
        param.dtype is torch.bfloat16 and param.element_size() == 2
        for param in actual_params
    )
    assert all(
        master.dtype is torch.float32 and master.element_size() == 4
        for master in actual_masters
    )
    assert all(
        param.untyped_storage().data_ptr() != master.untyped_storage().data_ptr()
        for param, master in zip(actual_params, actual_masters, strict=True)
    )
    inputs = [
        torch.randn(
            3, 4, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(seed)
        )
        for seed in range(5)
    ]

    for x in inputs:
        actual_optimizer.zero_grad()
        expected_optimizer.zero_grad()
        actual_loss = actual_model(x).float().square().mean()
        expected_loss = expected_model(x).float().square().mean()
        actual_loss.backward()
        expected_loss.backward()
        _assert_loss_and_gradients_bitwise_equal(
            actual_loss, expected_loss, actual_model, expected_model
        )
        assert actual_optimizer.step()[0]
        assert expected_optimizer.step()[0]
        for actual_param, expected_param in zip(
            actual_model.parameters(), expected_model.parameters(), strict=True
        ):
            assert torch.equal(actual_param, expected_param)


def test_bitwise_loss_and_gradient_check_detects_perturbation():
    actual_model = nn.Linear(2, 1, bias=False)
    expected_model = copy.deepcopy(actual_model)
    actual_model.weight.grad = torch.tensor([[1.0, 2.0]])
    expected_model.weight.grad = torch.tensor([[1.0, 2.5]])

    with pytest.raises(AssertionError):
        _assert_loss_and_gradients_bitwise_equal(
            torch.tensor(1.0), torch.tensor(1.0), actual_model, expected_model
        )


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_fsdp2_pipeline_preserves_prefetch_depth(depth: int):
    ps = SimpleNamespace(pp_size=4)

    assert fsdp2_optimizer._fsdp2_prefetch_depth(ps, default_depth=depth) == depth


@pytest.mark.parametrize("field", ["mesh_dim_name", "device_type"])
def test_fsdp2_config_rejects_empty_names(field: str):
    with pytest.raises(ValueError, match=field):
        FSDP2Config(**{field: ""})


def test_fsdp2_config_normalizes_unit_and_leaf_modules():
    cfg = FSDP2Config(unit_modules=[nn.Linear], leaf_module_names=["embed"])

    assert cfg.unit_modules == (nn.Linear,)
    assert cfg.leaf_module_names == ("embed",)
    assert isinstance(fsdp2_available(), bool)


def test_fsdp2_optimizer_offloads_dtensor_state_without_extra_knob(monkeypatch):
    model = ToyModel()
    torch_optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    optimizer = FSDP2Optimizer(torch_optimizer, model.parameters())
    calls: list[bool] = []

    def fake_move_optimizer_state_to_cpu(
        _optimizer, _offloaded_state, *, include_dtensor_state
    ):
        calls.append(include_dtensor_state)

    monkeypatch.setattr(
        fsdp2_optimizer, "move_optimizer_state_to_cpu", fake_move_optimizer_state_to_cpu
    )

    optimizer.offload_state_to_cpu()

    assert calls == [True]
    assert not hasattr(optimizer, "optimizer_offload_dtensor_state")


def test_fsdp2_grad_sync_enabled_propagates_to_fsdp2_roots():
    class FakeFSDP2Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[tuple[bool, bool]] = []

        def set_requires_gradient_sync(
            self, requires_gradient_sync: bool, *, recurse: bool
        ) -> None:
            self.calls.append((requires_gradient_sync, recurse))

    param = nn.Parameter(torch.tensor([1.0]))
    root = FakeFSDP2Root()
    optimizer = FSDP2Optimizer(
        torch.optim.SGD([param], lr=0.0), [param], fsdp_modules=[root]
    )

    optimizer.grad_sync_enabled = True
    optimizer.grad_sync_enabled = False

    assert root.calls == [(True, True), (False, True)]


def test_fsdp2_shard_placement_prefers_first_divisible_dimension():
    placement_for_two = build_fsdp2_shard_placement_fn(2)
    placement_for_three = build_fsdp2_shard_placement_fn(3)

    assert placement_for_two(nn.Parameter(torch.empty(3, 4))).dim == 1
    assert placement_for_three(nn.Parameter(torch.empty(3, 4))).dim == 0


def test_fsdp2_shard_placement_rejects_invalid_group_size():
    with pytest.raises(ValueError, match="positive"):
        build_fsdp2_shard_placement_fn(0)


def test_fsdp2_rejects_invalid_unit_path():
    with pytest.raises(ValueError, match="Invalid FSDP2 unit module path"):
        fsdp2_wrap._resolve_unit_module_types(("Linear",))


def test_fsdp2_rejects_non_module_unit_path():
    with pytest.raises(TypeError, match="does not resolve"):
        fsdp2_wrap._resolve_unit_module_types(("math.sqrt",))


def test_wrap_fsdp2_requires_distributed_when_mesh_is_not_provided(monkeypatch):
    monkeypatch.setattr(
        fsdp2_wrap, "_load_fully_shard", lambda: lambda module, **kwargs: module
    )

    with pytest.raises(RuntimeError, match="torch.distributed"):
        fsdp2_wrap.wrap_fsdp2(ToyModel(), ParallelState(), FSDP2Config())


def test_wrap_fsdp2_wraps_units_then_root_and_preserves_param_attrs(monkeypatch):
    model = ToyModel()
    model.block.proj.weight.tensor_model_parallel = True
    calls: list[nn.Module] = []

    def fake_fully_shard(module, **kwargs):
        calls.append(module)
        for param in module.parameters():
            vars(param).clear()
        module._fake_fsdp2_kwargs = kwargs
        return module

    monkeypatch.setattr(fsdp2_wrap, "_load_fully_shard", lambda: fake_fully_shard)

    result = fsdp2_wrap.wrap_fsdp2(
        model,
        ParallelState(),
        FSDP2Config(unit_modules=(ToyBlock,), reshard_after_forward=False),
        mesh=SimpleNamespace(name="mesh"),
    )

    assert result is model
    assert calls == [model.block, model]
    assert model.block.proj.weight.tensor_model_parallel is True
    assert model._fake_fsdp2_kwargs["reshard_after_forward"] is False
    assert model._fake_fsdp2_kwargs["mesh"].name == "mesh"


def test_wrap_fsdp2_accepts_unit_module_import_paths(monkeypatch):
    model = ToyModel()
    calls: list[nn.Module] = []

    def fake_fully_shard(module, **kwargs):
        calls.append(module)
        return module

    monkeypatch.setattr(fsdp2_wrap, "_load_fully_shard", lambda: fake_fully_shard)

    fsdp2_wrap.wrap_fsdp2(
        model,
        ParallelState(),
        FSDP2Config(unit_modules=("torch.nn.modules.linear.Linear",), wrap_root=False),
        mesh=SimpleNamespace(name="mesh"),
    )

    assert calls == [model.block.proj, model.out]


def test_wrap_fsdp2_uses_container_order_without_nested_unit_duplicates(monkeypatch):
    model = nn.Module()
    model.layers = nn.ModuleDict(
        {"10": NestedToyBlock(), "2": ToyBlock(), "11": ToyBlock()}
    )
    calls: list[nn.Module] = []

    def fake_fully_shard(module, **kwargs):
        calls.append(module)
        module._fake_fsdp2_kwargs = kwargs
        return module

    monkeypatch.setattr(fsdp2_wrap, "_load_fully_shard", lambda: fake_fully_shard)

    fsdp2_wrap.wrap_fsdp2(
        model,
        ParallelState(),
        FSDP2Config(unit_modules=(ToyBlock,), reshard_after_forward=True),
        mesh=SimpleNamespace(name="mesh"),
    )

    assert calls == [model.layers["10"], model.layers["2"], model.layers["11"], model]
    assert model.layers["10"]._fake_fsdp2_kwargs["reshard_after_forward"] is True
    assert not hasattr(model.layers["10"].inner, "_fake_fsdp2_kwargs")
    assert model.layers["11"]._fake_fsdp2_kwargs["reshard_after_forward"] is False


def test_wrap_fsdp2_configures_default_forward_prefetch(monkeypatch):
    model = TwoBlockModel()
    calls: list[nn.Module] = []

    def fake_fully_shard(module, **kwargs):
        calls.append(module)
        module._forward_prefetch = None
        module._backward_prefetch = None
        module._fake_fsdp2_kwargs = kwargs

        def set_forward_prefetch(modules, *, _module=module):
            _module._forward_prefetch = list(modules)

        def set_backward_prefetch(modules, *, _module=module):
            _module._backward_prefetch = list(modules)

        module.set_modules_to_forward_prefetch = set_forward_prefetch
        module.set_modules_to_backward_prefetch = set_backward_prefetch
        return module

    monkeypatch.setattr(fsdp2_wrap, "_load_fully_shard", lambda: fake_fully_shard)

    fsdp2_wrap.wrap_fsdp2(
        model,
        ParallelState(),
        FSDP2Config(unit_modules=(ToyBlock,), reshard_after_forward=True),
        mesh=SimpleNamespace(name="mesh"),
    )

    assert calls == [model.block0, model.block1, model]
    assert model.block0._fake_fsdp2_kwargs["reshard_after_forward"] is True
    assert model.block1._fake_fsdp2_kwargs["reshard_after_forward"] is False
    assert model._fake_fsdp2_kwargs["reshard_after_forward"] is False
    assert model._forward_prefetch == [model.block0]
    assert model.block0._forward_prefetch == [model.block1]
    assert model.block1._backward_prefetch is None


def test_wrap_fsdp2_prefetch_depths(monkeypatch):
    model = nn.Sequential(ToyBlock(), ToyBlock(), ToyBlock())

    def fake_fully_shard(module, **kwargs):
        module._forward_prefetch = None
        module._backward_prefetch = None

        def set_forward_prefetch(modules, *, _module=module):
            _module._forward_prefetch = list(modules)

        def set_backward_prefetch(modules, *, _module=module):
            _module._backward_prefetch = list(modules)

        module.set_modules_to_forward_prefetch = set_forward_prefetch
        module.set_modules_to_backward_prefetch = set_backward_prefetch
        return module

    monkeypatch.setattr(fsdp2_wrap, "_load_fully_shard", lambda: fake_fully_shard)

    fsdp2_wrap.wrap_fsdp2(
        model,
        ParallelState(),
        FSDP2Config(
            unit_modules=(ToyBlock,),
            wrap_root=False,
            forward_prefetch_depth=2,
            backward_prefetch_depth=2,
        ),
        mesh=SimpleNamespace(name="mesh"),
    )

    assert model[0]._forward_prefetch == [model[1], model[2]]
    assert model[1]._forward_prefetch == [model[2]]
    assert model[2]._backward_prefetch == [model[1], model[0]]


def test_clip_grads_with_sharded_norm_scales_cpu_grads_once():
    p0 = nn.Parameter(torch.ones(2))
    p1 = nn.Parameter(torch.ones(2))
    p0.grad = torch.tensor([3.0, 4.0])
    p1.grad = torch.tensor([0.0, 12.0])

    clip_grads_with_sharded_norm_([p0, p1], max_norm=6.5, total_norm=torch.tensor(13.0))

    scale = 6.5 / (13.0 + 1.0e-6)
    torch.testing.assert_close(p0.grad, torch.tensor([3.0, 4.0]) * scale)
    torch.testing.assert_close(p1.grad, torch.tensor([0.0, 12.0]) * scale)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for NCCL scalar test."
)
def test_all_reduce_scalar_moves_cpu_value_to_nccl_device(monkeypatch):
    group = object()
    reduced_devices: list[str] = []

    def fake_all_reduce(value, *, op, group):
        assert group is fake_all_reduce.group
        assert op == dist.ReduceOp.SUM
        reduced_devices.append(value.device.type)
        value.add_(5.0)

    fake_all_reduce.group = group
    monkeypatch.setattr(fsdp2_grad_clip.dist, "get_backend", lambda _group: "nccl")
    monkeypatch.setattr(fsdp2_grad_clip.dist, "all_reduce", fake_all_reduce)

    value = torch.tensor(7.0)
    all_reduce_scalar_(value, op=dist.ReduceOp.SUM, group=group)

    assert reduced_devices == ["cuda"]
    assert value.device.type == "cpu"
    torch.testing.assert_close(value, torch.tensor(12.0))


def test_fsdp2_optimizer_uses_scalar_all_reduce_for_all_norm_groups(monkeypatch):
    groups = SimpleNamespace(
        dp_cp=object(), tp=object(), replicated=object(), expert=object(), pp=object()
    )
    reduced_groups: list[object] = []

    def fake_all_reduce_scalar(value, *, op, group):
        assert value.ndim == 0
        assert op == dist.ReduceOp.SUM
        reduced_groups.append(group)

    monkeypatch.setattr(fsdp2_optimizer, "all_reduce_scalar_", fake_all_reduce_scalar)
    monkeypatch.setattr(fsdp2_optimizer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp2_optimizer.dist, "get_world_size", lambda _group: 2)

    sharded = nn.Parameter(torch.tensor([1.0]))
    replicated = nn.Parameter(torch.tensor([1.0]))
    expert = nn.Parameter(torch.tensor([1.0]))
    tp_replicated = nn.Parameter(torch.tensor([1.0]))
    sharded.grad = torch.tensor([2.0])
    replicated.grad = torch.tensor([3.0])
    expert.grad = torch.tensor([4.0])
    tp_replicated.grad = torch.tensor([5.0])

    optimizer = FSDP2Optimizer(
        torch.optim.SGD([sharded, replicated, expert, tp_replicated], lr=0.0),
        [sharded, replicated, expert, tp_replicated],
        ParallelState(dp_cp_group=groups.dp_cp, tp_group=groups.tp, pp_group=groups.pp),
        clip_grad=100.0,
        replicated_grad_params=[replicated],
        replicated_grad_norm_group=groups.replicated,
        expert_sharded_grad_params=[expert],
        expert_sharded_grad_norm_group=groups.expert,
        tp_replicated_grad_params=[tp_replicated],
    )

    assert optimizer.clip_grad_norm() == pytest.approx(
        (2.0**2 + 3.0**2 + 4.0**2 + 5.0**2) ** 0.5
    )
    assert reduced_groups == [
        groups.dp_cp,
        groups.tp,
        groups.replicated,
        groups.expert,
        groups.pp,
    ]


def test_fp32_adamw_state_dict_roundtrip_cpu():
    param = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    optimizer = build_adamw_optimizer(
        [{"params": [param], "weight_decay": 0.0}],
        all_params=[param],
        lr=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        foreach=False,
        use_fp32_master=True,
        cpu_update=False,
        opt=SimpleNamespace(),
    )
    param.grad = torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    optimizer.step()
    state = optimizer.state_dict()
    expected = {
        key: state[key][0].clone()
        for key in ("master_params", "exp_avgs", "exp_avg_sqs")
    }
    for key, value in expected.items():
        state[key][0] = SimpleNamespace(_local_tensor=value)

    loaded_param = nn.Parameter(torch.tensor([9.0, 9.0], dtype=torch.bfloat16))
    loaded_optimizer = build_adamw_optimizer(
        [{"params": [loaded_param], "weight_decay": 0.0}],
        all_params=[loaded_param],
        lr=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        foreach=False,
        use_fp32_master=True,
        cpu_update=False,
        opt=SimpleNamespace(),
    )
    loaded_optimizer.load_state_dict(state)
    loaded_state = loaded_optimizer.state_dict()

    assert loaded_state["step_count"] == state["step_count"]
    torch.testing.assert_close(
        loaded_param, expected["master_params"].to(torch.bfloat16)
    )
    for key, value in expected.items():
        torch.testing.assert_close(loaded_state[key][0], value)
    assert loaded_state["steps"] == state["steps"]


@pytest.mark.parametrize("cpu_update", [False, True])
def test_fp32_adamw_load_matches_uninterrupted_next_step_cpu(cpu_update: bool):
    def build(initial_value: torch.Tensor):
        param = nn.Parameter(initial_value.clone().to(dtype=torch.bfloat16))
        optimizer = build_adamw_optimizer(
            [{"params": [param], "weight_decay": 0.0}],
            all_params=[param],
            lr=0.1,
            weight_decay=0.0,
            betas=(0.9, 0.99),
            eps=1.0e-8,
            foreach=False,
            use_fp32_master=True,
            cpu_update=cpu_update,
            opt=SimpleNamespace(),
        )
        return param, optimizer

    initial = torch.tensor([1.0, -2.0], dtype=torch.float32)
    first_grad = torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    second_grad = torch.tensor([-0.125, 0.375], dtype=torch.bfloat16)

    ckpt_param, ckpt_optimizer = build(initial)
    direct_param, direct_optimizer = build(initial)
    loaded_param, loaded_optimizer = build(initial)

    ckpt_param.grad = first_grad.clone()
    ckpt_optimizer.step()
    direct_param.grad = first_grad.clone()
    direct_optimizer.step()

    saved_param = ckpt_param.detach().clone()
    saved_state = copy.deepcopy(ckpt_optimizer.state_dict())

    with torch.no_grad():
        loaded_param.copy_(saved_param)
    loaded_optimizer.load_state_dict(saved_state)

    direct_param.grad = second_grad.clone()
    direct_optimizer.step()
    loaded_param.grad = second_grad.clone()
    loaded_optimizer.step()

    torch.testing.assert_close(loaded_param, direct_param, atol=0.0, rtol=0.0)
    direct_state = direct_optimizer.state_dict()
    loaded_state = loaded_optimizer.state_dict()
    assert loaded_state["step_count"] == direct_state["step_count"]
    for key in ("master_params", "exp_avgs", "exp_avg_sqs"):
        torch.testing.assert_close(
            loaded_state[key][0], direct_state[key][0], atol=0.0, rtol=0.0
        )
    assert loaded_state["steps"] == direct_state["steps"]
