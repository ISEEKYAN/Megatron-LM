# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Correctness tests for the FSDP2 Muon lowering.

The core guarantee (AC#3) is that the sharded ``local shard -> bounded
full-matrix gather -> Newton-Schulz -> local reshard`` path produces, on every
data-parallel rank, exactly the local slice of an **independent full-matrix**
Muon reference. We verify this with real gloo DTensors at DP in {1, 2, 4} over
several shard layouts (row-even, column, row-padded), covering momentum EMA,
Newton-Schulz, weight decay and the update across multiple steps.

The reference below re-implements the Muon step on the *unsharded* matrix and
exercises none of the DTensor gather/reshard machinery, so a match isolates the
sharding lowering as correct. It reuses the trusted ``newton_schulz`` /
``muon_update_scale`` primitives (the orthogonalization math itself is a port of
the pinned Megatron reference and is checked separately for orthogonality).
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.lite.primitive.optimizers.fsdp2.muon import (
    FP32Muon,
    _fp32_matmul_precision,
    build_muon_chained_optimizer,
    muon_update_scale,
    newton_schulz_orthogonalize,
    split_muon_and_fallback_params,
)


# ---------------------------------------------------------------------------
# Independent full-matrix Muon reference (no sharding).
# ---------------------------------------------------------------------------
def _orthogonalize_full(update, *, steps, coeff, scale_mode, extra, split_shapes, matmul_prec):
    with _fp32_matmul_precision(matmul_prec):
        if split_shapes:
            grad_shape = update.shape
            qkv_dim = sum(split_shapes)
            groups = grad_shape[0] // qkv_dim
            parts = torch.split(update.view(groups, qkv_dim, -1), split_shapes, dim=1)
            parts = [p.reshape(-1, grad_shape[-1]) for p in parts]
            orth_parts = []
            for p in parts:
                o = newton_schulz_orthogonalize(p, steps, coeff)
                o = o * muon_update_scale(p.size(-2), p.size(-1), scale_mode) * extra
                orth_parts.append(o.view(groups, -1, grad_shape[-1]))
            return torch.cat(orth_parts, dim=1).view(grad_shape)
        o = newton_schulz_orthogonalize(update, steps, coeff)
        return o * muon_update_scale(update.size(-2), update.size(-1), scale_mode) * extra


def _oracle_muon(full_weight, grads, *, lr, wd, momentum, steps, coeff, scale_mode, extra,
                 nesterov, decoupled, split_shapes, matmul_prec):
    master = full_weight.clone().to(torch.float32)
    buffer = torch.zeros_like(master)
    for grad in grads:
        grad = grad.clone().to(torch.float32)
        if wd != 0.0:
            if decoupled:
                master.mul_(1.0 - lr * wd)
            else:
                grad = grad.add(master, alpha=wd)
        buffer.lerp_(grad, 1.0 - momentum)
        update = grad.lerp(buffer, momentum) if nesterov else buffer
        orth = _orthogonalize_full(
            update, steps=steps, coeff=coeff, scale_mode=scale_mode, extra=extra,
            split_shapes=split_shapes, matmul_prec=matmul_prec,
        )
        master.add_(orth, alpha=-lr)
    return master


# ---------------------------------------------------------------------------
# Distributed worker: run FP32Muon over a real DTensor-sharded param.
# ---------------------------------------------------------------------------
_MUON_KWARGS = dict(
    lr=0.1, momentum=0.9, weight_decay=0.05, nesterov=False, decoupled=True,
    steps=5, coeff="quintic", scale_mode="spectral", extra=1.0, matmul_prec="medium",
)


def _shard_worker(rank, world_size, store_path, rows, cols, num_steps, split_shapes, matmul_prec):
    from torch.distributed import DeviceMesh
    from torch.distributed.tensor import distribute_tensor

    from megatron.lite.primitive.optimizers.fsdp2.wrap import build_fsdp2_shard_placement_fn

    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size)
    try:
        mesh = DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("dp",))

        torch.manual_seed(20260711)
        full_weight = torch.randn(rows, cols, dtype=torch.float32)
        grads = [torch.randn(rows, cols, dtype=torch.float32) for _ in range(num_steps)]

        placement = build_fsdp2_shard_placement_fn(world_size)(full_weight)
        param = torch.nn.Parameter(distribute_tensor(full_weight.clone(), mesh, [placement]))
        if split_shapes:
            param.is_qkv = True
            param.qkv_split_shapes = list(split_shapes)

        opt = FP32Muon(
            [{"params": [param], "weight_decay": _MUON_KWARGS["weight_decay"]}],
            lr=_MUON_KWARGS["lr"],
            momentum=_MUON_KWARGS["momentum"],
            weight_decay=_MUON_KWARGS["weight_decay"],
            nesterov=_MUON_KWARGS["nesterov"],
            use_decoupled_weight_decay=_MUON_KWARGS["decoupled"],
            split_qkv=bool(split_shapes),
            num_ns_steps=_MUON_KWARGS["steps"],
            coefficient_type=_MUON_KWARGS["coeff"],
            scale_mode=_MUON_KWARGS["scale_mode"],
            extra_scale_factor=_MUON_KWARGS["extra"],
            fp32_matmul_prec=matmul_prec,
        )
        for grad in grads:
            param.grad = distribute_tensor(grad.clone(), mesh, [placement])
            opt.step()

        oracle_master = _oracle_muon(
            full_weight, grads, split_shapes=split_shapes,
            **{k: _MUON_KWARGS[k] for k in
               ("lr", "momentum", "steps", "coeff", "scale_mode", "extra",
                "nesterov", "decoupled")},
            wd=_MUON_KWARGS["weight_decay"],
            matmul_prec=matmul_prec,
        )
        expected_local = distribute_tensor(oracle_master, mesh, [placement]).to_local()
        got_local = param.detach().to_local()

        if not torch.allclose(got_local, expected_local, atol=1e-5, rtol=1e-4):
            max_diff = (got_local - expected_local).abs().max().item()
            raise AssertionError(
                f"rank {rank}/{world_size} shard mismatch (placement={placement}, "
                f"shape=({rows},{cols}), split={split_shapes}, prec={matmul_prec}): "
                f"max_diff={max_diff}"
            )
    finally:
        dist.destroy_process_group()


def _run_shard_case(world_size, rows, cols, num_steps, split_shapes, matmul_prec=None):
    if matmul_prec is None:
        matmul_prec = _MUON_KWARGS["matmul_prec"]
    with tempfile.TemporaryDirectory() as tmp:
        store_path = os.path.join(tmp, "store")
        mp.spawn(
            _shard_worker,
            args=(world_size, store_path, rows, cols, num_steps, split_shapes, matmul_prec),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_muon_sharded_matches_full_matrix_oracle_row_even(world_size):
    # rows divisible by world size -> Shard(0), even shards.
    _run_shard_case(world_size, rows=8, cols=12, num_steps=3, split_shapes=None)


@pytest.mark.parametrize("world_size", [2, 4])
def test_muon_sharded_matches_full_matrix_oracle_column_shard(world_size):
    # rows not divisible, cols divisible -> Shard(1).
    _run_shard_case(world_size, rows=10, cols=12, num_steps=3, split_shapes=None)


@pytest.mark.parametrize("world_size", [4])
def test_muon_sharded_matches_full_matrix_oracle_row_padded(world_size):
    # neither dim divisible -> Shard(0) with padded last shard.
    _run_shard_case(world_size, rows=10, cols=14, num_steps=3, split_shapes=None)


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_muon_sharded_qkv_split_matches_oracle(world_size):
    # Fused QKV weight: rows = num_query_groups * (q + k + v) head slices.
    # 2 groups * (4 + 2 + 2) = 16 rows, sharded over DP.
    _run_shard_case(world_size, rows=16, cols=12, num_steps=2, split_shapes=(4, 2, 2))


@pytest.mark.parametrize("world_size", [2, 4])
def test_muon_distributed_ns_matches_full_matrix_oracle(world_size):
    # rows >= cols and Shard(0) -> `_should_use_distributed_ns` is True, so the
    # step routes through `newton_schulz_tp(tp_mode="distributed")` (the rewrite's
    # whole point: no gather, NS split across ranks via all-reduce). This asserts
    # the *distributed* NS path is per-value equal to the unsharded full-matrix
    # oracle -- not just that it avoids the gather (that is checked separately in
    # ``test_muon_distributed_ns_avoids_redundant_gather``). rows=16 is divisible
    # by 2 and 4 so the row-shards stay even.
    #
    # We pin ``highest`` fp32 matmul precision here: the distributed NS reduces a
    # per-rank Gram matrix via all-reduce, so its matmul/accumulation *order*
    # differs from the single full-matrix NS the oracle runs. Under exact fp32
    # both compute the identical Newton-Schulz iteration, so they agree to fp32
    # round-off (~1e-6). See ``test_muon_distributed_ns_medium_precision_bounded``
    # for the production ``medium`` precision case, where that different order
    # combined with reduced-precision matmul produces a *bounded* ~1e-2 drift --
    # the same benign divergence the Megatron distributed (DistOpt) arm has vs a
    # naive reference, not an algorithmic error.
    _run_shard_case(
        world_size, rows=16, cols=8, num_steps=3, split_shapes=None, matmul_prec="highest"
    )


@pytest.mark.parametrize("world_size", [2])
def test_muon_distributed_ns_medium_precision_bounded(world_size):
    # Same distributed-NS route as above but at the production ``medium`` matmul
    # precision. The distributed Gram all-reduce runs a different reduced-precision
    # accumulation order than the oracle's single full-matrix NS, so the per-value
    # match loosens from ~1e-6 (highest) to a *bounded* few-1e-2. This pins that
    # the divergence is a reduced-precision artifact (bounded, expected), not an
    # unbounded correctness failure. Threshold is generous but finite: a real bug
    # (e.g. wrong transpose / partition_dim) would blow past it by orders.
    with tempfile.TemporaryDirectory() as tmp:
        store_path = os.path.join(tmp, "store")
        mp.spawn(
            _bounded_drift_worker,
            args=(world_size, store_path),
            nprocs=world_size,
            join=True,
        )


def _bounded_drift_worker(rank, world_size, store_path):
    from torch.distributed import DeviceMesh
    from torch.distributed.tensor import distribute_tensor

    from megatron.lite.primitive.optimizers.fsdp2.wrap import build_fsdp2_shard_placement_fn

    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size)
    try:
        mesh = DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("dp",))
        torch.manual_seed(20260711)
        rows, cols, num_steps = 16, 8, 3
        full_weight = torch.randn(rows, cols, dtype=torch.float32)
        grads = [torch.randn(rows, cols, dtype=torch.float32) for _ in range(num_steps)]

        placement = build_fsdp2_shard_placement_fn(world_size)(full_weight)
        param = torch.nn.Parameter(distribute_tensor(full_weight.clone(), mesh, [placement]))
        opt = FP32Muon(
            [{"params": [param], "weight_decay": _MUON_KWARGS["weight_decay"]}],
            lr=_MUON_KWARGS["lr"], momentum=_MUON_KWARGS["momentum"],
            weight_decay=_MUON_KWARGS["weight_decay"],
            use_decoupled_weight_decay=_MUON_KWARGS["decoupled"],
            num_ns_steps=_MUON_KWARGS["steps"], coefficient_type=_MUON_KWARGS["coeff"],
            scale_mode=_MUON_KWARGS["scale_mode"], extra_scale_factor=_MUON_KWARGS["extra"],
            fp32_matmul_prec="medium",
        )
        for grad in grads:
            param.grad = distribute_tensor(grad.clone(), mesh, [placement])
            opt.step()

        oracle_master = _oracle_muon(
            full_weight, grads, split_shapes=None,
            **{k: _MUON_KWARGS[k] for k in
               ("lr", "momentum", "steps", "coeff", "scale_mode", "extra",
                "nesterov", "decoupled")},
            wd=_MUON_KWARGS["weight_decay"], matmul_prec="medium",
        )
        expected_local = distribute_tensor(oracle_master, mesh, [placement]).to_local()
        got_local = param.detach().to_local()
        max_diff = (got_local - expected_local).abs().max().item()
        # Loose (few-1e-2) but finite: reduced-precision drift, not a broken algo.
        assert max_diff < 5e-2, f"rank {rank}: unbounded distributed-NS drift {max_diff}"
        # And it must be genuinely larger than the highest-precision agreement,
        # confirming the gap is the reduced-precision matmul, not noise.
        assert max_diff > 1e-4, f"rank {rank}: unexpectedly tight at medium ({max_diff})"
    finally:
        dist.destroy_process_group()


def _distributed_ns_worker(rank, world_size, store_path):
    from unittest.mock import patch

    from torch.distributed import DeviceMesh
    from torch.distributed.tensor import DTensor, distribute_tensor

    from megatron.lite.primitive.optimizers.fsdp2.wrap import build_fsdp2_shard_placement_fn

    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size)
    try:
        mesh = DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("dp",))
        torch.manual_seed(42)
        # out_features >= in_features so row-shard distributed NS is valid.
        full_weight = torch.randn(12, 8, dtype=torch.float32)
        placement = build_fsdp2_shard_placement_fn(world_size)(full_weight)
        param = torch.nn.Parameter(distribute_tensor(full_weight.clone(), mesh, [placement]))
        param.grad = distribute_tensor(torch.randn(12, 8), mesh, [placement])

        opt = FP32Muon(
            [{"params": [param], "weight_decay": 0.0}],
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0,
        )

        def _forbid_all_gather(*args, **kwargs):
            raise AssertionError("redundant all_gather called during distributed Muon NS")

        def _forbid_full_tensor(self):
            raise AssertionError("redundant full_tensor gather called during distributed Muon NS")

        with patch("torch.distributed.all_gather", side_effect=_forbid_all_gather):
            with patch.object(DTensor, "full_tensor", _forbid_full_tensor):
                opt.step()
    finally:
        dist.destroy_process_group()


def test_muon_distributed_ns_avoids_redundant_gather():
    """Distributed NS must not all_gather / full_tensor the momentum matrix."""
    with tempfile.TemporaryDirectory() as tmp:
        store_path = os.path.join(tmp, "store")
        mp.spawn(_distributed_ns_worker, args=(2, store_path), nprocs=2, join=True)


# ---------------------------------------------------------------------------
# Non-distributed CPU unit tests: primitives, routing, facade.
# ---------------------------------------------------------------------------
def test_newton_schulz_produces_near_orthogonal_rows():
    torch.manual_seed(0)
    g = torch.randn(6, 16, dtype=torch.float32)
    with torch.no_grad(), _fp32_matmul_precision("highest"):
        o = newton_schulz_orthogonalize(g, steps=5, coefficient_type="quintic")
    # Rows should be near-orthonormal: O @ O^T ~= I for the short dimension.
    gram = o @ o.mT
    assert torch.allclose(gram, torch.eye(6), atol=0.1)


def test_muon_update_scale_modes():
    assert muon_update_scale(8, 4, "spectral") == pytest.approx(8**0.5)
    assert muon_update_scale(8, 4, "unit_rms_norm") == pytest.approx((8 / 4) ** 0.5)
    with pytest.raises(ValueError):
        muon_update_scale(8, 4, "bogus")


def test_newton_schulz_rejects_non_fp32_and_non_2d():
    with pytest.raises(TypeError):
        newton_schulz_orthogonalize(torch.randn(4, 4, dtype=torch.bfloat16), steps=3)
    with pytest.raises(ValueError):
        newton_schulz_orthogonalize(torch.randn(4, 4, 4), steps=3)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8))  # 2D matrix -> Muon
        self.norm = torch.nn.Parameter(torch.randn(8))       # 1D -> Adam fallback


def test_routing_splits_matrix_vs_fallback():
    model = _Toy()
    # mimic the pre-wrap tagging: matrix weight is Muon-managed, 1D norm is not.
    model.weight.is_managed_by_layer_wise_optimizer = True
    model.norm.is_managed_by_layer_wise_optimizer = False
    muon_params, fallback_params, names = split_muon_and_fallback_params([model])
    assert muon_params == [model.weight]
    assert fallback_params == [model.norm]
    assert set(names.values()) == {"chunk0.weight", "chunk0.norm"}


def test_chained_facade_steps_muon_and_fallback_together():
    weight = torch.nn.Parameter(torch.randn(8, 8))
    norm = torch.nn.Parameter(torch.randn(8))
    weight.grad = torch.randn(8, 8)
    norm.grad = torch.randn(8)

    muon = FP32Muon([{"params": [weight], "weight_decay": 0.01}], lr=0.1, momentum=0.9,
                    weight_decay=0.01)
    from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW

    adam = FP32AdamW([{"params": [norm], "weight_decay": 0.0}], lr=0.1, weight_decay=0.0,
                     betas=(0.9, 0.999), eps=1e-8)
    chained = build_muon_chained_optimizer(muon, adam)

    before_w, before_n = weight.detach().clone(), norm.detach().clone()
    chained.step()
    assert not torch.equal(before_w, weight.detach())
    assert not torch.equal(before_n, norm.detach())
    # Facade exposes both children's param groups.
    assert len(chained.param_groups) == 2


def test_muon_state_dict_roundtrip_resumes_identically():
    torch.manual_seed(7)
    initial_weight = torch.randn(8, 12)
    fixed_grads = [torch.randn(8, 12) for _ in range(3)]

    def _build():
        w = torch.nn.Parameter(initial_weight.clone())
        opt = FP32Muon([{"params": [w], "weight_decay": 0.03}], lr=0.1, momentum=0.9,
                       weight_decay=0.03)
        return w, opt

    # Reference run: 3 uninterrupted steps.
    w_ref, opt_ref = _build()
    for g in fixed_grads:
        w_ref.grad = g.clone()
        opt_ref.step()

    # Interrupted run: 1 step, checkpoint, reload into a fresh optimizer, 2 more steps.
    w_a, opt_a = _build()
    w_a.grad = fixed_grads[0].clone()
    opt_a.step()
    ckpt = opt_a.state_dict()

    w_b = torch.nn.Parameter(w_a.detach().clone())
    opt_b = FP32Muon([{"params": [w_b], "weight_decay": 0.03}], lr=0.1, momentum=0.9,
                     weight_decay=0.03)
    opt_b.load_state_dict(ckpt)
    for g in fixed_grads[1:]:
        w_b.grad = g.clone()
        opt_b.step()

    assert torch.allclose(w_b.detach(), w_ref.detach(), atol=1e-6)


def test_muon_bf16_model_dtype_keeps_fp32_master_and_rounds_param():
    # Promoted params run in fp32 but carry the original bf16 model dtype; the
    # master stays fp32 while the param mirrors a bf16-rounded copy each step.
    torch.manual_seed(3)
    param = torch.nn.Parameter(torch.randn(8, 12, dtype=torch.float32))
    param._fsdp2_model_param_dtype = torch.bfloat16
    param.grad = torch.randn(8, 12, dtype=torch.float32)

    opt = FP32Muon([{"params": [param], "weight_decay": 0.01}], lr=0.1, momentum=0.9,
                   weight_decay=0.01)
    master = opt.state[param]["master_param"]
    assert master is not param  # promoted -> master is a separate fp32 clone
    assert master.dtype is torch.float32

    opt.step()
    # Param mirrors the bf16-rounded master (round-trip through bf16 is idempotent).
    rounded = master.to(torch.bfloat16).to(torch.float32)
    assert torch.equal(param.detach(), rounded)
    assert master.dtype is torch.float32


def _build_muon_adam_facade(weight_init, norm_init):
    from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW
    from megatron.lite.primitive.optimizers.fsdp2.optimizer import FSDP2Optimizer

    weight = torch.nn.Parameter(weight_init.clone())
    norm = torch.nn.Parameter(norm_init.clone())
    muon = FP32Muon([{"params": [weight], "weight_decay": 0.02}], lr=0.1, momentum=0.9,
                    weight_decay=0.02)
    adam = FP32AdamW([{"params": [norm], "weight_decay": 0.0}], lr=0.1, weight_decay=0.0,
                     betas=(0.9, 0.999), eps=1e-8)
    chained = build_muon_chained_optimizer(muon, adam)
    facade = FSDP2Optimizer(chained, [weight, norm], ps=None, clip_grad=1.0)
    return weight, norm, facade


def test_facade_checkpoint_and_offload_roundtrip_with_mixed_state():
    # Mixed state: one Muon matrix + one Adam fallback vector, driven through the
    # real FSDP2Optimizer facade (step / grad-norm / state / offload).
    torch.manual_seed(11)
    weight_init = torch.randn(8, 12)
    norm_init = torch.randn(12)
    weight_grads = [torch.randn(8, 12) for _ in range(3)]
    norm_grads = [torch.randn(12) for _ in range(3)]

    # Reference: 3 uninterrupted facade steps.
    w_ref, n_ref, opt_ref = _build_muon_adam_facade(weight_init, norm_init)
    for wg, ng in zip(weight_grads, norm_grads, strict=True):
        w_ref.grad, n_ref.grad = wg.clone(), ng.clone()
        opt_ref.step()

    # Interrupted: 1 step, offload+reload state (runtime offload; CPU no-op but
    # exercises the shared movement path over the Muon child), checkpoint, resume.
    # The real GPU->CPU->GPU DTensor offload lifecycle of the Muon master +
    # momentum state is verified on CUDA in ``test_muon_fsdp2_offload_gpu.py``.
    w_a, n_a, opt_a = _build_muon_adam_facade(weight_init, norm_init)
    w_a.grad, n_a.grad = weight_grads[0].clone(), norm_grads[0].clone()
    opt_a.step()
    opt_a.offload_state_to_cpu()
    opt_a.load_state_to_device()
    ckpt = opt_a.state_dict()

    w_b, n_b, opt_b = _build_muon_adam_facade(
        w_a.detach().clone(), n_a.detach().clone()
    )
    opt_b.load_state_dict(ckpt)
    for wg, ng in zip(weight_grads[1:], norm_grads[1:], strict=True):
        w_b.grad, n_b.grad = wg.clone(), ng.clone()
        opt_b.step()

    assert torch.allclose(w_b.detach(), w_ref.detach(), atol=1e-6)
    assert torch.allclose(n_b.detach(), n_ref.detach(), atol=1e-6)
