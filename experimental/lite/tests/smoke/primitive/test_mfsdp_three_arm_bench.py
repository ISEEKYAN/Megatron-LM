# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import json
import os
import statistics
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
import torch.distributed as dist

from megatron.lite.model.qwen3_moe.common import is_expert_param
from megatron.lite.primitive.optimizers.fsdp2.adamw import to_local_tensor
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.handle import ModelHandle


pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]

_MCORE_COMMIT = "00309a0199dc590060aa0995b6f4a371d8db9761"
_MLITE_BASE_COMMIT = "62295f9b306d70a8180e907b7c51b3ef293ea007"
_ARMS = ("mcore_mfsdp", "mlite_mfsdp", "mlite_fsdp2")
_TOPOLOGY = (2, 2, 1, 2, 2)
_WARMUP_STEPS = 5
_MEASURE_STEPS = 20
_PRECISION_STEPS = 3
_OPTIMIZER = "torch.optim.AdamW"
_COMPUTE_DTYPE = "bfloat16"
_MAIN_PARAM_DTYPE = "bfloat16"
_TOKENS_PER_STEP = 2 * 64
_LOSS_REL_TOL = 1.0e-2
_MLITE_ABLATIONS = {
    "bucket": (False, True),
    "ag_overlap": (False, True),
    "rs_overlap": (False, True),
    "prefetch": (False, True),
    "double_buffer": (False, True),
    "ub_zero_copy": (False, True),
    "nccl_registered_buffer": (False, True),
    "gdr": (False, True),
}


@dataclass
class _Arm:
    name: str
    handle: ModelHandle
    feature_probe: Callable[[], dict[str, Any]]


class _MCoreOptimizerAdapter:
    """Expose the MLite runtime tuple contract without changing Torch AdamW."""

    def __init__(self, optimizer: torch.optim.Optimizer, model_chunks: list[torch.nn.Module]):
        self.optimizer = optimizer
        self._model_chunks = model_chunks
        self.grad_sync_enabled = False

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> tuple[bool, float, int]:
        local_sq = torch.zeros((), device="cuda", dtype=torch.float32)
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    grad = to_local_tensor(param.grad)
                    local_sq.add_(grad.detach().float().square().sum())
        self.optimizer.step()
        return True, float(local_sq.sqrt()), 0


@pytest.fixture(scope="module", autouse=True)
def _cuda_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        if world_size > 1:
            pytest.fail("Three-arm M-FSDP benchmark launched without CUDA.")
        pytest.skip("CUDA is required for the three-arm M-FSDP benchmark.")
    if world_size != 8:
        pytest.skip("Three-arm M-FSDP benchmark requires exactly 8 ranks.")
    if int(os.environ.get("SLURM_NNODES", "0")) != 1:
        pytest.fail("Three-arm M-FSDP benchmark requires one Slurm node.")

    _assert_source_contract()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created_pg = True
    yield
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


def _assert_source_contract() -> None:
    marker = os.environ.get("MCORE_COMMIT_FILE")
    if not marker:
        pytest.fail("MCORE_COMMIT_FILE must identify the staged NVIDIA MCore source.")
    observed = Path(marker).read_text().strip()
    if observed != _MCORE_COMMIT:
        pytest.fail(f"MCore source mismatch: expected {_MCORE_COMMIT}, got {observed}.")
    observed_mlite = os.environ.get("MLITE_COMMIT")
    if observed_mlite != _MLITE_BASE_COMMIT:
        pytest.fail(
            f"MLite source mismatch: expected {_MLITE_BASE_COMMIT}, got {observed_mlite}."
        )


def _parallel_config() -> ParallelConfig:
    tp, ep, etp, pp, cp = _TOPOLOGY
    return ParallelConfig(tp=tp, ep=ep, etp=etp, pp=pp, vpp=1, cp=cp)


def _model_config():
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig

    return Qwen3MoEConfig(
        num_hidden_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=64,
        max_position_embeddings=4096,
        layer_types=["full_attention", "full_attention"],
    )


def _optimizer_config(overrides: dict[str, Any] | None = None) -> OptimizerConfig:
    cfg = OptimizerConfig(
        optimizer="adam",
        lr=1.0e-3,
        min_lr=0.0,
        weight_decay=0.0,
        clip_grad=1000.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
    )
    cfg.override_optimizer_config = {
        "mfsdp_sharding_strategy": "optim_grads_params",
        "use_fused_optimizer": False,
        "megatron_fsdp_main_params_dtype": torch.bfloat16,
        "megatron_fsdp_main_grads_dtype": torch.bfloat16,
        "megatron_fsdp_grad_comm_dtype": torch.bfloat16,
        "fsdp2_use_fp32_master": False,
        "adamw_foreach": False,
        "fsdp2_use_te_fused_adam": False,
        **(overrides or {}),
    }
    return cfg


def _new_bundle(backend: str | None, *, seed: int, overrides=None):
    from megatron.lite.model.qwen3_moe.lite import protocol

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    impl_cfg = protocol.ImplConfig(
        parallel=_parallel_config(),
        optimizer=backend,
        optimizer_config=_optimizer_config(overrides),
        use_deepep=False,
        use_thd=True,
        deterministic=True,
    )
    return protocol.build_model(_model_config(), impl_cfg=impl_cfg), impl_cfg


def _model_handle(bundle, optimizer, finalize_grads) -> ModelHandle:
    extras = dict(bundle.extras)
    extras.update(
        {
            "model_chunks": bundle.chunks,
            "model_cfg": bundle.extras["model_cfg"],
            "forward_step": bundle.forward_step,
            "finalize_grads": finalize_grads,
        }
    )
    return ModelHandle(
        model=bundle.chunks,
        optimizer=optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=_parallel_config()),
        _extras=extras,
    )


def _optimizer_chain(optimizer: Any) -> tuple[str, ...]:
    result = []
    seen = set()
    current = optimizer
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        cls = type(current)
        result.append(f"{cls.__module__}.{cls.__qualname__}")
        current = getattr(current, "_inner_optimizer", None) or getattr(
            current, "optimizer", None
        )
    return tuple(result)


def _assert_torch_adamw(arm: _Arm) -> None:
    chain = _optimizer_chain(arm.handle._optimizer)
    assert chain[-1] == "torch.optim.adamw.AdamW", (arm.name, chain)
    for group in arm.handle._optimizer.param_groups:
        assert tuple(group["betas"]) == (0.9, 0.999)
        assert float(group["eps"]) == 1.0e-8
        assert float(group["weight_decay"]) == 0.0
        for param in group["params"]:
            assert param.dtype is torch.bfloat16
    if dist.get_rank() == 0:
        print(
            "[MFSDP_OPTIMIZER] "
            f"arm={arm.name} optimizer_chain={' > '.join(chain)} "
            "optimizer=torch.optim.AdamW foreach=false master_param_dtype=bfloat16",
            flush=True,
        )


def _build_mlite_arm(
    backend: str, *, seed: int, overrides=None, label: str | None = None
) -> _Arm:
    bundle, _impl_cfg = _new_bundle(backend, seed=seed, overrides=overrides)
    hook = bundle.extras.get("post_model_load_hook")
    assert callable(hook)
    updates = hook()
    optimizer = updates["optimizer"]
    finalize = updates.get("finalize_grads")
    name = label or ("mlite_mfsdp" if backend == "mfsdp" else "mlite_fsdp2")
    handle = _model_handle(bundle, optimizer, finalize)
    probe_counts = _install_mlite_probe_counts(handle) if backend == "mfsdp" else {}
    return _Arm(name, handle, lambda: _mlite_feature_probe(handle, probe_counts))


def _build_mcore_arm(*, seed: int) -> _Arm:
    from torch.distributed.device_mesh import DeviceMesh

    from megatron.core.distributed.fsdp.src.megatron_fsdp import (
        MixedPrecisionPolicy,
        fully_shard,
    )
    from megatron.lite.model.qwen3_moe.lite.model import TransformerLayer
    from megatron.lite.primitive.optimizers.mfsdp.config import (
        annotate_parallel_parameters,
    )

    bundle, _impl_cfg = _new_bundle(None, seed=seed)
    ps = bundle.parallel_state
    dense_mesh = DeviceMesh.from_group(
        ps.dp_cp_group, "cuda", mesh_dim_names=("fsdp",)
    )
    expert_mesh = DeviceMesh.from_group(
        ps.ep_dp_group, "cuda", mesh_dim_names=("fsdp",)
    )
    wrapped_chunks = []
    torch_optimizers = []
    for chunk in bundle.chunks:
        annotate_parallel_parameters(
            chunk,
            is_expert_param,
            tp_size=ps.tp_size,
            etp_size=ps.etp_size,
        )
        optimizer = torch.optim.AdamW(
            chunk.parameters(),
            lr=1.0e-3,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
            foreach=False,
        )
        wrapped, optimizer = fully_shard(
            chunk,
            optimizer,
            device_mesh=dense_mesh,
            dp_shard_dim="fsdp",
            tp_dim=None,
            expt_device_mesh=expert_mesh,
            fsdp_group_ag=ps.dp_cp_group,
            expt_fsdp_group_ag=ps.ep_dp_group,
            fsdp_unit_modules=(TransformerLayer,),
            zero_dp_strategy="optim_grads_params",
            mixed_precision_policy=MixedPrecisionPolicy(
                main_params_dtype=torch.bfloat16,
                main_grads_dtype=torch.bfloat16,
                grad_comm_dtype=torch.bfloat16,
            ),
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            sync_model_each_microbatch=True,
            preproc_state_dict_for_dcp_ckpt=False,
            average_in_collective=True,
        )
        wrapped_chunks.append(wrapped)
        torch_optimizers.append(optimizer)
    assert len(torch_optimizers) == 1
    bundle.chunks[:] = wrapped_chunks
    adapter = _MCoreOptimizerAdapter(torch_optimizers[0], wrapped_chunks)
    handle = _model_handle(bundle, adapter, None)
    return _Arm("mcore_mfsdp", handle, lambda: _mcore_feature_probe(handle))


def _fixed_batches(*, seed: int) -> list[PackedBatch]:
    result = []
    for microbatch in range(2):
        generator = torch.Generator(device="cuda").manual_seed(seed + microbatch)
        result.append(
            PackedBatch(
                input_ids=torch.randint(
                    0, 64, (64,), generator=generator, device="cuda"
                ),
                labels=torch.randint(
                    0, 64, (64,), generator=generator, device="cuda"
                ),
                cu_seqlens=torch.tensor([0, 64], device="cuda", dtype=torch.int32),
                max_seqlen=64,
            )
        )
    return result


def _run_step(arm: _Arm, *, seed: int) -> tuple[float, float, float, float]:
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.zero_grad(arm.handle)
    original_forward_step = arm.handle._extras["forward_step"]
    forward_events = []

    def timed_forward_step(model, batch):
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        output = original_forward_step(model, batch)
        finished.record()
        forward_events.append((started, finished))
        return output

    arm.handle._extras["forward_step"] = timed_forward_step
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    after_fb = torch.cuda.Event(enable_timing=True)
    after_optimizer = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        result = runtime.forward_backward(
            arm.handle, iter(_fixed_batches(seed=seed)), None, num_microbatches=2
        )
    finally:
        arm.handle._extras["forward_step"] = original_forward_step
    after_fb.record()
    success, _grad_norm, _num_zeros = runtime.optimizer_step(arm.handle)
    after_optimizer.record()
    torch.cuda.synchronize()
    assert success
    loss = result.model_output.loss
    assert loss is not None and torch.isfinite(loss)
    fb_ms = start.elapsed_time(after_fb)
    forward_ms = sum(started.elapsed_time(finished) for started, finished in forward_events)
    optimizer_ms = after_fb.elapsed_time(after_optimizer)
    return float(loss.detach().float().cpu()), forward_ms, fb_ms, optimizer_ms


def _profile_communication_cuda_ms(arm: _Arm, *, seed: int) -> float:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _run_step(arm, seed=seed)
    total_us = 0.0
    for event in prof.key_averages():
        name = event.key.lower()
        if not any(token in name for token in ("nccl", "all_gather", "reduce_scatter", "alltoall")):
            continue
        total_us += float(
            getattr(
                event,
                "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
        )
    return _max_across_ranks(total_us / 1000.0)


def _relative_diff(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-12)


def _max_across_ranks(value: float) -> float:
    tensor = torch.tensor(value, device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def _benchmark(arms: list[_Arm], *, seed: int) -> dict[str, Any]:
    assert len({arm.name for arm in arms}) == len(arms)
    precision_losses = {arm.name: [] for arm in arms}
    for step in range(_PRECISION_STEPS):
        ordered = arms[step % len(arms) :] + arms[: step % len(arms)]
        for arm in ordered:
            loss, _forward_ms, _fb_ms, _optimizer_ms = _run_step(
                arm, seed=seed + step
            )
            precision_losses[arm.name].append(loss)

    reference = precision_losses[arms[0].name]
    precision_max_rel = {}
    for arm in arms:
        value = max(
            _relative_diff(left, right)
            for left, right in zip(reference, precision_losses[arm.name], strict=True)
        )
        value = _max_across_ranks(value)
        assert value <= _LOSS_REL_TOL, (arm.name, value, precision_losses)
        precision_max_rel[arm.name] = value

    for step in range(_WARMUP_STEPS):
        ordered = arms[step % len(arms) :] + arms[: step % len(arms)]
        for arm in ordered:
            _run_step(arm, seed=seed + 100 + step)

    samples = {
        arm.name: {
            "step_ms": [],
            "forward_ms": [],
            "backward_schedule_ms": [],
            "forward_backward_ms": [],
            "optimizer_ms": [],
        }
        for arm in arms
    }
    peak_memory = {arm.name: 0 for arm in arms}
    for step in range(_MEASURE_STEPS):
        ordered = arms[step % len(arms) :] + arms[: step % len(arms)]
        for arm in ordered:
            torch.cuda.reset_peak_memory_stats()
            _loss, forward_ms, fb_ms, optimizer_ms = _run_step(
                arm, seed=seed + 1000 + step
            )
            samples[arm.name]["forward_ms"].append(forward_ms)
            samples[arm.name]["backward_schedule_ms"].append(
                max(fb_ms - forward_ms, 0.0)
            )
            samples[arm.name]["forward_backward_ms"].append(fb_ms)
            samples[arm.name]["optimizer_ms"].append(optimizer_ms)
            samples[arm.name]["step_ms"].append(fb_ms + optimizer_ms)
            peak_memory[arm.name] = max(
                peak_memory[arm.name], torch.cuda.max_memory_allocated()
            )

    metrics = {}
    for arm in arms:
        arm_samples = samples[arm.name]
        step_ms = arm_samples["step_ms"]
        p50 = statistics.median(step_ms)
        metrics[arm.name] = {
            "step_ms_p50": p50,
            "step_ms_p95": _percentile(step_ms, 0.95),
            "step_ms_min": min(step_ms),
            "step_ms_max": max(step_ms),
            "tokens_per_s": _TOKENS_PER_STEP * dist.get_world_size() / (p50 / 1000.0),
            "forward_backward_ms_p50": statistics.median(
                arm_samples["forward_backward_ms"]
            ),
            "forward_ms_p50": statistics.median(arm_samples["forward_ms"]),
            "backward_schedule_ms_p50": statistics.median(
                arm_samples["backward_schedule_ms"]
            ),
            "optimizer_ms_p50": statistics.median(arm_samples["optimizer_ms"]),
            "communication_cuda_kernel_ms_overlapped": _profile_communication_cuda_ms(
                arm, seed=seed + 2000
            ),
            "communication_time_note": (
                "sum of profiled collective CUDA kernels; kernels may overlap compute"
            ),
            "peak_memory_gib": peak_memory[arm.name] / 1024**3,
            "precision_max_loss_rel": precision_max_rel[arm.name],
            "feature_probe": arm.feature_probe(),
        }
    return metrics


def _install_mlite_probe_counts(handle: ModelHandle) -> dict[str, int]:
    counts = {"ag_launches": 0, "rs_launches": 0, "start_prefetch_calls": 0}
    for chunk in handle._extras["model_chunks"]:
        all_gather = chunk.all_gather_pipeline
        original_gather = all_gather.async_bucket_gather

        def counted_gather(*args, _original=original_gather, **kwargs):
            counts["ag_launches"] += 1
            return _original(*args, **kwargs)

        all_gather.async_bucket_gather = counted_gather
        grad_reduce = chunk.grad_reduce_pipeline
        original_reduce = grad_reduce.reduce_gradients

        def counted_reduce(*args, _original=original_reduce, **kwargs):
            counts["rs_launches"] += 1
            return _original(*args, **kwargs)

        grad_reduce.reduce_gradients = counted_reduce
        for bucket in grad_reduce.buckets:
            if bucket.grad_ready_callback is not None:
                bucket.grad_ready_callback = counted_reduce
        original_start = chunk.start_param_sync

        def counted_start(self, *args, _original=original_start, **kwargs):
            counts["start_prefetch_calls"] += 1
            return _original(*args, **kwargs)

        chunk.start_param_sync = types.MethodType(counted_start, chunk)
    return counts


def _mlite_feature_probe(
    handle: ModelHandle, probe_counts: dict[str, int]
) -> dict[str, Any]:
    chunks = handle._extras["model_chunks"]
    buckets = [bucket for chunk in chunks for bucket in chunk.param_sync.buckets]
    allocator = chunks[0].param_and_grad_buffer.allocator
    user_buffer = allocator.user_buffer
    return {
        "bucket_count": len(buckets),
        "ag_overlap": chunks[0].all_gather_pipeline.overlap,
        "rs_overlap": any(bucket.grad_ready_callback is not None for bucket in buckets),
        "prefetch": chunks[0].mfsdp_config.all_gather_in_start_param_sync,
        "allocator": type(allocator).__name__,
        "double_buffer_slots": len(getattr(allocator, "_slots", {})),
        "nccl_ub_requested": chunks[0].mfsdp_config.nccl_ub,
        "nccl_ub_active": bool(user_buffer is not None and user_buffer.active),
        "symmetric_registration": bool(
            user_buffer is not None and user_buffer.active and user_buffer.symmetric
        ),
        "gdr_verified": False,
        **probe_counts,
    }


def _mcore_feature_probe(handle: ModelHandle) -> dict[str, Any]:
    chunk = handle._extras["model_chunks"][0]
    return {
        "implementation": f"{type(chunk).__module__}.{type(chunk).__qualname__}",
        "bucket_count": len(chunk.param_and_grad_buffer.parameter_groups),
        "ag_overlap": bool(chunk.ddp_config.overlap_param_gather),
        "rs_overlap": bool(chunk.ddp_config.overlap_grad_reduce),
        "double_buffer": bool(chunk.ddp_config.fsdp_double_buffer),
        "nccl_ub": bool(chunk.ddp_config.nccl_ub),
    }


def _ablation_overrides(feature: str, enabled: bool) -> dict[str, Any] | None:
    if feature == "bucket":
        return {"bucket_size": 4096 if enabled else None}
    if feature == "ag_overlap":
        return {"overlap_param_gather": enabled}
    if feature == "rs_overlap":
        return {"overlap_grad_reduce": enabled}
    if feature == "prefetch":
        return {"fsdp_all_gather_in_start_param_sync": enabled}
    if feature == "double_buffer":
        return {"fsdp_double_buffer": enabled}
    if feature in {"ub_zero_copy", "nccl_registered_buffer"}:
        return {"nccl_ub": enabled, "fsdp_double_buffer": enabled}
    if feature == "gdr":
        return None
    raise AssertionError(feature)


def test_mfsdp_three_arm_torch_adamw_benchmark():
    arms = [
        _build_mcore_arm(seed=7311),
        _build_mlite_arm("mfsdp", seed=7311),
        _build_mlite_arm("fsdp2", seed=7311),
    ]
    assert tuple(arm.name for arm in arms) == _ARMS
    for arm in arms:
        _assert_torch_adamw(arm)
        ps = arm.handle._parallel_state
        assert (ps.tp_size, ps.ep_size, ps.etp_size, ps.pp_size, ps.cp_size) == _TOPOLOGY
    metrics = _benchmark(arms, seed=9100)
    if dist.get_rank() == 0:
        print(
            "[MFSDP_THREE_ARM] "
            + json.dumps(
                {
                    "mcore_commit": _MCORE_COMMIT,
                    "mlite_commit": _MLITE_BASE_COMMIT,
                    "optimizer": _OPTIMIZER,
                    "topology": "tp2_ep2_etp1_pp2_cp2",
                    "warmup_steps": _WARMUP_STEPS,
                    "measure_steps": _MEASURE_STEPS,
                    "metrics": metrics,
                    "apex_fused_adam": "N/A: no common Apex optimizer path across all three arms",
                },
                sort_keys=True,
            ),
            flush=True,
        )


def test_mlite_mfsdp_feature_ablation():
    results = {}
    cache = {}
    for feature, states in _MLITE_ABLATIONS.items():
        results[feature] = {}
        off_overrides = _ablation_overrides(feature, False)
        on_overrides = _ablation_overrides(feature, True)
        if off_overrides is None or on_overrides is None:
            for enabled in states:
                results[feature][str(enabled).lower()] = {
                    "status": "N/A",
                    "reason": "MLite M-FSDP exposes no verifiable GDR control or hit marker",
                }
            continue
        key = (
            feature,
            tuple(sorted(off_overrides.items())),
            tuple(sorted(on_overrides.items())),
        )
        if key not in cache:
            arms = [
                _build_mlite_arm(
                    "mfsdp",
                    seed=8421,
                    overrides=off_overrides,
                    label=f"{feature}_off",
                ),
                _build_mlite_arm(
                    "mfsdp",
                    seed=8421,
                    overrides=on_overrides,
                    label=f"{feature}_on",
                ),
            ]
            for arm in arms:
                _assert_torch_adamw(arm)
            cache[key] = _benchmark(arms, seed=10100)
        pair_metrics = cache[key]
        for enabled in states:
            metrics = pair_metrics[f"{feature}_{'on' if enabled else 'off'}"]
            probe = metrics["feature_probe"]
            status = "measured"
            if enabled and feature == "ub_zero_copy":
                status = "unimplemented_copy_path"
            elif enabled and feature == "nccl_registered_buffer":
                if not probe["nccl_ub_active"]:
                    status = "unimplemented_fallback"
            elif enabled and feature == "prefetch":
                if probe["start_prefetch_calls"] == 0:
                    status = "unimplemented_no_production_hit"
            elif enabled and feature == "double_buffer":
                if probe["double_buffer_slots"] == 0:
                    status = "unimplemented_no_production_hit"
            results[feature][str(enabled).lower()] = {
                "status": status,
                "metrics": metrics,
            }

    for feature in ("bucket", "ag_overlap", "rs_overlap", "double_buffer"):
        assert results[feature]["false"]["status"] == "measured"
        assert results[feature]["true"]["status"] == "measured"
    off_bucket_count = results["bucket"]["false"]["metrics"]["feature_probe"][
        "bucket_count"
    ]
    on_bucket_count = results["bucket"]["true"]["metrics"]["feature_probe"][
        "bucket_count"
    ]
    assert on_bucket_count > off_bucket_count
    if dist.get_rank() == 0:
        print(
            "[MFSDP_ABLATION] "
            + json.dumps(
                {
                    "mlite_commit": _MLITE_BASE_COMMIT,
                    "optimizer": _OPTIMIZER,
                    "topology": "tp2_ep2_etp1_pp2_cp2",
                    "results": results,
                },
                sort_keys=True,
            ),
            flush=True,
        )
