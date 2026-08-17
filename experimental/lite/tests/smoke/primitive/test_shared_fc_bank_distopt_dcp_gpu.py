"""Two-phase dist-opt DCP reshard proof for an isolated shared FC LoRA bank."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.model.qwen3_moe.common import is_expert_param
from megatron.lite.model.qwen3_moe.lite.checkpoint import EXPERT_CLASSIFIER, PLACEMENT_FN
from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict, dcp
from megatron.lite.primitive.modules.multi_lora_bank import MultiLoraTrainingState
from megatron.lite.primitive.optimizers.megatron_wrap import (
    build_dist_opt_stack,
    finalize_dist_opt_grads,
)
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpus(2), pytest.mark.distributed]

_BANK_NAME = MultiLoraTrainingState.parameter_name(
    "layers.0.moe.experts._fc1_weight_0", "a"
)


def _skip_or_fail(message: str) -> None:
    if os.environ.get("MLITE_SHARED_FC_DCP_RUNNER") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module", autouse=True)
def _two_rank_cuda_process_group():
    """Make the standalone torchrun carrier establish its own NCCL group."""
    from megatron.core import parallel_state as mpu

    mpu_was_initialized = mpu.is_initialized()
    if not torch.cuda.is_available():
        _skip_or_fail("CUDA is required for the shared-FC DCP smoke.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        _skip_or_fail("run this smoke through two-rank torchrun")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created = True
    yield
    try:
        if not mpu_was_initialized and mpu.is_initialized():
            mpu.destroy_model_parallel()
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()


class SharedFCBank(nn.Module):
    """Adapter-only state: no native Qwen expert key may enter this proof."""

    def __init__(self):
        super().__init__()
        self.register_parameter(
            _BANK_NAME,
            nn.Parameter(
                torch.arange(4096, device="cuda", dtype=torch.bfloat16).reshape(2, 32, 64)
            ),
        )
        parameter = getattr(self, _BANK_NAME)
        parameter.allreduce = False
        parameter.tensor_model_parallel = False

    def forward(self):
        return getattr(self, _BANK_NAME).float().square().mean()


def _phase() -> str:
    phase = os.environ.get("MLITE_SHARED_FC_DCP_PHASE")
    if phase not in {"save", "load"}:
        _skip_or_fail("set MLITE_SHARED_FC_DCP_PHASE to save or load")
    return phase


def _artifact_dir() -> Path:
    raw = os.environ.get("MLITE_SHARED_FC_DCP_ARTIFACT_DIR")
    if raw is None:
        _skip_or_fail("set MLITE_SHARED_FC_DCP_ARTIFACT_DIR")
    return Path(raw)


def _build(parallel: ParallelConfig):
    ps = init_parallel(parallel)
    model = SharedFCBank()
    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_experts=2,
        moe_intermediate_size=16,
        add_bias_linear=False,
    )
    engine_cfg = SimpleNamespace(
        model_name="shared_fc_bank",
        parallel=parallel,
        optimizer=OptimizerConfig(optimizer="adam", lr=1.0e-3, weight_decay=0.0),
        deterministic=True,
    )
    chunks, optimizer = build_dist_opt_stack(
        [model], model_cfg=model_cfg, engine_cfg=engine_cfg, ps=ps, is_expert=is_expert_param
    )
    attach_model_sharded_state_dict(
        chunks, ps, get_placements=PLACEMENT_FN, is_expert=EXPERT_CLASSIFIER
    )
    return chunks, optimizer, ps, _production_finalize_grads(chunks, optimizer)


def _production_finalize_grads(chunks, optimizer):
    """Use the runtime's MCore DDP finalization before the outer optimizer step."""

    def finalize_grads():
        finalize_dist_opt_grads(chunks, optimizer)

    return finalize_grads


def _inner_optimizers(optimizer):
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        for item in chained:
            yield from _inner_optimizers(item)
        return
    yield optimizer


def _model_param_range_or_none(optimizer, parameter):
    """A non-owning dist-opt leaf has no model-param map on this MCore surface."""
    if not hasattr(optimizer, "model_param_gbuf_map"):
        return None
    try:
        return optimizer._get_model_param_range_map(parameter)
    except KeyError:
        return None


def _parameter_group_for_master(optimizer, master):
    for group in optimizer.optimizer.param_groups:
        if any(candidate is master for candidate in group["params"]):
            assert "step" in group
            return group
    raise AssertionError("master parameter has no optimizer param group")


def _local_optimizer_slices(chunks, optimizer):
    model = getattr(chunks[0], "module", chunks[0])
    parameter = dict(model.named_parameters())[_BANK_NAME]
    result = []
    for wrapped in _inner_optimizers(optimizer):
        ranges = _model_param_range_or_none(wrapped, parameter)
        if ranges is None:
            continue
        param_range = ranges["param"]
        master = parameter.main_param
        if master is None:
            continue
        state = wrapped.optimizer.state[master]
        param_group = _parameter_group_for_master(wrapped, master)
        assert master.numel() == param_range.end - param_range.start
        result.append(
            {
                "range": (param_range.start, param_range.end),
                "fp32_param": master.detach().cpu().clone(),
                "exp_avg": state["exp_avg"].detach().cpu().clone(),
                "exp_avg_sq": state["exp_avg_sq"].detach().cpu().clone(),
                "step": _step_value(param_group["step"]),
            }
        )
    return result


def _global_optimizer_oracle(chunks, optimizer):
    parameter = dict(getattr(chunks[0], "module", chunks[0]).named_parameters())[_BANK_NAME]
    local = _local_optimizer_slices(chunks, optimizer)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    assert len(gathered) == 2 and all(len(rank_slices) == 1 for rank_slices in gathered)
    source_ranges = sorted(item["range"] for rank_slices in gathered for item in rank_slices)
    assert source_ranges[0][0] == 0
    assert source_ranges[0][1] == source_ranges[1][0]
    assert source_ranges[1][1] == parameter.numel()
    all_slices = [item for rank_slices in gathered for item in rank_slices]
    model_gathered = [None] * dist.get_world_size()
    source_model = parameter.detach().cpu().clone()
    dist.all_gather_object(model_gathered, source_model)
    assert all(torch.equal(model, source_model) for model in model_gathered)
    result = {"model": source_model.float()}
    for key in ("fp32_param", "exp_avg", "exp_avg_sq"):
        size = parameter.numel()
        full = torch.empty(size, dtype=all_slices[0][key].dtype)
        coverage = torch.zeros(size, dtype=torch.int)
        for item in all_slices:
            start, end = item["range"]
            assert 0 <= start <= end <= size
            assert end - start == item[key].numel()
            full[start:end].copy_(item[key].reshape(-1))
            coverage[start:end] += 1
        assert torch.equal(coverage, torch.ones_like(coverage)), f"invalid global {key} coverage"
        result[key] = full
    steps = [item["step"] for item in all_slices]
    assert len(set(steps)) == 1 and steps[0] > 0
    result["step"] = steps[0]
    assert result["exp_avg"].abs().max() > 0 and result["exp_avg_sq"].abs().max() > 0
    return result


def _assert_loaded_slices(chunks, optimizer, oracle):
    model = dict(getattr(chunks[0], "module", chunks[0]).named_parameters())[_BANK_NAME]
    torch.testing.assert_close(model.cpu().float(), oracle["model"], rtol=0, atol=0)
    local = _local_optimizer_slices(chunks, optimizer)
    assert local, "every load rank must own the complete shared bank"
    expected = (0, model.numel())
    assert [item["range"] for item in local] == [expected]
    for item in local:
        start, end = item["range"]
        assert item["step"] == oracle["step"]
        for key in ("fp32_param", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(item[key].reshape(-1), oracle[key][start:end], rtol=0, atol=0)


def _assert_dense_replicated_state(chunk, phase: str):
    state = chunk.sharded_state_dict()
    tensor = state[_BANK_NAME]
    assert tuple(tensor.global_shape) == tuple(tensor.local_shape)
    assert tuple(tensor.global_offset) == (0,) * len(tensor.local_shape)
    assert tuple(tensor.axis_fragmentations) == (1,) * len(tensor.local_shape)
    assert tensor.flattened_range is None
    replicas = [tuple(tensor.replica_id)]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, replicas[0])
    expected = (
        {(0, 0, 0), (0, 1, 0)} if phase == "save" else {(0, 0, 0), (0, 0, 1)}
    )
    assert set(gathered) == expected
    assert len(gathered) == len(expected)


def _step_value(step) -> int:
    return int(step.detach().cpu().item()) if torch.is_tensor(step) else int(step)


def _poison_optimizer_state(chunks, optimizer, oracle, finalize_grads):
    """Materialize target Adam state before making its values detectably wrong."""
    loss = getattr(chunks[0], "module", chunks[0])()
    loss.backward()
    finalize_grads()
    optimizer.step()
    optimizer.zero_grad()
    for parameter in chunks[0].parameters():
        parameter.data.fill_(123)
    for wrapped in _inner_optimizers(optimizer):
        for group in getattr(wrapped, "shard_fp32_from_float16_groups", ()):
            for master in group:
                master.data.fill_(321)
                state = wrapped.optimizer.state[master]
                state["exp_avg"].fill_(322)
                state["exp_avg_sq"].fill_(323)
                param_group = _parameter_group_for_master(wrapped, master)
                if torch.is_tensor(param_group["step"]):
                    param_group["step"].fill_(99)
                else:
                    param_group["step"] = 99
    local = _local_optimizer_slices(chunks, optimizer)
    for item in local:
        start, end = item["range"]
        assert not torch.equal(item["fp32_param"].reshape(-1), oracle["fp32_param"][start:end])
        assert not torch.equal(item["exp_avg"].reshape(-1), oracle["exp_avg"][start:end])
        assert not torch.equal(item["exp_avg_sq"].reshape(-1), oracle["exp_avg_sq"][start:end])
        assert item["step"] != oracle["step"]
    has_local_state = torch.tensor(int(bool(local)), device="cuda")
    dist.all_reduce(has_local_state, op=dist.ReduceOp.MAX)
    assert has_local_state.item() == 1, "target topology must materialize Adam state"


def test_shared_fc_bank_distopt_dcp_tp2_ep1_to_tp1_ep2():
    if not torch.cuda.is_available() or dist.get_world_size() != 2:
        pytest.skip("run via two-rank CUDA torchrun")
    phase, artifact = _phase(), _artifact_dir()
    if phase == "save":
        parallel = ParallelConfig(tp=2, ep=1, etp=1, pp=1, cp=1)
    else:
        parallel = ParallelConfig(tp=1, ep=2, etp=1, pp=1, cp=1)
    chunks, optimizer, _ps, finalize_grads = _build(parallel)
    assert set(dict(getattr(chunks[0], "module", chunks[0]).named_parameters())) == {_BANK_NAME}
    _assert_dense_replicated_state(chunks[0], phase)
    checkpoint_dir = artifact / "checkpoint"
    if phase == "save":
        loss = getattr(chunks[0], "module", chunks[0])()
        loss.backward()
        finalize_grads()
        optimizer.step()
        optimizer.zero_grad()
        oracle = _global_optimizer_oracle(chunks, optimizer)
        if dist.get_rank() == 0:
            artifact.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        torch.save(oracle, artifact / f"oracle_rank_{dist.get_rank()}.pt")
        dcp.save_training_checkpoint(
            chunks[0], optimizer, 1, str(checkpoint_dir), use_dcp=True, save_optimizer=True
        )
        dist.barrier()
        return
    oracle = torch.load(artifact / f"oracle_rank_{dist.get_rank()}.pt", weights_only=True)
    _poison_optimizer_state(chunks, optimizer, oracle, finalize_grads)
    assert dcp.load_training_checkpoint(
        chunks[0], optimizer, str(checkpoint_dir), use_dcp=True, load_optimizer=True
    ) == 1
    _assert_loaded_slices(chunks, optimizer, oracle)
