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
from megatron.lite.primitive.optimizers.megatron_wrap import build_dist_opt_stack
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu, pytest.mark.distributed]

_BANK_NAME = MultiLoraTrainingState.parameter_name(
    "layers.0.moe.experts._fc1_weight_0", "a"
)


class SharedFCBank(nn.Module):
    """Adapter-only state: no native Qwen expert key may enter this proof."""

    def __init__(self):
        super().__init__()
        self.register_parameter(
            _BANK_NAME,
            nn.Parameter(torch.arange(16, device="cuda", dtype=torch.bfloat16).reshape(2, 2, 4)),
        )
        parameter = getattr(self, _BANK_NAME)
        parameter.allreduce = False
        parameter.tensor_model_parallel = False

    def forward(self):
        return getattr(self, _BANK_NAME).float().square().mean()


def _phase() -> str:
    phase = os.environ.get("MLITE_SHARED_FC_DCP_PHASE")
    if phase not in {"save", "load"}:
        pytest.skip("set MLITE_SHARED_FC_DCP_PHASE to save or load")
    return phase


def _artifact_dir() -> Path:
    raw = os.environ.get("MLITE_SHARED_FC_DCP_ARTIFACT_DIR")
    if raw is None:
        pytest.skip("set MLITE_SHARED_FC_DCP_ARTIFACT_DIR")
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
    return chunks, optimizer, ps


def _inner_optimizers(optimizer):
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        for item in chained:
            yield from _inner_optimizers(item)
        return
    yield optimizer


def _local_optimizer_slices(chunks, optimizer):
    model = getattr(chunks[0], "module", chunks[0])
    parameter = dict(model.named_parameters())[_BANK_NAME]
    result = []
    for wrapped in _inner_optimizers(optimizer):
        try:
            ranges = wrapped._get_model_param_range_map(parameter)
        except KeyError:
            continue
        world_range = ranges["gbuf_world"]
        master = parameter.main_param
        if master is None:
            continue
        state = wrapped.optimizer.state[master]
        result.append(
            {
                "range": (world_range.start, world_range.end),
                "fp32_param": master.detach().cpu().clone(),
                "exp_avg": state["exp_avg"].detach().cpu().clone(),
                "exp_avg_sq": state["exp_avg_sq"].detach().cpu().clone(),
                "step": state["step"],
            }
        )
    return result


def _global_optimizer_oracle(chunks, optimizer):
    local = _local_optimizer_slices(chunks, optimizer)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    all_slices = [item for rank_slices in gathered for item in rank_slices]
    assert all_slices, "at least one rank must own a dist-opt shard"
    result = {"model": dict(getattr(chunks[0], "module", chunks[0]).named_parameters())[_BANK_NAME].detach().cpu().float().clone()}
    for key in ("fp32_param", "exp_avg", "exp_avg_sq"):
        size = max(item["range"][1] for item in all_slices)
        full = torch.empty(size, dtype=all_slices[0][key].dtype)
        covered = torch.zeros(size, dtype=torch.bool)
        for item in all_slices:
            start, end = item["range"]
            full[start:end].copy_(item[key].reshape(-1))
            covered[start:end] = True
        assert covered.all(), f"incomplete global {key} oracle"
        result[key] = full
    steps = {item["step"] for item in all_slices}
    assert len(steps) == 1 and next(iter(steps)) > 0
    result["step"] = next(iter(steps))
    assert result["exp_avg"].abs().max() > 0 and result["exp_avg_sq"].abs().max() > 0
    return result


def _assert_loaded_slices(chunks, optimizer, oracle):
    model = dict(getattr(chunks[0], "module", chunks[0]).named_parameters())[_BANK_NAME]
    torch.testing.assert_close(model.cpu().float(), oracle["model"], rtol=0, atol=0)
    for item in _local_optimizer_slices(chunks, optimizer):
        start, end = item["range"]
        assert item["step"] == oracle["step"]
        for key in ("fp32_param", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(item[key].reshape(-1), oracle[key][start:end], rtol=0, atol=0)


def _assert_dense_replicated_state(chunk):
    state = chunk.sharded_state_dict()
    tensor = state[_BANK_NAME]
    assert tuple(tensor.global_shape) == tuple(tensor.local_shape)
    assert tuple(tensor.rank_offsets) == ()
    replicas = [tuple(tensor.replica_id)]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, replicas[0])
    assert all(isinstance(replica, tuple) and len(replica) == 3 for replica in gathered)
    assert sum(replica == (0, 0, 0) for replica in gathered) == 1


def test_shared_fc_bank_distopt_dcp_tp2_ep1_to_tp1_ep2():
    if not torch.cuda.is_available() or dist.get_world_size() != 2:
        pytest.skip("run via two-rank CUDA torchrun")
    phase, artifact = _phase(), _artifact_dir()
    if phase == "save":
        parallel = ParallelConfig(tp=2, ep=1, etp=1, pp=1, cp=1)
    else:
        parallel = ParallelConfig(tp=1, ep=2, etp=1, pp=1, cp=1)
    chunks, optimizer, _ps = _build(parallel)
    assert set(dict(getattr(chunks[0], "module", chunks[0]).named_parameters())) == {_BANK_NAME}
    _assert_dense_replicated_state(chunks[0])
    checkpoint_dir = artifact / "checkpoint"
    if phase == "save":
        loss = getattr(chunks[0], "module", chunks[0])()
        loss.backward()
        optimizer.finish_grad_sync()
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
    for parameter in chunks[0].parameters():
        parameter.data.fill_(123)
    for wrapped in _inner_optimizers(optimizer):
        for group in getattr(wrapped, "shard_fp32_from_float16_groups", ()):
            for master in group:
                master.data.fill_(321)
                state = wrapped.optimizer.state[master]
                state["exp_avg"].fill_(322)
                state["exp_avg_sq"].fill_(323)
                state["step"] = type(state["step"])(99)
    assert dcp.load_training_checkpoint(
        chunks[0], optimizer, str(checkpoint_dir), use_dcp=True, load_optimizer=True
    ) == 1
    _assert_loaded_slices(chunks, optimizer, oracle)
