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


def _snapshot(chunks, optimizer):
    model = getattr(chunks[0], "module", chunks[0])
    parameter = dict(model.named_parameters())[_BANK_NAME]
    result = {"model": parameter.detach().cpu().float().clone(), "states": []}
    for wrapped in _inner_optimizers(optimizer):
        masters = getattr(wrapped, "shard_fp32_from_float16_groups", ())
        for group in masters:
            for master in group:
                state = wrapped.optimizer.state[master]
                result["states"].append(
                    {
                        "fp32_param": master.detach().cpu().clone(),
                        "exp_avg": state["exp_avg"].detach().cpu().clone(),
                        "exp_avg_sq": state["exp_avg_sq"].detach().cpu().clone(),
                        "step": int(state["step"]),
                    }
                )
    assert result["states"], "dist-opt must expose fp32 master state"
    assert all(item["exp_avg"].abs().max() > 0 for item in result["states"])
    assert all(item["exp_avg_sq"].abs().max() > 0 for item in result["states"])
    assert all(item["step"] > 0 for item in result["states"])
    return result


def _assert_equal(actual, expected):
    assert actual.keys() == expected.keys()
    torch.testing.assert_close(actual["model"], expected["model"], rtol=0, atol=0)
    assert len(actual["states"]) == len(expected["states"])
    for got, want in zip(actual["states"], expected["states"], strict=True):
        assert got["step"] == want["step"]
        for key in ("fp32_param", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(got[key], want[key], rtol=0, atol=0)


def _assert_dense_replicated_state(chunk):
    state = chunk.sharded_state_dict()
    tensor = state[_BANK_NAME]
    assert tuple(tensor.global_shape) == tuple(tensor.local_shape)
    assert tuple(tensor.rank_offsets) == ()
    replicas = [tensor.replica_id == (0, 0, 0)]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, replicas[0])
    assert sum(gathered) == 1


def test_shared_fc_bank_distopt_dcp_tp2_ep1_to_tp1_ep2(tmp_path):
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
        optimizer.step()
        optimizer.zero_grad()
        oracle = _snapshot(chunks, optimizer)
        if dist.get_rank() == 0:
            artifact.mkdir(parents=True)
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
                state["step"] = 99
    assert dcp.load_training_checkpoint(
        chunks[0], optimizer, str(checkpoint_dir), use_dcp=True, load_optimizer=True
    ) == 1
    _assert_equal(_snapshot(chunks, optimizer), oracle)
