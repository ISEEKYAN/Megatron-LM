# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Two-GPU production lifecycle proof for model-owned Qwen3-MoE multi-LoRA."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.ckpt import dcp
from megatron.lite.primitive.train_step import run_microbatch_loop
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch

pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]


def _qwen_symbols():
    pytest.importorskip(
        "transformer_engine.pytorch",
        reason="production multi-LoRA EP2 smoke requires Transformer Engine.",
    )
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite import model, protocol

    return Qwen3MoEConfig, model, protocol


@pytest.fixture(scope="module", autouse=True)
def _ep2_cuda_group():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the production multi-LoRA EP2 smoke.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("run this smoke through torchrun with exactly two ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created = False
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")
        created = True
    yield
    if created:
        dist.destroy_process_group()


def _config():
    Qwen3MoEConfig, _model, _protocol = _qwen_symbols()
    return Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        max_position_embeddings=16,
        layer_types=["full_attention"],
    )


def _build_bundle(*, ep: int, model_seed: int):
    _Qwen3MoEConfig, _model, protocol = _qwen_symbols()
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    return protocol.build_model(
        _config(),
        impl_cfg=protocol.ImplConfig(
            parallel=ParallelConfig(tp=1, ep=ep, etp=1, pp=1, cp=1),
            optimizer="dist_opt",
            optimizer_config=OptimizerConfig(
                optimizer="adam", lr=1.0e-3, weight_decay=0.0, clip_grad=1.0
            ),
            multi_lora={"names": ("alpha", "bravo"), "rank": 2, "alpha": 4},
            use_deepep=False,
            deterministic=True,
        ),
    )


def _main_grad(param: torch.Tensor) -> torch.Tensor:
    grad = getattr(param, "main_grad", None)
    if grad is None:
        grad = param.grad
    assert grad is not None
    return grad


def _phase_artifact_dir() -> Path:
    raw = os.environ.get("MLITE_MULTI_LORA_PHASE_ARTIFACT_DIR")
    if raw is None:
        pytest.skip("run through the two-phase production multi-LoRA sbatch")
    return Path(raw)


def _phase() -> str:
    phase = os.environ.get("MLITE_MULTI_LORA_PHASE")
    if phase not in {"ep1_oracle", "ep2_verify"}:
        pytest.skip("set MLITE_MULTI_LORA_PHASE to ep1_oracle or ep2_verify")
    return phase


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    cpu = tensor.detach().cpu().contiguous()
    return {
        "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
    }


def _named_parameters(bundle) -> dict[str, torch.Tensor]:
    parameters: dict[str, torch.Tensor] = {}
    for chunk in bundle.chunks:
        module = getattr(chunk, "module", chunk)
        parameters.update(dict(module.named_parameters()))
    return parameters


def _checkpoint_semantics(bundle) -> dict[str, dict[str, object]]:
    """Independent dense/expert/bank semantic value records for DCP resharding."""
    parameters = _named_parameters(bundle)
    dense_candidates = sorted(
        name
        for name in parameters
        if ".experts." not in name and "multi_lora_training_state" not in name
    )
    assert dense_candidates, "tiny Qwen must expose a dense parameter"
    dense_name = dense_candidates[0]
    expert_records: dict[str, dict[str, object]] = {}
    local_experts = _config().num_experts // bundle.parallel_state.ep_size
    for name, parameter in parameters.items():
        match = re.search(r"\.experts\.(fc[12])\.weight(\d+)$", name)
        if match is None:
            continue
        global_expert = bundle.parallel_state.ep_rank * local_experts + int(
            match.group(2)
        )
        expert_records[f"{match.group(1)}.expert{global_expert}"] = _tensor_record(
            parameter
        )
    assert set(expert_records) == {
        f"{fc}.expert{expert}"
        for fc in ("fc1", "fc2")
        for expert in range(
            bundle.parallel_state.ep_rank * local_experts,
            (bundle.parallel_state.ep_rank + 1) * local_experts,
        )
    }
    state = bundle.extras["multi_lora_training_state"]
    bank_records = {
        name: _tensor_record(parameter) for name, parameter in state.named_parameters()
    }
    assert set(bank_records) == _expected_semantic_bank_keys(state)
    return {
        "dense": {dense_name: _tensor_record(parameters[dense_name])},
        "experts": expert_records,
        "banks": bank_records,
    }


def _poison_parameters(bundle) -> None:
    """Make pre-load values observably unlike a checkpoint from another seed."""
    with torch.no_grad():
        for parameter in _named_parameters(bundle).values():
            parameter.add_(0.125)


def _actual_topology(bundle) -> dict[str, int]:
    """Record dense/expert decomposition from live state and process groups."""
    ps = bundle.parallel_state
    dense_dp_group_size = (
        dist.get_world_size(ps.dp_group) if ps.dp_group is not None else 1
    )
    expert_dp_group_size = (
        dist.get_world_size(ps.ep_dp_group) if ps.ep_dp_group is not None else 1
    )
    assert ps.dp_size == dense_dp_group_size
    assert ps.expert_dp_size == expert_dp_group_size
    return {
        "world": dist.get_world_size(),
        "tp": ps.tp_size,
        "ep": ps.ep_size,
        "etp": ps.etp_size,
        "pp": ps.pp_size,
        "cp": ps.cp_size,
        "dp": dense_dp_group_size,
        "dp_rank": ps.dp_rank,
        "expert_dp": expert_dp_group_size,
        "expert_dp_rank": ps.expert_dp_rank,
    }


def _impl_contract(*, ep: int) -> dict[str, object]:
    """All implementation knobs that define this production evidence."""
    return {
        "parallel": {"tp": 1, "ep": ep, "etp": 1, "pp": 1, "cp": 1},
        "optimizer": "dist_opt",
        "optimizer_config": {
            "optimizer": "adam",
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "clip_grad": 1.0,
        },
        "deterministic": True,
        "use_deepep": False,
        "multi_lora": {
            "names": ["alpha", "bravo"],
            "rank": 2,
            "alpha": 4,
            "use_rslora": False,
        },
    }


def _fixed_local_router(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Route every token to expert zero, available in both EP1 and EP2."""
    return (
        torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype),
        torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.long),
    )


def _initialize_nonzero_banks(bundle) -> None:
    """Give A and B reproducible nonzero factors on every rank."""
    state = bundle.extras["multi_lora_training_state"]
    for index, parameter in enumerate(state.parameters()):
        values = torch.arange(
            parameter.numel(), device=parameter.device, dtype=torch.float32
        ).reshape_as(parameter)
        with torch.no_grad():
            parameter.copy_((values + index + 1).to(parameter.dtype) / 32)
        dist.broadcast(parameter, src=0)


def _configure_fixed_router(bundle) -> None:
    for chunk in bundle.chunks:
        qwen = getattr(chunk, "module", chunk)
        qwen.layers[0].moe.router.forward = _fixed_local_router


def _bank_parameters(bundle) -> dict[str, torch.Tensor]:
    state = bundle.extras["multi_lora_training_state"]
    parameters = dict(state.named_parameters())
    assert set(parameters) == _expected_semantic_bank_keys(state)
    return parameters


def _distributed_optimizer_leaves(optimizer) -> tuple[object, ...]:
    """Return dist-opt leaves without assuming a Qwen dense/expert chain shape."""
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is None:
        return (optimizer,)
    children = tuple(chained)
    assert children, "empty ChainedOptimizer cannot own distributed parameters"
    return tuple(
        leaf for child in children for leaf in _distributed_optimizer_leaves(child)
    )


def _group_ranks(group) -> tuple[int, ...]:
    """Make collective ownership comparable rather than relying on group size."""
    return tuple(dist.get_process_group_ranks(group))


def _optimizer_param_range(
    bundle, parameter: torch.Tensor
) -> tuple[int, object, object, object, object] | None:
    """Resolve this rank's optional dist-opt shard for one dense bank.

    ``model_param_gbuf_map`` describes locally owned reduce-scatter ranges, so
    a valid rank may own none of a small bank.  Global ownership is checked
    only after gathering the local records below.
    """
    owners = [
        (index, child)
        for index, child in enumerate(_distributed_optimizer_leaves(bundle.optimizer))
        if parameter in getattr(child, "model_param_gbuf_map", {})
    ]
    assert len(owners) <= 1, "one local bank range cannot belong to two dist-opt leaves"
    if not owners:
        return None
    leaf_index, optimizer = owners[0]
    assert getattr(parameter, "allreduce", None) is True
    gbuf_index, dtype, bucket_index = optimizer.model_param_gbuf_map[parameter]
    range_map = optimizer.gbuf_ranges[gbuf_index][dtype][bucket_index]
    param_range = range_map["param_map"][parameter]
    buffer = optimizer.buffers[gbuf_index]
    bucket = buffer.buckets[bucket_index]
    dense_dp_ranks = _group_ranks(bundle.parallel_state.dp_group)
    assert _group_ranks(optimizer.data_parallel_group) == dense_dp_ranks
    assert _group_ranks(buffer.data_parallel_group) == dense_dp_ranks
    if bundle.parallel_state.ep_size > 1:
        assert dense_dp_ranks != _group_ranks(bundle.parallel_state.ep_dp_group)
    return leaf_index, optimizer, param_range, buffer, bucket


def _dense_owner_record(
    bundle, leaf_index: int, optimizer, buffer
) -> dict[str, object]:
    """Serialize the leaf/group contract for cross-rank ownership validation."""
    return {
        "leaf_index": leaf_index,
        "owner_group_ranks": _group_ranks(optimizer.data_parallel_group),
        "buffer_group_ranks": _group_ranks(buffer.data_parallel_group),
    }


def _assert_dense_owner_contract(
    bundle, name: str, records: list[dict[str, object]]
) -> None:
    """Prove every local shard maps to one dense leaf, never the EP child."""
    assert records, f"no distributed-optimizer shard found for semantic bank {name}"
    dense_dp_ranks = _group_ranks(bundle.parallel_state.dp_group)
    assert len({record["leaf_index"] for record in records}) == 1
    assert all(record["owner_group_ranks"] == dense_dp_ranks for record in records)
    assert all(record["buffer_group_ranks"] == dense_dp_ranks for record in records)
    if bundle.parallel_state.ep_size > 1:
        expert_dp_ranks = _group_ranks(bundle.parallel_state.ep_dp_group)
        assert all(record["owner_group_ranks"] != expert_dp_ranks for record in records)
        assert all(
            record["buffer_group_ranks"] != expert_dp_ranks for record in records
        )


def _local_full_bank_contributions(bundle) -> dict[str, torch.Tensor]:
    """Read unsynchronized full gradients for the independent DP oracle arm."""
    return {
        name: _main_grad(parameter).detach().float().cpu().clone()
        for name, parameter in _bank_parameters(bundle).items()
    }


def _dense_dp_absolute_oracle(
    bundle, local_contributions: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Independently average rank-local full grads and audit live normalization."""
    gathered = [None] * dist.get_world_size(bundle.parallel_state.dp_group)
    dist.all_gather_object(
        gathered, local_contributions, group=bundle.parallel_state.dp_group
    )
    expected_keys = set(local_contributions)
    assert all(set(contribution) == expected_keys for contribution in gathered)
    local_factors: dict[str, dict[str, object]] = {}
    for name, parameter in _bank_parameters(bundle).items():
        owned = _optimizer_param_range(bundle, parameter)
        if owned is None:
            continue
        leaf_index, owner, _param_range, buffer, bucket = owned
        local_factors[name] = {
            **_dense_owner_record(bundle, leaf_index, owner, buffer),
            "factor": float(bucket.gradient_scaling_factor),
        }
    gathered_factors = [None] * len(gathered)
    dist.all_gather_object(
        gathered_factors, local_factors, group=bundle.parallel_state.dp_group
    )
    expected: dict[str, torch.Tensor] = {}
    factors: dict[str, float] = {}
    expected_factor = 1.0 / len(gathered)
    for name in sorted(expected_keys):
        records = [
            factor_map[name] for factor_map in gathered_factors if name in factor_map
        ]
        _assert_dense_owner_contract(bundle, name, records)
        seen = {record["factor"] for record in records}
        assert len(seen) == 1, f"missing or inconsistent dense-DP factor for {name}"
        factor = seen.pop()
        assert factor == expected_factor, (
            f"dense-DP bucket factor for {name} is {factor}, expected {expected_factor}"
        )
        factors[name] = factor
        expected[name] = torch.stack(
            [contribution[name] for contribution in gathered]
        ).mean(0)
    return expected, factors


def _reconstruct_optimizer_owned_bank_grads(bundle) -> dict[str, torch.Tensor]:
    """Rebuild each logical bank grad from dist-opt's owned reduce-scatter shards."""
    local_shards: dict[str, dict[str, object]] = {}
    for name, parameter in _bank_parameters(bundle).items():
        owned = _optimizer_param_range(bundle, parameter)
        if owned is None:
            continue
        leaf_index, owner, param_range, buffer, bucket = owned
        # ``gbuf_world_in_bucket`` identifies the reduce-scatter output view;
        # ``param`` maps that view back to the logical bank tensor's offsets.
        reduced = bucket.grad_data.view(-1)[
            param_range["gbuf_world_in_bucket"].start : param_range[
                "gbuf_world_in_bucket"
            ].end
        ].detach()
        local_shards[name] = {
            **_dense_owner_record(bundle, leaf_index, owner, buffer),
            "start": param_range["param"].start,
            "end": param_range["param"].end,
            "shape": tuple(parameter.shape),
            "dtype": str(reduced.dtype),
            "values": reduced.float().cpu().clone(),
        }

    gathered = [None] * dist.get_world_size(bundle.parallel_state.dp_group)
    dist.all_gather_object(gathered, local_shards, group=bundle.parallel_state.dp_group)
    expected_keys = set(_bank_parameters(bundle))
    assert set().union(*(set(shards) for shards in gathered)) == expected_keys
    reconstructed: dict[str, torch.Tensor] = {}
    for name in sorted(expected_keys):
        shape = tuple(_bank_parameters(bundle)[name].shape)
        flat = torch.empty(int(torch.tensor(shape).prod()), dtype=torch.float32)
        ranges = []
        records = []
        for shards in gathered:
            if name not in shards:
                continue
            shard = shards[name]
            records.append(shard)
            start, end = int(shard["start"]), int(shard["end"])
            values = shard["values"]
            assert end - start == values.numel()
            ranges.append((start, end))
            flat[start:end].copy_(values)
        _assert_dense_owner_contract(bundle, name, records)
        offset = 0
        for start, end in sorted(ranges):
            assert start == offset, f"non-contiguous dist-opt shard coverage for {name}"
            offset = end
        assert offset == flat.numel(), f"incomplete dist-opt shard coverage for {name}"
        reconstructed[name] = flat.reshape(shape)
    return reconstructed


def _run_unsynchronized_reference(
    bundle, batch: PackedBatch
) -> tuple[float, dict[str, torch.Tensor]]:
    """Take rank-local contributions without permitting DDP gradient sync."""
    chunk = bundle.chunks[0]
    qwen = getattr(chunk, "module", chunk)
    dispatcher = qwen.layers[0].moe.dispatcher
    dispatch_calls = []
    original_dispatch = dispatcher.dispatch

    def capture_dispatch(*args, **kwargs):
        dispatch_calls.append(True)
        return original_dispatch(*args, **kwargs)

    dispatcher.dispatch = capture_dispatch
    try:
        output = bundle.forward_step(chunk, batch)
        assert output["loss"].isfinite()
        bundle.optimizer.grad_sync_enabled = False
        output["loss"].backward()
        assert dispatch_calls == [True]
    finally:
        dispatcher.dispatch = original_dispatch
    return float(output["loss"].detach().cpu()), _local_full_bank_contributions(bundle)


def _run_production_forward_and_finalize(
    bundle, batch: PackedBatch
) -> tuple[float, dict[str, torch.Tensor]]:
    """Run the runtime microbatch lifecycle then reconstruct reduced dist-opt shards."""
    chunk = bundle.chunks[0]
    qwen = getattr(chunk, "module", chunk)
    dispatcher = qwen.layers[0].moe.dispatcher
    dispatch_calls = []
    original_dispatch = dispatcher.dispatch

    def capture_dispatch(*args, **kwargs):
        dispatch_calls.append(True)
        return original_dispatch(*args, **kwargs)

    dispatcher.dispatch = capture_dispatch
    try:
        output = run_microbatch_loop(
            chunk,
            iter((batch,)),
            1,
            bundle.forward_step,
            optimizer=bundle.optimizer,
            dist_opt=True,
        )
        assert output["loss"].isfinite()
        assert bundle.optimizer.grad_sync_enabled is True
        assert dispatch_calls == [True]
    finally:
        dispatcher.dispatch = original_dispatch
    # MCore finalize_model_grads owns the DDP finish; do not duplicate it here.
    bundle.finalize_grads()
    return float(
        output["loss"].detach().cpu()
    ), _reconstruct_optimizer_owned_bank_grads(bundle)


def _clear_gradients(bundle) -> None:
    for chunk in bundle.chunks:
        chunk.zero_grad_buffer()
    bundle.optimizer.zero_grad()


def _batch() -> PackedBatch:
    torch.manual_seed(4100 + dist.get_rank())
    return PackedBatch(
        input_ids=torch.randint(0, 64, (4,), device="cuda"),
        labels=torch.randint(0, 64, (4,), device="cuda"),
        seq_lens=torch.tensor([4], device="cuda", dtype=torch.int64),
        extras={"multi_lora_slots": {0: torch.tensor([0, 0, 1, 1], device="cuda")}},
    )


def _batch_record(batch: PackedBatch) -> dict[str, dict[str, object]]:
    return {
        "input_ids": _tensor_record(batch.input_ids),
        "labels": _tensor_record(batch.labels),
        "seq_lens": _tensor_record(batch.seq_lens),
        "slots": _tensor_record(batch.extras["multi_lora_slots"][0]),
    }


def _expected_semantic_bank_keys(state) -> set[str]:
    """Fixed tiny-Qwen surface contract: one layer, fc1/fc2, A/B factors."""
    assert state.local_layer_indices == (0,)
    surfaces = (
        "layers.0.moe.experts._fc1_weight_0",
        "layers.0.moe.experts._fc2_weight_0",
    )
    assert set(state.registry.banks) == set(surfaces)
    return {
        state.parameter_name(surface, factor)
        for surface in surfaces
        for factor in ("a", "b")
    }


@pytest.mark.timeout(120)
def test_ep2_production_builder_distopt_finalize_and_identity_roundtrip(tmp_path):
    """EP1 oracle and EP2-local DCP prove exactly one EP2 reduction."""
    _Qwen3MoEConfig, model, protocol = _qwen_symbols()
    phase = _phase()
    artifact_dir = _phase_artifact_dir()
    checkpoint_dir = artifact_dir / "production_checkpoint"
    manifest_path = artifact_dir / "manifest.json"
    complete_path = artifact_dir / "COMPLETE"
    phase_b_receipt_path = artifact_dir / "EP2_COMPLETE"
    candidate_sha = os.environ.get("MLITE_CANDIDATE_SHA")
    candidate_tree = os.environ.get("MLITE_CANDIDATE_TREE_SHA")
    candidate_diff = os.environ.get("MLITE_CANDIDATE_DIFF_SHA")
    assert candidate_sha and candidate_tree and candidate_diff, (
        "candidate commit, tree, and diff hashes must bind the phase artifact"
    )

    if phase == "ep1_oracle":
        if dist.get_rank() == 0:
            assert not artifact_dir.exists(), (
                "phase artifact directory already exists; refuse stale COMPLETE/checkpoint reuse"
            )
            artifact_dir.mkdir(parents=True)
        dist.barrier()
        bundle = _build_bundle(ep=1, model_seed=3100)
        ep1_topology = _actual_topology(bundle)
        assert ep1_topology["dp"] == 2
        assert ep1_topology["expert_dp"] == 2
        ep1_topologies = [None, None]
        dist.all_gather_object(ep1_topologies, ep1_topology)
        _initialize_nonzero_banks(bundle)
        _configure_fixed_router(bundle)
        batch = _batch()
        loss, local_contributions = _run_unsynchronized_reference(bundle, batch)
        expected_keys = _expected_semantic_bank_keys(
            bundle.extras["multi_lora_training_state"]
        )
        assert set(local_contributions) == expected_keys
        assert all(
            gradient.abs().max() > 0 for gradient in local_contributions.values()
        )
        oracle_grads, normalization_by_key = _dense_dp_absolute_oracle(
            bundle, local_contributions
        )
        assert set(oracle_grads) == expected_keys
        gathered_contributions = [None] * dist.get_world_size(
            bundle.parallel_state.dp_group
        )
        dist.all_gather_object(
            gathered_contributions,
            local_contributions,
            group=bundle.parallel_state.dp_group,
        )
        assert any(
            not torch.equal(
                gathered_contributions[0][name], gathered_contributions[1][name]
            )
            for name in expected_keys
        ), "rank-specific DP batches must produce distinct local contributions"
        losses = [None, None]
        batches = [None, None]
        oracle_tensors_by_rank = [None, None]
        local_contribution_tensors_by_rank = [None, None]
        dist.all_gather_object(losses, loss)
        dist.all_gather_object(batches, _batch_record(batch))
        dist.all_gather_object(
            oracle_tensors_by_rank,
            {name: _tensor_record(value) for name, value in oracle_grads.items()},
        )
        dist.all_gather_object(
            local_contribution_tensors_by_rank,
            {
                name: _tensor_record(value)
                for name, value in local_contributions.items()
            },
        )
        dist.barrier()
        oracle_path = artifact_dir / f"ep1_bank_grads_rank_{dist.get_rank():05d}.pt"
        local_path = (
            artifact_dir / f"ep1_local_bank_grads_rank_{dist.get_rank():05d}.pt"
        )
        torch.save(oracle_grads, oracle_path)
        torch.save(local_contributions, local_path)
        dist.barrier()
        if dist.get_rank() == 0:
            manifest = {
                "schema_version": 2,
                "candidate": {
                    "commit": candidate_sha,
                    "tree": candidate_tree,
                    "diff": candidate_diff,
                },
                "test_sha256": _sha256(Path(__file__)),
                "model_config": _config().to_dict(),
                "topology": {"ep1_oracle": ep1_topologies, "ep2_verify": None},
                "impl_config": {
                    "ep1_oracle": _impl_contract(ep=1),
                    "ep2_verify": _impl_contract(ep=2),
                },
                "seeds": {"model": 3100, "batch_base": 4100},
                "batch_by_rank": batches,
                "loss_by_rank": losses,
                "loss_normalization": "model token-loss mean",
                "dense_dp_gradient_scaling_by_key": normalization_by_key,
                "semantic_bank_keys": sorted(expected_keys),
                "semantic_bank_tensors_by_rank": oracle_tensors_by_rank,
                "local_contribution_tensors_by_rank": local_contribution_tensors_by_rank,
                "oracle_files": {
                    f"rank_{rank:05d}": _sha256(
                        artifact_dir / f"ep1_bank_grads_rank_{rank:05d}.pt"
                    )
                    for rank in range(2)
                },
                "local_contribution_files": {
                    f"rank_{rank:05d}": _sha256(
                        artifact_dir / f"ep1_local_bank_grads_rank_{rank:05d}.pt"
                    )
                    for rank in range(2)
                },
            }
            temporary_manifest = manifest_path.with_name(f".manifest.{os.getpid()}.tmp")
            temporary_manifest.write_text(
                json.dumps(manifest, sort_keys=True, indent=2)
            )
            os.replace(temporary_manifest, manifest_path)
            complete_payload = {
                "commit": candidate_sha,
                "manifest_sha256": _sha256(manifest_path),
                "phase": "ep1_oracle",
            }
            temporary_complete = complete_path.with_name(f".COMPLETE.{os.getpid()}.tmp")
            temporary_complete.write_text(json.dumps(complete_payload, sort_keys=True))
            os.replace(temporary_complete, complete_path)
        dist.barrier()
        return

    complete = json.loads(complete_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert complete == {
        "commit": candidate_sha,
        "manifest_sha256": _sha256(manifest_path),
        "phase": "ep1_oracle",
    }
    assert manifest["candidate"] == {
        "commit": candidate_sha,
        "tree": candidate_tree,
        "diff": candidate_diff,
    }
    assert manifest["test_sha256"] == _sha256(Path(__file__))
    assert manifest["schema_version"] == 2
    assert manifest["seeds"] == {
        "model": 3100,
        "batch_base": 4100,
    }
    assert manifest["loss_normalization"] == "model token-loss mean"
    assert set(manifest["dense_dp_gradient_scaling_by_key"]) == set(
        manifest["semantic_bank_keys"]
    )
    assert manifest["model_config"] == _config().to_dict()
    assert manifest["impl_config"] == {
        "ep1_oracle": _impl_contract(ep=1),
        "ep2_verify": _impl_contract(ep=2),
    }
    assert manifest["topology"]["ep1_oracle"][dist.get_rank()] == {
        "world": 2,
        "tp": 1,
        "ep": 1,
        "etp": 1,
        "pp": 1,
        "cp": 1,
        "dp": 2,
        "dp_rank": dist.get_rank(),
        "expert_dp": 2,
        "expert_dp_rank": dist.get_rank(),
    }
    assert manifest["topology"]["ep2_verify"] is None
    batch = _batch()
    expected_batch = manifest["batch_by_rank"][dist.get_rank()]
    assert _batch_record(batch) == expected_batch
    for rank in range(2):
        path = artifact_dir / f"ep1_bank_grads_rank_{rank:05d}.pt"
        assert _sha256(path) == manifest["oracle_files"][f"rank_{rank:05d}"]
        local_path = artifact_dir / f"ep1_local_bank_grads_rank_{rank:05d}.pt"
        assert (
            _sha256(local_path)
            == manifest["local_contribution_files"][f"rank_{rank:05d}"]
        )
    oracle_path = artifact_dir / f"ep1_bank_grads_rank_{dist.get_rank():05d}.pt"
    oracle_grads = torch.load(oracle_path, map_location="cpu", weights_only=True)
    assert {
        name: _tensor_record(value) for name, value in oracle_grads.items()
    } == manifest["semantic_bank_tensors_by_rank"][dist.get_rank()]

    bundle = _build_bundle(ep=2, model_seed=3100)
    ep2_topology = _actual_topology(bundle)
    assert ep2_topology == {
        "world": 2,
        "tp": 1,
        "ep": 2,
        "etp": 1,
        "pp": 1,
        "cp": 1,
        "dp": 2,
        "dp_rank": dist.get_rank(),
        "expert_dp": 1,
        "expert_dp_rank": 0,
    }
    state = bundle.extras["multi_lora_training_state"]
    assert state is not None
    assert bundle.extras["optimizer_backend"] == "dist_opt"
    assert callable(bundle.finalize_grads)
    assert all(
        getattr(parameter, "allreduce", None) is True
        for parameter in state.parameters()
    )
    expected_keys = _expected_semantic_bank_keys(state)
    assert set(oracle_grads) == expected_keys == set(manifest["semantic_bank_keys"])
    _initialize_nonzero_banks(bundle)

    # The real production injection creates model-owned sidecars.  They must
    # never request the legacy explicit EP all-reduce; dense dist-opt finalize
    # owns the one reduction for these registered parameters.
    kwargs = {}
    probe_batch = type(
        "Batch",
        (),
        {
            "extras": {
                "multi_lora_slots": {0: torch.zeros(1, device="cuda", dtype=torch.long)}
            }
        },
    )()
    protocol._inject_multi_lora_sidecars(kwargs, probe_batch, state)
    sidecar = kwargs["multi_lora_sidecars"][0]
    assert sidecar.requires_explicit_ep_sync is False
    assert model._sidecar_ep_sync_group(bundle.parallel_state, sidecar) is None

    # This is intentionally an EP2-to-EP2 DCP round trip.  Do not turn this
    # smoke back into a cross-EP reshard test: that is a known production DCP
    # limitation outside multi-LoRA's ownership contract.
    checkpoint_semantics = _checkpoint_semantics(bundle)
    dcp.save_training_checkpoint(
        bundle.chunks,
        bundle.optimizer,
        1,
        str(checkpoint_dir),
        config=_config(),
        ps=bundle.parallel_state,
        use_dcp=True,
        save_optimizer=False,
    )
    dist.barrier()
    checkpoint_files = {
        str(path.relative_to(checkpoint_dir)): _sha256(path)
        for path in sorted(checkpoint_dir.rglob("*"))
        if path.is_file()
    }
    assert checkpoint_files
    checkpoint_files_by_rank = [None] * dist.get_world_size()
    dist.all_gather_object(checkpoint_files_by_rank, checkpoint_files)
    assert all(
        rank_checkpoint_files == checkpoint_files
        for rank_checkpoint_files in checkpoint_files_by_rank
    )
    _poison_parameters(bundle)
    assert {
        str(path.relative_to(checkpoint_dir)): _sha256(path)
        for path in sorted(checkpoint_dir.rglob("*"))
        if path.is_file()
    } == checkpoint_files
    pre_load_semantics = _checkpoint_semantics(bundle)
    for category in ("dense", "experts", "banks"):
        assert set(pre_load_semantics[category]) == set(checkpoint_semantics[category])
        for name in checkpoint_semantics[category]:
            assert (
                pre_load_semantics[category][name]["sha256"]
                != checkpoint_semantics[category][name]["sha256"]
            )
    assert (
        dcp.load_training_checkpoint(
            bundle.chunks,
            bundle.optimizer,
            str(checkpoint_dir),
            config=_config(),
            ps=bundle.parallel_state,
            use_dcp=True,
            load_optimizer=False,
        )
        == 1
    )
    post_load_semantics = _checkpoint_semantics(bundle)
    assert post_load_semantics == checkpoint_semantics
    _configure_fixed_router(bundle)
    _loss, ep2_grads = _run_production_forward_and_finalize(bundle, batch)
    assert set(ep2_grads) == expected_keys
    for name, oracle in oracle_grads.items():
        assert (
            _tensor_record(oracle)
            == manifest["semantic_bank_tensors_by_rank"][dist.get_rank()][name]
        )
        torch.testing.assert_close(ep2_grads[name].cpu(), oracle, rtol=0, atol=0)

    # This actual production-forward negative control restores explicit EP
    # sync after clearing the correct arm's gradients.  It must double EP1.
    _clear_gradients(bundle)
    original_sidecar = protocol.MoELoraSidecar

    def force_explicit_sync(*args, **kwargs):
        kwargs["requires_explicit_ep_sync"] = True
        return original_sidecar(*args, **kwargs)

    protocol.MoELoraSidecar = force_explicit_sync
    try:
        _loss, doubled_grads = _run_production_forward_and_finalize(bundle, batch)
    finally:
        protocol.MoELoraSidecar = original_sidecar
    assert set(doubled_grads) == expected_keys
    for name, oracle in oracle_grads.items():
        torch.testing.assert_close(
            doubled_grads[name].cpu(), oracle * 2, rtol=0, atol=0
        )
    for name, param in state.named_parameters():
        assert name.startswith("bank_") and "." not in name
    ep2_topologies = [None, None]
    dist.all_gather_object(ep2_topologies, ep2_topology)
    if dist.get_rank() == 0:
        receipt = {
            "commit": candidate_sha,
            "manifest_sha256": _sha256(manifest_path),
            "phase": "ep2_verify",
            "topology_by_rank": ep2_topologies,
            "checkpoint_files": checkpoint_files,
        }
        temporary_receipt = phase_b_receipt_path.with_name(
            f".EP2_COMPLETE.{os.getpid()}.tmp"
        )
        temporary_receipt.write_text(json.dumps(receipt, sort_keys=True))
        os.replace(temporary_receipt, phase_b_receipt_path)
    dist.barrier()
