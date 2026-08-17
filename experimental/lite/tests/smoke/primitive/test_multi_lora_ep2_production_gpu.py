# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Two-GPU production lifecycle proof for model-owned Qwen3-MoE multi-LoRA."""

from __future__ import annotations

import hashlib
import json
import os
import re
import traceback
from pathlib import Path
from typing import Callable

import lora_distributed_test_utils as lora_dist_utils
import megatron.lite.runtime.contracts.config as runtime_config
import pytest
import torch
import torch.distributed as dist
from megatron.lite.primitive.ckpt import dcp
from megatron.lite.primitive.train_step import run_microbatch_loop
from megatron.lite.runtime.contracts.data import PackedBatch

pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]

_TINY_QWEN_EXPERTS = 4


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
    if int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4):
        pytest.skip("run this smoke through torchrun with 2 or 4 ranks")
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
        num_experts=_TINY_QWEN_EXPERTS,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        max_position_embeddings=16,
        layer_types=["full_attention"],
    )


def _build_bundle(*, tp: int = 1, ep: int, model_seed: int):
    _Qwen3MoEConfig, _model, protocol = _qwen_symbols()
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    return protocol.build_model(
        _config(),
        impl_cfg=protocol.ImplConfig(
            parallel=runtime_config.ParallelConfig(tp=tp, ep=ep, etp=1, pp=1, cp=1),
            optimizer="dist_opt",
            optimizer_config=runtime_config.OptimizerConfig(
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
        pytest.skip("run through the two-phase production multi-LoRA orchestrator")
    return Path(raw)


def _phase() -> str:
    phase = os.environ.get("MLITE_MULTI_LORA_PHASE")
    if phase not in {"ep2_oracle", "ep2_verify"}:
        pytest.skip("set MLITE_MULTI_LORA_PHASE to ep2_oracle or ep2_verify")
    return phase


def _phase_parallel_config() -> dict[str, int]:
    """Select checkpoint topology from the launcher, not an EP-only name."""
    world = dist.get_world_size()
    if world == 2:
        return {"world": 2, "tp": 1, "ep": 2}
    if world == 4:
        return {"world": 4, "tp": 2, "ep": 2}
    raise AssertionError(f"unsupported phase world size {world}; expected 2 or 4")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    cpu = tensor.detach().cpu().contiguous()
    return {
        "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
    }


def _write_phase_a_flight_record(
    artifact_dir: Path, *, remove: tuple[str, ...] = (), **updates: object
) -> dict[str, object]:
    """Atomically persist rank-local Phase A progress without a collective."""
    path = artifact_dir / f"phase_a_rank_{dist.get_rank():05d}.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    for key in remove:
        record.pop(key, None)
    record.update(updates)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True))
    os.replace(temporary, path)
    return record


def _run_phase_a_stage(artifact_dir: Path, stage: str, action):
    """Record any rank-local Phase A failure before a later collective."""
    _write_phase_a_flight_record(artifact_dir, current_stage=stage)
    try:
        value = action()
    except Exception as error:
        _write_phase_a_flight_record(
            artifact_dir,
            stage_error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    _write_phase_a_flight_record(
        artifact_dir, remove=("stage_error", "traceback"), **{f"{stage}_done": True}
    )
    return value


def _named_parameters(bundle) -> dict[str, torch.Tensor]:
    parameters: dict[str, torch.Tensor] = {}
    for chunk in bundle.chunks:
        module = getattr(chunk, "module", chunk)
        parameters.update(dict(module.named_parameters()))
    return parameters


def _checkpoint_semantics(bundle) -> dict[str, dict[str, object]]:
    """Bind every named model parameter plus dense/expert/bank DCP semantics."""
    parameters = _named_parameters(bundle)
    state = bundle.extras["multi_lora_training_state"]
    bank_records = {
        name: _tensor_record(parameter) for name, parameter in state.named_parameters()
    }
    assert set(bank_records) == _expected_semantic_bank_keys(state)
    bank_prefix = "multi_lora_training_state."
    model_records = {
        name: _tensor_record(parameter)
        for name, parameter in parameters.items()
        if not name.startswith(bank_prefix)
    }
    assert set(parameters) == set(model_records) | {
        f"{bank_prefix}{name}" for name in bank_records
    }
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
    return {
        "model": model_records,
        "dense": {dense_name: _tensor_record(parameters[dense_name])},
        "experts": expert_records,
        "banks": bank_records,
    }


def _poison_parameters(bundle) -> None:
    """Make pre-load values observably unlike a checkpoint from another seed."""
    with torch.no_grad():
        for parameter in _named_parameters(bundle).values():
            parameter.add_(0.125)


def _actual_topology(bundle) -> dict[str, object]:
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
        "tp_group_ranks": sorted(dist.get_process_group_ranks(ps.tp_group)),
        "ep_group_ranks": sorted(dist.get_process_group_ranks(ps.ep_group)),
        "dp_group_ranks": sorted(dist.get_process_group_ranks(ps.dp_group)),
        "expert_dp_group_ranks": sorted(dist.get_process_group_ranks(ps.ep_dp_group)),
    }


def _assert_tp_ep_membership(bundle, *, tp: int, ep: int) -> None:
    topology = _actual_topology(bundle)
    assert topology["tp"] == tp and topology["ep"] == ep
    assert len(topology["tp_group_ranks"]) == tp
    assert len(topology["ep_group_ranks"]) == ep
    assert dist.get_rank() in topology["tp_group_ranks"]
    assert dist.get_rank() in topology["ep_group_ranks"]


@pytest.mark.timeout(seconds=120)
def test_tp2_ep1_attention_multi_lora_production_groups():
    """GPU production builder must expose a real TP2 attention-bank path."""
    if dist.get_world_size() != 2:
        pytest.skip("TP2/EP1 production smoke requires exactly two ranks")
    bundle = _build_bundle(tp=2, ep=1, model_seed=3201)
    _assert_tp_ep_membership(bundle, tp=2, ep=1)
    state = bundle.extras["multi_lora_training_state"]
    qkv, proj = state.attention_banks_for_layer(0)
    assert qkv.partition.rank_partitioned_a and proj.partition.output_partitioned_b
    assert qkv.a_bank.requires_grad and proj.b_bank.requires_grad
    _run_bank_gradient_contract(bundle)


@pytest.mark.timeout(seconds=120)
def test_tp2_ep2_attention_and_expert_multi_lora_production_groups():
    """Four-rank production construction keeps TP attention orthogonal to EP experts."""
    if dist.get_world_size() != 4:
        pytest.skip("TP2/EP2 production smoke requires exactly four ranks")
    bundle = _build_bundle(tp=2, ep=2, model_seed=3202)
    _assert_tp_ep_membership(bundle, tp=2, ep=2)
    state = bundle.extras["multi_lora_training_state"]
    fc1, fc2 = state.banks_for_layer(0)
    qkv, proj = state.attention_banks_for_layer(0)
    assert fc1.partition.tp_size == fc2.partition.tp_size == 1
    assert qkv.partition.rank_partitioned_a and proj.partition.output_partitioned_b
    _run_bank_gradient_contract(bundle)


def _impl_contract(*, tp: int = 1, ep: int) -> dict[str, object]:
    """All implementation knobs that define this production evidence."""
    return {
        "parallel": {"tp": tp, "ep": ep, "etp": 1, "pp": 1, "cp": 1},
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
    """Route every token to expert zero for production empty-rank coverage."""
    return (
        torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype),
        torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.long),
    )


def _phase_a_balanced_router(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Route TP-local rows across both EP partitions in Phase A."""
    experts_per_ep = _TINY_QWEN_EXPERTS // 2
    return (
        torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype),
        ((torch.arange(x.shape[0], device=x.device) % 2) * experts_per_ep).view(-1, 1),
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


def _configure_phase_a_balanced_router(bundle) -> None:
    """Bind Phase A oracle and Phase B parity runs to the same route graph."""
    for chunk in bundle.chunks:
        qwen = getattr(chunk, "module", chunk)
        qwen.layers[0].moe.router.forward = _phase_a_balanced_router


def _bank_parameters(bundle) -> dict[str, torch.Tensor]:
    state = bundle.extras["multi_lora_training_state"]
    parameters = dict(state.named_parameters())
    assert set(parameters) == _expected_semantic_bank_keys(state)
    return parameters


def _bank_surface_kind(bundle, parameter: torch.Tensor) -> str:
    """Classify encoded bank parameters through the registry's native surface."""
    state = bundle.extras["multi_lora_training_state"]
    matches = []
    for surface, bank in state.registry.banks.items():
        if parameter is bank.a_bank or parameter is bank.b_bank:
            matches.append(surface)
    assert (
        len(matches) == 1
    ), f"bank parameter has ambiguous/missing registry surface: {matches}"
    surface = matches[0]
    if ".moe.experts._fc" in surface:
        return "fc"
    if ".attn." in surface:
        return "attention"
    raise AssertionError(f"unsupported model-owned bank surface: {surface}")


def _run_bank_gradient_contract(bundle) -> None:
    """Exercise mixed token routing and inspect each semantic A/B slot axis."""
    state = bundle.extras["multi_lora_training_state"]
    with torch.no_grad():
        for bank in state.registry.banks.values():
            bank.a_bank[0].fill_(0.125)
            bank.b_bank[0].fill_(0.125)
            bank.a_bank[1].fill_(0.25)
            bank.b_bank[1].fill_(0.25)
    _configure_fixed_router(bundle)
    batch = _production_batch(bundle)
    parameters = _bank_parameters(bundle)

    def assert_slots(gradients, *, active: tuple[int, ...]) -> None:
        assert set(gradients) == _expected_semantic_bank_keys(state)
        for name, gradient in gradients.items():
            assert gradient.shape == parameters[name].shape
            assert name.rsplit("_", 1)[-1] in {"a", "b"}
            for slot in range(gradient.shape[0]):
                if slot in active:
                    assert gradient[slot].abs().max() > 0, (name, slot)
                else:
                    assert gradient[slot].eq(0).all(), (name, slot)

    # Independent miss arm: all factors are nonzero but every token selects
    # alpha.  Its bravo gradients must be exactly zero before the mixed arm.
    batch.extras["multi_lora_slots"][0].zero_()
    loss, gradients = _run_production_forward_and_finalize(bundle, batch)
    assert loss == loss
    assert_slots(gradients, active=(0,))
    _clear_gradients(bundle)

    # A separate lifecycle establishes that the same semantic A/B slot axis
    # receives gradients for both adapters under mixed token routing.
    batch.extras["multi_lora_slots"][0].copy_(
        torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.long)
    )
    loss, gradients = _run_production_forward_and_finalize(bundle, batch)
    assert loss == loss
    assert_slots(gradients, active=(0, 1))


def _adapter_export_oracle(bundle) -> dict[str, dict[str, torch.Tensor]]:
    """Export both slots through the real TP materialization + native mapper."""
    import megatron.lite.model.qwen3_moe.lite.checkpoint as qwen_checkpoint
    from megatron.lite.primitive.ckpt.hf_weights import VLLM_LORA_NAME_PREFIX

    state = bundle.extras["multi_lora_training_state"]
    expected_modules = {
        f"model.layers.0.mlp.experts.{expert}.{projection}"
        for expert in range(_config().num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    } | {
        f"model.layers.0.self_attn.{projection}"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    expected_keys = {
        f"{VLLM_LORA_NAME_PREFIX}{module}.lora_{factor}.weight"
        for module in expected_modules
        for factor in ("A", "B")
    }
    exported = {
        name: state.registry.export_hf_state(
            name, qwen_checkpoint.Qwen3MoEWeightSpec(_config()), bundle.parallel_state
        )
        for name in ("alpha", "bravo")
    }
    for name, tensors in exported.items():
        assert set(tensors) == expected_keys, name
        assert all(tensor.ndim == 2 for tensor in tensors.values())
        assert all(
            tensor.shape[0 if key.endswith("lora_A.weight") else 1]
            == state.registry.rank
            for key, tensor in tensors.items()
        )
    assert any(
        not torch.equal(exported["alpha"][key], exported["bravo"][key])
        for key in expected_keys
    )
    records = {
        name: {key: _tensor_record(value) for key, value in tensors.items()}
        for name, tensors in exported.items()
    }
    gathered = [None] * bundle.parallel_state.dp_size
    dist.all_gather_object(gathered, records, group=bundle.parallel_state.dp_group)
    assert all(record == gathered[0] for record in gathered)
    return exported


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
    a valid rank may own none of a small bank.  Owner-group membership is
    checked only after gathering that group's local records below.
    """
    is_fc = _bank_surface_kind(bundle, parameter) == "fc"
    owners = [
        (index, child)
        for index, child in enumerate(_distributed_optimizer_leaves(bundle.optimizer))
        if parameter in getattr(child, "model_param_gbuf_map", {})
    ]
    assert len(owners) <= 1, "one local bank range cannot belong to two dist-opt leaves"
    if not owners:
        return None
    leaf_index, optimizer = owners[0]
    _assert_model_owned_bank_allreduce(parameter, is_fc=is_fc)
    gbuf_index, dtype, bucket_index = optimizer.model_param_gbuf_map[parameter]
    range_map = optimizer.gbuf_ranges[gbuf_index][dtype][bucket_index]
    param_range = range_map["param_map"][parameter]
    buffer = optimizer.buffers[gbuf_index]
    bucket = buffer.buckets[bucket_index]
    expected_group = lora_dist_utils.select_lora_bank_owner_group(
        bundle.parallel_state, is_expert_bank=is_fc
    )
    expected_ranks = _group_ranks(expected_group)
    assert _group_ranks(optimizer.data_parallel_group) == expected_ranks
    assert _group_ranks(buffer.data_parallel_group) == expected_ranks
    return leaf_index, optimizer, param_range, buffer, bucket


def _assert_model_owned_bank_allreduce(parameter: torch.Tensor, *, is_fc: bool) -> None:
    """Check explicit optimizer ownership without relying on bool identity."""
    value = getattr(parameter, "allreduce", None)
    assert value is not None, "model-owned LoRA bank is missing allreduce ownership"
    # Megatron may attach a bool-like scalar rather than Python's singleton.
    assert bool(value) == (not is_fc)


def _owner_group_record(
    bundle, leaf_index: int, optimizer, buffer
) -> dict[str, object]:
    """Serialize one optimizer leaf's owner-group contract."""
    return {
        "leaf_index": leaf_index,
        "owner_group_ranks": _group_ranks(optimizer.data_parallel_group),
        "buffer_group_ranks": _group_ranks(buffer.data_parallel_group),
    }


def _assert_owner_group_contract(
    bundle, name: str, records: list[dict[str, object]]
) -> None:
    """Prove every local shard maps to the intended owner-group leaf."""
    assert records, f"no distributed-optimizer shard found for semantic bank {name}"
    is_fc = _bank_surface_kind(bundle, _bank_parameters(bundle)[name]) == "fc"
    expected_ranks = _group_ranks(
        lora_dist_utils.select_lora_bank_owner_group(
            bundle.parallel_state, is_expert_bank=is_fc
        )
    )
    assert len({record["leaf_index"] for record in records}) == 1
    assert all(record["owner_group_ranks"] == expected_ranks for record in records)
    assert all(record["buffer_group_ranks"] == expected_ranks for record in records)


def _local_full_bank_contributions(bundle) -> dict[str, torch.Tensor]:
    """Read unsynchronized full gradients for the independent DP oracle arm."""
    return {
        name: _main_grad(parameter).detach().cpu().clone()
        for name, parameter in _bank_parameters(bundle).items()
    }


def _local_owner_factor_record(bundle, name: str) -> dict[str, object] | None:
    """Inspect one local factor only inside its owner-lane error envelope."""
    parameter = _bank_parameters(bundle)[name]
    owned = _optimizer_param_range(bundle, parameter)
    if owned is None:
        return None
    leaf_index, owner, _param_range, buffer, bucket = owned
    return {
        **_owner_group_record(bundle, leaf_index, owner, buffer),
        "factor": float(bucket.gradient_scaling_factor),
    }


def _bank_sync_absolute_oracle(
    bundle,
    local_contributions: dict[str, torch.Tensor],
    *,
    preflight_done: Callable[[], None] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Reference FC and attention gradients over their respective owner groups."""
    expected: dict[str, torch.Tensor] = {}
    records = lora_dist_utils.preflight_lora_bank_collective_order(
        local_contributions,
        lambda name: lora_dist_utils.build_lora_collective_descriptor(
            _bank_surface_kind(bundle, _bank_parameters(bundle)[name]),
            local_contributions[name],
            _bank_parameters(bundle)[name].dtype,
        ),
    )
    print(
        f"MULTI_LORA_COLLECTIVE_PREFLIGHT rank={dist.get_rank()} records={records}",
        flush=True,
    )
    if preflight_done is not None:
        preflight_done()
    for name, _kind, _shape, _dtype in records:
        contribution = local_contributions[name]
        if _bank_surface_kind(bundle, _bank_parameters(bundle)[name]) == "fc":
            bank_dtype = _bank_parameters(bundle)[name].dtype
            assert bank_dtype is torch.bfloat16
            assert contribution.dtype is torch.float32
            value = contribution.to("cuda", dtype=bank_dtype)
            dist.all_reduce(value, group=bundle.parallel_state.ep_group)
            value = value.float()
            dist.all_reduce(value, group=bundle.parallel_state.ep_dp_group)
            value.div_(bundle.parallel_state.expert_dp_size)
        else:
            assert (
                _bank_surface_kind(bundle, _bank_parameters(bundle)[name])
                == "attention"
            )
            assert contribution.dtype is torch.float32
            value = contribution.to("cuda")
            value = value.float()
            dist.all_reduce(value, group=bundle.parallel_state.dp_group)
            value.div_(bundle.parallel_state.dp_size)
        expected[name] = value.cpu()
    expected_keys = set(local_contributions)
    optimizer_expected: dict[str, torch.Tensor] = {}
    factors: dict[str, float] = {}
    for name in sorted(expected_keys):
        is_fc = _bank_surface_kind(bundle, _bank_parameters(bundle)[name]) == "fc"
        owner_group = lora_dist_utils.select_lora_bank_owner_group(
            bundle.parallel_state, is_expert_bank=is_fc
        )
        expected_factor = 1.0 / (
            bundle.parallel_state.expert_dp_size
            if is_fc
            else bundle.parallel_state.dp_size
        )

        def validate_records(records) -> None:
            _assert_owner_group_contract(bundle, name, records)
            seen = {record["factor"] for record in records}
            assert (
                len(seen) == 1
            ), f"missing or inconsistent owner-group factor for {name}"
            factor = seen.pop()
            assert (
                factor == expected_factor
            ), f"owner-group factor for {name} is {factor}, expected {expected_factor}"

        records = lora_dist_utils.gather_owner_factor_records_or_raise(
            owner_group,
            lambda: _local_owner_factor_record(bundle, name),
            validate_records,
        )
        factor = records[0]["factor"]
        factors[name] = factor
        optimizer_expected[name] = expected[name]
    return optimizer_expected, factors


def _reconstruct_optimizer_owned_bank_grads(bundle) -> dict[str, torch.Tensor]:
    """Rebuild each logical bank grad from dist-opt's owned reduce-scatter shards."""
    parameters = _bank_parameters(bundle)
    reconstructed: dict[str, torch.Tensor] = {}
    for name in sorted(parameters):
        parameter = parameters[name]
        is_fc = _bank_surface_kind(bundle, parameter) == "fc"
        group = (
            bundle.parallel_state.ep_dp_group
            if is_fc
            else bundle.parallel_state.dp_group
        )
        local: dict[str, object] = {"error": None, "shard": None}
        # Ownership metadata is local and may be absent.  Never raise before
        # the owner group has collectively observed every rank's result.
        try:
            owned = _optimizer_param_range(bundle, parameter)
            if owned is not None:
                leaf_index, owner, param_range, buffer, bucket = owned
                reduced = bucket.grad_data.view(-1)[
                    param_range["gbuf_world_in_bucket"]
                    .start : param_range["gbuf_world_in_bucket"]
                    .end
                ].detach()
                local["shard"] = {
                    **_owner_group_record(bundle, leaf_index, owner, buffer),
                    "start": param_range["param"].start,
                    "end": param_range["param"].end,
                    "shape": tuple(parameter.shape),
                    "dtype": str(reduced.dtype),
                    "values": reduced.float().cpu().clone(),
                }
        except Exception as error:  # reported symmetrically below
            local["error"] = f"{type(error).__name__}: {error}"
        gathered = _gather_optimizer_owned_bank_shards(local, group=group)
        errors = [record["error"] for record in gathered if record["error"]]
        assert not errors, f"owner metadata error for {name}: {errors}"
        records = [
            record["shard"] for record in gathered if record["shard"] is not None
        ]
        shape = tuple(parameter.shape)
        flat = torch.empty(int(torch.tensor(shape).prod()), dtype=torch.float32)
        ranges = []
        for shard in records:
            start, end = int(shard["start"]), int(shard["end"])
            values = shard["values"]
            assert end - start == values.numel()
            ranges.append((start, end))
            flat[start:end].copy_(values)
        _assert_owner_group_contract(bundle, name, records)
        offset = 0
        for start, end in sorted(ranges):
            assert start == offset, f"non-contiguous dist-opt shard coverage for {name}"
            offset = end
        assert offset == flat.numel(), f"incomplete dist-opt shard coverage for {name}"
        reconstructed[name] = flat.reshape(shape)
    return reconstructed


def _gather_optimizer_owned_bank_shards(
    local_shard: dict[str, object], *, group
) -> list[dict[str, object]]:
    """Gather one semantic bank on its dist-opt owner group, non-owners included."""
    gathered: list[dict[str, object] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, local_shard, group=group)
    assert all(shards is not None for shards in gathered)
    return [shards for shards in gathered if shards is not None]


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
    router = qwen.layers[0].moe.router
    original_router = router.forward
    router.forward = _phase_a_balanced_router
    _Qwen3MoEConfig, _model, protocol = _qwen_symbols()
    original_sidecar = protocol.MoELoraSidecar

    def disable_explicit_sync(*args, **kwargs):
        kwargs["requires_explicit_ep_sync"] = False
        return original_sidecar(*args, **kwargs)

    protocol.MoELoraSidecar = disable_explicit_sync
    try:
        output = bundle.forward_step(chunk, batch)
        assert output["loss"].isfinite()
        bundle.optimizer.grad_sync_enabled = False
        output["loss"].backward()
        assert dispatch_calls == [True]
    finally:
        dispatcher.dispatch = original_dispatch
        router.forward = original_router
        protocol.MoELoraSidecar = original_sidecar
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


def _batch(*, dense_dp_rank: int) -> PackedBatch:
    # TP ranks share one dense-DP replica batch; replicas differ by dp rank.
    torch.manual_seed(4100 + dense_dp_rank)
    return PackedBatch(
        input_ids=torch.randint(0, 64, (4,), device="cuda"),
        labels=torch.randint(0, 64, (4,), device="cuda"),
        seq_lens=torch.tensor([4], device="cuda", dtype=torch.int64),
        extras={"multi_lora_slots": {0: torch.tensor([0, 0, 1, 1], device="cuda")}},
    )


def _production_batch(bundle) -> PackedBatch:
    batch = _batch(dense_dp_rank=bundle.parallel_state.dp_rank)
    record = _batch_record(batch)
    if bundle.parallel_state.tp_size == 1:
        return batch
    gathered = [None] * dist.get_world_size(bundle.parallel_state.tp_group)
    dist.all_gather_object(gathered, record, group=bundle.parallel_state.tp_group)
    assert all(item == gathered[0] for item in gathered)
    return batch


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
        "layers.0.attn.qkv.linear.weight",
        "layers.0.attn.proj.linear.weight",
    )
    assert set(state.registry.banks) == set(surfaces)
    return {
        state.parameter_name(surface, factor)
        for surface in surfaces
        for factor in ("a", "b")
    }


@pytest.mark.timeout(seconds=120)
def test_production_builder_distopt_finalize_and_identity_roundtrip(tmp_path):
    """Same-topology TP/EP oracle and DCP prove each bank's one reduction."""
    _Qwen3MoEConfig, model, protocol = _qwen_symbols()
    phase = _phase()
    phase_config = _phase_parallel_config()
    artifact_dir = _phase_artifact_dir()
    checkpoint_dir = artifact_dir / "production_checkpoint"
    manifest_path = artifact_dir / "manifest.json"
    complete_path = artifact_dir / "COMPLETE"
    phase_b_receipt_path = artifact_dir / "EP2_COMPLETE"
    candidate_sha = os.environ.get("MLITE_CANDIDATE_SHA")
    candidate_tree = os.environ.get("MLITE_CANDIDATE_TREE_SHA")
    candidate_diff = os.environ.get("MLITE_CANDIDATE_DIFF_SHA")
    assert (
        candidate_sha and candidate_tree and candidate_diff
    ), "candidate commit, tree, and diff hashes must bind the phase artifact"

    if phase == "ep2_oracle":
        if dist.get_rank() == 0:
            assert (
                not artifact_dir.exists()
            ), "phase artifact directory already exists; refuse stale COMPLETE/checkpoint reuse"
            artifact_dir.mkdir(parents=True)
        dist.barrier()
        bundle = _build_bundle(
            tp=phase_config["tp"], ep=phase_config["ep"], model_seed=3100
        )
        oracle_topology = _actual_topology(bundle)
        assert {
            key: oracle_topology[key] for key in ("world", "tp", "ep")
        } == phase_config
        assert set(oracle_topology) == {
            "world",
            "tp",
            "ep",
            "etp",
            "pp",
            "cp",
            "dp",
            "dp_rank",
            "expert_dp",
            "expert_dp_rank",
            "tp_group_ranks",
            "ep_group_ranks",
            "dp_group_ranks",
            "expert_dp_group_ranks",
        }
        oracle_topologies = [None] * phase_config["world"]
        dist.all_gather_object(oracle_topologies, oracle_topology)
        _initialize_nonzero_banks(bundle)
        _configure_fixed_router(bundle)
        batch = _production_batch(bundle)
        adapter_exports = _adapter_export_oracle(bundle)
        export_path = artifact_dir / f"adapter_exports_rank_{dist.get_rank():05d}.pt"
        torch.save(adapter_exports, export_path)
        dist.barrier()
        _write_phase_a_flight_record(artifact_dir, export_complete=True)
        parameter_semantics = _run_phase_a_stage(
            artifact_dir, "checkpoint_semantics", lambda: _checkpoint_semantics(bundle)
        )
        loss, local_contributions = _run_phase_a_stage(
            artifact_dir,
            "reference",
            lambda: _run_unsynchronized_reference(bundle, batch),
        )
        expected_keys = _run_phase_a_stage(
            artifact_dir,
            "key_validation",
            lambda: _expected_semantic_bank_keys(
                bundle.extras["multi_lora_training_state"]
            ),
        )

        def validate_local_keys() -> None:
            assert set(local_contributions) == expected_keys

        _run_phase_a_stage(artifact_dir, "key_validation", validate_local_keys)
        oracle_grads, normalization_by_key = _run_phase_a_stage(
            artifact_dir,
            "oracle_collective",
            lambda: _bank_sync_absolute_oracle(
                bundle,
                local_contributions,
                preflight_done=lambda: _write_phase_a_flight_record(
                    artifact_dir, preflight_done=True
                ),
            ),
        )

        def validate_oracle() -> None:
            assert set(oracle_grads) == expected_keys
            assert all(gradient.abs().max() > 0 for gradient in oracle_grads.values())

        _run_phase_a_stage(artifact_dir, "oracle_validation", validate_oracle)
        _write_phase_a_flight_record(artifact_dir, oracle_done=True)
        gathered_contributions = [None] * dist.get_world_size(
            bundle.parallel_state.dp_group
        )
        dist.all_gather_object(
            gathered_contributions,
            local_contributions,
            group=bundle.parallel_state.dp_group,
        )
        assert all(
            any(
                contribution[name].abs().max() > 0
                for contribution in gathered_contributions
            )
            for name in expected_keys
        ), "each bank must receive a contribution from at least one EP rank"
        assert any(
            not torch.equal(
                gathered_contributions[0][name], gathered_contributions[1][name]
            )
            for name in expected_keys
        ), "rank-specific DP batches must produce distinct local contributions"
        losses = [None] * phase_config["world"]
        batches = [None] * phase_config["world"]
        oracle_tensors_by_rank = [None] * phase_config["world"]
        local_contribution_tensors_by_rank = [None] * phase_config["world"]
        parameter_semantics_by_rank = [None] * phase_config["world"]
        adapter_export_records_by_rank = [None] * phase_config["world"]
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
        dist.all_gather_object(parameter_semantics_by_rank, parameter_semantics)
        dist.all_gather_object(
            adapter_export_records_by_rank,
            {
                adapter: {key: _tensor_record(value) for key, value in tensors.items()}
                for adapter, tensors in adapter_exports.items()
            },
        )
        dist.barrier()
        oracle_path = artifact_dir / f"ep2_bank_grads_rank_{dist.get_rank():05d}.pt"
        local_path = (
            artifact_dir / f"ep2_local_bank_grads_rank_{dist.get_rank():05d}.pt"
        )
        torch.save(oracle_grads, oracle_path)
        torch.save(local_contributions, local_path)
        dist.barrier()
        if dist.get_rank() == 0:
            manifest = {
                "schema_version": 3,
                "candidate": {
                    "commit": candidate_sha,
                    "tree": candidate_tree,
                    "diff": candidate_diff,
                },
                "test_sha256": _sha256(Path(__file__)),
                "model_config": _config().to_dict(),
                "topology": {"oracle": oracle_topologies, "verify": None},
                "impl_config": {
                    "oracle": _impl_contract(
                        tp=phase_config["tp"], ep=phase_config["ep"]
                    ),
                    "verify": _impl_contract(
                        tp=phase_config["tp"], ep=phase_config["ep"]
                    ),
                },
                "seeds": {"model": 3100, "batch_base": 4100},
                "batch_by_rank": batches,
                "loss_by_rank": losses,
                "loss_normalization": "model token-loss mean",
                "router_contract": {
                    "oracle_and_verify": "balanced_global_experts_0_2",
                    "production_smoke": "expert0_only",
                },
                "dense_dp_gradient_scaling_by_key": normalization_by_key,
                "semantic_bank_keys": sorted(expected_keys),
                "semantic_bank_tensors_by_rank": oracle_tensors_by_rank,
                "local_contribution_tensors_by_rank": local_contribution_tensors_by_rank,
                "parameter_semantics_by_rank": parameter_semantics_by_rank,
                "adapter_export_tensors_by_rank": adapter_export_records_by_rank,
                "adapter_export_files": {
                    f"rank_{rank:05d}": _sha256(
                        artifact_dir / f"adapter_exports_rank_{rank:05d}.pt"
                    )
                    for rank in range(phase_config["world"])
                },
                "oracle_files": {
                    f"rank_{rank:05d}": _sha256(
                        artifact_dir / f"ep2_bank_grads_rank_{rank:05d}.pt"
                    )
                    for rank in range(phase_config["world"])
                },
                "local_contribution_files": {
                    f"rank_{rank:05d}": _sha256(
                        artifact_dir / f"ep2_local_bank_grads_rank_{rank:05d}.pt"
                    )
                    for rank in range(phase_config["world"])
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
                "phase": "ep2_oracle",
            }
            temporary_complete = complete_path.with_name(f".COMPLETE.{os.getpid()}.tmp")
            temporary_complete.write_text(json.dumps(complete_payload, sort_keys=True))
            os.replace(temporary_complete, complete_path)
        dist.barrier()
        _write_phase_a_flight_record(artifact_dir, phase_a_complete=True)
        return

    complete = json.loads(complete_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert complete == {
        "commit": candidate_sha,
        "manifest_sha256": _sha256(manifest_path),
        "phase": "ep2_oracle",
    }
    assert manifest["candidate"] == {
        "commit": candidate_sha,
        "tree": candidate_tree,
        "diff": candidate_diff,
    }
    assert manifest["test_sha256"] == _sha256(Path(__file__))
    assert manifest["schema_version"] == 3
    assert manifest["seeds"] == {"model": 3100, "batch_base": 4100}
    assert manifest["loss_normalization"] == "model token-loss mean"
    assert manifest["router_contract"] == {
        "oracle_and_verify": "balanced_global_experts_0_2",
        "production_smoke": "expert0_only",
    }
    assert set(manifest["dense_dp_gradient_scaling_by_key"]) == set(
        manifest["semantic_bank_keys"]
    )
    assert manifest["model_config"] == _config().to_dict()
    expected_impl = _impl_contract(tp=phase_config["tp"], ep=phase_config["ep"])
    assert manifest["impl_config"] == {"oracle": expected_impl, "verify": expected_impl}
    assert (
        manifest["topology"]["oracle"][dist.get_rank()]["world"]
        == phase_config["world"]
    )
    assert manifest["topology"]["oracle"][dist.get_rank()]["tp"] == phase_config["tp"]
    assert manifest["topology"]["oracle"][dist.get_rank()]["ep"] == phase_config["ep"]
    assert manifest["topology"]["verify"] is None
    local_contributions_by_rank = []
    for rank in range(phase_config["world"]):
        path = artifact_dir / f"ep2_bank_grads_rank_{rank:05d}.pt"
        assert _sha256(path) == manifest["oracle_files"][f"rank_{rank:05d}"]
        local_path = artifact_dir / f"ep2_local_bank_grads_rank_{rank:05d}.pt"
        assert (
            _sha256(local_path)
            == manifest["local_contribution_files"][f"rank_{rank:05d}"]
        )
    oracle_path = artifact_dir / f"ep2_bank_grads_rank_{dist.get_rank():05d}.pt"
    oracle_grads = torch.load(oracle_path, map_location="cpu", weights_only=True)
    assert {
        name: _tensor_record(value) for name, value in oracle_grads.items()
    } == manifest["semantic_bank_tensors_by_rank"][dist.get_rank()]
    for rank in range(phase_config["world"]):
        local_path = artifact_dir / f"ep2_local_bank_grads_rank_{rank:05d}.pt"
        local_contributions = torch.load(
            local_path, map_location="cpu", weights_only=True
        )
        assert set(local_contributions) == set(manifest["semantic_bank_keys"])
        assert {
            name: _tensor_record(value) for name, value in local_contributions.items()
        } == manifest["local_contribution_tensors_by_rank"][rank]
        local_contributions_by_rank.append(local_contributions)
    adapter_export_path = (
        artifact_dir / f"adapter_exports_rank_{dist.get_rank():05d}.pt"
    )
    assert (
        _sha256(adapter_export_path)
        == manifest["adapter_export_files"][f"rank_{dist.get_rank():05d}"]
    )
    adapter_export_oracle = torch.load(
        adapter_export_path, map_location="cpu", weights_only=True
    )
    assert {
        adapter: {key: _tensor_record(value) for key, value in tensors.items()}
        for adapter, tensors in adapter_export_oracle.items()
    } == manifest["adapter_export_tensors_by_rank"][dist.get_rank()]

    bundle = _build_bundle(
        tp=phase_config["tp"], ep=phase_config["ep"], model_seed=3100
    )
    verify_topology = _actual_topology(bundle)
    assert verify_topology == manifest["topology"]["oracle"][dist.get_rank()]
    state = bundle.extras["multi_lora_training_state"]
    assert state is not None
    assert bundle.extras["optimizer_backend"] == "dist_opt"
    assert callable(bundle.finalize_grads)
    batch = _production_batch(bundle)
    expected_batch = manifest["batch_by_rank"][dist.get_rank()]
    assert _batch_record(batch) == expected_batch
    for name, parameter in state.named_parameters():
        _assert_model_owned_bank_allreduce(
            parameter, is_fc=_bank_surface_kind(bundle, parameter) == "fc"
        )
    expected_keys = _expected_semantic_bank_keys(state)
    assert set(oracle_grads) == expected_keys == set(manifest["semantic_bank_keys"])
    _initialize_nonzero_banks(bundle)
    assert (
        _checkpoint_semantics(bundle)
        == manifest["parameter_semantics_by_rank"][dist.get_rank()]
    )

    # FC sidecars are EP-replicated expert parameters; attention remains on
    # its TP carrier and is dense-DP owned independently of this flag.
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
    assert sidecar.requires_explicit_ep_sync is True
    assert model._sidecar_ep_sync_group(bundle.parallel_state, sidecar) is not None

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
    for category in ("model", "dense", "experts", "banks"):
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
    assert (
        post_load_semantics == manifest["parameter_semantics_by_rank"][dist.get_rank()]
    )
    assert _batch_record(batch) == expected_batch
    restored_exports = _adapter_export_oracle(bundle)
    for adapter, tensors in restored_exports.items():
        assert set(tensors) == set(adapter_export_oracle[adapter])
        for key, value in tensors.items():
            assert torch.equal(value.cpu(), adapter_export_oracle[adapter][key])
    assert _config().to_dict() == manifest["model_config"]
    assert expected_impl == manifest["impl_config"]["verify"]
    assert (
        manifest["router_contract"]["oracle_and_verify"]
        == "balanced_global_experts_0_2"
    )
    _configure_phase_a_balanced_router(bundle)
    _loss, production_grads = _run_production_forward_and_finalize(bundle, batch)
    assert set(production_grads) == expected_keys
    for name, oracle in oracle_grads.items():
        assert (
            _tensor_record(oracle)
            == manifest["semantic_bank_tensors_by_rank"][dist.get_rank()][name]
        )
        torch.testing.assert_close(production_grads[name].cpu(), oracle, rtol=0, atol=0)

    # Production owns FC synchronization explicitly.  Disabling it is the
    # negative control: FC must diverge while TP attention remains unchanged.
    _clear_gradients(bundle)
    original_sidecar = protocol.MoELoraSidecar

    def disable_explicit_sync(*args, **kwargs):
        kwargs["requires_explicit_ep_sync"] = False
        return original_sidecar(*args, **kwargs)

    protocol.MoELoraSidecar = disable_explicit_sync
    try:
        _loss, unsynced_grads = _run_production_forward_and_finalize(bundle, batch)
    finally:
        protocol.MoELoraSidecar = original_sidecar
    assert set(unsynced_grads) == expected_keys
    for name, oracle in oracle_grads.items():
        if _bank_surface_kind(bundle, _bank_parameters(bundle)[name]) == "fc":
            group_ranks = tuple(
                dist.get_process_group_ranks(bundle.parallel_state.ep_dp_group)
            )
            expected = (
                sum(
                    (
                        local_contributions_by_rank[rank][name].float()
                        for rank in group_ranks
                    )
                ).float()
                / bundle.parallel_state.expert_dp_size
            )
            torch.testing.assert_close(
                unsynced_grads[name].cpu(), expected, rtol=0, atol=0
            )
            assert not torch.equal(expected, oracle), name
        else:
            torch.testing.assert_close(
                unsynced_grads[name].cpu(), oracle, rtol=0, atol=0
            )
    for name, param in state.named_parameters():
        assert name.startswith("bank_") and "." not in name
    verify_topologies = [None] * phase_config["world"]
    dist.all_gather_object(verify_topologies, verify_topology)
    if dist.get_rank() == 0:
        receipt = {
            "commit": candidate_sha,
            "manifest_sha256": _sha256(manifest_path),
            "phase": "ep2_verify",
            "topology_by_rank": verify_topologies,
            "checkpoint_files": checkpoint_files,
        }
        temporary_receipt = phase_b_receipt_path.with_name(
            f".EP2_COMPLETE.{os.getpid()}.tmp"
        )
        temporary_receipt.write_text(json.dumps(receipt, sort_keys=True))
        os.replace(temporary_receipt, phase_b_receipt_path)
    dist.barrier()
