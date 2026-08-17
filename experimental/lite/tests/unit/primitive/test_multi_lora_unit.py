# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for the independent dense multi-LoRA primitive."""

# isort: off
import importlib
import sys
import types
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import multi_lora_reference
import pytest
import torch
import torch.nn as nn
from megatron.lite.model.qwen3_moe.lite import multi_lora
from megatron.lite.model.qwen3_moe.lite.checkpoint import (
    PLACEMENT_FN,
    Qwen3MoEWeightSpec,
)
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.ckpt.hf_weights import (
    VLLM_LORA_NAME_PREFIX,
    export_hf_lora_bank_adapter,
)
from megatron.lite.primitive.ckpt.identity import (
    model_checkpoint_identity_metadata,
    require_checkpoint_identity_match,
)
from megatron.lite.primitive.ckpt import dcp
from megatron.lite.primitive.ckpt import hf_weights
from megatron.lite.primitive.modules import multi_lora_kernel
from megatron.lite.primitive.modules import multi_lora_bank
from megatron.lite.primitive.modules.lora import LoraSpec
from megatron.lite.primitive.modules.multi_lora import BatchedLoraDelta
from megatron.lite.primitive.modules.multi_lora_bank import (
    DenseLoraBank,
    LoraBankPartition,
    MultiLoraSpec,
    MultiLoraTrainingState,
    NamedLoraBankRegistry,
    apply_batched_lora_delta,
    validate_multi_lora_parallel_support,
)
from torch.distributed.tensor import Shard

# isort: on


def _inputs(dtype=torch.float64):
    x = torch.tensor(
        [[1.0, 2.0], [-1.0, 3.0], [2.0, -2.0]], dtype=dtype, requires_grad=True
    )
    a_bank = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 1.0], [-1.0, 1.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    b_bank = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 1.0], [2.0, 0.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    indices = torch.tensor([0, 1, 1])
    return x, a_bank, b_bank, indices


def test_bgmv_mixed_precision_promotion_covers_all_scratch_dot_boundaries():
    """FP32 scratch must not leave an unpromoted BF16 ``tl.dot`` boundary."""
    source = Path(multi_lora_kernel.multi_lora_bgmv.__file__).read_text()

    assert source.count("_promote_mixed_dot_operands(") == 5
    for kernel_name in (
        "_bgmv_shrink_kernel",
        "_bgmv_expand_kernel",
        "_bgmv_shrink_transpose_kernel",
        "_bgmv_fused_fwd_kernel",
    ):
        kernel_source = source.split(f"def {kernel_name}", maxsplit=1)[1].split(
            "@triton.autotune", maxsplit=1
        )[0]
        assert "_promote_mixed_dot_operands(" in kernel_source
    fused_source = source.split("def _bgmv_fused_fwd_kernel", maxsplit=1)[1].split(
        "@triton.jit", maxsplit=1
    )[0]
    assert "_promote_mixed_dot_operands(hidden, b)" in fused_source


def _bf16_ground_truth_result(name, actual, ground_truth, eager):
    """Allow at most one final-output ULP from the FP32 mathematical oracle."""
    actual32 = actual.float()
    eager32 = eager.float()
    ground_bf16 = ground_truth.detach().to(torch.bfloat16)
    actual_bf16 = actual.detach().to(torch.bfloat16)
    lower = torch.nextafter(ground_bf16, torch.full_like(ground_bf16, -float("inf")))
    upper = torch.nextafter(ground_bf16, torch.full_like(ground_bf16, float("inf")))
    # One output ULP means exactly one adjacent representable BF16 value in the
    # direction of the actual output.  Do not replace the subnormal spacing at
    # zero with ``finfo.tiny`` (the smallest *normal* value).
    direction_step = torch.where(
        actual_bf16 >= ground_bf16,
        upper.float() - ground_bf16.float(),
        ground_bf16.float() - lower.float(),
    )
    ground_ulp_error = (
        actual_bf16.float() - ground_bf16.float()
    ).abs() / direction_step
    ground_p99_ulp = torch.quantile(ground_ulp_error, 0.99)
    ground_max_ulp = ground_ulp_error.max()
    eager_p99 = torch.quantile((actual32 - eager32).abs(), 0.99)
    report = (
        f"{name}: fp32_ground_truth_max_ulp={ground_max_ulp.item():.9g} "
        f"p99_ulp={ground_p99_ulp.item():.9g}; "
        f"eager_bf16_p99_diagnostic={eager_p99.item():.9g}"
    )
    return report, bool(((actual_bf16 >= lower) & (actual_bf16 <= upper)).all())


@pytest.mark.parametrize("reference", [2.0, -2.0, 0.5, -0.5, 0.0])
def test_bf16_output_ulp_gate_accepts_one_adjacent_value_and_rejects_two(reference):
    """Cover normal, signed power boundaries, and zero/subnormal ULP spacing."""
    ground_truth = torch.tensor([reference], dtype=torch.bfloat16)
    eager = ground_truth.clone()
    for direction in (-float("inf"), float("inf")):
        one_ulp = torch.nextafter(
            ground_truth, torch.full_like(ground_truth, direction)
        )
        two_ulp = torch.nextafter(one_ulp, torch.full_like(one_ulp, direction))
        _, one_passes = _bf16_ground_truth_result(
            "one_ulp", one_ulp, ground_truth, eager
        )
        _, two_passes = _bf16_ground_truth_result(
            "two_ulp", two_ulp, ground_truth, eager
        )
        assert one_passes
        assert not two_passes


def test_batched_lora_delta_matches_dense_reference_for_sorted_slots():
    x, a_bank, b_bank, indices = _inputs()

    actual = BatchedLoraDelta.apply(x, a_bank, b_bank, indices, 0.25)
    expected = multi_lora_reference.dense_lora_delta_reference(
        x, a_bank, b_bank, indices, 0.25
    )

    torch.testing.assert_close(actual, expected)


def test_batched_lora_linear_stage_cpu_oracle_forward_backward_and_empty():
    torch.manual_seed(9)
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, 2, 3, dtype=torch.float64, requires_grad=True)
    slots = torch.tensor([0, 0, 2, 2], dtype=torch.int64)
    actual = multi_lora_kernel.batched_lora_linear_stage(x, weight, slots, scale=0.75)
    actual.square().sum().backward()
    actual_grads = (x.grad.detach().clone(), weight.grad.detach().clone())
    ref_x = x.detach().clone().requires_grad_()
    ref_w = weight.detach().clone().requires_grad_()
    reference = (
        torch.stack([ref_w[slot] @ row for row, slot in zip(ref_x, slots)]) * 0.75
    )
    reference.square().sum().backward()
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_grads[0], ref_x.grad)
    torch.testing.assert_close(actual_grads[1], ref_w.grad)
    assert actual_grads[1][1].eq(0).all()
    with pytest.raises(ValueError, match="slots sorted"):
        multi_lora_kernel.batched_lora_linear_stage(
            x.detach(), weight.detach(), torch.tensor([2, 0, 2, 0]), scale=0.75
        )
    empty = multi_lora_kernel.batched_lora_linear_stage(
        torch.empty(0, 3, dtype=torch.bfloat16),
        torch.ones(3, 2, 3, dtype=torch.bfloat16),
        torch.empty(0, dtype=torch.int64),
        output_dtype=torch.bfloat16,
        max_g_size_hint=4,
    )
    assert empty.shape == (0, 2) and empty.dtype is torch.bfloat16


def test_attention_bank_delta_matches_dense_oracle_and_only_updates_selected_slots():
    """The generic attention carrier keeps per-token slots and zero banks inert."""
    x = torch.tensor([[1.0, 2.0], [3.0, -1.0], [2.0, 4.0]], requires_grad=True)
    a_bank = nn.Parameter(torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]], [[3.0, 1.0]]]))
    b_bank = nn.Parameter(torch.tensor([[[2.0]], [[-1.0]], [[0.0]]]))
    slots = torch.tensor([1, 0, 1], dtype=torch.int64)
    actual = apply_batched_lora_delta(
        DenseLoraBank(a_bank, b_bank), x, slots, scale=0.5
    )
    expected = torch.stack(
        [0.5 * (b_bank[slot] @ a_bank[slot] @ row) for row, slot in zip(x, slots)]
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert a_bank.grad[2].eq(0).all() and b_bank.grad[2].eq(0).all()

    zero_bank = DenseLoraBank(
        nn.Parameter(torch.ones(1, 1, 2)), nn.Parameter(torch.zeros(1, 1, 1))
    )
    assert (
        apply_batched_lora_delta(
            zero_bank, x.detach(), torch.zeros(3, dtype=torch.int64), scale=1.0
        )
        .eq(0)
        .all()
    )


def test_tp_attention_builder_uses_linear_lora_partition_shapes(monkeypatch):
    """TP2 attention banks must have the same local A/B layouts as LinearLoRA."""
    fake_model = types.ModuleType("megatron.lite.model.qwen3_moe.lite.model")
    fake_model.MTPLossAutoScaler = type("MTPLossAutoScaler", (), {})
    fake_model.Qwen3MoEModel = nn.Module
    monkeypatch.setitem(
        sys.modules, "megatron.lite.model.qwen3_moe.lite.model", fake_model
    )
    protocol = importlib.import_module("megatron.lite.model.qwen3_moe.lite.protocol")

    class FakeLayer:
        layer_idx = 0
        attn = SimpleNamespace(
            qkv=SimpleNamespace(
                local_out=4, linear=SimpleNamespace(), use_sp=True, tp_size=2
            ),
            proj=SimpleNamespace(local_in=2, use_sp=True),
        )

    class FakeChunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.layers = [FakeLayer()]

    config = Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        vocab_size=8,
        num_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        layer_types=["full_attention"],
    )
    state = protocol._build_multi_lora_training_state(
        [FakeChunk()], config, MultiLoraSpec(names=("alpha", "bravo"), rank=4)
    )
    qkv, proj = state.attention_banks_for_layer(0)
    # QKV: A rank is TP-local, B keeps the local column-parallel output.
    assert qkv.a_bank.shape == (2, 2, 4)
    assert qkv.b_bank.shape == (2, 4, 4)
    # O-proj: A sees row-parallel local input, B is output-partitioned.
    assert proj.a_bank.shape == (2, 4, 2)
    assert proj.b_bank.shape == (2, 2, 4)
    # Registry-owned metadata must drive both optimizer placement and export
    # materialization; no second names/slot authority is allowed.
    assert qkv.partition.rank_partitioned_a is True
    assert proj.partition.output_partitioned_b is True
    # Optimizer metadata follows the explicit bank partition, not parameter
    # dimensionality: attention factors are TP shards, dense expert banks are
    # replicas and therefore need the data-parallel all-reduce hook.
    fc1, fc2 = state.banks_for_layer(0)
    for bank in (qkv, proj):
        for parameter in (bank.a_bank, bank.b_bank):
            assert parameter.tensor_model_parallel is True
            assert parameter.allreduce is True
    for bank in (fc1, fc2):
        for parameter in (bank.a_bank, bank.b_bank):
            assert parameter.tensor_model_parallel is False
            assert parameter.allreduce is False


def test_tp_attention_builder_rejects_rank_not_divisible_by_tp():
    with pytest.raises(ValueError, match="rank.*divisible.*TP"):
        validate_multi_lora_parallel_support(
            MultiLoraSpec(names=("alpha",), rank=3),
            tp_size=2,
            etp_size=1,
            use_deepep=False,
        )


def test_attention_bank_checkpoint_placement_never_shards_slot_dimension():
    qkv_a = MultiLoraTrainingState.parameter_name(
        "layers.0.attn.qkv.linear.weight", "a"
    )
    qkv_b = MultiLoraTrainingState.parameter_name(
        "layers.0.attn.qkv.linear.weight", "b"
    )
    proj_a = MultiLoraTrainingState.parameter_name(
        "layers.0.attn.proj.linear.weight", "a"
    )
    proj_b = MultiLoraTrainingState.parameter_name(
        "layers.0.attn.proj.linear.weight", "b"
    )
    # [slot, rank, input/output]: TP must use rank/output/input axes only.
    assert PLACEMENT_FN(qkv_a)[-1] == Shard(1)
    assert PLACEMENT_FN(qkv_b)[-1] == Shard(1)
    assert PLACEMENT_FN(proj_a)[-1] == Shard(2)
    assert PLACEMENT_FN(proj_b)[-1] == Shard(1)


def test_qkv_tp_carrier_collective_trace_has_hidden_rank_gather(monkeypatch):
    calls = []
    monkeypatch.setattr(
        multi_lora_bank,
        "_gather_sequence_parallel",
        lambda value, group: (
            calls.append(("sp_gather", value.shape)) or torch.cat((value, value), dim=0)
        ),
    )
    monkeypatch.setattr(
        multi_lora_bank,
        "_all_gather_last_dim",
        lambda value, group, reduce_backward: (
            calls.append(("rank_gather", value.shape, reduce_backward))
            or torch.cat((value, value), dim=-1)
        ),
        raising=False,
    )

    monkeypatch.setattr(
        multi_lora_kernel,
        "batched_lora_linear_stage",
        lambda value, weight, slots, **kwargs: (
            calls.append(("A" if weight.shape[1] == 2 else "B", value.shape))
            or value.new_zeros((value.shape[0], weight.shape[1]))
        ),
    )
    bank = DenseLoraBank(
        torch.ones(2, 2, 4),
        torch.ones(2, 4, 4),
        LoraBankPartition(tp_size=2, rank_partitioned_a=True),
    )
    result = apply_batched_lora_delta(
        bank,
        torch.ones(2, 1, 4),
        torch.tensor([0, 1], dtype=torch.int64),
        scale=1.0,
        tp_group=object(),
        sequence_parallel_input=True,
    )
    assert calls == [
        ("sp_gather", (2, 4)),
        ("sp_gather", (2, 1)),
        ("A", (4, 4)),
        ("rank_gather", (4, 2), True),
        ("B", (4, 4)),
    ]
    # The mocked SP all-gather doubles token rows at TP2, so the post-
    # collective output must not be reshaped with the pre-gather row count.
    assert result.shape == (4, 1, 4)


def test_qkv_sp_accepts_global_slots_without_second_slot_gather(monkeypatch):
    """Production SP passes local rows but logical-global adapter slots."""
    calls = []

    def gather(value, _group):
        calls.append(tuple(value.shape))
        return torch.cat((value, value + 1), dim=0)

    monkeypatch.setattr(multi_lora_bank, "_gather_sequence_parallel", gather)
    monkeypatch.setattr(
        multi_lora_bank,
        "_all_gather_last_dim",
        lambda value, _group, reduce_backward: torch.cat((value, value), dim=-1),
    )
    bank = DenseLoraBank(
        torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 4) / 16,
        torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4) / 16,
        LoraBankPartition(tp_size=2, rank_partitioned_a=True),
    )
    local_rows = torch.randn(2, 1, 4, requires_grad=True)
    global_slots = torch.tensor([0, 1, 0, 1])
    actual = apply_batched_lora_delta(
        bank,
        local_rows,
        global_slots,
        scale=0.5,
        tp_group=object(),
        sequence_parallel_input=True,
    )
    explicit_rows = torch.cat(
        (local_rows.detach(), local_rows.detach() + 1), dim=0
    ).requires_grad_()
    expected = apply_batched_lora_delta(
        bank, explicit_rows, global_slots, scale=0.5, tp_group=object()
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        local_rows.grad, explicit_rows.grad[:2] + explicit_rows.grad[2:]
    )
    assert calls == [(2, 4)]


def test_qkv_sp_rejects_slots_neither_local_nor_global(monkeypatch):
    monkeypatch.setattr(
        multi_lora_bank,
        "_gather_sequence_parallel",
        lambda value, _group: torch.cat((value, value), dim=0),
    )
    bank = DenseLoraBank(
        torch.ones(2, 2, 4),
        torch.ones(2, 4, 4),
        LoraBankPartition(tp_size=2, rank_partitioned_a=True),
    )
    with pytest.raises(ValueError, match="SP LoRA indices"):
        apply_batched_lora_delta(
            bank,
            torch.ones(2, 1, 4),
            torch.tensor([0, 1, 0]),
            scale=1.0,
            tp_group=object(),
            sequence_parallel_input=True,
        )


def test_proj_tp_carrier_collective_trace_has_local_b_then_output_gather(monkeypatch):
    calls = []
    monkeypatch.setattr(
        multi_lora_bank,
        "_all_reduce_sum",
        lambda value, group: calls.append(("rank_reduce", value.shape)) or value,
    )
    monkeypatch.setattr(
        multi_lora_bank,
        "_all_gather_last_dim",
        lambda value, group, reduce_backward: (
            calls.append(("output_gather", value.shape, reduce_backward))
            or torch.cat((value, value), dim=-1)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        multi_lora_bank,
        "_scatter_sequence_parallel",
        lambda value, group, rank: (
            calls.append(("sp_scatter", value.shape, rank)) or value[rank::2]
        ),
    )

    monkeypatch.setattr(
        multi_lora_kernel,
        "batched_lora_linear_stage",
        lambda value, weight, slots, **kwargs: (
            calls.append(("A" if weight.shape[1] == 4 else "B", value.shape))
            or value.new_zeros((value.shape[0], weight.shape[1]))
        ),
    )
    bank = DenseLoraBank(
        torch.ones(2, 4, 2),
        torch.ones(2, 2, 4),
        LoraBankPartition(tp_size=2, output_partitioned_b=True),
    )
    result = apply_batched_lora_delta(
        bank,
        torch.ones(2, 1, 2),
        torch.tensor([0, 1], dtype=torch.int64),
        scale=1.0,
        tp_group=object(),
        tp_rank=1,
        input_parallel_reduce=True,
        sequence_parallel_scatter_output=True,
    )
    assert calls == [
        ("A", (2, 2)),
        ("rank_reduce", (2, 4)),
        ("B", (2, 4)),
        # LinearLoRA.forward uses the default reduce_backward=False here: the
        # hidden gradient already crossed the preceding all-reduce.
        ("output_gather", (2, 2), False),
        ("sp_scatter", (2, 4), 1),
    ]
    # The SP scatter returns one rank's half of the rows; reshape follows its
    # actual result rather than the two input rows.
    assert result.shape == (1, 1, 4)


def test_tp_named_export_materializes_internal_partitions_before_base_gathers(
    monkeypatch,
):
    """Adapter-internal gathers precede native QKV-B/O-proj-A placement gathers."""

    class PartitionedBank:
        def __init__(self, a_bank, b_bank, partition):
            self.a_bank = a_bank
            self.b_bank = b_bank
            self.partition = partition

        @property
        def slots(self):
            return self.a_bank.shape[0]

    qkv_surface = "layers.0.attn.qkv.linear.weight"
    proj_surface = "layers.0.attn.proj.linear.weight"
    # Rank 0's local QKV-A and O-proj-B shards use values distinguishable from
    # the mocked rank 1 shard returned by allgather_concat below.
    qkv_a = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    qkv_b = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    proj_a = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    proj_b = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    # This is the red contract for the future partition-aware validator: its
    # current homogeneous-rank guard is itself the missing capability, so keep
    # the export assertion independently reachable.
    monkeypatch.setattr(NamedLoraBankRegistry, "__post_init__", lambda self: None)
    registry = NamedLoraBankRegistry(
        banks={
            qkv_surface: PartitionedBank(
                qkv_a, qkv_b, LoraBankPartition(tp_size=2, rank_partitioned_a=True)
            ),
            proj_surface: PartitionedBank(
                proj_a, proj_b, LoraBankPartition(tp_size=2, output_partitioned_b=True)
            ),
        },
        names={"alpha": 0},
        rank=4,
        alpha=4,
        base_model_identity={},
    )
    calls = []
    materialized = []
    events = []

    def _materialize(value):
        materialized.append(tuple(value.shape))
        events.append(("materialize", tuple(value.shape)))
        return value

    def _gather(value, world_size, group, *, dim):
        calls.append((tuple(value.shape), dim, value.flatten()[0].item()))
        events.append(("gather", tuple(value.shape), dim))
        return torch.cat((value, value + 100), dim=dim)

    monkeypatch.setattr(hf_weights, "allgather_concat", _gather)
    monkeypatch.setattr(hf_weights, "_materialize_dtensor", _materialize)
    config = Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        vocab_size=8,
        num_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        layer_types=["full_attention"],
    )
    exported = registry.export_hf_state(
        "alpha",
        Qwen3MoEWeightSpec(config),
        SimpleNamespace(tp_size=2, tp_group=object(), etp_size=1, etp_group=None),
    )
    assert calls == [
        ((2, 4), 0, 0.0),  # QKV rank-partitioned A
        ((4, 4), 0, 0.0),  # QKV column-parallel B
        ((2, 4), 0, 0.0),  # O-proj output-partitioned B
        ((4, 2), 1, 0.0),  # O-proj row-parallel A
    ]
    # The selected pair is materialized before the metadata-directed gather.
    # The legacy exporter may materialize its plain inputs again, but it must
    # not introduce a second partition gather.
    assert events[:3] == [
        ("materialize", (2, 4)),
        ("materialize", (4, 4)),
        ("gather", (2, 4), 0),
    ]
    proj_internal = events.index(("gather", (2, 4), 0), 3)
    assert events[proj_internal - 2 : proj_internal + 1] == [
        ("materialize", (4, 2)),
        ("materialize", (2, 4)),
        ("gather", (2, 4), 0),
    ]
    assert materialized
    q_prefix = f"{VLLM_LORA_NAME_PREFIX}model.layers.0.self_attn"
    assert exported[f"{q_prefix}.q_proj.lora_A.weight"].shape == (4, 4)
    assert exported[f"{q_prefix}.k_proj.lora_B.weight"].shape == (2, 4)
    assert exported[f"{q_prefix}.v_proj.lora_B.weight"].shape == (2, 4)
    assert exported[f"{q_prefix}.o_proj.lora_A.weight"].shape == (4, 4)
    assert exported[f"{q_prefix}.o_proj.lora_B.weight"].shape == (4, 4)
    torch.testing.assert_close(
        exported[f"{q_prefix}.q_proj.lora_A.weight"],
        torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [100.0, 101.0, 102.0, 103.0],
                [104.0, 105.0, 106.0, 107.0],
            ]
        ),
    )
    # Native fused QKV B is gathered on output rows before native_to_hf splits
    # it; the mocked rank-1 rows therefore appear in K/V rather than Q.
    torch.testing.assert_close(
        exported[f"{q_prefix}.k_proj.lora_B.weight"],
        torch.tensor([[100.0, 101.0, 102.0, 103.0], [104.0, 105.0, 106.0, 107.0]]),
    )
    torch.testing.assert_close(
        exported[f"{q_prefix}.v_proj.lora_B.weight"],
        torch.tensor([[108.0, 109.0, 110.0, 111.0], [112.0, 113.0, 114.0, 115.0]]),
    )
    torch.testing.assert_close(
        exported[f"{q_prefix}.o_proj.lora_A.weight"],
        torch.tensor(
            [
                [0.0, 1.0, 100.0, 101.0],
                [2.0, 3.0, 102.0, 103.0],
                [4.0, 5.0, 104.0, 105.0],
                [6.0, 7.0, 106.0, 107.0],
            ]
        ),
    )
    torch.testing.assert_close(
        exported[f"{q_prefix}.o_proj.lora_B.weight"],
        torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [100.0, 101.0, 102.0, 103.0],
                [104.0, 105.0, 106.0, 107.0],
            ]
        ),
    )


def test_batched_lora_delta_backward_matches_reference_for_repeated_slots():
    x, a_bank, b_bank, _ = _inputs()
    indices = torch.tensor([0, 1, 1])
    grad_out = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.0, 4.0]], dtype=x.dtype)

    actual = BatchedLoraDelta.apply(x, a_bank, b_bank, indices, 0.75)
    actual.backward(grad_out)
    actual_grads = tuple(t.grad.detach().clone() for t in (x, a_bank, b_bank))

    ref_x, ref_a, ref_b, _ = _inputs()
    expected = multi_lora_reference.dense_lora_delta_reference(
        ref_x, ref_a, ref_b, indices, 0.75
    )
    expected.backward(grad_out)

    for actual_grad, reference_tensor in zip(
        actual_grads, (ref_x, ref_a, ref_b), strict=True
    ):
        torch.testing.assert_close(actual_grad, reference_tensor.grad)


def test_triton_eligible_dense_backward_propagates_kernel_failure(monkeypatch):
    """An eligible BGMV backward failure must not be silently rerouted to eager."""
    x, a_bank, b_bank, indices = _inputs(dtype=torch.float32)

    def eager_forward(x, a_bank, b_bank, slots, scale):
        hidden = torch.bmm(a_bank.index_select(0, slots), x.unsqueeze(-1)).squeeze(-1)
        delta = torch.bmm(b_bank.index_select(0, slots), hidden.unsqueeze(-1)).squeeze(-1)
        return delta * scale, hidden

    class SentinelError(RuntimeError):
        pass

    def raise_sentinel(*_args, **_kwargs):
        raise SentinelError("dense backward sentinel")

    monkeypatch.setattr(multi_lora_kernel, "dense_batched_lora_forward", eager_forward)
    monkeypatch.setattr(multi_lora_kernel, "_can_use_triton", lambda *_args: True)
    monkeypatch.setattr(multi_lora_kernel.multi_lora_bgmv, "bgmv_bwd", raise_sentinel)

    with pytest.raises(SentinelError, match="dense backward sentinel"):
        BatchedLoraDelta.apply(x, a_bank, b_bank, indices, 0.75).sum().backward()


def test_triton_eligible_stage_backward_propagates_kernel_failure(monkeypatch):
    """An eligible stage backward failure must not be silently rerouted to eager."""
    x = torch.ones(2, 3, dtype=torch.float32, requires_grad=True)
    weight = torch.ones(2, 4, 3, dtype=torch.float32, requires_grad=True)
    slots = torch.tensor([0, 1], dtype=torch.int64)

    def eager_stage_forward(x, weight, slots, **_kwargs):
        return torch.bmm(weight.index_select(0, slots), x.unsqueeze(-1)).squeeze(-1)

    class SentinelError(RuntimeError):
        pass

    def raise_sentinel(*_args, **_kwargs):
        raise SentinelError("stage backward sentinel")

    monkeypatch.setattr(
        multi_lora_kernel, "dense_batched_lora_stage_forward", eager_stage_forward
    )
    monkeypatch.setattr(multi_lora_kernel, "_can_use_triton", lambda *_args: True)
    monkeypatch.setattr(
        multi_lora_kernel.multi_lora_bgmv, "bgmv_stage_bwd", raise_sentinel
    )

    with pytest.raises(SentinelError, match="stage backward sentinel"):
        multi_lora_kernel.batched_lora_linear_stage(x, weight, slots).sum().backward()


def test_fp32_backward_reference_matches_float64_autograd_truth():
    """Validate FP32-single-cast backward against random float64 autograd truth."""
    torch.manual_seed(23)
    # Values originate as FP32, then become float64 exactly.  Float64 autograd
    # supplies an independent high-precision derivative truth for those exact
    # inputs; it does not duplicate the oracle's FP32 contraction code.
    x = torch.randn(5, 3, dtype=torch.float32).double().requires_grad_(True)
    a_bank = torch.randn(3, 2, 3, dtype=torch.float32).double().requires_grad_(True)
    b_bank = torch.randn(3, 4, 2, dtype=torch.float32).double().requires_grad_(True)
    indices = torch.tensor([0, 0, 1, 2, 2])
    grad_output = torch.randn(5, 4, dtype=torch.float32).double()
    output = multi_lora_reference.dense_lora_delta_reference(
        x, a_bank, b_bank, indices, 0.75
    )
    output.backward(grad_output)
    expected = tuple(value.grad.detach() for value in (x, a_bank, b_bank))
    actual = multi_lora_reference.dense_lora_backward_reference_fp32_single_cast(
        x.detach(), grad_output, a_bank.detach(), b_bank.detach(), indices, 0.75
    )
    for observed, ground_truth in zip(actual, expected, strict=True):
        # The oracle performs contractions and reductions in FP32, so its
        # intermediate rounding is part of the intended implementation.  It
        # then widens that FP32 result back to the caller's float64 dtype.  Check
        # this exact widening property, then compare both sides as FP32 with
        # torch's standard FP32 tolerance rather than fitting a threshold to
        # this particular sample.  The BF16 contract remains a separate final
        # output single-cast check in the CUDA parity test.
        torch.testing.assert_close(observed, observed.float().double(), rtol=0, atol=0)
        torch.testing.assert_close(observed.float(), ground_truth.float())


def test_dense_lora_reference_passes_float64_gradcheck():
    x, a_bank, b_bank, indices = _inputs()
    assert torch.autograd.gradcheck(
        lambda x_, a_, b_: multi_lora_reference.dense_lora_delta_reference(
            x_, a_, b_, indices, 0.75
        ),
        (x, a_bank, b_bank),
    )


def test_batched_lora_delta_passes_gradcheck():
    x, a_bank, b_bank, _ = _inputs()
    indices = torch.tensor([0, 1, 1])

    assert torch.autograd.gradcheck(
        lambda x_, a_, b_: BatchedLoraDelta.apply(x_, a_, b_, indices, 0.5),
        (x, a_bank, b_bank),
    )


def test_dense_lora_bank_delegates_to_operator_without_copying_tensors():
    x, a_bank, b_bank, indices = _inputs(dtype=torch.float32)
    bank = DenseLoraBank(a_bank, b_bank)

    assert bank.a_bank is a_bank
    assert bank.b_bank is b_bank
    torch.testing.assert_close(
        bank.delta(x, indices, scale=2.0),
        BatchedLoraDelta.apply(x, a_bank, b_bank, indices, 2.0),
    )


@pytest.mark.parametrize(
    ("out_features", "rank", "expected"),
    [(128, 32, False), (256, 8, False), (256, 24, False), (256, 32, True)],
)
def test_bgmv_fused_selection_has_one_production_predicate(
    out_features, rank, expected
):
    """The public predicate is exactly the production launch decision."""
    assert multi_lora_kernel.use_fused_bgmv(out_features, rank) is expected


@pytest.mark.skipif(
    not torch.cuda.is_available() or not multi_lora_kernel._TRITON_AVAILABLE,
    reason="requires CUDA Triton BGMV",
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(torch.float32, 2e-3, 2e-3), (torch.bfloat16, 8e-2, 5e-2)],
)
@pytest.mark.parametrize(("out_features", "rank"), [(128, 8), (256, 8), (256, 32)])
def test_cuda_triton_bgmv_training_matches_independent_oracle(
    monkeypatch, out_features, rank, dtype, rtol, atol
):
    """Exercise shrink and production fused launch regimes with full backward."""
    torch.manual_seed(7)
    tokens, slots, in_features = 5, 4, 16
    indices = torch.tensor([0, 0, 1, 2, 2], device="cuda", dtype=torch.int64)
    values = [
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for shape in (
            (tokens, in_features),
            (slots, rank, in_features),
            (slots, out_features, rank),
        )
    ]
    x, a_bank, b_bank = values
    backward_calls, saved_hidden_dtypes, backward_hidden_dtypes = 0, [], []
    internal_scratch_dtypes = []
    original_empty = multi_lora_kernel.multi_lora_bgmv.torch.empty
    original_forward = multi_lora_kernel.multi_lora_bgmv.bgmv_fwd
    original_backward = multi_lora_kernel.multi_lora_bgmv.bgmv_bwd

    def captured_empty(*size, **kwargs):
        if size == (tokens, rank):
            internal_scratch_dtypes.append(kwargs["dtype"])
        return original_empty(*size, **kwargs)

    def captured_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        saved_hidden_dtypes.append(result[1].dtype)
        return result

    def counted_backward(*args, **kwargs):
        nonlocal backward_calls
        backward_calls += 1
        backward_hidden_dtypes.append(kwargs["hidden"].dtype)
        return original_backward(*args, **kwargs)

    monkeypatch.setattr(
        multi_lora_kernel.multi_lora_bgmv.torch, "empty", captured_empty
    )
    monkeypatch.setattr(multi_lora_kernel.multi_lora_bgmv, "bgmv_fwd", captured_forward)
    monkeypatch.setattr(multi_lora_kernel.multi_lora_bgmv, "bgmv_bwd", counted_backward)
    actual = BatchedLoraDelta.apply(x, a_bank, b_bank, indices, 0.75)
    reference_values = [value.detach().clone().requires_grad_(True) for value in values]
    reference = multi_lora_reference.dense_lora_delta_reference(
        *reference_values, indices, 0.75
    )
    if dtype is torch.bfloat16:
        fp32_forward = multi_lora_reference.dense_lora_delta_reference_fp32_single_cast(
            *reference_values, indices, 0.75
        )
        forward_report, forward_passes = _bf16_ground_truth_result(
            "forward_delta", actual, fp32_forward, reference
        )
        assert forward_passes, forward_report
    else:
        torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    grad_out = torch.randn_like(actual)
    actual.backward(grad_out)
    reference.backward(grad_out)
    assert backward_calls == 1
    assert saved_hidden_dtypes == [torch.float32]
    assert backward_hidden_dtypes == [torch.float32]
    assert internal_scratch_dtypes == [torch.float32, torch.float32]
    assert actual.dtype is dtype
    assert [(value.grad.dtype, value.grad.shape) for value in values] == [
        (dtype, x.shape),
        (dtype, a_bank.shape),
        (dtype, b_bank.shape),
    ]
    if dtype is torch.bfloat16:
        fp32_single_cast = (
            multi_lora_reference.dense_lora_backward_reference_fp32_single_cast(
                reference_values[0],
                grad_out,
                reference_values[1],
                reference_values[2],
                indices,
                0.75,
            )
        )
        reports, failures = [], []
    for name, actual_value, reference_value, fp32_value in zip(
        ("grad_input", "grad_A", "grad_B"),
        values,
        reference_values,
        fp32_single_cast if dtype is torch.bfloat16 else (None,) * 3,
        strict=True,
    ):
        if dtype is torch.bfloat16:
            report, passed = _bf16_ground_truth_result(
                name, actual_value.grad, fp32_value, reference_value.grad
            )
            reports.append(report)
            if not passed:
                failures.append(f"{name}: FP32-ground-truth max exceeds one BF16 ULP")
        else:
            torch.testing.assert_close(
                actual_value.grad, reference_value.grad, rtol=rtol, atol=atol
            )
    if dtype is torch.bfloat16:
        # Eager BF16 is intentionally diagnostic-only: its reduction order is
        # backend dependent, while the gate above specifies the math contract.
        print("\n".join(reports))
        assert not failures, "\n".join([*reports, *failures])
    torch.testing.assert_close(a_bank.grad[3], torch.zeros_like(a_bank.grad[3]))
    torch.testing.assert_close(b_bank.grad[3], torch.zeros_like(b_bank.grad[3]))


def _named_registry():
    return NamedLoraBankRegistry(
        banks={
            "model.layers.0.q_proj": DenseLoraBank(
                torch.arange(12, dtype=torch.float32).view(2, 2, 3),
                torch.arange(16, dtype=torch.float32).view(2, 4, 2),
            ),
            "model.layers.0.o_proj": DenseLoraBank(
                torch.arange(12, 24, dtype=torch.float32).view(2, 2, 3),
                torch.arange(16, 32, dtype=torch.float32).view(2, 4, 2),
            ),
        },
        names={"alpha": 0, "bravo": 1},
        rank=2,
        alpha=4,
        base_model_identity={"name": "tiny-qwen", "revision": "abc123"},
        lora_spec=LoraSpec(enabled=True, rank=2, alpha=4, dropout=0.125),
    )


def test_named_registry_export_reuses_hf_mapping_and_scaling_contract():
    """Named multi-LoRA export must take the production HF export route."""
    registry = NamedLoraBankRegistry(
        banks={
            "layers.0.moe.experts.fc1.weight0": DenseLoraBank(
                torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
                torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]),
            )
        },
        names={"alpha": 0},
        rank=2,
        alpha=4,
        base_model_identity={},
        lora_spec=LoraSpec(enabled=True, rank=2, alpha=4, use_rslora=True),
    )

    class Spec:
        num_experts = 1

        @staticmethod
        def tp_spec(name):
            assert name == "layers.0.moe.experts.fc1.weight0"
            return None

        @staticmethod
        def is_expert(name):
            return "experts" in name

        @staticmethod
        def native_to_hf(name, tensor):
            assert name == "layers.0.moe.experts.fc1.weight0"
            gate, up = tensor.chunk(2, dim=0)
            return [
                ("model.layers.0.mlp.experts.0.gate_proj.weight", gate),
                ("model.layers.0.mlp.experts.0.up_proj.weight", up),
            ]

    ps = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        etp_size=1,
        etp_rank=0,
        etp_group=None,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
    )
    state = registry.export_hf_state("alpha", Spec(), ps)
    prefix = VLLM_LORA_NAME_PREFIX + "model.layers.0.mlp.experts.0"
    assert set(state) == {
        f"{prefix}.gate_proj.lora_A.weight",
        f"{prefix}.gate_proj.lora_B.weight",
        f"{prefix}.up_proj.lora_A.weight",
        f"{prefix}.up_proj.lora_B.weight",
    }
    # vLLM FusedMoE uses alpha/rank, while training chose rsLoRA alpha/sqrt(rank).
    torch.testing.assert_close(
        state[f"{prefix}.gate_proj.lora_B.weight"],
        registry.banks["layers.0.moe.experts.fc1.weight0"].b_bank[0, :2] * 2**0.5,
    )


def test_named_registry_export_gathers_tensor_parallel_output_before_mapping(
    monkeypatch,
):
    """A bank's local TP B rows must never be exported as a partial adapter."""
    registry = NamedLoraBankRegistry(
        banks={
            "layers.0.attn.qkv.linear.weight": DenseLoraBank(
                torch.ones(1, 2, 2), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
            )
        },
        names={"alpha": 0},
        rank=2,
        alpha=2,
        base_model_identity={},
    )

    class Spec:
        num_experts = 0
        tp_spec = staticmethod(lambda name: (0, 0))
        is_expert = staticmethod(lambda name: False)
        native_to_hf = staticmethod(
            lambda name, tensor: [("model.layers.0.self_attn.q_proj.weight", tensor)]
        )

    import megatron.lite.primitive.ckpt.hf_weights as hf_weights

    monkeypatch.setattr(
        hf_weights,
        "allgather_concat",
        lambda tensor, world_size, group, dim: torch.cat(
            (tensor, tensor + 10), dim=dim
        ),
    )
    ps = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        tp_group="tp",
        etp_size=1,
        etp_rank=0,
        etp_group=None,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
    )
    state = registry.export_hf_state("alpha", Spec(), ps)
    key = f"{VLLM_LORA_NAME_PREFIX}model.layers.0.self_attn.q_proj.lora_B.weight"
    torch.testing.assert_close(
        state[key], torch.tensor([[1.0, 2.0], [3.0, 4.0], [11.0, 12.0], [13.0, 14.0]])
    )


def test_named_registry_rejects_duplicate_or_unknown_slots():
    bank = DenseLoraBank(torch.ones(2, 1, 1), torch.ones(2, 1, 1))
    with pytest.raises(ValueError, match="duplicate"):
        NamedLoraBankRegistry(
            banks={"q_proj": bank},
            names={"alpha": 0, "bravo": 0},
            rank=1,
            alpha=1,
            base_model_identity={},
        )
    with pytest.raises(KeyError, match="Unknown"):
        _named_registry().slot_for("not-registered")


def test_model_owned_training_state_discovers_optimizer_params_and_builds_sidecars():
    """The production sidecar factory must share registry parameter objects."""
    fc1 = DenseLoraBank(
        torch.nn.Parameter(torch.ones(2, 2, 3)),
        torch.nn.Parameter(torch.zeros(2, 4, 2)),
    )
    fc2 = DenseLoraBank(
        torch.nn.Parameter(torch.ones(2, 2, 2)),
        torch.nn.Parameter(torch.zeros(2, 3, 2)),
    )
    registry = NamedLoraBankRegistry(
        banks={
            "layers.0.moe.experts._fc1_weight_0": fc1,
            "layers.0.moe.experts._fc2_weight_0": fc2,
        },
        names={"alpha": 0, "bravo": 1},
        rank=2,
        alpha=4,
        base_model_identity={},
        lora_spec=LoraSpec(enabled=True, rank=2, alpha=4),
    )
    state = MultiLoraTrainingState(
        registry,
        {
            0: (
                "layers.0.moe.experts._fc1_weight_0",
                "layers.0.moe.experts._fc2_weight_0",
            )
        },
    )

    sidecar = multi_lora.MoELoraSidecar(
        *state.banks_for_layer(0),
        lora_indices=torch.tensor([0, 1], dtype=torch.int64),
        scale=state.scale,
    )

    assert sidecar.fc1 is fc1
    assert sidecar.fc2 is fc2
    assert {id(parameter) for parameter in state.parameters()} == {
        id(fc1.a_bank),
        id(fc1.b_bank),
        id(fc2.a_bank),
        id(fc2.b_bank),
    }
    assert state.registry.slot_for("bravo") == 1
    assert sidecar.requires_explicit_ep_sync is True


def test_model_owned_checkpoint_identity_is_pp_global_and_rejects_contract_changes():
    """Same-shaped banks must not silently restore into reordered adapter names."""

    def build_state(names, *, alpha=2, rank=1, layer_idx=0):
        fc1_surface = f"layers.{layer_idx}.moe.experts._fc1_weight_0"
        fc2_surface = f"layers.{layer_idx}.moe.experts._fc2_weight_0"
        registry = NamedLoraBankRegistry(
            banks={
                fc1_surface: DenseLoraBank(
                    nn.Parameter(torch.ones(2, rank, 1)),
                    nn.Parameter(torch.ones(2, 2, rank)),
                ),
                fc2_surface: DenseLoraBank(
                    nn.Parameter(torch.ones(2, rank, 2)),
                    nn.Parameter(torch.ones(2, 1, rank)),
                ),
            },
            names={name: slot for slot, name in enumerate(names)},
            rank=rank,
            alpha=alpha,
            base_model_identity={},
        )
        return MultiLoraTrainingState(registry, {layer_idx: (fc1_surface, fc2_surface)})

    source = nn.Module()
    source.add_module("multi_lora_training_state", build_state(("alpha", "bravo")))
    pp_stage_one = nn.Module()
    pp_stage_one.add_module(
        "multi_lora_training_state", build_state(("alpha", "bravo"), layer_idx=7)
    )
    restored = nn.Module()
    restored.add_module("multi_lora_training_state", build_state(("bravo", "alpha")))
    saved_identity = model_checkpoint_identity_metadata(source)
    assert saved_identity["chunk0.multi_lora_training_state"]["schema_version"] == 1
    assert saved_identity == model_checkpoint_identity_metadata(pp_stage_one)
    with pytest.raises(ValueError, match="identity mismatch"):
        require_checkpoint_identity_match(restored, saved_identity)
    for changed in (
        build_state(("alpha", "bravo"), alpha=4),
        build_state(("alpha", "bravo"), rank=2),
    ):
        target = nn.Module()
        target.add_module("multi_lora_training_state", changed)
        with pytest.raises(ValueError, match="identity mismatch"):
            require_checkpoint_identity_match(target, saved_identity)


def test_semantic_bank_parameter_names_survive_reordered_registry_and_pp_stages(
    tmp_path,
):
    """Tensor checkpoint keys bind each global native surface, not insertion order."""
    surfaces = (
        "layers.0.moe.experts._fc1_weight_0",
        "layers.0.moe.experts._fc2_weight_0",
        "layers.1.moe.experts._fc1_weight_0",
        "layers.1.moe.experts._fc2_weight_0",
    )

    def make_state(surface_order, *, fill_offset):
        banks = {}
        for index, surface in enumerate(surface_order):
            banks[surface] = DenseLoraBank(
                nn.Parameter(
                    torch.full((1, 1, 1), fill_offset + surfaces.index(surface) * 10.0)
                ),
                nn.Parameter(
                    torch.full(
                        (1, 1, 1), fill_offset + surfaces.index(surface) * 10.0 + 1
                    )
                ),
            )
        registry = NamedLoraBankRegistry(
            banks=banks, names={"alpha": 0}, rank=1, alpha=1, base_model_identity={}
        )
        return MultiLoraTrainingState(
            registry, {0: (surfaces[0], surfaces[1]), 1: (surfaces[2], surfaces[3])}
        )

    source = nn.Module()
    source_state = make_state(tuple(reversed(surfaces)), fill_offset=10.0)
    source.add_module("multi_lora_training_state", source_state)
    target = nn.Module()
    target_state = make_state(surfaces, fill_offset=-100.0)
    target.add_module("multi_lora_training_state", target_state)
    dcp.save_training_checkpoint(
        source,
        torch.optim.SGD(source.parameters(), lr=0.1),
        1,
        str(tmp_path),
        use_dcp=False,
    )
    dcp.load_training_checkpoint(
        target,
        torch.optim.SGD(target.parameters(), lr=0.1),
        str(tmp_path),
        use_dcp=False,
    )
    for surface in surfaces:
        source_bank = source_state.registry.banks[surface]
        target_bank = target_state.registry.banks[surface]
        torch.testing.assert_close(target_bank.a_bank, source_bank.a_bank)
        torch.testing.assert_close(target_bank.b_bank, source_bank.b_bank)

    stage_zero = {
        MultiLoraTrainingState.parameter_name(surfaces[0], factor)
        for factor in ("a", "b")
    }
    stage_one = {
        MultiLoraTrainingState.parameter_name(surfaces[2], factor)
        for factor in ("a", "b")
    }
    assert stage_zero.isdisjoint(stage_one)
    assert all("." not in name for name in stage_zero | stage_one)


def test_model_owned_fc_sidecar_has_explicit_ep_gradient_owner(
    monkeypatch, transformer_engine_import_stub
):
    """Model-owned replicated FC banks retain their explicit EP owner."""
    state = MultiLoraTrainingState(
        NamedLoraBankRegistry(
            banks={
                "layers.0.moe.experts._fc1_weight_0": DenseLoraBank(
                    nn.Parameter(torch.ones(1, 1, 1)), nn.Parameter(torch.ones(1, 2, 1))
                ),
                "layers.0.moe.experts._fc2_weight_0": DenseLoraBank(
                    nn.Parameter(torch.ones(1, 1, 2)), nn.Parameter(torch.ones(1, 1, 1))
                ),
            },
            names={"alpha": 0},
            rank=1,
            alpha=1,
            base_model_identity={},
        ),
        {
            0: (
                "layers.0.moe.experts._fc1_weight_0",
                "layers.0.moe.experts._fc2_weight_0",
            )
        },
    )
    injected = {}

    class Batch:
        extras = {"multi_lora_slots": {0: torch.tensor([0], dtype=torch.int64)}}

    fake_model = types.ModuleType("megatron.lite.model.qwen3_moe.lite.model")
    fake_model.MTPLossAutoScaler = type("MTPLossAutoScaler", (), {})
    fake_model.Qwen3MoEModel = nn.Module
    monkeypatch.setitem(
        sys.modules, "megatron.lite.model.qwen3_moe.lite.model", fake_model
    )
    protocol = importlib.import_module("megatron.lite.model.qwen3_moe.lite.protocol")
    protocol._inject_multi_lora_sidecars(injected, Batch(), state)
    owned = injected["multi_lora_sidecars"][0]
    assert owned.requires_explicit_ep_sync is True

    # Test the real model.py selector, not a hand-passed ``None`` argument.
    transformer_engine_import_stub()
    sys.modules.pop("megatron.lite.model.qwen3_moe.lite.model", None)
    model = importlib.import_module("megatron.lite.model.qwen3_moe.lite.model")
    ps = SimpleNamespace(ep_size=2, ep_group=object())
    assert model._sidecar_ep_sync_group(ps, owned) is ps.ep_group

    # The legacy external contract still selects one explicit EP group for
    # each of its two model call sites (fc1 and fc2).
    group = object()
    external = multi_lora.MoELoraSidecar(
        *state.banks_for_layer(0),
        lora_indices=torch.tensor([0], dtype=torch.int64),
        scale=1.0,
    )
    assert external.requires_explicit_ep_sync is True
    ps.ep_group = group
    assert model._sidecar_ep_sync_group(ps, external) is group
    # Both model call sites consume this same selector; an inverted flag would
    # change the result for both fc1 and fc2 before any kernel is launched.
    assert model._sidecar_ep_sync_group(ps, external) is group


def test_pipeline_stage_filters_remote_slots_and_rejects_missing_local_slot(
    monkeypatch,
):
    """Each PP stage constructs sidecars only for its own global layer IDs."""
    registry = NamedLoraBankRegistry(
        banks={
            "layers.1.moe.experts._fc1_weight_0": DenseLoraBank(
                nn.Parameter(torch.ones(1, 1, 1)), nn.Parameter(torch.ones(1, 2, 1))
            ),
            "layers.1.moe.experts._fc2_weight_0": DenseLoraBank(
                nn.Parameter(torch.ones(1, 1, 2)), nn.Parameter(torch.ones(1, 1, 1))
            ),
        },
        names={"alpha": 0},
        rank=1,
        alpha=1,
        base_model_identity={},
    )
    stage_one = MultiLoraTrainingState(
        registry,
        {
            1: (
                "layers.1.moe.experts._fc1_weight_0",
                "layers.1.moe.experts._fc2_weight_0",
            )
        },
    )
    assert stage_one.local_layer_indices == (1,)
    fake_model = types.ModuleType("megatron.lite.model.qwen3_moe.lite.model")
    fake_model.MTPLossAutoScaler = type("MTPLossAutoScaler", (), {})
    fake_model.Qwen3MoEModel = nn.Module
    monkeypatch.setitem(
        sys.modules, "megatron.lite.model.qwen3_moe.lite.model", fake_model
    )
    protocol = importlib.import_module("megatron.lite.model.qwen3_moe.lite.protocol")

    class Batch:
        extras = {
            "multi_lora_slots": {
                0: torch.tensor([0], dtype=torch.int64),
                1: torch.tensor([0], dtype=torch.int64),
            }
        }

    kwargs = {}
    protocol._inject_multi_lora_sidecars(kwargs, Batch(), stage_one)
    assert set(kwargs["multi_lora_sidecars"]) == {1}

    class MissingLocalBatch:
        extras = {"multi_lora_slots": {0: torch.tensor([0], dtype=torch.int64)}}

    with pytest.raises(ValueError, match=r"missing local pipeline layers: \[1\]"):
        protocol._inject_multi_lora_sidecars({}, MissingLocalBatch(), stage_one)


def test_model_owned_fc2_bank_uses_out_features_by_rank_layout():
    """fc2's B bank is [slots, hidden_size, rank], not transposed."""
    fc2 = DenseLoraBank(
        torch.nn.Parameter(torch.ones(2, 2, 2)),
        torch.nn.Parameter(torch.zeros(2, 3, 2)),
    )
    output = fc2.delta(
        torch.ones(2, 2), torch.tensor([0, 1], dtype=torch.int64), scale=1.0
    )
    assert output.shape == (2, 3)


def test_production_builder_owns_native_banks_and_injects_sidecars(monkeypatch):
    """Builder wiring, not a hand-made bank, owns the train/export lifecycle."""

    fake_model = types.ModuleType("megatron.lite.model.qwen3_moe.lite.model")
    fake_model.MTPLossAutoScaler = type("MTPLossAutoScaler", (), {})
    fake_model.Qwen3MoEModel = nn.Module
    monkeypatch.setitem(
        sys.modules, "megatron.lite.model.qwen3_moe.lite.model", fake_model
    )
    qwen3_moe_protocol = importlib.import_module(
        "megatron.lite.model.qwen3_moe.lite.protocol"
    )

    class FakeLayer:
        layer_idx = 0
        attn = SimpleNamespace(
            qkv=SimpleNamespace(local_out=8, linear=SimpleNamespace()),
            proj=SimpleNamespace(local_in=4),
        )

    class FakeChunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.layers = [FakeLayer()]

    chunk = FakeChunk()
    config = Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        vocab_size=8,
        num_experts=3,
        num_experts_per_tok=1,
        moe_intermediate_size=6,
        layer_types=["full_attention"],
    )
    with pytest.raises(ValueError, match="cannot be enabled together"):
        qwen3_moe_protocol.build_model(
            config,
            impl_cfg=qwen3_moe_protocol.ImplConfig(
                lora=LoraSpec(enabled=True, rank=2),
                multi_lora=MultiLoraSpec(names=("alpha",), rank=2),
            ),
        )
    with pytest.raises(ValueError, match="dist_opt only"):
        qwen3_moe_protocol.build_model(
            config,
            impl_cfg=qwen3_moe_protocol.ImplConfig(
                optimizer="fsdp2", multi_lora=MultiLoraSpec(names=("alpha",), rank=2)
            ),
        )
    state = qwen3_moe_protocol._build_multi_lora_training_state(
        [chunk], config, MultiLoraSpec(names=("alpha", "bravo"), rank=24)
    )
    assert state is chunk.multi_lora_training_state
    fc1, fc2 = state.banks_for_layer(0)
    qkv, proj = state.attention_banks_for_layer(0)
    assert fc1.a_bank.shape == (2, 24, 4)
    assert fc1.b_bank.shape == (2, 12, 24)
    assert fc2.a_bank.shape == (2, 24, 6)
    assert fc2.b_bank.shape == (2, 4, 24)
    assert qkv.a_bank.shape == (2, 24, 4)
    assert qkv.b_bank.shape == (2, 8, 24)
    assert proj.a_bank.shape == (2, 24, 4)
    assert proj.b_bank.shape == (2, 4, 24)
    for bank in (fc1, fc2):
        for parameter in (bank.a_bank, bank.b_bank):
            assert parameter.tensor_model_parallel is False
            assert parameter.allreduce is False
    for bank in (qkv, proj):
        for parameter in (bank.a_bank, bank.b_bank):
            assert parameter.tensor_model_parallel is False
            assert parameter.allreduce is True
    assert set(state.registry.banks) == {
        "layers.0.moe.experts._fc1_weight_0",
        "layers.0.moe.experts._fc2_weight_0",
        "layers.0.attn.qkv.linear.weight",
        "layers.0.attn.proj.linear.weight",
    }
    assert {id(parameter) for parameter in state.parameters()} == {
        id(fc1.a_bank),
        id(fc1.b_bank),
        id(fc2.a_bank),
        id(fc2.b_bank),
        id(state.registry.banks["layers.0.attn.qkv.linear.weight"].a_bank),
        id(state.registry.banks["layers.0.attn.qkv.linear.weight"].b_bank),
        id(state.registry.banks["layers.0.attn.proj.linear.weight"].a_bank),
        id(state.registry.banks["layers.0.attn.proj.linear.weight"].b_bank),
    }
    optimizer_parameter_ids = {
        id(parameter)
        for group in torch.optim.SGD(chunk.parameters(), lr=0.1).param_groups
        for parameter in group["params"]
    }
    assert {
        id(parameter) for parameter in state.parameters()
    } <= optimizer_parameter_ids
    checkpoint_state = chunk.state_dict()
    assert {
        "multi_lora_training_state."
        + MultiLoraTrainingState.parameter_name(surface, factor)
        for surface in (
            "layers.0.moe.experts._fc1_weight_0",
            "layers.0.moe.experts._fc2_weight_0",
            "layers.0.attn.qkv.linear.weight",
            "layers.0.attn.proj.linear.weight",
        )
        for factor in ("a", "b")
    } <= set(checkpoint_state)
    selected = state.registry.select("alpha")
    assert len(selected) == 4
    selected_fc1_a, selected_fc1_b = selected["layers.0.moe.experts._fc1_weight_0"]
    selected_fc2_a, selected_fc2_b = selected["layers.0.moe.experts._fc2_weight_0"]
    torch.testing.assert_close(selected_fc1_a, fc1.a_bank[0])
    torch.testing.assert_close(selected_fc1_b, fc1.b_bank[0])
    torch.testing.assert_close(selected_fc2_a, fc2.a_bank[0])
    torch.testing.assert_close(selected_fc2_b, fc2.b_bank[0])

    ps = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        etp_size=1,
        etp_rank=0,
        etp_group=None,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
    )
    generated = list(
        export_hf_lora_bank_adapter(
            selected,
            spec=Qwen3MoEWeightSpec(config),
            ps=ps,
            train_scale=state.scale,
            rank=state.registry.rank,
            alpha=state.registry.alpha,
            use_rslora=False,
        )
    )
    expected_modules = {
        f"model.layers.0.mlp.experts.{expert_idx}.{projection}"
        for expert_idx in range(config.num_experts)
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
    generated_keys = [key for key, _ in generated]
    assert set(generated_keys) == expected_keys
    assert len(generated_keys) == len(expected_keys) == (config.num_experts * 3 + 4) * 2
    assert len(generated_keys) == len(set(generated_keys))
    assert (
        set(state.registry.export_hf_state("alpha", Qwen3MoEWeightSpec(config), ps))
        == expected_keys
    )

    bundle = ModelBundle(
        chunks=[chunk],
        parallel_state=SimpleNamespace(),
        forward_step=partial(
            qwen3_moe_protocol._forward_step_bshd, multi_lora_state=state
        ),
        extras={"multi_lora_registry": state.registry},
    )
    batch = SimpleNamespace(
        input_ids=torch.tensor([1, 2]),
        labels=None,
        extras={"multi_lora_slots": {0: torch.tensor([0, 1], dtype=torch.int64)}},
    )
    output = bundle.forward_step(lambda **kwargs: kwargs, batch)
    sidecar = output["multi_lora_sidecars"][0]
    assert bundle.extras["multi_lora_registry"] is state.registry
    assert sidecar.fc1 is fc1 and sidecar.fc2 is fc2
    assert sidecar.qkv is qkv and sidecar.proj is proj
    assert sidecar.lora_indices is batch.extras["multi_lora_slots"][0]


def test_runtime_exports_named_adapter_from_model_owned_registry():
    """The runtime must find the registry on the model, not only handle extras."""
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    registry = object()
    captured = {}
    chunk = nn.Module()
    chunk.multi_lora_training_state = SimpleNamespace(registry=registry)

    class Protocol:
        @staticmethod
        def export_hf_lora_adapter(chunks, model_cfg, ps, **kwargs):
            captured.update(chunks=chunks, model_cfg=model_cfg, ps=ps, kwargs=kwargs)
            yield "adapter.weight", torch.ones(1)

    handle = SimpleNamespace(
        _extras={"model_chunks": [chunk], "protocol": Protocol(), "model_cfg": "cfg"},
        _model=chunk,
        _parallel_state="ps",
    )
    runtime = SimpleNamespace(multi_lora_registry=MegatronLiteRuntime.multi_lora_registry)

    exported = list(MegatronLiteRuntime.export_weights(runtime, handle, multi_lora_name="alpha"))
    assert exported[0][0] == "adapter.weight"
    torch.testing.assert_close(exported[0][1], torch.ones(1))
    assert captured["chunks"] == [chunk]
    assert captured["model_cfg"] == "cfg"
    assert captured["ps"] == "ps"
    assert captured["kwargs"] == {
        "multi_lora_registry": registry,
        "multi_lora_name": "alpha",
    }


def test_multi_lora_parallel_contract_allows_tp_and_rejects_etp_and_deepep():
    spec = multi_lora_bank.MultiLoraSpec(names=("alpha",), rank=2)
    with pytest.raises(ValueError, match="ETP"):
        validate_multi_lora_parallel_support(
            spec, tp_size=1, etp_size=2, use_deepep=False
        )
    validate_multi_lora_parallel_support(spec, tp_size=2, etp_size=1, use_deepep=False)
    with pytest.raises(ValueError, match="DeepEP"):
        validate_multi_lora_parallel_support(
            spec, tp_size=1, etp_size=1, use_deepep=True
        )


@pytest.mark.parametrize(
    ("bad_indices", "message"),
    [
        (torch.tensor([0, 2, 1]), "out of range"),
        (torch.tensor([-1, 0, 1]), "out of range"),
        (torch.tensor([1, 0, 1]), "monotonically non-decreasing"),
        (torch.tensor([0, 1, 1], dtype=torch.int32), "torch.int64"),
        (torch.tensor([[0, 1, 0]]), "one-dimensional"),
    ],
)
def test_batched_lora_delta_rejects_non_prototype_indices(bad_indices, message):
    x, a_bank, b_bank, _ = _inputs()

    with pytest.raises((IndexError, ValueError), match=message):
        BatchedLoraDelta.apply(x, a_bank, b_bank, bad_indices, 1.0)
    with pytest.raises((IndexError, ValueError), match=message):
        multi_lora_reference.dense_lora_delta_reference(
            x, a_bank, b_bank, bad_indices, 1.0
        )


def test_qwen_moe_sidecar_sorts_then_restores_unsorted_slots():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    a_bank = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    b_bank = torch.tensor([[[2.0]], [[3.0]]])
    slots = torch.tensor([1, 0, 1], dtype=torch.int64)
    bank = DenseLoraBank(a_bank, b_bank)

    actual = multi_lora.apply_dense_lora_delta(bank, x, slots, scale=1.0)
    expected_rows = [
        b_bank[slot] @ (a_bank[slot] @ row) for row, slot in zip(x, slots, strict=True)
    ]
    expected = torch.stack(expected_rows)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("tp_rank", "expected"), [(0, [0, 1]), (1, [2, 3])])
def test_moe_sp_global_slots_follow_contiguous_scatter_order(
    transformer_engine_import_stub, tp_rank, expected
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite.model import _local_moe_lora_indices

    slots = torch.tensor([0, 1, 2, 3])
    actual = _local_moe_lora_indices(slots, local_rows=2, tp_size=2, tp_rank=tp_rank)
    assert actual.tolist() == expected
    # Local input is already in SP-scatter order and must not be sliced again.
    assert (
        _local_moe_lora_indices(actual, local_rows=2, tp_size=2, tp_rank=tp_rank)
        is actual
    )
    with pytest.raises(ValueError, match="local or TP-global"):
        _local_moe_lora_indices(
            torch.tensor([0, 1, 2]), local_rows=2, tp_size=2, tp_rank=tp_rank
        )


def test_ep_gradient_sync_is_identity_in_forward_and_all_reduces_in_backward(
    monkeypatch,
):
    calls = []
    param = torch.tensor([2.0], requires_grad=True)
    group = object()

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda actual: 2)

    def fake_all_reduce(grad, *, op, group):
        calls.append((grad.clone(), op, group))
        grad.add_(3.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    output = multi_lora._AllReduceGradient.apply(param, group)
    torch.testing.assert_close(output, param)
    output.sum().backward()

    assert len(calls) == 1
    assert calls[0][2] is group
    torch.testing.assert_close(param.grad, torch.tensor([4.0]))


@pytest.mark.parametrize(
    ("etp_size", "use_deepep", "message"), [(2, False, "ETP"), (1, True, "DeepEP")]
)
def test_qwen_moe_sidecar_rejects_unverified_parallel_modes(
    transformer_engine_import_stub, etp_size, use_deepep, message
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite.model import MoELayer

    layer = MoELayer.__new__(MoELayer)
    nn.Module.__init__(layer)
    layer.ps = SimpleNamespace(etp_size=etp_size)
    layer._use_deepep_requested = use_deepep
    sidecar = multi_lora.MoELoraSidecar(
        fc1=DenseLoraBank(torch.ones(1, 1, 2), torch.ones(1, 2, 1)),
        fc2=DenseLoraBank(torch.ones(1, 1, 1), torch.ones(1, 2, 1)),
        lora_indices=torch.zeros(1, dtype=torch.int64),
        scale=1.0,
    )

    with pytest.raises(RuntimeError, match=message):
        layer(torch.ones(1, 2), multi_lora_sidecar=sidecar)
