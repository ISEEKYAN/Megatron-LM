# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engram storage contracts: one trainability switch, resident, never widened.

The released design carries the table as FP8 rows with E8M0 block scales and
selects trainability once, at construction. Both arms must work, the frozen arm
must not allocate a master, and a parent dtype cast must move neither the FP8
storage nor the FP32 master -- casting either one silently changes what the
optimizer updates and what the checkpoint round-trips.
"""

import pytest
import torch
from megatron.lite.primitive.modules.engram_lookup import EngramTable
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8

ROWS, WIDTH = 6, 64


def _table(**kwargs):
    generator = torch.Generator().manual_seed(11)
    source = torch.randn(ROWS, WIDTH, generator=generator)
    weight, scale = quantize_block_fp8(source, (1, 32), scale_format="e8m0")
    return EngramTable(weight, scale, **kwargs), source


def _dequantized(table):
    return table.weight.float() * table.scale.float().repeat_interleave(32, -1)


def test_frozen_arm_holds_no_master():
    table, _ = _table(trainable=False)
    assert table.master is None
    assert [name for name, _ in table.named_parameters()] == []
    assert table.weight.dtype == torch.float8_e4m3fn
    assert table.scale.dtype == torch.float8_e8m0fnu


def test_trainable_arm_adds_an_fp32_master_matching_the_stored_rows():
    table, _ = _table(trainable=True)
    assert isinstance(table.master, torch.nn.Parameter)
    assert table.master.dtype == torch.float32
    assert table.master.requires_grad
    # The master is the dequantized table, not a fresh init: resuming from a
    # frozen checkpoint must not perturb the rows the model already serves.
    assert torch.equal(table.master.detach(), _dequantized(table))


@pytest.mark.parametrize("trainable", [False, True])
def test_parent_bfloat16_cast_widens_neither_storage_nor_master(trainable):
    table, _ = _table(trainable=trainable)
    before = None if table.master is None else table.master.detach().clone()
    table.bfloat16()
    assert table.weight.dtype == torch.float8_e4m3fn, "FP8 storage was widened"
    assert table.scale.dtype == torch.float8_e8m0fnu, "block scales were widened"
    if before is not None:
        assert table.master.dtype == torch.float32, "master was rounded to bf16"
        assert torch.equal(table.master.detach(), before)
    # Only the emitted dtype follows the cast.
    assert table.output_dtype == torch.bfloat16
    assert table(torch.tensor([1])).dtype == torch.bfloat16


def test_refresh_storage_republishes_the_master_and_is_a_no_op_when_frozen():
    table, _ = _table(trainable=True)
    stale = _dequantized(table)
    with torch.no_grad():
        table.master.mul_(0.5)
    table.refresh_storage()
    published = _dequantized(table)
    assert not torch.equal(published, stale)
    # Storage reproduces the master within the FP8 grid it is stored on.
    assert torch.allclose(published, table.master.detach(), rtol=0.15)

    frozen, _ = _table(trainable=False)
    kept = frozen.weight.float().clone()
    frozen.refresh_storage()
    assert torch.equal(frozen.weight.float(), kept)


def test_power_of_two_rescaling_is_absorbed_by_the_e8m0_scale():
    # Halving every row leaves the E4M3 mantissas untouched because the block
    # scale is a power of two and moves instead. Pinned so that a future switch
    # to a non-E8M0 scale format shows up here rather than as drifting rows.
    table, _ = _table(trainable=True)
    mantissas = table.weight.float().clone()
    scales = table.scale.float().clone()
    with torch.no_grad():
        table.master.mul_(0.5)
    table.refresh_storage()
    assert torch.equal(table.weight.float(), mantissas)
    assert torch.equal(table.scale.float(), scales * 0.5)


def test_forward_does_not_mutate_or_release_storage():
    # "No host offload is performed": the rows a lookup reads must still be the
    # module's own resident buffers afterwards, unchanged and on-device.
    table, _ = _table(trainable=True)
    weight_ptr = table.weight.data_ptr()
    scale_ptr = table.scale.data_ptr()
    snapshot = table.weight.float().clone()
    table(torch.tensor([0, 1, 2])).sum().backward()
    assert table.weight.data_ptr() == weight_ptr
    assert table.scale.data_ptr() == scale_ptr
    assert torch.equal(table.weight.float(), snapshot)
    assert table.master.grad is not None


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda w, s: (w.float(), s), "Expected FP8 table"),
        (lambda w, s: (w, s.float()), "Expected FP8 table"),
        (lambda w, s: (w[:, :31], s), "Expected FP8 table"),
        (lambda w, s: (w, s[:, :1]), "Expected FP8 table"),
        (lambda w, s: (w.reshape(2, 3, WIDTH), s), "Expected FP8 table"),
    ],
)
def test_malformed_storage_is_rejected(mutate, message):
    generator = torch.Generator().manual_seed(11)
    weight, scale = quantize_block_fp8(
        torch.randn(ROWS, WIDTH, generator=generator), (1, 32), scale_format="e8m0"
    )
    with pytest.raises(ValueError, match=message):
        EngramTable(*mutate(weight, scale))


def test_owner_transport_preserves_compact_rows_and_backward(monkeypatch):
    from megatron.lite.primitive.modules.owner_row_transport import (
        _FixedCapacityAllToAll,
    )

    group = object()
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda g: 2)
    calls = []

    def exchange(output, padded, *, group):
        assert padded.shape == (2, 2, 2)
        calls.append(group)
        output.copy_(padded.flip(0))

    monkeypatch.setattr(torch.distributed, 'all_to_all_single', exchange)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    output = _FixedCapacityAllToAll.apply(x, (1, 2), (2, 1), 2, group)
    assert torch.equal(output, x.detach()[[1, 2, 0]])
    gradient = torch.tensor([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]])
    output.backward(gradient)
    assert torch.equal(x.grad, gradient[[2, 0, 1]])
    assert calls == [group, group]
