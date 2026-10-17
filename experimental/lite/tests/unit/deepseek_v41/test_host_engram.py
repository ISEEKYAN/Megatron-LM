# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host-resident frozen Engram tables: same bytes out, never moved, shareable.

A host-resident table keeps its FP8 rows and E8M0 scales in host memory. Module
device moves must leave that storage in place (no copy either way), lookups must
return exactly the bytes a device-resident table returns, and a second frozen
model may alias the first one's storage only when the bytes are identical.
"""

import pytest
import torch
from megatron.lite.primitive.modules import engram_lookup as memory
from megatron.lite.primitive.modules.engram_lookup import EngramTable
from megatron.lite.primitive.modules.row_memory_build import build_row_memories
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8

ROWS, WIDTH = 6, 64


def _storage(seed=11):
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(ROWS, WIDTH, generator=generator)
    return quantize_block_fp8(source, (1, 32), scale_format="e8m0")


def _bytes(tensor):
    return tensor.view(torch.uint8)


def test_host_table_matches_resident_table_bitwise():
    weight, scale = _storage()
    resident = EngramTable(weight, scale)
    host = EngramTable(weight, scale, host_resident=True)
    ids = torch.tensor([[5, 0, 3], [3, 3, 1]])
    for got, want in zip(host.lookup_fp8(ids), resident.lookup_fp8(ids)):
        if want is None:
            assert got is None
        else:
            assert torch.equal(_bytes(got), _bytes(want))
    assert torch.equal(_bytes(host(ids)), _bytes(resident(ids)))


def test_device_moves_leave_host_storage_in_place():
    weight, scale = _storage()
    host = EngramTable(weight, scale, host_resident=True)
    resident = EngramTable(weight, scale)
    kept = host.weight, host.scale
    # meta stands in for an accelerator on CPU-only hosts.
    host.to("meta")
    resident.to("meta")
    assert host.weight is kept[0] and host.scale is kept[1]
    assert host.weight.device.type == "cpu"
    assert resident.weight.is_meta, "resident tables still follow the module"
    host.bfloat16()
    assert host.weight is kept[0], "a dtype cast must not copy host storage"
    assert host.output_dtype == torch.bfloat16


def test_host_residency_requires_a_frozen_table():
    weight, scale = _storage()
    with pytest.raises(ValueError, match="frozen"):
        EngramTable(weight, scale, trainable=True, host_resident=True)


def test_share_storage_aliases_identical_bytes_and_rejects_different_ones():
    weight, scale = _storage()
    first = EngramTable(weight, scale, host_resident=True)
    second = EngramTable(weight, scale, host_resident=True)
    second.share_storage(first)
    assert second.weight is first.weight and second.scale is first.scale
    assert dict(second.named_buffers())["weight"] is first.weight

    other = EngramTable(*_storage(seed=12), host_resident=True)
    with pytest.raises(ValueError, match="differs"):
        other.share_storage(first)
    with pytest.raises(ValueError, match="host-resident"):
        EngramTable(weight, scale).share_storage(first)


def test_builder_places_host_tables_on_the_host_without_a_full_temporary():
    def projection(fan_in, fan_out):
        return torch.nn.Linear(fan_in, fan_out, bias=False, device="meta")

    constructors = (
        memory.prime_buckets,
        memory.hash_multipliers,
        memory.NgramHash,
        memory.RowLookup,
        memory.EngramTable,
        memory.ShardedEngramTable,
        memory.Engram,
    )
    with torch.device("meta"):
        _, memories = build_row_memories(
            layer_ids=[0],
            row_counts=[ROWS],
            order=2,
            heads=1,
            vocabulary=5,
            width=WIDTH,
            compressed_vocabulary=5,
            pad_id=0,
            token_map=None,
            token_count=5,
            hidden_size=8,
            copies=1,
            eps=1e-6,
            trainable=False,
            group=None,
            group_size=1,
            local_range=(0, 1),
            constructors=constructors,
            projection=projection,
            host_resident=True,
        )
    table = memories[0].embed
    assert table.host_resident
    assert table.weight.device.type == "cpu" and table.weight.is_contiguous()
    assert torch.count_nonzero(_bytes(table.weight)) == 0
    assert torch.equal(
        _bytes(table.scale), torch.full((ROWS, WIDTH // 32), 127, dtype=torch.uint8)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_lookup_copies_only_gathered_rows_and_matches_resident_bytes():
    weight, scale = _storage()
    resident = EngramTable(weight, scale).cuda()
    host = EngramTable(weight, scale, host_resident=True).cuda()
    assert host.weight.device.type == "cpu" and host.weight.is_pinned()
    ids = torch.tensor([[5, 0, 3], [3, 3, 1]], device="cuda")
    out = host(ids)
    assert out.is_cuda
    assert torch.equal(_bytes(out), _bytes(resident(ids)))
