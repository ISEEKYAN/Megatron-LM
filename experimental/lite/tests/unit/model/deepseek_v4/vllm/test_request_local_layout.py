from __future__ import annotations

import inspect

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_request_local_layout_maps_cp_boundary_local_and_compressed_rows() -> None:
    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.request_local_layout import (
        build_request_local_layout,
    )

    cu_seqlens = torch.tensor([0, 12, 20], dtype=torch.int32, device="cuda")
    cu_seqlens_compressed = torch.tensor(
        [0, 3, 5], dtype=torch.int32, device="cuda"
    )
    seq_to_rank_row = torch.tensor(
        [0, 3, 1, 4, 2], dtype=torch.int32, device="cuda"
    )
    physical = torch.full((10, 1, 4), -1, dtype=torch.int32, device="cuda")
    # Rank-local queries are global positions [10, 20). Physical workspace is
    # [boundary positions 8,9 | local positions 10..19 | rank-major compressed].
    physical[0, 0] = torch.tensor([0, 1, 2, 12], device="cuda")
    physical[2, 0] = torch.tensor([4, 16, -1, -1], device="cuda")
    original_physical = physical.clone()

    local, row_map = build_request_local_layout(
        physical,
        cu_seqlens,
        cu_seqlens_compressed,
        seq_to_rank_row,
        global_start=10,
        l_local=10,
        d_window=2,
        physical_workspace_rows=17,
    )
    torch.cuda.synchronize()

    assert local[0, 0].tolist() == [8, 9, 10, 12]
    assert local[2, 0].tolist() == [0, 8, -1, -1]
    sentinel = 17
    expected = [
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        0,
        1,
        2,
        3,
        12,
        15,
        13,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        16,
        14,
    ]
    assert row_map.tolist() == expected

    # The optimized canonical workspace may order rows differently from the
    # legacy unique/searchsorted compaction, but every lowered index must still
    # select the exact same physical KV value.
    workspace = torch.arange(
        1, 18, dtype=torch.int32, device="cuda"
    ).view(17, 1)
    valid_rows = row_map < workspace.shape[0]
    request_workspace = workspace.index_select(
        0, row_map.clamp_max(workspace.shape[0] - 1).to(torch.int64)
    )
    request_workspace.masked_fill_(~valid_rows.unsqueeze(1), 0)
    for row, request_offset in ((0, 0), (2, 15)):
        valid = original_physical[row, 0] >= 0
        physical_values = workspace.index_select(
            0, original_physical[row, 0, valid].to(torch.int64)
        )
        local_values = request_workspace.index_select(
            0, request_offset + local[row, 0, valid].to(torch.int64)
        )
        assert torch.equal(physical_values, local_values)

    differentiable_workspace = torch.randn(
        17, 2, dtype=torch.float32, device="cuda", requires_grad=True
    )
    gathered = differentiable_workspace.index_select(
        0, row_map.clamp_max(16).to(torch.int64)
    )
    gathered.masked_fill_(~valid_rows.unsqueeze(1), 0)
    gathered.sum().backward()
    # Every physical row occurs exactly once in this fixture; sentinel rows
    # must not leak gradient through their clamped gather source.
    assert torch.equal(
        differentiable_workspace.grad,
        torch.ones_like(differentiable_workspace),
    )


def test_attention_request_local_path_has_no_dynamic_set_remap() -> None:
    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.module import (
        VLLMAttention,
    )

    source = inspect.getsource(VLLMAttention._forward_training_attention)
    assert "torch.unique" not in source
    assert "torch.searchsorted" not in source
    assert "selected_workspace" not in source
    assert ".detach().cpu().tolist()" not in source
    assert "cu_seqlens[seq_idx].item()" not in source
    assert source.count("flash_mla_sparse_fwd(") == 2


def test_request_local_layout_supports_uncompressed_layers() -> None:
    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.request_local_layout import (
        build_request_local_layout,
    )

    cu = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
    cu_compressed = torch.zeros_like(cu)
    seq_to_rank = torch.empty(0, dtype=torch.int32, device="cuda")
    physical = torch.tensor(
        [[[0, 3]], [[1, 3]], [[2, 3]], [[3, -1]], [[4, 7]], [[5, 7]], [[6, 7]], [[7, -1]]],
        dtype=torch.int32,
        device="cuda",
    )
    local, row_map = build_request_local_layout(
        physical,
        cu,
        cu_compressed,
        seq_to_rank,
        global_start=0,
        l_local=8,
        d_window=0,
        physical_workspace_rows=8,
    )
    torch.cuda.synchronize()
    assert local[0, 0].tolist() == [0, 3]
    assert local[4, 0].tolist() == [0, 3]
    assert row_map.tolist() == list(range(8))


@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("cp_size", [1, 2])
def test_pack_request_local_indices_uses_one_packed_coordinate(
    batch_size: int,
    cp_size: int,
) -> None:
    from megatron.lite.model.deepseek_v4.vllm.primitive.attention.module import (
        _pack_request_local_indices,
    )

    lengths = torch.tensor(
        [2048 if index % 2 == 0 else 6144 for index in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )
    compressed = lengths // 4
    cu = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), lengths.cumsum(0)))
    cu_compressed = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.int32), compressed.cumsum(0))
    )
    total = int(lengths.sum().cpu())
    local_rows = total // cp_size
    global_start = 0 if cp_size == 1 else local_rows
    request_indices = torch.zeros((local_rows, 1, 2), dtype=torch.int32, device="cuda")
    request_indices[..., 1] = -1
    packed = _pack_request_local_indices(
        request_indices, cu, cu_compressed, global_start=global_start
    )

    global_queries = torch.arange(
        global_start, global_start + local_rows, dtype=torch.int32, device="cuda"
    )
    token_request_ids = torch.searchsorted(
        cu[1:].contiguous(), global_queries, right=True
    )
    expected_offsets = (cu[:-1] + cu_compressed[:-1]).index_select(
        0, token_request_ids
    )
    assert torch.equal(packed[:, 0, 0], expected_offsets)
    assert packed[:, 0, 1].eq(-1).all()
