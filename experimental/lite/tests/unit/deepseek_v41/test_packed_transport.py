# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real Core/Gloo transport against the full packed document reference."""

import datetime
import hashlib
import json
import multiprocessing
import sys
from pathlib import Path

import pytest
import torch


def _geometry(bounds, rank=0, size=1, group=None):
    from megatron.lite.primitive.modules.attention.packed_geometry import (
        PackedCPGeometry,
    )

    return PackedCPGeometry(bounds, rank, size, group)


def test_cp1_boundary_is_fixed_zero_and_compressed_map_is_core_order():
    from megatron.lite.primitive.modules.attention.packed_transport import (
        gather_packed_compressed_rows,
        packed_boundary_hidden,
    )

    geometry = _geometry((0, 3, 8))
    hidden = torch.arange(24, dtype=torch.float32).reshape(8, 3).requires_grad_()
    boundary = packed_boundary_hidden(hidden, geometry, window_size=2, ratio=2)
    assert boundary.shape == (2, 3)
    assert torch.equal(boundary, torch.zeros_like(boundary))
    assert packed_boundary_hidden(hidden, geometry, window_size=0).shape == (0, 3)
    mapping = torch.tensor([2, 0, 1], dtype=torch.int32)
    local = hidden[:3]
    rank_major, ordered = gather_packed_compressed_rows(local, mapping, geometry)
    assert rank_major is local
    assert torch.equal(ordered, local[[2, 0, 1]])
    ordered.sum().backward()
    assert torch.equal(hidden.grad[:3], torch.ones_like(hidden[:3]))
    assert torch.equal(hidden.grad[3:], torch.zeros_like(hidden[3:]))


def test_short_pack_rejects_multi_hop_before_communication():
    from megatron.lite.primitive.modules.attention.packed_transport import (
        packed_boundary_hidden,
    )

    for rank in range(3):
        geometry = _geometry((0, 1), rank, 3)
        real_rows = 1 if rank == 0 else 0
        with pytest.raises(ValueError, match=r'T=1, cp=3, width=1, d=128'):
            packed_boundary_hidden(torch.zeros(real_rows, 3), geometry, window_size=128)


def _worker(rank, size, init_file, bounds, window, queue):
    import torch.distributed as dist
    from megatron.lite.primitive.modules.attention.packed_indices import (
        packed_sparse_metadata,
    )
    from megatron.lite.primitive.modules.attention.packed_transport import (
        gather_packed_compressed_rows,
        packed_boundary_hidden,
    )

    original_current_device = torch.cuda.current_device
    allocations = {'_gather_along_first_dim': 0, '_reduce_scatter_along_first_dim': 0}

    def cpu_buffer_device():
        # CPU gate only: Core hardcodes current_device for its temporary
        # collective buffers. Keep its actual Gloo communication and VJP.
        import megatron.core.tensor_parallel.mappings as mappings

        frame = sys._getframe(1)
        assert (
            Path(frame.f_code.co_filename).resolve()
            == Path(mappings.__file__).resolve()
        )
        assert frame.f_code.co_name in allocations
        allocations[frame.f_code.co_name] += 1
        return torch.device('cpu')

    try:
        torch.cuda.current_device = cpu_buffer_device
        torch.set_num_threads(1)
        dist.init_process_group(
            'gloo',
            rank=rank,
            world_size=size,
            init_method=f'file://{init_file}',
            timeout=datetime.timedelta(seconds=60),
        )
        geometry = _geometry(bounds, rank, size, dist.group.WORLD)
        full = torch.arange(bounds[-1] * 3, dtype=torch.float32).reshape(1, -1, 3)
        real_rows = max(0, min(geometry.width, geometry.total - geometry.global_start))
        hidden = geometry.slice(full).squeeze(0)[:real_rows].requires_grad_()
        boundary = packed_boundary_hidden(hidden, geometry, window_size=window, ratio=2)
        padded_full = torch.nn.functional.pad(
            full, (0, 0, 0, geometry.padded_total - geometry.total)
        )
        expected = torch.cat((full.new_zeros((1, window, 3)), padded_full), 1)[
            0, geometry.global_start : geometry.global_start + window
        ]
        assert torch.equal(boundary, expected), (rank, 'halo')
        boundary.sum().backward()
        expected_grad = hidden.new_zeros((geometry.width, 3))
        if rank + 1 < size:
            expected_grad[-window:] = 1
        expected_grad = expected_grad[:real_rows]
        assert torch.equal(hidden.grad, expected_grad), (rank, 'halo backward')

        # The halo may contain a previous document, but metadata must never
        # select it. This exercises A2 addresses against the actual A3 buffer.
        indices, lengths = packed_sparse_metadata(
            geometry, window_size=window, lookback_rows=window
        )
        buffer = torch.cat((boundary.detach(), hidden.detach()), 0)
        for row in range(indices.shape[0]):
            global_row = geometry.global_start + row
            doc = next(i for i, end in enumerate(bounds[1:]) if global_row < end)
            selected = indices[row, : lengths[row]].long()
            reference = full[
                0, max(bounds[doc], global_row - window + 1) : global_row + 1
            ]
            assert torch.equal(buffer[selected], reference), (rank, doc, global_row)

        for ratio in (1, 2):
            values = (
                geometry.slice(full).squeeze(0)[:real_rows].detach().requires_grad_()
            )
            halo = packed_boundary_hidden(
                values, geometry, window_size=window, ratio=ratio
            )
            capacity = geometry.width // ratio + 1
            grouped, ids, positions, comp_cu, row_map = geometry.compressor_input(
                torch.nn.functional.pad(values, (0, 0, 0, geometry.width - real_rows)),
                halo,
                ratio,
                halo=ratio - 1,
                capacity=capacity,
            )
            compressed = grouped.reshape(capacity, ratio, 3).sum(1)
            # Core row_map has fixed capacity, with -1 tail slots beyond its
            # compressed prefix. Only the prefix is the real sequence domain.
            logical_map = row_map[: int(comp_cu[-1])]
            _, ordered = gather_packed_compressed_rows(
                compressed, logical_map, geometry
            )
            reference = torch.cat(
                [
                    full[0, begin : begin + ((end - begin) // ratio) * ratio]
                    .reshape(-1, ratio, 3)
                    .sum(1)
                    for begin, end in zip(bounds, bounds[1:])
                ]
            )
            assert torch.equal(ordered, reference), (rank, ratio, 'compressed')
            # Each rank consumes every global compressed row. Core's backward
            # sums both replicas before halo gradients return to their owners.
            ordered.sum().backward()
            reference_grad = torch.zeros_like(full)
            for begin, end in zip(bounds, bounds[1:]):
                cutoff = begin + ((end - begin) // ratio) * ratio
                reference_grad[:, begin:cutoff] = size
            expected_grad = geometry.slice(reference_grad).squeeze(0)[:real_rows]
            assert torch.equal(values.grad, expected_grad), (
                rank,
                ratio,
                'compressed backward',
            )
        assert allocations == {
            '_gather_along_first_dim': 2,
            '_reduce_scatter_along_first_dim': 2,
        }, allocations
        receipt = {
            'rank': rank,
            'ok': True,
            'cp_size': size,
            'bounds': bounds,
            'width': geometry.width,
            'halo_rows': window,
            'halo_sha256': hashlib.sha256(
                boundary.detach().contiguous().numpy().tobytes()
            ).hexdigest(),
            'compressed_ratios': [1, 2],
            'real_rows': real_rows,
            'cpu_buffer_device_allocations': allocations,
            'cpu_adapter_scope': 'CPU gate only: torch.cuda.current_device intercepted only in Core mappings.py _gather_along_first_dim/_reduce_scatter_along_first_dim buffer allocation; actual Gloo communication and VJP unchanged; GPU gate unaffected',
            'value_and_gradient_equal': True,
            'scope': 'real Core/Gloo halo and compressed gather; torch metadata reference; no native CUDA writer',
        }
        Path(init_file).with_name(f'rank-{rank}.json').write_text(json.dumps(receipt))
        queue.put((rank, None))
    except BaseException as error:
        queue.put((rank, repr(error)))
        raise
    finally:
        torch.cuda.current_device = original_current_device
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize(
    'bounds,size,window',
    [((0, 3, 8), 2, 2), ((0, 2, 9), 2, 2), ((0, 2, 7), 2, 2), ((0, 1), 3, 1)],
)
def test_gloo_halo_and_compressed_rows_match_full_documents(
    tmp_path, bounds, size, window
):
    context = multiprocessing.get_context('fork')
    queue = context.Queue()
    workers = [
        context.Process(
            target=_worker, args=(rank, size, tmp_path / 'pg', bounds, window, queue)
        )
        for rank in range(size)
    ]
    try:
        for worker in workers:
            worker.start()
        results = dict(queue.get(timeout=90) for _ in workers)
        for worker in workers:
            worker.join(timeout=10)
        assert results == {rank: None for rank in range(size)}, results
        assert all(worker.exitcode == 0 for worker in workers)
        receipts = [
            json.loads((tmp_path / f'rank-{rank}.json').read_text())
            for rank in range(size)
        ]
        assert all(r['ok'] and r['rank'] == rank for rank, r in enumerate(receipts))
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
