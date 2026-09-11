"""Run under the existing Slurm/container pytest runner with four visible GPUs."""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous):
    from megatron.lite.model.deepseek_v41.lite.parallel import EngramLayout
    from megatron.lite.model.deepseek_v41.lite.engram import EngramTable, ShardedEngramTable
    from megatron.lite.primitive.modules.engram_lookup import RowLookup
    from megatron.lite.primitive.quantization.ds41_fp8 import quantize_swa

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=4)
    try:
        # Deliberately not contiguous global ranks: these are two replicas of
        # two row shards, using the same four WORLD ranks, not a new axis.
        for total_rows in (7, 1):
            layout = EngramLayout(total_rows, ((0, 2), (1, 3)), world_size=4)
            groups, replica_groups = layout.create_groups()
            replica, shard = layout.coordinates(rank)
            lookup = RowLookup(layout.boundaries, groups[replica])
            begin, end = layout.row_intervals[shard]
            source = torch.arange(total_rows * 256, device='cuda').reshape(total_rows, 256).float() / 256
            quantized = quantize_swa(source)
            values = quantized.values[begin:end]
            scales = quantized.scale[begin:end]
            requests = [
                torch.tensor([[total_rows - 1, 0, total_rows - 1, 0] * 6], device='cuda'),
                torch.empty((0, 24), device='cuda', dtype=torch.int64),
            ]
            ids = requests[shard]
            actual_values, actual_scales = lookup.raw_rows(values, scales, ids)
            assert torch.equal(actual_values.view(torch.uint8), quantized.values.view(torch.uint8)[ids])
            assert torch.equal(actual_scales.view(torch.uint8), quantized.scale.view(torch.uint8)[ids])
            assert actual_values.flatten(-2).shape[-1] == 6144
            for trainable in (False, True):
                table = ShardedEngramTable(values, scales, lookup, trainable=trainable)
                reference = EngramTable(quantized.values, quantized.scale, trainable=trainable)
                actual = table(ids)
                torch.testing.assert_close(actual, reference(ids), atol=0, rtol=0)
                if trainable:
                    # Every rank participates, even the rank with zero requests
                    # or zero owned rows. Distinct weights expose permutation bugs.
                    coeff = torch.arange(ids.numel(), device='cuda').reshape(ids.shape).float() + 1
                    (actual.float() * coeff[..., None]).sum().backward()
                    expected = torch.zeros_like(source)
                    for request in requests:
                        for i, row in enumerate(request.flatten().tolist()):
                            expected[row] += i + 1
                    assert table.master.grad.dtype == torch.float32
                    torch.testing.assert_close(table.master.grad, expected[begin:end], atol=0, rtol=0)
                else:
                    assert not list(table.parameters())
                    assert all(t.device.type == 'cuda' for t in table.buffers())
            # Invalid IDs on one requester must make both group ranks fail,
            # rather than stranding a peer inside the following all-to-all.
            bad = ids.clone()
            if bad.numel():
                bad[0, 0] = total_rows
            try:
                lookup.raw_rows(values, scales, bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid peer ID was not rejected')
            for group in groups + replica_groups:
                if group != dist.GroupMember.NON_GROUP_MEMBER:
                    dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()


def test_lookup_nccl_bytes_and_gradients(tmp_path):
    # Fail instead of silently skipping missing acceptance infrastructure.
    assert os.environ.get('SLURM_JOB_ID'), 'Distributed acceptance must run in Slurm'
    assert torch.cuda.device_count() >= 4, 'Four GPUs required for row shards and replicas'
    mp.spawn(_worker, args=('file://' + str(tmp_path / 'rendezvous'),), nprocs=4, join=True)
