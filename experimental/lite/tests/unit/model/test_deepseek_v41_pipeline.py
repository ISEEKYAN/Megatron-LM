# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paired PP state lifetime and typed transport; full C4 protocol is separate."""

import pytest
import torch


def test_paired_pipeline_owner_returns_are_once_and_generation_scoped():
    from megatron.lite.model.deepseek_v41.lite import pipeline

    owner = torch.tensor([2.0, 5.0], requires_grad=True)
    mix = torch.tensor([0.25, 0.75], requires_grad=True)
    payload = pipeline.PairedPayload(
        h=owner.reshape(1, 1, 2, 1),
        p=mix.reshape(1, 1, 2),
        ced_h=owner.reshape(1, 1, 2, 1),
        ced_p=mix.reshape(1, 1, 2),
    )
    ledger = pipeline.PipelineLedger()
    tag = pipeline.PipelineTag(3, 7, 1, 2)
    ledger.publish(tag, payload, consumers=(20, 24, 36))
    assert ledger.read(tag).ced_p is payload.ced_p
    with pytest.raises(RuntimeError, match='generation'):
        ledger.read(pipeline.PipelineTag(3, 7, 1, 1))
    for consumer, vector in [(36, [-2.0, 7.0]), (20, [1.0, 2.0]), (24, [3.0, -5.0])]:
        grads = {
            name: torch.tensor(vector).reshape(tensor.shape)
            for name, tensor in payload.differentiable().items()
        }
        ledger.return_gradients(tag, consumer, grads)
        with pytest.raises(RuntimeError, match='duplicate'):
            ledger.return_gradients(tag, consumer, grads)
        assert owner.grad is None
        if consumer != 24:
            with pytest.raises(RuntimeError, match='missing'):
                ledger.backward(tag)
    ledger.backward(tag)
    # Two distinct paths (current and CED), each with summed [2,4] cotangent.
    torch.testing.assert_close(owner.grad, torch.tensor([4.0, 8.0]), atol=0, rtol=0)
    torch.testing.assert_close(mix.grad, torch.tensor([4.0, 8.0]), atol=0, rtol=0)
    with pytest.raises(RuntimeError, match='generation'):
        ledger.read(tag)
    with pytest.raises(RuntimeError, match='generation'):
        ledger.publish(tag, payload, consumers=(20,))
    with pytest.raises(ValueError, match='pair'):
        pipeline.PairedPayload(h=payload.h, p=payload.p, ced_h=payload.ced_h)


def _pipeline_worker(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist
    from megatron.lite.model.deepseek_v41.lite import pipeline
    from megatron.lite.primitive.parallel import tensor_payload as transport
    from torch.utils.checkpoint import checkpoint

    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group(
        'nccl',
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    group = dist.group.WORLD
    tags = [pipeline.PipelineTag(7, mb, 1, 3) for mb in range(2)]
    try:
        if rank == 0:
            x = torch.nn.Parameter(torch.tensor([2.0, 5.0], device=device))
            mix = torch.nn.Parameter(torch.tensor([0.25, 0.75], device=device))
            optimizer = torch.optim.SGD([x, mix], lr=0.125)
            ledger = pipeline.PipelineLedger()
            for mb, tag in enumerate(tags):
                current = x * (mb + 1)
                payload = pipeline.PairedPayload(
                    h=(3 * current).reshape(1, 1, 2, 1),
                    p=torch.tensor([0.75, 0.25], device=device).reshape(1, 1, 2),
                    ced_h=current.reshape(1, 1, 2, 1),
                    ced_p=(
                        mix + torch.tensor([mb * 0.125, -mb * 0.125], device=device)
                    ).reshape(1, 1, 2),
                    kv=current.square().reshape(1, 1, 2),
                    index_k=((torch.arange(32, device=device) % 3) - 1)
                    .float()
                    .to(torch.float8_e4m3fn),
                    index_scale=torch.tensor([2.0], device=device).to(
                        torch.float8_e8m0fnu
                    ),
                    topk=torch.tensor([2**40 + mb], device=device, dtype=torch.int64),
                    positions=torch.tensor(
                        [2**33 + mb], device=device, dtype=torch.int64
                    ),
                )
                ledger.publish(tag, payload, consumers=(1, 2, 3))
                for peer in (1, 2, 3):
                    transport.send_tensor_payload(
                        payload.tensors(),
                        (*tag.as_tuple(), 0),
                        peer=peer,
                        group=group,
                        device=device,
                    )
            for tag in reversed(tags):
                for peer in (1, 2, 3):
                    gradients = transport.recv_tensor_payload(
                        (*tag.as_tuple(), 1), peer=peer, group=group, device=device
                    )
                    ledger.return_gradients(
                        tag,
                        peer,
                        {
                            name: value
                            for name, value in zip(pipeline.PAYLOAD_FIELDS, gradients)
                            if value is not None
                        },
                    )
                ledger.backward(tag)
            ledger.assert_quiescent()
            ledger.finish_step(7)
            coefficients = torch.tensor([2.0, 4.0], device=device)
            expected_x = torch.zeros_like(x)
            expected_mix = torch.zeros_like(mix)
            for mb in range(2):
                current = x.detach() * (mb + 1)
                current_mix = mix.detach() + torch.tensor(
                    [mb * 0.125, -mb * 0.125], device=device
                )
                expected_x += (
                    (mb + 1)
                    * (
                        3 * torch.tensor([0.75, 0.25], device=device)
                        + current_mix
                        + 2 * current
                    )
                    * coefficients
                )
                expected_mix += current * coefficients
            torch.testing.assert_close(x.grad, expected_x, atol=0, rtol=0)
            torch.testing.assert_close(mix.grad, expected_mix, atol=0, rtol=0)
            before_x, before_mix = x.detach().clone(), mix.detach().clone()
            optimizer.step()
            torch.testing.assert_close(x, before_x - 0.125 * expected_x, atol=0, rtol=0)
            torch.testing.assert_close(
                mix, before_mix - 0.125 * expected_mix, atol=0, rtol=0
            )
            with pytest.raises(RuntimeError, match='generation'):
                ledger.read(tags[0])
            with pytest.raises(RuntimeError, match='generation'):
                transport.send_tensor_payload(
                    payload.tensors(), (99, 0), peer=1, group=group, device=device
                )
        else:
            received = []
            for mb, tag in enumerate(tags):
                payload = pipeline.PairedPayload.from_tensors(
                    transport.recv_tensor_payload(
                        (*tag.as_tuple(), 0), peer=0, group=group, device=device
                    )
                )
                assert (
                    payload.topk.dtype == torch.int64
                    and payload.topk.item() == 2**40 + mb
                )
                assert payload.positions.item() == 2**33 + mb
                assert payload.index_k.dtype == torch.float8_e4m3fn
                assert payload.index_scale.dtype == torch.float8_e8m0fnu
                assert payload.index_scale.float().item() == 2.0
                torch.testing.assert_close(
                    payload.index_k.float(),
                    ((torch.arange(32, device=device) % 3) - 1).float(),
                    atol=0,
                    rtol=0,
                )
                assert not payload.index_k.requires_grad
                received.append(payload)
            coefficient = torch.tensor(
                [[1.0, 2.0], [3.0, -5.0], [-2.0, 7.0]][rank - 1], device=device
            )
            for tag, payload in reversed(list(zip(tags, received))):

                def loss(h, p, ced_h, ced_p, kv):
                    return (
                        coefficient
                        * (
                            h.reshape(2) * p.reshape(2)
                            + ced_h.reshape(2) * ced_p.reshape(2)
                            + kv.reshape(2)
                        )
                    ).sum()

                result = checkpoint(
                    loss,
                    payload.h,
                    payload.p,
                    payload.ced_h,
                    payload.ced_p,
                    payload.kv,
                    use_reentrant=False,
                )
                fields = payload.differentiable()
                gradients = dict(
                    zip(fields, torch.autograd.grad(result, tuple(fields.values())))
                )
                transport.send_tensor_payload(
                    tuple(gradients.get(name) for name in pipeline.PAYLOAD_FIELDS),
                    (*tag.as_tuple(), 1),
                    peer=0,
                    group=group,
                    device=device,
                )
            if rank == 1:
                with pytest.raises(RuntimeError, match='generation'):
                    transport.recv_tensor_payload(
                        (98, 0), peer=0, group=group, device=device
                    )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(4)
def test_paired_pipeline_nccl_interleaving_recompute_and_owner_step(tmp_path):
    import os

    import torch.multiprocessing as mp

    assert os.getenv('SLURM_JOB_ID'), 'GPU pipeline validation requires Slurm'
    if torch.cuda.device_count() < 4:
        pytest.skip('Requires the declared four-GPU allocation')
    mp.spawn(
        _pipeline_worker, args=(f'file://{tmp_path}/rendezvous',), nprocs=4, join=True
    )
