# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""host_engram is a residency switch only: same partition, same bytes out.

The tiny 4-layer prefix (frozen Engram table on layer 1, MoE, CSA owner layer)
is built twice from the same seed, once with host-resident Engram storage, and
must produce bitwise-equal head hidden states and parameter gradients. CPU Gloo
runs replicate the table (sharded lookups are CUDA-only); the sharded host gather
is covered by the GPU acceptance run.
"""

import runpy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_PARITY = Path(__file__).with_name('test_cp_ep_parity.py')


def _worker(rank, directory, ep, world, host):
    parity = runpy.run_path(str(_PARITY))
    if world > 1:
        fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
        import megatron.core.fp8_utils  # noqa: F401
        import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
        import megatron.core.transformer.hyper_connection  # noqa: F401

        fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
        torch.set_num_threads(1)
        dist.init_process_group(
            'gloo',
            init_method=f'file://{directory}/store-{host}',
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=300),
        )
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.modules.engram_lookup import EngramTable
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.manual_seed(351)
    impl = replace(parity['_impl'](ParallelConfig(ep=ep)), host_engram=host)
    model = protocol.build_model(parity['_prefix_config'](), impl_cfg=impl).chunks[0]
    tables = [m for m in model.modules() if isinstance(m, EngramTable)]
    assert tables and all(t.host_resident == host for t in tables)
    ids, cu = parity['_inputs']()
    out = model(ids, cu_seqlens=cu, return_head_hidden=True)['head_hidden']
    (out * parity['_objective'](out)).sum().backward()
    torch.save(
        dict(
            hidden=out.detach(),
            grads={
                n: p.grad.clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            },
        ),
        f'{directory}/{host}-{rank}.pt',
    )
    if world > 1:
        dist.destroy_process_group()


@pytest.mark.parametrize('ep,world', [(1, 1), (2, 2)])
def test_host_engram_is_bitwise_at_the_same_partition(v41_core_te, tmp_path, ep, world):
    torch.set_num_threads(1)
    for host in (False, True):
        if world == 1:
            _worker(0, str(tmp_path), ep, world, host)
        else:
            mp.start_processes(
                _worker,
                args=(str(tmp_path), ep, world, host),
                nprocs=world,
                join=True,
                start_method='spawn',
            )
    for rank in range(world):
        resident = torch.load(tmp_path / f'False-{rank}.pt')
        host = torch.load(tmp_path / f'True-{rank}.pt')
        assert torch.equal(host['hidden'], resident['hidden']), rank
        assert host['grads'].keys() == resident['grads'].keys() and resident['grads']
        unequal = [
            name
            for name, value in resident['grads'].items()
            if not torch.equal(host['grads'][name], value)
        ]
        assert not unequal, (rank, unequal[:8])
