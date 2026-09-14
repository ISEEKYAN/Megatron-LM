"""CP sequence owners and EP row owners overlap, with distinct adjoint routes."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def state(rank=0):
    return SimpleNamespace(
        cp_size=2,
        ep_size=2,
        dp_size=1,
        expert_dp_size=1,
        tp_size=1,
        etp_size=1,
        pp_size=1,
        cp_rank=rank,
        ep_rank=rank,
        cp_group='cp',
        ep_group='ep',
        dp_cp_group='dense',
        ep_dp_group='expert',
    )


def test_cp_ep_contract_accepts_only_verified_overlap():
    from megatron.lite.model.qwen3_8_flash_next.cp_ep import validate_cp_ep_contract

    for rank in range(2):
        groups = {'cp': [0, 1], 'ep': [0, 1], 'dense': [0, 1], 'expert': [rank]}
        with patch.object(
            dist, 'get_process_group_ranks', side_effect=groups.__getitem__
        ):
            validate_cp_ep_contract(state(rank), ple_owner_sharding=True)
            with pytest.raises(
                NotImplementedError, match='QWEN38_CP_COMBINATION_NOT_VALIDATED'
            ):
                validate_cp_ep_contract(state(rank), ple_owner_sharding=False)
            groups['ep'] = [1, 0]
            with pytest.raises(ValueError, match='QWEN38_CP_EP_OWNER_GROUPS'):
                validate_cp_ep_contract(state(rank), ple_owner_sharding=True)


@pytest.mark.parametrize(
    'field,value',
    [
        ('tp_size', 2),
        ('dp_size', 2),
        ('ep_size', 4),
        ('cp_size', 4),
        ('expert_dp_size', 2),
    ],
)
def test_cp_ep_unverified_combinations_stay_closed(field, value):
    from megatron.lite.model.qwen3_8_flash_next.cp_ep import validate_cp_ep_contract

    ps = state()
    setattr(ps, field, value)
    with pytest.raises(
        NotImplementedError, match='QWEN38_CP_COMBINATION_NOT_VALIDATED'
    ):
        validate_cp_ep_contract(ps, ple_owner_sharding=True)


def _halo_worker(rank, store):
    from megatron.lite.model.qwen3_8_flash_next.cp import (
        qwen3_8_flash_next_cp_left_halo,
        shard_batch_for_qwen3_8_flash_next_cp,
    )
    from megatron.lite.model.qwen3_8_flash_next.engram import (
        Qwen3_8_FlashNextEngramTableConfig,
    )

    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo', init_method=f'file://{store}', rank=rank, world_size=2
    )
    try:
        cp_group = dist.new_group([0, 1])
        ep_group = dist.new_group([0, 1])
        mesh = SimpleNamespace(
            size=lambda: 2, get_local_rank=lambda: rank, get_group=lambda: cp_group
        )
        ids = torch.tensor([[0, 5, 2, 7, 4, 1, 6, 3, 0, 5, 2, 7, 4]])
        _, batch, _ = shard_batch_for_qwen3_8_flash_next_cp(
            mesh, None, {'input_ids': ids, 'cu_seqlens': torch.tensor([0, 5, 13])}
        )
        context = batch['_qwen3_8_flash_next_cp_context']
        full = (torch.arange(24).reshape(8, 3).float() / 8).requires_grad_()
        table = Qwen3_8_FlashNextEngramTableConfig(8, 3).build(
            process_group=ep_group, device='cpu', dtype=torch.float32
        )
        with torch.no_grad():
            table.weight.copy_(full[rank * 4 : (rank + 1) * 4])
        local = table(batch['input_ids'])
        halo = qwen3_8_flash_next_cp_left_halo(local, context, history=3)
        positions = torch.arange(rank * 8, (rank + 1) * 8)
        starts = torch.tensor([0, 5, 13])[
            torch.bucketize(positions, torch.tensor([5, 13]), right=True)
        ]
        valid = (positions < 13).reshape(1, 8, 1)
        prior = (positions - 3 >= starts).reshape(1, 8, 1)
        output = (local + torch.cat((halo, local), 1)[:, :8] * prior / 2) * valid
        padded = torch.nn.functional.pad(ids, (0, 3))
        reference = torch.nn.functional.embedding(padded, full)
        global_positions = torch.arange(16)
        global_starts = torch.tensor([0, 5, 13])[
            torch.bucketize(global_positions, torch.tensor([5, 13]), right=True)
        ]
        expected = (
            reference
            + torch.nn.functional.pad(reference, (0, 0, 3, 0))[:, :16]
            * (global_positions - 3 >= global_starts).reshape(1, 16, 1)
            / 2
        )
        expected = expected * (global_positions < 13).reshape(1, 16, 1)
        dy = torch.arange(48).reshape(1, 16, 3).float() / 16
        assert torch.equal(
            output, expected[:, rank * 8 : (rank + 1) * 8]
        ), 'CP_EP_PLE_HALO_FORWARD_BITWISE'
        output.backward(dy[:, rank * 8 : (rank + 1) * 8])
        expected.backward(dy)
        assert torch.equal(
            table.weight.grad, full.grad[rank * 4 : (rank + 1) * 4]
        ), 'CP_EP_PLE_HALO_OWNER_GRAD_BITWISE'
        assert table.weight.grad.count_nonzero() > 0, 'CP_EP_PLE_OWNER_NONEMPTY'
    finally:
        dist.destroy_process_group()


def test_cp_ep_real_owner_lookup_and_halo_adjoint(tmp_path):
    mp.spawn(_halo_worker, args=(str(tmp_path / 'store'),), nprocs=2, join=True)


def test_model_wires_cp_ep_with_owner_flag(transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_8_flash_next import model as module
    from megatron.lite.model.qwen3_8_flash_next.protocol import build_model_config
    from test_qwen38_training import tiny_training_config

    built = []

    def layer(config, ps, layer_idx, **kwargs):
        built.append((ps.cp_size, ps.ep_size, kwargs['ple_owner_sharding']))
        return torch.nn.Identity()

    monkeypatch.setattr(module, 'Qwen38Layer', layer)
    groups = {'cp': [0, 1], 'ep': [0, 1], 'dense': [0, 1], 'expert': [0]}
    with patch.object(dist, 'get_process_group_ranks', side_effect=groups.__getitem__):
        module.Qwen38Model(
            build_model_config(tiny_training_config()), state(), ple_owner_sharding=True
        )
        assert built == [(2, 2, True), (2, 2, True)], 'CP_EP_MODEL_OWNER_WIRING'
        with pytest.raises(
            NotImplementedError, match='QWEN38_CP_COMBINATION_NOT_VALIDATED'
        ):
            module.Qwen38Model(build_model_config(tiny_training_config()), state())
