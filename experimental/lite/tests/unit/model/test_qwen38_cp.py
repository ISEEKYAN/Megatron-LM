from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.model.qwen3_8_flash_next import cp
from megatron.lite.model.registry import get_model_package, resolve_model_type_from_hf


def mesh(rank, size):
    return SimpleNamespace(
        size=lambda: size, get_local_rank=lambda: rank, get_group=lambda: 'test-group'
    )


@pytest.mark.parametrize('packed', [False, True])
def test_contiguous_packed_padded(packed):
    ids = torch.arange(7).reshape(1, 7)
    batch = {'input_ids': ids, 'labels': ids.clone()}
    if packed:
        batch['seq_lens'] = torch.tensor([3, 4, -1000])
    else:
        batch['attention_mask'] = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    outputs = []
    for rank in range(2):
        _, out, layout = cp.shard_batch_for_qwen3_8_flash_next_cp(
            mesh(rank, 2), None, batch
        )
        context = out['_qwen3_8_flash_next_cp_context']
        expected = torch.nn.functional.pad(ids, (0, 1))[:, rank * 4 : (rank + 1) * 4]
        assert torch.equal(out['input_ids'], expected), 'CP_RANK_INTERVAL'
        assert (
            context.local_sequence_start == rank * 4
            and context.local_sequence_end == (rank + 1) * 4
        ), 'CP_CONTEXT_INTERVAL'
        assert layout.padded_seq_len == 8, 'CP_PAD_ALIGNMENT'
        if packed:
            assert context.global_cu_seqlens.tolist() == [
                0,
                3,
                7,
            ], 'CP_REAL_PACKED_BOUNDARIES'
        else:
            assert context.global_sequence_lengths.tolist() == [5], 'CP_PADDED_LENGTH'
        outputs.append(out['input_ids'])
    assert torch.equal(torch.cat(outputs, 1)[:, :7], ids), 'CP_RECONSTRUCTION'


def test_cp_gather_and_halo_delegate(monkeypatch):
    parts = [
        torch.arange(4.0).reshape(1, 4, 1).requires_grad_(),
        torch.ones(1, 4, 1, requires_grad=True),
    ]
    _, batch, _ = cp.shard_batch_for_qwen3_8_flash_next_cp(
        mesh(1, 2), None, {'input_ids': torch.ones(1, 8, dtype=torch.long)}
    )
    ctx = batch['_qwen3_8_flash_next_cp_context']
    monkeypatch.setattr(cp, '_all_gather_cp', lambda tensor, group: parts)
    monkeypatch.setattr(
        cp,
        '_gather_contiguous_tail',
        lambda tensor, **kwargs: [p[:, -kwargs['tail_len'] :] for p in parts],
    )
    assert torch.equal(
        cp.qwen3_8_flash_next_cp_all_gather(parts[1], ctx), torch.cat(parts, 1)
    ), 'CP_GATHER_ORDER'
    halo = cp.qwen3_8_flash_next_cp_left_halo(parts[1], ctx, history=3)
    assert halo.flatten().tolist() == [1.0, 2.0, 3.0], 'CP_LEFT_HALO'
    halo.sum().backward()
    assert (
        parts[0].grad.flatten().tolist() == [0.0, 1.0, 1.0, 1.0]
        and parts[1].grad is not None
    ), 'CP_HALO_GRAD_ANCHOR'


@pytest.mark.parametrize(
    'alias',
    ['qwen4_exp', 'qwen4_exp_text', 'qwen3_8_flash_next', 'qwen3_8_flash_next_text'],
)
def test_registry(alias):
    assert (
        resolve_model_type_from_hf({'model_type': alias}) == 'qwen3_8_flash_next'
    ), 'QWEN38_REGISTRY'
    assert hasattr(
        get_model_package('qwen3_8_flash_next'), 'Qwen3_8_FlashNextTextConfig'
    ), 'QWEN38_PACKAGE'


@pytest.mark.parametrize(
    'tag',
    [
        'CP_CONTIGUOUS_INTERVAL',
        'CP_GLOBAL_METADATA',
        'CP_PACKED_BOUNDARIES',
        'CP_GROUP_REQUIRED',
        'CP_HALO_SHAPE_HISTORY',
        'CP_PACKED_LENGTHS',
        'CP_PACKED_OVERFLOW',
        'CP_TP_UNSUPPORTED',
        'CP_PAD_MULTIPLE',
    ],
)
def test_cp_guards(tag):
    ctx = cp.Qwen3_8_FlashNextCPContext(
        None,
        0,
        2,
        torch.ones(1, 8, dtype=torch.long),
        torch.zeros(1, 8, dtype=torch.bool),
        0,
        4,
    )
    try:
        if tag == 'CP_CONTIGUOUS_INTERVAL':
            replace(ctx, local_sequence_start=1)
        elif tag == 'CP_GLOBAL_METADATA':
            replace(ctx, global_padding_mask=torch.zeros(1, 7, dtype=torch.bool))
        elif tag == 'CP_PACKED_BOUNDARIES':
            replace(ctx, global_cu_seqlens=torch.tensor([1, 8]))
        elif tag == 'CP_GROUP_REQUIRED':
            cp.qwen3_8_flash_next_cp_all_gather(torch.ones(1, 4, 1), ctx)
        elif tag == 'CP_HALO_SHAPE_HISTORY':
            cp.qwen3_8_flash_next_cp_left_halo(torch.ones(1, 4, 1), ctx, history=-1)
        elif tag == 'CP_PACKED_LENGTHS':
            cp.packed_boundaries_from_seq_lens(torch.tensor([0]))
        elif tag == 'CP_PACKED_OVERFLOW':
            cp.packed_boundaries_from_seq_lens(torch.tensor([5]), total_tokens=4)
        elif tag == 'CP_TP_UNSUPPORTED':
            cp.shard_batch_for_qwen3_8_flash_next_cp(mesh(0, 2), mesh(0, 2), {})
        else:
            cp.shard_batch_for_qwen3_8_flash_next_cp(
                mesh(0, 2),
                None,
                {'input_ids': torch.ones(1, 8, dtype=torch.long)},
                pad_multiple=0,
            )
    except Exception as error:
        assert tag in str(error), f'GUARD_{tag}: wrong failure {error!r}'
    else:
        assert False, f'GUARD_{tag}: missing rejection'
