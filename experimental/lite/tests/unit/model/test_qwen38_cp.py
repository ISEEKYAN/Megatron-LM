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
def test_cp_int32_hash_matches_int64(packed):
    from megatron.lite.model.qwen3_8_flash_next.engram import (
        Qwen3_8_FlashNextNGramEmbedding,
    )

    m = Qwen3_8_FlashNextNGramEmbedding(None)
    ids = torch.tensor([[248319, 9, 248044, 11, 13, 200001]], dtype=torch.int32)
    _, batch, _ = cp.shard_batch_for_qwen3_8_flash_next_cp(
        mesh(0, 1), None, {'input_ids': ids}
    )
    ids = batch['_qwen3_8_flash_next_cp_context'].global_input_ids
    cu = torch.tensor([0, 3, 8]) if packed else None
    actual = m.hash_ids(ids, cu)
    assert actual.dtype == torch.int64
    assert torch.equal(actual, m.hash_ids(ids.long(), cu)), 'CP_INT32_HASH_EXACT'


@pytest.mark.parametrize(
    'key,fill',
    [
        ('input_ids', 42),
        ('labels', -100),
        ('position_ids', 0),
        ('attention_mask', 0),
        ('padding_mask', True),
        ('loss_mask', 0),
    ],
)
def test_cp_padding_fill(key, fill):
    ids = torch.arange(5).reshape(1, 5)
    value = ids.bool() if key == 'padding_mask' else ids.clone()
    batch = {'input_ids': ids, key: value}
    _, out, _ = cp.shard_batch_for_qwen3_8_flash_next_cp(
        mesh(1, 2), None, batch, padding_token_id=42
    )
    assert torch.equal(out[key][:, :1], value[:, 4:])
    assert (out[key][:, 1:] == fill).all(), 'CP_PAD_FILL'


@pytest.mark.parametrize('value', [None, torch.ones(1, 5)])
def test_cp_unknown_batch_key_rejected(value):
    with pytest.raises(ValueError, match='CP_UNKNOWN_BATCH_KEYS.*token_weights'):
        cp.shard_batch_for_qwen3_8_flash_next_cp(
            mesh(1, 2),
            None,
            {'input_ids': torch.ones(1, 5, dtype=torch.long), 'token_weights': value},
        )


@pytest.mark.parametrize('metadata', ['none', 'seq_lens', 'cu_seqlens'])
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_ple_cp_padding_matches_separate_sequences(metadata, dtype):
    from megatron.lite.model.qwen3_8_flash_next.engram import (
        Qwen3_8_FlashNextNGramEmbedding,
        Qwen3_8_FlashNextPLELayer,
    )

    torch.manual_seed(38)
    embedding = Qwen3_8_FlashNextNGramEmbedding(
        lambda ids: (ids.remainder(31).float() / 31).unsqueeze(-1)
    )
    m = Qwen3_8_FlashNextPLELayer(
        embedding, hidden_size=2, hc_count=4, ple_embed_dim=16, dtype=torch.float32
    )
    with torch.no_grad():
        m.conv1d.weight.fill_(0.2)
    # Left, interior, right and alignment padding; nonzero pad IDs expose leakage.
    ids = torch.tensor([[99, 99, 1, 2, 3, 4, 99, 5, 6, 7, 8, 99, 99]], dtype=dtype)
    batch = {'input_ids': ids, 'padding_mask': ids == 99}
    if metadata == 'seq_lens':
        batch[metadata] = torch.tensor([7, 6, -1000])
    elif metadata == 'cu_seqlens':
        batch[metadata] = torch.tensor([0, 7, 13])
    _, out, _ = cp.shard_batch_for_qwen3_8_flash_next_cp(mesh(0, 1), None, batch)
    x = torch.randn(1, out['input_ids'].shape[1], 8, requires_grad=True)
    actual = m(x, out['input_ids'], cp_context=out['_qwen3_8_flash_next_cp_context'])
    for a, b in [(2, 6), (7, 11)]:
        expected = m(x[:, a:b], ids[:, a:b].long())
        torch.testing.assert_close(actual[:, a:b], expected, atol=1e-6, rtol=1e-5)
    actual[:, [2, 3, 4, 5, 7, 8, 9, 10]].sum().backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad[:, [0, 1, 6, 11, 12, 13, 14, 15]] == 0).all(), 'PLE_PAD_GRAD'


def test_unassembled_runtime_diagnostic():
    from megatron.lite.model.registry import (
        get_train_runtime_module,
        resolve_runtime_model_name,
    )

    for resolve, args in [
        (get_train_runtime_module, ()),
        (resolve_runtime_model_name, ('lite',)),
    ]:
        with pytest.raises(ValueError, match='not yet assembled'):
            resolve('qwen3_8_flash_next', *args)


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
