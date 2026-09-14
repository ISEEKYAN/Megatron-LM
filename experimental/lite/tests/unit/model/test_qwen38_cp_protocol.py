from types import SimpleNamespace

import pytest
import torch
from megatron.lite.model.qwen3_8_flash_next.protocol import _forward_step
from megatron.lite.runtime.contracts import PackedBatch


class Capture:
    def __init__(self, rank, *, tp=1):
        self.ps = SimpleNamespace(cp_size=2, cp_rank=rank, cp_group='cp', tp_size=tp)

    def __call__(self, **kwargs):
        return kwargs


@pytest.mark.parametrize('rank', [0, 1])
def test_cp_protocol_shifts_globally_then_slices(rank):
    ids = torch.arange(13)
    # A masked target inside the second doc; the CP boundary is inside that doc.
    mask = torch.ones(13, dtype=torch.bool)
    mask[10] = False
    batch = PackedBatch(ids, ids.clone(), torch.tensor([5, 8]), loss_mask=mask)

    class Wrapper:
        module = Capture(rank)

        def __call__(self, **kwargs):
            return self.module(**kwargs)

    actual = _forward_step(Wrapper(), batch)
    expected_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0]])
    expected_labels = torch.tensor(
        [[1, 2, 3, 4, -100, 6, 7, 8, 9, -100, 11, 12, -100, -100, -100, -100]]
    )
    expected_positions = torch.tensor(
        [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0]]
    )
    interval = slice(rank * 8, (rank + 1) * 8)
    assert torch.equal(
        actual['input_ids'], expected_ids[:, interval]
    ), 'CP_PROTOCOL_CONTIGUOUS_IDS'
    assert torch.equal(
        actual['labels'], expected_labels[:, interval]
    ), 'CP_PROTOCOL_GLOBAL_SHIFT'
    assert torch.equal(
        actual['position_ids'], expected_positions[:, interval]
    ), 'CP_PROTOCOL_DOCUMENT_POSITIONS'
    assert actual['cu_seqlens'].tolist() == [
        0,
        5,
        13,
        16,
    ], 'CP_PROTOCOL_PHYSICAL_BOUNDARIES'
    context = actual['cp_context']
    assert context.global_cu_seqlens.tolist() == [0, 5, 13]
    assert (
        int((~context.global_padding_mask).sum()) == 13
    ), 'CP_PROTOCOL_REAL_ROUTER_TOKENS'
    assert int(actual['loss_token_count']) == 10, 'CP_PROTOCOL_GLOBAL_LOSS_TOKENS'
    assert torch.equal(ids, torch.arange(13)), 'CP_PROTOCOL_INPUT_NOT_MUTATED'


def test_cp_protocol_single_document_without_labels():
    ids = torch.arange(5)
    actual = _forward_step(Capture(1), PackedBatch(ids, None, torch.tensor([5])))
    assert actual['input_ids'].tolist() == [
        [4, 0, 0, 0]
    ], 'CP_PROTOCOL_CONTIGUOUS_SINGLE_DOCUMENT'
    assert actual['labels'] is None and actual['loss_token_count'] is None
    assert actual['cu_seqlens'].tolist() == [0, 5, 8]
    assert actual['cp_context'].global_cu_seqlens.tolist() == [0, 5]


def test_cp_protocol_preserves_tp_combination_guard():
    ids = torch.arange(5)
    with pytest.raises(ValueError, match='CP_TP_UNSUPPORTED'):
        _forward_step(
            Capture(0, tp=2), PackedBatch(ids, ids.clone(), torch.tensor([5]))
        )
