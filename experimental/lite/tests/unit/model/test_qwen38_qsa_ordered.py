"""Serial-order QSA adjoints must continue prior BF16 accumulation."""

from types import SimpleNamespace

import torch


def test_qsa_owners_follow_physical_contiguous_intervals():
    from megatron.lite.model.qwen3_8_flash_next.qsa_ordered import document_owners

    context = SimpleNamespace(local_sequence_length=8)
    assert document_owners(context, 5, 13) == (0, 1), 'QSA_ORDERED_CONTIGUOUS_OWNERS'
    assert document_owners(context, 8, 13) == (1,), 'QSA_ORDERED_SINGLE_OWNER'


def test_qsa_selected_adjoints_continue_bf16_prefix():
    from megatron.lite.model.qwen3_8_flash_next.qsa_ordered import accumulate_selected

    state = torch.full((2, 1, 1, 1, 128), 256, dtype=torch.bfloat16)
    contribution = (
        torch.tensor([1, -256], dtype=torch.bfloat16)
        .reshape(1, 2, 1, 1, 1)
        .expand(1, 2, 1, 1, 128)
        .contiguous()
    )
    indices = torch.zeros(1, 2, 1, dtype=torch.long)
    batch = torch.zeros(1, 1, 1, dtype=torch.long)
    accumulate_selected(
        state, contribution, contribution, batch, indices, document_start=0, rank=1
    )
    assert not bool(state.count_nonzero()), 'QSA_ORDERED_BF16_PREFIX_CONTINUATION'
    partial = torch.zeros_like(state)
    accumulate_selected(
        partial, contribution, contribution, batch, indices, document_start=0, rank=1
    )
    assert bool(
        (partial + 256).count_nonzero()
    ), 'QSA_ORDERED_PARTIAL_SUM_IS_NOT_PREFIX'


def test_qsa_single_owner_gradients_match_native_index():
    from megatron.lite.model.qwen3_8_flash_next.qsa_ordered import ordered_select_kv

    torch.manual_seed(95)
    local = [
        torch.randn(1, 8, 1, 128, dtype=torch.bfloat16).requires_grad_()
        for _ in range(2)
    ]
    reference = [x.detach().clone().requires_grad_() for x in local]
    context = SimpleNamespace(
        rank=1,
        local_sequence_length=8,
        local_sequence_start=8,
        local_sequence_end=16,
        group=None,
    )
    batch = torch.zeros(1, 1, 1, dtype=torch.long)
    indices = torch.tensor([[[2, 0, 2], [1, 2, 0]]])
    selected = ordered_select_kv(
        local[0],
        local[1],
        local[0][:, :3].detach(),
        local[1][:, :3].detach(),
        batch,
        indices,
        context,
        8,
        11,
    )
    dy = [torch.randn_like(x) for x in selected]
    torch.autograd.backward(selected, dy)
    torch.autograd.backward([x[:, :3][batch, indices] for x in reference], dy)
    for a, b in zip(local, reference):
        assert torch.equal(
            a.grad.view(torch.uint8), b.grad.view(torch.uint8)
        ), 'QSA_ORDERED_TRUE_SERIAL_BITWISE'
