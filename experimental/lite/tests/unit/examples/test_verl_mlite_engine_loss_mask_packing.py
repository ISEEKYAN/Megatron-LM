# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Regression coverage for THD loss-mask packing in the MLite VERL engine.

VERL hands the actor-update batch a *response-only* ``loss_mask`` /
``response_mask`` nested tensor (shape ``[bsz, response_len]``), while
``input_ids`` is the full ``[prompt; response]`` packed sequence. The engine
must expand that mask to the full sequence length before the model protocol
packs it against the ``input_ids`` seq_lens. Returning the response-only mask
unchanged left ``loss_mask`` shorter than the declared seq_lens and crashed the
update step inside ``_nested_from_packed_tensor``::

    torch.narrow(0, offset, 206): start + length exceeds dimension size (128)

These tests exercise the pure ``_loss_mask_for_packing`` static logic end-to-end
with ``_nested_from_packed_tensor``; no GPU, model init, or torch.distributed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))


from megatron.lite.model.deepseek_v4.lite.protocol import _nested_from_packed_tensor

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def _require_verl() -> None:
    pytest.importorskip("verl", reason="VERL is required for this optional example test.")


def _tensor_dict(data, batch_size):
    from tensordict import TensorDict

    return TensorDict(data, batch_size=batch_size)


def _loss_mask_for_packing(micro_batch, input_ids):
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    return MegatronLiteEngine._loss_mask_for_packing(micro_batch, input_ids)


def _full_input_ids(full_lengths: list[int]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(
        [torch.arange(length) for length in full_lengths], layout=torch.jagged
    )


def _response_mask_row(response_len: int) -> torch.Tensor:
    # Include an internal zero to model a masked span (e.g. tool output); the
    # whole valid response span must be preserved, not collapsed to its sum.
    row = torch.ones(response_len, dtype=torch.float32)
    if response_len > 4:
        row[3] = 0.0
    return row


def _assert_packs_to_full(packed_mask, input_ids, full_lengths, response_lengths):
    seq_lens = input_ids.offsets().diff().to(torch.int64)
    # Must be full-length: sum of response lengths alone would be too short.
    assert int(packed_mask.values().numel()) == sum(full_lengths)
    # Packing against the full seq_lens must not overflow (the original crash).
    nested = _nested_from_packed_tensor(packed_mask.values().contiguous(), seq_lens)
    rows = nested.unbind(0)
    for i, (total, resp) in enumerate(zip(full_lengths, response_lengths, strict=True)):
        prompt_len = total - resp
        # Prompt positions are masked out (zeros); response span is kept intact.
        assert torch.count_nonzero(rows[i][:prompt_len]) == 0
        assert rows[i][prompt_len:].numel() == resp


def test_nested_response_only_loss_mask_expands_to_full_length():
    """The production crash case: response-only nested mask + full input_ids."""
    full_lengths = [206, 130, 40]
    response_lengths = [78, 30, 20]  # sum 128 < a single full length (206)
    input_ids = _full_input_ids(full_lengths)
    loss_mask = torch.nested.as_nested_tensor(
        [_response_mask_row(r) for r in response_lengths], layout=torch.jagged
    )
    micro_batch = _tensor_dict(
        {"input_ids": input_ids, "loss_mask": loss_mask}, batch_size=[len(full_lengths)]
    )

    packed = _loss_mask_for_packing(micro_batch, input_ids)
    _assert_packs_to_full(packed, input_ids, full_lengths, response_lengths)


def test_full_length_nested_loss_mask_is_unchanged():
    """A response covering the whole sequence (prompt_len == 0) still round-trips."""
    full_lengths = [100, 100]
    response_lengths = [100, 100]
    input_ids = _full_input_ids(full_lengths)
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(r, dtype=torch.float32) for r in response_lengths], layout=torch.jagged
    )
    micro_batch = _tensor_dict(
        {"input_ids": input_ids, "loss_mask": loss_mask}, batch_size=[len(full_lengths)]
    )

    packed = _loss_mask_for_packing(micro_batch, input_ids)
    _assert_packs_to_full(packed, input_ids, full_lengths, response_lengths)


def test_dense_and_nested_masks_match_with_unsupervised_response_tokens():
    from verl.workers.utils.padding import left_right_2_no_padding

    full_lengths, response_lengths = [206, 130, 40, 7], [78, 30, 20, 0]
    width, prompt_width = 206, 128
    attention = torch.zeros(4, width, dtype=torch.long)
    dense = torch.zeros(4, width - prompt_width)
    responses = [_response_mask_row(n) for n in response_lengths]
    responses[0][-1] = 0  # A real trailing token, not right padding.
    responses[1].zero_()  # Entirely unsupervised response still has a length.
    for i, (total, n) in enumerate(zip(full_lengths, response_lengths, strict=True)):
        attention[i, prompt_width - (total - n) : prompt_width + n] = 1
        dense[i, :n] = responses[i]
    batch = left_right_2_no_padding(
        _tensor_dict(
            {
                "input_ids": torch.arange(width).repeat(4, 1),
                "position_ids": torch.arange(width).repeat(4, 1),
                "attention_mask": attention,
                "response_mask": dense,
            },
            batch_size=[4],
        )
    )
    packed_dense = _loss_mask_for_packing(batch, batch["input_ids"])
    batch["loss_mask"] = torch.nested.as_nested_tensor(responses, layout=torch.jagged)
    packed_nested = _loss_mask_for_packing(batch, batch["input_ids"])
    assert torch.equal(packed_dense.offsets(), packed_nested.offsets())
    assert torch.equal(packed_dense.values(), packed_nested.values())
    for row, total, response in zip(
        packed_dense.unbind(), full_lengths, responses, strict=True
    ):
        assert torch.equal(
            row, torch.cat([response.new_zeros(total - response.numel()), response])
        )


@pytest.mark.parametrize("attention", [None, torch.ones(1, 6), torch.ones(1, 3)])
def test_dense_mask_rejects_missing_or_mismatched_attention(attention):
    ids = _full_input_ids([5])
    batch = _tensor_dict({"loss_mask": torch.tensor([[1.0, 0.0, 1.0, 0.0]])}, [1])
    if attention is not None:
        batch["attention_mask"] = attention
    with pytest.raises(
        ValueError, match="Dense loss mask requires matching full attention_mask"
    ):
        _loss_mask_for_packing(batch, ids)


def test_response_longer_than_input_is_rejected():
    """A response mask longer than its input sequence is a hard error, not silent."""
    full_lengths = [40]
    input_ids = _full_input_ids(full_lengths)
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(64, dtype=torch.float32)], layout=torch.jagged
    )
    micro_batch = _tensor_dict({"input_ids": input_ids, "loss_mask": loss_mask}, batch_size=[1])
    with pytest.raises(ValueError, match="tokens but packed input"):
        _loss_mask_for_packing(micro_batch, input_ids)
