# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed loss-mask fallback contract: the padded->packed relocation is only
valid for a contiguous all-ones response prefix; interior zeros (multi-turn /
tool-use masks) must fail loudly instead of silently training the wrong tokens.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from verl_mlite.engine.mlite_engine import MegatronLiteEngine

pytestmark = pytest.mark.mlite


def _packed_input_ids(rows):
    return torch.nested.as_nested_tensor(
        [torch.arange(n, dtype=torch.long) for n in rows], layout=torch.jagged
    )


def test_contiguous_response_mask_relocates_to_packed_tail():
    input_ids = _packed_input_ids([6, 5])
    micro = TensorDict(
        {"loss_mask": torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32)},
        batch_size=[2],
    )
    out = MegatronLiteEngine._loss_mask_for_packing(micro, input_ids)
    rows = list(out.unbind(0))
    assert rows[0].tolist() == [0, 0, 0, 1, 1, 1]  # 3 response tokens at the tail
    assert rows[1].tolist() == [0, 0, 0, 1, 1]


def test_interior_zero_mask_raises_instead_of_misaligning():
    # Multi-turn shape: [1,1,0,0,1,1] (sum=4). The old fallback copied
    # [1,1,0,0] onto the packed tail — training observation tokens and
    # skipping half the real response. Must raise now.
    input_ids = _packed_input_ids([6])
    micro = TensorDict(
        {"loss_mask": torch.tensor([[1, 1, 0, 0, 1, 1]], dtype=torch.float32)},
        batch_size=[1],
    )
    with pytest.raises(ValueError, match="contiguous all-ones prefix"):
        MegatronLiteEngine._loss_mask_for_packing(micro, input_ids)


def test_nested_mask_passthrough_unchanged():
    input_ids = _packed_input_ids([4])
    nested = torch.nested.as_nested_tensor(
        [torch.tensor([0.0, 1.0, 0.0, 1.0])], layout=torch.jagged
    )
    micro = TensorDict({"loss_mask": nested}, batch_size=[1])
    out = MegatronLiteEngine._loss_mask_for_packing(micro, input_ids)
    assert out is nested  # nested masks are position-aligned already: no fallback
