# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Router-replay helpers owned by the VERL-to-MLite connector."""

import torch


def build_r3_replay_mask(
    input_ids: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """Mark causal rows whose recorded routes affect response log probabilities.

    Rollout has no recorded route for the final response token because its logits
    are not consumed. When a sample has a response, replay every preceding model
    row; samples without a response remain entirely native.
    """
    if not getattr(input_ids, "is_nested", False):
        raise TypeError("R3 router replay requires jagged input_ids")

    total_lens = input_ids.offsets().diff()
    response_lens = response_mask.sum(dim=-1).to(
        device=total_lens.device, dtype=total_lens.dtype
    )
    if response_lens.numel() != total_lens.numel():
        raise ValueError(
            "R3 response_mask batch size must match jagged input_ids: "
            f"got {response_lens.numel()} and {total_lens.numel()}"
        )

    replay_lens = torch.where(
        response_lens > 0, total_lens - 1, torch.zeros_like(total_lens)
    )
    suffix_lens = total_lens - replay_lens
    values = torch.tensor([True, False], dtype=torch.bool, device=total_lens.device)
    values = values.repeat(total_lens.numel())
    counts = torch.stack((replay_lens, suffix_lens), dim=1).flatten()
    mask_values = torch.repeat_interleave(values, counts)
    return torch.nested.nested_tensor_from_jagged(
        mask_values, offsets=input_ids.offsets()
    )
