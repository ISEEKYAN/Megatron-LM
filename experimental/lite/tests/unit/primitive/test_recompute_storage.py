# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.lite.primitive import recompute


def test_discard_and_recompute_output_storage_uses_the_existing_autograd_graph(
    monkeypatch,
):
    monkeypatch.setattr(
        recompute,
        "_get_share_storage",
        lambda: lambda dst, src: dst.set_(src),
    )
    x = torch.tensor([2.0], requires_grad=True)
    output = x.square()
    downstream = output * 3.0

    recompute.discard_and_recompute_output_storage(
        output, downstream, torch.square, x
    )

    assert output.untyped_storage().nbytes() == 0
    downstream.sum().backward()
    torch.testing.assert_close(output, torch.tensor([4.0]))
    torch.testing.assert_close(x.grad, torch.tensor([12.0]))
