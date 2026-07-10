# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import torch


def test_packed_verl_batch_keeps_configured_mtp_path_enabled():
    from megatron.lite.model.deepseek_v4.lite.protocol import _prepare_packed_batch_kwargs
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.contracts import PackedBatch

    tokens = torch.arange(8, dtype=torch.long)
    batch = PackedBatch(
        input_ids=tokens,
        labels=tokens.clone(),
        loss_mask=torch.ones_like(tokens, dtype=torch.float32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
    )
    kwargs = _prepare_packed_batch_kwargs(SimpleNamespace(ps=ParallelState()), batch)
    assert kwargs["enable_mtp"] is True
