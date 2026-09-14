# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Communication must preserve the configured dtype in every send/recv mode."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from megatron.lite.primitive.parallel import pipeline as pl


@pytest.mark.parametrize('dtype', [None, torch.float32])
@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('batched', [False, True])
def test_pipeline_wire_dtype(dtype, dynamic, batched):
    # These FP32 low bits must survive both forward and backward transport.
    value = torch.tensor([1.000123, -0.50317, 3.141592]).reshape(3, 1, 1)
    expected_dtype = torch.bfloat16 if dtype is None else dtype
    expected = value.to(expected_dtype)
    sent = []
    request = SimpleNamespace(wait=lambda: None)

    def send(tensor, peer, group=None):
        sent.append(tensor.clone())
        return request

    def receive(tensor, peer, group=None):
        assert tensor.dtype == expected_dtype, 'PP_RECV_DTYPE'
        tensor.copy_(expected)
        return request

    def batch(ops):
        return [op.op(op.tensor, op.peer, op.group) for op in ops]

    ps = SimpleNamespace(pp_group=None, pp_next_rank=1, pp_prev_rank=1)
    with patch.object(pl.dist, 'get_rank', return_value=0), patch.object(
        pl.dist,
        'P2POp',
        side_effect=lambda op, tensor, peer, group: SimpleNamespace(
            op=op, tensor=tensor, peer=peer, group=group
        ),
    ), patch.object(pl.dist, 'isend', side_effect=send), patch.object(
        pl.dist, 'irecv', side_effect=receive
    ), patch.object(
        pl.dist, 'batch_isend_irecv', side_effect=batch
    ), patch.object(
        pl, '_pipeline_device', return_value=torch.device('cpu')
    ), patch.object(
        pl, '_communicate_shapes', return_value=(value.shape, value.shape)
    ):
        fwd, bwd = pl._send_recv_pipeline(
            value,
            value,
            True,
            True,
            ps,
            value.shape,
            batch_p2p=batched,
            dynamic_shape=dynamic,
            pipeline_dtype=dtype,
        )
    assert len(sent) == 2, 'PP_BIDIRECTIONAL_SEND'
    for actual in [*sent, fwd, bwd]:
        assert actual.dtype == expected_dtype, 'PP_WIRE_DTYPE'
        torch.testing.assert_close(
            actual, expected, atol=0, rtol=0, msg='PP_WIRE_EXACT'
        )
    assert fwd.requires_grad
