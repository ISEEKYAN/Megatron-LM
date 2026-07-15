# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit test for THD dynamic-batch P2P shapes in the 1F1B schedule.

Under THD + dynamic batching every micro-batch packs a different token count, so
the inter-stage hidden — and its P2P recv buffer — is a different shape per
micro-batch. The legacy code sized ONE fixed buffer from the first batch, which
truncated the recv of later, larger micro-batches (NCCL size mismatch -> hang).

These tests drive the *real* ``_1f1b_schedule`` on a single rank with the actual
dist transfer mocked out, asserting that each recv site is handed the exact shape
of its own micro-batch (off-by-one in the recv<->mb map = silent deadlock on GPU).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.primitive.parallel import pipeline as pl


def _make_ps(pp_size: int, pp_rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        pp_size=pp_size,
        pp_rank=pp_rank,
        pp_is_first=(pp_rank == 0),
        pp_is_last=(pp_rank == pp_size - 1),
        pp_prev_rank=pp_rank - 1,
        pp_next_rank=pp_rank + 1,
        pp_group=None,
    )


class _MockModel(torch.nn.Module):
    """Records the input_tensor set on it so we can catch a None/garbage input."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.weight = torch.nn.Parameter(torch.ones(hidden, hidden))
        self._input_tensor = None

    def set_input_tensor(self, t):
        self._input_tensor = t


def _run_schedule(pp_size, pp_rank, seq_lens, hidden=8):
    """Run _1f1b_schedule for one rank; return (recorded_fwd, recorded_bwd).

    dist transfer is mocked: recv returns the buffer it was handed (proving the
    caller sized it correctly). We record every recv's concrete shape.
    """
    ps = _make_ps(pp_size, pp_rank)
    num_mb = len(seq_lens)
    batches = [{"S": s} for s in seq_lens]

    def shape_fn(batch):
        return (int(batch["S"]), 1, hidden)

    fwd_shapes = [shape_fn(b) for b in batches]

    model = _MockModel(hidden)
    recorded_fwd: list[tuple[int, ...]] = []
    recorded_bwd: list[tuple[int, ...]] = []

    def fake_srp(
        send_fwd, send_bwd, recv_fwd, recv_bwd, ps_, tensor_shape,
        *, fwd_recv_buf=None, bwd_recv_buf=None, batch_p2p=True, clone_recv=False,
    ):
        fwd_out = None
        bwd_out = None
        if recv_fwd:
            assert fwd_recv_buf is not None, "recv_fwd must be handed a pre-sized buffer"
            recorded_fwd.append(tuple(fwd_recv_buf.shape))
            fwd_recv_buf.grad = None
            fwd_recv_buf.requires_grad_()
            fwd_out = fwd_recv_buf
        if recv_bwd:
            assert bwd_recv_buf is not None, "recv_bwd must be handed a pre-sized buffer"
            recorded_bwd.append(tuple(bwd_recv_buf.shape))
            # a grad tensor matching the sent-forward hidden of this mb
            bwd_out = torch.ones_like(bwd_recv_buf)
        return fwd_out, bwd_out

    def forward_step_fn(m, batch):
        s = int(batch["S"])
        if ps.pp_is_first:
            # first stage fabricates its own hidden from the (packed) input
            base = torch.ones(s, 1, hidden, requires_grad=True)
        else:
            inp = unwrap_model(m)._input_tensor
            assert inp is not None, "middle/last stage forwarded with a None input"
            assert tuple(inp.shape) == (s, 1, hidden), (
                f"input shape {tuple(inp.shape)} != expected {(s, 1, hidden)} for mb S={s}"
            )
            base = inp
        # hidden carries this mb's exact inter-stage shape [S, 1, H]
        hidden_t = base * m.weight.sum()
        out = {"hidden_states": hidden_t}
        if ps.pp_is_last:
            out["loss"] = hidden_t.float().sum()
        return out

    # Force CPU allocation for the flat recv buffers (module hard-codes cuda).
    real_empty = torch.empty

    def cpu_empty(*a, **kw):
        kw.pop("device", None)
        return real_empty(*a, **kw)

    orig_srp = pl._send_recv_pipeline
    orig_empty = pl.torch.empty
    pl._send_recv_pipeline = fake_srp
    pl.torch.empty = cpu_empty
    try:
        pl._1f1b_schedule(
            forward_step_fn,
            model,
            iter([(b, None) for b in batches]),
            num_mb,
            SimpleNamespace(num_microbatches=num_mb),
            ps,
            fwd_shapes[0],
            shape_fn=shape_fn,
        )
    finally:
        pl._send_recv_pipeline = orig_srp
        pl.torch.empty = orig_empty

    return recorded_fwd, recorded_bwd, fwd_shapes


VARLEN = [5, 9, 3, 7]  # deliberately non-uniform, ascending & descending mix


@pytest.mark.parametrize("pp_rank", [1, 2])
def test_middle_stage_recv_shapes_match_each_microbatch(pp_rank):
    """Every fwd/bwd recv on a middle stage is sized for its own micro-batch."""
    recorded_fwd, recorded_bwd, fwd_shapes = _run_schedule(4, pp_rank, VARLEN)
    # A middle stage receives a forward input for every mb (in forward order)
    assert recorded_fwd == fwd_shapes, (recorded_fwd, fwd_shapes)
    # and a backward grad for every mb (backward processes oldest-first == fwd order)
    assert recorded_bwd == fwd_shapes, (recorded_bwd, fwd_shapes)


def test_last_stage_recv_shapes_match_each_microbatch():
    recorded_fwd, recorded_bwd, fwd_shapes = _run_schedule(4, 3, VARLEN)
    # last stage receives every forward input, sends no forward -> no bwd recv
    assert recorded_fwd == fwd_shapes, (recorded_fwd, fwd_shapes)
    assert recorded_bwd == [], recorded_bwd


def test_first_stage_recv_shapes_match_each_microbatch():
    recorded_fwd, recorded_bwd, fwd_shapes = _run_schedule(4, 0, VARLEN)
    # first stage never receives a forward input; it receives a bwd grad per mb
    assert recorded_fwd == [], recorded_fwd
    assert recorded_bwd == fwd_shapes, (recorded_bwd, fwd_shapes)


def test_legacy_fixed_shape_still_uniform_when_no_shape_fn():
    """Without shape_fn the schedule keeps the single fixed tensor_shape."""
    ps = _make_ps(4, 1)
    uniform = [6, 6, 6, 6]
    recorded_fwd, recorded_bwd, _ = _run_schedule(4, 1, uniform)
    assert recorded_fwd == [(6, 1, 8)] * 4
    assert recorded_bwd == [(6, 1, 8)] * 4
    del ps
