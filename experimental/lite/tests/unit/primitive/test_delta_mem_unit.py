# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""δ-mem primitive contract tests (skills/primitive/principle.md).

Reference semantics: declare-lab/delta-Mem ``deltamem/core/delta_impl.py``
(arXiv:2605.12357). The ``_reference_*`` functions below are verbatim-excerpted
(signature-adapted to standalone form) from that repository (CC-BY-4.0,
https://github.com/declare-lab/delta-Mem) so the contract is bitwise against the
pinned upstream math on fp32 CPU, self-contained in this file.

Contract layers:
1. bitwise fp32 CPU vs the reference excerpts (projections + sequential scan);
2. equation-level (Eq. 4-12) naive re-derivation at 1e-6 — guards against
   copying a misunderstanding verbatim;
3. invariants: read-before-write, mask freezes state + zeroes reads, λ=1 ⇒
   identity (not decay), zero state ⇒ bitwise-zero steer, scan == step decode,
   SSW one-write-per-message + silent per-token fallback, init contract,
   trainable-param formula, float64 gradcheck;
4. single-GPU proxy at the upstream 1e-5 envelope (skipped without CUDA).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from megatron.lite.primitive.modules.delta_mem import (
    DeltaMemConfig,
    DeltaMemory,
    delta_mem_scaling,
    normalize_delta_mem_config,
)

pytestmark = pytest.mark.mlite

HIDDEN = 16
QUERY_OUT = 24
OUTPUT_OUT = 16
RANK = 4


# --- reference excerpts (declare-lab/delta-Mem, deltamem/core/delta_impl.py) ---


def _reference_sequence_projections(
    hidden_states, memory_q_proj, memory_k_proj, memory_v_proj, beta_proj, beta_bias,
    *, num_states, rank,
):
    """Excerpt of ``_memory_sequence_projections`` + ``_normalize_memory_projection``
    (couple_lambda=True, normalize_qk=True, rankwise_gates=True)."""
    state_read_dim = num_states * rank
    gate_dim = state_read_dim

    def normalize(projected):
        if num_states > 1:
            projected = projected.view(*projected.shape[:-1], num_states, rank)
            projected = torch.tanh(projected)
            projected = F.normalize(projected, dim=-1, eps=1e-6)
            return projected.reshape(*projected.shape[:-2], state_read_dim)
        projected = torch.tanh(projected)
        return F.normalize(projected, dim=-1, eps=1e-6)

    packed_gates = F.linear(hidden_states, beta_proj)
    packed_memory_weight = torch.cat([memory_q_proj, memory_k_proj, memory_v_proj], dim=0)
    packed_memory = F.linear(hidden_states, packed_memory_weight)
    memory_q, memory_k, memory_v = torch.split(
        packed_memory, [state_read_dim, state_read_dim, state_read_dim], dim=-1
    )
    memory_q = normalize(memory_q)
    memory_k = normalize(memory_k)
    beta = torch.sigmoid(
        packed_gates + beta_bias.view(*([1] * (hidden_states.dim() - 1)), gate_dim)
    ).unsqueeze(-1)
    lam = 1.0 - beta
    return memory_q, memory_k, memory_v, beta, lam


def _reference_affine_scan_torch(
    state, memory_q_seq, memory_k_seq, memory_v_seq, keep_seq, erase_seq, write_seq,
    token_mask=None,
):
    """Verbatim excerpt of ``_memory_affine_scan_torch``."""
    batch_size, seq_len, _ = memory_q_seq.shape
    current_state = state
    read_steps = []

    for token_idx in range(seq_len):
        q_t = memory_q_seq[:, token_idx, :]
        k_t = memory_k_seq[:, token_idx, :]
        v_t = memory_v_seq[:, token_idx, :]
        keep_t = keep_seq[:, token_idx, :].unsqueeze(-1)
        erase_t = erase_seq[:, token_idx, :].unsqueeze(-1)
        write_t = write_seq[:, token_idx, :].unsqueeze(-1)

        read_t = torch.einsum("bij,bj->bi", current_state, q_t)

        if token_mask is not None:
            valid = token_mask[:, token_idx].view(batch_size, 1)
            read_t = read_t * valid.to(dtype=read_t.dtype)

        pred_t = torch.einsum("bij,bj->bi", current_state, k_t)
        write_outer = v_t.unsqueeze(-1) * k_t.unsqueeze(1)
        pred_outer = pred_t.unsqueeze(-1) * k_t.unsqueeze(1)
        next_state = keep_t * current_state - erase_t * pred_outer + write_t * write_outer

        if token_mask is not None:
            valid_state = token_mask[:, token_idx].view(batch_size, 1, 1).to(dtype=next_state.dtype)
            current_state = next_state * valid_state + current_state * (1.0 - valid_state)
        else:
            current_state = next_state

        read_steps.append(read_t)

    reads = torch.stack(read_steps, dim=1)
    return current_state, reads


# --- helpers ---


def _make_module(rank=RANK, num_states=1, granularity="token", seed=0, output_init="zero"):
    torch.manual_seed(seed)
    return DeltaMemory(
        HIDDEN,
        QUERY_OUT,
        OUTPUT_OUT,
        DeltaMemConfig(
            rank=rank,
            num_states=num_states,
            write_granularity=granularity,
            output_init=output_init,
        ),
    )


def _random_inputs(batch=2, seq=7, seed=1, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq, HIDDEN, generator=gen, dtype=dtype)
    mask = torch.ones(batch, seq, dtype=torch.bool)
    mask[0, -2:] = False  # padding tail on sample 0
    return x, mask


# --- 1. bitwise vs reference excerpts ---


def test_projections_bitwise_match_reference_excerpt():
    for num_states in (1, 4):
        module = _make_module(num_states=num_states, seed=3)
        x, _ = _random_inputs(seed=4)
        ours = module.project(x)
        theirs = _reference_sequence_projections(
            x,
            module.memory_q_proj,
            module.memory_k_proj,
            module.memory_v_proj,
            module.beta_proj,
            module.beta_bias,
            num_states=num_states,
            rank=RANK,
        )
        for mine, ref in zip(ours, theirs):
            assert torch.equal(mine, ref)


def test_scan_bitwise_matches_reference_excerpt():
    module = _make_module(seed=5)
    x, mask = _random_inputs(seed=6)
    q, k, v, beta, lam = module.project(x)
    state0 = module.init_state(x.size(0), dtype=x.dtype)
    keep, erase, write = module._update_coefficients(beta, lam)
    for token_mask in (None, mask):
        ref_state, ref_reads = _reference_affine_scan_torch(
            state0, q, k, v, keep, erase, write, token_mask=token_mask
        )
        our_state, our_reads = module.scan(state0, q, k, v, beta, lam, token_mask=token_mask)
        assert torch.equal(our_state, ref_state)
        assert torch.equal(our_reads, ref_reads)


def test_msw_scan_bitwise_matches_reference_plumbing():
    # Multi-head-state path: reference flattens [B,N,r,r] -> [B*N,r,r], permutes the
    # per-sub-state projections, scans, then concatenates reads back to N*r.
    n = 4
    module = _make_module(num_states=n, seed=7)
    x, mask = _random_inputs(seed=8)
    b, t = x.shape[:2]
    q, k, v, beta, lam = module.project(x)
    state0 = module.init_state(b, dtype=x.dtype)
    keep, erase, write = module._update_coefficients(beta, lam)

    q_f = q.view(b, t, n, RANK).permute(0, 2, 1, 3).reshape(b * n, t, RANK)
    k_f = k.view(b, t, n, RANK).permute(0, 2, 1, 3).reshape(b * n, t, RANK)
    v_f = v.view(b, t, n, RANK).permute(0, 2, 1, 3).reshape(b * n, t, RANK)
    keep_f = keep.permute(0, 2, 1, 3).reshape(b * n, t, RANK)
    erase_f = erase.permute(0, 2, 1, 3).reshape(b * n, t, RANK)
    write_f = write.permute(0, 2, 1, 3).reshape(b * n, t, RANK)
    mask_f = mask.unsqueeze(1).expand(b, n, t).reshape(b * n, t)
    ref_state, ref_reads = _reference_affine_scan_torch(
        state0.reshape(b * n, RANK, RANK), q_f, k_f, v_f, keep_f, erase_f, write_f,
        token_mask=mask_f,
    )
    ref_state = ref_state.reshape(b, n, RANK, RANK)
    ref_reads = ref_reads.reshape(b, n, t, RANK).permute(0, 2, 1, 3).reshape(b, t, n * RANK)

    our_state, our_reads = module.scan(state0, q, k, v, beta, lam, token_mask=mask)
    assert torch.equal(our_state, ref_state)
    assert torch.equal(our_reads, ref_reads)


# --- 2. equation-level naive re-derivation (Eq. 4-12) ---


def test_naive_equation_semantics():
    module = _make_module(seed=9)
    x, _ = _random_inputs(batch=1, seq=5, seed=10)
    q, k, v, beta, lam = module.project(x)
    state0 = module.init_state(1, dtype=x.dtype)
    our_state, our_reads = module.scan(state0, q, k, v, beta, lam)

    s = state0[0].double()
    for t_idx in range(x.size(1)):
        q_t = q[0, t_idx].double()
        k_t = k[0, t_idx].double()
        v_t = v[0, t_idx].double()
        beta_t = beta[0, t_idx, :, 0].double()
        lam_t = 1.0 - beta_t
        read_t = s @ q_t  # Eq. 6: read BEFORE write
        assert torch.allclose(our_reads[0, t_idx].double(), read_t, atol=1e-6)
        # Eq. 10: S = Diag(λ)S + Diag(β)(v − Sk)kᵀ
        s = torch.diag(lam_t) @ s + torch.diag(beta_t) @ torch.outer(v_t - s @ k_t, k_t)
    assert torch.allclose(our_state[0].double(), s, atol=1e-6)

    # Eq. 4-5 shapes/normalization: unit-norm q/k rows, β = σ(W_β x + b) in (0, 1)
    assert torch.allclose(q.norm(dim=-1), torch.ones_like(q.norm(dim=-1)), atol=1e-4)
    assert torch.all((beta > 0) & (beta < 1))
    assert torch.equal(lam, 1.0 - beta)


# --- 3. invariants ---


def test_scan_equals_stepwise_decode():
    module = _make_module(seed=11)
    # Nonzero Δ-heads so (Δq, Δo) carry the reads: with the default zero init the
    # steer assertions would be vacuously zeros-vs-zeros, and a read-after-write
    # bug would pass unnoticed (reads never feed the state update).
    gen = torch.Generator().manual_seed(30)
    with torch.no_grad():
        module.delta_q_proj.copy_(torch.randn(QUERY_OUT, RANK, generator=gen))
        module.delta_o_proj.copy_(torch.randn(OUTPUT_OUT, RANK, generator=gen))
    x, _ = _random_inputs(batch=2, seq=6, seed=12)
    state = module.init_state(2, dtype=x.dtype)
    dq_full, do_full, state_full = module(x, state)
    assert torch.count_nonzero(dq_full) > 0  # non-vacuity guard
    assert torch.count_nonzero(do_full) > 0

    state_step = module.init_state(2, dtype=x.dtype)
    dq_steps, do_steps = [], []
    for t_idx in range(x.size(1)):
        dq_t, do_t, state_step = module(x[:, t_idx : t_idx + 1], state_step)
        dq_steps.append(dq_t)
        do_steps.append(do_t)
    # Tolerance justification (recorded downgrade from the bitwise target): the
    # chunked path projects [B,T,H] in one gemm, the decode path [B,1,H] per step
    # — same math, different reduction shapes, so end-to-end bitwise equality is
    # not guaranteed across gemm kernels. Bitwise coverage of the scan itself
    # (identical inputs) lives in test_scan_bitwise_matches_reference_excerpt and
    # test_msw_chunked_scan_composes_bitwise_with_carried_state.
    assert torch.allclose(state_full, state_step, atol=1e-6, rtol=0)
    assert torch.allclose(dq_full, torch.cat(dq_steps, dim=1), atol=1e-6, rtol=0)
    assert torch.allclose(do_full, torch.cat(do_steps, dim=1), atol=1e-6, rtol=0)


def test_zero_state_zero_steer_bitwise():
    module = _make_module(seed=13, output_init="base_slice_fixed")
    gen = torch.Generator().manual_seed(14)
    module.base_slice_init_(
        torch.randn(QUERY_OUT, HIDDEN, generator=gen),
        torch.randn(OUTPUT_OUT, HIDDEN, generator=gen),
    )
    x, _ = _random_inputs(batch=2, seq=1, seed=15)
    state0 = module.init_state(2, dtype=x.dtype)
    # Read-before-write: at t=0 the read comes from S_0 = 0, so the steer is
    # bitwise zero even with writes enabled and non-zero Δ-heads.
    dq, do, state1 = module(x, state0)
    assert torch.count_nonzero(dq) == 0
    assert torch.count_nonzero(do) == 0
    assert torch.count_nonzero(state1) > 0  # the write itself happened

    # Frozen zero state: reads stay zero for any length.
    dq, do, _ = module(_random_inputs(batch=2, seq=5, seed=16)[0], state0, write_enabled=False)
    assert torch.count_nonzero(dq) == 0
    assert torch.count_nonzero(do) == 0


def test_mask_freezes_state_and_zeroes_reads():
    module = _make_module(seed=17)
    x, _ = _random_inputs(batch=1, seq=2, seed=18)
    q, k, v, beta, lam = module.project(x)
    state0 = module.init_state(1, dtype=x.dtype)
    mask = torch.tensor([[True, False]])
    state_masked, reads = module.scan(state0, q, k, v, beta, lam, token_mask=mask)
    state_first_only, _ = module.scan(
        state0, q[:, :1], k[:, :1], v[:, :1], beta[:, :1], lam[:, :1]
    )
    assert torch.equal(state_masked, state_first_only)
    assert torch.count_nonzero(reads[:, 1]) == 0


def test_unit_keep_gate_is_identity_not_decay():
    module = _make_module(seed=19)
    x, _ = _random_inputs(batch=1, seq=3, seed=20)
    q, k, v, _, _ = module.project(x)
    state0 = torch.randn(1, RANK, RANK)
    ones = torch.ones(1, x.size(1), RANK)
    zeros = torch.zeros(1, x.size(1), RANK)
    # β = 0 ⇒ keep = λ = 1, erase = write = 0 ⇒ S_t = S_{t-1} bitwise.
    final_state, reads = module.affine_scan_torch(state0, q, k, v, ones, zeros, zeros)
    assert torch.equal(final_state, state0)
    assert torch.equal(reads[:, 0], torch.einsum("bij,bj->bi", state0, q[:, 0]))


def test_init_contract():
    module = _make_module(seed=21, output_init="base_slice_fixed")
    # W_β weight zero, bias −1.5 ⇒ β = σ(−1.5) exactly, everywhere.
    assert torch.count_nonzero(module.beta_proj) == 0
    assert torch.equal(module.beta_bias, torch.full((RANK,), -1.5))
    x, _ = _random_inputs(seed=22)
    beta = module.project(x)[3]
    expected = torch.sigmoid(torch.full_like(beta, -1.5))
    assert torch.equal(beta, expected)
    # Δ-heads zero before base_slice_init_ (S_0 = 0 alone guarantees zero steer).
    assert torch.count_nonzero(module.delta_q_proj) == 0
    assert torch.count_nonzero(module.delta_o_proj) == 0
    # base_slice_fixed: first min(ref_width, rank, in) base columns, col-L2-normed
    # in float32 (eps 1e-6), × online_gain, remaining columns zero.
    gen = torch.Generator().manual_seed(23)
    base_q = torch.randn(QUERY_OUT, HIDDEN, generator=gen)
    base_o = torch.randn(OUTPUT_OUT, HIDDEN, generator=gen)
    module.base_slice_init_(base_q, base_o)
    width = min(8, RANK, HIDDEN)
    for head, base in ((module.delta_q_proj, base_q), (module.delta_o_proj, base_o)):
        expected_slice = F.normalize(base[:, :width].float(), dim=0, eps=1e-6) * 0.05
        assert torch.equal(head[:, :width], expected_slice)
        assert torch.count_nonzero(head[:, width:]) == 0
    # kaiming_uniform(a=√5) on the W^m trio, pinned by RNG replay: construction
    # after manual_seed consumes exactly the three kaiming draws, in q/k/v order
    # (torch.empty and the zero/const fills consume no RNG).
    torch.manual_seed(21)
    for name in ("memory_q_proj", "memory_k_proj", "memory_v_proj"):
        expected_w = torch.empty(RANK, HIDDEN)
        torch.nn.init.kaiming_uniform_(expected_w, a=math.sqrt(5))
        assert torch.equal(getattr(module, name), expected_w)


def test_trainable_param_count_formula():
    # Per layer, branches (q, o): N·r·(3d) memory trio + N·r·d + N·r gate + N·r·d_q
    # + N·r·d_out Δ-heads = N·r·(4d + d_q + d_out) + N·r. With d_out = d this is the
    # design note's r·(5d + d_q) + r at N=1 (4,866,336 on Qwen3-4B r=8, 36 layers).
    for n in (1, 4):
        module = _make_module(num_states=n)
        total = sum(p.numel() for p in module.parameters())
        assert all(p.requires_grad for p in module.parameters())
        expected = n * RANK * (4 * HIDDEN + QUERY_OUT + OUTPUT_OUT) + n * RANK
        assert total == expected
    # Design-note goldens at real Qwen3-4B dims (d=2560, d_q=4096, 36 layers),
    # verified on instantiated modules, not just literal arithmetic.
    per_layer = sum(
        p.numel()
        for p in DeltaMemory(2560, 4096, 2560, DeltaMemConfig(rank=8)).parameters()
    )
    assert per_layer * 36 == 4_866_336  # paper Appendix C: 4.87M (TSW/SSW)
    per_layer_msw = sum(
        p.numel()
        for p in DeltaMemory(2560, 4096, 2560, DeltaMemConfig(rank=8, num_states=4)).parameters()
    )
    assert per_layer_msw * 36 == 19_465_344  # paper Appendix C: 19.47M (MSW, N=4)


def test_message_mean_write_and_silent_token_fallback():
    module = _make_module(granularity="message", seed=25)
    gen = torch.Generator().manual_seed(99)
    with torch.no_grad():
        module.delta_q_proj.copy_(torch.randn(QUERY_OUT, RANK, generator=gen))
        module.delta_o_proj.copy_(torch.randn(OUTPUT_OUT, RANK, generator=gen))
    x, _ = _random_inputs(batch=1, seq=6, seed=26)
    state0 = module.init_state(1, dtype=x.dtype)
    message_ids = torch.tensor([[-1, 0, 0, 1, 1, 1]])

    dq, do, state_msg = module(x, state0, message_ids=message_ids)
    # One write per message: means of tokens {1,2} and {3,4,5} pushed through the
    # same projections, scanned over the 2-slot message axis.
    means = torch.stack([x[0, 1:3].mean(dim=0), x[0, 3:6].mean(dim=0)]).unsqueeze(0)
    mq, mk, mv, mbeta, mlam = module.project(means)
    expected_state, _ = module.scan(state0, mq, mk, mv, mbeta, mlam)
    assert torch.equal(state_msg, expected_state)
    # All token reads use the PRE-chunk state (state0 = 0 here ⇒ zero steer).
    assert torch.count_nonzero(dq) == 0
    assert torch.count_nonzero(do) == 0

    # Silent fallback: no active id in the chunk ⇒ per-token writes, evolving reads
    # (reference `_message_write_inputs` returns None; an all-−1 chunk does NOT skip).
    token_module = _make_module(granularity="token", seed=25)
    token_module.load_state_dict(module.state_dict())
    dq_fb, do_fb, state_fb = module(x, state0, message_ids=torch.full_like(message_ids, -1))
    dq_tok, do_tok, state_tok = token_module(x, state0)
    assert torch.equal(state_fb, state_tok)
    assert torch.equal(dq_fb, dq_tok)
    assert torch.equal(do_fb, do_tok)
    assert torch.count_nonzero(dq_fb) > 0  # evolving-state reads, unlike SSW above


def test_write_disabled_reads_frozen_state():
    module = _make_module(seed=27)
    gen = torch.Generator().manual_seed(33)
    with torch.no_grad():
        module.delta_q_proj.copy_(torch.randn(QUERY_OUT, RANK, generator=gen))
        module.delta_o_proj.copy_(torch.randn(OUTPUT_OUT, RANK, generator=gen))
    x, mask = _random_inputs(seed=28)
    state0 = torch.randn(2, RANK, RANK)
    q = module.project(x)[0]
    dq, do, state_out = module(x, state0, token_mask=mask, write_enabled=False)
    assert state_out is state0  # no write, same tensor
    assert torch.count_nonzero(dq) > 0
    # Naive frozen-read re-derivation (float64), independent of token_reads:
    # r_t = S q^m_t, masked positions read zero.
    reads_naive = torch.zeros(2, x.size(1), RANK, dtype=torch.float64)
    for b in range(2):
        for t_idx in range(x.size(1)):
            if mask[b, t_idx]:
                reads_naive[b, t_idx] = state0[b].double() @ q[b, t_idx].double()
    assert torch.allclose(
        dq.double(), reads_naive @ module.delta_q_proj.double().T * module.scale, atol=1e-6
    )
    assert torch.allclose(
        do.double(), reads_naive @ module.delta_o_proj.double().T * module.scale, atol=1e-6
    )


def test_config_normalize_and_gating():
    cfg = normalize_delta_mem_config(
        {"rank": 8, "num_state_heads": 4, "memory_write_granularity": "message_mean"}
    )
    assert cfg.enabled and cfg.num_states == 4 and cfg.write_granularity == "message"
    assert cfg.scale == delta_mem_scaling(8, 16.0) == 2.0
    assert not normalize_delta_mem_config({"enabled": False, "rank": 8}).enabled
    assert not normalize_delta_mem_config(None).enabled
    with pytest.raises(TypeError, match="delta_mem config"):
        normalize_delta_mem_config(object())
    with pytest.raises(ValueError, match="write_granularity"):
        normalize_delta_mem_config({"rank": 4, "write_granularity": "sentence_mean"})
    with pytest.raises(ValueError, match="rank > 0"):
        DeltaMemory(HIDDEN, QUERY_OUT, OUTPUT_OUT, DeltaMemConfig(rank=0))
    with pytest.raises(ValueError, match="branches"):
        DeltaMemConfig(rank=4, branches=("q", "k"))


def test_gradcheck_scan_float64():
    torch.manual_seed(29)
    b, t, r = 1, 3, 2
    state = torch.randn(b, r, r, dtype=torch.float64, requires_grad=True)
    q = torch.randn(b, t, r, dtype=torch.float64, requires_grad=True)
    k = torch.randn(b, t, r, dtype=torch.float64, requires_grad=True)
    v = torch.randn(b, t, r, dtype=torch.float64, requires_grad=True)
    gate = torch.rand(b, t, r, dtype=torch.float64, requires_grad=True)

    def run(state_, q_, k_, v_, gate_):
        final_state, reads = DeltaMemory.affine_scan_torch(
            state_, q_, k_, v_, 1.0 - gate_, gate_, gate_
        )
        return final_state, reads

    assert torch.autograd.gradcheck(run, (state, q, k, v, gate), atol=1e-6)


def test_msw_equation_level_naive_rederivation():
    # Non-self-referential MSW check: N != r so a transposed gate grouping cannot
    # hide, beta_proj randomized so β is not uniform, carried-over NONZERO state.
    n, r = 2, 3
    torch.manual_seed(41)
    module = DeltaMemory(HIDDEN, QUERY_OUT, OUTPUT_OUT, DeltaMemConfig(rank=r, num_states=n))
    gen = torch.Generator().manual_seed(42)
    with torch.no_grad():
        module.beta_proj.copy_(0.5 * torch.randn(n * r, HIDDEN, generator=gen))
    x = torch.randn(1, 4, HIDDEN, generator=gen)
    state0 = torch.randn(1, n, r, r, generator=gen)
    q, k, v, beta, lam = module.project(x)
    our_state, our_reads = module.scan(state0, q, k, v, beta, lam)
    frozen_reads = module.token_reads(state0, q)

    wq = module.memory_q_proj.double()
    wk = module.memory_k_proj.double()
    wv = module.memory_v_proj.double()
    wb = module.beta_proj.double()
    bias = module.beta_bias.double()
    s = state0[0].double().clone()
    for t_idx in range(x.size(1)):
        h = x[0, t_idx].double()
        beta_full = torch.sigmoid(wb @ h + bias)
        reads_t, frozen_t = [], []
        for i in range(n):
            rows = slice(i * r, (i + 1) * r)

            def norm_proj(w):
                z = torch.tanh((w @ h)[rows])
                return z / z.norm().clamp_min(1e-6)

            q_i, k_i = norm_proj(wq), norm_proj(wk)
            v_i = (wv @ h)[rows]
            beta_i = beta_full[rows]
            frozen_t.append(state0[0, i].double() @ q_i)
            reads_t.append(s[i] @ q_i)  # read BEFORE this position's write
            s[i] = torch.diag(1.0 - beta_i) @ s[i] + torch.diag(beta_i) @ torch.outer(
                v_i - s[i] @ k_i, k_i
            )
        assert torch.allclose(our_reads[0, t_idx].double(), torch.cat(reads_t), atol=1e-6)
        assert torch.allclose(frozen_reads[0, t_idx].double(), torch.cat(frozen_t), atol=1e-6)
    assert torch.allclose(our_state[0].double(), s, atol=1e-6)


def test_msw_chunked_scan_composes_bitwise_with_carried_state():
    # State re-entry below the projection layer: scanning chunk 2 from chunk 1's
    # final [B,N,r,r] state must equal the full scan bitwise (identical inputs).
    n = 4
    module = _make_module(num_states=n, seed=43)
    x, mask = _random_inputs(batch=2, seq=6, seed=44)
    q, k, v, beta, lam = module.project(x)
    state0 = torch.randn(2, n, RANK, RANK, generator=torch.Generator().manual_seed(45))
    full_state, full_reads = module.scan(state0, q, k, v, beta, lam, token_mask=mask)
    mid_state, reads_a = module.scan(
        state0, q[:, :3], k[:, :3], v[:, :3], beta[:, :3], lam[:, :3], token_mask=mask[:, :3]
    )
    end_state, reads_b = module.scan(
        mid_state, q[:, 3:], k[:, 3:], v[:, 3:], beta[:, 3:], lam[:, 3:], token_mask=mask[:, 3:]
    )
    assert torch.equal(full_state, end_state)
    assert torch.equal(full_reads, torch.cat([reads_a, reads_b], dim=1))


def test_ssw_single_token_decode_write():
    # D6 decode clause: one generated token carrying its message id becomes a
    # single-token "message mean" write into that id's slot; its read uses the
    # pre-write session state.
    module = _make_module(granularity="message", seed=61)
    gen = torch.Generator().manual_seed(62)
    with torch.no_grad():
        module.delta_q_proj.copy_(torch.randn(QUERY_OUT, RANK, generator=gen))
        module.delta_o_proj.copy_(torch.randn(OUTPUT_OUT, RANK, generator=gen))
    x = torch.randn(1, 1, HIDDEN, generator=gen)
    state0 = torch.randn(1, RANK, RANK, generator=gen)  # carried session state
    dq, do, state1 = module(x, state0, message_ids=torch.tensor([[2]]))

    means = torch.zeros(1, 3, HIDDEN)
    means[0, 2] = x[0, 0]  # slot 2 = the decode token itself; slots 0-1 inactive
    mmask = torch.tensor([[False, False, True]])
    mq, mk, mv, mbeta, mlam = module.project(means)
    expected_state, _ = module.scan(state0, mq, mk, mv, mbeta, mlam, token_mask=mmask)
    assert torch.equal(state1, expected_state)
    assert torch.count_nonzero(state1 - state0) > 0  # the write happened
    # Read from the PRE-write state (naive float64 re-derivation).
    q = module.project(x)[0]
    read_naive = state0[0].double() @ q[0, 0].double()
    assert torch.allclose(
        dq[0, 0].double(), module.delta_q_proj.double() @ read_naive * module.scale, atol=1e-6
    )


def test_ssw_mask_excludes_padding_from_message_means():
    # Message means pool ACTIVE tokens only: active = (id >= 0) AND unmasked
    # (reference _message_write_inputs).
    module = _make_module(granularity="message", seed=63)
    gen = torch.Generator().manual_seed(64)
    x = torch.randn(1, 4, HIDDEN, generator=gen)
    state0 = module.init_state(1, dtype=x.dtype)
    message_ids = torch.tensor([[0, 0, 0, -1]])
    mask = torch.tensor([[True, True, False, True]])  # token 2 is padding
    _, _, state1 = module(x, state0, token_mask=mask, message_ids=message_ids)

    mean_active = x[0, :2].mean(dim=0)  # tokens 0-1 only: id 0 ∧ unmasked
    mq, mk, mv, mbeta, mlam = module.project(mean_active.view(1, 1, HIDDEN))
    expected_state, _ = module.scan(state0, mq, mk, mv, mbeta, mlam)
    assert torch.equal(state1, expected_state)


def test_ssw_message_aligned_chunking_policy():
    # Contract chunking policy: message-ALIGNED chunks. Under it the session STATE
    # composes with whole-chunk ingestion; per-token READS intentionally differ
    # (D6 chunk-dependence: chunked ingestion lets later messages read earlier
    # messages' writes, whole-chunk ingestion reads only the pre-chunk state).
    module = _make_module(granularity="message", seed=71)
    gen = torch.Generator().manual_seed(72)
    with torch.no_grad():
        module.delta_q_proj.copy_(torch.randn(QUERY_OUT, RANK, generator=gen))
        module.delta_o_proj.copy_(torch.randn(OUTPUT_OUT, RANK, generator=gen))
    x = torch.randn(1, 5, HIDDEN, generator=gen)
    ids = torch.tensor([[0, 0, 1, 1, 1]])
    state0 = torch.randn(1, RANK, RANK, generator=gen)

    dq_whole, _, state_whole = module(x, state0, message_ids=ids)
    _, _, state_after_msg0 = module(x[:, :2], state0, message_ids=ids[:, :2])
    dq_chunk2, _, state_chunked = module(x[:, 2:], state_after_msg0, message_ids=ids[:, 2:])
    # Tolerance justification (recorded): chunking changes the means-tensor gemm
    # shapes ([1,2,H] whole vs [1,1,H]+[1,2,H] chunked), so composition is exact
    # only to the last ulp, not bitwise — same rationale as scan==decode above.
    assert torch.allclose(state_whole, state_chunked, atol=1e-6, rtol=0)
    # Chunk-dependence pinned: message-1 tokens read state_after_msg0 when chunked
    # vs state0 when whole — the steers must differ.
    assert torch.count_nonzero(dq_whole[:, 2:] - dq_chunk2) > 0


def test_one_hot_key_query_round_trip_elementary():
    # §6 oracle, below tanh/L2: drive the elementary scan directly. β=1 ⇒ keep=0,
    # erase=write=1 ⇒ S₁ = v e_jᵀ from S₀=0; reading with q = e_j recovers v bitwise.
    gen = torch.Generator().manual_seed(81)
    v = torch.randn(1, 1, RANK, generator=gen)
    ones = torch.ones(1, 1, RANK)
    zeros = torch.zeros(1, 1, RANK)
    for j in range(RANK):
        k = torch.zeros(1, 1, RANK)
        k[0, 0, j] = 1.0
        s1, _ = DeltaMemory.affine_scan_torch(torch.zeros(1, RANK, RANK), k, k, v, zeros, ones, ones)
        assert torch.equal(s1[0], torch.outer(v[0, 0], k[0, 0]))
        _, reads = DeltaMemory.affine_scan_torch(s1, k, k, torch.zeros_like(v), ones, zeros, zeros)
        assert torch.equal(reads[0, 0], v[0, 0])


def test_bf16_within_recorded_band():
    # bf16 layer of the bitwise-or-threshold contract. Observed max-abs deviation
    # vs the fp32 CPU path on these seeds: ~2e-2 for Δq/Δo, ~1e-2 for the state
    # (bf16 has ~3 decimal digits; the scan re-quantizes per step like the
    # reference, which keeps S in backbone dtype — design note D9). Band = 4×.
    module = _make_module(seed=51)
    gen = torch.Generator().manual_seed(52)
    with torch.no_grad():
        module.delta_q_proj.copy_(torch.randn(QUERY_OUT, RANK, generator=gen))
        module.delta_o_proj.copy_(torch.randn(OUTPUT_OUT, RANK, generator=gen))
    x, mask = _random_inputs(seed=53)
    dq32, do32, s32 = module(x, module.init_state(2, dtype=x.dtype), token_mask=mask)

    module16 = _make_module(seed=51).to(torch.bfloat16)
    module16.load_state_dict(
        {k_: v_.to(torch.bfloat16) for k_, v_ in module.state_dict().items()}
    )
    dq16, do16, s16 = module16(
        x.to(torch.bfloat16),
        module16.init_state(2, dtype=torch.bfloat16),
        token_mask=mask,
    )
    assert (dq16.float() - dq32).abs().max() < 8e-2
    assert (do16.float() - do32).abs().max() < 8e-2
    assert (s16.float() - s32).abs().max() < 4e-2


# --- 4. single-GPU proxy (upstream 1e-5 envelope) ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="single-GPU proxy needs CUDA")
def test_gpu_proxy_matches_cpu_fp32():
    module = _make_module(seed=31)
    x, mask = _random_inputs(seed=32)
    state0 = module.init_state(2, dtype=x.dtype)
    dq_cpu, do_cpu, state_cpu = module(x, state0, token_mask=mask)

    module_gpu = module.to("cuda")
    dq_gpu, do_gpu, state_gpu = module_gpu(
        x.cuda(), state0.cuda(), token_mask=mask.cuda()
    )
    assert torch.allclose(dq_cpu, dq_gpu.cpu(), atol=1e-5)
    assert torch.allclose(do_cpu, do_gpu.cpu(), atol=1e-5)
    assert torch.allclose(state_cpu, state_gpu.cpu(), atol=1e-5)
