# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.modules.lora import (
    GroupedLinearLoRA,
    LinearLoRA,
    SharedGroupedLinearLoRA,
    freeze_non_lora_params,
    lora_scaling,
    normalize_lora_config,
    olora_tail_factors,
    trainable_param_stats,
)

pytestmark = pytest.mark.mlite


def test_lora_config_aliases_and_trainable_param_accounting():
    cfg = normalize_lora_config({"enabled": True, "rank": 2, "alpha": 6, "targets": ["qkv", "fc2"]})

    assert cfg.enabled
    assert cfg.scale == 3.0
    assert cfg.targets() == {"linear_qkv", "linear_fc2"}
    assert cfg.targets_module("qkv")
    assert cfg.targets_module("linear_fc2")
    assert not normalize_lora_config({"enabled": False, "rank": 8}).enabled
    with pytest.raises(TypeError, match="LoRA config"):
        normalize_lora_config(object())

    class TinyAdapterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(3, 2)
            self.lora_adapter = nn.Linear(3, 2)

    model = TinyAdapterModel()
    stats = freeze_non_lora_params(model)

    assert stats["lora_tensors"] == 2
    assert stats["frozen_tensors"] == 2
    assert not model.base.weight.requires_grad
    assert model.lora_adapter.weight.requires_grad
    assert trainable_param_stats(model) == {
        "trainable_tensors": 2,
        "trainable_numel": model.lora_adapter.weight.numel() + model.lora_adapter.bias.numel(),
    }


def test_rslora_uses_sqrt_rank_scaling():
    # standard LoRA: alpha/rank; rsLoRA: alpha/sqrt(rank). rank=4, alpha=8 -> 2.0 vs 4.0.
    assert lora_scaling(4, 8, use_rslora=False) == 2.0
    assert lora_scaling(4, 8, use_rslora=True) == 4.0
    assert lora_scaling(4, None, use_rslora=True) == 4 / (4**0.5)  # alpha defaults to rank

    cfg = normalize_lora_config({"rank": 4, "alpha": 8, "use_rslora": True})
    assert cfg.use_rslora
    assert cfg.scale == 4.0
    assert normalize_lora_config({"rank": 4, "alpha": 8}).scale == 2.0  # default off

    assert LinearLoRA(2, 1, rank=4, alpha=8).scale == 2.0
    assert LinearLoRA(2, 1, rank=4, alpha=8, use_rslora=True).scale == 4.0
    assert SharedGroupedLinearLoRA(2, 2, 2, rank=4, alpha=8, use_rslora=True).scale == 4.0
    assert GroupedLinearLoRA(2, 2, 2, rank=4, alpha=8, use_rslora=True).scale == 4.0

    # modules record use_rslora so the adapter round-trip can invert scale->alpha correctly
    assert LinearLoRA(2, 1, rank=4, alpha=8, use_rslora=True).use_rslora is True
    assert LinearLoRA(2, 1, rank=4, alpha=8).use_rslora is False


def test_rslora_forward_scales_delta_by_sqrt_rank():
    # identical adapter weights, rsLoRA output = sqrt(rank) x standard output.
    # alpha=8 != rank=4 so the std path is non-trivially scaled (2.0), not 1.0.
    torch.manual_seed(0)
    a = torch.randn(4, 3)
    b = torch.randn(2, 4)
    std = LinearLoRA(3, 2, rank=4, alpha=8, dropout=0.0)
    rs = LinearLoRA(3, 2, rank=4, alpha=8, dropout=0.0, use_rslora=True)
    with torch.no_grad():
        for layer in (std, rs):
            layer.lora_a.copy_(a)
            layer.lora_b.copy_(b)
    x = torch.randn(1, 3)
    torch.testing.assert_close(rs(x), std(x) * (4**0.5))


def test_olora_tail_factors_are_minor_subspace_without_sigma():
    # B0=U_-r, A0=V_-r^T from the SMALLEST r singular values, orthonormal (no Sigma scaling).
    torch.manual_seed(0)
    out_f, in_f, rank = 16, 12, 3
    w = torch.randn(out_f, in_f)
    u, _s, vh = torch.linalg.svd(w, full_matrices=False)  # singular values descending
    b0, a0 = olora_tail_factors(w, rank)

    assert b0.shape == (out_f, rank)
    assert a0.shape == (rank, in_f)
    # orthonormal => NO singular-value scaling injected (unlike MiLoRA's Sigma^1/2)
    torch.testing.assert_close(b0.t() @ b0, torch.eye(rank), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(a0 @ a0.t(), torch.eye(rank), atol=1e-4, rtol=1e-4)
    # spans the MINOR subspace (smallest r vectors), up to per-vector sign
    torch.testing.assert_close(b0.abs(), u[:, -rank:].abs(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(a0.abs(), vh[-rank:, :].abs(), atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError, match="exceeds"):
        olora_tail_factors(w, rank=13)


def test_linear_lora_olora_tail_preserves_init_output():
    # PiSSA-style residual: effective output unchanged at init, but adapter delta is NON-zero.
    torch.manual_seed(0)
    in_f, out_f, rank = 12, 16, 3
    layer = LinearLoRA(in_f, out_f, rank, alpha=6)  # scale = 6/3 = 2
    w = torch.randn(out_f, in_f)
    w0 = w.clone()
    x = torch.randn(4, in_f)
    base_out = x @ w0.t()

    layer.olora_tail_init_(w)  # sets lora_a/b, subtracts scale*B0@A0 from w in place

    effective = x @ w.t() + layer(x)  # residual base + adapter
    torch.testing.assert_close(effective, base_out, atol=1e-4, rtol=1e-4)
    assert layer.lora_b.abs().sum() > 0  # unlike standard zero-init B


def test_shared_grouped_olora_tail_preserves_every_expert_output():
    # one shared adapter; subtract same delta from each expert -> each expert output preserved.
    torch.manual_seed(0)
    n_exp, in_f, out_f, rank = 3, 12, 16, 2
    layer = SharedGroupedLinearLoRA(n_exp, in_f, out_f, rank, alpha=4)  # scale = 2
    ws = [torch.randn(out_f, in_f) for _ in range(n_exp)]
    w0s = [w.clone() for w in ws]

    layer.olora_tail_init_(ws)

    x = torch.randn(5, in_f)
    shared = (x @ layer.lora_a.t()) @ layer.lora_b.t() * layer.scale
    for w_after, w0 in zip(ws, w0s, strict=True):
        torch.testing.assert_close(x @ w_after.t() + shared, x @ w0.t(), atol=1e-4, rtol=1e-4)


def test_olora_tail_init_rejects_tp_sharded_weight():
    # tp>1 / partitioned surfaces need a distributed SVD; we guard against silent wrong init.
    # rank must be divisible by the partition size (4 % 2 == 0) so construction succeeds and
    # the guard — not the constructor — is what rejects the sharded surface.
    layer = LinearLoRA(12, 16, rank=4, alpha=6, rank_partitioned_a=True, rank_partition_size=2)
    with pytest.raises(NotImplementedError, match="tp=1"):
        layer.olora_tail_init_(torch.randn(16, 12))


def test_r3_router_replay_pins_selection_and_stays_differentiable():
    # R3 (arXiv:2606.02437 §3) via the upstream-PR#49-shaped RouterReplay registry:
    # RECORD captures the fresh top-k; REPLAY_FORWARD pins the SELECTION to it while
    # recomputing SCORES from live logits -> indices frozen, probabilities differentiable.
    from types import SimpleNamespace

    from megatron.lite.primitive.modules.router import (
        RouterReplay,
        RouterReplayAction,
        TopKRouter,
        attach_router_replay,
        detach_router_replay,
    )

    cfg = SimpleNamespace(
        num_experts_per_tok=2, num_experts=8, router_aux_loss_coef=0.0, hidden_size=16
    )
    ps = SimpleNamespace(tp_size=1, tp_group=None)
    # router_gating_linear routes through the TE GEMM when TE is installed, which is
    # CUDA-only — run on the GPU there, and on CPU only in TE-less dev environments.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router = TopKRouter(cfg, ps, compute_aux_loss=False).to(device)
    assert attach_router_replay(router) == 1
    torch.manual_seed(0)
    x = torch.randn(5, 16, device=device)

    with torch.no_grad():
        RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
        _, idx0 = router(x)
        recorded = RouterReplay.get_recorded_data()
        # no cursor advanced: bare-router records land under microbatch key 0
        assert len(recorded) == 1 and torch.equal(recorded[0][0], idx0)

        # perturb the gate (simulates the rollout->train policy drift that causes TIM)
        router.gate.weight.add_(torch.randn_like(router.gate.weight) * 3.0)
        RouterReplay.set_global_router_replay_action(None)
        _, idx_noreplay = router(x)
        assert not torch.equal(idx_noreplay, idx0)  # WITHOUT replay: top-k flips (the TIM)

        RouterReplay.set_replay_data([idx0])
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        scores_r, idx_r = router(x)
    assert torch.equal(idx_r, idx0)  # WITH replay: selection pinned despite the drift
    # Regression vs gathering from post-topk scatter-zeroed dense probs (upstream PR#49
    # bug): drifted replayed experts must get LIVE nonzero scores matching the router's
    # native normalization (post-softmax mode: softmax over the replayed-k logits).
    assert scores_r.min() > 0
    with torch.no_grad():
        manual_logits = torch.nn.functional.linear(x.float(), router.gate.weight.float())
        expected = torch.softmax(manual_logits.gather(1, idx0), dim=-1)
    torch.testing.assert_close(
        scores_r.float(), expected, atol=2e-2, rtol=2e-2  # loose: TE gemm vs fp32 matmul
    )

    router.gate.weight.requires_grad_(True)
    RouterReplay.set_replay_data([idx0])
    s, _ = router(x)
    # Post-softmax scores sum to 1 per token, so s.sum() is a CONSTANT and its gradient
    # to the gate is mathematically zero — the old s.sum().backward() check only passed
    # on softmax-backward FP noise, which flips to exactly 0.0 depending on allocator/
    # reduction state across tests. Use a non-uniform functional (distinct per-k weights)
    # whose gradient is genuinely nonzero, so this tests real grad flow to the gate.
    weights = torch.arange(1, s.size(-1) + 1, device=s.device, dtype=torch.float32)
    (s.float() * weights).sum().backward()
    assert router.gate.weight.grad is not None and router.gate.weight.grad.abs().sum() > 0

    # REPLAY_BACKWARD pops one queued tensor per (re)forward — exhaustion must fail
    # loudly. The FIFO fills at REPLAY time (not set_replay_data time) and only when
    # the chunk hook armed queue_backward_replays (full recompute re-runs the router).
    # Call router.forward directly: recompute re-forwards bypass Module.__call__ and
    # thus the chunk hooks, which would otherwise re-manage the flag and the action.
    RouterReplay.clear_global_indices()
    RouterReplay.set_replay_data([idx0])
    inst = RouterReplay.global_router_replay_instances[0]
    inst.queue_backward_replays = True
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    with torch.no_grad():
        router.forward(x)  # replayed forward queues exactly one backward entry
        assert len(inst.replay_backward_list) == 1
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
        _, idx_bwd = router.forward(x)  # the recompute re-forward consumes it
        assert torch.equal(idx_bwd, idx0)
        with pytest.raises(RuntimeError, match="exhausted"):
            router.forward(x)

    detach_router_replay(router)
    assert router.router_replay is None
    assert RouterReplay.global_router_replay_instances == []
    with torch.no_grad():
        _, idx_off = router(x)
    assert torch.equal(idx_off, idx_noreplay)  # replay fully off after detach


def test_r3_router_replay_sentinel_tokens_fall_back_to_live_routing():
    # Unmappable-routing masking contract (arXiv:2605.13779 §6.3): sentinel -1 marks tokens whose
    # rollout routes cannot be mapped to this batch — replay must keep their FRESH
    # selection AND scores (live routing, bitwise vs no replay) while mapped tokens
    # in the same pass still pin to the recorded indices. Never silently wrong:
    # zero-filled rows would otherwise replay as expert 0.
    from types import SimpleNamespace

    from megatron.lite.primitive.modules.router import (
        RouterReplay,
        RouterReplayAction,
        TopKRouter,
        attach_router_replay,
        detach_router_replay,
    )

    cfg = SimpleNamespace(
        num_experts_per_tok=2, num_experts=8, router_aux_loss_coef=0.0, hidden_size=16
    )
    ps = SimpleNamespace(tp_size=1, tp_group=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router = TopKRouter(cfg, ps, compute_aux_loss=False).to(device)
    assert attach_router_replay(router) == 1
    torch.manual_seed(3)
    x = torch.randn(5, 16, device=device)

    with torch.no_grad():
        RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
        _, idx0 = router(x)
        # drift the gate so fresh routing provably departs from the record
        router.gate.weight.add_(torch.randn_like(router.gate.weight) * 3.0)
        RouterReplay.set_global_router_replay_action(None)
        fresh_scores, fresh_idx = router(x)
        assert not torch.equal(fresh_idx, idx0)

        # int16-safe sentinel rows (ingest dtype) on tokens 1 and 3
        target = idx0.clone().to(torch.int16)
        target[1] = -1
        target[3] = -1
        RouterReplay.clear_global_indices()
        RouterReplay.set_replay_data([target])
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        scores_r, idx_r = router(x)

    mapped = torch.tensor([True, False, True, False, True], device=device)
    assert torch.equal(idx_r[mapped], idx0[mapped])  # mapped: pinned despite drift
    assert torch.equal(idx_r[~mapped], fresh_idx[~mapped])  # unmappable: live routing
    assert torch.equal(scores_r[~mapped], fresh_scores[~mapped])  # bitwise fresh scores
    assert scores_r[mapped].min() > 0  # replayed experts keep live nonzero scores

    detach_router_replay(router)


def test_r3_router_replay_keys_record_and_replay_by_microbatch(transformer_engine_import_stub):
    # R3 phase 2 (arXiv:2606.02437 §3, plan WS1): RECORD/REPLAY storage is keyed by the
    # class-level microbatch cursor so N microbatches per step round-trip correctly;
    # double-record without cursor advance and missing replay keys fail loudly.
    transformer_engine_import_stub()
    from types import SimpleNamespace

    from megatron.lite.primitive.modules.router import (
        RouterReplay,
        RouterReplayAction,
        TopKRouter,
        attach_router_replay,
        detach_router_replay,
    )

    cfg = SimpleNamespace(
        num_experts_per_tok=2, num_experts=8, router_aux_loss_coef=0.0, hidden_size=16
    )
    ps = SimpleNamespace(tp_size=1, tp_group=None)
    router = TopKRouter(cfg, ps, compute_aux_loss=False)
    assert attach_router_replay(router) == 1
    torch.manual_seed(1)
    x0, x1 = torch.randn(5, 16), torch.randn(5, 16)

    with torch.no_grad():
        RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
        RouterReplay.current_microbatch = 0
        _, idx_mb0 = router(x0)
        RouterReplay.current_microbatch = 1
        _, idx_mb1 = router(x1)
        recorded = RouterReplay.get_recorded_data()
        assert set(recorded[0]) == {0, 1}
        assert torch.equal(recorded[0][0], idx_mb0) and torch.equal(recorded[0][1], idx_mb1)
        assert not torch.equal(idx_mb0, idx_mb1)  # distinct inputs -> keyed data differs

        # double-record without cursor advance must fail loudly
        with pytest.raises(RuntimeError, match="already recorded microbatch 1"):
            router(x1)

        # replay round-trip: get_recorded_data feeds set_replay_data (dict-keyed),
        # each microbatch replays ITS OWN routing even with swapped lookup order
        RouterReplay.set_global_router_replay_action(None)
        RouterReplay.clear_global_indices()
        RouterReplay.set_replay_data(recorded)
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        RouterReplay.current_microbatch = 1
        _, idx_r1 = router(x0)  # drifted input, pinned mb1 selection
        assert torch.equal(idx_r1, idx_mb1)
        RouterReplay.current_microbatch = 0
        _, idx_r0 = router(x1)
        assert torch.equal(idx_r0, idx_mb0)

        # missing microbatch key fails loudly instead of replaying the wrong routing
        RouterReplay.current_microbatch = 2
        with pytest.raises(RuntimeError, match="microbatch 2"):
            router(x0)

    detach_router_replay(router)
    assert RouterReplay.current_microbatch is None


def test_r3_chunk_pre_hook_advances_private_microbatch_schedule(transformer_engine_import_stub):
    # Plan WS1 §1.2: the engine loads a microbatch schedule; each chunk's forward
    # pre-hook pops a PRIVATE copy (one chunk forward == one microbatch), so VPP-style
    # interleaving across chunks stays keyed correctly and over-consumption raises.
    transformer_engine_import_stub()
    from types import SimpleNamespace

    from megatron.lite.primitive.modules.router import (
        RouterReplay,
        RouterReplayAction,
        TopKRouter,
        attach_router_replay,
        detach_router_replay,
    )

    cfg = SimpleNamespace(
        num_experts_per_tok=2, num_experts=8, router_aux_loss_coef=0.0, hidden_size=16
    )
    ps = SimpleNamespace(tp_size=1, tp_group=None)

    class TwoRouterChunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.r0 = TopKRouter(cfg, ps, compute_aux_loss=False)
            self.r1 = TopKRouter(cfg, ps, compute_aux_loss=False)

        def forward(self, x):
            (s0, _), (s1, _) = self.r0(x), self.r1(x)
            return s0.sum() + s1.sum()

    torch.manual_seed(2)
    chunk0, chunk1 = TwoRouterChunk(), TwoRouterChunk()
    # PP/VPP registry order: reset on the first chunk, append on the rest
    assert attach_router_replay(chunk0, reset=True) == 2
    assert attach_router_replay(chunk1, reset=False) == 2
    assert len(RouterReplay.global_router_replay_instances) == 4

    x0, x1 = torch.randn(3, 16), torch.randn(3, 16)
    with torch.no_grad():
        RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
        RouterReplay.load_microbatch_schedule(range(2))
        # interleaved chunk order (VPP-style); per chunk the order is monotone 0..N-1
        chunk0(x0)
        chunk1(x0)
        chunk0(x1)
        chunk1(x1)
        for per_router in RouterReplay.get_recorded_data():
            assert set(per_router) == {0, 1}

        # a 5th chunk forward exceeds either chunk's private schedule copy
        with pytest.raises(RuntimeError, match="schedule exhausted"):
            chunk0(x0)

        # hook is inert while no action is armed (normal training forwards)
        RouterReplay.set_global_router_replay_action(None)
        chunk0(x0)

        # reloading the schedule refreshes the per-chunk copies (next train step)
        RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
        RouterReplay.clear_global_indices()
        RouterReplay.load_microbatch_schedule(range(2))
        chunk0(x0)
        assert RouterReplay.current_microbatch == 0

    detach_router_replay(chunk0)
    detach_router_replay(chunk1)
    assert RouterReplay.microbatch_schedule is None


def test_r3_replay_backward_recompute_matches_no_recompute(transformer_engine_import_stub):
    # Plan WS3 (mirrors verl transformer_impl.py REPLAY_BACKWARD dance): with full
    # activation recompute the backward re-runs the router forward AFTER the microbatch
    # cursor has advanced (1F1B interleave), so recompute must drain the per-router
    # FIFO in forward order instead of consulting the cursor. Contract: (i) recompute
    # topk == forward topk bitwise, (ii) grads under replay+recompute == grads under
    # replay without recompute bitwise (same CPU fp32 op sequence), (iii) the FIFO is
    # fully drained after backward and leftovers fail loudly.
    transformer_engine_import_stub()
    import copy
    from types import SimpleNamespace

    from megatron.lite.primitive.modules.router import (
        RouterReplay,
        RouterReplayAction,
        TopKRouter,
        attach_router_replay,
        detach_router_replay,
    )
    from megatron.lite.primitive.recompute import wrap_checkpoint

    cfg = SimpleNamespace(
        num_experts_per_tok=2, num_experts=8, router_aux_loss_coef=0.0, hidden_size=16
    )
    ps = SimpleNamespace(tp_size=1, tp_group=None)

    class RouterBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = TopKRouter(cfg, ps, compute_aux_loss=False)

        def forward(self, x):
            scores, indices = self.router(x)
            # selection-dependent output: replaying the wrong microbatch's indices
            # changes both the value (indices term) and the gate grads (scores term)
            return x * scores.sum(-1, keepdim=True) + indices.float().sum(-1, keepdim=True)

    class Chunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([RouterBlock(), RouterBlock()])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    torch.manual_seed(3)
    chunk = Chunk()
    ref = copy.deepcopy(chunk)  # identical weights, no recompute
    # grad-requiring inputs: the reentrant CheckpointFunction threads gradients
    # through its tensor args (parameters reach it via the recompute closure)
    x0, x1 = torch.randn(4, 16, requires_grad=True), torch.randn(4, 16, requires_grad=True)

    # cross-swapped targets (mb0 <- x1's routing, mb1 <- x0's) so a recompute that
    # consulted the advanced cursor or fresh logits would pick DIFFERENT indices
    assert attach_router_replay(ref, reset=True) == 2
    RouterReplay.load_microbatch_schedule(range(2))
    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    with torch.no_grad():
        ref(x1)
        ref(x0)
    targets = RouterReplay.get_recorded_data()
    assert all(not torch.equal(t[0], t[1]) for t in targets)
    RouterReplay.set_global_router_replay_action(None)
    RouterReplay.clear_global_indices()

    def replay_interleaved(model):
        # 1F1B-style interleave: both forwards run before the first backward, so
        # mb0's recompute happens after the cursor moved on to mb1
        RouterReplay.set_replay_data([dict(t) for t in targets])
        RouterReplay.load_microbatch_schedule(range(2))
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        loss_mb0 = model(x0).square().mean()
        loss_mb1 = model(x1).square().mean()
        loss_mb0.backward()
        loss_mb1.backward()
        RouterReplay.set_global_router_replay_action(None)
        return loss_mb0.detach(), loss_mb1.detach()

    ref_losses = replay_interleaved(ref)
    ref_grads = [p.grad.clone() for p in ref.parameters()]
    RouterReplay.assert_backward_replay_drained()  # no recompute -> nothing queued
    detach_router_replay(ref)

    assert attach_router_replay(chunk, reset=True, recompute_replay=True) == 2
    for block in chunk.blocks:
        wrap_checkpoint(block, preserve_rng_state=False)
    emitted = {id(block): [] for block in chunk.blocks}
    for block in chunk.blocks:
        block.router.register_forward_hook(
            lambda module, args, out, key=id(block): emitted[key].append(out[1])
        )

    losses = replay_interleaved(chunk)
    RouterReplay.assert_backward_replay_drained()  # (iii) recompute consumed the FIFO
    assert torch.equal(losses[0], ref_losses[0]) and torch.equal(losses[1], ref_losses[1])
    for ref_grad, grad in zip(ref_grads, (p.grad for p in chunk.parameters()), strict=True):
        torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)  # (ii) bitwise
    for indices in emitted.values():
        # forward mb0, forward mb1, recompute mb0, recompute mb1
        assert len(indices) == 4
        assert torch.equal(indices[2], indices[0]) and torch.equal(indices[3], indices[1])  # (i)

    # leftover FIFO entries (a replayed grad-enabled forward whose backward never ran)
    # must fail loudly instead of leaking into the next step
    RouterReplay.set_replay_data([dict(t) for t in targets])
    RouterReplay.load_microbatch_schedule(range(1))
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    chunk(x0)
    with pytest.raises(RuntimeError, match="unconsumed"):
        RouterReplay.assert_backward_replay_drained()
    detach_router_replay(chunk)


def test_linear_lora_forward_backward_matches_low_rank_delta():
    layer = LinearLoRA(3, 2, rank=2, alpha=4, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        layer.lora_b.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    output = layer(x)

    torch.testing.assert_close(output, torch.tensor([[10.0, 22.0]]))
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[8.0, 12.0, 0.0]]))


def test_grouped_lora_respects_per_expert_splits():
    layer = GroupedLinearLoRA(2, 2, 2, rank=1, alpha=1, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]))
        layer.lora_b.copy_(torch.tensor([[[2.0], [3.0]], [[5.0], [7.0]]]))

    x = torch.tensor([[2.0, 9.0], [4.0, 1.0], [6.0, 3.0]])
    output = layer(x, [1, 2])

    torch.testing.assert_close(output, torch.tensor([[4.0, 6.0], [5.0, 7.0], [15.0, 21.0]]))
    with pytest.raises(ValueError, match="expected 2 splits"):
        layer(x, [3])


def test_shared_grouped_lora_uses_one_adapter_for_all_experts():
    layer = SharedGroupedLinearLoRA(2, 2, 2, rank=1, alpha=2, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[1.0, -1.0]]))
        layer.lora_b.copy_(torch.tensor([[2.0], [3.0]]))

    output = layer(torch.tensor([[3.0, 1.0], [4.0, 7.0]]), [1, 1])

    torch.testing.assert_close(output, torch.tensor([[8.0, 12.0], [-12.0, -18.0]]))


def test_mrope_interleaves_text_height_and_width_sections():
    from megatron.lite.primitive.modules.mrope import MultimodalRotaryEmbedding

    base = torch.arange(3 * 2 * 6, dtype=torch.float32).reshape(3, 2, 6)

    interleaved = MultimodalRotaryEmbedding._apply_interleaved_mrope(base, mrope_section=[1, 1, 1])

    expected = base[0].clone()
    expected[..., 1] = base[1, ..., 1]
    expected[..., 2] = base[2, ..., 2]
    torch.testing.assert_close(interleaved, expected)


def test_mtp_aux_loss_scaler_threads_independent_gradient(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler

    MTPLossAutoScaler.set_loss_scale(torch.tensor(0.125))
    output = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    mtp_loss = torch.tensor(4.0, requires_grad=True)

    MTPLossAutoScaler.apply(output * 3.0, mtp_loss).sum().backward()

    torch.testing.assert_close(output.grad, torch.full_like(output, 3.0))
    torch.testing.assert_close(mtp_loss.grad, torch.tensor(0.125))
    MTPLossAutoScaler.main_loss_backward_scale = 1.0


def test_gated_delta_static_helpers_are_finite_and_shape_stable(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet

    alpha = torch.tensor([[[0.0, 1.0], [-1.0, 2.0]]])
    beta = torch.tensor([[[0.0, 2.0], [-2.0, 4.0]]])

    g, beta_sigmoid = GatedDeltaNet._compute_g_and_beta(torch.zeros(2), torch.ones(2), alpha, beta)

    assert g.shape == alpha.shape
    assert beta_sigmoid.shape == beta.shape
    assert torch.isfinite(g).all()
    assert torch.isfinite(beta_sigmoid).all()
    assert torch.all(g < 0)


def test_olora_plain_qr_factors_and_output_preservation():
    """Plain OLoRA (QR, arXiv:2406.01775): orthonormal B0 and init-time output
    preservation via the shared PiSSA-style residual write-back."""
    import torch

    from megatron.lite.primitive.modules.lora import LinearLoRA, olora_factors

    torch.manual_seed(0)
    w = torch.randn(64, 96)
    b0, a0 = olora_factors(w, 8)
    assert b0.shape == (64, 8) and a0.shape == (8, 96)
    assert (b0.T @ b0 - torch.eye(8)).abs().max() < 1e-5

    lora = LinearLoRA(96, 64, rank=8, alpha=32, use_rslora=True)
    base = w.clone()
    x = torch.randn(4, 96)
    y_ref = x @ base.T
    lora.olora_init_(base)
    y = x @ base.T + lora(x)
    assert (y - y_ref).abs().max() < 1e-3
