# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""R3 router-replay closed loop through the real model forward (arXiv:2606.02437 §3).

Record routing on a forward-only pass, drift the gates (the rollout->train policy
drift that causes TIM), then replay the recorded routing during training forwards:
the sparse path is pinned, losses are deterministic, and detach fully restores
normal routing. Phase 2: two microbatches per forward_backward, keyed by the
chunk-hook cursor — each microbatch replays ITS OWN routing and swapped keys
provably change the result. Single GPU; exercises MoELayer/dispatcher, not just
the bare router.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.deterministic import set_deterministic
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu]


def _qwen3_moe_symbols():
    te = pytest.importorskip(
        "transformer_engine.pytorch",
        reason="router replay smoke requires real Transformer Engine.",
    )
    assert hasattr(te, "Linear")
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite import protocol

    return Qwen3MoEConfig, protocol


@pytest.fixture(scope="module", autouse=True)
def _single_gpu_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the router replay smoke test.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29557")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created_pg = True
    yield
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


def _build_handle() -> tuple[ModelHandle, object]:
    Qwen3MoEConfig, protocol = _qwen3_moe_symbols()
    torch.manual_seed(4242)
    torch.cuda.manual_seed_all(4242)
    model_cfg = Qwen3MoEConfig(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        max_position_embeddings=16,
        layer_types=["full_attention", "full_attention"],
    )
    parallel = ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1)
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer="dist_opt",
        optimizer_config=OptimizerConfig(
            optimizer="adam", lr=1.0e-3, weight_decay=0.0, clip_grad=1.0
        ),
        use_deepep=False,
        deterministic=True,
    )
    bundle = protocol.build_model(model_cfg, impl_cfg=impl_cfg)
    extras = dict(bundle.extras)
    extras.update(
        {
            "model_chunks": bundle.chunks,
            "forward_step": bundle.forward_step,
            "finalize_grads": bundle.finalize_grads,
            "protocol": protocol,
        }
    )
    handle = ModelHandle(
        # single chunk unwrapped at pp=1, matching runtime.build_model's convention
        model=bundle.chunks[0] if len(bundle.chunks) == 1 else bundle.chunks,
        optimizer=bundle.optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=parallel),
        _extras=extras,
    )
    return handle, model_cfg


def _batches(vocab_size: int) -> list[PackedBatch]:
    torch.manual_seed(1357)
    return [
        PackedBatch(
            input_ids=torch.randint(0, vocab_size, (8,), device="cuda"),
            labels=torch.randint(0, vocab_size, (8,), device="cuda"),
            seq_lens=torch.full((2,), 4, dtype=torch.int64, device="cuda"),
        )
        for _ in range(2)  # sequential draws: distinct data per microbatch
    ]


def _forward_loss(
    runtime: MegatronLiteRuntime, handle: ModelHandle, batches: list[PackedBatch]
) -> float:
    result = runtime.forward_backward(
        handle, iter(batches), None, num_microbatches=len(batches), forward_only=True
    )
    # post-#68 contract: loss is optional in metrics; the canonical scalar lives on
    # model_output.loss (None only when the model computes no loss, e.g. no labels;
    # with several microbatches it is the LAST one's loss — determinism still holds).
    loss = result.model_output.loss
    assert loss is not None, "labels were provided; forward-only must surface a loss"
    return float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)


def test_router_replay_record_then_replay_closed_loop():
    from megatron.lite.primitive.modules.router import (
        RouterReplay,
        RouterReplayAction,
        attach_router_replay,
        detach_router_replay,
    )

    if dist.get_world_size() != 1:
        pytest.skip("router replay closed-loop smoke is single-GPU.")

    set_deterministic(2026)
    handle, model_cfg = _build_handle()
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    batches = _batches(model_cfg.vocab_size)

    # attach chunk-by-chunk preserving global layer order (reset only on the first)
    chunks = handle._extras["model_chunks"]
    count = 0
    for i, chunk in enumerate(chunks):
        count += attach_router_replay(chunk, reset=(i == 0))
    assert count == model_cfg.num_hidden_layers  # one router per MoE layer

    # 1) RECORD on a forward-only pass (the "rollout" stand-in); the chunk-hook
    # cursor keys each microbatch's routing under its schedule index
    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    RouterReplay.load_microbatch_schedule(range(len(batches)))
    loss_rollout = _forward_loss(runtime, handle, batches)
    recorded = RouterReplay.get_recorded_data()
    assert len(recorded) == count
    assert all(set(r) == {0, 1} for r in recorded)
    assert all(
        t.shape[-1] == model_cfg.num_experts_per_tok for r in recorded for t in r.values()
    )
    # distinct microbatch data routes distinctly, so the keying is load-bearing
    assert any(not torch.equal(r[0], r[1]) for r in recorded)

    # 2) drift every gate (rollout->train mismatch); routing must actually flip.
    # Re-recording needs a clear (double-record fails loudly) and a fresh schedule.
    with torch.no_grad():
        for chunk in chunks:
            for name, param in chunk.named_parameters():
                if "gate.weight" in name and param.shape[-1] == model_cfg.hidden_size:
                    param.add_(torch.randn_like(param) * 3.0)
    RouterReplay.clear_global_indices()
    RouterReplay.load_microbatch_schedule(range(len(batches)))
    loss_drift = _forward_loss(runtime, handle, batches)
    drifted = RouterReplay.get_recorded_data()
    assert any(
        not torch.equal(a[mb], b[mb])
        for a, b in zip(recorded, drifted, strict=True)
        for mb in (0, 1)
    )

    # 3) replay the ORIGINAL rollout routing on the drifted model, per microbatch
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    RouterReplay.set_replay_data(list(recorded))
    RouterReplay.load_microbatch_schedule(range(len(batches)))
    loss_replay_1 = _forward_loss(runtime, handle, batches)
    RouterReplay.load_microbatch_schedule(range(len(batches)))
    loss_replay_2 = _forward_loss(runtime, handle, batches)
    assert loss_replay_1 == loss_replay_2  # deterministic under pinned routing
    assert loss_replay_1 != loss_drift  # pinned sparse path != drifted free routing

    # 3b) swapping the microbatch keys replays the WRONG routing per microbatch —
    # proof the replay is keyed, not just fanned out batch-wide
    RouterReplay.set_replay_data([{0: r[1], 1: r[0]} for r in recorded])
    RouterReplay.load_microbatch_schedule(range(len(batches)))
    loss_swapped = _forward_loss(runtime, handle, batches)
    assert loss_swapped != loss_replay_1

    # 4) detach restores normal routing exactly
    for chunk in chunks:
        detach_router_replay(chunk)
    assert RouterReplay.global_router_replay_instances == []
    loss_detached = _forward_loss(runtime, handle, batches)
    assert loss_detached == loss_drift
