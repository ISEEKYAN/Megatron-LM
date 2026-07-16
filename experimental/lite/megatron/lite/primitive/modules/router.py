# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE router implementations: TopKRouter (softmax) and SigmoidTopKRouter."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler
from megatron.lite.primitive.utils.moe import (
    compute_routing_scores_for_aux_loss,
    router_gating_linear,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
)

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel import ParallelState


class RouterReplayAction(Enum):
    """Action for a router-replay step (mirrors the verl/mcore contract)."""

    RECORD = "record"  # capture the topk expert indices for later replay
    REPLAY_FORWARD = "replay_forward"  # force the recorded indices in the forward pass
    REPLAY_BACKWARD = "replay_backward"  # force them again during backward recompute


class RouterReplay:
    """R3 record/replay of MoE routing decisions (arXiv:2606.02437 §3).

    Architecture follows upstream mlite PR#49 / verl ``router_replay_patch``:
    one instance per MoE router, registered in ``global_router_replay_instances``
    in attach order so the runtime can address per-layer routing tensors
    positionally, with global set-data/set-action fan-out. Record/replay storage
    is keyed by microbatch index (upstream reference: verl
    ``router_replay_utils.set_router_replay_data`` feeds one microbatch at a
    time); the cursor advances via the chunk forward pre-hook, so keyed lookups
    fail loudly on schedule/order bugs instead of replaying the wrong microbatch.

    Score semantics deliberately follow the verl/slime reference, NOT upstream
    PR#49's ``apply(probs_dense, ...)``: the reference gathers replayed scores
    from the FULL dense score tensor (every expert has a live score), whereas
    PR#49 gathers from the post-topk scatter-zeroed dense probs — any replayed
    index outside the CURRENT top-k (i.e. exactly the drifted tokens R3 exists
    to fix) reads score 0.0 there, silently zeroing that expert's contribution
    and its gate gradient. Here the router hands ``apply_from_logits`` the raw
    gating logits instead: pre-softmax mode gathers from softmax(logits);
    post-softmax mode softmaxes the gathered logits over the replayed k —
    matching each mode's native score normalization.
    """

    global_router_replay_instances: list["RouterReplay"] = []

    # ── class-level microbatch cursor (fanned out globally) ──
    # RECORD/REPLAY storage is keyed by microbatch index so N microbatches per
    # optimizer step record and replay correctly. The router cannot know the
    # microbatch index; the chunk forward pre-hook (attach_router_replay) pops it
    # from a per-chunk copy of ``microbatch_schedule`` — one chunk forward == one
    # microbatch, monotone 0..N-1 per chunk under mlite's schedules (incl. 1F1B).
    current_microbatch: int | None = None
    microbatch_schedule: list[int] | None = None
    _schedule_generation: int = 0

    # ── global controls (one call fans out to every layer) ──
    @staticmethod
    def set_replay_data(
        all_layers_topk_indices: list[torch.Tensor | dict[int, torch.Tensor]],
    ) -> None:
        """Set per-layer replay targets: a plain tensor pins microbatch 0
        (single-microbatch compat), a dict keys targets by microbatch index —
        symmetric with :meth:`get_recorded_data`."""
        instances = RouterReplay.global_router_replay_instances
        if len(all_layers_topk_indices) != len(instances):
            raise ValueError(
                f"router replay expects {len(instances)} per-layer tensors, "
                f"got {len(all_layers_topk_indices)}."
            )
        for inst, idx in zip(instances, all_layers_topk_indices, strict=True):
            inst.set_target_indices(idx)

    @staticmethod
    def set_replay_data_for_microbatch(
        microbatch_idx: int, all_layers_topk_indices: list[torch.Tensor]
    ) -> None:
        instances = RouterReplay.global_router_replay_instances
        if len(all_layers_topk_indices) != len(instances):
            raise ValueError(
                f"router replay expects {len(instances)} per-layer tensors, "
                f"got {len(all_layers_topk_indices)}."
            )
        for inst, idx in zip(instances, all_layers_topk_indices, strict=True):
            inst.targets_by_mb[microbatch_idx] = idx

    @staticmethod
    def get_recorded_data() -> list[dict[int, torch.Tensor]]:
        return [
            inst.get_recorded_indices() for inst in RouterReplay.global_router_replay_instances
        ]

    @staticmethod
    def clear_global_indices() -> None:
        for inst in RouterReplay.global_router_replay_instances:
            inst.clear_indices()

    @staticmethod
    def set_global_router_replay_action(action: RouterReplayAction | None) -> None:
        for inst in RouterReplay.global_router_replay_instances:
            inst.router_replay_action = action

    @staticmethod
    def clear_global_router_replay_instances() -> None:
        RouterReplay.global_router_replay_instances.clear()

    @staticmethod
    def load_microbatch_schedule(schedule: Iterable[int]) -> None:
        """Load the microbatch-index schedule one forward_backward will consume.

        Each attached chunk's forward pre-hook pops a private copy in order, so
        multi-chunk (VPP) interleaving stays correct; a stale per-chunk copy is
        refreshed via the generation counter on the next load."""
        RouterReplay.microbatch_schedule = list(schedule)
        RouterReplay._schedule_generation += 1
        RouterReplay.current_microbatch = None

    @staticmethod
    def clear_microbatch_schedule() -> None:
        RouterReplay.microbatch_schedule = None
        RouterReplay._schedule_generation += 1
        RouterReplay.current_microbatch = None

    @staticmethod
    def assert_backward_replay_drained() -> None:
        """End-of-step invariant: recompute consumed every queued backward replay."""
        leftover = sum(
            len(inst.replay_backward_list)
            for inst in RouterReplay.global_router_replay_instances
        )
        if leftover:
            raise RuntimeError(
                f"router replay backward FIFO has {leftover} unconsumed entries; "
                "recompute/forward mismatch."
            )

    @staticmethod
    def _resolved_microbatch() -> int:
        # No cursor (bare router / single-microbatch legacy callers) == microbatch 0.
        mb = RouterReplay.current_microbatch
        return 0 if mb is None else mb

    # ── per-instance state ──
    def __init__(self) -> None:
        self.targets_by_mb: dict[int, torch.Tensor] = {}
        self.recorded_by_mb: dict[int, torch.Tensor] = {}
        self.router_replay_action: RouterReplayAction | None = None
        self.replay_backward_list: list[torch.Tensor] = []
        # Armed per chunk-forward when full recompute will re-run this router in
        # backward: each replayed forward then queues its target for the recompute.
        self.queue_backward_replays: bool = False
        RouterReplay.global_router_replay_instances.append(self)

    def set_target_indices(self, topk_indices: torch.Tensor | dict[int, torch.Tensor]) -> None:
        if isinstance(topk_indices, dict):
            self.targets_by_mb = dict(topk_indices)
        else:
            self.targets_by_mb = {0: topk_indices}

    def get_recorded_indices(self) -> dict[int, torch.Tensor]:
        return dict(self.recorded_by_mb)

    def clear_indices(self) -> None:
        self.recorded_by_mb = {}
        self.targets_by_mb = {}
        self.replay_backward_list = []

    def apply_from_logits(
        self,
        logits: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
        *,
        use_pre_softmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay hook the router calls after computing its own fresh topk.

        ``logits`` are the raw dense gating logits ``[tokens, num_experts]``;
        the fresh ``topk_scores``/``topk_indices`` pass through unchanged unless
        an action is armed.
        """
        action = self.router_replay_action
        if action is None:
            return topk_scores, topk_indices
        if action == RouterReplayAction.RECORD:
            mb = RouterReplay._resolved_microbatch()
            if mb in self.recorded_by_mb:
                raise RuntimeError(
                    f"router replay already recorded microbatch {mb}; advance the "
                    "microbatch cursor (or clear) before recording again."
                )
            self.recorded_by_mb[mb] = topk_indices
            return topk_scores, topk_indices
        if action == RouterReplayAction.REPLAY_FORWARD:
            mb = RouterReplay._resolved_microbatch()
            target = self.targets_by_mb.get(mb)
            if target is None:
                raise RuntimeError(
                    "router replay is in replay mode but no target indices were set "
                    f"for microbatch {mb}."
                )
            if self.queue_backward_replays:
                # Full-recompute backwards re-run this forward and drain the FIFO in
                # forward order (append here, not in set_target_indices, so FIFO
                # order == actual replayed-forward order).
                self.replay_backward_list.append(target)
        elif action == RouterReplayAction.REPLAY_BACKWARD:
            if not self.replay_backward_list:
                raise RuntimeError(
                    "router replay backward list exhausted; recompute/forward mismatch."
                )
            target = self.replay_backward_list.pop(0)
        else:  # pragma: no cover - enum is closed
            return topk_scores, topk_indices
        target = target.to(device=logits.device).long().view(logits.size(0), topk_indices.size(-1))
        # Unmappable-routing masking contract (arXiv:2605.13779 §6.3): sentinel -1 marks tokens whose rollout routes
        # could not be mapped to this batch (zero-filled capture gaps, re-tokenized
        # spans). They keep the FRESH selection and scores — live routing — instead
        # of replaying a silently wrong route (0 is a valid expert id).
        valid = target.ge(0).all(dim=-1, keepdim=True)
        target = torch.where(valid, target, topk_indices)
        logits_f = logits.float()
        if use_pre_softmax:
            probs = torch.softmax(logits_f, dim=-1).gather(1, target)
        else:
            probs = torch.softmax(logits_f.gather(1, target), dim=-1)
        probs = torch.where(valid, probs, topk_scores.float())
        return probs.to(topk_scores.dtype), target


def _is_replay_capable_router(module: nn.Module) -> bool:
    return hasattr(module, "router_replay") and not getattr(
        module, "_router_replay_exclude", False
    )


def attach_router_replay(
    model: nn.Module, *, reset: bool = True, recompute_replay: bool = False
) -> int:
    """Enable router replay on every replay-capable MoE router in ``model``.

    Walks the module tree and gives each router a fresh :class:`RouterReplay`,
    registered in module-traversal (== layer) order — no model constructor
    changes needed. ``reset=False`` appends to the existing registry so a
    multi-chunk (PP/VPP) model can attach chunk-by-chunk in global layer order.
    Returns the attached router count.

    Also registers a forward pre-hook on ``model`` (the chunk root): one chunk
    forward == one microbatch, so the hook advances the class-level microbatch
    cursor from this chunk's private copy of ``RouterReplay.microbatch_schedule``.
    Recompute re-forwards bypass ``Module.__call__`` on the chunk (checkpointing
    calls the saved function directly), so the hook never refires under recompute.

    ``recompute_replay=True`` (caller gates it on a recompute config that re-runs
    the router in backward, e.g. full/moe recompute) arms the upstream
    REPLAY_BACKWARD dance (verl ``transformer_impl.py``): the chunk post-hook
    flips replayed routers to REPLAY_BACKWARD so backward re-forwards drain the
    per-router FIFO in forward order, and the next microbatch's pre-hook flips
    them back to REPLAY_FORWARD — otherwise recompute could pick a different
    fresh topk than the forward, corrupting the sparse path's gradient wiring.
    """
    if reset:
        RouterReplay.clear_global_router_replay_instances()
    for handle in getattr(model, "_router_replay_hook_handles", ()):
        handle.remove()
    chunk_replays: list[RouterReplay] = []
    for module in model.modules():
        if _is_replay_capable_router(module):
            module.router_replay = RouterReplay()
            chunk_replays.append(module.router_replay)

    cursor_state: dict = {"generation": None, "queue": None}

    def _chunk_pre_hook(module, args) -> None:
        queue_backward = recompute_replay and torch.is_grad_enabled()
        for replay in chunk_replays:
            # previous microbatch's recompute is done; replay live forwards again
            if replay.router_replay_action is RouterReplayAction.REPLAY_BACKWARD:
                replay.router_replay_action = RouterReplayAction.REPLAY_FORWARD
            replay.queue_backward_replays = queue_backward
        if all(replay.router_replay_action is None for replay in chunk_replays):
            return
        if cursor_state["generation"] != RouterReplay._schedule_generation:
            template = RouterReplay.microbatch_schedule
            cursor_state["queue"] = None if template is None else deque(template)
            cursor_state["generation"] = RouterReplay._schedule_generation
        if cursor_state["queue"] is None:
            return  # no schedule loaded: single-microbatch (key 0) semantics
        if not cursor_state["queue"]:
            raise RuntimeError(
                "router replay microbatch schedule exhausted: more chunk forwards "
                "than scheduled microbatches."
            )
        RouterReplay.current_microbatch = cursor_state["queue"].popleft()

    def _chunk_post_hook(module, args, output) -> None:
        # Recompute re-forwards must consume the FIFO (forward order), not the
        # microbatch cursor, which may have advanced by the time backward runs.
        if not torch.is_grad_enabled():
            return  # forward-only pass: no backward, no recompute
        for replay in chunk_replays:
            if replay.router_replay_action is RouterReplayAction.REPLAY_FORWARD:
                replay.router_replay_action = RouterReplayAction.REPLAY_BACKWARD

    handles = [model.register_forward_pre_hook(_chunk_pre_hook)]
    if recompute_replay:
        handles.append(model.register_forward_hook(_chunk_post_hook))
    model._router_replay_hook_handles = handles
    return len(chunk_replays)


def detach_router_replay(model: nn.Module) -> None:
    for handle in getattr(model, "_router_replay_hook_handles", ()):
        handle.remove()
    if hasattr(model, "_router_replay_hook_handles"):
        del model._router_replay_hook_handles
    for module in model.modules():
        if _is_replay_capable_router(module):
            module.router_replay = None
    RouterReplay.clear_global_router_replay_instances()
    RouterReplay.clear_microbatch_schedule()


def _ordered_topk_from_routing_map(
    probs_dense: torch.Tensor, routing_map: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_ids = torch.arange(
        probs_dense.size(-1), device=probs_dense.device, dtype=torch.long
    ).expand_as(routing_map)
    masked_ids = torch.where(
        routing_map, expert_ids, torch.full_like(expert_ids, probs_dense.size(-1))
    )
    topk_indices = torch.sort(masked_ids, dim=-1).values[:, :topk]
    topk_scores = torch.gather(probs_dense, dim=-1, index=topk_indices)
    return topk_scores, topk_indices


class TopKRouter(nn.Module):
    """TopK gating with optional high-precision router logits/probabilities."""

    def __init__(
        self,
        config,
        ps: ParallelState,
        *,
        router_bias_rate: float = 0.0,
        compute_aux_loss: bool = True,
        use_pre_softmax: bool = False,
        moe_router_fusion: bool = False,
        router_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if router_bias_rate > 0:
            raise NotImplementedError(
                "expert-bias EMA is not implemented in the primitive router; "
                "use load_balancing_type='none' or extend ParallelState."
            )
        self.topk = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.aux_loss_coeff = config.router_aux_loss_coef
        self.router_bias_rate = router_bias_rate
        self.compute_aux_loss = compute_aux_loss
        self.use_pre_softmax = use_pre_softmax
        self.moe_router_fusion = moe_router_fusion
        self.router_dtype = router_dtype

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.register_buffer(
            "expert_bias", torch.zeros(config.num_experts, dtype=torch.float32), persistent=False
        )
        # R3 replay slot: populated by attach_router_replay(); None = replay off.
        self.router_replay: RouterReplay | None = None

        self._aux_loss_group = ps.tp_group if ps.tp_size > 1 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_dtype = self.router_dtype or x.dtype
        logits = router_gating_linear(x, self.gate.weight, None, router_dtype)
        logits = logits.view(-1, self.num_experts)
        num_tokens = logits.size(0)
        if self.moe_router_fusion:
            probs_dense, _ = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.use_pre_softmax,
                score_function="softmax",
                fused=True,
            )
            topk_scores, topk_indices = torch.topk(probs_dense, k=self.topk, dim=-1)
        else:
            probs_dense, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.use_pre_softmax,
                score_function="softmax",
                fused=False,
            )
            topk_scores, topk_indices = _ordered_topk_from_routing_map(
                probs_dense, routing_map, self.topk
            )
        if self.router_replay is not None:
            # R3 (arXiv:2606.02437 §3): pin the discrete selection to the recorded rollout
            # routing while keeping the scores live from the current logits, so gradients
            # still reach the gate and upstream adapters despite the frozen sparse path.
            topk_scores, topk_indices = self.router_replay.apply_from_logits(
                logits, topk_scores, topk_indices, use_pre_softmax=self.use_pre_softmax
            )
        if self.router_dtype is None:
            topk_scores = topk_scores.to(x.dtype)

        # A zero/None coefficient means no load-balancing objective (the RL
        # reference backend runs moe_router_load_balancing_type='none'); do not
        # attach an aux-loss gradient in that case.
        apply_aux_loss = (
            self.compute_aux_loss
            and bool(self.aux_loss_coeff)
            and self.training
            and torch.is_grad_enabled()
        )
        if apply_aux_loss:
            routing_map, aux_scores = compute_routing_scores_for_aux_loss(
                logits, self.topk, score_function="softmax", fused=self.moe_router_fusion
            )
            tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
            total_num_tokens = num_tokens
            if self._aux_loss_group is not None:
                dist.all_reduce(tokens_per_expert, group=self._aux_loss_group)
                total_num_tokens = num_tokens * dist.get_world_size(group=self._aux_loss_group)
            aux_loss = switch_load_balancing_loss_func(
                aux_scores,
                tokens_per_expert,
                total_num_tokens,
                self.topk,
                self.num_experts,
                self.aux_loss_coeff,
                fused=False,
            )
            topk_scores = MoEAuxLossAutoScaler.apply(topk_scores, aux_loss)

        return topk_scores, topk_indices


class SigmoidTopKRouter(nn.Module):
    """Sigmoid-family TopK router for DeepSeek-style MoE."""

    def __init__(
        self,
        config,
        ps: ParallelState,
        *,
        router_bias_rate: float = 0.0,
        compute_aux_loss: bool = True,
        use_pre_softmax: bool = False,
        moe_router_fusion: bool = False,
    ):
        super().__init__()
        if router_bias_rate > 0:
            raise NotImplementedError(
                "expert-bias EMA is not implemented in the primitive router; "
                "use load_balancing_type='none' or extend ParallelState."
            )
        self.topk = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.aux_loss_coeff = getattr(config, "aux_loss_alpha", 0.0)
        self.scaling_factor = config.routed_scaling_factor
        self.score_function = getattr(config, "scoring_func", "sigmoid")
        self.router_bias_rate = router_bias_rate
        self.compute_aux_loss = compute_aux_loss
        self.use_pre_softmax = use_pre_softmax
        self.moe_router_fusion = moe_router_fusion

        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
            persistent=False,
        )

        self._aux_loss_group = ps.tp_group if ps.tp_size > 1 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        logits = logits.view(-1, self.num_experts)
        num_tokens = logits.size(0)
        probs_dense, routing_map = topk_routing_with_score_function(
            logits,
            self.topk,
            score_function=self.score_function,
            expert_bias=self.expert_bias.to(logits.dtype),
            scaling_factor=(self.scaling_factor or None),
            fused=self.moe_router_fusion,
        )
        topk_scores, topk_indices = _ordered_topk_from_routing_map(
            probs_dense, routing_map, self.topk
        )
        topk_scores = topk_scores.to(logits.dtype)

        apply_aux_loss = (
            self.compute_aux_loss
            and bool(self.aux_loss_coeff)
            and self.training
            and torch.is_grad_enabled()
        )
        if apply_aux_loss:
            _, aux_scores = compute_routing_scores_for_aux_loss(
                logits, self.topk, score_function=self.score_function, fused=self.moe_router_fusion
            )
            tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
            total_num_tokens = num_tokens
            if self._aux_loss_group is not None:
                dist.all_reduce(tokens_per_expert, group=self._aux_loss_group)
                total_num_tokens = num_tokens * dist.get_world_size(group=self._aux_loss_group)
            aux_loss = switch_load_balancing_loss_func(
                aux_scores,
                tokens_per_expert,
                total_num_tokens,
                self.topk,
                self.num_experts,
                self.aux_loss_coeff,
                fused=False,
            )
            topk_scores = MoEAuxLossAutoScaler.apply(topk_scores, aux_loss)

        return topk_scores, topk_indices


__all__ = ["SigmoidTopKRouter", "TopKRouter"]
