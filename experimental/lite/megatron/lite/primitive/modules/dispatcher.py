# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Token dispatcher: AllToAll and DeepEP dispatch/combine."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.modules.moe import _AllToAll
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import ensure_divisible
from megatron.lite.primitive.utils.moe import permute, unpermute

# Token count alignment for the static-capacity A2A shape. #5258 pulls this from
# ``moe.fused_a2a`` (``HYBRIDEP_TOKEN_ALIGNMENT``); MLite keeps its own dispatch
# primitive, so the alignment is a plain contract constant here.
_DEFAULT_TOKEN_ALIGNMENT = 8


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    return ((int(value) + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class StaticCapacityConfig:
    """Fixed-capacity / device-driven dispatch contract (mirrors upstream #5258).

    This is the MLite side of the "How upstream captures dropless MoE dispatch"
    contract: a fixed per-rank token budget plus device-driven per-expert padding
    so the whole MoE ``TransformerBlock`` chunk is CUDA-graph safe. MLite copies the
    *contract* (static shapes, on-device padding/tail-zero, a device overflow flag
    read only outside capture) and reuses its own permute/unpermute primitive rather
    than porting the HybridEP fused kernel.

    Attributes:
        num_tokens: static per-rank input token count ``M`` (derived from
            ``max_seqlen_per_dp_cp_rank``, divided by TP under sequence parallelism,
            aligned up). Every EP rank issues the same compile-time-known A2A shape.
        expert_capacity: per-expert, per-source-rank token budget ``C``. Dispatch and
            combine buffers are pre-sized to ``num_experts * C``; a routed count above
            ``C`` for any expert raises the device overflow flag (fail-loud, no silent
            truncation of the training signal — the runner reruns the microbatch eager).
    """

    num_tokens: int
    expert_capacity: int


def compute_static_capacity(
    *,
    max_seqlen_per_dp_cp_rank: int,
    num_experts: int,
    moe_router_topk: int,
    tensor_model_parallel_size: int = 1,
    sequence_parallel: bool = False,
    capacity_factor: float = 1.0,
    token_alignment: int = _DEFAULT_TOKEN_ALIGNMENT,
) -> StaticCapacityConfig:
    """Resolve the #5258 static-capacity contract from static config only.

    Returns the per-rank token budget ``M`` and per-expert capacity ``C``. All inputs
    come from static model/parallel config (no device-to-host read), so this may run
    ahead of capture. ``C`` is sized from the routed load ``M * topk / num_experts``
    scaled by ``capacity_factor``; the runtime detects the rare over-``C`` microbatch
    via the device overflow flag and reruns it eager (dropless is preserved by the
    eager fallback, never by dropping tokens inside the graph).
    """
    if max_seqlen_per_dp_cp_rank <= 0:
        raise ValueError("max_seqlen_per_dp_cp_rank must be positive")
    if capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")

    num_tokens = int(max_seqlen_per_dp_cp_rank)
    if sequence_parallel and tensor_model_parallel_size > 1:
        num_tokens = ensure_divisible(num_tokens, tensor_model_parallel_size)
    num_tokens = _align_up(num_tokens, token_alignment)

    per_expert = math.ceil(capacity_factor * num_tokens * moe_router_topk / num_experts)
    expert_capacity = _align_up(per_expert, token_alignment)
    return StaticCapacityConfig(num_tokens=num_tokens, expert_capacity=expert_capacity)

try:
    import deep_ep  # pyright: ignore[reportMissingImports]
    from deep_ep.utils import EventHandle, EventOverlap  # pyright: ignore[reportMissingImports]
except ImportError:
    deep_ep = None  # type: ignore
    EventHandle = None  # type: ignore
    EventOverlap = None  # type: ignore


def _hidden_bytes(hidden_size: int) -> int:
    return hidden_size * 2


def _build_deepep_buffer(group: dist.ProcessGroup, hidden_size: int):
    if deep_ep is None:
        raise RuntimeError("DeepEP buffer requested but deep_ep is not installed.")

    group_size = dist.get_world_size(group=group)
    hidden_bytes = _hidden_bytes(hidden_size)
    num_nvl_bytes = 0
    num_rdma_bytes = 0

    for config in (
        deep_ep.Buffer.get_dispatch_config(group_size),
        deep_ep.Buffer.get_combine_config(group_size),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group_size), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group_size), num_rdma_bytes
        )

    return deep_ep.Buffer(group=group, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=num_rdma_bytes)


def _use_moe_permute_fusion() -> bool:
    return os.environ.get("MEGATRON_LITE_MOE_PERMUTE_FUSION", "0") == "1"


def _tensor_hidden_bytes(x: torch.Tensor) -> int:
    return x.size(1) * max(x.element_size(), 2)


class _DeepEPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        buffer,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        num_experts: int,
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ):
        previous_event = (
            EventOverlap(EventHandle())
            if async_finish and EventHandle is not None and EventOverlap is not None
            else None
        )
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            topk_indices,
            num_experts=num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        (recv_hidden, recv_indices, recv_probs, recv_per_expert, handle, after_event) = (
            buffer.dispatch(
                hidden_states.contiguous(),
                topk_idx=topk_indices,
                topk_weights=topk_scores.float(),
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                previous_event=event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        )
        if async_finish:
            after_event.current_stream_wait()

        ctx.buffer = buffer
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        recv_per_expert_tensor = torch.tensor(
            recv_per_expert, dtype=torch.int64, device=recv_hidden.device
        )
        return recv_hidden, recv_indices, recv_probs, recv_per_expert_tensor, handle

    @staticmethod
    def backward(
        ctx, grad_recv_hidden, grad_recv_indices, grad_recv_probs, grad_recv_per_expert, grad_handle
    ):
        del grad_recv_indices, grad_recv_per_expert, grad_handle
        previous_event = (
            EventOverlap(EventHandle())
            if ctx.async_finish and EventHandle is not None and EventOverlap is not None
            else None
        )
        grad_scores = None if grad_recv_probs is None else grad_recv_probs.float()
        grad_hidden, grad_topk_scores, after_event = ctx.buffer.combine(
            grad_recv_hidden.contiguous(),
            ctx.handle,
            topk_weights=grad_scores,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return None, grad_hidden, None, grad_topk_scores, None, None, None


class _DeepEPCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        buffer,
        rank_grouped: torch.Tensor,
        handle,
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ):
        previous_event = (
            EventOverlap(EventHandle())
            if async_finish and EventHandle is not None and EventOverlap is not None
            else None
        )
        combined, _, after_event = buffer.combine(
            rank_grouped,
            handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined

    @staticmethod
    def backward(ctx, grad_output):
        previous_event = (
            EventOverlap(EventHandle())
            if ctx.async_finish and EventHandle is not None and EventOverlap is not None
            else None
        )
        grad_rank_grouped, _, _, _, _, after_event = ctx.buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return None, grad_rank_grouped, None, None, None


class TokenDispatcher:

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        ps: ParallelState,
        *,
        use_deepep: bool = True,
        moe_permute_fusion: bool | None = None,
        static_capacity: StaticCapacityConfig | None = None,
    ):
        self.ps = ps
        self.num_experts = num_experts
        self.ep_size = ps.ep_size
        self.num_local_experts = ensure_divisible(num_experts, ps.ep_size)
        self.moe_permute_fusion = (
            _use_moe_permute_fusion() if moe_permute_fusion is None else bool(moe_permute_fusion)
        )

        # Fixed-capacity / device-driven dispatch (mirrors #5258). When configured it
        # replaces the host-driven A2A path with static shapes so the MoE chunk is
        # CUDA-graph safe; it is mutually exclusive with the DeepEP kernel path.
        self.static_capacity = static_capacity
        self._over_budget: torch.Tensor | None = None

        self.use_deepep = (
            use_deepep
            and deep_ep is not None
            and ps.ep_size > 1
            and static_capacity is None
        )
        if self.use_deepep:
            assert ps.tp_ep_group is not None
            self.buffer = _build_deepep_buffer(ps.tp_ep_group, hidden_size)

        self._row_id_map: torch.Tensor | None = None
        self._restore_shape: tuple | None = None
        self._input_splits: list[int] | None = None
        self._output_splits: list[int] | None = None
        self._static_splits: list[int] | None = None
        self._handle = None
        self._deepep_event = None

        if self.ep_size > 1 and self.num_local_experts > 1:
            chunk_idxs = torch.arange(self.ep_size * self.num_local_experts)
            self._sort_by_experts = (
                chunk_idxs.reshape(self.ep_size, self.num_local_experts).T.ravel().tolist()
            )
            self._restore_by_ranks = (
                chunk_idxs.reshape(self.num_local_experts, self.ep_size).T.ravel().tolist()
            )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_scores: torch.Tensor, topk_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.static_capacity is not None:
            return self._dispatch_static(hidden_states, topk_scores, topk_indices)
        if self.ep_size <= 1:
            return self._dispatch_local(hidden_states, topk_scores, topk_indices)
        if self.use_deepep:
            return self._dispatch_deepep(hidden_states, topk_scores, topk_indices)
        dispatched, tpe, sorted_scores = self._dispatch_alltoall(
            hidden_states, topk_scores, topk_indices
        )
        return dispatched, tpe, sorted_scores

    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        if self.static_capacity is not None:
            return self._combine_static(expert_output)
        if self.ep_size <= 1:
            return self._combine_local(expert_output)
        if self.use_deepep:
            return self._combine_deepep(expert_output)
        return self._combine_alltoall(expert_output)

    def submit_deepep_combine(
        self, expert_output: torch.Tensor, *, allocate_on_comm_stream: bool = False
    ):
        if not self.use_deepep:
            raise RuntimeError("submit_deepep_combine requires DeepEP combine.")
        rank_grouped = unpermute(
            expert_output,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=self.moe_permute_fusion,
        )
        previous_event = (
            EventOverlap(EventHandle())
            if EventHandle is not None and EventOverlap is not None
            else None
        )
        combined = self.buffer.combine(
            rank_grouped,
            self._handle,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        event = None
        if isinstance(combined, tuple):
            if len(combined) >= 3:
                event = combined[2]
            combined = combined[0]
        return {"combined": combined, "event": event}

    def finish_deepep_combine(self, state):
        if not self.use_deepep:
            raise RuntimeError("finish_deepep_combine requires DeepEP combine.")
        event = state.get("event")
        if event is not None:
            event.current_stream_wait()
        self._row_id_map = None
        self._restore_shape = None
        self._handle = None
        self._local_tpe_list = None
        return state["combined"]

    def _dispatch_local(self, hidden_states, topk_scores, topk_indices):
        t, h = hidden_states.shape
        e = self.num_experts

        routing_map = torch.zeros(t, e, dtype=torch.bool, device=hidden_states.device)
        routing_map.scatter_(1, topk_indices, True)
        num_out = int(routing_map.sum().item())

        probs_2d = torch.zeros(t, e, dtype=topk_scores.dtype, device=hidden_states.device)
        probs_2d.scatter_(1, topk_indices, topk_scores)

        permuted, permuted_probs, sorted_indices = permute(
            hidden_states,
            routing_map,
            probs=probs_2d,
            num_out_tokens=num_out,
            fused=self.moe_permute_fusion,
        )[:3]

        self._row_id_map = sorted_indices
        self._restore_shape = hidden_states.shape

        tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
        return permuted, tokens_per_expert, permuted_probs

    def _combine_local(self, expert_output):
        result = unpermute(
            expert_output,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=self.moe_permute_fusion,
        )
        self._row_id_map = None
        self._restore_shape = None
        return result

    # ------------------------------------------------------------------
    # Static-capacity / device-driven dispatch (mirrors upstream #5258).
    #
    # Every tensor shape below is compile-time known and no ``.item()`` /
    # ``.tolist()`` / ``.cpu()`` runs on dispatch metadata, so the whole MoE chunk
    # is CUDA-graph safe. The only host-visible capacity signal is a boolean device
    # flag (``self._over_budget``); ``raise_if_over_budget`` reads it exclusively
    # OUTSIDE the captured region.
    # ------------------------------------------------------------------

    @property
    def over_budget(self) -> torch.Tensor | None:
        """Device boolean flag set during the last static dispatch (or ``None``).

        Stays on device; never read to host inside a captured region. The runner
        checks it after replay to discard and eagerly rerun an over-budget
        microbatch.
        """
        return self._over_budget

    def raise_if_over_budget(self) -> None:
        """Host-side fail-loud check for the static-capacity contract.

        Must be called only OUTSIDE the captured region (it performs a device→host
        read). Raises when the last dispatch routed more tokens to some expert than
        the static ``expert_capacity`` budget, so over-capacity never silently
        truncates the training signal.
        """
        if self._over_budget is None:
            return
        if bool(self._over_budget.item()):
            cfg = self.static_capacity
            raise RuntimeError(
                "Static-capacity MoE dispatch over budget: a routed expert exceeded "
                f"expert_capacity={cfg.expert_capacity if cfg else '?'} "
                "(num_tokens="
                f"{cfg.num_tokens if cfg else '?'}). Rerun this microbatch eager "
                "or raise moe_expert_rank_capacity_factor."
            )

    def _build_capacity_routing(self, topk_scores, topk_indices, num_tokens, num_experts, device):
        """Device-side routing_map / probs_2d build plus the device overflow flag.

        No host reads: the overflow flag is computed with ``(counts > C).any()`` and
        left on device.
        """
        routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device=device)
        routing_map.scatter_(1, topk_indices, True)
        probs_2d = torch.zeros(num_tokens, num_experts, dtype=topk_scores.dtype, device=device)
        probs_2d.scatter_(1, topk_indices, topk_scores)

        capacity = self.static_capacity.expert_capacity
        counts = routing_map.sum(dim=0)
        self._over_budget = (counts > capacity).any()
        return routing_map, probs_2d, counts

    def _zero_capacity_tail(self, permuted, permuted_probs, counts):
        """Zero the unused per-expert tail entirely on device (no host read).

        ``drop_and_pad`` fills a slot beyond an expert's routed count with a real
        (unrouted) token whose prob is already 0; explicitly zeroing the hidden and
        prob tail makes the padded contract literal (matches #5258 ``_zero_hybridep_padding``)
        and removes any ``0 * inf`` hazard on the inert rows. Numerically inert for
        the parity path since the tail probs are 0 regardless.
        """
        c = self.static_capacity.expert_capacity
        num_out = permuted.size(0)
        slot = torch.arange(num_out, device=permuted.device)
        pos_in_expert = slot % c
        expert_of_slot = slot // c
        valid = pos_in_expert < counts.index_select(0, expert_of_slot)
        permuted = permuted * valid.unsqueeze(1).to(permuted.dtype)
        permuted_probs = permuted_probs * valid.to(permuted_probs.dtype)
        return permuted, permuted_probs

    def _dispatch_static(self, hidden_states, topk_scores, topk_indices):
        cfg = self.static_capacity
        t, _h = hidden_states.shape
        e = self.num_experts
        c = cfg.expert_capacity
        device = hidden_states.device

        if t != cfg.num_tokens:
            raise ValueError(
                "Static-capacity dispatch requires a fixed token count "
                f"num_tokens={cfg.num_tokens}, got {t}. Pad the input to the static "
                "budget (max-aligned THD) before dispatch."
            )

        routing_map, probs_2d, counts = self._build_capacity_routing(
            topk_scores, topk_indices, t, e, device
        )

        # Pad-to-capacity permute: fixed [E*C, H] buffer, grouped by global expert.
        # ``drop_and_pad`` keeps the first C tokens per expert (all routed tokens when
        # under budget); padding slots carry prob 0 so they are numerically inert.
        num_out = e * c
        permuted, permuted_probs, sorted_indices = permute(
            hidden_states,
            routing_map,
            probs=probs_2d,
            num_out_tokens=num_out,
            # drop_and_pad is only honored by the argsort (non-fused) permute path;
            # the fused kernel ignores it, so force the correctness path here.
            fused=False,
            drop_and_pad=True,
        )[:3]
        permuted, permuted_probs = self._zero_capacity_tail(permuted, permuted_probs, counts)
        self._row_id_map = sorted_indices
        self._restore_shape = hidden_states.shape

        nle = self.num_local_experts
        if self.ep_size <= 1:
            # No A2A: each local expert already holds its C-token slot.
            self._local_tpe_list = [c] * nle
            tokens_per_expert = torch.full((nle,), c, dtype=torch.int64, device=device)
            return permuted, tokens_per_expert, permuted_probs

        # Static, symmetric all-to-all: every rank sends/receives exactly nle*C tokens
        # per peer. Splits are compile-time constants (no host read of any count).
        rank_split = [nle * c] * self.ep_size
        self._static_splits = rank_split
        recv_flat = _AllToAll.apply(permuted, rank_split, rank_split, self.ps.ep_group)
        recv_scores = _AllToAll.apply(
            permuted_probs.unsqueeze(-1), rank_split, rank_split, self.ps.ep_group
        )

        # Reorder recv layout (src_rank, local_expert) -> (local_expert, src_rank) with
        # a pure reshape/permute (static, autograd-safe, no host read).
        dispatched = self._regroup_ranks_to_experts(recv_flat, self.ep_size, nle, c)
        permuted_probs_out = self._regroup_ranks_to_experts(recv_scores, self.ep_size, nle, c)

        # Each local expert receives exactly ep_size*C tokens (C from every source rank).
        per_local = self.ep_size * c
        self._local_tpe_list = [per_local] * nle
        tokens_per_expert = torch.full((nle,), per_local, dtype=torch.int64, device=device)
        return dispatched, tokens_per_expert, permuted_probs_out.squeeze(-1)

    @staticmethod
    def _regroup_ranks_to_experts(flat, ep_size, num_local_experts, capacity):
        # flat: [ep_size * num_local_experts * capacity, ...] ordered (rank, local_expert)
        # -> [num_local_experts * ep_size * capacity, ...] ordered (local_expert, rank)
        tail = flat.shape[1:]
        reshaped = flat.view(ep_size, num_local_experts, capacity, *tail)
        regrouped = reshaped.permute(1, 0, 2, *range(3, reshaped.dim())).contiguous()
        return regrouped.reshape(num_local_experts * ep_size * capacity, *tail)

    @staticmethod
    def _regroup_experts_to_ranks(flat, ep_size, num_local_experts, capacity):
        # Inverse of _regroup_ranks_to_experts.
        tail = flat.shape[1:]
        reshaped = flat.view(num_local_experts, ep_size, capacity, *tail)
        regrouped = reshaped.permute(1, 0, 2, *range(3, reshaped.dim())).contiguous()
        return regrouped.reshape(ep_size * num_local_experts * capacity, *tail)

    def _combine_static(self, expert_output):
        cfg = self.static_capacity
        c = cfg.expert_capacity
        nle = self.num_local_experts

        if self.ep_size <= 1:
            result = unpermute(
                expert_output,
                self._row_id_map,
                restore_shape=self._restore_shape,
                # drop_and_pad row_id_map has duplicate/padding indices; the argsort
                # (non-fused) scatter-add path handles them, a bijection-assuming
                # fused kernel may not.
                fused=False,
            )
            self._reset_static_state()
            return result

        # Inverse reorder, then symmetric A2A back to the source ranks.
        rank_grouped = self._regroup_experts_to_ranks(expert_output, self.ep_size, nle, c)
        combined = _AllToAll.apply(
            rank_grouped, self._static_splits, self._static_splits, self.ps.ep_group
        )
        result = unpermute(
            combined,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=False,
        )
        self._reset_static_state()
        return result

    def _reset_static_state(self):
        self._row_id_map = None
        self._restore_shape = None
        self._local_tpe_list = None
        self._static_splits = None
        # ``_over_budget`` is intentionally retained until the next dispatch so the
        # runner can call ``raise_if_over_budget`` after combine/replay.

    def _dispatch_alltoall(self, hidden_states, topk_scores, topk_indices):
        t, h = hidden_states.shape
        e = self.num_experts

        routing_map = torch.zeros(t, e, dtype=torch.bool, device=hidden_states.device)
        routing_map.scatter_(1, topk_indices, True)
        # Use the actual number of routed (token, expert) pairs from routing_map
        # rather than t * topk: hash routing (ds4) can map a token's topk slots to
        # DUPLICATE experts, which scatter_ dedups, so t*topk would overcount and
        # leave permuted.size(0) != sum(input_splits) (all-to-all split mismatch).
        # Unique-topk routers (every other model) have routing_map.sum() == t*topk,
        # so this is a no-op for them.
        num_out = int(routing_map.sum().item())

        probs_2d = torch.zeros(t, e, dtype=topk_scores.dtype, device=hidden_states.device)
        probs_2d.scatter_(1, topk_indices, topk_scores)

        permuted, permuted_probs, sorted_indices = permute(
            hidden_states,
            routing_map,
            probs=probs_2d,
            num_out_tokens=num_out,
            fused=self.moe_permute_fusion,
        )[:3]
        self._row_id_map = sorted_indices
        self._restore_shape = hidden_states.shape

        tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
        tpe_by_rank = tokens_per_expert.view(self.ep_size, self.num_local_experts).sum(dim=1)
        self._input_splits = tpe_by_rank.tolist()

        global_tpe_flat = tokens_per_expert.new_empty(self.ep_size * e)
        dist.all_gather_into_tensor(global_tpe_flat, tokens_per_expert, group=self.ps.ep_group)
        global_tpe_2d = global_tpe_flat.view(self.ep_size, e)
        ep_rank = dist.get_rank(group=self.ps.ep_group)
        my_start = ep_rank * self.num_local_experts
        recv_tpe_2d = global_tpe_2d[:, my_start : my_start + self.num_local_experts].contiguous()
        self._output_splits = recv_tpe_2d.sum(dim=1).tolist()

        recv_flat = _AllToAll.apply(
            permuted, self._input_splits, self._output_splits, self.ps.ep_group
        )
        recv_scores = _AllToAll.apply(
            permuted_probs.unsqueeze(-1), self._input_splits, self._output_splits, self.ps.ep_group
        )

        if self.num_local_experts > 1:
            chunk_sizes = recv_tpe_2d.ravel().tolist()
            chunks = torch.split(recv_flat, chunk_sizes, dim=0)
            score_chunks = torch.split(recv_scores, chunk_sizes, dim=0)
            sort_idxs = self._sort_by_experts
            restore_idxs = self._restore_by_ranks
            dispatched = torch.cat([chunks[i] for i in sort_idxs], dim=0)
            permuted_probs_out = torch.cat([score_chunks[i] for i in sort_idxs], dim=0)
            self._combine_chunk_sizes = [chunk_sizes[i] for i in sort_idxs]
            self._combine_restore_idxs = restore_idxs
        else:
            dispatched = recv_flat
            permuted_probs_out = recv_scores
            self._combine_chunk_sizes = None
            self._combine_restore_idxs = None

        recv_tpe = recv_tpe_2d.sum(dim=0)
        return dispatched, recv_tpe, permuted_probs_out.squeeze(-1)

    def _combine_alltoall(self, expert_output):
        if self._combine_chunk_sizes is not None:
            chunks = torch.split(expert_output, self._combine_chunk_sizes, dim=0)
            restore_idxs = (
                self._combine_restore_idxs
                if self._combine_restore_idxs is not None
                else self._restore_by_ranks
            )
            rank_grouped = torch.cat([chunks[i] for i in restore_idxs], dim=0)
        else:
            rank_grouped = expert_output

        combined = _AllToAll.apply(
            rank_grouped, self._output_splits, self._input_splits, self.ps.ep_group
        )
        result = unpermute(
            combined,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=self.moe_permute_fusion,
        )
        self._row_id_map = None
        self._restore_shape = None
        self._input_splits = None
        self._output_splits = None
        self._combine_chunk_sizes = None
        self._combine_restore_idxs = None
        self._local_tpe_list = None
        return result

    def submit_deepep_dispatch(
        self, hidden_states, topk_scores, topk_indices, *, allocate_on_comm_stream: bool = False
    ):
        if not self.use_deepep:
            raise RuntimeError("submit_deepep_dispatch requires DeepEP dispatch.")
        previous_event = (
            EventOverlap(EventHandle())
            if EventHandle is not None and EventOverlap is not None
            else None
        )
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = self.buffer.get_dispatch_layout(
            topk_indices,
            num_experts=self.num_experts,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        topk_scores = topk_scores.float()
        recv_hidden, recv_indices, recv_probs, recv_per_expert, handle, event = (
            self.buffer.dispatch(
                hidden_states,
                topk_idx=topk_indices,
                topk_weights=topk_scores,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                previous_event=event,
                async_finish=True,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        )
        return {
            "recv_hidden": recv_hidden,
            "recv_indices": recv_indices,
            "recv_probs": recv_probs,
            "recv_per_expert": recv_per_expert,
            "handle": handle,
            "event": event,
        }

    def finish_deepep_dispatch(self, state):
        if not self.use_deepep:
            raise RuntimeError("finish_deepep_dispatch requires DeepEP dispatch.")
        self._handle = state["handle"]
        self._deepep_event = state["event"]
        self.wait_dispatch_event()
        return self._finish_deepep_dispatch(
            state["recv_hidden"],
            state["recv_indices"],
            state["recv_probs"],
            state["recv_per_expert"],
        )

    def _finish_deepep_dispatch(
        self,
        recv_hidden: torch.Tensor,
        recv_indices: torch.Tensor,
        recv_probs: torch.Tensor,
        recv_per_expert,
    ):
        if isinstance(recv_per_expert, torch.Tensor):
            recv_per_expert = [int(x) for x in recv_per_expert.detach().cpu().tolist()]
        local_tpe = torch.tensor(
            recv_per_expert[: self.num_local_experts], dtype=torch.int64, device=recv_hidden.device
        )
        self._local_tpe_list = [int(x) for x in recv_per_expert[: self.num_local_experts]]
        rows = recv_hidden.size(0)
        recv_indices = recv_indices.to(torch.long)
        routing_map = torch.zeros(
            rows, self.num_local_experts, dtype=torch.bool, device=recv_hidden.device
        )
        probs_2d = torch.zeros(
            rows, self.num_local_experts, dtype=recv_probs.dtype, device=recv_hidden.device
        )
        valid = recv_indices >= 0
        row_ids = torch.arange(rows, device=recv_hidden.device).unsqueeze(1)
        row_ids = row_ids.expand_as(recv_indices)[valid]
        expert_ids = recv_indices[valid]
        routing_map[row_ids, expert_ids] = True
        probs_2d[row_ids, expert_ids] = recv_probs[valid]
        num_out = sum(int(x) for x in recv_per_expert)
        dispatched, permuted_probs, sorted_indices = permute(
            recv_hidden,
            routing_map,
            probs=probs_2d,
            num_out_tokens=num_out,
            fused=self.moe_permute_fusion,
        )[:3]
        self._row_id_map = sorted_indices
        self._restore_shape = recv_hidden.shape
        if os.environ.get("MEGATRON_LITE_DEEPEP_DEBUG_METADATA") == "1":
            ep_rank = dist.get_rank(group=self.ps.ep_group)
            print(
                "[DEEPEP_METADATA] "
                f"ep_rank={ep_rank} recv_rows={int(recv_hidden.shape[0])} "
                f"expert_rows={int(dispatched.shape[0])} "
                f"recv_indices_shape={tuple(recv_indices.shape)} "
                f"recv_per_expert_len={len(recv_per_expert)} "
                f"recv_per_expert_sum={sum(int(x) for x in recv_per_expert)} "
                f"recv_per_expert_head={recv_per_expert[: self.num_local_experts]} "
                f"local_tpe_sum={int(local_tpe.sum().item())}",
                flush=True,
            )
        if os.environ.get("MEGATRON_LITE_DEEPEP_SKIP_DISPATCH_METADATA_CHECK") != "1" and int(
            local_tpe.sum().item()
        ) != int(dispatched.shape[0]):
            ep_rank = dist.get_rank(group=self.ps.ep_group)
            raise RuntimeError(
                "DeepEP dispatch metadata mismatch: "
                f"ep_rank={ep_rank} dispatched_tokens={int(dispatched.shape[0])} "
                f"local_tpe={local_tpe.tolist()} recv_per_expert_len={len(recv_per_expert)}"
            )
        return dispatched, local_tpe, permuted_probs

    def _dispatch_deepep(self, hidden_states, topk_scores, topk_indices):
        if torch.is_grad_enabled():
            recv_hidden, recv_indices, recv_probs, recv_per_expert, handle = _DeepEPDispatch.apply(
                self.buffer,
                hidden_states,
                topk_indices,
                topk_scores.float(),
                self.num_experts,
                False,
                False,
            )
            self._handle = handle
            self._deepep_event = None
            return self._finish_deepep_dispatch(
                recv_hidden, recv_indices, recv_probs, recv_per_expert
            )
        state = self.submit_deepep_dispatch(
            hidden_states, topk_scores, topk_indices, allocate_on_comm_stream=False
        )
        return self.finish_deepep_dispatch(state)

    def wait_dispatch_event(self):
        if self._deepep_event is not None:
            self._deepep_event.current_stream_wait()
            self._deepep_event = None

    def _combine_deepep(self, expert_output):
        rank_grouped = unpermute(
            expert_output,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=self.moe_permute_fusion,
        )
        if torch.is_grad_enabled():
            combined = _DeepEPCombine.apply(self.buffer, rank_grouped, self._handle, False, False)
        else:
            combined = self.buffer.combine(rank_grouped, self._handle)
        if isinstance(combined, tuple):
            combined = combined[0]
        self._row_id_map = None
        self._restore_shape = None
        self._handle = None
        self._local_tpe_list = None
        return combined


__all__ = ["StaticCapacityConfig", "TokenDispatcher", "compute_static_capacity"]
