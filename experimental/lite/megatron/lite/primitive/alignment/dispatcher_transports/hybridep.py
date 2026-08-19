"""HybridEP adapter that reproduces the vLLM DeepEP-LL route contract."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

try:
    import deep_ep
except ImportError:  # pragma: no cover - depends on the runtime image
    deep_ep = None


_buffer = None
_buffer_capacity = 0
_buffer_signature = None


@dataclass
class HybridEPRouteState:
    buffer: object
    handle: object
    source_indices: torch.Tensor
    source_weights: torch.Tensor
    source_output_index: torch.Tensor
    source_all_routes_valid: bool


@dataclass
class HybridEPDispatchResult:
    hidden: torch.Tensor
    tokens_per_expert: torch.Tensor
    probs: torch.Tensor
    state: HybridEPRouteState


def is_available() -> bool:
    return deep_ep is not None and hasattr(deep_ep, "HybridEPBuffer")


def require_available() -> None:
    if not is_available():
        raise RuntimeError(
            "hybridep requires a merged DeepEP runtime exporting "
            "both Buffer and HybridEPBuffer"
        )


def validate_topology(group: dist.ProcessGroup) -> int:
    domain_size_value = os.environ.get(
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"
    )
    if domain_size_value is None:
        raise RuntimeError(
            "hybridep requires explicit "
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"
        )
    domain_size = int(domain_size_value)
    if domain_size <= 0:
        raise RuntimeError(
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN must be positive"
        )
    world_size = dist.get_world_size(group=group)
    if world_size % domain_size:
        raise RuntimeError(
            f"EP world size {world_size} is not divisible by NVLink-domain "
            f"size {domain_size}"
        )
    return domain_size


def detect_accessible_ranks(group: dist.ProcessGroup) -> int:
    """Return the runtime-detected NVLink/MNNVL domain size."""

    require_available()
    try:
        hybrid_ep_cpp = importlib.import_module("hybrid_ep_cpp")
        allocator = hybrid_ep_cpp.ExtendedMemoryAllocator()
        detected = int(allocator.detect_accessible_ranks(group))
    except (ImportError, AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "hybridep could not detect the accessible NVLink/MNNVL ranks"
        ) from exc
    if detected <= 0:
        raise RuntimeError(
            f"hybridep detected an invalid accessible-rank count: {detected}"
        )
    return detected


def _get_buffer(
    group: dist.ProcessGroup,
    hidden_size: int,
    num_local_experts: int,
    required_route_capacity: int,
):
    require_available()
    domain_size = validate_topology(group)
    capacity = int(
        os.environ.get(
            "MLITE_HYBRIDEP_MAX_ROUTE_TOKENS_PER_RANK",
            str(required_route_capacity),
        )
    )
    if capacity < required_route_capacity:
        raise RuntimeError(
            f"HybridEP route capacity {capacity} is below required "
            f"{required_route_capacity}"
        )

    signature = (group, hidden_size, num_local_experts, domain_size)
    global _buffer, _buffer_capacity, _buffer_signature
    rebuild = (
        _buffer is None
        or getattr(_buffer, "runtime", None) is None
        or _buffer_signature != signature
        or _buffer_capacity < capacity
    )
    if rebuild:
        # Accessible-rank discovery is a collective topology probe.  Running
        # it on every token dispatch adds cross-tray synchronization to the
        # hot path; validate only when the process-wide buffer is rebuilt.
        detected_domain_size = detect_accessible_ranks(group)
        if detected_domain_size != domain_size:
            raise RuntimeError(
                "requested HybridEP NVLink-domain size does not match runtime "
                f"detection: requested={domain_size}, "
                f"detected={detected_domain_size}"
            )
        _buffer = deep_ep.HybridEPBuffer(
            group=group,
            hidden_dim=hidden_size,
            max_num_of_tokens_per_rank=capacity,
            num_local_experts=num_local_experts,
            use_fp8=False,
        )
        _buffer_capacity = capacity
        _buffer_signature = signature
    else:
        detected_domain_size = int(
            getattr(
                _buffer,
                "_mlite_detected_nvlink_domain_ranks",
                domain_size,
            )
        )
    setattr(
        _buffer,
        "_mlite_detected_nvlink_domain_ranks",
        detected_domain_size,
    )
    actual_domain_size = getattr(
        _buffer, "num_of_hybrid_ep_ranks_per_nvlink_domain", None
    )
    if (
        actual_domain_size is not None
        and int(actual_domain_size) != domain_size
    ):
        raise RuntimeError(
            "HybridEP runtime ignored the requested NVLink-domain size: "
            f"{actual_domain_size} != {domain_size}"
        )
    return _buffer


class _DispatchRoutes(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        buffer,
        route_hidden: torch.Tensor,
        route_indices: torch.Tensor,
        num_experts: int,
        num_local_experts: int,
    ):
        (
            dispatched,
            _,
            _,
            tokens_per_expert,
            handle,
        ) = buffer.dispatch_with_permute(
            hidden=route_hidden.contiguous(),
            topk_idx=route_indices.reshape(-1, 1).contiguous(),
            topk_weights=None,
            num_of_experts=num_experts,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=128,
            num_permuted_tokens=None,
            non_blocking=False,
        )
        ctx.buffer = buffer
        ctx.handle = handle
        return dispatched, tokens_per_expert, handle

    @staticmethod
    def backward(ctx, grad_dispatched, grad_tokens_per_expert, grad_handle):
        del grad_tokens_per_expert, grad_handle
        grad_routes, _ = ctx.buffer.combine_with_unpermute(
            hidden=grad_dispatched.contiguous(),
            handle=ctx.handle,
            pad_multiple=128,
        )
        return None, grad_routes, None, None, None


class _CombineRoutes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, expert_output: torch.Tensor, handle):
        source_routes, _ = buffer.combine_with_unpermute(
            hidden=expert_output.contiguous(),
            handle=handle,
            pad_multiple=128,
        )
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.num_permuted_tokens = expert_output.shape[0]
        return source_routes

    @staticmethod
    def backward(ctx, grad_source_routes):
        grad_expert, _, _, _, _ = ctx.buffer.dispatch_with_permute(
            hidden=grad_source_routes.contiguous(),
            scaling_factor=None,
            num_permuted_tokens=ctx.num_permuted_tokens,
            pad_multiple=128,
            handle=ctx.handle,
        )
        return None, grad_expert, None


def dispatch_routes(
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    num_experts: int,
    num_local_experts: int,
    group: dist.ProcessGroup,
) -> HybridEPDispatchResult:
    valid = (topk_indices >= 0) & (topk_indices < num_experts)
    positions = torch.nonzero(valid, as_tuple=False)
    if positions.shape[0] == 0:
        raise RuntimeError("hybridep received no valid routes")
    token_rows = positions[:, 0]
    topk_slots = positions[:, 1]
    route_hidden = hidden_states.index_select(0, token_rows).contiguous()
    route_indices = topk_indices[token_rows, topk_slots].reshape(-1)
    source_output_index = torch.full_like(
        topk_indices, -1, dtype=torch.long
    )
    source_output_index[token_rows, topk_slots] = torch.arange(
        positions.shape[0],
        device=topk_indices.device,
        dtype=torch.long,
    )
    buffer = _get_buffer(
        group,
        hidden_states.shape[1],
        num_local_experts,
        positions.shape[0],
    )
    expert_hidden, padded_tokens_per_expert, handle = _DispatchRoutes.apply(
        buffer,
        route_hidden,
        route_indices,
        num_experts,
        num_local_experts,
    )
    if not isinstance(padded_tokens_per_expert, torch.Tensor):
        raise TypeError("HybridEP tokens_per_expert must be a tensor")
    tokens_per_expert = padded_tokens_per_expert.to(
        device=hidden_states.device, dtype=torch.int64
    )
    state = HybridEPRouteState(
        buffer=buffer,
        handle=handle,
        source_indices=topk_indices,
        source_weights=topk_scores,
        source_output_index=source_output_index,
        source_all_routes_valid=(
            positions.shape[0] == topk_indices.numel()
        ),
    )
    return HybridEPDispatchResult(
        hidden=expert_hidden,
        tokens_per_expert=tokens_per_expert,
        probs=torch.zeros(
            expert_hidden.shape[0],
            device=expert_hidden.device,
            dtype=torch.float32,
        ),
        state=state,
    )


def combine_routes(
    expert_output: torch.Tensor, state: HybridEPRouteState
) -> torch.Tensor:
    return _CombineRoutes.apply(state.buffer, expert_output, state.handle)
