"""Regular mLite HybridEP transport using its native top-k interface."""

from __future__ import annotations

import importlib
import os

import torch
import torch.distributed as dist

try:
    import deep_ep
except ImportError:  # pragma: no cover - runtime-image dependent
    deep_ep = None


_buffer = None
_buffer_signature = None


def is_available() -> bool:
    return deep_ep is not None and hasattr(deep_ep, "HybridEPBuffer")


def require_available() -> None:
    if not is_available():
        raise RuntimeError(
            "hybridep requires a merged DeepEP runtime exporting HybridEPBuffer"
        )


def validate_topology(group: dist.ProcessGroup) -> int:
    value = os.environ.get("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN")
    if value is None:
        raise RuntimeError(
            "hybridep requires explicit "
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"
        )
    try:
        domain_size = int(value)
    except ValueError as exc:
        raise RuntimeError(
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN must be an integer"
        ) from exc
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


def get_buffer(
    group: dist.ProcessGroup,
    hidden_size: int,
    num_local_experts: int,
    required_capacity: int,
    capacity: int,
):
    require_available()
    domain_size = validate_topology(group)
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError("HybridEP token capacity must be a positive integer")
    if capacity < required_capacity:
        raise RuntimeError(
            f"HybridEP token capacity {capacity} is below required "
            f"{required_capacity}"
        )

    signature = (
        group,
        hidden_size,
        num_local_experts,
        domain_size,
        capacity,
    )
    global _buffer, _buffer_signature
    if (
        _buffer is None
        or getattr(_buffer, "runtime", None) is None
        or _buffer_signature != signature
    ):
        detected = detect_accessible_ranks(group)
        if detected != domain_size:
            raise RuntimeError(
                "requested HybridEP NVLink-domain size does not match runtime "
                f"detection: requested={domain_size}, detected={detected}"
            )
        _buffer = deep_ep.HybridEPBuffer(
            group=group,
            hidden_dim=hidden_size,
            max_num_of_tokens_per_rank=capacity,
            num_local_experts=num_local_experts,
            use_fp8=False,
        )
        _buffer_signature = signature
        setattr(_buffer, "_mlite_detected_nvlink_domain_ranks", detected)

    actual = getattr(
        _buffer, "num_of_hybrid_ep_ranks_per_nvlink_domain", domain_size
    )
    if int(actual) != domain_size:
        raise RuntimeError(
            "HybridEP runtime topology differs from the requested domain size: "
            f"{actual} != {domain_size}"
        )
    return _buffer


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        buffer,
        hidden: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        num_local_experts: int,
    ):
        valid = (topk_indices >= 0) & (topk_indices < num_experts)
        positions = torch.nonzero(valid, as_tuple=False)
        token_rows = positions[:, 0]
        topk_slots = positions[:, 1]
        route_experts = topk_indices[token_rows, topk_slots].long()
        routing_map = torch.zeros(
            hidden.shape[0],
            num_experts,
            dtype=torch.bool,
            device=hidden.device,
        )
        routing_map[token_rows, route_experts] = True
        probs = torch.zeros(
            hidden.shape[0],
            num_experts,
            dtype=torch.float32,
            device=hidden.device,
        )
        probs.index_put_(
            (token_rows, route_experts),
            topk_weights[token_rows, topk_slots].float(),
            accumulate=True,
        )
        dispatched, permuted_weights, _, tokens_per_expert, handle = (
            buffer.dispatch_with_permute(
                hidden=hidden.contiguous(),
                routing_map=routing_map,
                probs=probs,
                scaling_factor=None,
                num_of_experts_per_rank=num_local_experts,
                pad_multiple=128,
                num_permuted_tokens=None,
                non_blocking=False,
            )
        )
        if not isinstance(tokens_per_expert, torch.Tensor):
            tokens_per_expert = torch.tensor(
                tokens_per_expert, device=dispatched.device, dtype=torch.int64
            )
        else:
            tokens_per_expert = tokens_per_expert.to(
                device=dispatched.device, dtype=torch.int64
            )
        if permuted_weights is None:
            raise RuntimeError("HybridEP dispatch did not return router weights")
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.topk_indices = topk_indices
        ctx.num_experts = num_experts
        return (
            dispatched,
            permuted_weights.reshape(-1),
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(
        ctx,
        grad_dispatched,
        grad_permuted_weights,
        grad_tokens_per_expert,
        grad_handle,
    ):
        del grad_tokens_per_expert, grad_handle
        grad_hidden, grad_probs = ctx.buffer.combine_with_unpermute(
            hidden=grad_dispatched.contiguous(),
            probs=(
                None
                if grad_permuted_weights is None
                else grad_permuted_weights.contiguous()
            ),
            handle=ctx.handle,
            pad_multiple=128,
        )
        grad_topk_weights = None
        if grad_probs is not None:
            valid = (ctx.topk_indices >= 0) & (
                ctx.topk_indices < ctx.num_experts
            )
            safe_indices = ctx.topk_indices.clamp(
                min=0, max=ctx.num_experts - 1
            )
            grad_topk_weights = grad_probs.gather(1, safe_indices)
            grad_topk_weights = grad_topk_weights.masked_fill(~valid, 0)
        return (
            None,
            grad_hidden,
            None,
            grad_topk_weights,
            None,
            None,
        )


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, expert_output: torch.Tensor, handle):
        combined, _ = buffer.combine_with_unpermute(
            hidden=expert_output.contiguous(),
            handle=handle,
            pad_multiple=128,
        )
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.num_permuted_tokens = expert_output.shape[0]
        return combined

    @staticmethod
    def backward(ctx, grad_output):
        grad_expert, _, _, _, _ = ctx.buffer.dispatch_with_permute(
            hidden=grad_output.contiguous(),
            scaling_factor=None,
            num_permuted_tokens=ctx.num_permuted_tokens,
            pad_multiple=128,
            handle=ctx.handle,
        )
        return None, grad_expert, None


__all__ = [
    "Combine",
    "Dispatch",
    "detect_accessible_ranks",
    "get_buffer",
    "is_available",
    "require_available",
    "validate_topology",
]
