"""Normal DeepEP transport used to reproduce the vLLM DeepEP-LL contract."""

from __future__ import annotations

import torch
import torch.distributed as dist

try:
    import deep_ep
    from deep_ep.utils import EventHandle, EventOverlap
except ImportError:  # pragma: no cover - depends on the runtime image
    deep_ep = None
    EventHandle = None
    EventOverlap = None


_buffer = None


def is_available() -> bool:
    return deep_ep is not None and hasattr(deep_ep, "Buffer")


def require_available() -> None:
    if not is_available():
        raise RuntimeError("deepep requested but deep_ep.Buffer is unavailable")


def initialize(*, num_sms: int = 20) -> None:
    require_available()
    deep_ep.Buffer.set_num_sms(num_sms)


def new_event_overlap():
    if EventHandle is None or EventOverlap is None:
        return None
    return EventOverlap(EventHandle())


def configure_deterministic_allocator() -> None:
    """Avoid DeepEP's deterministic ``torch.empty`` stream race."""

    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    ):
        torch.utils.deterministic.fill_uninitialized_memory = False


def hidden_bytes(hidden_size: int) -> int:
    return hidden_size * 2


def tensor_hidden_bytes(tensor: torch.Tensor) -> int:
    return tensor.size(1) * max(tensor.element_size(), 2)


def get_buffer(group: dist.ProcessGroup, payload_hidden_bytes: int):
    """Return the process-wide normal-DeepEP communication buffer."""

    require_available()
    configure_deterministic_allocator()
    group_size = dist.get_world_size(group=group)
    num_nvl_bytes = 0
    num_rdma_bytes = 0
    for config in (
        deep_ep.Buffer.get_dispatch_config(group_size),
        deep_ep.Buffer.get_combine_config(group_size),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(
                payload_hidden_bytes, group_size
            ),
            num_nvl_bytes,
        )
        if group_size > torch.cuda.device_count():
            num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(
                    payload_hidden_bytes, group_size
                ),
                num_rdma_bytes,
            )

    global _buffer
    if (
        _buffer is None
        or getattr(_buffer, "runtime", None) is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = deep_ep.Buffer(
            group=group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            explicitly_destroy=True,
        )
    return _buffer


def build_buffer(group: dist.ProcessGroup, hidden_size: int):
    return get_buffer(group, hidden_bytes(hidden_size))


def dispatch_raw(
    group,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    num_experts: int,
    *,
    async_finish: bool,
    allocate_on_comm_stream: bool,
):
    """Run normal DeepEP dispatch without owning autograd state."""

    buffer = get_buffer(group, tensor_hidden_bytes(hidden_states))
    hidden_states = hidden_states.contiguous()
    topk_indices = topk_indices.contiguous()
    topk_scores = topk_scores.float().contiguous()
    previous_event = new_event_overlap() if async_finish else None
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
    result = buffer.dispatch(
        hidden_states,
        topk_idx=topk_indices,
        topk_weights=topk_scores,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    if async_finish:
        result[-1].current_stream_wait()
    return result


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        group,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        num_experts: int,
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ):
        (
            recv_hidden,
            recv_indices,
            recv_probs,
            recv_per_expert,
            handle,
            _,
        ) = dispatch_raw(
            group,
            hidden_states,
            topk_indices,
            topk_scores,
            num_experts,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        recv_per_expert_tensor = torch.tensor(
            recv_per_expert, dtype=torch.int64
        )
        return (
            recv_hidden,
            recv_indices,
            recv_probs,
            recv_per_expert_tensor,
            handle,
        )

    @staticmethod
    def backward(
        ctx,
        grad_recv_hidden,
        grad_recv_indices,
        grad_recv_probs,
        grad_recv_per_expert,
        grad_handle,
    ):
        del grad_recv_indices, grad_recv_per_expert, grad_handle
        previous_event = (
            new_event_overlap() if ctx.async_finish else None
        )
        buffer = get_buffer(
            ctx.group, tensor_hidden_bytes(grad_recv_hidden)
        )
        grad_scores = (
            None if grad_recv_probs is None else grad_recv_probs.float()
        )
        grad_hidden, grad_topk_scores, after_event = buffer.combine(
            grad_recv_hidden.contiguous(),
            ctx.handle,
            topk_weights=grad_scores,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return (
            None,
            grad_hidden,
            None,
            grad_topk_scores,
            None,
            None,
            None,
        )


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        group,
        rank_grouped: torch.Tensor,
        handle,
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ):
        buffer = get_buffer(group, tensor_hidden_bytes(rank_grouped))
        previous_event = new_event_overlap() if async_finish else None
        combined, _, after_event = buffer.combine(
            rank_grouped,
            handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined

    @staticmethod
    def backward(ctx, grad_output):
        previous_event = (
            new_event_overlap() if ctx.async_finish else None
        )
        buffer = get_buffer(ctx.group, tensor_hidden_bytes(grad_output))
        grad_rank_grouped, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return None, grad_rank_grouped, None, None, None
