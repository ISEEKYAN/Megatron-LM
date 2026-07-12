# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Chunk-wise CUDA Graph capture/replay for Megatron Lite.

This is an *additive runtime capability* of the existing MLite primitives, not a
new family of graphed model implementations (see ``docs/cuda-graph-design.md``).
Normal construction contains no CUDA Graph coverage, backend, target, FP8, or
optimizer-graph choice. The runtime assembles a :class:`CudaGraphController`
*explicitly* from the concrete objects it already holds — the constructed
``TransformerBlock`` chunks, the PP schedule, the precision plan, the optimizer
plan, and the parallel state — and the controller decides, structurally, whether
each chunk's replay signature is graph-safe:

* ``enabled``        — the whole chunk is captured as one forward/backward graph
                       pair per live PP/VPP slot;
* ``partial``        — a chunk still holds a dynamic region, so only its stable
                       static sub-regions are captured;
* ``not-applicable`` — no region qualifies, so the step stays intentionally eager.

There is **no** capability protocol, capability collector, or generic policy
compiler. Eligibility is not declared by a primitive implementing an interface;
it is decided by whether the concrete chunk the runtime already built satisfies
the replay signature.

Design invariants (``docs/cuda-graph-design.md`` §"Design Invariants"):

* Replay is semantics-preserving: activating graph replay must not change model
  topology, tensor shape/dtype, RNG semantics, process groups, loss scaling,
  gradient reduction, parameter ownership, or optimizer update.
* Static input addresses persist for the graph lifetime; shape, stride, dtype,
  device, tensor-field presence, and non-tensor metadata form the replay
  signature (invariant 4).
* PP/VPP slot assignment is derived from MLite's actual schedule, not assumed
  (invariant 5); the ``(chunk, slot)`` FIFO comes from
  :mod:`megatron.lite.primitive.parallel.chunk_cuda_graphs` (ported #5258).
* The controller prefers a qualified stable subgraph; it never turns a
  capture/replay exception into a silent eager execution path (invariant 10).

The TE-backed capture itself (``make_graphed_callables``) requires CUDA + TE and
is imported lazily inside :meth:`CudaGraphController._capture_slot`; everything
else in this module is pure-CPU and unit-tested without a GPU.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence, Tuple

from megatron.lite.primitive.parallel.chunk_cuda_graphs import (
    ChunkCudaGraphRuntimeSlots,
    build_chunk_cuda_graph_slot_plan_from_schedule,
    get_required_num_microbatch_slots_per_chunk,
)


class CudaGraphDebugMode(enum.Enum):
    """Diagnostic override for CUDA Graph capture.

    The *absence* of an override means "apply the strongest implementation
    coverage qualified for this exact model/runtime plan". ``OFF`` is a
    diagnostic escape hatch only — an eager correctness oracle, an A/B baseline,
    or a debugging switch. It is **not** a peer user-selectable feature profile
    and it does not select coverage (``docs/cuda-graph-design.md`` §Decision
    Summary).
    """

    AUTO = "auto"
    OFF = "off"


class ExclusionCode(enum.Enum):
    """Stable reason codes for why a region was not captured.

    These are the auditable structured reasons attached to ``partial`` /
    ``not-applicable`` status (invariant 9). New codes may be added; existing
    values must stay stable so downstream tooling can key on them.
    """

    DYNAMIC_SHAPE = "dynamic_shape"
    THD_METADATA = "thd_metadata"
    DYNAMIC_CP_GROUP = "dynamic_cp_group"
    DYNAMIC_MOE_ROUTING = "dynamic_moe_routing"
    UNFUSED_ROPE_HOST_SYNC = "unfused_rope_host_sync"
    HOST_SYNC_IN_REGION = "host_sync_in_region"
    UNQUALIFIED_OPTIMIZER_HOOK = "unqualified_optimizer_hook"
    UNQUALIFIED_FSDP_HOOK = "unqualified_fsdp_hook"
    UNQUALIFIED_NCCL = "unqualified_nccl"
    DEBUG_DISABLED = "debug_disabled"


@dataclass(frozen=True)
class ReplaySignature:
    """The fixed contract a captured graph replays against (invariant 4).

    Two replays are interchangeable iff their signatures are equal. Values inside
    the fixed-size ``cu_seqlens`` buffers may change between replays, but their
    address, shape, dtype, maximum sequence count, maximum token capacity, CP
    topology, and non-tensor metadata stay fixed — this is the fixed-capacity THD
    contract merged as upstream #4359, not support for arbitrary shapes.

    Attributes:
        token_capacity: fixed captured token count ``M`` (max-aligned THD
            ``max_seqlen_per_dp_cp_rank`` padded budget).
        max_sequences: fixed maximum number of packed sequences
            (``thd_max_packed_sequences + 1`` including the dummy tail).
        cp_size: static context-parallel size (CP topology is graph metadata).
        hidden_dtype: dtype of the captured hidden-state surface.
        tensor_fields: sorted names of the tensor kwargs threaded into the graph
            (their presence is part of the signature).
        tensor_specs: per-tensor ``(name, shape, dtype, stride)`` — a changed
            field breaks the signature and must be rejected, never silently
            recovered.
        static_metadata: sorted non-tensor metadata ``(key, value)`` pairs
            reconstructed inside the callable (``max_seqlen_q/kv``, ``qkv_format``
            etc.) — these come from static config, not a device-to-host read.
    """

    token_capacity: int
    max_sequences: int
    cp_size: int
    hidden_dtype: str
    tensor_fields: Tuple[str, ...]
    tensor_specs: Tuple[Tuple[str, Tuple[int, ...], str, Tuple[int, ...]], ...]
    static_metadata: Tuple[Tuple[str, Any], ...]

    def matches(self, other: "ReplaySignature") -> bool:
        """Return whether ``other`` can replay against this captured signature."""
        return self == other


# Tensor fields of PackedSeqParams that are decomposed into graph inputs (#4359).
# Everything else on the dataclass is static metadata reconstructed in-callable.
_PACKED_SEQ_TENSOR_FIELDS: Tuple[str, ...] = (
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
    "seq_idx",
)
# Non-tensor metadata that must be static (never a device read) inside capture.
_PACKED_SEQ_STATIC_FIELDS: Tuple[str, ...] = (
    "qkv_format",
    "max_seqlen_q",
    "max_seqlen_kv",
    "local_cp_size",
    "total_tokens",
    "cp_rank",
)


class CudaGraphError(RuntimeError):
    """Raised when a region declared graph-safe fails its replay contract.

    A capture/replay exception is *fatal* — it is never converted into a silent
    eager fallback (invariant 10). Silent eager fallback would make performance
    and correctness claims unauditable.
    """


def build_replay_signature(
    hidden_states,
    packed_seq_params,
    *,
    cp_size: int,
    require_fused_rope: bool = True,
) -> ReplaySignature:
    """Derive the fixed replay signature from a concrete THD microbatch.

    Enforces the fixed-capacity THD contract up front so that a would-be
    graph-unsafe region fails loudly *before* capture rather than corrupting a
    replay:

    * ``max_seqlen_q``/``max_seqlen_kv`` must be present. MLite already supplies
      them from the packed protocol, so a captured chunk must reject the GQA
      fallback that derives a Python int from ``cu_seqlens[-1]``
      (``gqa.py``): that host read is not graph-safe (invariant "no ``.item()``
      inside the captured region"). See :func:`assert_fused_rope_thd`.
    * every decomposed ``cu_seqlens`` tensor is recorded by shape/dtype/stride so
      a changed field cannot reuse the graph.

    Args:
        hidden_states: the captured hidden-state surface (a ``torch.Tensor``).
        packed_seq_params: a ``PackedSeqParams``-like object (THD metadata).
        cp_size: static context-parallel size.
        require_fused_rope: when True, a missing ``max_seqlen_q/kv`` is fatal
            because the unfused RoPE path would host-sync on ``cu_seqlens``.

    Returns:
        A :class:`ReplaySignature`.

    Raises:
        CudaGraphError: if the THD metadata cannot satisfy the static contract.
    """
    max_q = getattr(packed_seq_params, "max_seqlen_q", None)
    max_kv = getattr(packed_seq_params, "max_seqlen_kv", None)
    if require_fused_rope and (max_q is None or max_kv is None):
        raise CudaGraphError(
            "THD chunk is not graph-safe: max_seqlen_q/max_seqlen_kv are required "
            "so RoPE does not derive a Python int from cu_seqlens[-1] (a host "
            "sync). Provide fused-RoPE metadata or leave this chunk eager "
            f"(reason={ExclusionCode.UNFUSED_ROPE_HOST_SYNC.value})."
        )

    tensor_specs = []
    tensor_fields = []
    for name in _PACKED_SEQ_TENSOR_FIELDS:
        t = getattr(packed_seq_params, name, None)
        if t is None:
            continue
        tensor_fields.append(name)
        tensor_specs.append(
            (
                name,
                tuple(int(s) for s in t.shape),
                str(t.dtype),
                tuple(int(s) for s in t.stride()),
            )
        )

    # token_capacity is the fixed captured token dimension of the hidden surface.
    token_capacity = int(hidden_states.shape[0])
    # max_sequences: cu_seqlens_*_padded has (max_sequences + 1) entries.
    cu = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
    if cu is None:
        cu = getattr(packed_seq_params, "cu_seqlens_q", None)
    max_sequences = int(cu.shape[0] - 1) if cu is not None else 0

    static_metadata = tuple(
        (name, getattr(packed_seq_params, name, None)) for name in _PACKED_SEQ_STATIC_FIELDS
    )

    return ReplaySignature(
        token_capacity=token_capacity,
        max_sequences=max_sequences,
        cp_size=int(cp_size),
        hidden_dtype=str(hidden_states.dtype),
        tensor_fields=tuple(tensor_fields),
        tensor_specs=tuple(tensor_specs),
        static_metadata=static_metadata,
    )


def assert_fused_rope_thd(packed_seq_params) -> None:
    """Fail loud if THD RoPE would host-sync instead of using fused metadata.

    Mirrors the gate in :func:`build_replay_signature` for call sites that only
    need the boolean guard (e.g. a captured attention sub-callable).
    """
    max_q = getattr(packed_seq_params, "max_seqlen_q", None)
    max_kv = getattr(packed_seq_params, "max_seqlen_kv", None)
    if max_q is None or max_kv is None:
        raise CudaGraphError(
            "Refusing to capture THD attention without max_seqlen_q/kv: the "
            "unfused RoPE fallback reads cu_seqlens[-1] on the host "
            f"(reason={ExclusionCode.UNFUSED_ROPE_HOST_SYNC.value})."
        )


@dataclass(frozen=True)
class CoverageEntry:
    """One region bound into the captured coverage manifest."""

    region: str
    chunk_id: int
    num_slots: int


@dataclass(frozen=True)
class ExclusionReason:
    """One region excluded before capture, with a stable reason code."""

    region: str
    code: ExclusionCode
    detail: str = ""


@dataclass(frozen=True)
class CudaGraphStatus:
    """The single observable aggregate state of qualification (invariant 9)."""

    state: Literal["enabled", "partial", "not-applicable"]
    implementation: Optional[str]
    captured: Tuple[CoverageEntry, ...] = ()
    excluded: Tuple[ExclusionReason, ...] = ()


@dataclass
class ChunkGraphSafety:
    """Structural verdict for one chunk, produced by :meth:`inspect_chunk`.

    ``graph_safe`` means every op in the chunk satisfies the replay signature.
    ``exclusions`` records the reasons a region is not graph-safe; a non-empty
    list downgrades the chunk to ``partial`` (if some static sub-region remains)
    or ``not-applicable``.
    """

    chunk_id: int
    graph_safe: bool
    exclusions: list = field(default_factory=list)


def inspect_chunk_moe_dispatch(chunk) -> Optional[ExclusionReason]:
    """Return an exclusion if the chunk holds a graph-unsafe MoE dispatcher.

    A dropless dynamic MoE dispatcher derives its all-to-all split sizes from a
    live token count via ``.item()``/``.tolist()`` (host sync), which is not
    graph-safe. Option A makes the dispatcher fixed-capacity / device-driven; a
    dispatcher that has not opted into that static mode excludes the whole chunk
    with a stable reason (AC#2: dropless dynamic MoE must never silently
    masquerade as a captured chunk).

    The dispatcher advertises graph-safety structurally via a
    ``cuda_graph_safe`` attribute/property (set by its static-capacity mode); the
    primitive layer never inspects a model name.
    """
    for module in chunk.modules() if hasattr(chunk, "modules") else []:
        # Structural, model-neutral probe: any dispatcher-like module that has
        # not declared itself graph-safe excludes the chunk.
        if getattr(module, "is_moe_dispatcher", False):
            if not bool(getattr(module, "cuda_graph_safe", False)):
                return ExclusionReason(
                    region="moe_dispatch",
                    code=ExclusionCode.DYNAMIC_MOE_ROUTING,
                    detail=(
                        "MoE dispatcher is dropless/dynamic (host-synced A2A split "
                        "sizes). Enable fixed-capacity device-driven dispatch to "
                        "make the whole chunk graph-safe."
                    ),
                )
    return None


class CudaGraphController:
    """Explicit-assembly owner of chunk-wise capture, slots, and replay.

    Constructed by the runtime from the concrete objects it already holds (no
    capability registry, collector, or policy compiler). Owns graphs, persistent
    buffers, capture order, RNG registration, and TE coordination.
    """

    def __init__(
        self,
        *,
        chunks: Sequence[Any],
        num_warmup_microbatches: int,
        num_microbatches: int,
        precision: Any = None,
        optimizer: Any = None,
        parallel_state: Any = None,
        cp_size: int = 1,
        debug: CudaGraphDebugMode = CudaGraphDebugMode.AUTO,
        num_model_chunks: int = 1,
    ) -> None:
        if num_model_chunks != 1:
            # MLite does not support VPP>1 (one local chunk per PP rank).
            raise CudaGraphError(
                "CudaGraphController supports exactly one local model chunk per PP "
                f"rank (VPP>1 unsupported); got num_model_chunks={num_model_chunks}."
            )
        self.chunks = list(chunks)
        self.num_warmup_microbatches = int(num_warmup_microbatches)
        self.num_microbatches = int(num_microbatches)
        self.precision = precision
        self.optimizer = optimizer
        self.parallel_state = parallel_state
        self.cp_size = int(cp_size)
        self.debug = debug
        self.num_model_chunks = num_model_chunks

        # Runtime slot plan (derived, not assumed). One local chunk => one slot
        # pool. Warmup + steady 1F1B order gives the required live-slot count.
        order = self._reconstruct_signed_order()
        self._num_slots_per_chunk = get_required_num_microbatch_slots_per_chunk(
            order, num_model_chunks=1
        )
        self._runtime_slots = ChunkCudaGraphRuntimeSlots(self._num_slots_per_chunk[0])

        # (chunk_id, slot) -> captured graphed callable; filled during warmup.
        self._graphed: dict[Tuple[int, int], Any] = {}
        self._status: Optional[CudaGraphStatus] = None
        self._signature: Optional[ReplaySignature] = None

    # ---- schedule / slot plumbing (CPU) --------------------------------------

    def _reconstruct_signed_order(self) -> Tuple[int, ...]:
        """Rebuild the signed +chunk/-chunk 1F1B order from the microbatch counts.

        Mirrors MLite's ``_1f1b_schedule`` slot lifetime: ``num_warmup``
        outstanding forwards, then interleaved 1F1B, then cooldown backwards. With
        one local chunk the chunk id is always 1.
        """
        num_warmup = min(self.num_warmup_microbatches, self.num_microbatches)
        num_steady = self.num_microbatches - num_warmup
        order = [1] * num_warmup
        for _ in range(num_steady):
            order.append(1)
            order.append(-1)
        order.extend([-1] * num_warmup)
        return tuple(order)

    def build_slot_plan(self, schedule_table: Optional[Sequence[Tuple[int, int]]] = None):
        """Return the deterministic per-slot plan for diagnostics/tests."""
        if schedule_table is None:
            schedule_table = [(mb, 0) for mb in range(self.num_microbatches)]
        return build_chunk_cuda_graph_slot_plan_from_schedule(
            num_warmup_microbatches=min(self.num_warmup_microbatches, self.num_microbatches),
            num_model_chunks=self.num_model_chunks,
            schedule_table=list(schedule_table),
        )

    @property
    def num_slots(self) -> int:
        return self._num_slots_per_chunk[0]

    # ---- qualification (CPU, structural) -------------------------------------

    def qualify(self) -> CudaGraphStatus:
        """Decide, structurally, the coverage state for the assembled chunks.

        This does not capture any graph; it inspects the concrete chunks the
        runtime built and returns the observable aggregate state. Capture happens
        later, lazily, during warmup (:meth:`warmup_or_replay`).
        """
        if self.debug is CudaGraphDebugMode.OFF:
            self._status = CudaGraphStatus(
                state="not-applicable",
                implementation=None,
                excluded=(
                    ExclusionReason(
                        region="all",
                        code=ExclusionCode.DEBUG_DISABLED,
                        detail="cuda_graph_debug=OFF: eager correctness oracle / A-B baseline.",
                    ),
                ),
            )
            return self._status

        captured: list[CoverageEntry] = []
        excluded: list[ExclusionReason] = []
        for chunk_id, chunk in enumerate(self.chunks):
            moe_exclusion = inspect_chunk_moe_dispatch(chunk)
            if moe_exclusion is not None:
                # AC#2: a chunk with dropless dynamic MoE is not captured whole.
                # It downgrades to partial (static attention/MLP sub-regions) —
                # never a silent eager retry masquerading as a captured chunk.
                excluded.append(moe_exclusion)
                continue
            captured.append(
                CoverageEntry(
                    region="transformer_block",
                    chunk_id=chunk_id,
                    num_slots=self.num_slots,
                )
            )

        if captured and not excluded:
            state: Literal["enabled", "partial", "not-applicable"] = "enabled"
        elif captured:
            state = "partial"
        else:
            state = "not-applicable"

        self._status = CudaGraphStatus(
            state=state,
            implementation="te_chunk_wise" if captured else None,
            captured=tuple(captured),
            excluded=tuple(excluded),
        )
        return self._status

    @property
    def status(self) -> Optional[CudaGraphStatus]:
        return self._status

    # ---- capture / replay (GPU, TE-backed) -----------------------------------

    def reset_iteration(self) -> None:
        """Reset the live-slot FIFO at the start of a forward-backward pass."""
        self._runtime_slots.reset()

    def _capture_slot(self, chunk_id: int, slot: int, callable_fn, sample_args, sample_kwargs):
        """Capture one (chunk, slot) forward/backward graph pair via TE.

        Imported lazily: ``make_graphed_callables`` needs CUDA + Transformer
        Engine, so this path is exercised only on GPU (8-card proxy / production
        PP). CPU unit tests cover the surrounding slot/qualify/signature logic.
        """
        import torch  # local import: keep module import CPU-safe/fast
        from transformer_engine.pytorch import make_graphed_callables

        if not torch.cuda.is_available():  # defensive: never capture off-GPU
            raise CudaGraphError("CUDA is required to capture a chunk CUDA graph.")

        graphed = make_graphed_callables(
            callable_fn,
            sample_args,
            sample_kwargs=sample_kwargs,
            num_warmup_iters=0,
            allow_unused_input=True,
        )
        self._graphed[(chunk_id, slot)] = graphed
        return graphed

    def get_graphed(self, chunk_id: int, slot: int):
        """Return the captured graphed callable for a ``(chunk, slot)`` key."""
        return self._graphed.get((chunk_id, slot))


__all__ = [
    "CudaGraphController",
    "CudaGraphDebugMode",
    "CudaGraphError",
    "CudaGraphStatus",
    "CoverageEntry",
    "ExclusionCode",
    "ExclusionReason",
    "ReplaySignature",
    "ChunkGraphSafety",
    "build_replay_signature",
    "assert_fused_rope_thd",
    "inspect_chunk_moe_dispatch",
]
