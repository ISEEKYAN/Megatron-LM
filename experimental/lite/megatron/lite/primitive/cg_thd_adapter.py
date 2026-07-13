# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""THD static-input adapter for CUDA Graph capture (chunk-wise, THD-only).

A CUDA graph boundary cannot depend on an arbitrary Python dataclass, on a
device-to-host read taken *inside* the captured region, or on a tensor whose
address/shape/dtype changes between replays. Merged Megatron ``dev`` #4359
bridges variable-length THD to CUDA Graphs by fixing the *physical*
representation while letting the *values* inside fixed-size buffers change:

* ``PackedSeqParams`` is decomposed into a small set of tensor kwargs plus static
  (non-tensor) metadata, and reconstructed *inside* the graphed callable.
  ``max_seqlen_q/kv`` and friends come from static config, never a capture-time
  host read.
* the four ``cu_seqlens`` tensors are padded to a fixed maximum sequence count
  (``thd_max_packed_sequences + 1``), token-like tensors to a fixed token
  capacity (``max_seqlen_per_dp_cp_rank``), and a padding mask keeps padded
  tokens out of loss/router accounting.

Open main-branch #5672 records the concrete RoPE caveat this adapter enforces:
an *unfused* THD RoPE path that derives a Python int from ``cu_seqlens[-1]``
(``gqa.py`` lines 204-213) host-syncs and is **not** graph-safe; only the fused
RoPE path — for which MLite already supplies ``max_seqlen_q/kv`` from the packed
protocol — qualifies. The graph path must therefore *reject* that fallback
loudly rather than silently host-sync inside a captured region.

Scope (CPU-only, this task): the decompose/reconstruct helpers, the fixed
``cu_seqlens``/padding-mask buffers with fail-loud over-capacity, and the
fused-RoPE / GQA-fallback gate. This module performs **no** actual capture and
changes **no** eager numerics; the capture/replay wiring and MoE dispatch live
in sibling tasks.

Naming and field lists are kept identical to the CUDA Graph controller spine so
the two decompositions unify without churn at integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

# Tensor fields of PackedSeqParams that are threaded into the graph as inputs
# (#4359). Their *values* may change per replay while address/shape/dtype stay
# fixed. Everything else is static metadata reconstructed in-callable.
_PACKED_SEQ_TENSOR_FIELDS: Tuple[str, ...] = (
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
    "seq_idx",
)
# The four ``cu_seqlens`` tensors that get fixed-capacity persistent buffers.
_CU_SEQLENS_FIELDS: Tuple[str, ...] = (
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
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


class ThdStaticInputError(RuntimeError):
    """Raised when a THD microbatch cannot satisfy the static-capture contract.

    A capture/replay contract violation is *fatal*: it is never converted into a
    silent eager fallback, because a silent fallback would make performance and
    correctness claims unauditable. (Aligns with the controller spine's
    ``CudaGraphError``; kept as a distinct type here so this adapter is
    self-contained and the integrator can unify the two.)
    """


# ---------------------------------------------------------------------------
# #4359 decompose / reconstruct
# ---------------------------------------------------------------------------


def decompose_packed_seq_params(
    packed_seq_params: Any,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    """Split ``PackedSeqParams`` into graph-threaded tensors + static metadata.

    Returns ``(tensor_kwargs, static_meta)`` where ``tensor_kwargs`` maps a field
    name to a live ``torch.Tensor`` (the graph inputs), and ``static_meta`` maps
    a field name to a fixed non-tensor value plus the static ``cp_group`` handle
    (a process group, reattached verbatim on reconstruction, never captured).
    """
    tensor_kwargs: Dict[str, Tensor] = {}
    for name in _PACKED_SEQ_TENSOR_FIELDS:
        value = getattr(packed_seq_params, name, None)
        if value is not None:
            tensor_kwargs[name] = value
    static_meta: Dict[str, Any] = {
        name: getattr(packed_seq_params, name, None) for name in _PACKED_SEQ_STATIC_FIELDS
    }
    # cp_group is a process-group handle: static per capture, kept out of the
    # tensor signature and reattached verbatim on reconstruction.
    static_meta["cp_group"] = getattr(packed_seq_params, "cp_group", None)
    return tensor_kwargs, static_meta


def reconstruct_packed_seq_params(
    tensor_kwargs: Dict[str, Tensor], static_meta: Dict[str, Any]
) -> PackedSeqParams:
    """Rebuild a ``PackedSeqParams`` inside the graphed callable (#4359).

    The inverse of :func:`decompose_packed_seq_params`. Only the decomposed
    tensor kwargs and static metadata are used; nothing is read from the host.
    """
    fields: Dict[str, Any] = {}
    fields.update(tensor_kwargs)
    fields.update(static_meta)
    allowed = PackedSeqParams.__dataclass_fields__
    return PackedSeqParams(**{k: v for k, v in fields.items() if k in allowed})


# ---------------------------------------------------------------------------
# Fused-RoPE / GQA-fallback gate (#5672 caveat)
# ---------------------------------------------------------------------------


def assert_fused_rope_thd(packed_seq_params: Any) -> None:
    """Reject the graph-unsafe GQA THD-RoPE fallback (``gqa.py`` 204-213).

    When ``max_seqlen_q``/``max_seqlen_kv`` are absent, the GQA THD path derives
    ``seq_len_for_rope = int(cu_seqlens_q[-1])`` — a device-to-host read that
    host-syncs and is not graph-safe (#5672). MLite already supplies these from
    the packed protocol, so a captured chunk must fail loud here rather than
    silently host-sync inside the captured region.

    Raises:
        ThdStaticInputError: if fused-RoPE metadata is missing.
    """
    max_q = getattr(packed_seq_params, "max_seqlen_q", None)
    max_kv = getattr(packed_seq_params, "max_seqlen_kv", None)
    if max_q is None or max_kv is None:
        raise ThdStaticInputError(
            "THD chunk is not graph-safe: max_seqlen_q/max_seqlen_kv are required "
            "so RoPE uses fused metadata instead of deriving a Python int from "
            "cu_seqlens[-1] (a host sync; see gqa.py:204-213 and upstream #5672). "
            "Provide fused-RoPE metadata or leave this chunk eager."
        )


# ``reject_gqa_python_int_fallback`` is the same gate under the name that reads
# as the graph-path intent at the GQA call site.
reject_gqa_python_int_fallback = assert_fused_rope_thd


# Stable exclusion-reason code (kept identical to the controller spine's
# ``ExclusionCode.UNFUSED_ROPE_HOST_SYNC`` so the two decompositions unify).
UNFUSED_ROPE_HOST_SYNC = "unfused_rope_host_sync"


@dataclass(frozen=True)
class ThdRopeSafety:
    """Whether a THD chunk's RoPE is graph-safe, and why not if it isn't.

    The controller uses this to *narrow the captured boundary and report* rather
    than either silently host-syncing inside capture or catching a failed
    whole-attention capture and retrying eagerly. ``assert_fused_rope_thd`` is the
    fatal gate for a chunk already declared whole-graphable; this classifier is
    the non-raising decision for what coverage a chunk qualifies for.
    """

    fused_rope_available: bool
    exclusion_reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.fused_rope_available


def classify_thd_rope(packed_seq_params: Any) -> ThdRopeSafety:
    """Classify a THD chunk's RoPE graph-safety without raising.

    When fused-RoPE metadata (``max_seqlen_q/kv``) is present the whole THD
    attention (incl. RoPE and projections) may be captured. When it is absent the
    safe boundary narrows to the TE THD attention core and the RoPE/projection
    exclusion is *reported* with :data:`UNFUSED_ROPE_HOST_SYNC` — never silently
    left eager, and never a caught-and-retried whole-attention capture.
    """
    max_q = getattr(packed_seq_params, "max_seqlen_q", None)
    max_kv = getattr(packed_seq_params, "max_seqlen_kv", None)
    if max_q is None or max_kv is None:
        return ThdRopeSafety(False, UNFUSED_ROPE_HOST_SYNC)
    return ThdRopeSafety(True, None)


# ---------------------------------------------------------------------------
# Fixed-capacity cu_seqlens + padding-mask buffers
# ---------------------------------------------------------------------------


@dataclass
class StaticThdInputBuffers:
    """Persistent, fixed-address THD metadata surfaces for CUDA-graph replay.

    The four ``cu_seqlens`` tensors live at fixed addresses with a fixed shape
    (``max_sequences + 1``) and dtype; :meth:`copy_in` writes each replay's real
    values into them in place, pads the tail as dummy sequence(s), and rebuilds a
    boolean padding mask of length ``token_capacity`` that is ``True`` exactly on
    real (non-padded) tokens. Over-capacity is fatal (:class:`ThdStaticInputError`).

    Attributes:
        token_capacity: fixed captured token count ``M`` (max-aligned THD budget).
        max_sequences: fixed maximum number of packed sequences ``S`` (buffers hold
            ``S + 1`` cumulative offsets; a real batch may use fewer).
        cu_seqlens: fixed-address buffer per ``cu_seqlens_*`` field.
        padding_mask: ``(M,)`` bool, ``True`` on real tokens (rebuilt each copy_in).
        num_real_sequences: sequences written by the most recent :meth:`copy_in`.
        real_tokens: real (non-padded) token count from the most recent copy_in.
    """

    token_capacity: int
    max_sequences: int
    cu_seqlens: Dict[str, Tensor]
    padding_mask: Tensor
    index_dtype: torch.dtype
    num_real_sequences: int = 0
    real_tokens: int = 0

    @property
    def buffer_length(self) -> int:
        """Fixed cu_seqlens buffer length ``S + 1``."""
        return self.max_sequences + 1

    def copy_in(
        self, packed_seq_params: Any, *, dummy_tail_sequence: bool = False
    ) -> None:
        """Copy one replay's variable THD metadata into the fixed buffers.

        Runs *outside* the captured region: it may launch device ops and take a
        host read of the capacity counters (the over-capacity signal), but it
        never mutates the buffers' address/shape/dtype. After it returns the
        buffers replay the same graph with new values.

        Args:
            packed_seq_params: source THD metadata for this microbatch.
            dummy_tail_sequence: if True, the max-alignment tail
                ``[real_tokens, token_capacity)`` is represented as a single
                dummy sequence in ``cu_seqlens_*_padded`` (matching #4359's
                dummy-tail option for attention); the padding mask still excludes
                those tokens. If False (default) the unused tail slots are
                zero-length dummies.

        Raises:
            ThdStaticInputError: if the real sequence count or token count
                exceeds the fixed capacity, or required cu_seqlens are missing.
        """
        assert_fused_rope_thd(packed_seq_params)

        padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
        if padded is None:
            padded = getattr(packed_seq_params, "cu_seqlens_q", None)
        if padded is None:
            raise ThdStaticInputError(
                "Static THD copy_in requires cu_seqlens_q(_padded) to size the "
                "packed region."
            )

        num_real = int(padded.numel()) - 1
        if num_real < 0:
            raise ThdStaticInputError("cu_seqlens must have at least one entry.")
        # A dummy tail sequence needs one extra slot beyond the real sequences.
        needed_sequences = num_real + (1 if dummy_tail_sequence else 0)
        if needed_sequences > self.max_sequences:
            raise ThdStaticInputError(
                f"THD microbatch has {num_real} sequences"
                + (" (+1 dummy tail)" if dummy_tail_sequence else "")
                + f" but the captured buffers hold at most {self.max_sequences}. "
                "Increase max_sequences or leave this chunk eager."
            )
        real_tokens = int(padded[-1].item())
        if real_tokens > self.token_capacity:
            raise ThdStaticInputError(
                f"THD microbatch packs {real_tokens} tokens but the captured "
                f"token capacity is {self.token_capacity}. Increase token_capacity "
                "or leave this chunk eager."
            )

        for name in _CU_SEQLENS_FIELDS:
            source = getattr(packed_seq_params, name, None)
            if source is None:
                continue
            buf = self.cu_seqlens.get(name)
            if buf is None:
                continue
            src_len = int(source.numel())
            if src_len != num_real + 1:
                raise ThdStaticInputError(
                    f"cu_seqlens field {name!r} has {src_len} entries but "
                    f"cu_seqlens_q(_padded) implies {num_real + 1}; all four "
                    "cu_seqlens must describe the same sequence count."
                )
            # In-place copy keeps the buffer's address/shape/dtype fixed.
            buf[: num_real + 1].copy_(source.to(dtype=buf.dtype))
            # Fill the fixed-capacity tail so downstream ops see a valid,
            # non-decreasing offset array (dummy sequences).
            tail_value = real_tokens
            if dummy_tail_sequence and name.endswith("_padded"):
                # One dummy sequence spanning the max-alignment tail up to M.
                if self.token_capacity > real_tokens:
                    buf[num_real + 1] = self.token_capacity
                    buf[num_real + 2 :] = self.token_capacity
                else:
                    buf[num_real + 1 :] = self.token_capacity
            else:
                buf[num_real + 1 :] = tail_value

        self.num_real_sequences = num_real
        self.real_tokens = real_tokens
        self._rebuild_padding_mask()

    def _rebuild_padding_mask(self) -> None:
        """Recompute ``padding_mask`` (True on real tokens) from the buffers.

        Vectorized and host-sync-free: real tokens are those inside a real
        (unpadded) per-sequence span. Per-sequence pad *and* the max-alignment
        tail are both excluded. When the source only carries padded cu_seqlens
        (MLite's current convention, unpadded == padded), the only distinction is
        the max-alignment tail — which is still honestly masked.
        """
        mask = self.padding_mask
        mask.zero_()
        num_real = self.num_real_sequences
        if num_real == 0 or self.real_tokens == 0:
            return

        padded = self.cu_seqlens["cu_seqlens_q_padded"][: num_real + 1].to(torch.long)
        unpadded = self.cu_seqlens.get("cu_seqlens_q")
        if unpadded is None:
            unpadded = padded
        else:
            unpadded = unpadded[: num_real + 1].to(torch.long)

        real_len = (unpadded[1:] - unpadded[:-1]).clamp(min=0)  # (num_real,)
        token_idx = torch.arange(
            self.token_capacity, device=mask.device, dtype=torch.long
        )
        # segment(token) = index of the padded span the token falls in.
        seg = torch.searchsorted(padded[1:].contiguous(), token_idx, right=True)
        seg = seg.clamp(max=num_real - 1)
        seg_start = padded[:-1][seg]
        seg_real_len = real_len[seg]
        valid = (
            (token_idx < padded[-1])
            & ((token_idx - seg_start) < seg_real_len)
        )
        mask[:] = valid

    def to_packed_seq_params(
        self, static_meta: Optional[Dict[str, Any]] = None
    ) -> PackedSeqParams:
        """Reconstruct a ``PackedSeqParams`` pointing at the fixed buffers.

        The returned object shares the persistent buffer tensors, so the graph
        captured against it replays against the same addresses.
        """
        tensor_kwargs = {name: self.cu_seqlens[name] for name in self.cu_seqlens}
        meta = dict(static_meta) if static_meta is not None else {}
        return reconstruct_packed_seq_params(tensor_kwargs, meta)


def allocate_static_thd_buffers(
    *,
    token_capacity: int,
    max_sequences: int,
    device: Any = "cpu",
    index_dtype: torch.dtype = torch.int32,
    include_kv: bool = True,
) -> StaticThdInputBuffers:
    """Allocate persistent fixed-capacity THD metadata surfaces.

    Args:
        token_capacity: fixed captured token count ``M``.
        max_sequences: fixed maximum packed-sequence count ``S`` (buffers get
            ``S + 1`` cumulative offsets).
        device: buffer device.
        index_dtype: cu_seqlens dtype (int32 to match TE/#4359).
        include_kv: allocate the ``*_kv`` buffers too (self-attention shares the
            q buffers logically, but distinct addresses match the decomposed
            four-tensor contract).

    Raises:
        ThdStaticInputError: for non-positive capacities.
    """
    if token_capacity <= 0:
        raise ThdStaticInputError(f"token_capacity must be > 0, got {token_capacity}.")
    if max_sequences <= 0:
        raise ThdStaticInputError(f"max_sequences must be > 0, got {max_sequences}.")

    length = max_sequences + 1
    names = list(_CU_SEQLENS_FIELDS)
    if not include_kv:
        names = [n for n in names if not n.startswith("cu_seqlens_kv")]
    cu_seqlens = {
        name: torch.zeros(length, dtype=index_dtype, device=device) for name in names
    }
    padding_mask = torch.zeros(token_capacity, dtype=torch.bool, device=device)
    return StaticThdInputBuffers(
        token_capacity=int(token_capacity),
        max_sequences=int(max_sequences),
        cu_seqlens=cu_seqlens,
        padding_mask=padding_mask,
        index_dtype=index_dtype,
    )


__all__ = [
    "UNFUSED_ROPE_HOST_SYNC",
    "ThdRopeSafety",
    "ThdStaticInputError",
    "StaticThdInputBuffers",
    "allocate_static_thd_buffers",
    "assert_fused_rope_thd",
    "classify_thd_rope",
    "decompose_packed_seq_params",
    "reconstruct_packed_seq_params",
    "reject_gqa_python_int_fallback",
]
