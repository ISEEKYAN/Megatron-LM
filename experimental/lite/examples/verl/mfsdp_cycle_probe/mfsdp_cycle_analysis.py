# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure analysis helpers for the M-FSDP training-side per-cycle retention probe.

These functions carry no torch / CUDA / distributed dependency so the scientific
core of the probe — the per-cycle retention slope and the allocation-stack
attribution — is unit-testable on CPU without a GPU. ``mfsdp_cycle_probe.py``
imports them at the end of a run to turn the raw per-cycle CSV rows and the
``torch.cuda.memory._snapshot()`` dict into the numbers that answer the task:

  * per-cycle retention MiB/cycle for each arm (least-squares slope over the
    steady-state cycles, after a warmup),
  * the ``mfsdp - fsdp2`` retention delta (the gold-standard A/B answer),
  * the top-N allocation call-stacks that are still LIVE at end of run, ranked
    by retained bytes (the "where does it leak" attribution),
  * the effect of the ``expandable_segments`` secondary axis.

Everything here operates on plain dicts/lists so a test can feed synthetic
samples and assert the arithmetic without a training stack. See
``docs/mfsdp-training-side-cycle-retention-probe.md``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

_MIB = 1024 * 1024


def _least_squares_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Slope of the best-fit line y = a*x + b; 0.0 if fewer than 2 distinct x."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return num / denom


def per_cycle_retention_mib(
    samples: Iterable[Mapping[str, object]],
    *,
    phase: str,
    metric: str = "device_used_MiB",
    warmup_cycles: int = 3,
) -> dict[str, float]:
    """Steady-state per-cycle retention for one metric of one phase.

    ``samples`` is the list of per-(cycle,phase) rows the probe emits. We take
    the rows matching ``phase`` with ``cycle > warmup_cycles`` (the first few
    cycles carry one-off CUDA-context / cudnn / autotune allocations that are NOT
    per-cycle leakage) and return both the least-squares slope (MiB/cycle) and
    the first-to-last net delta over the steady-state window.

    Returning the slope AND the raw net delta lets the reader tell a genuine
    monotone leak (slope ≈ delta/Δcycle, both positive) from one-off overhead
    (slope ≈ 0 even if an early jump made the absolute value high).
    """
    rows = [
        r
        for r in samples
        if str(r.get("phase")) == phase and int(r["cycle"]) > warmup_cycles
    ]
    rows.sort(key=lambda r: int(r["cycle"]))
    if len(rows) < 2:
        return {
            "slope_mib_per_cycle": 0.0,
            "net_delta_mib": 0.0,
            "first_cycle": rows[0]["cycle"] if rows else None,
            "last_cycle": rows[-1]["cycle"] if rows else None,
            "n_points": len(rows),
        }
    xs = [float(int(r["cycle"])) for r in rows]
    ys = [float(r[metric]) for r in rows]
    return {
        "slope_mib_per_cycle": _least_squares_slope(xs, ys),
        "net_delta_mib": ys[-1] - ys[0],
        "first_cycle": int(rows[0]["cycle"]),
        "last_cycle": int(rows[-1]["cycle"]),
        "n_points": len(rows),
    }


def retention_delta_mib(mfsdp_slope: float, fsdp2_slope: float) -> float:
    """The gold-standard answer: how much MORE mfsdp retains per cycle than fsdp2."""
    return mfsdp_slope - fsdp2_slope


def dedup_host_storage_entries(
    raw: Iterable[tuple[int, int, tuple[int, ...]]],
) -> list[tuple[int, tuple[int, ...]]]:
    """Deduplicate raw ``(data_ptr, nbytes, shape)`` host-tensor records by
    storage ``data_ptr``.

    The probe's ``gc`` walk yields one record per live CPU tensor, but many
    tensors alias one storage (an optimizer offload buffer is narrowed into per-
    parameter views). Counting each storage once — keyed by ``data_ptr`` — is
    what makes ``host_tensor_MiB`` a true resident-bytes figure rather than a sum
    inflated by aliasing. Returns ``(nbytes, shape)`` pairs (first-seen shape per
    storage) ready for ``summarize_host_storages``. Pure: no torch, CPU-testable.
    """
    seen: dict[int, tuple[int, tuple[int, ...]]] = {}
    for data_ptr, nbytes, shape in raw:
        if data_ptr in seen:
            continue
        seen[int(data_ptr)] = (int(nbytes), tuple(shape))
    return list(seen.values())


@dataclass(frozen=True)
class StackRetention:
    """One allocation call-stack that is still live at end of run."""

    frames: tuple[str, ...]
    retained_bytes: int
    num_blocks: int

    @property
    def retained_mib(self) -> float:
        return self.retained_bytes / _MIB

    def top_frame(self) -> str:
        return self.frames[0] if self.frames else "<no-frame>"


def _frames_of(block: Mapping[str, object]) -> tuple[str, ...]:
    """Render a block's allocation frames as ``file:line:function`` strings.

    Torch's snapshot stores frames as dicts with ``filename``/``line``/``name``.
    We keep them outermost-first (the allocation site last is torch-internal;
    the earliest user frame is most informative), matching what the pytorch
    memory-viz tool shows.
    """
    frames = block.get("frames") or []
    rendered: list[str] = []
    for fr in frames:
        if isinstance(fr, Mapping):
            filename = fr.get("filename", "?")
            line = fr.get("line", "?")
            name = fr.get("name", "?")
            rendered.append(f"{filename}:{line}:{name}")
        else:
            rendered.append(str(fr))
    return tuple(rendered)


def top_live_allocation_stacks(
    snapshot: Mapping[str, object],
    *,
    top_n: int = 15,
    key_frames: int = 6,
) -> list[StackRetention]:
    """Aggregate LIVE allocations from a torch memory snapshot by call-stack.

    ``snapshot`` is the dict returned by ``torch.cuda.memory._snapshot()`` (or
    unpickled from ``_dump_snapshot``). We walk every segment's blocks, keep
    those in an active/allocated state, group them by the first ``key_frames``
    frames of their allocation stack, and return the ``top_n`` groups by total
    retained bytes. This is the "which call-stack is holding the per-cycle
    residue" attribution — run it on the end-of-run snapshot of each arm and
    diff mfsdp vs fsdp2.

    Robust to the two shapes torch has used: blocks carrying their own
    ``frames``/``history``, or segments carrying ``blocks`` with per-block
    ``frames``.
    """
    groups: dict[tuple[str, ...], list[int]] = {}

    def _record(nbytes: int, frames: tuple[str, ...]) -> None:
        if nbytes <= 0:
            return
        key = frames[:key_frames]
        bucket = groups.setdefault(key, [0, 0])
        bucket[0] += nbytes
        bucket[1] += 1

    for segment in snapshot.get("segments", []) or []:
        for block in segment.get("blocks", []) or []:
            state = block.get("state")
            # torch marks live blocks "active_allocated"; some versions only
            # emit allocated blocks with a non-empty history.
            if state is not None and state not in (
                "active_allocated",
                "allocated",
            ):
                continue
            nbytes = int(block.get("size", block.get("requested_size", 0)) or 0)
            frames = _frames_of(block)
            if not frames:
                history = block.get("history") or []
                if history and isinstance(history[0], Mapping):
                    frames = _frames_of(history[0])
            _record(nbytes, frames)

    ranked = sorted(groups.items(), key=lambda kv: kv[1][0], reverse=True)
    out: list[StackRetention] = []
    for frames, (nbytes, nblocks) in ranked[:top_n]:
        out.append(StackRetention(frames=frames, retained_bytes=nbytes, num_blocks=nblocks))
    return out


def diff_live_stacks(
    mfsdp_stacks: Sequence[StackRetention],
    fsdp2_stacks: Sequence[StackRetention],
) -> list[tuple[str, float]]:
    """(top-frame, mfsdp_MiB - fsdp2_MiB) for stacks present in mfsdp.

    Positive entries are call-stacks mfsdp retains more of than fsdp2 — the
    concrete attribution of the per-cycle retention gap.
    """
    fsdp2_by_frame: dict[str, float] = {}
    for s in fsdp2_stacks:
        fsdp2_by_frame[s.top_frame()] = fsdp2_by_frame.get(s.top_frame(), 0.0) + s.retained_mib
    out: list[tuple[str, float]] = []
    for s in mfsdp_stacks:
        delta = s.retained_mib - fsdp2_by_frame.get(s.top_frame(), 0.0)
        out.append((s.top_frame(), delta))
    out.sort(key=lambda kv: kv[1], reverse=True)
    return out


# ── gold-standard A/B combine (pure; consumes per-arm summary dicts) ─────────

_COMBINE_PHASES = ("woke", "exported", "asleep")
_COMBINE_METRICS = ("device_used_MiB", "torch_reserved_MiB", "torch_alloc_MiB")


def stack_retentions_from_summary(
    summary: Mapping[str, object],
) -> list[StackRetention]:
    """Rebuild the ``StackRetention`` list a probe arm serialized into its summary.

    ``mfsdp_cycle_probe.py`` writes each arm's top live allocation stacks under
    ``summary["live_stacks"]`` as plain dicts; this turns them back into
    ``StackRetention`` so ``diff_live_stacks`` can consume them at combine time.
    """
    out: list[StackRetention] = []
    for d in summary.get("live_stacks", []) or []:
        if not isinstance(d, Mapping):
            continue
        frames = tuple(str(f) for f in (d.get("frames") or []))
        out.append(
            StackRetention(
                frames=frames,
                retained_bytes=int(d.get("retained_bytes", 0) or 0),
                num_blocks=int(d.get("num_blocks", 0) or 0),
            )
        )
    return out


def _slope(summary: Mapping[str, object], phase: str, metric: str) -> float:
    retention = summary.get("retention") or {}
    per_phase = retention.get(phase) or {}
    per_metric = per_phase.get(metric) or {}
    return float(per_metric.get("slope_mib_per_cycle", 0.0) or 0.0)


def combine_gold_standard(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """The 2×2 gold-standard answer: ``mfsdp − fsdp2`` per ``expandable_segments``.

    ``summaries`` is the list of per-arm summary dicts (one per arm the probe
    ran). We index by ``(optimizer, expandable_segments)`` and, for each
    ``expandable_segments`` value that has BOTH an mfsdp and an fsdp2 arm, emit:

      * ``delta``: for every (phase, metric), the mfsdp slope, the fsdp2 slope,
        and ``retention_delta_mib`` = mfsdp − fsdp2 (positive = mfsdp retains
        more per cycle — the leak the task is chasing);
      * ``stack_diff``: ``diff_live_stacks`` over the two arms' live allocation
        stacks (which call-stack mfsdp holds that fsdp2 does not).

    Arms that lack a same-``expandable_segments`` counterpart are reported under
    ``unpaired`` so a missing arm is visible rather than silently dropped.
    """
    by_key: dict[tuple[object, object], Mapping[str, object]] = {}
    for s in summaries:
        by_key[(s.get("optimizer"), s.get("expandable_segments"))] = s

    exp_values = sorted(
        {k[1] for k in by_key}, key=lambda v: (v is None, str(v))
    )
    pairs: list[dict[str, object]] = []
    paired_keys: set[tuple[object, object]] = set()
    for exp in exp_values:
        m = by_key.get(("mfsdp", exp))
        f = by_key.get(("fsdp2", exp))
        if m is None or f is None:
            continue
        paired_keys.add(("mfsdp", exp))
        paired_keys.add(("fsdp2", exp))
        delta: dict[str, dict[str, dict[str, float]]] = {}
        for phase in _COMBINE_PHASES:
            delta[phase] = {}
            for metric in _COMBINE_METRICS:
                m_slope = _slope(m, phase, metric)
                f_slope = _slope(f, phase, metric)
                delta[phase][metric] = {
                    "mfsdp_slope_mib_per_cycle": m_slope,
                    "fsdp2_slope_mib_per_cycle": f_slope,
                    "delta_mib_per_cycle": retention_delta_mib(m_slope, f_slope),
                }
        pairs.append(
            {
                "expandable_segments": exp,
                "mfsdp_tag": m.get("tag"),
                "fsdp2_tag": f.get("tag"),
                "delta": delta,
                "stack_diff": diff_live_stacks(
                    stack_retentions_from_summary(m),
                    stack_retentions_from_summary(f),
                ),
            }
        )

    unpaired = [
        {"optimizer": opt, "expandable_segments": exp, "tag": by_key[(opt, exp)].get("tag")}
        for (opt, exp) in by_key
        if (opt, exp) not in paired_keys
    ]
    return {
        "gold_standard_AB": pairs,
        "unpaired_arms": unpaired,
        "phases": list(_COMBINE_PHASES),
        "metrics": list(_COMBINE_METRICS),
    }


__all__ = [
    "StackRetention",
    "dedup_host_storage_entries",
    "per_cycle_retention_mib",
    "retention_delta_mib",
    "top_live_allocation_stacks",
    "diff_live_stacks",
    "stack_retentions_from_summary",
    "combine_gold_standard",
]
