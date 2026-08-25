"""Bitwise trace contracts for mLite versus official-vLLM forward gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch


SCHEMA = "mlite-vllm-forward-parity/v1"
OFFICIAL_ROLE = "official-vllm"
CANDIDATE_ROLE = "mlite-vllm"
REQUIRED_FORWARD_OPS = frozenset(
    {
        "mhc.pre",
        "mhc.post",
        "moe.input_quant",
        "moe.w13_quant",
        "moe.gate_up",
        "moe.activation_quant",
        "moe.w2_quant",
        "moe.output",
        "indexer.topk",
        "indexer.attention",
    }
)


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    sequence_lengths: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.sequence_lengths)

    @property
    def token_count(self) -> int:
        return sum(self.sequence_lengths)

    @property
    def cu_seqlens(self) -> tuple[int, ...]:
        boundaries = [0]
        for length in self.sequence_lengths:
            boundaries.append(boundaries[-1] + length)
        return tuple(boundaries)


SYNTHETIC_CASES = (
    SyntheticCase("bs1-2k-left", (2047,)),
    SyntheticCase("bs1-2k", (2048,)),
    SyntheticCase("bs1-2k-right", (2049,)),
    SyntheticCase("bs4-ragged-2k", (1, 511, 2048, 2049)),
    SyntheticCase("bs4-ragged-6k", (3, 1025, 6143, 6144)),
    SyntheticCase("bs32-2k", (2048,) * 32),
    SyntheticCase(
        "bs32-ragged-2k6k",
        tuple(2047 + (index % 3) for index in range(16))
        + tuple(6143 + (index % 3) for index in range(16)),
    ),
)


@dataclass(frozen=True)
class TraceTensor:
    op: str
    tensor: str
    value: torch.Tensor


class ParityTrace:
    """Ordered operator trace; values are detached to prevent later mutation."""

    def __init__(self) -> None:
        self._entries: list[TraceTensor] = []

    @property
    def entries(self) -> tuple[TraceTensor, ...]:
        return tuple(self._entries)

    def add(self, op: str, **tensors: torch.Tensor) -> None:
        if not op or not tensors:
            raise ValueError("a parity trace entry requires an op and tensors")
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{op}.{name} is not a tensor")
            self._entries.append(TraceTensor(op, name, value.detach().clone()))


def _location(value: torch.Tensor, flat_index: int) -> tuple[int, ...]:
    if value.ndim == 0:
        return ()
    coordinates = []
    remaining = flat_index
    for stride, size in zip(value.stride(), value.shape, strict=True):
        coordinate = remaining // stride
        coordinates.append(min(coordinate, size - 1))
        remaining %= stride
    return tuple(coordinates)


def assert_tensor_equal(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    op: str,
    tensor: str,
) -> None:
    prefix = f"bitwise parity failed: op={op} tensor={tensor}"
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"{prefix} shape={tuple(candidate.shape)} reference_shape={tuple(reference.shape)} "
            f"dtype={candidate.dtype} reference_dtype={reference.dtype}"
        )
    if reference.dtype != candidate.dtype:
        raise AssertionError(
            f"{prefix} shape={tuple(candidate.shape)} dtype={candidate.dtype} "
            f"reference_dtype={reference.dtype}"
        )
    reference_cpu = reference.detach().cpu().contiguous()
    candidate_cpu = candidate.detach().cpu().contiguous()
    if torch.equal(reference_cpu, candidate_cpu):
        return
    unequal = torch.ne(reference_cpu, candidate_cpu)
    if reference_cpu.is_floating_point():
        unequal |= torch.isnan(reference_cpu) ^ torch.isnan(candidate_cpu)
    flat_index = int(unequal.flatten().nonzero()[0].item())
    index = _location(reference_cpu, flat_index)
    raise AssertionError(
        f"{prefix} shape={tuple(candidate.shape)} dtype={candidate.dtype} "
        f"first_mismatch={index} candidate={candidate_cpu[index].item()!r} "
        f"reference={reference_cpu[index].item()!r}"
    )


def assert_trace_equal(reference: ParityTrace, candidate: ParityTrace) -> None:
    reference_entries = reference.entries
    candidate_entries = candidate.entries
    if len(reference_entries) != len(candidate_entries):
        raise AssertionError(
            "bitwise parity failed: trace length "
            f"candidate={len(candidate_entries)} reference={len(reference_entries)}"
        )
    for expected, actual in zip(reference_entries, candidate_entries, strict=True):
        if (expected.op, expected.tensor) != (actual.op, actual.tensor):
            raise AssertionError(
                "bitwise parity failed: trace order "
                f"candidate={actual.op}.{actual.tensor} "
                f"reference={expected.op}.{expected.tensor}"
            )
        assert_tensor_equal(
            expected.value,
            actual.value,
            op=expected.op,
            tensor=expected.tensor,
        )


def require_forward_ops(trace: ParityTrace) -> None:
    present = {entry.op for entry in trace.entries}
    missing = REQUIRED_FORWARD_OPS - present
    if missing:
        raise AssertionError(
            "forward parity trace is incomplete; missing ops: "
            + ", ".join(sorted(missing))
        )


def input_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_trace(
    path: str | Path,
    trace: ParityTrace,
    *,
    role: str,
    case: str,
    inputs_sha256: str,
) -> None:
    if role not in (OFFICIAL_ROLE, CANDIDATE_ROLE):
        raise ValueError(f"unsupported parity role: {role}")
    torch.save(
        {
            "schema": SCHEMA,
            "role": role,
            "case": case,
            "inputs_sha256": inputs_sha256,
            "entries": [
                {"op": item.op, "tensor": item.tensor, "value": item.value.cpu()}
                for item in trace.entries
            ],
        },
        Path(path),
    )


def load_trace(path: str | Path, *, expected_role: str) -> tuple[dict, ParityTrace]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} artifact")
    if payload.get("role") != expected_role:
        raise ValueError(
            f"{path} role={payload.get('role')!r}; expected {expected_role!r}"
        )
    trace = ParityTrace()
    for entry in payload.get("entries", ()):
        trace.add(entry["op"], **{entry["tensor"]: entry["value"]})
    return payload, trace


def compare_trace_artifacts(
    official_path: str | Path, candidate_path: str | Path
) -> None:
    official_meta, official = load_trace(
        official_path, expected_role=OFFICIAL_ROLE
    )
    candidate_meta, candidate = load_trace(
        candidate_path, expected_role=CANDIDATE_ROLE
    )
    for field in ("case", "inputs_sha256"):
        if official_meta.get(field) != candidate_meta.get(field):
            raise AssertionError(
                f"parity artifact {field} differs: "
                f"candidate={candidate_meta.get(field)!r} "
                f"reference={official_meta.get(field)!r}"
            )
    require_forward_ops(official)
    require_forward_ops(candidate)
    assert_trace_equal(official, candidate)


def cases_named(names: Iterable[str]) -> tuple[SyntheticCase, ...]:
    requested = set(names)
    selected = tuple(case for case in SYNTHETIC_CASES if case.name in requested)
    missing = requested - {case.name for case in selected}
    if missing:
        raise KeyError(", ".join(sorted(missing)))
    return selected
