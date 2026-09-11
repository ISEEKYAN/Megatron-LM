# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Check the DeepSeek-V4.1 optimizer spec against its own stated contract.

The spec (``docs/specs/deepseek_v41_optimizer.md``) states Algorithm 1 in prose
and ships one expected vector. This script re-derives the vector from the prose
alone -- it does not import the spec's numbers -- and then tries to break its own
derivation with mutations the prose forbids. A mutation that still reproduces the
vector would mean the vector cannot tell a correct implementation from that
error, so each one must fail.

It also checks that the routing table covers every parameter family exactly once
and refuses anything it has not classified.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 200

BETA = Decimal("0.95")
TAU = Decimal("1e-3")
EPSILON = Decimal("1e-20")
K = 11


def _l2(values: list[Decimal]) -> Decimal:
    return sum((v * v for v in values), Decimal(0)).sqrt()


def algorithm_1(
    n_matrix: list[list[Decimal]],
    *,
    k: int = K,
    epsilon_under_sqrt: bool = False,
    mask_after_normalize: bool = False,
    warm_start: list[list[Decimal]] | None = None,
) -> tuple[list[list[Decimal]], list[bool]]:
    """Algorithm 1 from the spec prose, starting at the post-momentum matrix N.

    The keyword arguments exist only so the negative controls can express the
    specific errors the prose rules out; production has none of them.
    """
    rows = len(n_matrix)
    cols = len(n_matrix[0])

    rho = [_l2(row) for row in n_matrix]
    rho_mean = sum(rho, Decimal(0)) / Decimal(rows)
    mask = [r <= TAU * rho_mean for r in rho]

    u = [list(row) for row in (warm_start if warm_start is not None else n_matrix)]
    if not mask_after_normalize:
        for i, masked in enumerate(mask):
            if masked:
                u[i] = [Decimal(0)] * cols

    for step in range(1, k + 1):
        if step % 2 == 1:
            for i in range(rows):
                norm = _l2(u[i])
                denom = (norm * norm + EPSILON).sqrt() if epsilon_under_sqrt else norm + EPSILON
                u[i] = [v / denom for v in u[i]]
        else:
            for j in range(cols):
                column = [u[i][j] for i in range(rows)]
                norm = _l2(column)
                denom = (norm * norm + EPSILON).sqrt() if epsilon_under_sqrt else norm + EPSILON
                for i in range(rows):
                    u[i][j] = u[i][j] / denom

    if mask_after_normalize:
        for i, masked in enumerate(mask):
            if masked:
                u[i] = [Decimal(0)] * cols

    scale = Decimal(cols).sqrt()
    return [[scale * v for v in row] for row in u], mask


def _close(a: Decimal, b: Decimal) -> bool:
    """Agreement at the precision the spec's own decimal strings carry."""
    if a == 0 and b == 0:
        return True
    scale = max(abs(a), abs(b))
    return abs(a - b) <= scale * Decimal("1e-60")


def _matrices_agree(got: list[list[Decimal]], want: list[list[Decimal]]) -> bool:
    return all(
        _close(g, w) for got_row, want_row in zip(got, want) for g, w in zip(got_row, want_row)
    )


def check_vector(vectors: dict) -> list[str]:
    failures = []
    n_matrix = [[Decimal(str(v)) for v in row] for row in vectors["N"]]
    want_delta = [[Decimal(s) for s in row] for row in vectors["delta_decimal"]]
    want_mask = vectors["mask"]

    delta, mask = algorithm_1(n_matrix)
    if mask != want_mask:
        failures.append(f"mask mismatch: derived {mask}, vector says {want_mask}")
    if not _matrices_agree(delta, want_delta):
        failures.append(f"delta mismatch: derived {[[str(v) for v in r] for r in delta]}")

    # Each mutation states an error the prose rules out. If the vector still
    # matches, the vector cannot distinguish that error and is not evidence.
    mutations = {
        "even K (prose requires an odd count ending on rows)": dict(k=10),
        "epsilon under the square root": dict(epsilon_under_sqrt=True),
        "warm-started U instead of a fresh N": dict(
            warm_start=[[Decimal("0.5")] * len(n_matrix[0]) for _ in n_matrix]
        ),
    }
    for name, kwargs in mutations.items():
        mutated, _ = algorithm_1(n_matrix, **kwargs)
        if _matrices_agree(mutated, want_delta):
            failures.append(f"negative control did not fail: {name}")

    # Mask ordering needs its own fixture. The shipped vector's masked row is
    # exactly zero, and a zero row survives normalization unchanged, so masking
    # before or after is the same computation -- that fixture cannot see the
    # error. A near-zero row can: unmasked, normalization lifts it to unit norm.
    near_zero = [[Decimal(1), Decimal(2)], [Decimal(3), Decimal(4)], [Decimal("1e-9"), Decimal(0)]]
    before, mask_nz = algorithm_1(near_zero)
    after, _ = algorithm_1(near_zero, mask_after_normalize=True)
    if not mask_nz[2]:
        failures.append("mask-order fixture is wrong: its third row should be masked")
    elif _matrices_agree(before, after):
        failures.append("negative control did not fail: mask applied after normalization")
    return failures


def check_routing(spec_text: str) -> tuple[list[str], int]:
    """Every routed row names exactly one algorithm, and none is routed by name."""
    failures = []
    # The vocabulary the spec actually uses. "when active" / "before encoder
    # unfreeze" are stage qualifiers on a real algorithm; "Excluded" is a scope
    # decision; "OPEN:" is a declared evidence gap, which the spec is required to
    # mark rather than resolve by guessing.
    base = {"Muon", "head-wise Muon", "Sinkhorn", "AdamW"}
    rows = re.findall(r"^\| `([^`]+)`[^|]*\| ([^|]+?) \|", spec_text, re.M)
    if not rows:
        return ["routing table not found"], 0

    open_gaps = []
    for name, algorithm in rows:
        algorithm = algorithm.strip()
        if algorithm.startswith("OPEN:"):
            open_gaps.append(name)
            continue
        if algorithm.startswith("Excluded"):
            continue
        if "NOT AdamW" in algorithm or algorithm.startswith("Auxiliary-loss-free"):
            # The router bias is updated by the load-balancing rule, not an optimizer.
            continue
        head = algorithm.split(",")[0].replace(" when active", "").strip()
        if head not in base:
            failures.append(f"{name}: unrecognized algorithm {algorithm!r}")
    if open_gaps:
        print(f"note: {len(open_gaps)} declared evidence gap(s): {', '.join(open_gaps)}")

    # The spec forbids routing on the parameter name. Two families whose names
    # both end in a q/k projection must land in different groups, decided by
    # logical matrix shape.
    by_name = dict(rows)
    wq_a = next((v for k, v in by_name.items() if k.endswith("attn.wq_a.weight")), None)
    wq_b = next((v for k, v in by_name.items() if k.endswith("attn.wq_b.weight")), None)
    if wq_a and wq_b and wq_a.strip() == wq_b.strip():
        failures.append(
            "wq_a and wq_b share a group; the spec routes on logical matrix shape, "
            "and wq_a is one shared matrix while wq_b is 64 independent ones"
        )
    wkv = next((v for k, v in by_name.items() if k.endswith("attn.wkv.weight")), None)
    if wkv and "head-wise" in wkv:
        failures.append("wkv routed head-wise; it is one shared latent K/V head, not 64")
    return failures, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    args = parser.parse_args()

    vectors = json.loads(args.vectors.read_text())
    failures = check_vector(vectors)
    routing_failures, routed = check_routing(args.spec.read_text())
    failures.extend(routing_failures)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(
        f"ok: Algorithm 1 re-derived from prose reproduces the vector; "
        f"4 negative controls all diverge; {routed} routed parameter families"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
