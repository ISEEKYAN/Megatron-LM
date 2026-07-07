#!/usr/bin/env python3
"""Compare FLA packed-THD backward with per-sequence dense references."""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F
from fla.modules.convolution import causal_conv1d
from fla.modules.l2norm import l2norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


DTYPE = torch.bfloat16
LENGTHS = (17, 63, 129, 257)
CONV_DIM = 512
NUM_HEADS = 8
HEAD_DIM = 128
CONV_KERNEL = 4
CONV_PAD_ALIGNMENT = 4096


def _metric(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.detach().float()
    reference = reference.detach().float()
    delta = candidate - reference
    denom = reference.square().sum().sqrt().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "max_abs": float(delta.abs().max().item()),
        "rel_l2": float(delta.square().sum().sqrt().div(denom).item()),
        "candidate_l2": float(candidate.square().sum().sqrt().item()),
        "reference_l2": float(reference.square().sum().sqrt().item()),
    }


def _cu_seqlens(lengths: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), device="cuda")


def _conv(x: torch.Tensor, weight: torch.Tensor, cu_seqlens=None) -> torch.Tensor:
    out, _ = causal_conv1d(
        x=x,
        weight=weight,
        bias=None,
        activation="silu",
        cu_seqlens=cu_seqlens,
    )
    return out


def probe_causal_conv() -> dict:
    total = sum(LENGTHS)
    cu = _cu_seqlens(LENGTHS)
    x_base = torch.randn(1, total, CONV_DIM, device="cuda", dtype=DTYPE)
    weight_base = torch.randn(CONV_DIM, CONV_KERNEL, device="cuda", dtype=DTYPE) * 0.02
    dy = torch.randn_like(x_base)

    x_packed = x_base.detach().clone().requires_grad_(True)
    weight_packed = weight_base.detach().clone().requires_grad_(True)
    out_packed = _conv(x_packed, weight_packed, cu)
    (out_packed * dy).float().sum().backward()

    x_dense = x_base.detach().clone().requires_grad_(True)
    weight_dense = weight_base.detach().clone().requires_grad_(True)
    dense_outputs = []
    start = 0
    for length in LENGTHS:
        dense_outputs.append(_conv(x_dense[:, start : start + length], weight_dense))
        start += length
    out_dense = torch.cat(dense_outputs, dim=1)
    (out_dense * dy).float().sum().backward()

    pad_n = -total % CONV_PAD_ALIGNMENT
    x_padded = F.pad(x_base, (0, 0, 0, pad_n)).detach().requires_grad_(True)
    weight_padded = weight_base.detach().clone().requires_grad_(True)
    cu_padded = cu.clone()
    cu_padded[-1] += pad_n
    out_padded_full = _conv(x_padded, weight_padded, cu_padded)
    out_padded = out_padded_full[:, :total]
    (out_padded * dy).float().sum().backward()

    return {
        "packed_vs_dense": {
            "output": _metric(out_packed, out_dense),
            "input_grad": _metric(x_packed.grad, x_dense.grad),
            "weight_grad": _metric(weight_packed.grad, weight_dense.grad),
        },
        "padded_packed_vs_dense": {
            "output": _metric(out_padded, out_dense),
            "input_grad": _metric(x_padded.grad[:, :total], x_dense.grad),
            "weight_grad": _metric(weight_padded.grad, weight_dense.grad),
        },
        "pad_n": pad_n,
    }


def _delta_rule(q, k, v, g, beta, cu_seqlens=None):
    out, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )
    return out


def probe_delta_rule() -> dict:
    total = sum(LENGTHS)
    cu = _cu_seqlens(LENGTHS)
    bases = {
        "q": torch.randn(1, total, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE),
        "k": torch.randn(1, total, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE),
        "v": torch.randn(1, total, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE),
        "g": -torch.rand(1, total, NUM_HEADS, device="cuda", dtype=torch.float32),
        "beta": torch.rand(1, total, NUM_HEADS, device="cuda", dtype=DTYPE),
    }
    bases["q"] = F.normalize(bases["q"].float(), dim=-1).to(DTYPE)
    bases["k"] = F.normalize(bases["k"].float(), dim=-1).to(DTYPE)
    dy = torch.randn_like(bases["v"])

    packed = {name: value.detach().clone().requires_grad_(True) for name, value in bases.items()}
    out_packed = _delta_rule(**packed, cu_seqlens=cu)
    (out_packed * dy).float().sum().backward()

    dense = {name: value.detach().clone().requires_grad_(True) for name, value in bases.items()}
    outputs = []
    start = 0
    for length in LENGTHS:
        inputs = {name: value[:, start : start + length] for name, value in dense.items()}
        outputs.append(_delta_rule(**inputs))
        start += length
    out_dense = torch.cat(outputs, dim=1)
    (out_dense * dy).float().sum().backward()

    result = {"output": _metric(out_packed, out_dense)}
    for name in packed:
        result[f"{name}_grad"] = _metric(packed[name].grad, dense[name].grad)
    return result


def probe_l2norm_launch_partition() -> dict:
    """Compare MCore's joint Q/K launch with mLite's two launches."""
    total = sum(LENGTHS)
    query_key_base = torch.randn(
        1, total, 2 * NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE
    )
    dy = torch.randn_like(query_key_base)

    joint_input = query_key_base.detach().clone().requires_grad_(True)
    joint = l2norm(joint_input.contiguous())
    (joint * dy).float().sum().backward()

    split_input = query_key_base.detach().clone().requires_grad_(True)
    query, key = split_input.chunk(2, dim=2)
    split = torch.cat(
        (l2norm(query.contiguous()), l2norm(key.contiguous())), dim=2
    )
    (split * dy).float().sum().backward()

    return {
        "output": _metric(split, joint),
        "input_grad": _metric(split_input.grad, joint_input.grad),
    }


def main() -> None:
    torch.manual_seed(20260706)
    torch.cuda.manual_seed_all(20260706)
    result = {
        "torch": torch.__version__,
        "lengths": LENGTHS,
        "causal_conv1d": probe_causal_conv(),
        "gated_delta_rule": probe_delta_rule(),
        "l2norm_joint_vs_split": probe_l2norm_launch_partition(),
    }
    flat_values = []
    for section in (
        result["causal_conv1d"],
        result["gated_delta_rule"],
        result["l2norm_joint_vs_split"],
    ):
        stack = [section]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, float):
                flat_values.append(value)
    print("GDN_THD_BACKWARD_PROBE " + json.dumps(result, sort_keys=True), flush=True)
    if not all(math.isfinite(value) for value in flat_values):
        raise RuntimeError("probe produced a non-finite metric")


if __name__ == "__main__":
    main()
