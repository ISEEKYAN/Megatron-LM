"""CPU semantic probes; extracts actual functions without importing GPU dependencies."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-model", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.official_model.read_bytes()).hexdigest() == (
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    ), "Official source snapshot changed; re-audit before updating the pin"
    root = Path(__file__).resolve().parents[1]
    ns = dict(torch=torch, Tensor=torch.Tensor, math=math, Any=object)
    extract(
        root / "megatron/lite/primitive/utils/rotary.py",
        [
            "_yarn_find_correction_dim",
            "_yarn_find_correction_range",
            "_yarn_linear_ramp_mask",
        ],
        ns,
    )
    extract(
        root / "megatron/lite/primitive/modules/attention/csa.py",
        [
            "build_rope_cos_sin",
            "build_yarn_rope_cos_sin",
            "build_compressed_rope_cos_sin",
            "apply_partial_rope",
        ],
        ns,
    )
    extract(args.official_model, ["precompute_freqs_cis", "apply_rotary_emb"], ns)
    cfg = SimpleNamespace(
        rotary_scaling_factor=16,
        beta_fast=32,
        beta_slow=1,
        original_max_position_embeddings=65536,
    )
    positions = torch.tensor([[0, 1, 127, 128, 65535, 65536, 131071, 1048575]])
    torch.manual_seed(41)
    x = torch.randn(1, positions.numel(), 2, 512)
    results = {}
    for ratio in (0, 1, 2):
        base = 160000 if ratio else 10000
        freq = ns["precompute_freqs_cis"](
            64, 1048576, 65536 if ratio else 0, base, 16, 32, 1
        )[positions[0]]
        ref = x.clone()
        ns["apply_rotary_emb"](ref[..., -64:], freq)

        def run(theta, yarn):
            cos, sin = ns["build_compressed_rope_cos_sin"](
                positions,
                64,
                theta,
                config=cfg,
                use_yarn=yarn,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            actual = ns["apply_partial_rope"](
                x.transpose(1, 2), cos, sin, 64
            ).transpose(1, 2)
            return actual, cos, sin

        actual, cos, sin = run(base, ratio != 0)
        torch.testing.assert_close(actual, ref, atol=1e-6, rtol=1e-6)
        restored = ns["apply_partial_rope"](
            actual.transpose(1, 2), cos, -sin, 64
        ).transpose(1, 2)
        torch.testing.assert_close(restored, x, atol=1e-6, rtol=1e-6)
        legacy, _, _ = run(160000 if ratio > 1 else 10000, ratio > 1)
        per_pos = (legacy - ref).abs().amax(dim=(0, 2, 3)).tolist()
        if ratio == 1:
            assert (
                max(per_pos[1:]) > 1
            ), "Negative control did not detect ratio=1 regression"
        else:
            torch.testing.assert_close(legacy, ref, atol=1e-6, rtol=1e-6)
        results[str(ratio)] = {
            "official_max_error": (actual - ref).abs().max().item(),
            "legacy_error_by_position": per_pos,
        }
    # A per-head RMS after wq_b changes attention probabilities, even before RoPE.
    q = torch.tensor([2.0, 0.0])
    k = torch.eye(2)
    official = (k @ q / math.sqrt(2)).softmax(-1)
    normalized = (
        k @ (q * torch.rsqrt(q.square().mean() + 1e-20)) / math.sqrt(2)
    ).softmax(-1)
    assert (official - normalized).abs().max() > 0.05
    # Exact index identity: both contractions sum d within each group, never across g.
    a = torch.randn(2, 3, 8, 16, dtype=torch.float64)
    w = torch.randn(8, 5, 16, dtype=torch.float64)
    grouped = torch.einsum("...gd,god->...go", a, w)
    released = torch.einsum("bsgd,grd->bsgr", a, w)
    torch.testing.assert_close(grouped, released, atol=0, rtol=0)
    wrong = torch.einsum("bsgd,grd->bsgr", a.flip(2), w)
    assert (wrong - released).abs().max() > 1
    print(
        json.dumps(
            {
                "positions": positions.tolist()[0],
                "rope": results,
                "q_rms_probability_delta": (official - normalized).abs().max().item(),
                "grouped_projection_max_error": (grouped - released).abs().max().item(),
            },
            indent=2,
        )
    )
    print("CSA2_CPU_SEMANTIC_PROBES_OK (not GPU or training parity)")


if __name__ == "__main__":
    main()
