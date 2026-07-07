# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Offline fp64 ground-truth verdict for dumped linear-layer wgrads.

Usage:
    python analyze_wgrad_dumps.py --mlite DIR --mcore DIR [--output-json PATH]

For every dumped module we recompute the exact wgrad in float64 from that
side's own captured (X, dY) and compare each backend's actual wgrad against
its own fp64 ground truth.  Key outputs per (layer, family):

  rel_err     ||wgrad_actual - ref64(own X,dY)|| / ||ref64||   -> convicts the
              GEMM/accumulation path of that side if large
  cancel      ||ref64|| / || |dY|^T |X| ||                     -> how strongly
              the token sum cancels (small = noise-floor dominated)
  ref_ratio   ||ref64_mlite|| / ||ref64_mcore||                -> divergence
              already present in the *inputs* (upstream), not the GEMM
  wg_ratio    ||wgrad_mlite|| / ||wgrad_mcore||                -> the observed
              family-norm divergence at this layer
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def _load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def _collect(dump_dir: Path) -> dict[tuple[int, str], dict[str, Path]]:
    """Map (layer_idx, family) -> {'X': path, 'dY': path, 'wgrad': path}."""
    out: dict[tuple[int, str], dict[str, Path]] = {}
    for path in sorted(dump_dir.glob("*.pt")):
        name = path.name
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None:
            continue
        layer = int(m.group(1))
        if "o_proj" in name or "out_proj" in name:
            family = "gdn_oproj"
        elif "full_attn.proj" in name or "linear_proj" in name:
            family = "attn_oproj"
        else:
            continue
        entry = out.setdefault((layer, family), {})
        if ".X." in name:
            idx = int(name.rsplit(".", 2)[-2])
            cur = entry.get("_x_idx", -1)
            if idx > cur:
                entry["X"] = path
                entry["_x_idx"] = idx
        elif ".dY." in name:
            idx = int(name.rsplit(".", 2)[-2])
            cur = entry.get("_dy_idx", 10**9)
            if idx < cur:
                entry["dY"] = path
                entry["_dy_idx"] = idx
        elif name.endswith(".weight.wgrad.pt"):
            entry["wgrad"] = path
    return out


def _side_metrics(entry: dict[str, Path]) -> dict | None:
    if not {"X", "dY", "wgrad"} <= set(entry):
        return None
    x = _load(entry["X"]).double()
    dy = _load(entry["dY"]).double()
    wg = _load(entry["wgrad"])
    wg_dtype = str(wg.dtype)
    wg = wg.double()
    x2 = x.reshape(-1, x.shape[-1])
    dy2 = dy.reshape(-1, dy.shape[-1])
    if dy2.shape[0] != x2.shape[0]:
        return {"error": f"token mismatch X={tuple(x2.shape)} dY={tuple(dy2.shape)}"}
    ref = dy2.t() @ x2
    if ref.shape != wg.shape:
        if ref.t().shape == wg.shape:
            ref = ref.t()
            absref = x2.abs().t() @ dy2.abs()
        else:
            return {"error": f"shape mismatch ref={tuple(ref.shape)} wgrad={tuple(wg.shape)}"}
    else:
        absref = dy2.abs().t() @ x2.abs()
    n_ref = ref.norm().item()
    n_wg = wg.norm().item()
    n_abs = absref.norm().item()
    return {
        "tokens": int(x2.shape[0]),
        "wgrad_dtype": wg_dtype,
        "norm_X": x2.norm().item(),
        "norm_dY": dy2.norm().item(),
        "norm_wgrad": n_wg,
        "norm_ref64": n_ref,
        "rel_err_vs_own_truth": (wg - ref).norm().item() / max(n_ref, 1e-300),
        "cancel": n_ref / max(n_abs, 1e-300),
        "_ref": ref,
        "_wg": wg,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlite", required=True)
    ap.add_argument("--mcore", required=True)
    ap.add_argument("--output-json", default=None)
    ns = ap.parse_args()

    ml = _collect(Path(ns.mlite))
    mc = _collect(Path(ns.mcore))
    keys = sorted(set(ml) | set(mc))
    rows = []
    for key in keys:
        layer, family = key
        row: dict = {"layer": layer, "family": family}
        m_side = _side_metrics(ml[key]) if key in ml else None
        c_side = _side_metrics(mc[key]) if key in mc else None
        for tag, side in (("mlite", m_side), ("mcore", c_side)):
            if side is None:
                row[tag] = None
                continue
            row[tag] = {k: v for k, v in side.items() if not k.startswith("_")}
        if m_side and c_side and "_ref" in m_side and "_ref" in c_side:
            row["wg_ratio"] = m_side["norm_wgrad"] / max(c_side["norm_wgrad"], 1e-300)
            row["ref_ratio"] = m_side["norm_ref64"] / max(c_side["norm_ref64"], 1e-300)
            if m_side["_ref"].shape == c_side["_ref"].shape:
                cross = (m_side["_ref"] - c_side["_ref"]).norm().item()
                row["cross_ref_rel"] = cross / max(c_side["norm_ref64"], 1e-300)
                wg_m, wg_c = m_side["_wg"].flatten(), c_side["_wg"].flatten()
                denom = wg_m.norm().item() * wg_c.norm().item()
                row["wg_cosine"] = float((wg_m @ wg_c).item() / max(denom, 1e-300))
        rows.append(row)

    text = json.dumps(rows, indent=2)
    print(text, flush=True)
    if ns.output_json:
        Path(ns.output_json).write_text(text + "\n", encoding="utf-8")

    print("\n=== VERDICT SUMMARY (per layer/family) ===")
    print(
        f"{'ly':>3} {'family':<11} {'wg_ratio':>9} {'ref_ratio':>9} "
        f"{'err_mlite':>10} {'err_mcore':>10} {'cancel_ml':>10} {'cancel_mc':>10}"
    )
    for row in rows:
        m, c = row.get("mlite"), row.get("mcore")
        if not m or not c or "rel_err_vs_own_truth" not in m or "rel_err_vs_own_truth" not in c:
            print(f"{row['layer']:>3} {row['family']:<11} (incomplete: {m and 'm'}{c and 'c'})")
            continue
        print(
            f"{row['layer']:>3} {row['family']:<11} {row.get('wg_ratio', float('nan')):>9.4f} "
            f"{row.get('ref_ratio', float('nan')):>9.4f} {m['rel_err_vs_own_truth']:>10.3e} "
            f"{c['rel_err_vs_own_truth']:>10.3e} {m['cancel']:>10.3e} {c['cancel']:>10.3e}"
        )


if __name__ == "__main__":
    main()
