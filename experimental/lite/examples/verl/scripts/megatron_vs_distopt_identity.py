# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SUPERSEDED single-process identity check — kept for provenance only.

>>> This script runs in ONE process with `pg_collection=None` (local Newton-Schulz).
>>> The moe panel correctly rejected that as not exercising the REAL distributed path.
>>> The admissible receipt is now `tp_distributed_muon_identity.py`, which proves the
>>> same construction identity under a live TP=2 process group with cross-rank
>>> all_reduce (job 14245178). Prefer that; this file is NOT the evidence of record.

Megatron-native Muon vs DistOpt-mlite Muon: real numerical identity receipt (AC#3(a)).

Context (moe BLOCKER C-BITWISE-REDEFINED): the earlier delivery asserted a
"bitwise by construction" verdict from prose alone ("same get_megatron_optimizer
path") with **no independent arm and no torch.equal evidence**. That is not
admissible. This script produces the missing real numerical evidence, honestly
scoped:

  MLite's DistOpt Muon is *not a second implementation* of Megatron Muon -- it
  lowers, through `build_dist_opt_optimizer_config`, into Megatron-Core's own
  `TensorParallelMuon` (`megatron/core/optimizer/emerging_optimizers.py`). There
  is no independent Megatron binary to diff. So this is a **construction
  identity**, and we prove it *numerically* rather than by assertion:

  (a) CONFIG IDENTITY -- the Megatron-Core `OptimizerConfig` produced by the
      MLite lowering (path A) is field-for-field equal to a directly hand-built
      native Megatron `OptimizerConfig` (path B), across every muon_* knob plus
      lr/weight_decay/clip. This is the regression guard that the muon_tp_mode
      propagation fix (this task) is about: if the lowering dropped any muon_*
      field, A != B here.

  (b) UPDATE IDENTITY (torch.equal) -- we build Megatron-Core's *native*
      `TensorParallelMuon` from BOTH configs, using Megatron's own
      `_kwargs_from_config` mapper, and step each on identical seeded params and
      grads. Final weights are compared with `torch.equal`. Bit-identical ==>
      the MLite lowering perturbs the Megatron Muon update by exactly zero.

  (c) NEGATIVE CONTROL -- a config with num_ns_steps perturbed by 1 yields a
      >0 max_abs weight delta, proving the torch.equal in (b) is *sensitive*
      (not a vacuous pass).

This deliberately does NOT claim a bitwise diff of two independent lowerings.
The only genuinely independent lowering (FSDP2 Muon) is deferred to the
1.13.5.5.6 rewrite per bayan's 05:29/05:45 directive and is not exercised here.
"""

from __future__ import annotations

import sys

import torch  # pyright: ignore[reportMissingImports]

from megatron.core.optimizer.emerging_optimizers import (  # pyright: ignore[reportMissingImports]
    TensorParallelMuon,
    _kwargs_from_config,
)
from megatron.core.optimizer.optimizer_config import (  # pyright: ignore[reportMissingImports]
    OptimizerConfig as CoreOptimizerConfig,
)
from megatron.lite.primitive.optimizers.megatron_wrap import (  # pyright: ignore[reportMissingImports]
    build_dist_opt_optimizer_config,
)
from megatron.lite.runtime.contracts.config import (  # pyright: ignore[reportMissingImports]
    OptimizerConfig as LiteOptimizerConfig,
)

# Same-contract hyperparameters as the GPU harness, with several NON-default muon
# knobs so the identity check is non-trivial (a lowering that silently reverts to
# a default would fail config identity + the update torch.equal).
HP = dict(
    lr=1e-5,
    weight_decay=0.1,
    clip_grad=1.0,
    muon_tp_mode="distributed",
    muon_momentum=0.9,          # non-default (default 0.95)
    muon_nesterov=True,
    muon_num_ns_steps=6,        # non-default (default 5)
    muon_coefficient_type="quintic",
    muon_scale_mode="spectral",
    muon_fp32_matmul_prec="high",  # non-default (default "medium")
    muon_extra_scale_factor=1.0,
)

# Every muon_* knob the MLite lowering is responsible for forwarding, plus the
# scalar training knobs. These are the fields whose identity we assert.
IDENTITY_FIELDS = [
    "optimizer",
    "lr",
    "weight_decay",
    "clip_grad",
    "muon_momentum",
    "muon_split_qkv",
    "muon_nesterov",
    "muon_scale_mode",
    "muon_fp32_matmul_prec",
    "muon_coefficient_type",
    "muon_num_ns_steps",
    "muon_tp_mode",
    "muon_extra_scale_factor",
    "muon_scalar_optimizer",
]


def _make_lite_config(**overrides):
    hp = dict(HP)
    hp.update(overrides)
    # LiteOptimizerConfig maps optimizer= -> optimizer_algorithm; set the rest by
    # constructor if accepted, else setattr (duck-typed lite path).
    try:
        return LiteOptimizerConfig(optimizer="muon", **hp)
    except TypeError:
        cfg = LiteOptimizerConfig(optimizer="muon", lr=hp["lr"], weight_decay=hp["weight_decay"], clip_grad=hp["clip_grad"])
        for k, v in hp.items():
            if k in ("lr", "weight_decay", "clip_grad"):
                continue
            setattr(cfg, k, v)
        return cfg


def _make_native_core_config(**overrides):
    hp = dict(HP)
    hp.update(overrides)
    return CoreOptimizerConfig(optimizer="muon", **hp)


def _config_identity(core_a, core_b):
    diffs = []
    for f in IDENTITY_FIELDS:
        va, vb = getattr(core_a, f, "<missing>"), getattr(core_b, f, "<missing>")
        if va != vb:
            diffs.append((f, va, vb))
    return diffs


def _build_native_muon(core_cfg, params):
    """Build Megatron-Core's native TensorParallelMuon via its own config mapper."""
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", core_cfg)
    kwargs["is_qkv_fn"] = lambda p: False
    kwargs["qkv_split_shapes"] = None
    kwargs["pg_collection"] = None  # single-tensor, unsharded: local Newton-Schulz
    return TensorParallelMuon(params, **kwargs)


def _run_muon(core_cfg, *, steps=5, seed=1234):
    """Step native TensorParallelMuon on identical seeded params+grads; return final weight."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(64, 48, generator=gen, dtype=torch.float32)
    p = torch.nn.Parameter(w.clone())
    opt = _build_native_muon(core_cfg, [p])
    ggen = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        p.grad = torch.randn(64, 48, generator=ggen, dtype=torch.float32)
        opt.step()
    return p.detach().clone()


def main() -> int:
    torch.use_deterministic_algorithms(False)

    lite = _make_lite_config()
    core_a = build_dist_opt_optimizer_config(lite)   # PATH A: MLite DistOpt lowering
    core_b = _make_native_core_config()              # PATH B: native Megatron config

    # (a) CONFIG IDENTITY
    diffs = _config_identity(core_a, core_b)
    print(f"[a] CONFIG_IDENTITY fields_checked={len(IDENTITY_FIELDS)} diffs={diffs}")
    if diffs:
        print("RESULT FAIL: config identity broken (MLite lowering dropped/altered a field)")
        return 1
    print("[a] CONFIG_IDENTITY_OK all fields equal (MLite lowering == native Megatron config)")

    # (b) UPDATE IDENTITY via native TensorParallelMuon driven by each config
    wa = _run_muon(core_a, steps=5, seed=1234)
    wb = _run_muon(core_b, steps=5, seed=1234)
    equal = torch.equal(wa, wb)
    max_abs = (wa - wb).abs().max().item()
    print(f"[b] UPDATE_IDENTITY torch.equal={equal} max_abs_delta={max_abs:.3e} "
          f"weight_shape={tuple(wa.shape)} steps=5")
    if not equal:
        print("RESULT FAIL: update identity broken")
        return 1
    print("[b] UPDATE_IDENTITY_OK final weights bit-identical (torch.equal)")

    # (c) NEGATIVE CONTROL: perturb num_ns_steps -> update MUST differ
    core_c = _make_native_core_config(muon_num_ns_steps=HP["muon_num_ns_steps"] + 1)
    wc = _run_muon(core_c, steps=5, seed=1234)
    neg_equal = torch.equal(wa, wc)
    neg_max_abs = (wa - wc).abs().max().item()
    print(f"[c] NEG_CONTROL(num_ns_steps+1) torch.equal={neg_equal} max_abs_delta={neg_max_abs:.3e}")
    if neg_equal or neg_max_abs == 0.0:
        print("RESULT FAIL: negative control did not diverge -> torch.equal is not sensitive")
        return 1
    print("[c] NEG_CONTROL_OK update torch.equal is sensitive (perturbation -> nonzero delta)")

    print("RESULT PASS: Megatron-native == DistOpt-mlite Muon "
          "(config identity + update torch.equal, sensitivity-controlled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
