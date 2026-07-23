# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU qualification smoke for the chunk-wise CUDA Graph controller spine.

The CPU unit tests cover the pure-CPU architecture (qualify / slot plan / replay
signature). This script qualifies the one thing the CPU box cannot reach: the
TE-backed capture/replay machinery on a real H100 + Transformer Engine.

Stage B (this script) captures a real ``transformer_engine.pytorch.Linear`` via
the controller's ``_capture_slot`` (which calls ``make_graphed_callables``),
replays it via ``get_graphed``, and checks the replayed forward matches the
eager forward. This is the direct hardware proof that the delivered capture /
replay path is wired correctly end to end.

Evidence discipline: a verdict JSON is always written; the ``NON_SKIP`` marker is
created *only* when the capture actually executed on CUDA+TE (skip != pass).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_EXPERIMENTAL_LITE_ROOT = Path(__file__).resolve().parents[2]
if str(_EXPERIMENTAL_LITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTAL_LITE_ROOT))

from megatron.lite.primitive.cuda_graph import CudaGraphController, CudaGraphError


def _capture_replay_stage(
    *,
    hidden: int,
    tokens: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Capture a real TE Linear via the controller and compare replay vs eager."""
    import transformer_engine.pytorch as te  # noqa: WPS433 (GPU-only import)

    torch.manual_seed(0)
    module = te.Linear(hidden, hidden, bias=True, params_dtype=dtype).cuda()

    # Eager reference on an independent, detached input (same values as replay
    # input) so the two forwards see identical weights and identical activations.
    base = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
    x_eager = base.detach().clone().requires_grad_(True)
    with torch.no_grad():
        out_eager = module(x_eager.detach())

    # One local model chunk; a single (chunk=0, slot=0) capture is all the
    # machinery needs to exercise the TE make_graphed_callables path.
    controller = CudaGraphController(
        chunks=[object()],
        num_warmup_microbatches=1,
        num_microbatches=1,
    )
    x_capture = base.detach().clone().requires_grad_(True)
    # TE make_graphed_callables needs warmup iters to populate need_bwd_dw_graph;
    # the controller default is 3 (job 13875020 failed with num_warmup_iters=0).
    controller._capture_slot(0, 0, module, (x_capture,), None)
    graphed = controller.get_graphed(0, 0)
    if graphed is None:
        raise CudaGraphError("controller.get_graphed(0, 0) returned None after capture")

    x_replay = base.detach().clone().requires_grad_(True)
    out_replay = graphed(x_replay)

    diff = (out_replay.detach().float() - out_eager.detach().float()).abs()
    max_abs = float(diff.max().item())
    allclose = torch.allclose(
        out_replay.detach().float(), out_eager.detach().float(), atol=atol, rtol=rtol
    )
    return {
        "stage": "capture_replay_te_linear",
        "device": torch.cuda.get_device_name(0),
        "hidden": hidden,
        "tokens": tokens,
        "dtype": str(dtype),
        "num_slots": controller.num_slots,
        "captured_slots": len(controller._graphed),
        "max_abs_diff": max_abs,
        "atol": atol,
        "rtol": rtol,
        "allclose": bool(allclose),
        "output_shape": list(out_replay.shape),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--marker", default=None, help="NON_SKIP marker path on pass")
    args = parser.parse_args(argv)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        verdict = {"skipped": True, "reason": "CUDA unavailable", "passed": False}
        out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        print(json.dumps(verdict, sort_keys=True))
        return 2  # skip != pass

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    try:
        result = _capture_replay_stage(
            hidden=args.hidden,
            tokens=args.tokens,
            dtype=dtype,
            atol=args.atol,
            rtol=args.rtol,
        )
    except Exception as exc:  # noqa: BLE001 — surface the real failure in the verdict
        verdict = {"skipped": False, "passed": False, "error": repr(exc)}
        out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        print(json.dumps(verdict, sort_keys=True))
        raise

    passed = bool(result["allclose"])
    verdict = {"skipped": False, "passed": passed, "result": result}
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, sort_keys=True))

    if passed and args.marker:
        Path(args.marker).write_text("NON_SKIP_CUDA_GRAPH_CAPTURE_REPLAY_PASSED\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
