# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fail-closed checks for the DeepSeek-V4 MLite SM100 DSA environment."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any


_MIN_CUDNN_FRONTEND = (1, 26, 0)


def _version_tuple(value: str) -> tuple[int, ...]:
    components = []
    for component in value.split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        components.append(int(digits))
    return tuple(components)


def validate_dependency_contract(
    *,
    capability: tuple[int, int],
    cudnn_frontend_version: str,
    flash_mla_sparse_fwd: Any,
    indexer_fwd_sm100: Any,
) -> dict[str, str | bool]:
    """Validate the exact kernel surface required by the GB200 MLite path."""
    if capability[0] != 10:
        raise RuntimeError(
            f"DS4 MLite validation requires an SM100 GPU, got {capability}"
        )
    if _version_tuple(cudnn_frontend_version) < _MIN_CUDNN_FRONTEND:
        raise RuntimeError(
            "DS4 SM100 requires nvidia-cudnn-frontend >=1.26.0, "
            f"got {cudnn_frontend_version}"
        )
    if flash_mla_sparse_fwd is None:
        raise RuntimeError("FlashMLA does not export flash_mla_sparse_fwd")
    if indexer_fwd_sm100 is None:
        raise RuntimeError("cudnn-frontend does not export indexer_fwd_sm100")
    return {
        "capability": f"{capability[0]}.{capability[1]}",
        "cudnn_frontend": cudnn_frontend_version,
        "flash_mla_sparse_fwd": True,
        "indexer_fwd_sm100": True,
    }


def probe() -> dict[str, str | bool]:
    """Import the production dependencies and verify MLite selects SM100."""
    import torch
    from cudnn.deepseek_sparse_attention.indexer_forward._interface import (
        indexer_fwd as indexer_fwd_sm100,
    )
    from flash_mla import flash_mla_sparse_fwd

    from megatron.lite.primitive.kernels.dsa_kernels import _select_indexer_forward

    device = torch.device("cuda", 0)
    capability = torch.cuda.get_device_capability(device)
    selected = _select_indexer_forward(device)
    if selected is not indexer_fwd_sm100:
        raise RuntimeError("MLite did not select cudnn-frontend indexer_fwd_sm100")
    report = validate_dependency_contract(
        capability=capability,
        cudnn_frontend_version=version("nvidia-cudnn-frontend"),
        flash_mla_sparse_fwd=flash_mla_sparse_fwd,
        indexer_fwd_sm100=indexer_fwd_sm100,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    print("DS4_SM100_DSA_ENV_READY", flush=True)
    return report


def verify_payload(payload: Path, *, expected_prompts: int) -> dict[str, int]:
    """Verify that the formal MLite forward emitted full FP32 distributions."""
    import torch

    rows = torch.load(payload, map_location="cpu", weights_only=True)
    if len(rows) != expected_prompts:
        raise RuntimeError(
            f"expected {expected_prompts} MLite prompt rows, got {len(rows)}"
        )
    token_count = 0
    vocab_size = None
    for index, row in enumerate(rows):
        token_ids = row.get("token_ids")
        logprobs = row.get("logprobs")
        if not isinstance(token_ids, torch.Tensor) or token_ids.dtype != torch.int32:
            raise RuntimeError(f"row {index} token_ids are not int32")
        if not isinstance(logprobs, torch.Tensor) or logprobs.dtype != torch.float32:
            raise RuntimeError(f"row {index} logprobs are not FP32")
        if logprobs.ndim != 2 or logprobs.shape[0] != token_ids.numel():
            raise RuntimeError(f"row {index} is not token aligned")
        if not token_ids.numel() or not logprobs.shape[1]:
            raise RuntimeError(f"row {index} is empty")
        if vocab_size is None:
            vocab_size = int(logprobs.shape[1])
        elif vocab_size != int(logprobs.shape[1]):
            raise RuntimeError("MLite prompt rows use different vocabulary sizes")
        token_count += token_ids.numel()
    report = {
        "prompt_count": len(rows),
        "token_count": token_count,
        "vocab_size": int(vocab_size or 0),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    print("DS4_MLITE_BF16_PAYLOAD_VERIFIED", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe")
    verify_parser = subparsers.add_parser("verify-payload")
    verify_parser.add_argument("--payload", type=Path, required=True)
    verify_parser.add_argument("--expected-prompts", type=int, default=36)
    args = parser.parse_args()
    if args.command == "probe":
        probe()
    else:
        verify_payload(args.payload, expected_prompts=args.expected_prompts)


if __name__ == "__main__":
    main()
