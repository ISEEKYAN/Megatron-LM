#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate the fixed DeepSeek-V4 DAPO launch contract."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

from packaging.version import Version


EXACT_DEPENDENCIES = {
    "vllm": "0.26.1rc1.dev682+g7aa248fcf",
    "flashinfer-python": "0.6.16.post3",
    "nvidia-cutlass-dsl": "4.6.2",
    "tilelang": "0.1.12",
}
MINIMUM_DEPENDENCIES = {
    "transformer-engine": "2.15.0",
    "nvidia-cudnn-frontend": "1.27.0",
}
EXPECTED_VERL_COMMIT = "01d2ab350c459c78f0350d3679d442e2a8ddba64"


def installed(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise SystemExit(f"DS4 dependency is missing: {name}") from exc


def validate_geometry(
    model_config: Path,
    rollout_tp: int,
    world_size: int,
    actor_pp: int,
    actor_cp: int,
    actor_ep: int,
    max_sequence_length: int,
    ppo_max_tokens_per_gpu: int,
) -> None:
    config = json.loads(model_config.read_text(encoding="utf-8"))
    o_groups = config.get("o_groups")
    if not isinstance(o_groups, int) or o_groups < 1 or o_groups % rollout_tp != 0:
        raise SystemExit(
            f"DS4 o_groups={o_groups} must be divisible by rollout_tp={rollout_tp}"
        )
    expected_topologies = {
        4: (1, 1, 4),
        32: (4, 2, 8),
        64: (4, 2, 8),
    }
    expected_topology = expected_topologies.get(world_size)
    if expected_topology is None:
        raise SystemExit(f"unsupported DS4 world_size={world_size}; expected 4, 32, or 64")
    for name, value in (("actor_pp", actor_pp), ("actor_cp", actor_cp), ("actor_ep", actor_ep)):
        if value < 1 or world_size % value:
            raise SystemExit(f"{name}={value} must divide world_size={world_size}")
    if (actor_pp, actor_cp, actor_ep) != expected_topology:
        raise SystemExit(
            f"DS4 world_size={world_size} actor topology must be "
            f"PP{expected_topology[0]}/CP{expected_topology[1]}/EP{expected_topology[2]}, got "
            f"PP{actor_pp}/CP{actor_cp}/EP{actor_ep}"
        )
    if max_sequence_length != 8192 or ppo_max_tokens_per_gpu != 4096:
        raise SystemExit(
            "DS4 production token contract requires max_sequence_length=8192 "
            f"and ppo_max_tokens_per_gpu=4096, got {max_sequence_length}/"
            f"{ppo_max_tokens_per_gpu}"
        )
    print(
        f"DS4_ROLLOUT_TP_PREFLIGHT_PASSED o_groups={o_groups} rollout_tp={rollout_tp}",
        flush=True,
    )


def verl_commit() -> str:
    declared = os.environ.get("VERL_COMMIT")
    root = os.environ.get("VERL_ROOT")
    if declared is None and root:
        try:
            declared = subprocess.check_output(
                ["git", "-C", root, "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if declared is None:
        raise SystemExit(
            f"VERL provenance is not verifiable; set VERL_COMMIT={EXPECTED_VERL_COMMIT}"
        )
    if not declared.startswith(EXPECTED_VERL_COMMIT):
        raise SystemExit(f"DS4 requires VERL {EXPECTED_VERL_COMMIT}, got {declared}")
    return declared


def validate_environment() -> None:
    actual = {
        name: installed(name) for name in EXACT_DEPENDENCIES | MINIMUM_DEPENDENCIES
    }
    bad_exact = {
        name: (actual[name], wanted)
        for name, wanted in EXACT_DEPENDENCIES.items()
        if Version(actual[name]) != Version(wanted)
    }
    bad_minimum = {
        name: (actual[name], wanted)
        for name, wanted in MINIMUM_DEPENDENCIES.items()
        if Version(actual[name]) < Version(wanted)
    }
    if bad_exact or bad_minimum:
        raise SystemExit(
            f"DS4 dependency mismatch: exact={bad_exact} minimum={bad_minimum}"
        )
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"DS4 requires Python 3.12, got {sys.version}")

    import cudnn
    import torch
    import transformer_engine.pytorch as te
    from cudnn import DSA

    if torch.version.cuda is None or Version(torch.version.cuda) < Version("12.8"):
        raise SystemExit(
            "DS4 requires CUDA-enabled PyTorch with CUDA >=12.8, "
            f"got {torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("DS4 requires a visible CUDA device")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 9:
        raise SystemExit(f"DS4 requires SM90 or newer, got capability={capability}")
    if "q_causal_offsets" not in inspect.signature(
        DSA.indexer_forward_wrapper
    ).parameters:
        raise SystemExit(
            "nvidia-cudnn-frontend lacks q_causal_offsets required by fused DSA CP"
        )

    declared_verl = verl_commit()
    print(
        "DS4_DEPENDENCY_CONTRACT_PASSED "
        f"verl={declared_verl} python={sys.version.split()[0]} "
        f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
        f"gpu_capability={capability[0]}.{capability[1]} "
        f"vllm={actual['vllm']} te={actual['transformer-engine']} "
        f"cudnn_frontend={actual['nvidia-cudnn-frontend']} "
        f"flashinfer={actual['flashinfer-python']} "
        f"cutlass={actual['nvidia-cutlass-dsl']} tilelang={actual['tilelang']} "
        f"te_origin={te.__file__} cudnn_origin={cudnn.__file__}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    geometry = subparsers.add_parser("geometry")
    geometry.add_argument("--model-config", type=Path, required=True)
    geometry.add_argument("--rollout-tp", type=int, required=True)
    geometry.add_argument("--world-size", type=int, required=True)
    geometry.add_argument("--actor-pp", type=int, required=True)
    geometry.add_argument("--actor-cp", type=int, required=True)
    geometry.add_argument("--actor-ep", type=int, required=True)
    geometry.add_argument("--max-sequence-length", type=int, required=True)
    geometry.add_argument("--ppo-max-tokens-per-gpu", type=int, required=True)
    subparsers.add_parser("environment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "geometry":
        validate_geometry(
            args.model_config,
            args.rollout_tp,
            args.world_size,
            args.actor_pp,
            args.actor_cp,
            args.actor_ep,
            args.max_sequence_length,
            args.ppo_max_tokens_per_gpu,
        )
    else:
        validate_environment()


if __name__ == "__main__":
    main()
