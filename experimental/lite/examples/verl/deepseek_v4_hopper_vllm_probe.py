# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate that the active vLLM binary exposes DeepSeek-V4 support."""

from __future__ import annotations

from importlib.metadata import version


def main() -> None:
    import vllm._C  # noqa: F401  # Force CUDA/Torch ABI resolution.
    from vllm.model_executor.models.registry import ModelRegistry

    architecture = "DeepseekV4ForCausalLM"
    if architecture not in ModelRegistry.get_supported_archs():
        raise RuntimeError(f"vLLM does not register {architecture}")

    print(
        "DS4_HOPPER_VLLM_ENV_PROBE_PASSED "
        f"vllm={version('vllm')} model=deepseek_v4 architecture={architecture}"
    )


if __name__ == "__main__":
    main()
