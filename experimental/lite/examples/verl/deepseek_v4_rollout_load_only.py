# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Load the DeepSeek V4 vLLM rollout model without starting RL workers."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])
    rollout_tp = int(os.environ["ROLLOUT_TP"])
    with (checkpoint_dir / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    o_groups = config.get("o_groups")
    if not isinstance(o_groups, int) or o_groups < 1:
        raise RuntimeError(f"DeepSeek V4 config has invalid o_groups={o_groups!r}")
    if rollout_tp < 1 or o_groups % rollout_tp != 0:
        raise RuntimeError(
            f"DeepSeek V4 o_groups={o_groups} must be divisible by "
            f"positive rollout_tp={rollout_tp}"
        )

    from vllm import LLM

    LLM(
        model=str(checkpoint_dir),
        tensor_parallel_size=rollout_tp,
        load_format="dummy",
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=384,
        max_num_seqs=32,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=float(
            os.environ.get("ROLLOUT_GPU_MEMORY_UTILIZATION", "0.60")
        ),
        kv_cache_dtype="fp8",
        enforce_eager=True,
        disable_log_stats=True,
        hf_overrides={
            "expert_dtype": "fp8",
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "scale_fmt": "float32",
                "weight_block_size": [128, 128],
            },
        },
    )
    print(
        "DS4_VLLM_LOAD_ONLY_PASSED "
        f"rollout_tp={rollout_tp} o_groups={o_groups} "
        f"local_groups={o_groups // rollout_tp}",
        flush=True,
    )


if __name__ == "__main__":
    main()
