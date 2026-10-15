# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Training adapter for the inference-owned fixed-K32 CUDA primitive."""


def batched_block32_mm(
    a,
    a_scale,
    weight,
    weight_scale,
    output_dtype,
    *,
    grouped=False,
    weight_scale_group_rows=1
):
    from vllm.model_executor.kernels.batch_invariant import block32_scaled_mm

    return block32_scaled_mm(
        a,
        a_scale,
        weight,
        weight_scale,
        output_dtype,
        grouped=grouped,
        weight_scale_group_rows=weight_scale_group_rows,
    )
