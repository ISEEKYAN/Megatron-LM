# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment arithmetic over the shared DS4.1 training state."""


def load_vllm_flashmla():
    """Import vLLM's FlashMLA ops; they must load before pip ``flash_mla``.

    Both extensions define the same C++ kernel symbols and the copy loaded first
    serves both, so pip FlashMLA (imported by Megatron Core attention modules)
    loaded first makes vLLM's paged BI kernel reject TOPK_LENGTH/BATCH_INVARIANT.
    """
    import sys

    loaded = list(sys.modules)  # insertion order = load order
    pip, ours = "flash_mla.cuda", "vllm._flashmla_C"
    if pip in loaded and (ours not in loaded or loaded.index(pip) < loaded.index(ours)):
        raise RuntimeError(
            "pip flash_mla was loaded before vLLM's FlashMLA, whose BI kernel then "
            "fails; import vllm.v1.attention.ops.flashmla first (the VERL MLite "
            "sitecustomize does when DSV41_ATTENTION_CORE=flashmla)"
        )
    import vllm.v1.attention.ops.flashmla  # noqa: F401
