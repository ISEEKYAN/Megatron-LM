# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inference-visible primitives with MLite-owned backward contracts."""

from .autograd import visible_forward
from .compressor import compressor
from .engram import engram_post, engram_reference
from .linear import bf16_fp32_linear, decoded_bf16_master, shared_swiglu
from .mhc import hc_collapse, mhc_coefficients, mhc_joint, mhc_post, reference_post
from .norm import qkv_rms_norm, rms_norm
from .probability import log_softmax
from .routing import router_topk
from .sparse import indexer_scores, indexer_select_positions, sparse_attention

__all__ = [
    'bf16_fp32_linear',
    'compressor',
    'decoded_bf16_master',
    'engram_post',
    'engram_reference',
    'hc_collapse',
    'indexer_scores',
    'indexer_select_positions',
    'log_softmax',
    'mhc_coefficients',
    'mhc_joint',
    'mhc_post',
    'qkv_rms_norm',
    'reference_post',
    'rms_norm',
    'router_topk',
    'shared_swiglu',
    'sparse_attention',
    'visible_forward',
]
