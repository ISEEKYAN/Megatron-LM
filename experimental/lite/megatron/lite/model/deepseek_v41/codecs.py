# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The release's four quantization surfaces and their only supported codecs."""
from types import MappingProxyType

from megatron.lite.primitive.quantization import mxfp4, mxfp8, nvfp4

CODECS = MappingProxyType(
    {
        ("main_kv", 16, "e4m3", "e2m1"): nvfp4.fake_quant_main_kv,
        ("index", 32, "e8m0", "e2m1"): mxfp4.fake_quant_index,
        ("swa", 32, "e8m0", "e4m3"): mxfp8.fake_quant_swa,
        ("linear", 32, "e8m0", "e4m3"): mxfp8.dynamic_fp8_linear,
    }
)


def attention_codecs():
    return {
        "main": CODECS[("main_kv", 16, "e4m3", "e2m1")],
        "index": CODECS[("index", 32, "e8m0", "e2m1")],
        "swa": CODECS[("swa", 32, "e8m0", "e4m3")],
    }
