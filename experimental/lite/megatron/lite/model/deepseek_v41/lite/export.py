# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Export DS4.1 weights in the release (= vLLM checkpoint) format for online resync.

Names, dtypes and quantized storage match the release checkpoint, so vLLM's own
``load_weights`` (layerwise reload) consumes the stream exactly as a cold load:

* block-FP8 linears: E4M3 codes + UE8M0 32x32 scales (``quantize_block_fp8``);
* routed experts: MXFP4 codes + UE8M0 1x32 scales (``quantize_mxfp4``);
* everything else in its release dtype (BF16, or FP32 for mHC/sink/router bias).

The quantizers are the ones the deployment forward applies to the BF16 master
each step, so the resynced vLLM weights are the training fake-quant weights.
Frozen Engram tables are never trained and are not exported.
"""
import re

import torch

from megatron.lite.model.deepseek_v41.lite.checkpoint import source_name
from megatron.lite.primitive.ckpt.hf_weights import parse_expert_idx
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

FP8_BLOCK = (32, 32)
_FP8_BLOCK_SUFFIXES = (
    ".attn.wq_a.weight",
    ".attn.wq_b.weight",
    ".attn.wkv.weight",
    ".attn.wo_a.weight",
    ".attn.wo_b.weight",
    ".attn.indexer.wq_b.weight",
    ".ffn.shared_experts.w1.weight",
    ".ffn.shared_experts.w2.weight",
    ".ffn.shared_experts.w3.weight",
    ".engram.wkv.weight",
)
_MXFP4 = re.compile(r"\.ffn\.experts\.\d+\.w[123]\.weight$")
_FP32 = re.compile(r"(\.hc_(attn|ffn)_(fn|base|scale)|\.attn\.attn_sink|\.ffn\.gate\.bias)$")
_EXPERT = re.compile(
    r"^layers\.(\d+)\.ffn\.(experts|shared_experts)\.linear_fc([12])\.weight(\d*)$"
)
_ENGRAM = re.compile(r"^engrams\.(\d+)\.(q_weight|k_weight|wkv\.weight)$")
_ALPHA = ("alpha_pre", "alpha_post", "alpha_res")


def release_format(key):
    """``"mxfp4"``, ``"fp8_block"`` or the release dtype of a release key."""
    if _MXFP4.search(key):
        return "mxfp4"
    if key.endswith(_FP8_BLOCK_SUFFIXES):
        return "fp8_block"
    return torch.float32 if _FP32.search(key) else torch.bfloat16


def release_names(name):
    """Release key(s) for a global native name; ``#slot`` marks one of three mHC scales."""
    if name.endswith(".ape") or name.startswith("hash."):
        return []
    match = _EXPERT.fullmatch(name)
    if match:
        layer, kind, fc, expert = match.groups()
        prefix = f"layers.{layer}.ffn.{kind}." + (f"{expert}." if kind == "experts" else "")
        return [prefix + "w1.weight", prefix + "w3.weight"] if fc == "1" else [prefix + "w2.weight"]
    match = _ENGRAM.fullmatch(name)
    if match:
        return [f"layers.{match[1]}.engram.{match[2]}"]
    if name.startswith("engrams."):
        return []  # frozen tables
    if name.endswith(_ALPHA):
        return [f"{source_name(name)}#{_ALPHA.index(name.rsplit('.', 1)[1])}"]
    return [source_name(name)]


class DeepseekV41WeightSpec:
    """Shared-exporter spec: EP/PP gather of native tensors under release names."""

    def __init__(self, config):
        self.config = config

    @property
    def num_experts(self):
        return self.config.n_routed_experts

    def weight_map(self):
        return {}

    def load_weight_map(self, base_model, ps, logical_state_keys):
        del base_model, ps
        # Only persistent buffers consult this map: the router bias is exported,
        # Engram tables and hash constants are not.
        return {
            name: release_names(name)
            for name in logical_state_keys
            if name.endswith(".ffn.router.expert_bias")
        }

    def native_to_hf(self, native_name, tensor):
        keys = release_names(native_name)
        if len(keys) == 2:
            return list(zip(keys, (t.contiguous() for t in tensor.chunk(2, dim=0))))
        return [(keys[0], tensor)] if keys else []

    def tp_spec(self, native_name):
        return None

    def qkv_spec(self, native_name):
        return None

    def is_expert(self, native_name):
        return ".ffn.experts." in native_name

    def expert_global_id(self, native_name):
        return parse_expert_idx(native_name) if self.is_expert(native_name) else None

    def expert_local_name(self, native_name, local_idx):
        return f"{native_name.rsplit('.weight', 1)[0]}.weight{local_idx}"


def to_release(key, tensor):
    """Serialize one release tensor (+ its scale for quantized storage)."""
    kind = release_format(key)
    if kind == "mxfp4":
        codes, scale = quantize_mxfp4(tensor)
        return [(key, codes), (key[: -len("weight")] + "scale", scale)]
    if kind == "fp8_block":
        codes, scale = quantize_block_fp8(tensor, FP8_BLOCK, scale_format="e8m0")
        return [(key, codes), (key[: -len("weight")] + "scale", scale)]
    return [(key, tensor.to(kind).contiguous())]


def export_hf_weights(chunks, model_cfg, ps, **kwargs):
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights as gather

    target = kwargs.pop("target", "release")
    if target not in ("release", "hf", "vllm"):
        raise ValueError(f"DS4.1 exports only the release/vLLM checkpoint format, got {target!r}")
    if kwargs.pop("resync_config", None) or kwargs.get("export_dtype"):
        raise ValueError("DS4.1 release export owns dtypes and quantization; no overrides")
    kwargs.pop("export_dtype", None)
    scales = {}
    for key, tensor in gather(
        chunks, DeepseekV41WeightSpec(model_cfg), ps, vocab_size=model_cfg.vocab_size, **kwargs
    ):
        if "#" in key:
            key, slot = key.split("#")
            parts = scales.setdefault(key, [None] * 3)
            parts[int(slot)] = tensor.reshape(1)
            if any(p is None for p in parts):
                continue
            tensor = torch.cat(scales.pop(key))
        yield from to_release(key, tensor)
    if scales:
        raise RuntimeError(f"Incomplete mHC scale export: {sorted(scales)[:4]}")


def save_hf_weights(chunks, path, model_cfg, ps, **kwargs):
    from megatron.lite.primitive.ckpt.hf_weights import stream_export_to_shards

    kwargs.pop("cpu", None)
    shard = int(kwargs.pop("shard_size_bytes", 5 * 1024**3))
    stream_export_to_shards(
        export_hf_weights(chunks, model_cfg, ps, rank0_only=True, cpu=True, **kwargs),
        path,
        shard_size_bytes=shard,
    )
