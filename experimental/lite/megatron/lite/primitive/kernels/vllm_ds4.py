# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Strict, optional adapters for the vLLM DeepSeek-V4 inference kernels.

The adapters in this module deliberately do not construct vLLM model objects,
read vLLM forward context, register parameters, or provide numerical fallbacks.
All optional dependencies are imported at the point of use.
"""

from __future__ import annotations

import importlib
import math
import os
import types
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

import torch
from torch import Tensor, nn

from megatron.lite.primitive.autograd import inference_only
from megatron.lite.primitive.quantization.deployment_block_fp8 import (
    quantize_block_fp8_weight,
)


def _symbol(module: str, name: str) -> Any:
    try:
        value = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            f"vLLM kernel entry {module}.{name} is unavailable; "
            "install the matching vLLM build and its compiled dependencies."
        ) from exc
    return value


def _op(namespace: str, name: str) -> Callable[..., Any]:
    try:
        op = getattr(getattr(torch.ops, namespace), name)
    except AttributeError as exc:
        raise NotImplementedError(
            f"required torch.ops.{namespace}.{name} is not registered by this vLLM build"
        ) from exc
    return op


def _tensors(values: Iterable[Any]) -> Iterable[Tensor]:
    for value in values:
        if isinstance(value, Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            yield from _tensors(value)


def _inference_only(*values: Any) -> None:
    # Do not reject trainable inputs before running the deployment kernel.
    # Callers attach ``primitive.autograd.inference_only`` to the produced
    # tensor, which preserves the real forward value and rejects backward at
    # the output boundary.  Keeping this helper as a no-op avoids changing the
    # validation call sites in the individual adapters.
    del values


def _output_boundary(value: Any, *dependencies: Any) -> Any:
    deps = tuple(_tensors(dependencies))
    if isinstance(value, Tensor):
        if value.requires_grad or any(tensor.requires_grad for tensor in deps):
            return inference_only(value, *deps)
        return value
    if isinstance(value, tuple):
        return tuple(_output_boundary(item, *deps) for item in value)
    if isinstance(value, list):
        return [_output_boundary(item, *deps) for item in value]
    return value


def _tensor(
    value: Tensor,
    name: str,
    *,
    ndim: int | None = None,
    dtype: torch.dtype | tuple[torch.dtype, ...] | None = None,
    contiguous: bool = True,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(value.shape)}")
    if dtype is not None:
        allowed = (dtype,) if isinstance(dtype, torch.dtype) else dtype
        if value.dtype not in allowed:
            raise TypeError(f"{name} dtype must be one of {allowed}, got {value.dtype}")
    if contiguous and not value.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous; shape={tuple(value.shape)} "
            f"stride={value.stride()}"
        )


def _validate_finite(stage: str, **tensors: Tensor) -> None:
    if os.environ.get("MLITE_VALIDATE_FINITE") != "1":
        return
    for name, tensor in tensors.items():
        if not tensor.is_floating_point():
            continue
        finite = torch.isfinite(
            tensor.float()
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else tensor
        )
        if not bool(finite.all()):
            raise FloatingPointError(
                f"MLITE_NONFINITE stage={stage} tensor={name} "
                f"dtype={tensor.dtype} shape={tuple(tensor.shape)} "
                f"nonfinite={int((~finite).sum().item())}"
            )


def _same_device(named: dict[str, Tensor]) -> None:
    devices = {value.device for value in named.values()}
    if len(devices) != 1:
        detail = ", ".join(f"{name}={value.device}" for name, value in named.items())
        raise ValueError(f"all tensors must be on one device ({detail})")


class _Adapter(nn.Module):
    """Parameter-free base with a stable inference-only contract."""

    def __init__(self) -> None:
        super().__init__()


class MHCKernel(str, Enum):
    PRE = "pre"
    PRE_BROADCAST = "pre_broadcast"
    POST = "post"
    POST_PRE = "post_pre"
    HEAD = "head"


_MHC_ENTRIES = {
    MHCKernel.PRE: "mhc_pre_tilelang",
    MHCKernel.PRE_BROADCAST: "mhc_pre_broadcast_tilelang",
    MHCKernel.POST: "mhc_post_tilelang",
    MHCKernel.POST_PRE: "mhc_fused_post_pre_tilelang",
    MHCKernel.HEAD: "hc_head_fused_kernel_tilelang",
}


class MHCTileLangAdapter(_Adapter):
    """Adapter for one of vLLM's five public TileLang mHC wrappers."""

    def __init__(self, kernel: MHCKernel | str) -> None:
        super().__init__()
        try:
            self.kernel = MHCKernel(kernel)
        except ValueError as exc:
            raise ValueError(f"unsupported mHC kernel {kernel!r}") from exc

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        _inference_only(args, tuple(kwargs.values()))
        tensors = list(_tensors((args, tuple(kwargs.values()))))
        if not tensors:
            raise TypeError("mHC adapter requires tensor arguments")
        for i, value in enumerate(tensors):
            _tensor(value, f"tensor[{i}]")
        _same_device({str(i): value for i, value in enumerate(tensors)})

        if self.kernel is MHCKernel.PRE:
            _validate_mhc_pre(args, broadcast=False)
        elif self.kernel is MHCKernel.PRE_BROADCAST:
            _validate_mhc_pre(args, broadcast=True)
        elif self.kernel is MHCKernel.POST:
            _validate_mhc_post(args)
        elif self.kernel is MHCKernel.POST_PRE:
            _validate_mhc_post_pre(args)
        elif self.kernel is MHCKernel.HEAD:
            _validate_mhc_head(args)

        fn = _symbol("vllm.model_executor.kernels.mhc.tilelang", _MHC_ENTRIES[self.kernel])
        return _output_boundary(fn(*args, **kwargs), args, tuple(kwargs.values()))


def _validate_mhc_pre(args: tuple[Any, ...], *, broadcast: bool) -> None:
    if len(args) < 4:
        raise TypeError("mHC pre requires residual, fn, hc_scale, and hc_base")
    residual, fn, scale, base = args[:4]
    _tensor(residual, "residual", ndim=2 if broadcast else None, dtype=torch.bfloat16)
    if not broadcast and residual.ndim < 3:
        raise ValueError("mHC pre residual must have shape (..., hc_mult, hidden_size)")
    _tensor(fn, "fn", ndim=2, dtype=torch.float32)
    _tensor(scale, "hc_scale", ndim=1, dtype=torch.float32)
    _tensor(base, "hc_base", ndim=1, dtype=torch.float32)
    hidden = residual.shape[-1]
    hc_mult = fn.shape[1] // hidden if broadcast else residual.shape[-2]
    expected = 2 * hc_mult + hc_mult * hc_mult
    if fn.shape != (expected, hc_mult * hidden):
        raise ValueError(f"fn must have shape {(expected, hc_mult * hidden)}")
    if scale.shape != (3,) or base.shape != (expected,):
        raise ValueError("invalid hc_scale or hc_base shape")


def _validate_mhc_post(args: tuple[Any, ...]) -> None:
    if len(args) < 4:
        raise TypeError("mHC post requires x, residual, post_mix, and comb_mix")
    x, residual, post, comb = args[:4]
    _tensor(x, "x", dtype=torch.bfloat16)
    _tensor(residual, "residual", dtype=torch.bfloat16)
    _tensor(post, "post_mix", dtype=torch.float32)
    _tensor(comb, "comb_mix", dtype=torch.float32)
    outer, mult, hidden = residual.shape[:-2], residual.shape[-2], residual.shape[-1]
    if x.shape != (*outer, hidden):
        raise ValueError("x and residual shapes are incompatible")
    if post.shape not in ((*outer, mult), (*outer, mult, 1)):
        raise ValueError("post_mix has an invalid shape")
    if comb.shape != (*outer, mult, mult):
        raise ValueError("comb_mix has an invalid shape")


def _validate_mhc_post_pre(args: tuple[Any, ...]) -> None:
    if len(args) < 7:
        raise TypeError("mHC post_pre requires seven leading tensor arguments")
    _validate_mhc_post(args[:4])
    residual = args[1]
    fn, scale, base = args[4:7]
    _validate_mhc_pre((residual, fn, scale, base), broadcast=False)


def _validate_mhc_head(args: tuple[Any, ...]) -> None:
    if len(args) < 4:
        raise TypeError("mHC head requires hidden_states, fn, hc_scale, and hc_base")
    hs, fn, scale, base = args[:4]
    _tensor(hs, "hidden_states", ndim=3, dtype=torch.bfloat16)
    _tensor(fn, "fn", ndim=2, dtype=torch.float32)
    _tensor(scale, "hc_scale", ndim=1, dtype=torch.float32)
    _tensor(base, "hc_base", ndim=1, dtype=torch.float32)
    mult, hidden = hs.shape[-2:]
    if fn.shape != (mult, mult * hidden) or scale.shape != (1,) or base.shape != (mult,):
        raise ValueError("invalid mHC head parameter shape")


class FusedQKVRMSNormAdapter(_Adapter):
    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        q_weight: Tensor,
        kv_weight: Tensor,
        eps: float,
    ) -> tuple[Tensor, Tensor]:
        _inference_only(q, kv, q_weight, kv_weight)
        for name, value in (("q", q), ("kv", kv)):
            _tensor(value, name, ndim=2, contiguous=False)
            if value.stride(-1) != 1:
                raise ValueError(f"{name} innermost dimension must be contiguous")
        for name, value in (("q_weight", q_weight), ("kv_weight", kv_weight)):
            _tensor(value, name, ndim=1)
        _same_device({"q": q, "kv": kv, "q_weight": q_weight, "kv_weight": kv_weight})
        if q.shape[0] != kv.shape[0]:
            raise ValueError("q and kv token dimensions must match")
        if q_weight.shape != (q.shape[1],) or kv_weight.shape != (kv.shape[1],):
            raise ValueError("RMSNorm weight dimensions must match q and kv")
        if q.dtype != kv.dtype or q_weight.dtype != q.dtype or kv_weight.dtype != kv.dtype:
            raise TypeError("q, kv, and both weights must have the same dtype")
        if eps <= 0:
            raise ValueError("eps must be positive")
        fn = _symbol("vllm.models.common.ops", "fused_q_kv_rmsnorm")
        return _output_boundary(
            fn(q, kv, q_weight, kv_weight, eps), q, kv, q_weight, kv_weight
        )


class CompressorKernelAdapter(_Adapter):
    """Explicit boundary for a caller-owned DS4 compressor kernel.

    The model owns only BF16 master parameters.  FP32 state/cache metadata and
    any deployment packing stay ephemeral and are supplied by ``operation``.
    """

    def forward(
        self,
        operation: Callable[..., Any],
        kv_score: Tensor,
        positions: Tensor,
        ape: Tensor,
        norm_weight: Tensor,
        *,
        compress_ratio: int,
        head_dim: int,
        metadata: Any,
    ) -> Any:
        _inference_only(kv_score, positions, ape, norm_weight)
        if not callable(operation):
            raise TypeError("compressor operation must be callable")
        _tensor(kv_score, "kv_score", ndim=2, dtype=torch.float32)
        _tensor(positions, "positions", ndim=1, dtype=(torch.int32, torch.int64))
        _tensor(ape, "ape", ndim=2, dtype=torch.float32)
        _tensor(norm_weight, "norm_weight", ndim=1, dtype=torch.bfloat16)
        if compress_ratio <= 1 or head_dim <= 0:
            raise ValueError("compressor ratio must exceed one and head_dim must be positive")
        coff = 2 if compress_ratio == 4 else 1
        expected_width = coff * head_dim
        if kv_score.shape != (positions.numel(), 2 * expected_width):
            raise ValueError("kv_score shape does not match the compressor contract")
        if ape.shape != (compress_ratio, expected_width):
            raise ValueError("ape shape does not match the compressor contract")
        if norm_weight.shape != (head_dim,):
            raise ValueError("compressor norm shape does not match head_dim")
        _same_device(
            {
                "kv_score": kv_score,
                "positions": positions,
                "ape": ape,
                "norm_weight": norm_weight,
            }
        )
        result = operation(
            kv_score=kv_score,
            positions=positions,
            ape=ape,
            norm_weight=norm_weight,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            metadata=metadata,
        )
        return _output_boundary(result, kv_score, ape, norm_weight)


class IndexerKernelAdapter(_Adapter):
    """Explicit boundary for DS4 indexer query/selection primitives."""

    def forward(
        self,
        operation: Callable[..., Any],
        qr: Tensor,
        index_q: Tensor,
        index_weights: Tensor,
        positions: Tensor,
        *,
        compress_ratio: int,
        topk: int,
        metadata: Any,
    ) -> Any:
        _inference_only(qr, index_q, index_weights, positions)
        if not callable(operation):
            raise TypeError("indexer operation must be callable")
        # ``qr`` is the first view returned by splitting fused_wqa_wkv and is
        # therefore intentionally allowed to be non-contiguous.
        _tensor(qr, "qr", ndim=2, dtype=torch.bfloat16, contiguous=False)
        _tensor(index_q, "index_q", ndim=3, dtype=torch.bfloat16)
        _tensor(index_weights, "index_weights", ndim=2, dtype=torch.bfloat16)
        _tensor(positions, "positions", ndim=1, dtype=(torch.int32, torch.int64))
        if compress_ratio != 4:
            raise ValueError("DeepSeek-V4 indexer is defined only for compress_ratio=4")
        if topk <= 0:
            raise ValueError("indexer topk must be positive")
        if qr.shape[0] != positions.numel() or index_q.shape[0] != qr.shape[0]:
            raise ValueError("indexer token dimensions do not match")
        if index_weights.shape != index_q.shape[:2]:
            raise ValueError("index_weights must have one value per token and index head")
        _same_device(
            {
                "qr": qr,
                "index_q": index_q,
                "index_weights": index_weights,
                "positions": positions,
            }
        )
        result = operation(
            qr=qr,
            index_q=index_q,
            index_weights=index_weights,
            positions=positions,
            compress_ratio=compress_ratio,
            topk=topk,
            metadata=metadata,
        )
        return _output_boundary(result, qr, index_q, index_weights)


class KVCacheLayout(str, Enum):
    FP8_DS_MLA = "fp8_ds_mla"
    PLAIN_BF16 = "plain_bf16"
    PLAIN_FP8_E4M3 = "plain_fp8_e4m3"


class DS4KVInsertAdapter(_Adapter):
    """Exact adapter for the three `_C` DS4 QNorm/RoPE/KV insert entries."""

    def __init__(self, layout: KVCacheLayout | str) -> None:
        super().__init__()
        self.layout = KVCacheLayout(layout)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        cache: Tensor,
        slot_mapping: Tensor,
        positions: Tensor,
        cos_sin_cache: Tensor,
        *,
        eps: float,
        block_size: int,
        padded_heads: int | None = None,
        q_out: Tensor | None = None,
        kv_scale: Tensor | None = None,
        q_scale_inv: Tensor | None = None,
    ) -> Tensor:
        _inference_only(
            q, kv, cache, slot_mapping, positions, cos_sin_cache, q_out, kv_scale, q_scale_inv
        )
        _tensor(q, "kv_insert.q", ndim=3)
        _tensor(kv, "kv", ndim=2)
        _tensor(slot_mapping, "slot_mapping", ndim=1, dtype=torch.int64)
        _tensor(positions, "positions", ndim=1, dtype=torch.int64)
        _tensor(
            cos_sin_cache,
            "cos_sin_cache",
            ndim=2,
            dtype=torch.float32,
        )
        if q.shape[0] != kv.shape[0] or q.shape[0] != positions.numel():
            raise ValueError("q, kv, and positions token dimensions must match")
        if slot_mapping.numel() != q.shape[0]:
            raise ValueError("slot_mapping must contain one entry per token")
        if kv.shape[1] != q.shape[2]:
            raise ValueError("kv width must match q head dimension")
        if block_size <= 0 or eps <= 0:
            raise ValueError("block_size and eps must be positive")
        named = {
            "q": q,
            "kv": kv,
            "cache": cache,
            "slot_mapping": slot_mapping,
            "positions": positions,
            "cos_sin_cache": cos_sin_cache,
        }
        named.update(
            {name: value for name, value in {
                "q_out": q_out, "kv_scale": kv_scale, "q_scale_inv": q_scale_inv
            }.items() if value is not None}
        )
        _same_device(named)

        if self.layout is KVCacheLayout.FP8_DS_MLA:
            _tensor(cache, "cache", ndim=2, dtype=torch.uint8)
            if padded_heads is None or padded_heads < q.shape[1]:
                raise ValueError("fp8_ds_mla requires padded_heads >= q heads")
            args = (
                q, kv, cache, slot_mapping, positions, cos_sin_cache,
                padded_heads, eps, block_size,
            )
            if q_out is not None:
                if q_out.shape != (q.shape[0], padded_heads, q.shape[2]):
                    raise ValueError("q_out has an invalid padded shape")
                _op("_C", "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out")(
                    q, kv, q_out, cache, slot_mapping, positions, cos_sin_cache,
                    padded_heads, eps, block_size,
                )
                return _output_boundary(
                    q_out, q, kv, cache, slot_mapping, positions, cos_sin_cache
                )
            return _output_boundary(
                _op("_C", "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")(*args),
                q,
                kv,
                cache,
                slot_mapping,
                positions,
                cos_sin_cache,
            )

        _tensor(cache, "cache", ndim=3)
        if cache.shape[-1] != kv.shape[-1] or cache.shape[1] != block_size:
            raise ValueError("plain cache must have shape (blocks, block_size, head_dim)")
        if self.layout is KVCacheLayout.PLAIN_BF16:
            if cache.dtype != torch.bfloat16 or q.dtype != torch.bfloat16:
                raise TypeError("plain_bf16 cache and q must be bfloat16")
            _op("_C", "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert")(
                q, kv, cache, slot_mapping, positions, cos_sin_cache, eps, block_size
            )
            return _output_boundary(
                q, q, kv, cache, slot_mapping, positions, cos_sin_cache
            )

        if cache.dtype != torch.float8_e4m3fn:
            raise TypeError("plain_fp8_e4m3 cache must be float8_e4m3fn")
        if q_out is None or q_out.dtype != torch.float8_e4m3fn or q_out.shape != q.shape:
            raise ValueError("plain_fp8_e4m3 requires a shape-matched float8 q_out")
        if kv_scale is None or q_scale_inv is None:
            raise ValueError("plain_fp8_e4m3 requires kv_scale and q_scale_inv")
        _op("_C", "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert")(
            q, kv, q_out, cache, slot_mapping, positions, cos_sin_cache,
            kv_scale, q_scale_inv, eps, block_size,
        )
        return _output_boundary(
            q_out,
            q,
            kv,
            cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            kv_scale,
            q_scale_inv,
        )


class FlashMLAAdapter(_Adapter):
    """Lowest explicit FlashMLA entries; no vLLM forward context is consulted."""

    def sparse(
        self,
        q: Tensor,
        kv: Tensor,
        indices: Tensor,
        *,
        sm_scale: float,
        attn_sink: Tensor | None = None,
        topk_length: Tensor | None = None,
        out: Tensor | None = None,
    ) -> Any:
        _inference_only(q, kv, indices, attn_sink, topk_length, out)
        _tensor(q, "flash_sparse.q", ndim=3)
        _tensor(kv, "kv", ndim=3)
        _tensor(indices, "indices", ndim=3, dtype=(torch.int32, torch.int64))
        if q.shape[0] != indices.shape[0] or kv.shape[-1] != q.shape[-1]:
            raise ValueError("FlashMLA sparse q/kv/indices shapes are incompatible")
        named = {"q": q, "kv": kv, "indices": indices}
        if attn_sink is not None:
            _tensor(attn_sink, "attn_sink", ndim=1, dtype=torch.float32)
            named["attn_sink"] = attn_sink
        if topk_length is not None:
            _tensor(topk_length, "topk_length", ndim=1, dtype=(torch.int32, torch.int64))
            named["topk_length"] = topk_length
        if out is not None:
            named["out"] = out
        _same_device(named)
        fn = _symbol("vllm.v1.attention.ops.flashmla", "flash_mla_sparse_fwd")
        result = fn(
            q=q, kv=kv, indices=indices, sm_scale=sm_scale,
            attn_sink=attn_sink, topk_length=topk_length, out=out,
        )
        return _output_boundary(result, q, kv, attn_sink, out)

    def paged(
        self,
        q: Tensor,
        k_cache: Tensor,
        *,
        tile_scheduler_metadata: Any,
        indices: Tensor,
        topk_length: Tensor,
        softmax_scale: float,
        attn_sink: Tensor,
        out: Tensor,
        extra_k_cache: Tensor | None = None,
        extra_indices_in_kvcache: Tensor | None = None,
        extra_topk_length: Tensor | None = None,
        head_dim_v: int = 512,
        is_fp8_kvcache: bool = True,
    ) -> Any:
        _inference_only(
            q, k_cache, indices, topk_length, attn_sink, out,
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
        )
        if tile_scheduler_metadata is None:
            raise NotImplementedError(
                "FlashMLA paged decode requires explicit tile_scheduler_metadata "
                "built by the matching vLLM metadata builder."
            )
        _tensor(q, "flash_paged.q", ndim=4)
        _tensor(k_cache, "k_cache", ndim=4)
        _tensor(indices, "indices", dtype=(torch.int32, torch.int64))
        _tensor(topk_length, "topk_length", dtype=(torch.int32, torch.int64))
        _same_device({"q": q, "k_cache": k_cache, "indices": indices,
                      "topk_length": topk_length, "attn_sink": attn_sink, "out": out})
        fn = _symbol("vllm.v1.attention.ops.flashmla", "flash_mla_with_kvcache")
        result = fn(
            q=q, k_cache=k_cache, block_table=None, head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata, cache_seqlens=None,
            is_fp8_kvcache=is_fp8_kvcache, indices=indices,
            topk_length=topk_length, softmax_scale=softmax_scale,
            attn_sink=attn_sink, extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length, out=out,
        )
        return _output_boundary(result, q, k_cache, attn_sink, out)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("use FlashMLAAdapter.sparse or .paged with explicit metadata")


class GateLinearAdapter(_Adapter):
    """Validate and invoke an already-constructed vLLM GateLinear explicitly."""

    def forward(self, gate: Callable[[Tensor], Any], hidden_states: Tensor) -> Tensor:
        _inference_only(hidden_states)
        _tensor(hidden_states, "hidden_states", ndim=2)
        if not callable(gate):
            raise TypeError("gate must be a callable vLLM GateLinear instance")
        result = gate(hidden_states)
        output = result[0] if isinstance(result, tuple) else result
        _tensor(output, "gate output", ndim=2)
        if output.shape[0] != hidden_states.shape[0]:
            raise ValueError("gate output token dimension does not match input")
        return _output_boundary(output, hidden_states)


class DS4TopKAdapter(_Adapter):
    def forward(
        self,
        gating_output: Tensor,
        correction_bias: Tensor,
        *,
        indices_dtype: torch.dtype,
        routed_scaling_factor: float,
    ) -> tuple[Tensor, Tensor]:
        _inference_only(gating_output, correction_bias)
        _tensor(gating_output, "gating_output", ndim=2, dtype=torch.float32)
        _tensor(correction_bias, "correction_bias", ndim=1, dtype=torch.float32)
        _same_device({"gating_output": gating_output, "correction_bias": correction_bias})
        if gating_output.shape[1] not in (256, 384) or correction_bias.shape != (
            gating_output.shape[1],
        ):
            raise ValueError("DSv4 top-k requires 256/384 experts and matching bias")
        if indices_dtype not in (torch.int32, torch.uint32, torch.int64):
            raise TypeError("unsupported top-k indices dtype")
        fn = _symbol("vllm.model_executor.layers.fused_moe.router.dsv4_topk", "dsv4_topk")
        return _output_boundary(
            fn(gating_output, correction_bias, indices_dtype, routed_scaling_factor),
            gating_output,
            correction_bias,
        )


class HashRouteAdapter(_Adapter):
    """Exact `topk_hash_softplus_sqrt` custom-op route."""

    def forward(
        self,
        gating_output: Tensor,
        input_tokens: Tensor,
        hash_indices_table: Tensor,
        *,
        topk: int,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        correction_bias: Tensor | None = None,
        indices_dtype: torch.dtype = torch.int32,
    ) -> tuple[Tensor, Tensor]:
        _inference_only(gating_output, input_tokens, hash_indices_table, correction_bias)
        _tensor(gating_output, "gating_output", ndim=2, dtype=torch.float32)
        _tensor(input_tokens, "input_tokens", ndim=1, dtype=(torch.int32, torch.int64))
        _tensor(hash_indices_table, "hash_indices_table", ndim=2,
                dtype=(torch.int32, torch.int64))
        if input_tokens.shape[0] != gating_output.shape[0]:
            raise ValueError("one input token id is required per router row")
        if hash_indices_table.shape[1] != topk:
            raise ValueError("hash table width must equal topk")
        if input_tokens.dtype != hash_indices_table.dtype:
            raise TypeError("input_tokens and hash_indices_table dtypes must match")
        if indices_dtype not in (torch.int32, torch.int64):
            raise TypeError("hash output indices must be int32 or int64")
        if os.getenv("MLITE_VALIDATE_INDICES") == "1" and input_tokens.numel():
            token_min, token_max = torch.aminmax(input_tokens)
            expert_min, expert_max = torch.aminmax(hash_indices_table)
            if int(token_min.item()) < 0 or int(token_max.item()) >= hash_indices_table.shape[0]:
                raise ValueError(
                    "hash router token IDs are outside tid2eid: "
                    f"min={int(token_min.item())}, max={int(token_max.item())}, "
                    f"rows={hash_indices_table.shape[0]}"
                )
            if int(expert_min.item()) < 0 or int(expert_max.item()) >= gating_output.shape[-1]:
                raise ValueError(
                    "tid2eid expert IDs are outside router logits: "
                    f"min={int(expert_min.item())}, max={int(expert_max.item())}, "
                    f"experts={gating_output.shape[-1]}"
                )
        _same_device({"gating_output": gating_output, "input_tokens": input_tokens,
                      "hash_indices_table": hash_indices_table})
        weights = torch.empty(
            gating_output.shape[0], topk, dtype=torch.float32, device=gating_output.device
        )
        ids = torch.empty(
            gating_output.shape[0], topk, dtype=indices_dtype, device=gating_output.device
        )
        token_expert = torch.empty_like(ids, dtype=torch.int32)
        _symbol("vllm._custom_ops", "topk_hash_softplus_sqrt")(
            weights, ids, token_expert, gating_output, renormalize,
            routed_scaling_factor, correction_bias, input_tokens,
            hash_indices_table, None,
        )
        return _output_boundary(
            (weights, ids),
            gating_output,
            correction_bias,
            input_tokens,
            hash_indices_table,
        )


class FusedExpertsAdapter(_Adapter):
    """Unquantized local-expert path through vLLM's actual `fused_experts` entry."""

    def forward(
        self,
        hidden_states: Tensor,
        w1: Tensor,
        w2: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        *,
        activation: Any,
        apply_router_weight_on_input: bool = False,
        global_num_experts: int = -1,
        expert_map: Tensor | None = None,
    ) -> Tensor:
        _inference_only(hidden_states, w1, w2, topk_weights, topk_ids, expert_map)
        _tensor(hidden_states, "hidden_states", ndim=2)
        _tensor(w1, "w1", ndim=3)
        _tensor(w2, "w2", ndim=3)
        _tensor(topk_weights, "topk_weights", ndim=2, dtype=torch.float32)
        _tensor(topk_ids, "topk_ids", ndim=2, dtype=(torch.int32, torch.int64))
        if topk_weights.shape != topk_ids.shape or topk_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("top-k tensors must have matching shape and token count")
        if w1.shape[0] != w2.shape[0] or w1.shape[2] != hidden_states.shape[1]:
            raise ValueError("expert weight shapes are incompatible with hidden_states")
        _same_device({"hidden_states": hidden_states, "w1": w1, "w2": w2,
                      "topk_weights": topk_weights, "topk_ids": topk_ids})
        fn = _symbol("vllm.model_executor.layers.fused_moe.fused_moe", "fused_experts")
        return _output_boundary(
            fn(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            ),
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            expert_map,
        )


@dataclass(frozen=True)
class GroupedFP8ExpertWeights:
    """Ephemeral vLLM/DeepGEMM weight tensors for all local experts."""

    w13: Tensor
    w2: Tensor
    w13_scale: Tensor
    w2_scale: Tensor


class GroupedDeepGemmExpertsAdapter(_Adapter):
    """BF16 masters into the official batched-DeepGEMM pipeline."""

    _EXPERTS_CLASS = "BatchedDeepGemmExperts"
    _PREPARE_CLASSES = {
        "DeepEPLLPrepareAndFinalize",
        "_NormalDeepEPAlignedPrepareAndFinalize",
    }

    def __init__(
        self,
        pack_grouped_weight: Callable[[Iterable[nn.Parameter]], Any] | None = None,
    ):
        super().__init__()
        if pack_grouped_weight is None:
            pack_grouped_weight = _symbol(
                "megatron.lite.primitive.quantization.deployment_block_fp8",
                "pack_grouped_block_fp8_weight",
            )
        self._pack_grouped_weight = pack_grouped_weight

    def pack(
        self,
        w13: Iterable[nn.Parameter],
        w2: Iterable[nn.Parameter],
    ) -> GroupedFP8ExpertWeights:
        w13 = tuple(w13)
        w2 = tuple(w2)
        if not w13 or len(w13) != len(w2):
            raise ValueError("grouped experts require equal non-empty w13/w2 masters")
        packed_w13 = self._pack_grouped_weight(w13)
        packed_w2 = self._pack_grouped_weight(w2)
        result = GroupedFP8ExpertWeights(
            w13=packed_w13.qweight,
            w2=packed_w2.qweight,
            w13_scale=packed_w13.scales,
            w2_scale=packed_w2.scales,
        )
        if result.w13.ndim != 3 or result.w2.ndim != 3:
            raise RuntimeError("grouped DeepGEMM weights must be rank-3")
        if result.w13.dtype != torch.float8_e4m3fn:
            raise RuntimeError("grouped DeepGEMM w13 must use E4M3")
        if result.w2.dtype != torch.float8_e4m3fn:
            raise RuntimeError("grouped DeepGEMM w2 must use E4M3")
        return result

    def forward(
        self,
        hidden_states: Tensor,
        w13: Iterable[nn.Parameter],
        w2: Iterable[nn.Parameter],
        topk_weights: Tensor,
        topk_ids: Tensor,
        *,
        build_kernel: Callable[[GroupedFP8ExpertWeights], Any],
        global_num_experts: int,
        expert_map: Tensor,
    ) -> Tensor:
        """Build a scale-bound official kernel and execute dispatch→GEMMs→combine."""

        w13 = tuple(w13)
        w2 = tuple(w2)
        _inference_only(hidden_states, topk_weights, topk_ids, expert_map)
        _tensor(hidden_states, "hidden_states", ndim=2, dtype=torch.bfloat16)
        _tensor(topk_weights, "topk_weights", ndim=2, dtype=torch.float32)
        _tensor(topk_ids, "topk_ids", ndim=2, dtype=torch.int64)
        _tensor(expert_map, "expert_map", ndim=1, dtype=torch.int32)
        if topk_weights.shape != topk_ids.shape:
            raise ValueError("top-k weights and ids must have identical shapes")
        if topk_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("top-k token count must match hidden_states")
        if expert_map.shape[0] != global_num_experts:
            raise ValueError("expert_map must contain every global expert")

        packed = self.pack(w13, w2)
        _validate_finite(
            "grouped_moe.input",
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            w13=packed.w13,
            w2=packed.w2,
            w13_scale=packed.w13_scale,
            w2_scale=packed.w2_scale,
        )
        kernel = build_kernel(packed)
        fused_experts = getattr(kernel, "fused_experts", None)
        prepare_finalize = getattr(kernel, "prepare_finalize", None)
        if (
            fused_experts is None
            or fused_experts.__class__.__name__ != self._EXPERTS_CLASS
        ):
            raise RuntimeError(
                "grouped MoE builder must return FusedMoEKernel with "
                f"{self._EXPERTS_CLASS}"
            )
        if (
            prepare_finalize is None
            or prepare_finalize.__class__.__name__ not in self._PREPARE_CLASSES
        ):
            raise RuntimeError(
                "grouped MoE builder must return FusedMoEKernel with "
                "a supported DeepEP prepare/finalize"
            )
        for name, expected in (
            ("w1_scale", packed.w13_scale),
            ("w2_scale", packed.w2_scale),
        ):
            actual = getattr(fused_experts, name, None)
            if (
                not isinstance(actual, Tensor)
                or actual.device != expected.device
                or actual.data_ptr() != expected.data_ptr()
            ):
                raise RuntimeError(
                    f"grouped MoE builder did not bind dynamic {name} "
                    "from the BF16 master packing"
                )
        output = kernel.apply(
            hidden_states=hidden_states,
            w1=packed.w13,
            w2=packed.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=_symbol(
                "vllm.model_executor.layers.fused_moe.activation", "MoEActivation"
            ).SILU,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=False,
        )
        return _output_boundary(
            output, hidden_states, topk_weights, topk_ids, expert_map, w13, w2
        )


class GroupedMoEKernelBuilderAdapter:
    """Construct official BatchedDeepGemm with LL or aligned normal DeepEP."""

    def __init__(
        self,
        deepep_buffer: Any,
        *,
        device: torch.device | str,
        num_experts: int,
        num_local_experts: int,
        experts_per_token: int,
        hidden_dim: int,
        intermediate_size: int,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = True,
    ) -> None:
        positive = {
            "num_experts": num_experts,
            "num_local_experts": num_local_experts,
            "experts_per_token": experts_per_token,
            "hidden_dim": hidden_dim,
            "intermediate_size": intermediate_size,
            "max_tokens_per_rank": max_tokens_per_rank,
            "num_dispatchers": num_dispatchers,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"grouped MoE sizes must be positive: {invalid}")
        self.deepep_buffer = deepep_buffer
        self.device = torch.device(device)
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_per_token = experts_per_token
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers = num_dispatchers
        self.use_fp8_dispatch = use_fp8_dispatch

    def __call__(self, packed: GroupedFP8ExpertWeights, *, dispatcher=None) -> Any:
        config_module = (
            "vllm.model_executor.layers.fused_moe.config"
        )
        activation = _symbol(
            "vllm.model_executor.layers.fused_moe.activation",
            "MoEActivation",
        )
        quant = _symbol(config_module, "fp8_w8a8_moe_quant_config")(
            w1_scale=packed.w13_scale,
            w2_scale=packed.w2_scale,
            block_shape=[128, 128],
        )
        parallel = _symbol(config_module, "FusedMoEParallelConfig").make_no_parallel()
        routing = _symbol(config_module, "RoutingMethodType")
        moe_config = _symbol(config_module, "FusedMoEConfig")(
            num_experts=self.num_experts,
            experts_per_token=self.experts_per_token,
            hidden_dim=self.hidden_dim,
            intermediate_size=self.intermediate_size,
            num_local_experts=self.num_local_experts,
            num_logical_experts=self.num_experts,
            activation=activation.SILU,
            device=self.device,
            routing_method=routing.TopK,
            moe_parallel_config=parallel,
            in_dtype=torch.bfloat16,
            max_num_tokens=self.max_tokens_per_rank,
        )
        if dispatcher is None:
            if self.deepep_buffer is None:
                raise ValueError(
                    "grouped MoE builder requires either an LL buffer or "
                    "an aligned normal-DeepEP dispatcher"
                )
            prepare = _symbol(
                "vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll",
                "DeepEPLLPrepareAndFinalize",
            )(
                self.deepep_buffer,
                max_tokens_per_rank=self.max_tokens_per_rank,
                num_dispatchers=self.num_dispatchers,
                use_fp8_dispatch=self.use_fp8_dispatch,
            )
        else:
            from megatron.lite.primitive.alignment.vllm_batched_prepare import (
                NormalDeepEPAlignedPrepareAndFinalize,
            )

            prepare = NormalDeepEPAlignedPrepareAndFinalize.build(
                dispatcher,
                max_tokens_per_rank=self.max_tokens_per_rank,
                num_dispatchers=self.num_dispatchers,
            )
        experts = _symbol(
            "vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe",
            "BatchedDeepGemmExperts",
        )(
            moe_config=moe_config,
            quant_config=quant,
            max_num_tokens=self.max_tokens_per_rank,
            num_dispatchers=self.num_dispatchers,
        )

        kernel_cls = _symbol(
            "vllm.model_executor.layers.fused_moe.modular_kernel",
            "FusedMoEKernel",
        )

        kernel = kernel_cls(prepare, experts)

        def runtime_allocate_buffers(
            impl,
            out_dtype,
            device,
            M_chunk,
            M_full,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        ):
            """Allocate modular-MoE temporaries from the mLite runtime."""
            workspace_dtype = impl.fused_experts.workspace_dtype(out_dtype)
            workspace13_shape, workspace2_shape, _ = (
                impl.fused_experts.workspace_shapes(
                    M_chunk,
                    N,
                    K,
                    top_k,
                    global_num_experts,
                    local_num_experts,
                    expert_tokens_meta,
                    activation,
                )
            )
            _, _, output_shape = impl.fused_experts.workspace_shapes(
                M_full,
                N,
                K,
                top_k,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
            common = torch.empty(
                max(math.prod(workspace13_shape), math.prod(output_shape)),
                dtype=workspace_dtype,
                device=device,
            )
            workspace13 = common[: math.prod(workspace13_shape)].view(
                workspace13_shape
            )
            output = common[: math.prod(output_shape)].view(output_shape)
            workspace2 = torch.empty(
                workspace2_shape, dtype=workspace_dtype, device=device
            )
            return workspace13, workspace2, output

        impl = getattr(kernel, "impl", None)
        if impl is None or not hasattr(impl, "_allocate_buffers"):
            raise NotImplementedError(
                "vLLM modular FusedMoEKernel does not expose _allocate_buffers"
            )
        impl._allocate_buffers = types.MethodType(runtime_allocate_buffers, impl)
        return kernel


class SharedExpertsAdapter(_Adapter):
    """Explicit shared-expert callable boundary, avoiding vLLM's layer registry."""

    def forward(self, shared_experts: Callable[[Tensor], Any], hidden_states: Tensor) -> Tensor:
        _inference_only(hidden_states)
        _tensor(hidden_states, "hidden_states", ndim=2)
        if not callable(shared_experts):
            raise TypeError("shared_experts must be an explicit callable")
        output = shared_experts(hidden_states)
        output = output[0] if isinstance(output, tuple) else output
        _tensor(output, "shared expert output")
        if output.shape != hidden_states.shape:
            raise ValueError("shared expert output shape must match hidden_states")
        return _output_boundary(output, hidden_states)


class OProjectionAdapter(_Adapter):
    """Official DS4 inverse-RoPE/grouped-FP8 output projection.

    BF16 parameters remain the only model state.  Both FP8 weights are packed
    ephemerally with the same vLLM deployment utilities used by the reference.
    """

    def forward(
        self,
        o: Tensor,
        positions: Tensor,
        cos_sin_cache: Tensor,
        wo_a: Tensor,
        wo_b: Tensor,
        *,
        n_groups: int,
        heads_per_group: int,
        nope_dim: int,
        rope_dim: int,
        o_lora_rank: int,
    ) -> Tensor:
        _inference_only(o, positions, cos_sin_cache, wo_a, wo_b)
        _tensor(o, "o", ndim=3, dtype=torch.bfloat16)
        _tensor(positions, "positions", ndim=1)
        _tensor(cos_sin_cache, "cos_sin_cache", dtype=torch.float32)
        _tensor(wo_a, "wo_a", ndim=2, dtype=torch.bfloat16)
        _tensor(wo_b, "wo_b", ndim=2, dtype=torch.bfloat16)
        if positions.shape[0] != o.shape[0]:
            raise ValueError("positions and o must have the same token count")
        if n_groups <= 0 or heads_per_group <= 0 or o_lora_rank <= 0:
            raise ValueError("o_proj group and rank sizes must be positive")
        if o.shape[1] != n_groups * heads_per_group:
            raise ValueError("o head count does not match grouped projection metadata")
        if o.shape[2] != nope_dim + rope_dim:
            raise ValueError("o head dimension does not match nope_dim + rope_dim")
        if wo_a.shape != (
            n_groups * o_lora_rank,
            heads_per_group * (nope_dim + rope_dim),
        ):
            raise ValueError("wo_a shape does not match grouped DS4 o_proj contract")
        if wo_b.shape[1] != n_groups * o_lora_rank:
            raise ValueError("wo_b input size does not match grouped DS4 o_proj rank")
        _same_device(
            {
                "o": o,
                "positions": positions,
                "cos_sin_cache": cos_sin_cache,
                "wo_a": wo_a,
                "wo_b": wo_b,
            }
        )

        post_process = _symbol(
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "deepgemm_post_process_fp8_weight_block",
        )
        official = _symbol(
            "vllm.models.deepseek_v4.nvidia.ops.o_proj",
            "deep_gemm_fp8_o_proj",
        )
        recipe_fn = _symbol(
            "vllm.models.deepseek_v4.nvidia.ops.o_proj",
            "compute_fp8_einsum_recipe",
        )

        with torch.inference_mode():
            canonical_wa = quantize_block_fp8_weight(wo_a)
            wa_q, wa_s = canonical_wa.qweight, canonical_wa.scales
            wa_q, wa_s = post_process(
                wq=wa_q,
                ws=wa_s,
                quant_block_shape=(128, 128),
                use_e8m0=True,
                is_bmm=True,
                bmm_batch_size=n_groups,
            )

            # The official helper only requires these two attributes from wo_a.
            packed_wa = type("_PackedGroupedWeight", (), {})()
            packed_wa.weight = wa_q
            packed_wa.weight_scale = wa_s

            canonical_wb = quantize_block_fp8_weight(wo_b)
            wb_q, wb_s = canonical_wb.qweight, canonical_wb.scales
            wb_q, wb_s = post_process(
                wq=wb_q,
                ws=wb_s,
                quant_block_shape=(128, 128),
                use_e8m0=True,
            )

            def packed_wb(value: Tensor) -> Tensor:
                activation_quant = _symbol(
                    "vllm.model_executor.layers.quantization.utils.fp8_utils",
                    "per_token_group_quant_fp8",
                )
                tma_aligned_scales = bool(
                    _symbol(
                        "vllm.envs",
                        "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES",
                    )
                )
                aq, a_s = activation_quant(
                    value,
                    128,
                    use_ue8m0=True,
                    column_major_scales=True,
                    tma_aligned_scales=tma_aligned_scales,
                )
                output = torch.empty(
                    value.shape[0],
                    wb_q.shape[0],
                    dtype=torch.bfloat16,
                    device=value.device,
                )
                gemm = _symbol("vllm.utils.deep_gemm", "fp8_gemm_nt")
                gemm(
                    (aq, a_s),
                    (wb_q, wb_s),
                    output,
                    is_deep_gemm_e8m0_used=True,
                )
                return output

            recipe, aligned = recipe_fn()
            output = official(
                o,
                positions,
                cos_sin_cache,
                packed_wa,
                packed_wb,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                o_lora_rank=o_lora_rank,
                einsum_recipe=recipe,
                tma_aligned_scales=aligned,
            )
        return _output_boundary(output, o, wo_a, wo_b)


class DeepEPMode(str, Enum):
    HIGH_THROUGHPUT = "high_throughput"
    LOW_LATENCY = "low_latency"


class DeepEPAdapter(_Adapter):
    """Explicit DeepEP handle/group adapter; lifecycle remains caller-owned."""

    def __init__(self, handle: Any, group: Any, mode: DeepEPMode | str) -> None:
        super().__init__()
        if handle is None or group is None:
            raise ValueError("DeepEP requires explicit handle and process group")
        self.handle = handle
        self.group = group
        self.mode = DeepEPMode(mode)

    def dispatch(self, x: Tensor, topk_ids: Tensor, **kwargs: Any) -> Any:
        _inference_only(x, topk_ids, tuple(kwargs.values()))
        _tensor(x, "x", ndim=2)
        _tensor(topk_ids, "topk_ids", ndim=2, dtype=torch.int64)
        if x.shape[0] != topk_ids.shape[0]:
            raise ValueError("x and topk_ids token dimensions must match")
        if self.mode is DeepEPMode.LOW_LATENCY:
            required = {"max_tokens_per_rank", "num_experts"}
            missing = required - kwargs.keys()
            if missing:
                raise ValueError(f"low-latency dispatch missing {sorted(missing)}")
            result = self.handle.low_latency_dispatch(
                x, topk_ids, kwargs.pop("max_tokens_per_rank"),
                kwargs.pop("num_experts"), **kwargs,
            )
            return _output_boundary(result, x)
        return _output_boundary(
            self.handle.dispatch(x=x, topk_idx=topk_ids, **kwargs), x
        )

    def combine(
        self,
        x: Tensor,
        handle: Any,
        *,
        topk_ids: Tensor | None = None,
        topk_weights: Tensor | None = None,
        out: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        _inference_only(x, topk_ids, topk_weights, out, tuple(kwargs.values()))
        _tensor(x, "x")
        if handle is None:
            raise ValueError("combine requires the handle returned by dispatch")
        if self.mode is DeepEPMode.LOW_LATENCY:
            if topk_ids is None or topk_weights is None:
                raise ValueError("low-latency combine requires topk_ids and topk_weights")
            result = self.handle.low_latency_combine(
                x, topk_ids, topk_weights, handle, out=out, **kwargs
            )
            return _output_boundary(result, x, topk_weights)
        return _output_boundary(
            self.handle.combine(
                x=x, handle=handle, topk_weights=topk_weights, **kwargs
            ),
            x,
            topk_weights,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("use DeepEPAdapter.dispatch and .combine explicitly")


__all__ = [
    "CompressorKernelAdapter",
    "DS4KVInsertAdapter",
    "DS4TopKAdapter",
    "DeepEPAdapter",
    "DeepEPMode",
    "FlashMLAAdapter",
    "FusedExpertsAdapter",
    "GroupedDeepGemmExpertsAdapter",
    "GroupedFP8ExpertWeights",
    "GroupedMoEKernelBuilderAdapter",
    "FusedQKVRMSNormAdapter",
    "GateLinearAdapter",
    "HashRouteAdapter",
    "IndexerKernelAdapter",
    "KVCacheLayout",
    "MHCKernel",
    "MHCTileLangAdapter",
    "OProjectionAdapter",
    "SharedExpertsAdapter",
]
