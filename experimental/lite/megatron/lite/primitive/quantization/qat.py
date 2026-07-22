# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Quantization-aware training (QAT) primitive for Megatron Lite.

Phase 1 scope (see ``docs/qat_cross_framework_design.md``): weight-only integer
QAT for ``int8`` (W8A16) and non-NVFP4 ``int4`` (W4A16-int). NVFP4 / MXFP4 / FP8
and any activation quantization are deferred to later, separately gated leaves.

The primitive enforces the three-state separation mandated by the design:

1. **Master weight** — the trainable parameter stays the original (BF16) weight.
   Fake quantization is applied through ``torch.nn.utils.parametrize`` so the
   raw parameter survives untouched as ``...parametrizations.weight.original``
   and remains what the optimizer updates. QAT never registers ``W_hat`` as a
   parameter, and never forces an FP32 master.
2. **fake-quant / STE** — the forward path uses ``W_hat = dequant(quant(W))``;
   the backward path is a straight-through estimator (optionally clipped to the
   representable range). Quantization ``scale``/``amax`` are *statistics*, held
   in non-trainable buffers, never weight copies.
3. **Deployment representation** — packed integer tensors + scales are produced
   only on demand (export / rollout refit) by :func:`quantize_weight` /
   :func:`pack_int4`; the training step, optimizer and checkpoint never consume
   packed weights.

This module is model-agnostic on purpose (``primitive.design`` replaceability):
it knows nothing about Qwen/GLM/Kimi names and never calls into rollout. Models
opt in from their ``protocol.build_model`` via :func:`apply_qat_to_chunks`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

# Supported phase-1 formats -> number of integer bits. Free-form strings are
# rejected; every enum must map to an exact quant/dequant contract.
_FORMAT_BITS: dict[str, int] = {
    "int8": 8,
    "int4": 4,
}

# Formats that are recognised as future work but explicitly not implemented in
# phase 1. Selecting one is a loud error naming the deferral, never a silent
# fallback to a different scale layout.
_DEFERRED_FORMATS: frozenset[str] = frozenset(
    {"nvfp4_w4a16", "nvfp4_w4a4", "mxfp4", "fp8", "fp8_e4m3"}
)

# Module leaf-names that must never be weight-quantized (numerically fragile /
# tiny). These are generic Megatron surface names, not model names.
_DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    "lm_head",
    "output_layer",
    "gate",  # MoE router gate
    "router",
    "embedding",
    "word_embeddings",
)


@dataclass(frozen=True)
class QATSpec:
    """Typed, explicit opt-in QAT configuration.

    Defaults are inert: ``enabled=False`` means callers get a bit-identical
    model with no quantizer nodes inserted.
    """

    enabled: bool = False
    format: str = "int8"
    group_size: int = 0  # 0 = per-tensor, -1 = per-output-channel, N>0 = block along in-features
    symmetric: bool = True
    ste_clip: bool = True  # zero grad outside representable range; False = pure pass-through
    ignore_patterns: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_IGNORE_PATTERNS)
    activation_bits: int | None = None  # weight-only in phase 1; W*A* is gated separately
    export_mode: str = "fake"  # "fake" (train) | "packed" (deploy snapshot)
    learnable_scales: bool = False  # LSQ future work; must be False in phase 1

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.format in _DEFERRED_FORMATS:
            raise ValueError(
                f"QAT format {self.format!r} is deferred to a later leaf and not implemented "
                "in phase 1 (int8/int4 only). It requires its own scale layout, export "
                "serializer and validation; do not alias it onto integer QAT."
            )
        if self.format not in _FORMAT_BITS:
            raise ValueError(
                f"Unknown QAT format {self.format!r}; phase-1 supports {sorted(_FORMAT_BITS)}."
            )
        if self.activation_bits is not None:
            raise ValueError(
                "activation quantization (W*A*) is not supported in phase 1; it needs a "
                "calibration/observer-freeze protocol and cross-DP amax sync before enabling."
            )
        if self.learnable_scales:
            raise ValueError("learnable_scales (LSQ) is deferred; phase 1 uses max calibration.")
        if self.export_mode not in ("fake", "packed"):
            raise ValueError(f"export_mode must be 'fake' or 'packed', got {self.export_mode!r}.")
        if self.group_size < -1:
            raise ValueError(f"group_size must be >= -1, got {self.group_size}.")

    @property
    def num_bits(self) -> int:
        return _FORMAT_BITS[self.format]

    def targets_module(self, name: str) -> bool:
        """True if a module should be quantized (no path component is on the ignore list).

        Matching is on dotted path *components* (exact, case-insensitive) so that
        e.g. the router leaf ``gate`` is skipped while the MLP linear ``gate_up``
        (a different component) is still quantized.
        """
        components = {c.lower() for c in name.split(".")}
        return not any(pat.lower() in components for pat in self.ignore_patterns)


def normalize_qat_spec(config: QATSpec | dict[str, Any] | None) -> QATSpec:
    if config is None:
        return QATSpec()
    if isinstance(config, QATSpec):
        return config
    if not isinstance(config, dict):
        raise TypeError(f"QAT config must be QATSpec, dict, or None, got {type(config)!r}.")
    values = dict(config)
    if "ignore_patterns" in values and not isinstance(values["ignore_patterns"], tuple):
        values["ignore_patterns"] = tuple(values["ignore_patterns"])
    return QATSpec(**values)


# ---------------------------------------------------------------------------
# Integer quant/dequant numerics
# ---------------------------------------------------------------------------


def _int_qrange(num_bits: int, symmetric: bool) -> tuple[int, int]:
    """Integer code range.

    Symmetric uses the restricted range ``[-(2^(b-1)-1), 2^(b-1)-1]`` (matches
    TensorRT/ModelOpt symmetric weight quant): scale = amax / (2^(b-1)-1) so the
    max magnitude maps exactly to the top code. Asymmetric (affine) uses the full
    unsigned range ``[0, 2^b-1]`` with a zero-point.
    """
    if symmetric:
        qmax = (1 << (num_bits - 1)) - 1
        return -qmax, qmax
    return 0, (1 << num_bits) - 1


def _reshape_for_groups(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    """Reshape a 2D ``[out, in]`` weight so the reduction dim is last.

    Returns ``(view, reduce_dim)``. ``reduce_dim`` is the axis over which amax is
    taken (kept as size-1 for broadcasting).
    """
    if group_size == 0:  # per-tensor
        return weight, -1  # sentinel: reduce over all elements
    if group_size == -1:  # per-output-channel: one scale per row
        return weight, 1
    # block along in-features
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"group_size={group_size} does not divide in_features={in_features}."
        )
    view = weight.reshape(out_features, in_features // group_size, group_size)
    return view, 2


def compute_amax(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Per-group max-abs statistic (detached; calibration is not differentiated)."""
    view, reduce_dim = _reshape_for_groups(weight.detach(), group_size)
    if reduce_dim == -1:
        return view.abs().amax()
    return view.abs().amax(dim=reduce_dim, keepdim=True)


def _compute_qparams(
    weight: torch.Tensor, num_bits: int, group_size: int, symmetric: bool
) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
    """Return ``(scale, zero_point, qmin, qmax)`` broadcastable to the grouped view."""
    qmin, qmax = _int_qrange(num_bits, symmetric)
    view, reduce_dim = _reshape_for_groups(weight.detach(), group_size)
    eps = torch.finfo(torch.float32).tiny
    if symmetric:
        amax = compute_amax(weight, group_size).float()
        scale = (amax / qmax).clamp_min(eps)
        return scale, None, qmin, qmax
    # affine
    if reduce_dim == -1:
        wmin = view.float().amin()
        wmax = view.float().amax()
    else:
        wmin = view.float().amin(dim=reduce_dim, keepdim=True)
        wmax = view.float().amax(dim=reduce_dim, keepdim=True)
    scale = ((wmax - wmin) / (qmax - qmin)).clamp_min(eps)
    zero_point = torch.round(qmin - wmin / scale)
    return scale, zero_point, qmin, qmax


class _FakeQuantizeSTE(torch.autograd.Function):
    """Q/DQ in forward, straight-through estimator in backward.

    ``scale``/``zero_point`` are treated as constants (calibration statistics),
    so no gradient flows to them. With ``clip=True`` the STE zeroes gradient for
    weights whose (pre-clamp) code falls outside ``[qmin, qmax]``.
    """

    @staticmethod
    def forward(ctx, weight, scale, zero_point, qmin, qmax, clip):  # type: ignore[override]
        orig_dtype = weight.dtype
        w = weight.float()
        if zero_point is None:
            q = torch.round(w / scale)
            q_clamped = q.clamp(qmin, qmax)
            w_hat = q_clamped * scale
        else:
            q = torch.round(w / scale + zero_point)
            q_clamped = q.clamp(qmin, qmax)
            w_hat = (q_clamped - zero_point) * scale
        if clip:
            ctx.save_for_backward((q >= qmin) & (q <= qmax))
            ctx.clip = True
        else:
            ctx.clip = False
        return w_hat.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        if ctx.clip:
            (mask,) = ctx.saved_tensors
            grad_output = grad_output * mask.to(grad_output.dtype)
        return grad_output, None, None, None, None, None


def fake_quantize_weight(weight: torch.Tensor, spec: QATSpec) -> torch.Tensor:
    """Differentiable (STE) fake-quantization of a 2D weight per ``spec``."""
    if weight.dim() != 2:
        raise ValueError(f"fake_quantize_weight expects a 2D [out, in] weight, got {tuple(weight.shape)}.")
    scale, zero_point, qmin, qmax = _compute_qparams(
        weight, spec.num_bits, spec.group_size, spec.symmetric
    )
    if spec.group_size > 0:
        out_features, in_features = weight.shape
        view = weight.reshape(out_features, in_features // spec.group_size, spec.group_size)
        w_hat = _FakeQuantizeSTE.apply(view, scale, zero_point, qmin, qmax, spec.ste_clip)
        return w_hat.reshape(out_features, in_features)
    return _FakeQuantizeSTE.apply(weight, scale, zero_point, qmin, qmax, spec.ste_clip)


# ---------------------------------------------------------------------------
# Deployment representation (packed integers + scales) — export only
# ---------------------------------------------------------------------------


def quantize_weight(weight: torch.Tensor, spec: QATSpec) -> dict[str, torch.Tensor]:
    """Produce the packed deployment snapshot for a BF16 weight.

    Returns a dict with the integer codes (``qweight``), ``scale`` and, for
    affine, ``zero_point``. The training step never calls this; it exists for
    export / rollout refit and for the round-trip validation contract.
    """
    scale, zero_point, qmin, qmax = _compute_qparams(
        weight, spec.num_bits, spec.group_size, spec.symmetric
    )
    view, _ = _reshape_for_groups(weight.detach(), spec.group_size)
    w = view.float()
    if zero_point is None:
        codes = torch.round(w / scale).clamp(qmin, qmax)
    else:
        codes = torch.round(w / scale + zero_point).clamp(qmin, qmax)
    codes = codes.reshape(weight.shape).to(torch.int8)
    out = {"qweight": codes, "scale": scale}
    if zero_point is not None:
        out["zero_point"] = zero_point
    return out


def dequantize_weight(packed: dict[str, torch.Tensor], spec: QATSpec) -> torch.Tensor:
    """Inverse of :func:`quantize_weight` — reconstruct the fake-quantized BF16 weight."""
    codes = packed["qweight"]
    scale = packed["scale"]
    out_features, in_features = codes.shape
    if spec.group_size > 0:
        codes_v = codes.reshape(out_features, in_features // spec.group_size, spec.group_size).float()
    else:
        codes_v = codes.float()
    if "zero_point" in packed:
        w = (codes_v - packed["zero_point"]) * scale
    else:
        w = codes_v * scale
    return w.reshape(out_features, in_features)


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack a 2D int4 code tensor (last dim even) into ``uint8`` (two codes/byte).

    Codes are the signed range ``[-7, 7]`` (or ``[0, 15]`` affine); they are
    offset into ``[0, 15]`` nibbles. Low nibble = even index, high nibble = odd.
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"pack_int4 needs an even last dim, got {codes.shape[-1]}.")
    ints = codes.to(torch.int32)
    lo = ints[..., 0::2] & 0x0F
    hi = ints[..., 1::2] & 0x0F
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor, *, signed: bool = True) -> torch.Tensor:
    """Inverse of :func:`pack_int4`. Returns int8 codes; sign-extends if ``signed``."""
    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)
    if signed:
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
    out = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return out.to(torch.int8)


# ---------------------------------------------------------------------------
# Parametrization: keep master weight, fake-quant on access
# ---------------------------------------------------------------------------


class WeightFakeQuant(nn.Module):
    """``torch.nn.utils.parametrize`` module that fake-quantizes a weight.

    Registered on a linear's ``weight``, it moves the trainable master weight to
    ``parametrizations.weight.original`` and returns ``W_hat`` on every access.
    A persistent ``amax`` buffer carries the calibration statistic into the
    checkpoint so quantizer state round-trips (design checkpoint contract).
    """

    def __init__(self, spec: QATSpec, weight_shape: torch.Size):
        super().__init__()
        self.spec = spec
        amax = compute_amax(torch.zeros(weight_shape), spec.group_size)
        # persistent so distckpt's named_buffers() loop saves/restores it.
        self.register_buffer("amax", amax.clone(), persistent=True)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.amax.copy_(compute_amax(weight, self.spec.group_size).to(self.amax.dtype))
        return fake_quantize_weight(weight, self.spec)


# ---------------------------------------------------------------------------
# Apply to modules / chunks (model-agnostic opt-in surface)
# ---------------------------------------------------------------------------


def _quantizable_weight_owner(module: nn.Module) -> nn.Module | None:
    """Return the sub-object that owns a 2D ``weight`` Parameter, if any.

    MLite parallel linears wrap the real GEMM as ``module.linear`` (TE) whose
    ``.weight`` is the parameter; plain ``nn.Linear`` owns ``.weight`` directly.
    Grouped expert linears expose 3D stacked weights and are handled by the
    caller (skipped here in phase 1).
    """
    inner = getattr(module, "linear", None)
    if isinstance(inner, nn.Module) and isinstance(getattr(inner, "weight", None), nn.Parameter):
        if inner.weight.dim() == 2:
            return inner
    if isinstance(getattr(module, "weight", None), nn.Parameter) and module.weight.dim() == 2:
        return module
    return None


def apply_qat_to_module(module: nn.Module, spec: QATSpec) -> bool:
    """Register weight fake-quant on a single module's 2D weight. Returns applied."""
    owner = _quantizable_weight_owner(module)
    if owner is None:
        return False
    if parametrize.is_parametrized(owner, "weight"):
        return False
    parametrize.register_parametrization(
        owner, "weight", WeightFakeQuant(spec, owner.weight.shape), unsafe=True
    )
    return True


def apply_qat_to_chunks(chunks, spec: QATSpec | dict[str, Any] | None) -> dict[str, int]:
    """Apply weight-only QAT to every eligible linear in the model chunks.

    Opt-in and inert by default: with a disabled spec nothing is registered and
    the model stays bit-identical. Must be called *after* chunk build / HF load
    and *before* optimizer construction so the optimizer captures the master
    ``weight.original`` parameter. Routers / lm_head / embeddings are skipped.
    """
    spec = normalize_qat_spec(spec)
    stats = {"quantized_modules": 0, "skipped_ignored": 0, "skipped_no_weight": 0}
    if not spec.enabled:
        return stats
    for chunk in chunks:
        for name, module in chunk.named_modules():
            if _quantizable_weight_owner(module) is None:
                continue
            if not spec.targets_module(name):
                stats["skipped_ignored"] += 1
                continue
            if apply_qat_to_module(module, spec):
                stats["quantized_modules"] += 1
    return stats


def qat_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect quantizer buffers (amax) for inspection / explicit persistence."""
    out = {}
    for name, buf in model.named_buffers():
        if name.endswith(".amax") and "parametrizations" in name:
            out[name] = buf.detach().clone()
    return out


__all__ = [
    "QATSpec",
    "WeightFakeQuant",
    "apply_qat_to_chunks",
    "apply_qat_to_module",
    "compute_amax",
    "dequantize_weight",
    "fake_quantize_weight",
    "normalize_qat_spec",
    "pack_int4",
    "qat_state_dict",
    "quantize_weight",
    "unpack_int4",
]
