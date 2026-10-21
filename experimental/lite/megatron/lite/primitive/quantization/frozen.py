# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Immutable release codes; logical Core weights have no BF16 backing storage.

Only LoRA release builds opt in. Matrix adjoints decode their own operands; no
model-wide decoded cache or master copy is retained. Small unquantized release
parameters keep their actual release dtype/storage.
"""
import math

import torch
from torch.utils._pytree import tree_map


class FrozenWeight(torch.Tensor):
    @staticmethod
    def __new__(cls, codes, scale, shape, kind, dtype=torch.bfloat16):
        if kind not in ('mxfp4', 'fp8_block') or codes.device != scale.device:
            raise ValueError('Frozen weights require colocated native codes/scales')
        rows, columns = math.prod(shape[:-1]), shape[-1]
        if kind == 'mxfp4':
            valid = codes.numel() * 2 == rows * columns and scale.numel() * 32 == rows * columns
        else:
            valid = codes.numel() == rows * columns and scale.numel() * 1024 == rows * columns
        expected_dtype = torch.int8 if kind == 'mxfp4' else torch.float8_e4m3fn
        if not valid or codes.dtype != expected_dtype or scale.dtype != torch.float8_e8m0fnu:
            raise ValueError('Native release weight/scale shape or dtype mismatch')
        return torch.Tensor._make_wrapper_subclass(
            cls, tuple(shape), dtype=dtype, device=codes.device, requires_grad=False
        )

    def __init__(self, codes, scale, shape, kind, dtype=torch.bfloat16):
        self.codes, self.scale, self.kind = codes, scale, kind

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        # Module._apply may keep a Parameter and replace only its TensorImpl.
        # Its Python-owned native payload must follow that storage transition.
        if (
            not isinstance(value, FrozenWeight)
            or value.shape != self.shape
            or value.kind != self.kind
        ):
            raise ValueError('Immutable native weights require matching native storage')
        torch.Tensor.data.__set__(self, value)
        self.codes, self.scale, self.kind = value.codes, value.scale, value.kind

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            raise ValueError('Native release base is immutable; train adapters only')
        return super().requires_grad_(False)

    def decoded(self):
        from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8
        from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4

        decode = dequantize_mxfp4 if self.kind == 'mxfp4' else dequantize_block_fp8
        args = () if self.kind == 'mxfp4' else ((32, 32),)
        return decode(self.codes, self.scale, *args).reshape(self.shape).to(self.dtype)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        first = args[0] if args else None
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            return FrozenWeight(first.codes, first.scale, first.shape, first.kind, first.dtype)
        if func == torch.ops.aten._to_copy.default:
            device = kwargs.get('device', first.device)
            return FrozenWeight(
                first.codes.to(device=device),
                first.scale.to(device=device),
                first.shape,
                first.kind,
                kwargs.get('dtype', first.dtype),
            )
        if func in (torch.ops.aten.view.default, torch.ops.aten.reshape.default):
            shape = list(args[1])
            if -1 in shape:
                shape[shape.index(-1)] = first.numel() // -math.prod(shape)
            if math.prod(shape) != first.numel():
                raise ValueError('Frozen weight reshape changes element count')
            return FrozenWeight(first.codes, first.scale, shape, first.kind, first.dtype)
        if func == torch.ops.aten.cat.default and all(isinstance(w, cls) for w in first):
            dimension = kwargs.get('dim', args[1] if len(args) > 1 else 0)
            if dimension != 0 or len({w.kind for w in first}) != 1:
                raise ValueError('Native fused weights support row concatenation only')
            return FrozenWeight(
                torch.cat([w.codes for w in first]),
                torch.cat([w.scale for w in first]),
                (sum(w.shape[0] for w in first), *first[0].shape[1:]),
                first[0].kind,
                first[0].dtype,
            )
        if func == torch.ops.aten.copy_.default and isinstance(first, cls):
            source = args[1]
            if (
                not isinstance(source, cls)
                or source.shape != first.shape
                or source.kind != first.kind
            ):
                raise ValueError('Frozen release weights cannot receive decoded/mutable values')
            if first.codes.data_ptr() != source.codes.data_ptr():
                first.codes.copy_(source.codes)
            if first.scale.data_ptr() != source.scale.data_ptr():
                first.scale.copy_(source.scale)
            return first
        if func._schema.is_mutable:
            raise RuntimeError(f'Immutable native weight rejects {func}')
        decode = lambda x: x.decoded() if isinstance(x, cls) else x
        return func(*tree_map(decode, args), **tree_map(decode, kwargs))


def native_payload(weight, kind):
    if isinstance(weight, FrozenWeight):
        if weight.kind != kind:
            raise ValueError(f'Expected {kind}, got native {weight.kind}')
        return weight.codes, weight.scale
    return None
