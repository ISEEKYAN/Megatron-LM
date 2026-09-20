# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Release weight policy; transfer and decoding live in hf_weights."""
from dataclasses import replace
from functools import partial

import torch
from megatron.lite.primitive.ckpt.binding_records import (
    DeferredModule,
    Rule,
    TensorBinding,
)
from megatron.lite.primitive.ckpt.hf_weights import (
    export_checkpoint as _export_checkpoint,
)
from megatron.lite.primitive.ckpt.hf_weights import load_bound_model, save_bound_model
from megatron.lite.primitive.modules.engram_lookup import (
    EngramTable,
    ShardedEngramTable,
)


def validate_execution(*, enable_dspark_execution: bool = False) -> None:
    if type(enable_dspark_execution) is not bool:
        raise TypeError("enable_dspark_execution must be bool")
    if enable_dspark_execution:
        raise NotImplementedError("DSpark execution is not implemented")


class DeepseekV41WeightSpec:
    optional_prefix = 'mtp.'

    @staticmethod
    def row_block(name):
        return 1 if name.endswith('.engram.embed.weight') else None

    @staticmethod
    def encode(name, tensor, encoding):
        from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
        from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

        if encoding == 'I8':
            return quantize_mxfp4(tensor)
        return quantize_block_fp8(
            tensor,
            (DeepseekV41WeightSpec.row_block(name) or 32, 32),
            scale_format='e8m0',
        )

    @staticmethod
    def row_shard(owner):
        return getattr(owner, "lookup", None)

    @staticmethod
    def frozen_storage(owner):
        if (
            isinstance(owner, (EngramTable, ShardedEngramTable))
            and owner.master is None
        ):
            return owner.weight, owner.scale

    @staticmethod
    def refresh_storage(owner):
        if isinstance(owner, (EngramTable, ShardedEngramTable)):
            owner.refresh_storage()

    @staticmethod
    def expand_bindings(model, available):
        if model.ps.ep_size > 1:
            experts = model.config.to_hf_dict()['text_config']['n_routed_experts']
            for name, binding in list(available.items()):
                if '.ffn.experts.' in name:
                    prefix, tail = name.split('.ffn.experts.')
                    suffix = tail.split('.', 1)[1]
                    for index in range(experts):
                        key = f'{prefix}.ffn.experts.{index}.{suffix}'
                        available.setdefault(key, replace(binding, release_key=key))
        return available


_SPEC = DeepseekV41WeightSpec()
export_checkpoint = partial(_export_checkpoint, spec=_SPEC)
save_model = partial(save_bound_model, spec=_SPEC)


def load_model(model, path, *, allow_missing_mtp=False):
    return load_bound_model(model, path, _SPEC, allow_missing_archive=allow_missing_mtp)


def export_hf_weights(
    chunks, model_cfg, ps, *, target=None, resync_config=None, **kwargs
):
    from .resync import transport_weights, validate_target

    deployment = validate_target(target, resync_config)
    if len(chunks) != 1:
        raise NotImplementedError('Export requires a single complete chunk')
    if kwargs.pop('include_mtp_only', False):
        raise NotImplementedError('V4.1_HF_EXPORT_MTP_ONLY_UNSUPPORTED')
    if kwargs.pop('include_local_prefixes', None) is not None:
        raise NotImplementedError('V4.1_HF_EXPORT_LOCAL_PREFIXES_UNSUPPORTED')
    limit = kwargs.pop('limit', None)
    if deployment and limit is not None:
        raise ValueError('DS4.1 resync requires a complete generation, without limit')
    if deployment:
        if kwargs.pop('row_chunks', True) is not True:
            raise ValueError('DS4.1 resync requires bounded row chunks')
        budget = kwargs.get('buffer_max_size_bytes', 5 * 1024**3)
        if type(budget) is not int or budget <= 0:
            raise ValueError('Invalid resync buffer budget')
        # Non-row codecs operate on a matrix. Refuse one whose conservative
        # workspace bound cannot fit; never silently exceed a small budget.
        for name, binding in chunks[0].tensor_bindings.items():
            if (
                binding.role != 'scale'
                and _SPEC.row_block(name) != 1
                and binding.encoding in ('I8', 'F8_E4M3')
                and binding.tensor.dtype
                in (torch.float32, torch.bfloat16, torch.float16)
                and binding.tensor.numel() * (64 if binding.encoding == 'I8' else 32)
                + 8192
                > budget // 2
            ):
                raise ValueError(f'Resync buffer too small for matrix codec: {name}')
        # Reserve source, packed payload, and overlapping iterator handoff views.
        kwargs['buffer_max_size_bytes'] = (
            kwargs.get('buffer_max_size_bytes', 5 * 1024**3) // 4
        )
        kwargs['row_chunks'] = True
        if kwargs.pop('export_dtype', None) not in (
            None,
            'bf16',
            'bfloat16',
            torch.bfloat16,
        ):
            raise ValueError('DS4.1 resync uses official mixed FP32/BF16 dtypes')
    weights = export_checkpoint(chunks[0], **kwargs)
    if deployment:
        weights = transport_weights(weights, deployment=True)
    for count, pair in enumerate(weights, 1):
        yield pair
        if limit is not None and count >= limit:
            break
