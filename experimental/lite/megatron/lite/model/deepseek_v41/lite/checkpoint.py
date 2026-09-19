# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Release weight policy; transfer and decoding live in hf_weights."""
from dataclasses import replace
from functools import partial

from megatron.lite.primitive.ckpt.binding_records import (
    DeferredModule,
    Rule,
    TensorBinding,
)
from megatron.lite.primitive.ckpt.hf_weights import (
    export_checkpoint as _export_checkpoint,
)
from megatron.lite.primitive.ckpt.hf_weights import (
    load_bound_model,
    save_bound_model,
)
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
    if target is not None or resync_config is not None:
        raise NotImplementedError(
            'V4.1_HF_SAVE_RESYNC_UNSUPPORTED: target/resync_config require '
            'a resync exporter; this entry point only exports archival HF weights'
        )
    if len(chunks) != 1:
        raise NotImplementedError('Export requires a single complete chunk')
    include_mtp_only = kwargs.pop('include_mtp_only', False)
    kwargs.pop('include_local_prefixes', None)
    limit = kwargs.pop('limit', None)
    if include_mtp_only:
        if kwargs:
            raise TypeError('MTP-only export does not accept additional options')
        return
    for count, pair in enumerate(export_checkpoint(chunks[0], **kwargs), 1):
        yield pair
        if limit is not None and count >= limit:
            break
