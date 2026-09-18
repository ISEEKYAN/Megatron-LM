# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Release bindings and owner policy for the shared checkpoint machinery."""

from dataclasses import replace
from functools import partial

import torch
from megatron.lite.primitive.ckpt import binding_checkpoint as checkpoint
from megatron.lite.primitive.ckpt import tensor_archive as archive
from megatron.lite.primitive.ckpt.tensor_archive import (
    CheckpointTensorStore,
    TensorEntry,
    _tensor,
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


class WeightSpec:
    optional_prefix = 'mtp.'
    staging_prefix = '.v41-export-'
    archive_required_message = (
        'Complete MTP/vision/aligner archival storage is required for export'
    )
    frozen_storage_message = 'Frozen Engram requires FP8 table storage'

    @staticmethod
    def row_block(name):
        return 1 if name.endswith('.engram.embed.weight') else None

    @staticmethod
    def row_shard(owner):
        return owner.lookup if isinstance(owner, ShardedEngramTable) else None

    @staticmethod
    def scale_block(binding):
        return (
            1 if binding.encoding == 'I8' or binding.role == 'engram_table' else 32,
            32,
        )

    @staticmethod
    def frozen_storage(owner):
        if isinstance(owner, EngramTable) and owner.master is None:
            return owner.weight, owner.scale

    @staticmethod
    def refresh_storage(owner):
        if isinstance(owner, EngramTable):
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


_SPEC = WeightSpec()
export_model = partial(checkpoint.export_model, spec=_SPEC)
export_checkpoint = partial(checkpoint.export_checkpoint, spec=_SPEC)
save_model = partial(checkpoint.save_model, spec=_SPEC)


def bind_checkpoint(model, records, *, store=None, allow_missing_mtp=False):
    return checkpoint.bind_checkpoint(
        model, records, spec=_SPEC, store=store, allow_missing_archive=allow_missing_mtp
    )


def load_model(model, path, *, allow_missing_mtp=False):
    return checkpoint.load_model(
        model, path, spec=_SPEC, allow_missing_archive=allow_missing_mtp
    )


def load_weight(store, name, *, output_dtype=torch.bfloat16):
    return archive.load_weight(
        store,
        name,
        output_dtype=output_dtype,
        row_block=_SPEC.row_block(name),
        read_tensor=_tensor,
    )


def export_hf_weights(chunks, model_cfg, ps, **kwargs):
    """Adapt online export options without changing the archival checkpoint stream."""
    if len(chunks) != 1:
        raise NotImplementedError('Single-rank V4.1 export requires one chunk')
    # Match Qwen3.5: no executable MTP path; local prefixes are a legacy hint.
    include_mtp_only = kwargs.pop('include_mtp_only', False)
    kwargs.pop('include_local_prefixes', None)
    if include_mtp_only:
        return
    limit = kwargs.pop('limit', None)
    for exported_params, pair in enumerate(export_checkpoint(chunks[0], **kwargs), 1):
        yield pair
        # Match the shared HF exporter's post-yield limit check.
        if limit is not None and exported_params >= limit:
            return
