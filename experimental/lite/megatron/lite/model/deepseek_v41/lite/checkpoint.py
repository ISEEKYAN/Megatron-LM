# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Release weight policy; transfer and decoding live in hf_weights."""
from dataclasses import replace
from functools import partial

import megatron.lite.primitive.ckpt.binding_records as _imports_binding_records
import megatron.lite.primitive.ckpt.hf_weights as _imports_hf_weights
import megatron.lite.primitive.modules.engram_lookup as _imports_engram_lookup
import torch

DeferredModule = _imports_binding_records.DeferredModule
Rule = _imports_binding_records.Rule
TensorBinding = _imports_binding_records.TensorBinding
_export_checkpoint = _imports_hf_weights.export_checkpoint
load_bound_model = _imports_hf_weights.load_bound_model
save_bound_model = _imports_hf_weights.save_bound_model
EngramTable = _imports_engram_lookup.EngramTable
ShardedEngramTable = _imports_engram_lookup.ShardedEngramTable


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
        import megatron.lite.primitive.quantization.block_fp8 as _imports_block_fp8

        quantize_block_fp8 = _imports_block_fp8.quantize_block_fp8
        from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

        if encoding == 'I8':
            return quantize_mxfp4(tensor)
        return quantize_block_fp8(
            tensor,
            (DeepseekV41WeightSpec.row_block(name) or 32, 32),
            scale_format='e8m0',
        )

    @staticmethod
    def codec_tile(name, encoding):
        # Scale-block row alignment; conservative bytes per source element for
        # codec workspace and for encoded weight plus scale (at most 1 + 1/32).
        if encoding == 'I8':
            return 1, 64, 2
        return DeepseekV41WeightSpec.row_block(name) or 32, 32, 2

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


def validate_resync_budget(model, budget):
    """Return the exporter share of budget; refuse before the first yield.

    Non-row codecs encode tiles, so each matrix needs one tile of workspace
    plus its whole encoded output within the share.
    """
    from megatron.lite.primitive.ckpt.hf_weights import plan_matrix_codec

    if type(budget) is not int or budget <= 0:
        raise ValueError('Invalid resync buffer budget')
    share = budget // 4
    for name, binding in model.tensor_bindings.items():
        if (
            binding.role != 'scale'
            and _SPEC.row_block(name) != 1
            and binding.encoding in ('I8', 'F8_E4M3')
            and binding.tensor.dtype in (torch.float32, torch.bfloat16, torch.float16)
        ):
            try:
                plan_matrix_codec(_SPEC, name, binding.tensor, binding.encoding, share)
            except ValueError:
                raise ValueError(
                    f'Resync buffer too small for matrix codec: {name}'
                ) from None
    return share


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
        # Reserve source, packed payload, and overlapping iterator handoff views.
        kwargs['buffer_max_size_bytes'] = validate_resync_budget(chunks[0], budget)
        kwargs['row_chunks'] = True
        if kwargs.pop('export_dtype', None) not in (
            None,
            'bf16',
            'bfloat16',
            torch.bfloat16,
        ):
            raise ValueError('DS4.1 resync uses official mixed FP32/BF16 dtypes')
    frozen = deployment and (resync_config or {}).get('freeze_engram', False)
    manifest = None
    spec = _SPEC
    if frozen:
        import torch.distributed as dist
        from megatron.lite.primitive.ckpt.frozen_storage import storage_digest

        from .resync import frozen_tables_transport

        model = chunks[0]
        manifest = {}
        for name, binding in model.tensor_bindings.items():
            if binding.role != 'engram_table':
                continue
            storage = _SPEC.frozen_storage(binding.owner)
            if storage is None:
                raise ValueError('freeze_engram requires tables without FP32 masters')
            local = storage_digest(*storage)
            lookup = _SPEC.row_shard(binding.owner)
            hashes = [local]
            if lookup is not None and lookup.group is not None:
                hashes = [None] * dist.get_world_size(lookup.group)
                dist.all_gather_object(hashes, local, group=lookup.group)
            manifest[name] = hashes
        if ps.pp_size > 1:
            manifests = [None] * ps.pp_size
            dist.all_gather_object(manifests, manifest, group=ps.pp_group)
            merged = {}
            for stage_manifest in manifests:
                if merged.keys() & stage_manifest.keys():
                    raise ValueError('Frozen Engram table has multiple PP owners')
                merged.update(stage_manifest)
            manifest = merged
        previous = getattr(model, '_resync_frozen_tables', None)
        if previous is not None and previous != manifest:
            raise ValueError('Frozen Engram storage changed since initial resync')
        reuse = previous is not None
        if reuse:

            class FrozenSpec(DeepseekV41WeightSpec):
                @staticmethod
                def skip_binding(binding):
                    return binding.role == 'engram_table'

            spec = FrozenSpec()
        yield frozen_tables_transport(manifest, reuse)
    weights = _export_checkpoint(
        chunks[0], spec=spec, local_stage=ps.pp_size > 1, **kwargs
    )
    if ps.pp_size > 1:
        import megatron.lite.primitive.ckpt.pipeline_stream as _imports_pipeline_stream

        broadcast_stage_stream = _imports_pipeline_stream.broadcast_stage_stream

        from .resync import decoded_weights

        # Encode each local stage before crossing PP; RowChunk planes travel
        # together and never materialize a full Engram table.
        weights = broadcast_stage_stream(
            transport_weights(weights, deployment=deployment), ps
        )
        weights = decoded_weights(weights)
    if deployment:
        weights = transport_weights(weights, deployment=True)
    for count, pair in enumerate(weights, 1):
        yield pair
        if limit is not None and count >= limit:
            break
    if manifest is not None:
        chunks[0]._resync_frozen_tables = manifest
