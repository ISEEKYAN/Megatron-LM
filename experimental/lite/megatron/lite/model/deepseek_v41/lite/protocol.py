# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text protocol with replicated data parallel training and explicit optimizer policy."""

import math
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from functools import partial

import megatron.lite.model.protocol_utils as _imports_protocol_utils
import megatron.lite.primitive.modules.router_replay as _imports_router_replay
import megatron.lite.primitive.modules.vision_training as _imports_vision_training
import megatron.lite.primitive.ops.linear_cross_entropy as _imports_linear_cross_entropy
import megatron.lite.primitive.parallel.state as _imports_state
import torch
from megatron.lite.model import protocol_utils as _protocol_utils
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.topology import pipeline_cuts
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.config_fields import project_fields
from megatron.lite.primitive.parallel.owned_ddp import wrap_owned_ddp

ParallelState = _imports_state.ParallelState
init_parallel = _imports_state.init_parallel
from megatron.lite.primitive.parallel.thd import roll_packed_thd_left
from megatron.lite.primitive.train_step import prepare_microbatches
from megatron.lite.runtime.contracts import ParallelConfig
from megatron.lite.runtime.contracts.loss import get_loss_context

from .checkpoint import export_hf_weights as _export_hf_weights_impl
from .checkpoint import load_model, save_model
from .optimizer_groups import OptimizerConfig, V41Optimizer

_pack_r3_replay_mask = _imports_protocol_utils.pack_r3_replay_mask
_pack_routed_experts = _imports_protocol_utils.pack_routed_experts
RouterReplay = _imports_router_replay.RouterReplay
RouterReplayAction = _imports_router_replay.RouterReplayAction
VisionSchedule = _imports_vision_training.VisionSchedule
VisionTrainability = _imports_vision_training.VisionTrainability
linear_cross_entropy = _imports_linear_cross_entropy.linear_cross_entropy

# HF checkpoints store trainable masters and byte-preserved archives.


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = None
    optimizer_config: OptimizerConfig | None = None
    device: str = 'cuda'
    dtype: torch.dtype = torch.bfloat16
    quantized: bool = True
    w4a8_experts: bool = False
    deployment_math: bool = False
    use_deepep: bool = False
    # DeepEP dispatch VJP reduces route gradients as all-to-all does (bitwise
    # dX across dispatchers); costs one gradient row per route on the wire.
    deepep_routed_backward: bool = False
    token_map: list[int] | None = None
    trainable_engram: bool = False
    shard_engram: bool = True
    # Frozen Engram tables stay in host memory; lookups copy only gathered rows.
    host_engram: bool = False
    text_only: bool = True
    vision_trainability: VisionTrainability | None = None
    external_vision_device: str | None = None
    gate_temperature: float = 1.0
    bias_rate: float = 0.001
    enable_dspark_execution: bool = False
    # Pipeline stage boundaries: one cut, increasing cuts (PP>2), or None for
    # the topology's default two-stage cut. Each must keep owners with readers.
    pipeline_split_layer: int | tuple[int, ...] | None = None
    # 'full': recompute every CSA-state-consuming layer and the FFN of each
    # KV/index source layer in backward (MCore full/uniform/1 granularity).
    # Selective (MCore module names): 'moe' recomputes every layer's FFN;
    # 'attn' recomputes the attention of layers that consume CSA state without
    # owning it. Give a list or a comma-separated string.
    recompute: str | list[str] | None = None

    def __post_init__(self):
        # Runtime/VERL configurations arrive as serialized mappings.
        if isinstance(self.optimizer_config, Mapping):
            object.__setattr__(
                self, "optimizer_config", OptimizerConfig(**self.optimizer_config)
            )
        if self.pipeline_split_layer is not None and not isinstance(
            self.pipeline_split_layer, int
        ):
            object.__setattr__(
                self, "pipeline_split_layer", tuple(self.pipeline_split_layer)
            )
        if isinstance(self.dtype, str):
            dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
            if self.dtype not in dtypes:
                raise ValueError("V4.1 residual dtype must be BF16 or FP32")
            object.__setattr__(self, "dtype", dtypes[self.dtype])


def build_model_config(source, **overrides):
    if overrides:
        raise ValueError('Apply overrides to the explicit nested source config')
    return (
        DeepseekV41Config(source)
        if isinstance(source, dict)
        else DeepseekV41Config.from_hf(source)
    )


def _cut_tuple(cuts):
    return () if cuts is None else (cuts,) if isinstance(cuts, int) else tuple(cuts)


def _validate_parallel(c, p, cut=20):
    """`cut` is the topology-validated stage boundary (or boundaries)."""
    if p.pp > 1 and (not c.text_only or c.external_vision_device is not None):
        raise NotImplementedError(
            'V4.1_PP_TEXT_ONLY: PP currently supports text-only training; use PP=1 for multimodal training'
        )
    if (
        p.pp > 1
        and c.pipeline_split_layer is not None
        and _cut_tuple(c.pipeline_split_layer) != _cut_tuple(cut)
    ):
        raise NotImplementedError(
            f'V4.1_PP_CSA2_PAYLOAD_UNSUPPORTED: expected topology-validated split {cut}'
        )
    if type(p.ep) is not int or p.ep < 1:
        raise ValueError('EP size must be a positive integer')
    if p.pp > 1 and (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() % (p.pp * p.ep) != 0
    ):
        raise ValueError(
            'V4.1_PP_WORLD: PP requires an initialized world divisible by PP * EP'
        )
    if type(p.cp) is not int or p.cp < 1:
        raise ValueError('CP size must be a positive integer')
    if c.use_deepep and p.cp > 1:
        # No CP x DeepEP parity gate exists; fail closed until one does.
        raise NotImplementedError(
            'V4.1_DEEPEP_CP_UNSUPPORTED: DeepEP dispatch with CP>1 is not gated; '
            'use the all-to-all dispatcher (use_deepep=False) or CP=1'
        )
    # EP-over-CP as in MCore and DeepSeek V4: dense ranks decompose as
    # (TP, CP, DP, PP) and experts as (ETP, EP, EDP, PP); the MoE dispatcher
    # exchanges every CP shard's tokens within the EP group.
    if p.cp > 1 and (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() % p.cp != 0
    ):
        raise ValueError('CP requires an initialized distributed world divisible by cp')
    if p.ep > 1 and (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() % p.ep != 0
    ):
        raise ValueError('EP requires an initialized distributed world divisible by ep')


def _pipeline_stages(model_cfg, c, p):
    """Every stage's ``[start, end)`` layer range, with owner-safe boundaries.

    Cuts come from ``pipeline_split_layer`` or, for a custom MCore
    ``pp_layout``, from the layer ranges that layout assigns to each rank.
    """
    count = len(model_cfg.topology)
    if p.pp == 1:
        return ((0, count),)
    requested = c.pipeline_split_layer
    if p.pp_layout is not None:
        import megatron.lite.primitive.parallel.pp as _imports_pp

        if requested is not None:
            raise ValueError('V4.1_PP_SPLIT: set pp_layout or pipeline_split_layer')
        layers = [
            _imports_pp.build_pipeline_chunk_layout(
                count,
                ParallelState(
                    pp_size=p.pp,
                    pp_rank=rank,
                    pp_is_first=rank == 0,
                    pp_is_last=rank == p.pp - 1,
                    pp_layout=p.pp_layout,
                ),
            ).layer_indices
            for rank in range(p.pp)
        ]
        if any(not ids for ids in layers) or [i for ids in layers for i in ids] != list(
            range(count)
        ):
            raise ValueError(
                'V4.1_PP_SPLIT: pp_layout must give every stage contiguous layers'
            )
        requested = tuple(ids[0] for ids in layers[1:])
    bounds = (0, *pipeline_cuts(model_cfg.topology, requested, p.pp), count)
    return tuple(zip(bounds, bounds[1:]))


def _construct_model(model_cfg, impl_cfg, ps, layer_range, stages=None):
    from .model import DeepseekV41Model

    c = impl_cfg
    with torch.device(impl_cfg.device):
        return DeepseekV41Model(
            model_cfg,
            parallel_state=ps,
            layer_range=layer_range,
            pipeline_split_layer=(
                tuple(start for start, _ in stages[1:])
                if stages is not None and len(stages) > 1
                else None
            ),
            **project_fields(
                vars(c),
                'token_map quantized w4a8_experts deployment_math use_deepep deepep_routed_backward '
                'trainable_engram shard_engram host_engram '
                'gate_temperature bias_rate enable_dspark_execution',
            ),
        )


def build_model(model_cfg, *, impl_cfg):
    c, p = impl_cfg, impl_cfg.parallel
    unsupported = [
        key
        for key, invalid in (
            ('tp', p.tp != 1),
            ('vpp', p.vpp != 1),
            ('etp', p.etp not in (None, 1)),
        )
        if invalid
    ]
    if unsupported:
        raise NotImplementedError(
            f'V4.1_UNSUPPORTED_PARALLELISM: {", ".join(unsupported)}; '
            'supported: DP, EP, contiguous CP (alone or with EP) and text-only PP '
            '(owner-safe cuts or pp_layout), in any combination; TP/VPP/ETP are unsupported'
        )
    stages = _pipeline_stages(model_cfg, c, p)
    _validate_parallel(c, p, cut=tuple(start for start, _ in stages[1:]))
    ps = ParallelState()
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        ps = init_parallel(replace(p, etp=1))
    if impl_cfg.optimizer not in (None, 'muon'):
        raise ValueError('V4.1 optimizer must be explicitly selected as muon')
    if impl_cfg.optimizer is None and impl_cfg.optimizer_config is not None:
        raise ValueError('optimizer_config requires selecting the V4.1 optimizer')
    if impl_cfg.optimizer == 'muon' and not isinstance(
        impl_cfg.optimizer_config, OptimizerConfig
    ):
        raise ValueError(
            'V4.1 Muon requires explicit optimizer_config including NS backend settings'
        )
    if impl_cfg.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('V4.1 residual dtype must be BF16 or FP32')
    if c.w4a8_experts and c.dtype != torch.bfloat16:
        raise ValueError('V4.1 W4A8 requires BF16 residuals')
    if c.deployment_math and (not c.w4a8_experts or not c.quantized):
        raise ValueError('Deployment math requires quantized W4A8 experts')
    if c.deployment_math and (c.dtype != torch.bfloat16 or p.tp != 1):
        # Contiguous CP is supported (gathered KV/compressor, split SWA writer);
        # each rank evaluates the deployment head on its own rows.
        raise ValueError('Deployment math requires BF16 and TP=1')
    layer_range = stages[ps.pp_rank] if p.pp > 1 else None
    model = _construct_model(model_cfg, impl_cfg, ps, layer_range, stages)
    model.pipeline_residual_dtype = impl_cfg.dtype
    from megatron.lite.primitive.recompute import parse_recompute_spec

    selective = parse_recompute_spec(impl_cfg.recompute)
    if selective == ['full']:
        selective = []
    elif not set(selective) <= {'moe', 'attn'} or len(set(selective)) != len(selective):
        raise ValueError(
            'V4.1 recompute must be None, "full" or a selection of "moe", "attn"'
        )
    model.recompute = 'full' if impl_cfg.recompute == 'full' else None
    from .model import owns_attention_state, recompute_attention, recompute_ffn

    for layer in model.layers if impl_cfg.recompute is not None else ():
        if layer is None:
            continue
        if impl_cfg.recompute == 'full' and owns_attention_state(layer):
            recompute_ffn(layer.ffn)
        if 'moe' in selective:
            recompute_ffn(layer.ffn)
        if 'attn' in selective and not owns_attention_state(layer):
            recompute_attention(layer.attn)
    import megatron.lite.primitive.modules.native_fp32_linear as _imports_native_fp32_linear
    from megatron.lite.primitive.modules.engram_lookup import EngramTable

    configure_residual_projections = (
        _imports_native_fp32_linear.configure_residual_projections
    )

    optimizing = impl_cfg.optimizer == 'muon'
    if optimizing and impl_cfg.device == 'meta':
        raise ValueError(
            'Materialize V4.1 parameters before constructing optimizer state'
        )
    if impl_cfg.vision_trainability is not None:
        impl_cfg.vision_trainability.apply(model)
    elif impl_cfg.text_only:
        for binding in model.parameter_bindings():
            if binding.role in ('vision', 'aligner', 'image_delimiter'):
                binding.tensor.requires_grad_(False)
    configure_residual_projections(model, c.dtype, optimizing, EngramTable)
    if model.engram_hash is not None:
        model.engram_hash.to(device=impl_cfg.device)
    optimizer = None
    if optimizing:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
                parameter.main_grad = None
        model.residual_dtype = impl_cfg.dtype
        optimizer = V41Optimizer(
            model,
            impl_cfg.optimizer_config,
            dp_group=ps.dp_cp_group if ps.cp_size > 1 else ps.dp_group,
            ps=ps,
        )
    if impl_cfg.external_vision_device is not None:
        if impl_cfg.vision_trainability is None:
            raise ValueError('External vision requires an explicit post-training mask')
        model.vision_schedule = VisionSchedule(model, impl_cfg.external_vision_device)
    execution_model = wrap_owned_ddp(
        model,
        ps,
        optimizing=optimizing,
        external_device=c.external_vision_device,
        row_tables=[
            b.engram.embed
            for b in model.layers
            if b is not None and b.engram is not None
        ],
        shard_group=model.engram_group,
        manual_dense_sync=optimizing and optimizer.host_update is not None,
    )
    transfer_extras = {}
    if optimizing and optimizer.host_update is not None:
        transfer_extras['pre_model_device_transfer_hook'] = (
            optimizer.host_update.prepare_model_transfer
        )
    return ModelBundle(
        [model],
        ps,
        optimizer=optimizer,
        finalize_grads=(
            optimizer.finalize_grads
            if optimizing
            and (
                ps.ep_size > 1
                or model.engram_group is not None
                or optimizer.host_update is not None
            )
            else None
        ),
        forward_step=partial(
            _forward_step, optimizer=optimizer, execution_model=execution_model
        ),
        extras={
            'model_cfg': model_cfg,
            **({'pipeline_dtype': torch.float32} if p.pp > 1 else {}),
            'vision_schedule': model.vision_schedule,
            'prepare_microbatches': partial(
                prepare_microbatches,
                dp_group=ps.dp_cp_group if ps.cp_size > 1 else ps.dp_group,
                cp_rank=ps.cp_rank,
                cp_size=ps.cp_size,
            ),
            'optimizer_backend': 'none' if optimizer is None else 'v41',
            'parameter_bindings': model.parameter_bindings,
            **transfer_extras,
        },
    )


def load_hf_weights(chunk, hf_path, model_cfg, ps):
    if not hf_path:
        raise ValueError('V4.1_HF_PATH_REQUIRED: provide a checkpoint directory')
    load_model(chunk, hf_path)


export_hf_weights = _export_hf_weights_impl


def save_hf_weights(
    chunks, path, model_cfg, ps, *, target=None, resync_config=None, **kwargs
):
    from .resync import decoded_weights, validate_target

    if validate_target(target, resync_config):
        import json
        from pathlib import Path

        import megatron.lite.primitive.ckpt.hf_weights as _imports_hf_weights

        stream_export_to_shards = _imports_hf_weights.stream_export_to_shards

        budget = kwargs.pop('buffer_max_size_bytes', 5 * 1024**3)
        if type(budget) is not int or budget <= 0:
            raise ValueError('Invalid resync buffer budget')
        # Saved HF checkpoints must be standalone, including frozen tables.
        resync_config = dict(resync_config or {})
        resync_config.pop('freeze_engram', None)
        resync_config.pop('resend_frozen_engram', None)
        weights = export_hf_weights(
            chunks,
            model_cfg,
            ps,
            target=target,
            resync_config=resync_config,
            buffer_max_size_bytes=budget,
            **kwargs,
        )
        stream_export_to_shards(
            decoded_weights(weights), str(path), shard_size_bytes=budget // 4
        )
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            (Path(path) / 'config.json').write_text(
                json.dumps(chunks[0].config.to_hf_dict(), indent=2) + '\n'
            )
        return
    if len(chunks) != 1:
        raise NotImplementedError('Single-rank V4.1 export requires one chunk')
    save_model(chunks[0], path, **kwargs)


def vocab_size(model_cfg):
    return model_cfg.to_hf_dict()['text_config']['vocab_size']


# Both replay inputs use V4's shared packers with V4.1's contiguous padding.
pack_routed_experts = partial(
    _pack_routed_experts, contiguous=True, contiguous_padding=True
)
pack_r3_replay_mask = partial(
    _pack_r3_replay_mask, contiguous=True, contiguous_padding=True
)


router_replay_roots = partial(_protocol_utils.router_replay_roots, contiguous=True)


def _cp_targets(batch, cp_context):
    """Shift full documents once, then optionally select this CP rank's tokens."""
    mask = (
        torch.ones_like(batch.labels, dtype=torch.float32)
        if batch.loss_mask is None
        else batch.loss_mask
    )
    labels, mask = (
        roll_packed_thd_left(value, cu_seqlens_padded=batch.cu_seqlens)[0]
        for value in (batch.labels, mask)
    )
    denominator = mask.sum().clamp_min(1)
    if cp_context is None:
        return labels, mask, denominator
    return (
        cp_context.slice(labels, seq_dim=0),
        cp_context.slice(mask, seq_dim=0),
        denominator,
    )


def text_output(hidden, weight, batch, *, cp_context=None, tp_group=None, logits=None):
    context = get_loss_context()
    temperature = 1.0 if context is None else context.temperature
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if logits is not None:
        from megatron.lite.primitive.modules import deployment_math

        # Under CP each rank holds the logits of its own rows; labels and the
        # loss mask are sliced to the same rows by _cp_targets.
        if tp_group is not None and torch.distributed.get_world_size(tp_group) != 1:
            raise ValueError('Deployment head probabilities require full-vocab TP1')
        logits = logits.float() / temperature
    if batch.labels is None:
        result = {
            "logits": (
                logits
                if logits is not None
                else torch.nn.functional.linear(hidden, weight) / temperature
            )
        }
        if context is not None and context.calculate_entropy:
            if logits is None:
                labels = torch.zeros(
                    hidden.shape[:-1], dtype=torch.long, device=hidden.device
                )
                _, result["entropy"] = linear_cross_entropy(
                    hidden, weight, labels, temperature, tp_group
                )
            else:
                distribution = deployment_math.log_softmax(logits)
                result["entropy"] = -(distribution.exp() * distribution).sum(-1)
        return result
    if batch.labels.shape != batch.input_ids.shape:
        raise ValueError("Labels must match packed input shape")
    if batch.loss_mask is not None and batch.loss_mask.shape != batch.labels.shape:
        raise ValueError("Loss mask must match packed input shape")
    # Under CP a rank's native loss needs the prepared global denominator. An
    # external loss_fn owns normalization, so the native loss is skipped there.
    native_loss = (
        cp_context is None
        or context is not None
        and context.normalization_denominator is not None
    )
    if not native_loss and (context is None or not context.external_normalization):
        raise ValueError('V4.1_CP_NORMALIZATION_REQUIRED: use prepare_microbatches')
    labels, mask, denominator = _cp_targets(batch, cp_context)
    if logits is None:
        log_probs, entropy = linear_cross_entropy(
            hidden, weight, labels, temperature, tp_group
        )
    else:
        distribution = deployment_math.log_softmax(logits)
        log_probs = distribution.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        entropy = (
            -(distribution.exp() * distribution).sum(-1)
            if context is not None and context.calculate_entropy
            else None
        )
    result = {}
    if native_loss:
        if context is not None and context.normalization_denominator is not None:
            denominator = context.normalization_denominator
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError("Loss denominator must be finite and positive")
        loss = -(log_probs * mask).sum() / denominator
        result["loss"] = loss * (1.0 if context is None else context.loss_scale)
    if context is None or context.return_log_probs:
        result["log_probs"] = log_probs
    if context is not None and context.calculate_entropy:
        result["entropy"] = entropy
    return result


@contextmanager
def _preserve_replay(instances):
    saved = [
        (r, r.router_replay_action, r.target_topk_idx, r.target_replay_mask)
        for r in instances
    ]
    try:
        yield
    finally:
        for replay, action, target, mask in saved:
            replay.router_replay_action = action
            replay.target_topk_idx, replay.target_replay_mask = target, mask


class PackedRouterReplay:
    """Bind one packed invocation to its routes before visiting logical samples.

    This adapter does not execute a model or choose a CP layout. Its offsets
    refer to the router-local token buffer, after protocol packing/CP/TP slicing.
    A recomputed *whole* packed invocation consumes one legacy FIFO entry per
    router, not one entry per sample. Checkpoints inside a sample should use
    ``router_replay_checkpoint_contexts`` to bind their own immutable targets.
    """

    def __init__(self, total_tokens):
        self.total_tokens = total_tokens
        self.entries = []
        self.next_begin = 0
        for replay in RouterReplay.global_router_replay_instances:
            action = replay.router_replay_action
            target, mask = replay.target_topk_idx, replay.target_replay_mask
            if action == RouterReplayAction.REPLAY_BACKWARD:
                if not replay.replay_backward_list:
                    raise RuntimeError('Packed replay backward target queue is empty')
                target = replay.replay_backward_list.pop(0)
                mask = replay.replay_backward_mask_list.pop(0)
            if action in (
                RouterReplayAction.REPLAY_FORWARD,
                RouterReplayAction.REPLAY_BACKWARD,
            ):
                if target is None or target.shape[0] != total_tokens:
                    raise ValueError(
                        'Packed replay target must match the local token buffer'
                    )
                if mask is not None and mask.numel() != total_tokens:
                    raise ValueError(
                        'Packed replay mask must match the local token buffer'
                    )
            self.entries.append((replay, action, target, mask, []))

    @contextmanager
    def sequence(self, begin, end):
        if begin != self.next_begin or not begin < end <= self.total_tokens:
            raise ValueError(
                'Packed replay ranges must partition the token buffer in order'
            )
        with _preserve_replay(entry[0] for entry in self.entries):
            for replay, action, target, mask, records in self.entries:
                if action == RouterReplayAction.RECORD:
                    replay.recorded_topk_idx = None
                elif action in (
                    RouterReplayAction.REPLAY_FORWARD,
                    RouterReplayAction.REPLAY_BACKWARD,
                ):
                    replay.router_replay_action = RouterReplayAction.REPLAY_FORWARD
                    replay.target_topk_idx = target[begin:end]
                    replay.target_replay_mask = (
                        None if mask is None else mask[begin:end]
                    )
            yield
            for replay, action, target, mask, records in self.entries:
                if action == RouterReplayAction.RECORD:
                    result = replay.recorded_topk_idx
                    if result is None or result.shape[0] != end - begin:
                        raise RuntimeError(
                            'Packed record did not visit every local router/token'
                        )
                    records.append(result.detach().clone())
            self.next_begin = end

    def finish(self):
        if self.next_begin != self.total_tokens:
            raise RuntimeError('Packed replay token partition is incomplete')
        for replay, action, target, mask, records in self.entries:
            if action == RouterReplayAction.RECORD:
                replay.recorded_topk_idx = torch.cat(records, dim=0)


def packed_paired_forward(
    sequence_forward,
    hidden,
    pre_mix,
    cu_seqlens,
    *,
    input_ids=None,
    image_mask=None,
    cp_context=None,
):
    """Run a pure sequence callable once over the packed slice.

    The callable returns (hidden, next_pre_mix) and receives the slice's
    PackedCPGeometry; attention and Engram hashes mark document boundaries
    from it, token-wise work runs over every row at once. Statistics stay
    outside forward. ``hidden`` holds this rank's real rows in packing order.
    """
    from megatron.lite.primitive.modules.attention.packed_geometry import (
        PackedCPGeometry,
    )
    from megatron.lite.primitive.utils.packed_seq import packed_sequence_ranges

    if hidden.ndim != 4 or hidden.shape[0] != 1 or pre_mix.shape != hidden.shape[:-1]:
        raise ValueError("Expected packed hidden [1,T,HC,D] and pre_mix [1,T,HC]")
    for tensor in (input_ids, image_mask):
        if tensor is not None and tensor.shape != hidden.shape[:2]:
            raise ValueError("Token inputs must match packed [1,T] dimensions")
    total = hidden.shape[1] if cp_context is None else cp_context.total_length
    ranges = list(packed_sequence_ranges(cu_seqlens, total))
    geometry = PackedCPGeometry(
        (0, *(end for _, end in ranges)),
        0 if cp_context is None else cp_context.rank,
        1 if cp_context is None else cp_context.size,
        None if cp_context is None else cp_context.group,
        hidden.device,
    )
    rows = sum(stop - start for start, stop in map(geometry.owned, range(len(ranges))))
    if rows != hidden.shape[1]:
        raise ValueError("Packed hidden must hold this rank's real rows")
    replay = PackedRouterReplay(hidden.shape[1]) if cp_context is None else None
    with replay.sequence(0, rows) if replay is not None else nullcontext():
        output = sequence_forward(
            hidden,
            pre_mix,
            input_ids=input_ids,
            image_mask=image_mask,
            geometry=geometry,
        )
    if replay is not None:
        replay.finish()
    return output


def packed_pipeline_forward(
    sequence_forward, hidden, pre_mix, cu_seqlens, *, previous_post=None, **kwargs
):
    """Carry the slice's deferred mHC operands across a PP boundary.

    packed_paired_forward runs the sequence once over the whole packed slice,
    so the incoming operands hold exactly this rank's rows and the outgoing
    ones are returned for the next stage unchanged in layout.
    """
    outgoing = []

    def slice_forward(h, pre, **slice_kwargs):
        h, pre, post = sequence_forward(
            h, pre, previous_post=previous_post, return_pending=True, **slice_kwargs
        )
        outgoing.append(post)
        return h, pre

    h, pre = packed_paired_forward(slice_forward, hidden, pre_mix, cu_seqlens, **kwargs)
    (post,) = outgoing
    return h, pre, post


def _validate_replay(model, batch):
    if batch.routed_experts is not None:
        routers = [
            module
            for root in router_replay_roots(model)
            for module in root.modules()
            if hasattr(module, 'router_replay')
        ]
        if not routers or any(
            module.router_replay is None
            or module.router_replay.router_replay_action
            != RouterReplayAction.REPLAY_FORWARD
            for module in routers
        ):
            raise RuntimeError('V4.1 routed inputs require an active replay driver')


def _validate_text_batch(batch, *, multimodal=False):
    if set(batch.extras) - ({'images', 'token_types'} if multimodal else set()):
        raise NotImplementedError(
            'Text-only protocol does not accept extra modality fields'
        )
    if batch.input_ids.ndim != 1 or batch.total_tokens != batch.input_ids.numel():
        raise ValueError('Expected packed 1-D tokens matching seq_lens')
    if batch.position_ids is not None and not torch.equal(
        batch.position_ids,
        torch.cat(
            [
                torch.arange(int(n), device=batch.input_ids.device)
                for n in batch.seq_lens
            ]
        ),
    ):
        raise ValueError('Only sequence-local positions are supported')


def _forward_step(model, batch, *, optimizer=None, execution_model=None):
    schedule = model.vision_schedule
    if schedule is not None and schedule.stage != 'idle':
        raise RuntimeError('Previous microbatch requires completed vision backward')
    try:
        return _forward_step_impl(
            model, batch, optimizer=optimizer, execution_model=execution_model
        )
    except Exception:
        if schedule is not None:
            schedule.abort()
        raise


def _forward_step_impl(model, batch, *, optimizer=None, execution_model=None):
    _validate_text_batch(batch, multimodal=model.ps.pp_size <= 1)
    _validate_replay(model, batch)
    precision = (
        torch.autocast(device_type=batch.input_ids.device.type, enabled=False)
        if hasattr(model, 'residual_dtype')
        else nullcontext()
    )
    modality = dict(batch.extras)
    cp_context = None
    ids = batch.input_ids[None]
    if model.ps.cp_size > 1:
        import megatron.lite.primitive.modules.attention.cp as _imports_cp

        ContiguousCPSequence = _imports_cp.ContiguousCPSequence

        if (
            modality
            or batch.routed_experts is not None
            or batch.r3_replay_mask is not None
        ):
            raise NotImplementedError(
                'CP text training does not yet accept modality or replay inputs'
            )
        cp_context = ContiguousCPSequence(
            batch.total_tokens, model.ps.cp_rank, model.ps.cp_size, model.ps.cp_group
        )
        ids = cp_context.slice(ids)
    if 'token_types' in modality:
        if modality['token_types'].shape != batch.input_ids.shape:
            raise ValueError('Packed token types must match the input IDs')
        modality['token_types'] = modality['token_types'][None]
    with precision:
        output = (model if execution_model is None else execution_model)(
            ids,
            cu_seqlens=batch.cu_seqlens,
            cp_context=cp_context,
            return_head_hidden=not model.deployment_math,
            **modality,
        )
    result = (
        {'hidden_states': output['hidden_states']}
        if 'hidden_states' in output
        else text_output(
            None if model.deployment_math else output['head_hidden'][0],
            None if model.deployment_math else output['head_weight'],
            batch,
            cp_context=cp_context,
            tp_group=model.ps.tp_group,
            logits=output['logits'][0] if model.deployment_math else None,
        )
    )
    if optimizer is not None and model.training and torch.is_grad_enabled():
        if 'modality_loads' not in output:
            raise NotImplementedError(
                'V4.1_PP_OPTIMIZER_UNSUPPORTED: stage modality loads are unavailable'
            )
        optimizer.accumulate_modality_loads(output['modality_loads'])
    if model.vision_schedule is not None and model.vision_schedule.stage != 'idle':
        result['backward'] = model.vision_schedule.backward
    return result


unpack_forward_output = partial(
    _protocol_utils.unpack_thd_forward_output, contiguous=True, unpadded=True
)
