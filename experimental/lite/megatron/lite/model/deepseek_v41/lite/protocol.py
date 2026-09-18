# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text protocol with replicated data parallel training and explicit optimizer policy."""

import math
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from functools import partial

import torch
from megatron.lite.model import protocol_utils as _protocol_utils
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.protocol_utils import (
    pack_r3_replay_mask as _pack_r3_replay_mask,
)
from megatron.lite.model.protocol_utils import (
    pack_routed_experts as _pack_routed_experts,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.config_fields import project_fields
from megatron.lite.primitive.modules.router_replay import (
    RouterReplay,
    RouterReplayAction,
)
from megatron.lite.primitive.modules.vision_training import (
    VisionSchedule,
    VisionTrainability,
)
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.parallel.owned_ddp import wrap_owned_ddp
from megatron.lite.primitive.parallel.state import ParallelState, init_parallel
from megatron.lite.primitive.parallel.thd import roll_packed_thd_left
from megatron.lite.primitive.train_step import prepare_microbatches
from megatron.lite.runtime.contracts import ParallelConfig
from megatron.lite.runtime.contracts.loss import get_loss_context

from .checkpoint import export_hf_weights as _export_hf_weights_impl
from .checkpoint import load_model, save_model
from .optimizer_groups import OptimizerConfig, V41Optimizer

# HF checkpoints store trainable masters and byte-preserved archives.


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = None
    optimizer_config: OptimizerConfig | None = None
    device: str = 'cuda'
    dtype: torch.dtype = torch.bfloat16
    quantized: bool = True
    token_map: list[int] | None = None
    trainable_engram: bool = False
    shard_engram: bool = True
    text_only: bool = True
    vision_trainability: VisionTrainability | None = None
    external_vision_device: str | None = None
    gate_temperature: float = 1.0
    bias_rate: float = 0.001
    enable_dspark_execution: bool = False
    pipeline_split_layer: int = 20

    def __post_init__(self):
        # Runtime/VERL configurations arrive as serialized mappings.
        if isinstance(self.optimizer_config, Mapping):
            object.__setattr__(
                self, "optimizer_config", OptimizerConfig(**self.optimizer_config)
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


UNSUPPORTED = (
    (
        lambda c, p: p.pp > 1
        and (not c.text_only or c.external_vision_device is not None),
        NotImplementedError,
        'V4.1_PP_TEXT_ONLY: PP currently supports text-only training; use PP=1 for multimodal training',
    ),
    (
        lambda c, p: p.pp > 1 and c.pipeline_split_layer != 20,
        NotImplementedError,
        'V4.1_PP_CSA2_PAYLOAD_UNSUPPORTED: only split layer 20 is supported; other cuts require transporting CSA2 owner state',
    ),
    (
        lambda c, p: p.pp > 1 and (p.ep != 1 or p.cp != 1),
        NotImplementedError,
        'V4.1_PP_COMBINATION_UNSUPPORTED: PP2 requires EP=CP=1',
    ),
    (
        lambda c, p: p.pp > 1 and c.optimizer is not None,
        NotImplementedError,
        'V4.1_PP_OPTIMIZER_UNSUPPORTED: PP2 currently supports model forward/backward; distributed optimizer training is not validated',
    ),
    (
        lambda c, p: p.pp > 1
        and (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() != 2
        ),
        ValueError,
        'V4.1_PP_WORLD: PP2 requires an initialized two-rank world',
    ),
    (
        lambda c, p: p.cp != 1 and p.ep != 1,
        NotImplementedError,
        'CP_AND_EP_NOT_SIMULTANEOUSLY_SUPPORTED: V4.1 requires EP=1 with CP>1; use CP-only or EP with CP=1',
    ),
    (
        lambda c, p: type(p.ep) is not int or p.ep < 1,
        ValueError,
        'EP size must be a positive integer',
    ),
    (
        lambda c, p: type(p.cp) is not int or p.cp < 1,
        ValueError,
        'CP size must be a positive integer',
    ),
    (
        lambda c, p: p.cp > 1
        and (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() != p.cp
        ),
        ValueError,
        'CP requires an initialized CP-only world',
    ),
    (
        lambda c, p: p.ep > 1
        and (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() < p.ep
        ),
        ValueError,
        'EP requires an initialized distributed world of at least ep ranks',
    ),
)


def build_model(model_cfg, *, impl_cfg):
    c, p = impl_cfg, impl_cfg.parallel
    unsupported = [
        key
        for key, invalid in (
            ('tp', p.tp != 1),
            ('vpp', p.vpp != 1),
            ('pp', p.pp not in (1, 2)),
            ('etp', p.etp not in (None, 1)),
            ('pp_layout', p.pp_layout is not None),
        )
        if invalid
    ]
    if unsupported:
        raise NotImplementedError(
            f'V4.1_UNSUPPORTED_PARALLELISM: {", ".join(unsupported)}; '
            'supported: DP, EP with CP=1, contiguous CP-only, or text-only PP2; '
            'TP/VPP/ETP, PP other than 1 or 2, and custom pipeline layouts are unsupported'
        )
    for invalid, error, message in UNSUPPORTED:
        if invalid(c, p):
            raise error(message)
    from .model import DeepseekV41Model

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
    layer_range = None
    if p.pp > 1:
        from megatron.lite.primitive.parallel.pp import build_pipeline_chunk_layout

        cut = impl_cfg.pipeline_split_layer
        count = model_cfg.to_hf_dict()['text_config']['num_hidden_layers']
        layout = build_pipeline_chunk_layout(
            count, replace(ps, pp_layout=f'Et*{cut}|t*{count - cut}L')
        )
        layer_range = (layout.layer_indices[0], layout.layer_indices[-1] + 1)
    with torch.device(impl_cfg.device):
        model = DeepseekV41Model(
            model_cfg,
            parallel_state=ps,
            layer_range=layer_range,
            **project_fields(
                vars(c),
                'token_map quantized trainable_engram shard_engram '
                'gate_temperature bias_rate enable_dspark_execution',
            ),
        )
    model.pipeline_residual_dtype = impl_cfg.dtype
    from megatron.lite.primitive.modules.engram_lookup import EngramTable
    from megatron.lite.primitive.modules.native_fp32_linear import (
        configure_residual_projections,
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
    )
    return ModelBundle(
        [model],
        ps,
        optimizer=optimizer,
        finalize_grads=(
            optimizer.finalize_grads
            if optimizing and (ps.ep_size > 1 or model.engram_group is not None)
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
        },
    )


def load_hf_weights(chunk, hf_path, model_cfg, ps):
    if hf_path:
        load_model(chunk, hf_path)


export_hf_weights = _export_hf_weights_impl


def save_hf_weights(
    chunks, path, model_cfg, ps, *, target=None, resync_config=None, **kwargs
):
    if target is not None or resync_config is not None:
        raise NotImplementedError(
            "V4.1_HF_SAVE_RESYNC_UNSUPPORTED: target/resync_config require "
            "a resync exporter; this entry point only saves archival HF weights"
        )
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
unpack_recorded_routed_experts = _protocol_utils.unpack_recorded_routed_experts


def validate_router_replay(chunks, action):
    if len(chunks) != 1:
        raise NotImplementedError("Replay requires one local PP chunk")
    router_replay_roots(chunks[0])


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


def text_output(hidden, weight, batch, *, cp_context=None, tp_group=None):
    context = get_loss_context()
    temperature = 1.0 if context is None else context.temperature
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if batch.labels is None:
        result = {"logits": torch.nn.functional.linear(hidden, weight) / temperature}
        if context is not None and context.calculate_entropy:
            labels = torch.zeros(
                hidden.shape[:-1], dtype=torch.long, device=hidden.device
            )
            _, result["entropy"] = linear_cross_entropy(
                hidden, weight, labels, temperature, tp_group
            )
        return result
    if batch.labels.shape != batch.input_ids.shape:
        raise ValueError("Labels must match packed input shape")
    if batch.loss_mask is not None and batch.loss_mask.shape != batch.labels.shape:
        raise ValueError("Loss mask must match packed input shape")
    labels, mask, denominator = _cp_targets(batch, cp_context)
    log_probs, entropy = linear_cross_entropy(
        hidden, weight, labels, temperature, tp_group
    )
    if context is not None and context.normalization_denominator is not None:
        denominator = context.normalization_denominator
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("Loss denominator must be finite and positive")
    loss = -(log_probs * mask).sum() / denominator
    if cp_context is not None and (
        context is None or context.normalization_denominator is None
    ):
        loss = loss * cp_context.size
    result = {"loss": loss * (1.0 if context is None else context.loss_scale)}
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
    """Run a pure sequence callable over each logical sample, preserving its graph.

    The callable returns (hidden, next_pre_mix), creates fresh sequence state
    per invocation, and keeps bias/statistic publication outside forward. RNG is
    consumed in sequence order, exactly as for independent calls. This is a
    correctness path; CP transport and document ownership belong to the primitive.
    """
    from contextlib import nullcontext

    from megatron.lite.primitive.utils.packed_seq import packed_sequence_ranges

    if hidden.ndim != 4 or hidden.shape[0] != 1 or pre_mix.shape != hidden.shape[:-1]:
        raise ValueError("Expected packed hidden [1,T,HC,D] and pre_mix [1,T,HC]")
    for tensor in (input_ids, image_mask):
        if tensor is not None and tensor.shape != hidden.shape[:2]:
            raise ValueError("Token inputs must match packed [1,T] dimensions")
    outputs, mixes = [], []
    replay = PackedRouterReplay(hidden.shape[1]) if cp_context is None else None
    total = hidden.shape[1] if cp_context is None else cp_context.total_length
    offset = 0
    for begin, end in packed_sequence_ranges(cu_seqlens, total):
        kwargs = {}
        if cp_context is not None:
            document = cp_context.document(begin, end)
            kwargs['cp_context'] = document
            begin, end = offset, offset + document.local_length
            offset = end
        if input_ids is not None:
            kwargs['input_ids'] = input_ids[:, begin:end]
        if image_mask is not None:
            kwargs['image_mask'] = image_mask[:, begin:end]
        with replay.sequence(begin, end) if replay is not None else nullcontext():
            h, p = sequence_forward(
                hidden[:, begin:end], pre_mix[:, begin:end], **kwargs
            )
        outputs.append(h)
        mixes.append(p)
    if replay is not None:
        replay.finish()
    return torch.cat(outputs, dim=1), torch.cat(mixes, dim=1)


def _validate_replay(model, batch):
    if batch.routed_experts is not None:
        from megatron.lite.primitive.modules.router_replay import RouterReplayAction

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
    if model.ps.pp_size > 1:
        _validate_text_batch(batch)
    else:
        _validate_text_batch(batch, multimodal=True)
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
        from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

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
            return_head_hidden=True,
            **modality,
        )
    result = (
        {'hidden_states': output['hidden_states']}
        if 'hidden_states' in output
        else text_output(
            output['head_hidden'][0],
            model.head.weight.float(),
            batch,
            cp_context=cp_context,
            tp_group=model.ps.tp_group,
        )
    )
    if optimizer is not None and model.training and torch.is_grad_enabled():
        optimizer.accumulate_modality_loads(output['modality_loads'])
    if model.vision_schedule is not None and model.vision_schedule.stage != 'idle':
        result['backward'] = model.vision_schedule.backward
    return result


unpack_forward_output = partial(
    _protocol_utils.unpack_thd_forward_output, contiguous=True, unpadded=True
)
