# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text protocol and PP-only local construction; mixed parallel routing is separate."""

import math
from contextlib import nullcontext
from dataclasses import dataclass, field, replace

import torch
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.protocol_utils import (
    pack_r3_replay_mask as _pack_r3_replay_mask,
)
from megatron.lite.model.protocol_utils import (
    pack_routed_experts as _pack_routed_experts,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.ckpt.hf_weights import allgather_concat
from megatron.lite.primitive.parallel.state import ParallelState, init_parallel
from megatron.lite.primitive.parallel.thd import roll_packed_thd_left
from megatron.lite.runtime.contracts import ParallelConfig
from torch.nn import functional as F

from .checkpoint import export_model, load_model, save_model
from .optimizer_groups import OptimizerConfig, V41Optimizer
from .training import VisionSchedule, VisionTrainability


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
    text_only: bool = True
    vision_trainability: VisionTrainability | None = None
    external_vision_device: str | None = None
    gate_temperature: float = 1.0
    bias_rate: float = 0.001
    enable_dspark_execution: bool = False


def build_model_config(source, **overrides):
    if overrides:
        raise ValueError('Apply overrides to the explicit nested source config')
    return (
        DeepseekV41Config(source)
        if isinstance(source, dict)
        else DeepseekV41Config.from_hf(source)
    )


def build_model(model_cfg, *, impl_cfg):
    from .model import DeepseekV41Model

    p = impl_cfg.parallel
    if (
        any(getattr(p, key) != 1 for key in ('tp', 'ep', 'cp', 'vpp'))
        or p.etp not in (None, 1)
        or p.pp_layout is not None
    ):
        raise NotImplementedError(
            'V4.1 stage construction supports PP only; TP/EP/CP/VPP remain pending'
        )
    ps = ParallelState()
    layer_range = None
    if p.pp > 1:
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() != p.pp
        ):
            raise ValueError(
                'PP stage construction requires exactly pp initialized ranks'
            )
        count = model_cfg.to_hf_dict()['text_config']['num_hidden_layers']
        if count % p.pp:
            raise ValueError('Equal PP stages must divide the real layer count')
        ps = init_parallel(p)
        width = count // p.pp
        layer_range = (ps.pp_rank * width, (ps.pp_rank + 1) * width)
    elif torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise NotImplementedError(
            'Data parallel construction requires the distributed integration'
        )
    if p.pp > 1 and impl_cfg.optimizer is not None:
        raise NotImplementedError(
            'Pipeline optimizer construction requires distributed routing integration'
        )
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
    with torch.device(impl_cfg.device):
        model = DeepseekV41Model(
            model_cfg,
            token_map=impl_cfg.token_map,
            quantized=impl_cfg.quantized,
            trainable_engram=impl_cfg.trainable_engram,
            gate_temperature=impl_cfg.gate_temperature,
            bias_rate=impl_cfg.bias_rate,
            enable_dspark_execution=impl_cfg.enable_dspark_execution,
            layer_range=layer_range,
        )
    from .attention import Linear
    from .engram import EngramTable

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
    # Cast residual projections first, preserving initialization rounding; native
    # FP32 compressors/norms and Engram byte storage keep their own contracts.
    for module in model.modules():
        if isinstance(module, (Linear, torch.nn.Embedding)):
            if not isinstance(module, Linear) or module.weight.dtype != torch.float32:
                module.to(dtype=impl_cfg.dtype)
            if optimizing and isinstance(module, Linear):
                module.native_fp32 = True
                module.weight.data = module.weight.data.float()
        if isinstance(module, EngramTable):
            module.output_dtype = impl_cfg.dtype
    if model.engram_hash is not None:
        model.engram_hash.to(device=impl_cfg.device)
    optimizer = None
    if optimizing:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
                parameter.main_grad = None
        model.residual_dtype = impl_cfg.dtype
        optimizer = V41Optimizer(model, impl_cfg.optimizer_config)
    if impl_cfg.external_vision_device is not None:
        if impl_cfg.vision_trainability is None:
            raise ValueError('External vision requires an explicit post-training mask')
        model.vision_schedule = VisionSchedule(model, impl_cfg.external_vision_device)
    return ModelBundle(
        [model],
        ps,
        optimizer=optimizer,
        forward_step=_forward_step,
        extras={
            'model_cfg': model_cfg,
            'vision_schedule': model.vision_schedule,
            'prepare_microbatches': prepare_microbatches,
            'optimizer_backend': 'none' if optimizer is None else 'v41',
            'parameter_bindings': model.parameter_bindings,
        },
    )


def prepare_microbatches(data_iter, count):
    """Use one valid-token denominator for all SFT microbatches (O16)."""
    from megatron.lite.runtime.contracts.loss import LossContext, split_loss_context

    if count < 1:
        raise ValueError('Microbatch count must be positive')
    items = [split_loss_context(next(data_iter)) for _ in range(count)]
    total = 0.0
    for batch, _ in items:
        if batch.labels is None:
            raise ValueError('SFT normalization requires labels')
        mask = (
            torch.ones_like(batch.labels, dtype=torch.float32)
            if batch.loss_mask is None
            else batch.loss_mask
        )
        if (
            mask.shape != batch.input_ids.shape
            or not torch.isfinite(mask).all()
            or (mask < 0).any()
        ):
            raise ValueError('Expected finite nonnegative token loss weights')
        offset = 0
        for length in batch.seq_lens.tolist():
            total += float(mask[offset + 1 : offset + length].sum())
            offset += length
    # The generic runtime divides every microbatch by count after this loss.
    denominator = max(total, 1.0) / count
    return [
        (
            batch,
            replace(context or LossContext(), normalization_denominator=denominator),
        )
        for batch, context in items
    ]


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


def _forward_step(model, batch):
    schedule = model.vision_schedule
    if schedule is not None and schedule.stage != 'idle':
        raise RuntimeError('Previous microbatch requires completed vision backward')
    try:
        return _forward_step_impl(model, batch)
    except Exception:
        if schedule is not None:
            schedule.abort()
        raise


def _forward_step_impl(model, batch):
    _validate_text_batch(batch, multimodal=True)
    _validate_replay(model, batch)
    precision = (
        torch.autocast(device_type=batch.input_ids.device.type, enabled=False)
        if hasattr(model, 'residual_dtype')
        else nullcontext()
    )
    modality = dict(batch.extras)
    if 'token_types' in modality:
        if modality['token_types'].shape != batch.input_ids.shape:
            raise ValueError('Packed token types must match the input IDs')
        modality['token_types'] = modality['token_types'][None]
    with precision:
        logits = model(batch.input_ids[None], cu_seqlens=batch.cu_seqlens, **modality)[
            'logits'
        ][0]
    result = _text_output(logits, batch)
    if model.vision_schedule is not None and model.vision_schedule.stage != 'idle':
        result['backward'] = model.vision_schedule.backward
    return result


def pipeline_forward_step(model, batch, *, start, end, payload=None, owners=(-1, -1)):
    """Model/protocol range boundary; a scheduler owns transport and generation.

    Packed multi-sequence/CP partitioning is a separate integration. Reject it
    explicitly until the scheduler can carry separate CSA2 state per sample.
    """
    _validate_text_batch(batch)
    if batch.routed_experts is not None or batch.r3_replay_mask is not None:
        raise NotImplementedError(
            'Pipeline routing replay requires scheduler integration'
        )
    if batch.seq_lens.numel() != 1:
        raise NotImplementedError(
            'Pipeline packed sequences require per-sample state routing'
        )
    payload, owners = model.forward_pipeline_range(
        batch.input_ids[None], start=start, end=end, payload=payload, owners=owners
    )
    result = {'pipeline_payload': payload, 'pipeline_owners': owners}
    if end == len(model.layers):
        result.update(_text_output(model.finish_pipeline(payload)[0], batch))
    return result


def packed_pipeline_forward_step(model, batch, *, start, end, state=None):
    """CP1 packed range with an independent carrier per unpadded sequence.

    State is an ordered tuple of (PairedPayload, owners) pairs. Transport must
    preserve this sequence order and all native compressed-state shapes. This
    entry does not partition CP or replace the distributed PP scheduler.
    """
    _validate_text_batch(batch)
    if batch.routed_experts is not None or batch.r3_replay_mask is not None:
        raise NotImplementedError(
            'Pipeline routing replay requires scheduler integration'
        )
    lengths = batch.seq_lens.tolist()
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError('Packed pipeline requires positive sequence lengths')
    if start == 0:
        if state is not None:
            raise ValueError('First packed range requires fresh sequence states')
        state = tuple((None, (-1, -1)) for _ in lengths)
    elif state is None or len(state) != len(lengths):
        raise ValueError('Packed pipeline requires one state per sequence')
    outputs = []
    offset = 0
    for length, (payload, owners) in zip(lengths, state):
        payload, owners = model.forward_pipeline_range(
            batch.input_ids[offset : offset + length][None],
            start=start,
            end=end,
            payload=payload,
            owners=owners,
        )
        outputs.append((payload, owners))
        offset += length
    result = {'packed_pipeline_state': tuple(outputs)}
    if end == len(model.layers):
        from .pipeline import PairedPayload

        final = PairedPayload(
            h=torch.cat([payload.h for payload, _ in outputs], dim=1),
            p=torch.cat([payload.p for payload, _ in outputs], dim=1),
        )
        result.update(_text_output(model.finish_pipeline(final)[0], batch))
    return result


def _text_output(logits, batch):
    from megatron.lite.runtime.contracts.loss import get_loss_context

    context = get_loss_context()
    temperature = 1.0 if context is None else context.temperature
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    logits = logits / temperature
    result = {'logits': logits}
    if batch.labels is not None:
        if batch.labels.shape != batch.input_ids.shape:
            raise ValueError('Labels must match packed input shape')
        labels = batch.labels.clone()
        mask = (
            torch.ones_like(labels, dtype=torch.float32)
            if batch.loss_mask is None
            else batch.loss_mask.clone()
        )
        if mask.shape != labels.shape:
            raise ValueError('Loss mask must match packed input shape')
        labels, mask = (
            roll_packed_thd_left(value, cu_seqlens_padded=batch.cu_seqlens)[0]
            for value in (labels, mask)
        )
        token_loss = F.cross_entropy(logits, labels, reduction='none')
        denominator = mask.sum().clamp_min(1)
        if context is not None and context.normalization_denominator is not None:
            denominator = context.normalization_denominator
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError('Loss denominator must be finite and positive')
        result['loss'] = (token_loss * mask).sum() / denominator
        if context is not None:
            result['loss'] = result['loss'] * context.loss_scale
        if context is None or context.return_log_probs:
            result['log_probs'] = -token_loss
    if context is not None and context.calculate_entropy:
        log_probs = logits.log_softmax(-1)
        result['entropy'] = -(log_probs.exp() * log_probs).sum(-1)
    return result


def unpack_forward_output(model, batch, output):
    if isinstance(output, dict):
        return {
            key: unpack_forward_output(model, batch, value)
            for key, value in output.items()
        }
    if (
        isinstance(output, torch.Tensor)
        and output.ndim > 0
        and output.shape[0] == batch.total_tokens
    ):
        return torch.nested.as_nested_tensor(
            list(output.split(batch.seq_lens.tolist()))
        )
    return output


def load_hf_weights(chunk, hf_path, model_cfg, ps):
    if hf_path:
        load_model(chunk, hf_path)


def _single(chunks):
    if len(chunks) != 1:
        raise NotImplementedError('Single-rank V4.1 export requires one chunk')
    return chunks[0]


def export_hf_weights(chunks, model_cfg, ps, **kwargs):
    if kwargs:
        raise ValueError('Unsupported export options')
    model = _single(chunks)
    if model.archival_store is None or set(model.archival_store.entries) != set(
        model.archival_bindings
    ):
        raise ValueError('Complete archival storage is required for export')
    yield from export_model(model)
    from .checkpoint import _tensor

    for key in model.archival_store.entries:
        yield key, _tensor(model.archival_store, key)


def save_hf_weights(chunks, path, model_cfg, ps, **kwargs):
    if kwargs:
        raise ValueError('Unsupported export options')
    save_model(_single(chunks), path)


def vocab_size(model_cfg):
    return model_cfg.to_hf_dict()['text_config']['vocab_size']


def pack_routed_experts(model, batch, routed_experts):
    """Use shared THD padding followed by contiguous CP and then TP slicing."""
    return _pack_routed_experts(
        model, batch, routed_experts, contiguous=True, contiguous_padding=True
    )


def pack_r3_replay_mask(model, batch):
    """Keep the causal replay mask in exactly the same token layout as routes."""
    return _pack_r3_replay_mask(model, batch, contiguous=True, contiguous_padding=True)


def router_replay_roots(chunk):
    """E stages retain global layer slots; absent/nonlocal slots contain None."""
    while hasattr(chunk, 'module'):
        chunk = chunk.module
    layers = chunk.layers
    start, end = getattr(chunk, 'local_layer_range', (0, len(layers)))
    if not 0 <= start < end <= len(layers):
        raise ValueError('Invalid V4.1 replay layer interval')
    if any((layer is not None) != (start <= i < end) for i, layer in enumerate(layers)):
        raise ValueError(
            'V4.1 replay requires contiguous stage-owned global layer slots'
        )
    return list(layers[start:end])


def validate_router_replay(chunks, action):
    """Fail before collectives for scheduler interfaces not implemented by E yet."""
    from megatron.lite.primitive.parallel.thd import parallel_state_from_model

    if len(chunks) != 1:
        raise NotImplementedError(
            'V4.1 replay requires one local PP chunk; VPP is not wired'
        )
    ps = parallel_state_from_model(chunks[0]) or ParallelState()
    router_replay_roots(chunks[0])
    if action == 'record' and ps.pp_size > 1:
        raise NotImplementedError(
            'V4.1 PP record requires E post-drain route collection; '
            'a collective inside stage forward would deadlock'
        )


def unpack_recorded_routed_experts(model, batch, recorded, *, pipeline_drained=False):
    """Invert local route packing. PP record needs E's post-drain scheduler hook."""
    import torch.distributed as dist
    from megatron.lite.primitive.parallel.thd import (
        parallel_state_from_model,
        thd_pack_meta,
    )

    ps = parallel_state_from_model(model) or ParallelState()
    if ps.pp_size > 1 and not pipeline_drained:
        validate_router_replay([model], 'record')
    if not recorded or any(row is None for row in recorded):
        raise RuntimeError('V4.1 record did not visit every local router')
    full = torch.stack(recorded, dim=1)
    for size, group in ((ps.tp_size, ps.tp_group), (ps.cp_size, ps.cp_group)):
        if size > 1:
            if group is None:
                raise RuntimeError(
                    'V4.1 route gather requires the corresponding parallel group'
                )
            full = allgather_concat(full, size, group, dim=0)
    if ps.pp_size > 1:
        if ps.pp_group is None:
            raise RuntimeError(
                'V4.1 PP route gather requires pp_group after pipeline drain'
            )
        widths = allgather_concat(
            torch.tensor([full.shape[1]], dtype=torch.long, device=full.device),
            ps.pp_size,
            ps.pp_group,
            dim=0,
        ).tolist()
        current = model
        while hasattr(current, 'module'):
            current = current.module
        expected_range = (sum(widths[: ps.pp_rank]), sum(widths[: ps.pp_rank + 1]))
        valid = torch.tensor(
            [getattr(current, 'local_layer_range', None) == expected_range],
            dtype=torch.int32,
            device=full.device,
        )
        dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=ps.pp_group)
        if not valid.item():
            raise ValueError(
                'V4.1 PP router counts disagree with global stage layer order'
            )
        padded = full.new_zeros(full.shape[0], max(widths), full.shape[2])
        padded[:, : full.shape[1]] = full
        parts = [torch.empty_like(padded) for _ in widths]
        dist.all_gather(parts, padded, group=ps.pp_group)
        full = torch.cat([part[:, :width] for part, width in zip(parts, widths)], dim=1)
    meta = thd_pack_meta(
        batch.seq_lens, tp_size=ps.tp_size, cp_size=ps.cp_size, contiguous=True
    )
    if full.shape[0] != int(meta.cu_seqlens_padded[-1]):
        raise ValueError('V4.1 recorded rows differ from the shared THD token layout')
    rows = [
        full[int(start) : int(start) + int(length)]
        for start, length in zip(meta.cu_seqlens_padded[:-1], meta.lengths)
    ]
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)
