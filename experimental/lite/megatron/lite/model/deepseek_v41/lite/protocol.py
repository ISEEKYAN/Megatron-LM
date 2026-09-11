# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text protocol and PP-only local construction; mixed parallel routing is separate."""

from dataclasses import dataclass, field

import torch
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel.state import ParallelState, init_parallel
from megatron.lite.runtime.contracts import ParallelConfig
from torch.nn import functional as F

from .checkpoint import export_model, load_model, save_model


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = None
    device: str = 'cuda'
    dtype: torch.dtype = torch.bfloat16
    quantized: bool = True
    token_map: list[int] | None = None
    trainable_engram: bool = False
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
    if impl_cfg.optimizer is not None:
        raise NotImplementedError(
            'Optimizer construction requires the V4.1 object-based routing integration'
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
    # Keep HC coefficients, router and normalization tensors in their native
    # precision. Only residual projections change dtype in diagnostic mode.
    for module in model.modules():
        from .attention import Linear
        from .engram import EngramTable

        if isinstance(module, (Linear, torch.nn.Embedding)):
            if isinstance(module, Linear) and module.weight.dtype == torch.float32:
                continue  # ratio>1 compressor projections have an FP32 contract
            module.to(dtype=impl_cfg.dtype)
        if isinstance(module, EngramTable):
            module.output_dtype = impl_cfg.dtype
    if model.engram_hash is not None:
        model.engram_hash.to(device=impl_cfg.device)
    return ModelBundle(
        [model],
        ps,
        forward_step=_forward_step,
        extras={
            'model_cfg': model_cfg,
            'optimizer_backend': 'none',
            'parameter_bindings': model.parameter_bindings,
        },
    )


def _validate_text_batch(batch):
    if batch.routed_experts is not None or batch.r3_replay_mask is not None:
        raise NotImplementedError('Routing replay requires the replay integration')
    if batch.extras:
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
    _validate_text_batch(batch)
    logits = model(batch.input_ids[None], cu_seqlens=batch.cu_seqlens)['logits'][0]
    return _text_output(logits, batch)


def pipeline_forward_step(model, batch, *, start, end, payload=None, owners=(-1, -1)):
    """Model/protocol range boundary; a scheduler owns transport and generation.

    Packed multi-sequence/CP partitioning is a separate integration. Reject it
    explicitly until the scheduler can carry separate CSA2 state per sample.
    """
    _validate_text_batch(batch)
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
        offset = 0
        for length in batch.seq_lens.tolist():
            end = offset + length
            labels[offset:end] = batch.labels[offset:end].roll(-1)
            mask[offset:end] = mask[offset:end].roll(-1)
            labels[end - 1], mask[end - 1] = 0, 0
            offset = end
        token_loss = F.cross_entropy(logits, labels, reduction='none')
        result['loss'] = (token_loss * mask).sum() / mask.sum().clamp_min(1)
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
    if model.archival_store is not None:
        from .checkpoint import _tensor

        for key in model.archival_store.entries:
            yield key, _tensor(model.archival_store, key)


def save_hf_weights(chunks, path, model_cfg, ps, **kwargs):
    if kwargs:
        raise ValueError('Unsupported export options')
    save_model(_single(chunks), path)


def vocab_size(model_cfg):
    return model_cfg.to_hf_dict()['text_config']['vocab_size']
