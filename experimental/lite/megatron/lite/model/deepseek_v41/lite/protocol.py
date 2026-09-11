# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text-only single-rank protocol. Distributed and optimizer routing are separate."""

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.runtime.contracts import ParallelConfig
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
        DeepseekV41Config(source) if isinstance(source, dict) else DeepseekV41Config.from_hf(source)
    )


def build_model(model_cfg, *, impl_cfg):
    from .model import DeepseekV41Model

    p = impl_cfg.parallel
    if (
        any(getattr(p, key) != 1 for key in ('tp', 'ep', 'cp', 'pp', 'vpp'))
        or p.etp not in (None, 1)
        or p.pp_layout is not None
    ):
        raise NotImplementedError('V4.1 assembly currently requires single-rank parallelism')
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise NotImplementedError('Distributed V4.1 construction requires the parallel integration')
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
        ParallelState(),
        forward_step=_forward_step,
        extras={
            'model_cfg': model_cfg,
            'optimizer_backend': 'none',
            'parameter_bindings': model.parameter_bindings,
        },
    )


def _forward_step(model, batch):
    if batch.routed_experts is not None or batch.r3_replay_mask is not None:
        raise NotImplementedError('Routing replay requires the replay integration')
    if batch.extras:
        raise NotImplementedError('Text-only protocol does not accept extra modality fields')
    if batch.input_ids.ndim != 1 or batch.total_tokens != batch.input_ids.numel():
        raise ValueError('Expected packed 1-D tokens matching seq_lens')
    if batch.position_ids is not None and not torch.equal(
        batch.position_ids,
        torch.cat([torch.arange(int(n), device=batch.input_ids.device) for n in batch.seq_lens]),
    ):
        raise ValueError('Only sequence-local positions are supported')
    from megatron.lite.runtime.contracts.loss import get_loss_context

    context = get_loss_context()
    temperature = 1.0 if context is None else context.temperature
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    logits = model(batch.input_ids[None], cu_seqlens=batch.cu_seqlens)['logits'][0] / temperature
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
        return {key: unpack_forward_output(model, batch, value) for key, value in output.items()}
    if (
        isinstance(output, torch.Tensor)
        and output.ndim > 0
        and output.shape[0] == batch.total_tokens
    ):
        return torch.nested.as_nested_tensor(list(output.split(batch.seq_lens.tolist())))
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
    if model.archival_store is None or set(model.archival_store.entries) != set(model.archival_bindings):
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
