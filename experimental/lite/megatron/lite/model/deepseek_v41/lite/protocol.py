# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text-only single-rank protocol. Distributed and optimizer routing are separate."""

from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.protocol_utils import (
    pack_r3_replay_mask as _pack_r3_replay_mask,
)
from megatron.lite.model.protocol_utils import (
    pack_routed_experts as _pack_routed_experts,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.runtime.contracts import ParallelConfig
from torch.nn import functional as F

from .checkpoint import export_model, load_model, save_model
from .optimizer_groups import OptimizerConfig, V41Optimizer


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
    optimizer = None
    if impl_cfg.optimizer == 'muon':
        if impl_cfg.device == 'meta':
            raise ValueError(
                'Materialize V4.1 parameters before constructing optimizer state'
            )
        # Persistent FP32 owners receive FP32 wgrad from the numerical providers.
        # No post-hoc BF16 gradient widening is used.
        from .attention import Linear

        for module in model.modules():
            if isinstance(module, Linear):
                module.native_fp32 = True
                module.weight.data = module.weight.data.float()
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
                p.main_grad = None
                p.register_post_accumulate_grad_hook(_publish_main_grad)
        model.residual_dtype = impl_cfg.dtype
        optimizer = V41Optimizer(model, impl_cfg.optimizer_config)
    return ModelBundle(
        [model],
        ParallelState(),
        optimizer=optimizer,
        forward_step=_forward_step,
        extras={
            'model_cfg': model_cfg,
            'optimizer_backend': 'none' if optimizer is None else 'v41',
            'parameter_bindings': model.parameter_bindings,
        },
    )


def _publish_main_grad(parameter):
    if parameter.grad.dtype != torch.float32:
        raise RuntimeError('V4.1 gradient producer did not return native FP32')
    parameter.main_grad = parameter.grad


def _forward_step(model, batch):
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
    precision = (
        torch.autocast(device_type=batch.input_ids.device.type, enabled=False)
        if hasattr(model, 'residual_dtype')
        else nullcontext()
    )
    with precision:
        logits = (
            model(batch.input_ids[None], cu_seqlens=batch.cu_seqlens)['logits'][0]
            / temperature
        )
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
            parts = [torch.empty_like(full) for _ in range(size)]
            dist.all_gather(parts, full.contiguous(), group=group)
            full = torch.cat(parts, dim=0)
    if ps.pp_size > 1:
        if ps.pp_group is None:
            raise RuntimeError(
                'V4.1 PP route gather requires pp_group after pipeline drain'
            )
        counts = [
            torch.empty(1, dtype=torch.long, device=full.device)
            for _ in range(ps.pp_size)
        ]
        dist.all_gather(
            counts,
            torch.tensor([full.shape[1]], dtype=torch.long, device=full.device),
            group=ps.pp_group,
        )
        widths = [int(count.item()) for count in counts]
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
