# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text protocol with replicated data parallel training and explicit optimizer policy."""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from functools import partial

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
from megatron.lite.primitive.modules.vision_training import (
    VisionSchedule,
    VisionTrainability,
)
from megatron.lite.primitive.packed_lm import _cp_targets, prepare_microbatches
from megatron.lite.primitive.packed_lm import text_output as _text_output
from megatron.lite.primitive.packed_lm import unpack_forward_output
from megatron.lite.primitive.parallel.state import ParallelState, init_parallel
from megatron.lite.runtime.contracts import ParallelConfig

from .checkpoint import export_hf_weights as _export_hf_weights_impl
from .checkpoint import load_model, save_model
from .optimizer_groups import OptimizerConfig, V41Optimizer

# HF checkpoints store trainable masters and byte-preserved archives.
HF_SAVE_SUPPORTS_RESYNC = False


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


def _reject_options(error, checks):
    # Evaluate in declaration order, retaining distributed short-circuit checks.
    for message, invalid in checks.items():
        if invalid():
            raise error(message)


def build_model(model_cfg, *, impl_cfg):
    c, p = impl_cfg, impl_cfg.parallel
    unsupported = [key for key in ('tp', 'vpp') if getattr(p, key) != 1]
    if p.pp not in (1, 2):
        unsupported.append('pp')
    if p.etp not in (None, 1):
        unsupported.append('etp')
    if p.pp_layout is not None:
        unsupported.append('pp_layout')
    if unsupported:
        raise NotImplementedError(
            f'V4.1_UNSUPPORTED_PARALLELISM: {", ".join(unsupported)}; '
            'supported: DP, EP with CP=1, contiguous CP-only, or text-only PP2; '
            'TP/VPP/ETP, PP other than 1 or 2, and custom pipeline layouts are unsupported'
        )
    if p.pp > 1:
        _reject_options(
            NotImplementedError,
            {
                'V4.1_PP_TEXT_ONLY: PP currently supports text-only training; use PP=1 for multimodal training': lambda: not c.text_only
                or c.external_vision_device is not None,
                'V4.1_PP_CSA2_PAYLOAD_UNSUPPORTED: only split layer 20 is supported; other cuts require transporting CSA2 owner state': lambda: c.pipeline_split_layer
                != 20,
                'V4.1_PP_COMBINATION_UNSUPPORTED: PP2 requires EP=CP=1': lambda: p.ep
                != 1
                or p.cp != 1,
                'V4.1_PP_OPTIMIZER_UNSUPPORTED: PP2 currently supports model forward/backward; distributed optimizer training is not validated': lambda: c.optimizer
                is not None,
            },
        )
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() != 2
        ):
            raise ValueError(
                'V4.1_PP_WORLD: PP2 requires an initialized two-rank world'
            )
    from .model import DeepseekV41Model

    if p.cp != 1 and p.ep != 1:
        raise NotImplementedError(
            'CP_AND_EP_NOT_SIMULTANEOUSLY_SUPPORTED: V4.1 requires EP=1 with CP>1; '
            'use CP-only or EP with CP=1'
        )
    if type(p.ep) is not int or p.ep < 1:
        raise ValueError("EP size must be a positive integer")
    if type(p.cp) is not int or p.cp < 1:
        raise ValueError('CP size must be a positive integer')
    if p.cp > 1 and (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() != p.cp
    ):
        raise ValueError('CP requires an initialized CP-only world')
    if p.ep > 1 and (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() < p.ep
    ):
        raise ValueError(
            "EP requires an initialized distributed world of at least ep ranks"
        )
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
            token_map=impl_cfg.token_map,
            quantized=impl_cfg.quantized,
            trainable_engram=impl_cfg.trainable_engram,
            shard_engram=impl_cfg.shard_engram,
            gate_temperature=impl_cfg.gate_temperature,
            bias_rate=impl_cfg.bias_rate,
            enable_dspark_execution=impl_cfg.enable_dspark_execution,
        )
    model.pipeline_residual_dtype = impl_cfg.dtype
    from megatron.lite.primitive.modules.engram_lookup import EngramTable

    from .attention import Linear

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
    execution_model = model
    if (ps.dp_size > 1 or ps.cp_size > 1) and optimizing:
        if impl_cfg.external_vision_device is not None:
            raise NotImplementedError(
                'DP external vision requires staged gradient synchronization'
            )
        from torch.nn.parallel import DistributedDataParallel

        # DDP synchronizes parameter initialization. Encoded Engram buffers need
        # byte collectives because NCCL does not accept their FP8 storage dtype.
        gradient_group = ps.dp_cp_group if ps.cp_size > 1 else ps.dp_group
        sharded = set()
        if model.engram_group is not None:
            for block in model.layers:
                if block.engram is not None:
                    table = block.engram.embed
                    sharded.update(id(tensor) for tensor in table.buffers())
                    if table.master is not None:
                        sharded.add(id(table.master))
        model._ddp_params_and_buffers_to_ignore = [
            name
            for name, tensor in (*model.named_parameters(), *model.named_buffers())
            if id(tensor) in sharded
        ]
        with torch.no_grad():
            for buffer in model.buffers():
                if id(buffer) in sharded:
                    continue
                value = buffer.contiguous().reshape(-1).view(torch.uint8)
                torch.distributed.broadcast(value, src=0, group=gradient_group)
                buffer.copy_(value.view(buffer.dtype).reshape(buffer.shape))
        if ps.ep_size > 1:
            expert_ids = {
                id(b.tensor) for b in model.parameter_bindings() if b.role == "expert"
            }
            model._ddp_params_and_buffers_to_ignore += [
                name
                for name, parameter in model.named_parameters()
                if id(parameter) in expert_ids
            ]
            for parameter in model.parameters():
                if id(parameter) in expert_ids:
                    torch.distributed.broadcast(
                        parameter.data,
                        src=torch.distributed.get_global_rank(ps.ep_dp_group, 0),
                        group=ps.ep_dp_group,
                    )
        execution_model = DistributedDataParallel(
            model,
            process_group=gradient_group,
            broadcast_buffers=False,
            find_unused_parameters=True,
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
            'prepare_microbatches': partial(prepare_microbatches, dp_group=ps.dp_group),
            'optimizer_backend': 'none' if optimizer is None else 'v41',
            'parameter_bindings': model.parameter_bindings,
        },
    )


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
        _validate_pipeline_batch(batch)
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
            ids, cu_seqlens=batch.cu_seqlens, cp_context=cp_context, **modality
        )
    result = (
        {'hidden_states': output['hidden_states']}
        if 'hidden_states' in output
        else _text_output(output['logits'][0], batch, cp_context=cp_context)
    )
    if optimizer is not None and model.training and torch.is_grad_enabled():
        optimizer.accumulate_modality_loads(output['modality_loads'])
    if model.vision_schedule is not None and model.vision_schedule.stage != 'idle':
        result['backward'] = model.vision_schedule.backward
    return result


def pipeline_forward_step(model, batch, *, start, end, payload=None, owners=(-1, -1)):
    """Single-sequence adapter for the shared paired range protocol."""
    _validate_pipeline_batch(batch)
    if batch.seq_lens.numel() != 1:
        raise NotImplementedError(
            'Pipeline packed sequences require per-sample state routing'
        )
    states, result = _pipeline_ranges(model, batch, start, end, ((payload, owners),))
    payload, owners = states[0]
    return {'pipeline_payload': payload, 'pipeline_owners': owners, **result}


def packed_pipeline_forward_step(model, batch, *, start, end, state=None):
    """CP1 THD adapter: one carrier per sample, in original sequence order."""
    _validate_pipeline_batch(batch)
    lengths = batch.seq_lens.tolist()
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError('Packed pipeline requires positive sequence lengths')
    if start == 0:
        if state is not None:
            raise ValueError('First packed range requires fresh sequence states')
        state = tuple((None, (-1, -1)) for _ in lengths)
    elif state is None or len(state) != len(lengths):
        raise ValueError('Packed pipeline requires one state per sequence')
    states, result = _pipeline_ranges(model, batch, start, end, state)
    return {'packed_pipeline_state': states, **result}


def _validate_pipeline_batch(batch):
    _validate_text_batch(batch)
    if batch.routed_experts is not None or batch.r3_replay_mask is not None:
        raise NotImplementedError(
            'Pipeline routing replay requires scheduler integration'
        )


def _pipeline_ranges(model, batch, start, end, states):
    from megatron.lite.primitive.modules.paired_payload import PairedPayload

    outputs = tuple(
        model.forward_pipeline_range(
            ids[None], start=start, end=end, payload=p, owners=o
        )
        for ids, (p, o) in zip(batch.input_ids.split(batch.seq_lens.tolist()), states)
    )
    result = {}
    if end == len(model.layers):
        final = PairedPayload(
            h=torch.cat([p.h for p, _ in outputs], dim=1),
            p=torch.cat([p.p for p, _ in outputs], dim=1),
        )
        result = _text_output(model.finish_pipeline(final)[0], batch)
    return outputs, result


def load_hf_weights(chunk, hf_path, model_cfg, ps):
    if hf_path:
        load_model(chunk, hf_path)


def _single(chunks):
    if len(chunks) != 1:
        raise NotImplementedError('Single-rank V4.1 export requires one chunk')
    return chunks[0]


def export_hf_weights(chunks, model_cfg, ps, **kwargs):
    yield from _export_hf_weights_impl(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(chunks, path, model_cfg, ps, **kwargs):
    save_model(_single(chunks), path, **kwargs)


def vocab_size(model_cfg):
    return model_cfg.to_hf_dict()['text_config']['vocab_size']


# Both replay inputs use V4's shared packers with V4.1's contiguous padding.
pack_routed_experts = partial(
    _pack_routed_experts, contiguous=True, contiguous_padding=True
)
pack_r3_replay_mask = partial(
    _pack_r3_replay_mask, contiguous=True, contiguous_padding=True
)


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
