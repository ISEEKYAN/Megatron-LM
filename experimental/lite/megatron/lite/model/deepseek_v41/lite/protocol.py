# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text protocol with replicated data parallel training and explicit optimizer policy."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import partial

import torch
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.protocol_utils import ( pack_r3_replay_mask as _pack_r3_replay_mask, )
from megatron.lite.model.protocol_utils import ( pack_routed_experts as _pack_routed_experts, )
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.config_fields import project_fields
from megatron.lite.primitive.modules.vision_training import ( VisionSchedule, VisionTrainability, )
from megatron.lite.primitive.packed_lm import _forward_step as _packed_step
from megatron.lite.primitive.packed_lm import ( prepare_microbatches, unpack_forward_output, )
from megatron.lite.primitive.parallel import route_records as _route_records
from megatron.lite.primitive.parallel.owned_ddp import wrap_owned_ddp
from megatron.lite.primitive.parallel.state import ParallelState, init_parallel
from megatron.lite.runtime.contracts import ParallelConfig

from .checkpoint import export_hf_weights as _export_hf_weights_impl
from .checkpoint import load_model, save_model

_forward_step = partial(_packed_step, model_name='V4.1')
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
            object.__setattr__( self, "optimizer_config", OptimizerConfig(**self.optimizer_config) )
        if isinstance(self.dtype, str):
            dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
            if self.dtype not in dtypes:
                raise ValueError("V4.1 residual dtype must be BF16 or FP32")
            object.__setattr__(self, "dtype", dtypes[self.dtype])


def build_model_config(source, **overrides):
    if overrides:
        raise ValueError('Apply overrides to the explicit nested source config')
    return ( DeepseekV41Config(source) if isinstance(source, dict)
        else DeepseekV41Config.from_hf(source) )


UNSUPPORTED = ( ( lambda c, p: p.pp > 1
        and (not c.text_only or c.external_vision_device is not None), NotImplementedError,
        'V4.1_PP_TEXT_ONLY: PP currently supports text-only training; use PP=1 for multimodal training',
    ), ( lambda c, p: p.pp > 1 and c.pipeline_split_layer != 20,
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
            'prepare_microbatches': partial(prepare_microbatches, dp_group=ps.dp_group),
            'optimizer_backend': 'none' if optimizer is None else 'v41',
            'parameter_bindings': model.parameter_bindings,
        },
    )


def load_hf_weights(chunk, hf_path, model_cfg, ps):
    if hf_path:
        load_model(chunk, hf_path)


export_hf_weights = _export_hf_weights_impl


def save_hf_weights(chunks, path, model_cfg, ps, **kwargs):
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


router_replay_roots = partial(_route_records.router_replay_roots, model_name='V4.1')
validate_router_replay = partial(
    _route_records.validate_router_replay, model_name='V4.1'
)
unpack_recorded_routed_experts = partial(
    _route_records.unpack_recorded_routed_experts, model_name='V4.1'
)
