# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.8 text training protocol for the existing MLite runtime."""
from dataclasses import dataclass, field

import torch
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.config import load_hf_config_dict
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig

from .config import Qwen3_8_FlashNextTextConfig


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = 'dist_opt'
    optimizer_config: OptimizerConfig | None = None
    # Explicit reduced table for scale proxies; None preserves all release primes.
    ngram_primes: tuple[int, ...] | None = None
    router_aux_loss_coef: float | None = None
    deterministic: bool = True


def build_model_config(source, **overrides):
    config = (
        dict(source) if isinstance(source, dict) else load_hf_config_dict(str(source))
    )
    text = dict(config.get('text_config', config))
    text.update(overrides)
    return Qwen3_8_FlashNextTextConfig.from_hf_dict(text)


def is_expert_param(name):
    return '.experts.' in name


def _forward_step(model, batch):
    lengths = batch.seq_lens.tolist()
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=batch.input_ids.device,
        dtype=torch.int32,
    )
    labels = None
    if batch.labels is not None:
        labels = torch.cat(
            [part.roll(-1) for part in batch.labels.flatten().split(lengths)]
        ).reshape(1, -1)
    return model(
        input_ids=batch.input_ids.reshape(1, -1),
        labels=labels,
        cu_seqlens=cu if len(lengths) > 1 else None,
    )


def unpack_forward_output(model, batch, output):
    from megatron.lite.model.protocol_utils import nested_from_packed

    return nested_from_packed(output.reshape(-1), batch.seq_lens)


def build_model(model_cfg, *, impl_cfg):
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

    from .model import Qwen38Model

    if impl_cfg.router_aux_loss_coef is not None:
        model_cfg.router_aux_loss_coef = impl_cfg.router_aux_loss_coef
    ps = init_parallel(impl_cfg.parallel)
    chunks = [
        Qwen38Model(model_cfg, ps, ngram_primes=impl_cfg.ngram_primes)
        .to(torch.bfloat16)
        .cuda()
    ]
    optimizer, finalize = None, None
    if impl_cfg.optimizer == 'dist_opt':
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_dist_opt_training_optimizer,
        )
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        optimizer, finalize = build_dist_opt_training_optimizer(
            chunks,
            model_cfg=model_cfg,
            impl_cfg=impl_cfg,
            ps=ps,
            model_name='qwen3_8_flash_next',
            is_expert=is_expert_param,
            deterministic=impl_cfg.deterministic,
        )
        register_training_hooks(chunks, optimizer)
        attach_model_sharded_state_dict(chunks, ps, is_expert=is_expert_param)
    elif impl_cfg.optimizer is not None:
        raise ValueError(f'Unsupported Qwen3.8 optimizer: {impl_cfg.optimizer}')
    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize,
        forward_step=_forward_step,
        extras={
            'model_cfg': model_cfg,
            'optimizer_backend': impl_cfg.optimizer or 'none',
            'pre_forward_hook': MoEAuxLossAutoScaler.set_loss_scale,
        },
    )


def load_hf_weights(chunk, hf_path, model_cfg, ps):
    if hf_path:
        raise NotImplementedError('QWEN38_HF_WEIGHT_LOADING_NOT_VALIDATED')


def vocab_size(model_cfg):
    return model_cfg.vocab_size
