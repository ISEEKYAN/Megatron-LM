# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.8 text training protocol for the existing MLite runtime."""

from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace

import torch
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.config import load_hf_config_dict
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.parallel.thd import parallel_state_from_model
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
    # Explicit checkpoint layout change. Owners use EP; replicas use expert-DP.
    ple_owner_sharding: bool = False


def build_model_config(source, **overrides):
    config = (
        dict(source) if isinstance(source, dict) else load_hf_config_dict(str(source))
    )
    if 'text_config' in config:
        config['text_config'] = {**config['text_config'], **overrides}
    else:
        config.update(overrides)
    return Qwen3_8_FlashNextTextConfig.from_hf_dict(config)


def is_expert_param(name, *, ple_owner_sharding=False):
    return '.experts.' in name or (
        ple_owner_sharding
        and name.endswith('.ple.ple_embedding.ngram_embedding.weight')
    )


def parameter_placements(name, *, ple_owner_sharding=False):
    from torch.distributed.tensor import Replicate, Shard

    from .tp import projection_shard

    # Match the existing Qwen3.5 GroupedLinear owner layout. weightN denotes
    # a local expert: these checkpoints currently require the same EP size.
    return [
        Replicate(),
        Replicate(),
        (
            Shard(0)
            if is_expert_param(name, ple_owner_sharding=ple_owner_sharding)
            else Replicate()
        ),
        Shard(0) if projection_shard(name) else Replicate(),
    ]


PLACEMENT_FN = parameter_placements
EXPERT_CLASSIFIER = is_expert_param


def _forward_step(model, batch):
    lengths = batch.seq_lens.tolist()
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=batch.input_ids.device,
        dtype=torch.int32,
    )
    labels = None
    if batch.labels is not None:
        targets = []
        masks = (
            [None] * len(lengths)
            if batch.loss_mask is None
            else batch.loss_mask.flatten().split(lengths)
        )
        for part, mask in zip(batch.labels.flatten().split(lengths), masks):
            target = part.roll(-1)
            if mask is not None:
                target = target.masked_fill(~mask.roll(-1).bool(), -100)
            target[-1] = -100
            targets.append(target)
        labels = torch.cat(targets).reshape(1, -1)
    kwargs = dict(
        input_ids=batch.input_ids.reshape(1, -1),
        labels=labels,
        cu_seqlens=cu if len(lengths) > 1 else None,
    )
    ps = parallel_state_from_model(model)
    if ps is not None and ps.cp_size > 1:
        from .cp import shard_batch_for_qwen3_8_flash_next_cp

        # Targets are already shifted over complete documents. Never roll a shard.
        positions = torch.cat(
            [torch.arange(n, device=batch.input_ids.device) for n in lengths]
        ).reshape(1, -1)
        cp_mesh = SimpleNamespace(
            size=lambda: ps.cp_size,
            get_local_rank=lambda: ps.cp_rank,
            get_group=lambda: ps.cp_group,
        )
        tp_mesh = SimpleNamespace(size=lambda: ps.tp_size)
        _, local, layout = shard_batch_for_qwen3_8_flash_next_cp(
            cp_mesh, tp_mesh, dict(kwargs, position_ids=positions, cu_seqlens=cu)
        )
        context = local.pop('_qwen3_8_flash_next_cp_context')
        physical_cu = cu
        if int(cu[-1]) < layout.padded_seq_len:
            physical_cu = torch.cat((cu, cu.new_tensor([layout.padded_seq_len])))
        kwargs = dict(
            local,
            cp_context=context,
            cu_seqlens=physical_cu,
            # Loss targets and real router tokens are different populations.
            # Router validity remains available in context.global_padding_mask.
            loss_token_count=None if labels is None else (labels != -100).sum(),
        )
    return model(**kwargs)


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
        # MCore TEGroupedLinear, nv/dev 0cd11658f4435, fetched
        # 2026-09-13T11:11:23Z: use native FP32 wgrad accumulation only
        # when dist_opt supplies main_grad; BF16-then-cast loses cancellation.
        Qwen38Model(
            model_cfg,
            ps,
            ngram_primes=impl_cfg.ngram_primes,
            ple_owner_sharding=impl_cfg.ple_owner_sharding,
            fuse_wgrad_accumulation=impl_cfg.optimizer == 'dist_opt',
        )
        .to(torch.bfloat16)
        .cuda()
    ]
    expert_classifier = partial(
        is_expert_param, ple_owner_sharding=impl_cfg.ple_owner_sharding
    )
    placements = partial(
        parameter_placements, ple_owner_sharding=impl_cfg.ple_owner_sharding
    )
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
            is_expert=expert_classifier,
            deterministic=impl_cfg.deterministic,
        )
        if ps.tp_size > 1:
            from .tp import finalize_replicated_experts

            finalize = partial(
                finalize_replicated_experts, chunks, finalize, ps.tp_size
            )
        register_training_hooks(chunks, optimizer)
        attach_model_sharded_state_dict(
            chunks, ps, get_placements=placements, is_expert=expert_classifier
        )
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


def save_hf_weights(chunks, path, model_cfg, ps, **kwargs):
    raise NotImplementedError('QWEN38_HF_EXPORT_NOT_VALIDATED_USE_NATIVE_CHECKPOINT')


def vocab_size(model_cfg):
    return model_cfg.vocab_size
