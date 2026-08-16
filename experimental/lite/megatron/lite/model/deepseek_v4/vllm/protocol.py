"""Training-capable protocol for the DeepSeek-V4 vLLM implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm.checkpoint import (
    EXPERT_CLASSIFIER,
    export_hf_weights,
    invalidate_bound_source_scales,
    load_hf_weights,
    save_hf_weights,
)
from megatron.lite.model.deepseek_v4.vllm.runtime_metadata import (
    DS4MoEKernelMetadataBuilderAdapter,
    DS4SparseAttentionMetadataBuilderAdapter,
    DS4SparseIndexerCompressorMetadataAdapter,
    ds4_vllm_forward_context,
    initialize_ds4_vllm_batch_invariance,
)
from megatron.lite.model.protocol_utils import add_loss_context_kwargs, nested_from_packed
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig

_CANONICAL_STAGES = frozenset(
    {"mhc", "linear", "kv_flashmla", "o_proj", "router_moe", "deepep"}
)
_ALIASES = {
    "attn": ("mhc", "linear", "kv_flashmla", "o_proj"),
    "moe": ("router_moe", "deepep"),
}
_SELECTABLE_MODULES = _CANONICAL_STAGES | _ALIASES.keys()


@dataclass(frozen=True)
class SelectorConfig:
    """Stable selector for global decoder layer IDs and module kinds."""

    global_layer_ids: tuple[int, ...] = ()
    module_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        layer_ids = tuple(self.global_layer_ids)
        requested_names = tuple(self.module_names)
        if any(type(layer_id) is not int or layer_id < 0 for layer_id in layer_ids):
            raise ValueError("global_layer_ids must contain non-negative integers.")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("global_layer_ids must not contain duplicates.")
        unknown = set(requested_names) - _SELECTABLE_MODULES
        if unknown:
            raise ValueError(f"Unknown selector module names: {sorted(unknown)}")
        if any(type(name) is not str for name in requested_names):
            raise TypeError("module_names must contain strings.")
        if len(set(requested_names)) != len(requested_names):
            raise ValueError("module_names must not contain duplicates.")
        module_names: list[str] = []
        for name in requested_names:
            expanded = _ALIASES.get(name, (name,))
            for stage in expanded:
                if stage not in module_names:
                    module_names.append(stage)
        object.__setattr__(self, "global_layer_ids", layer_ids)
        object.__setattr__(self, "module_names", tuple(module_names))

    def selects(self, global_layer_id: int, module_name: str) -> bool:
        return (
            global_layer_id in self.global_layer_ids
            and module_name in self.module_names
        )


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: Any | None = None
    optimizer_config: OptimizerConfig | None = None
    hf_path: str = ""
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    recompute: tuple[str, ...] = ()
    offload: tuple[str, ...] = ()
    use_thd: bool = False
    use_deepep: bool = False
    attention_backend_override: str | None = None
    deterministic: bool = True
    qat: Any | None = None
    mtp_enable: bool = False
    dsa_indexer_loss_coeff: float = 0.0
    max_tokens_per_rank: int = 8192

    def __post_init__(self) -> None:
        if isinstance(self.selector, Mapping):
            object.__setattr__(self, "selector", SelectorConfig(**dict(self.selector)))


def is_expert_param(name: str) -> bool:
    return EXPERT_CLASSIFIER(name)


def _finalize_replica_grads(model: nn.Module, parallel_state) -> None:
    """Average dense and expert gradients over their respective replicas."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = (
            parallel_state.ep_dp_group
            if is_expert_param(name)
            else parallel_state.dp_group
        )
        size = (
            parallel_state.expert_dp_size
            if is_expert_param(name)
            else parallel_state.dp_size
        )
        if group is not None and size > 1:
            dist.all_reduce(parameter.grad, group=group)
            parameter.grad.div_(size)


def build_model_config(source: str | Path | dict, **overrides) -> DeepseekV4Config:
    if isinstance(source, dict):
        config = DeepseekV4Config._from_hf_dict(source)
    else:
        config = DeepseekV4Config.from_hf(str(source))
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def _validate_contract(model_cfg: DeepseekV4Config, impl_cfg: ImplConfig) -> None:
    if impl_cfg.dsa_indexer_loss_coeff < 0.0:
        raise ValueError("dsa_indexer_loss_coeff must be >= 0")
    if impl_cfg.max_tokens_per_rank <= 0:
        raise ValueError("max_tokens_per_rank must be positive")
    if not 0 <= model_cfg.num_hash_layers <= model_cfg.num_hidden_layers:
        raise ValueError(
            "num_hash_layers is a zero-based prefix length and must be between "
            f"0 and num_hidden_layers; got {model_cfg.num_hash_layers} for "
            f"{model_cfg.num_hidden_layers} layers."
        )
    parallel = impl_cfg.parallel
    dimensions = {
        "tp": parallel.tp,
        "etp": 1 if parallel.etp is None else parallel.etp,
        "pp": parallel.pp,
        "vpp": parallel.vpp,
        "cp": parallel.cp,
    }
    unsupported = {name: size for name, size in dimensions.items() if size > 1}
    if unsupported:
        values = ", ".join(f"{name}={size}" for name, size in unsupported.items())
        raise NotImplementedError(
            f"DeepSeek V4 vLLM layer-0 keeps TP/ETP/PP/VPP/CP at one; got {values}."
        )
    if parallel.ep != 2 or not impl_cfg.use_deepep:
        raise NotImplementedError(
            "DeepSeek V4 vLLM layer-0 requires EP=2 and use_deepep=True."
        )
    if impl_cfg.mtp_enable:
        raise NotImplementedError("DeepSeek V4 vLLM skeleton does not support MTP yet.")
    disabled_features = {
        "recompute": impl_cfg.recompute,
        "offload": impl_cfg.offload,
        "use_thd": impl_cfg.use_thd,
        # The vLLM implementation always executes its explicit FlashMLA stage.
        # Accept the runtime/bench default spelling instead of treating the
        # equivalent "flash" selection as an unsupported feature.
        "attention_backend_override": (
            impl_cfg.attention_backend_override
            if impl_cfg.attention_backend_override not in (None, "flash")
            else None
        ),
        "qat": impl_cfg.qat,
    }
    enabled = [name for name, value in disabled_features.items() if value]
    if enabled:
        raise NotImplementedError(
            "DeepSeek V4 vLLM skeleton does not install training/runtime features: "
            + ", ".join(enabled)
        )
    invalid_ids = [
        layer_id
        for layer_id in impl_cfg.selector.global_layer_ids
        if layer_id >= model_cfg.num_hidden_layers
    ]
    if invalid_ids:
        raise ValueError(f"Selector layer IDs are outside the model: {invalid_ids}")
    selected_ids = impl_cfg.selector.global_layer_ids
    if selected_ids:
        expected_prefix = tuple(range(max(selected_ids) + 1))
        if selected_ids != expected_prefix:
            raise ValueError(
                "DeepSeek V4 vLLM selectors must form a contiguous global-layer "
                f"prefix starting at zero; got {selected_ids}, expected {expected_prefix}."
            )
    if selected_ids and max(selected_ids) >= 3:
        ratios = tuple(model_cfg.compress_ratios[:4])
        if ratios != (0, 0, 4, 128):
            raise ValueError(
                "DeepSeek V4 4-layer audit requires compress_ratios[:4] "
                f"to be (0, 0, 4, 128); got {ratios}."
            )
        if model_cfg.num_hash_layers != 3:
            raise ValueError(
                "DeepSeek V4 4-layer audit requires num_hash_layers=3."
            )
    if 0 in impl_cfg.selector.global_layer_ids:
        missing = _CANONICAL_STAGES - set(impl_cfg.selector.module_names)
        if missing:
            raise ValueError(
                "layer-0 candidate audit requires every execution stage; missing "
                + ", ".join(sorted(missing))
            )


def _roll_packed_targets(
    values: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
) -> torch.Tensor | None:
    if values is None or seq_lens is None:
        return values
    rolled = values.clone()
    offset = 0
    for length_value in seq_lens.detach().cpu().tolist():
        length = int(length_value)
        if length > 0:
            rolled[offset : offset + length - 1] = values[offset + 1 : offset + length]
            rolled[offset + length - 1] = 0
        offset += length
    if offset != values.numel():
        raise ValueError(
            f"packed sequence lengths sum to {offset}, but targets contain "
            f"{values.numel()} values"
        )
    return rolled


def _forward_step(
    model: nn.Module,
    batch,
    *,
    attention_metadata=None,
    moe_metadata=None,
) -> dict[str, torch.Tensor]:
    seq_lens = getattr(batch, "seq_lens", None)
    kwargs = {
        "input_ids": batch.input_ids,
        "position_ids": getattr(batch, "position_ids", None),
        "attention_metadata": (
            getattr(batch, "attention_metadata", None)
            if attention_metadata is None
            else attention_metadata
        ),
        "moe_metadata": (
            getattr(batch, "moe_metadata", None)
            if moe_metadata is None
            else moe_metadata
        ),
        "labels": _roll_packed_targets(getattr(batch, "labels", None), seq_lens),
        "loss_mask": _roll_packed_targets(getattr(batch, "loss_mask", None), seq_lens),
        "temperature": getattr(batch, "temperature", 1.0),
    }
    add_loss_context_kwargs(kwargs)
    return model(**kwargs)


def unpack_forward_output(model: nn.Module, batch, output) -> Any:
    del model
    return nested_from_packed(output, batch.seq_lens)


def build_model(model_cfg: DeepseekV4Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    _validate_contract(model_cfg, impl_cfg)
    initialize_ds4_vllm_batch_invariance()
    from megatron.lite.model.deepseek_v4.vllm.model import DeepseekV4Model

    parallel_state = init_parallel(impl_cfg.parallel)
    model = DeepseekV4Model(
        model_cfg,
        ps=parallel_state,
        use_deepep=impl_cfg.use_deepep,
        selected_layer_ids=impl_cfg.selector.global_layer_ids,
        selected_module_names=impl_cfg.selector.module_names,
        indexer_loss_coeff=impl_cfg.dsa_indexer_loss_coeff,
    )
    # The runtime loads replicated HF tensors through NCCL before the deferred
    # FSDP2 wrap.  Keep that lifecycle identical to the Lite protocol: masters
    # must already live on the rank-local CUDA device during checkpoint load.
    if torch.cuda.is_available():
        model = model.cuda()
        # The production vLLM GPUWorker creates this process-global manager
        # before constructing its model runner.  The mLite runtime directly
        # reuses vLLM sparse-indexer and attention kernels, so it must honor
        # the same lifecycle even though it does not instantiate GPUWorker.
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            is_workspace_manager_initialized,
        )

        if not is_workspace_manager_initialized():
            init_workspace_manager(next(model.parameters()).device, num_ubatches=1)
    selected_layers = impl_cfg.selector.global_layer_ids
    attention_builders = None
    moe_metadata = None

    def ensure_runtime_assets():
        nonlocal attention_builders, moe_metadata
        if attention_builders is not None and moe_metadata is not None:
            return attention_builders, moe_metadata
        device = next(model.parameters()).device
        attention_builders = {
            0: DS4SparseAttentionMetadataBuilderAdapter.from_hf(
                impl_cfg.hf_path,
                model_cfg,
                device=device,
            ),
            **{
                layer_idx: DS4SparseIndexerCompressorMetadataAdapter.from_hf(
                    impl_cfg.hf_path,
                    model_cfg,
                    layer_idx=layer_idx,
                    device=device,
                )
                for layer_idx in selected_layers
                if layer_idx > 0
            },
        }
        moe_metadata = {
            layer_idx: DS4MoEKernelMetadataBuilderAdapter(
                model_cfg,
                device=device,
                ep_size=parallel_state.ep_size,
                max_tokens_per_rank=impl_cfg.max_tokens_per_rank,
                layer_idx=layer_idx,
            ).build()
            for layer_idx in selected_layers
        }
        return attention_builders, moe_metadata
    from vllm.config import VllmConfig

    vllm_config = VllmConfig()

    def forward_step(model: nn.Module, batch) -> dict[str, torch.Tensor]:
        attention_metadata = getattr(batch, "attention_metadata", None)
        moe_metadata = getattr(batch, "moe_metadata", None)
        if (attention_metadata is None) != (moe_metadata is None):
            raise ValueError(
                "caller-owned attention_metadata and moe_metadata must be "
                "provided together"
            )
        current_attention_builders = None
        current_moe_metadata = None
        if attention_metadata is None or moe_metadata is None:
            current_attention_builders, current_moe_metadata = ensure_runtime_assets()
        seq_lens = getattr(batch, "seq_lens", None)
        if seq_lens is None:
            token_counts = [int(batch.input_ids.numel())]
        else:
            token_counts = [int(value) for value in seq_lens.detach().cpu().tolist()]
        if sum(token_counts) > impl_cfg.max_tokens_per_rank:
            raise ValueError(
                f"packed batch has {sum(token_counts)} tokens, exceeding "
                f"max_tokens_per_rank={impl_cfg.max_tokens_per_rank}"
            )
        if attention_metadata is None:
            assert current_attention_builders is not None
            attention_metadata = {
                layer_idx: (
                    current_attention_builders[layer_idx].build_prefill(token_counts[0])
                    if len(token_counts) == 1
                    else current_attention_builders[layer_idx].build_prefill_batch(token_counts)
                )
                for layer_idx in selected_layers
            }
        if moe_metadata is None:
            assert current_moe_metadata is not None
            moe_metadata = current_moe_metadata
        with ds4_vllm_forward_context(
            batch,
            parallel_state,
            vllm_config=vllm_config,
        ):
            return _forward_step(
                model,
                batch,
                attention_metadata=attention_metadata,
                moe_metadata=moe_metadata,
            )

    optimizer = None
    finalize_grads = None
    post_model_load_hook = None
    optimizer_backend = "none"
    if impl_cfg.optimizer is not None:
        config = impl_cfg.optimizer_config or OptimizerConfig()
        optimizer_name = (
            impl_cfg.optimizer
            if isinstance(impl_cfg.optimizer, str)
            else config.optimizer
        )
        if optimizer_name not in ("adam", "adamw", "sgd", "fsdp2"):
            raise ValueError(
                f"DS4 vLLM training supports adam/adamw/sgd/fsdp2, got {optimizer_name!r}"
            )
        if optimizer_name == "fsdp2":
            optimizer_backend = "fsdp2"

            def _post_model_load_hook():
                from megatron.lite.model.deepseek_v4.vllm.model import DeepseekV4Layer
                from megatron.lite.primitive.optimizers.fsdp2 import (
                    build_fsdp2_training_optimizer,
                )

                return {
                    "optimizer": build_fsdp2_training_optimizer(
                        [model],
                        config,
                        parallel_state,
                        unit_modules=(DeepseekV4Layer,),
                        expert_classifier=is_expert_param,
                        deterministic=impl_cfg.deterministic,
                        vpp=impl_cfg.parallel.vpp,
                        leaf_module_names=(),
                        # vLLM units intentionally contain both BF16 matrix
                        # masters and FP32 mHC/APE state. FSDP requires a
                        # uniform original shard dtype per unit; its FP32-shard
                        # path records and restores each parameter's model
                        # dtype for forward execution.
                        use_fp32_shards=True,
                        # DS4 vLLM kernels own their mixed-dtype boundaries
                        # (BF16 hidden states, FP32 RoPE/router/mHC metadata).
                        # Recursive FSDP casting corrupts those contracts.
                        cast_forward_inputs=False,
                    )
                }

            post_model_load_hook = _post_model_load_hook
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                foreach=False,
            )
            optimizer_backend = "sgd"
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.lr,
                betas=(config.adam_beta1 or 0.9, config.adam_beta2 or 0.999),
                eps=config.adam_eps or 1e-8,
                weight_decay=config.weight_decay,
            )
            optimizer_backend = "adamw"

        if optimizer_name != "fsdp2":

            def finalize_grads() -> None:
                _finalize_replica_grads(model, parallel_state)

    extras = {
        "model_cfg": model_cfg,
        "optimizer_backend": optimizer_backend,
        "selector": impl_cfg.selector,
    }
    if torch.cuda.is_available():
        from vllm.v1.worker.workspace import reset_workspace_manager

        extras["close_hook"] = reset_workspace_manager
    extras["post_optimizer_step_hook"] = lambda: invalidate_bound_source_scales(
        model
    )
    if post_model_load_hook is not None:
        extras["post_model_load_hook"] = post_model_load_hook

    return ModelBundle(
        chunks=[model],
        parallel_state=parallel_state,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=forward_step,
        extras=extras,
    )


def vocab_size(model_cfg: DeepseekV4Config) -> int | None:
    return model_cfg.vocab_size


__all__ = [
    "ImplConfig",
    "SelectorConfig",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "is_expert_param",
    "load_hf_weights",
    "save_hf_weights",
    "unpack_forward_output",
    "vocab_size",
]
