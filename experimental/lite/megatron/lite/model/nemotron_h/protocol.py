"""Nemotron native mlite protocol; no HF model wrapper or custom scheduler."""

from dataclasses import dataclass, field

from megatron.lite.model.protocol_utils import nested_from_packed
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.parallel.cp import contiguous_slice_for_cp
from megatron.lite.primitive.parallel.thd import (
    pack_nested_thd,
    parallel_state_from_model,
    thd_pack_meta,
    unpack_thd_to_nested,
)
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.loss import get_loss_context

from .checkpoint import load_hf_weights as _load_weights
from .config import NemotronHConfig
from .functional import visible_forward
from .mamba import SSMMeta
from .model import NemotronModel


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"
    recompute: list[str] = field(default_factory=list)
    optimizer_config: OptimizerConfig | None = None
    deterministic: bool = True


MODULE_MAP = {}


def build_model_config(source, **overrides):
    config = (
        NemotronHConfig._from_hf_dict(source)
        if isinstance(source, dict)
        else NemotronHConfig.from_hf(source)
    )
    for name, value in overrides.items():
        if not hasattr(config, name):
            raise ValueError(f"Unknown Nemotron config override: {name}")
        setattr(config, name, value)
    config.__post_init__()
    return config


def is_expert(name):
    return ".mixer.experts." in name


EXPERT_CLASSIFIER = is_expert


def PLACEMENT_FN(name):
    from torch.distributed.tensor import Replicate, Shard

    return [
        Replicate(),
        Replicate(),
        Shard(0) if is_expert(name) else Replicate(),
        Replicate(),
    ]


def _token_mean_loss(log_probs, local_mask, full_mask, cp_size):
    # Megatron averages dense and expert gradients over DP*CP.
    return -(log_probs * local_mask).sum() * cp_size / full_mask.sum().clamp_min(1)


def forward_step(model, batch):
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    ps = parallel_state_from_model(model)
    loss_mask = batch.loss_mask
    if loss_mask is None:
        loss_mask = batch.input_ids.new_ones(batch.input_ids.shape)
    packed = pack_nested_thd(
        nested_from_packed(batch.input_ids, batch.seq_lens),
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_rank=ps.cp_rank,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
        split_cp=False,
        labels=nested_from_packed(batch.labels, batch.seq_lens),
        loss_mask=nested_from_packed(loss_mask, batch.seq_lens),
        roll_labels=batch.labels is not None,
        roll_loss_mask=True,
    )

    def local(tensor):
        if tensor is None:
            return None
        return contiguous_slice_for_cp(
            tensor, ps.cp_rank, ps.cp_size, seq_dim=1
        ).reshape(-1)

    meta = SSMMeta(tuple(packed.cu_seqlens_padded.cpu().tolist()))
    output = model(local(packed.input_ids), meta=meta)
    if not ps.pp_is_last:
        return {"hidden_states": output}
    labels = local(packed.labels)
    if labels is None:
        return {"logits": output}
    context = get_loss_context()
    if context is not None and context.temperature != 1.0:
        output = output / context.temperature
    valid = labels >= 0
    labels = labels.clamp_min(0)[:, None]
    log_probs = visible_forward(
        compute_token_logprobs,
        lambda logits, ids: logits.float().log_softmax(-1).gather(-1, ids.long()),
        output,
        labels,
    ).reshape(-1)
    log_probs = log_probs.masked_fill(~valid, 0)
    mask = local(packed.loss_mask)
    mask = valid if mask is None else mask * valid
    full_mask = packed.labels >= 0
    if packed.loss_mask is not None:
        full_mask = full_mask * packed.loss_mask
    loss = _token_mean_loss(log_probs, mask, full_mask, ps.cp_size)
    result = {"log_probs": log_probs[None], "loss": loss}
    if context is not None and context.calculate_entropy:
        from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy

        result["entropy"] = vocab_parallel_entropy(output, ps.tp_group)[None]
    return result


def unpack_forward_output(model, batch, output):
    ps = parallel_state_from_model(model)
    meta = thd_pack_meta(
        batch.seq_lens, tp_size=ps.tp_size, cp_size=ps.cp_size, cp_group=ps.cp_group
    )
    return unpack_thd_to_nested(output, meta, contiguous=True)


def build_model(model_cfg, *, impl_cfg):
    p = impl_cfg.parallel
    if p.tp != 1 or (p.etp or 1) != 1 or p.vpp != 1:
        raise ValueError("Native Nemotron currently requires TP1/ETP1/VPP1")
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    init_batch_invariance()
    ps = init_parallel(p)
    count = model_cfg.num_hidden_layers
    start, end = (
        count * ps.pp_rank // ps.pp_size,
        count * (ps.pp_rank + 1) // ps.pp_size,
    )
    chunks = [NemotronModel(model_cfg, ps, layer_range=(start, end), device="cuda")]
    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    if recompute_spec:
        for chunk in chunks:
            apply_recompute(chunk.layers.values(), recompute_spec, MODULE_MAP)
    optimizer = finalize_grads = None
    if impl_cfg.optimizer == "dist_opt":
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_dist_opt_training_optimizer,
        )

        optimizer, finalize_grads = build_dist_opt_training_optimizer(
            chunks,
            model_cfg=model_cfg,
            impl_cfg=impl_cfg,
            ps=ps,
            model_name="nemotron_h",
            is_expert=is_expert,
            deterministic=impl_cfg.deterministic,
        )
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict

        attach_model_sharded_state_dict(
            chunks, ps, get_placements=PLACEMENT_FN, is_expert=is_expert
        )
    elif impl_cfg.optimizer is not None:
        raise ValueError(
            "Native Nemotron uses dist_opt, not the historical FSDP adapter"
        )
    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=forward_step,
        extras={
            "model_cfg": model_cfg,
            "optimizer_backend": impl_cfg.optimizer or "none",
        },
    )


def load_hf_weights(chunk, hf_path, model_cfg, ps):
    while hasattr(chunk, "module"):
        chunk = chunk.module
    _load_weights(chunk, hf_path)


def vocab_size(model_cfg):
    return model_cfg.vocab_size


def export_hf_weights(chunks, model_cfg, ps, **kwargs):
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights as export

    from .checkpoint import NemotronExport

    yield from export(chunks, NemotronExport(model_cfg), ps, **kwargs)
