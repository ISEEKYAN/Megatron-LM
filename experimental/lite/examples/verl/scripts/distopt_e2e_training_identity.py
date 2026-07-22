# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end TRAINING bitwise identity: mcore-native vs MLite-DistOpt Muon.

Context (bayan 2026-07-51 decisive gate). The config/kernel identity receipt
(``tp_distributed_muon_identity.py``, job 14245178, ``max_abs=0``) proved that
MLite's DistOpt *lowering* produces the same ``OptimizerConfig`` and that the
distributed Newton-Schulz *kernel* runs correctly on a real TP process group.
bayan's remaining requirement is stronger and different: prove that MLite's
DistOpt **integration wiring** -- the grad plumbing around that kernel -- is
correct end to end. Concretely, run a REAL training step through both stacks and
compare the trained weights:

    forward -> backward -> DDP grad bucketing/reduce -> DistributedOptimizer
    master-grad sharding -> finalize grads -> muon optimizer.step()

and check ``torch.equal`` on the resulting weights. ``max_abs == 0`` proves MLite's
integration (``_mark_dist_opt_parallel_attrs`` per-param tagging that governs
grad-norm accounting, DDP bucket layout, ``build_dist_opt_optimizer_config``,
``finalize_dist_opt_grads``, and the DistributedOptimizer master-grad path) is
bit-for-bit faithful to a raw Megatron-Core ``get_megatron_optimizer`` build;
``!= 0`` would catch a grad-wiring bug.

The two arms are genuinely independent stacks (NO forced-same-object, NO
pg_collection=None):

  * ARM-native  -- raw Megatron-Core: wrap the model with ``DistributedDataParallel``
    directly, hand-build the muon ``OptimizerConfig``, call ``get_megatron_optimizer``
    (using mpu global process groups), and finalize with ``finalize_model_grads``.
  * ARM-mlite   -- MLite's ``build_dist_opt_stack`` (which internally runs
    ``_mark_dist_opt_parallel_attrs`` + ``build_dist_opt_optimizer_config`` +
    ``DistributedDataParallel`` + ``get_megatron_optimizer`` with its OWN
    ``pg_collection`` built from the lite ``ParallelState``) and finalize with
    ``finalize_dist_opt_grads``.

Both arms start from an identical weight init (native's ``state_dict`` is copied
into mlite's model before wrapping) and consume identical per-rank data, so any
post-step weight delta is attributable to the integration wiring, nothing else.

Geometry: run under ``torchrun`` with ``world_size = tp_size * dp_size``. With
``TP=2, DP=2`` (4 GPUs) the run exercises the full chain: TP-sharded matmul comm
in fwd/bwd, DDP grad all-reduce across DP=2, DistributedOptimizer master-grad
sharding across DP=2, and the distributed cross-TP muon Newton-Schulz.

Non-vacuity: a NEGATIVE-CONTROL arm ("mlite-nofinalize") deliberately drops the
grad-finalization step (so DP grads are never reduced). It must DIVERGE from
ARM-native (``max_abs > 0``), proving the ``torch.equal`` check is sensitive to a
real grad-wiring defect and the primary ``max_abs == 0`` is not vacuous.
"""

from __future__ import annotations

import os

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from types import SimpleNamespace

from megatron.core.distributed import (  # pyright: ignore[reportMissingImports]
    DistributedDataParallel,
    DistributedDataParallelConfig,
)
from megatron.core.distributed.finalize_model_grads import (  # pyright: ignore[reportMissingImports]
    finalize_model_grads,
)
from megatron.core.optimizer import (  # pyright: ignore[reportMissingImports]
    OptimizerConfig as CoreOptimizerConfig,
    get_megatron_optimizer,
)
from megatron.core.transformer.enums import ModelType  # pyright: ignore[reportMissingImports]
from megatron.core.transformer.transformer_config import (  # pyright: ignore[reportMissingImports]
    TransformerConfig,
)
from megatron.lite.primitive.optimizers.megatron_wrap import (  # pyright: ignore[reportMissingImports]
    build_dist_opt_stack,
    finalize_dist_opt_grads,
)
from megatron.lite.primitive.parallel import init_parallel  # pyright: ignore[reportMissingImports]
from megatron.lite.runtime.contracts.config import (  # pyright: ignore[reportMissingImports]
    OptimizerConfig as LiteOptimizerConfig,
    ParallelConfig,
)

# ---- contract hyper-parameters ---------------------------------------------
H = 32                 # full hidden size (must divide by tp_size)
BATCH = 8
STEPS = 6
SEED = 1234
# muon knobs (several non-default -> a lowering that silently reverts would show
# up as a weight delta, not just a config diff).
MUON_HP = dict(
    optimizer="muon",
    lr=1e-3,
    weight_decay=0.1,
    clip_grad=1.0,
    muon_tp_mode="distributed",
    muon_momentum=0.9,
    muon_nesterov=True,
    muon_num_ns_steps=6,
    muon_coefficient_type="quintic",
    muon_scale_mode="spectral",
    muon_fp32_matmul_prec="high",
)


class _ReduceFromTP(torch.autograd.Function):
    """All-reduce (sum) over the TP group in forward; identity in backward.

    This is the standard row-parallel output reduction (mcore's
    ``_ReduceFromModelParallelRegion``): the full output is the sum of per-rank
    partials, and the grad w.r.t. each partial is the identity.
    """

    @staticmethod
    def forward(ctx, x, group):  # noqa: D401
        y = x.clone()
        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(y, group=group)
        return y

    @staticmethod
    def backward(ctx, grad):  # noqa: D401
        return grad, None


class TinyTPModel(nn.Module):
    """Column-parallel -> ReLU -> row-parallel MLP with real TP grad flow.

    ``w_col`` is sharded along dim0 (partition_dim=0, the column-parallel weight)
    and ``w_row`` along dim1 (partition_dim=1, the row-parallel weight). Both are
    2D + ``tensor_model_parallel=True`` so they are the exact param shape the
    distributed muon Newton-Schulz + dist-opt grad-norm accounting target.
    """

    def __init__(self, tp_size: int, tp_group):
        super().__init__()
        assert H % tp_size == 0, "H must divide tp_size"
        local = H // tp_size
        self._tp_group = tp_group
        # local shards; identical init is enforced later by state_dict copy.
        self.w_col = nn.Parameter(torch.empty(local, H, dtype=torch.bfloat16, device="cuda"))
        self.w_row = nn.Parameter(torch.empty(H, local, dtype=torch.bfloat16, device="cuda"))
        nn.init.normal_(self.w_col, std=0.05)
        nn.init.normal_(self.w_row, std=0.05)
        # mark the TP-partition metadata the way mcore's ColumnParallelLinear /
        # RowParallelLinear do at construction. Both arms build the SAME model, so
        # these attrs are present identically before either wrapping path runs;
        # what differs between arms is the *wiring code*, not these attrs.
        for p, dim in ((self.w_col, 0), (self.w_row, 1)):
            p.tensor_model_parallel = True
            p.partition_dim = dim
            p.partition_stride = 1

    def forward(self, x):
        # x is replicated across TP ranks (same on every tp rank).
        h = torch.relu(x @ self.w_col.t())          # (B, H/tp)  column-parallel output
        y_partial = h @ self.w_row.t()              # (B, H)     row-parallel partials
        y = _ReduceFromTP.apply(y_partial, self._tp_group)
        return y.float().square().mean()            # scalar loss


def _model_cfg():
    return SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=H,
        num_attention_heads=2,
        num_experts=None,          # dense: no expert params in this proof
        moe_intermediate_size=None,
        add_bias_linear=False,
    )


def _lite_engine_cfg(parallel):
    return SimpleNamespace(
        model_name="tiny_tp_mlp",
        parallel=parallel,
        optimizer=LiteOptimizerConfig(**MUON_HP),
        deterministic=False,
    )


def _native_transformer_cfg(p):
    """Equivalent TransformerConfig for the native DDP wrap (same VALUES as the
    mlite path builds internally -- independence is in the CODE, not the config
    numbers)."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=H,
        num_attention_heads=2,
        tensor_model_parallel_size=p.tp,
        pipeline_model_parallel_size=p.pp,
        context_parallel_size=p.cp,
        expert_model_parallel_size=p.ep,
        sequence_parallel=p.tp > 1,
        bf16=True,
        params_dtype=torch.bfloat16,
    )


def _build_native_arm(model: nn.Module, p, transformer_cfg):
    """Raw Megatron-Core stack: DDP + hand-built muon config + mpu groups."""
    transformer_cfg.finalize_model_grads_func = finalize_model_grads
    model.model_type = ModelType.encoder_or_decoder
    # native marking: dense params all-reduce; TP attrs already on the shards.
    for _name, param in model.named_parameters():
        if not hasattr(param, "allreduce"):
            param.allreduce = True
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_grad_reduce=False, grad_reduce_in_fp32=True
    )
    wrapped = DistributedDataParallel(transformer_cfg, ddp_config, model)
    opt_cfg = CoreOptimizerConfig(
        use_distributed_optimizer=True, bf16=True, params_dtype=torch.bfloat16, **MUON_HP
    )
    optimizer = get_megatron_optimizer(config=opt_cfg, model_chunks=[wrapped])

    def finalize():
        finalize_model_grads([wrapped])

    return wrapped, optimizer, finalize


def _build_mlite_arm(model: nn.Module, p, ps):
    """MLite DistOpt stack via the production build_dist_opt_stack path."""
    engine_cfg = _lite_engine_cfg(p)
    wrapped_chunks, optimizer = build_dist_opt_stack(
        [model], model_cfg=_model_cfg(), engine_cfg=engine_cfg, ps=ps
    )

    def finalize():
        finalize_dist_opt_grads(wrapped_chunks, optimizer)

    return wrapped_chunks[0], optimizer, finalize


def _inner_module(wrapped):
    return wrapped.module if hasattr(wrapped, "module") else wrapped


def _copy_init(src_wrapped, dst_wrapped):
    """Force dst model to start from src's exact weights (identical init)."""
    src = _inner_module(src_wrapped)
    dst = _inner_module(dst_wrapped)
    with torch.no_grad():
        s = dict(src.named_parameters())
        for name, p in dst.named_parameters():
            p.copy_(s[name])


def _step(wrapped, optimizer, finalize, x):
    optimizer.zero_grad()
    if hasattr(wrapped, "zero_grad_buffer"):
        wrapped.zero_grad_buffer()
    loss = wrapped(x)
    loss.backward()
    finalize()
    optimizer.step()  # return signature varies across mcore versions; ignore it
    return loss.detach()


def _local_weights(wrapped):
    m = _inner_module(wrapped)
    return {n: p.detach().float().clone() for n, p in m.named_parameters()}


def _max_abs(a: dict, b: dict) -> float:
    keys = sorted(set(a) | set(b))
    worst = 0.0
    for k in keys:
        d = (a[k] - b[k]).abs().max().item()
        worst = max(worst, d)
    return worst


def _global_max(local_val: float, device) -> float:
    t = torch.tensor([local_val], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)

    tp = int(os.environ.get("TP_SIZE", "2"))
    assert world % tp == 0, f"world={world} not divisible by tp={tp}"
    dp = world // tp
    parallel = ParallelConfig(tp=tp, ep=1, etp=1, pp=1, cp=1)

    # mpu globals for the NATIVE arm; lite ParallelState for the MLITE arm.
    from megatron.core import parallel_state as mpu  # pyright: ignore[reportMissingImports]

    if not mpu.is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
    ps = init_parallel(parallel)

    if rank == 0:
        print(
            f"[env] world={world} tp={tp} dp={dp} H={H} batch={BATCH} steps={STEPS} "
            f"muon_tp_mode={MUON_HP['muon_tp_mode']}",
            flush=True,
        )

    # ---- build both real, independent stacks from an identical init ----------
    torch.manual_seed(SEED + ps.tp_rank)  # seed shard init deterministically per tp-rank
    native_model = TinyTPModel(tp, ps.tp_group)
    torch.manual_seed(SEED + 777 + ps.tp_rank)  # different init; will be overwritten by copy
    mlite_model = TinyTPModel(tp, ps.tp_group)
    ctrl_model = TinyTPModel(tp, ps.tp_group)

    transformer_cfg_native = _native_transformer_cfg(parallel)
    native_wrapped, native_opt, native_fin = _build_native_arm(
        native_model, parallel, transformer_cfg_native
    )
    mlite_wrapped, mlite_opt, mlite_fin = _build_mlite_arm(mlite_model, parallel, ps)
    ctrl_wrapped, ctrl_opt, ctrl_fin = _build_mlite_arm(ctrl_model, parallel, ps)

    # identical init: copy native weights into mlite + control model buffers.
    _copy_init(native_wrapped, mlite_wrapped)
    _copy_init(native_wrapped, ctrl_wrapped)
    # DDP buffers hold the authoritative param copy; re-sync them from the .data
    # we just copied so the first step sees identical weights in all arms.
    for opt in (native_opt, mlite_opt, ctrl_opt):
        rmp = getattr(opt, "reload_model_params", None)
        if callable(rmp):
            rmp()

    # sanity: pre-step weights identical across arms
    pre = _max_abs(_local_weights(native_wrapped), _local_weights(mlite_wrapped))
    pre_g = _global_max(pre, device)
    if rank == 0:
        print(f"[init] PRE-STEP native-vs-mlite global_max_abs={pre_g:.3e} (expect 0)", flush=True)

    # ---- run identical training on both arms ---------------------------------
    per_step = []
    for step in range(STEPS):
        # data is data-parallel: same within a (dp_rank) across arms, differs per dp_rank.
        gen = torch.Generator(device=device).manual_seed(SEED + 1000 * (step + 1) + ps.dp_rank)
        x = torch.randn(BATCH, H, generator=gen, device=device, dtype=torch.bfloat16)

        l_native = _step(native_wrapped, native_opt, native_fin, x)
        l_mlite = _step(mlite_wrapped, mlite_opt, mlite_fin, x)
        # negative control: identical model+data but grads are NEVER finalized.
        l_ctrl = _step(ctrl_wrapped, ctrl_opt, lambda: None, x)

        local = _max_abs(_local_weights(native_wrapped), _local_weights(mlite_wrapped))
        gmax = _global_max(local, device)
        ctrl_local = _max_abs(_local_weights(native_wrapped), _local_weights(ctrl_wrapped))
        ctrl_g = _global_max(ctrl_local, device)
        per_step.append((step, gmax, ctrl_g))
        if rank == 0:
            print(
                f"[step {step}] loss native={l_native.item():.6f} mlite={l_mlite.item():.6f} | "
                f"native-vs-mlite global_max_abs={gmax:.3e} | "
                f"neg-control(no-finalize) global_max_abs={ctrl_g:.3e}",
                flush=True,
            )

    # ---- verdict -------------------------------------------------------------
    final_gmax = per_step[-1][1]
    final_ctrl = per_step[-1][2]
    equal = all(g == 0.0 for _, g, _ in per_step)
    ctrl_diverged = final_ctrl > 0.0
    if rank == 0:
        print(
            f"[B] E2E_TRAINING_IDENTITY steps={STEPS} all_steps_equal={equal} "
            f"final_native_vs_mlite_max_abs={final_gmax:.3e}",
            flush=True,
        )
        print(
            f"[D] NEG_CONTROL no-finalize final_max_abs={final_ctrl:.3e} diverged={ctrl_diverged} "
            f"(proves torch.equal is sensitive to a grad-wiring defect)",
            flush=True,
        )
        ok = equal and ctrl_diverged
        if ok:
            print(
                "RESULT PASS: mcore-native == MLite-DistOpt Muon end-to-end training "
                "(bitwise weights after real fwd/bwd/DDP/DistOpt/finalize/muon-step); "
                "neg-control diverged",
                flush=True,
            )
        else:
            print(
                f"RESULT FAIL: all_steps_equal={equal} ctrl_diverged={ctrl_diverged} "
                "(see per-step deltas above)",
                flush=True,
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
