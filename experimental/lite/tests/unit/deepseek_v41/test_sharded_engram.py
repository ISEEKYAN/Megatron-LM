# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from megatron.lite.primitive.parallel.state import ParallelState


def test_trainer_scheduler_preserves_model_lr_and_decay_policy():
    import ast
    import math
    from pathlib import Path
    from types import SimpleNamespace

    path = (
        Path(__file__).resolve().parents[3]
        / "examples/verl/verl_mlite/engine/mlite_engine.py"
    )
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "_MegatronLiteLRScheduler"
    )
    namespace = {"Any": object, "math": math}
    exec(compile(ast.Module([node], type_ignores=[]), str(path), "exec"), namespace)
    groups = [{"lr_mult": 5, "wd_mult": 0, "weight_decay": 0}, {"weight_decay": 0.1}]
    scheduler = namespace["_MegatronLiteLRScheduler"](
        SimpleNamespace(param_groups=groups),
        init_lr=0,
        max_lr=1e-4,
        min_lr=1e-4,
        lr_warmup_steps=0,
        lr_decay_steps=50,
        lr_decay_style="constant",
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=50,
        wd_incr_style="constant",
        wsd_decay_steps=None,
        lr_wsd_decay_style="constant",
    )
    scheduler.step()
    assert groups[0]["lr"] == 5e-4 and groups[0]["weight_decay"] == 0
    assert groups[1]["lr"] == 1e-4 and groups[1]["weight_decay"] == 0.1


def test_runtime_accepts_serialized_v41_optimizer_config():
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.backends.mlite.runtime import _build_impl_cfg

    config = _build_impl_cfg(
        protocol,
        MegatronLiteConfig(
            impl_cfg={
                "shard_engram": True,
                "dtype": "float32",
                "optimizer": "muon",
                "optimizer_config": {
                    "lr": 1e-4,
                    "ns_steps": 5,
                    "coefficient_type": "quintic",
                },
            }
        ),
    )
    assert config.dtype is torch.float32
    assert config.optimizer_config == protocol.OptimizerConfig(1e-4, 5, "quintic")
    assert config.shard_engram


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_model_allocates_only_owned_engram_rows(
    moe, model_config, monkeypatch, rank, trainable
):
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model
    from megatron.lite.primitive.modules.engram_lookup import ShardedEngramTable

    group = object()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda g: 3)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda g: rank)
    ps = ParallelState(dp_cp_group=group, dp_cp_size=3, dp_cp_rank=rank)
    with torch.device("meta"):
        model = DeepseekV41Model(
            model_config,
            parallel_state=ps,
            shard_engram=True,
            trainable_engram=trainable,
            quantized=False,
        )
    for index, total in zip(model.engram_layer_ids, [152, 220]):
        table = model.layers[index].engram.embed
        assert isinstance(table, ShardedEngramTable)
        boundaries = [total * i // 3 for i in range(4)]
        assert table.lookup.boundaries == tuple(boundaries)
        assert table.lookup.group is group
        assert table.weight.shape == (boundaries[rank + 1] - boundaries[rank], 32)
        assert table.scale.shape == (table.weight.shape[0], 1)
        assert (table.master is not None) == trainable
        if trainable:
            assert table.master.shape == table.weight.shape
            assert table.master.dtype == torch.float32


def _train_shards(rank, config, trainable, world, ep, cp, directory):
    from dataclasses import replace
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig
    from megatron.lite.primitive.train_step import run_microbatch_loop
    from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.manual_seed(19)
    dist.init_process_group(
        "nccl",
        init_method=(Path(directory) / "rdzv").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        impl = protocol.ImplConfig(
            parallel=ParallelConfig(ep=ep, cp=cp),
            device=f"cuda:{rank}",
            dtype=torch.float32,
            quantized=False,
            token_map=list(range(256)),
            trainable_engram=trainable,
            optimizer="muon",
            optimizer_config=OptimizerConfig(1e-4, 5, "quintic"),
        )
        reference = protocol.build_model(config, impl_cfg=impl)
        actual = protocol.build_model(config, impl_cfg=replace(impl, shard_engram=True))
        full, local = reference.chunks[0], actual.chunks[0]
        # Row-dependent nonzero values ensure lookup ownership and backward
        # affect the result; zero-filled tables would hide routing defects.
        for layer_id in full.engram_layer_ids:
            table = full.layers[layer_id].engram.embed
            value = (
                torch.arange(table.weight.numel(), device=rank).reshape(
                    table.weight.shape
                )
                % 7
                - 3
            ).float() / 8
            table.weight.copy_(value.to(table.weight.dtype))
            if table.master is not None:
                table.master.data.copy_(value)
                table.refresh_storage()
        state = full.state_dict()
        for name, value in local.state_dict().items():
            source = state[name]
            if ".engram.embed." in name:
                layer_id = int(name.split(".")[1])
                lookup = local.layers[layer_id].engram.embed.lookup
                source = source[lookup.boundaries[rank] : lookup.boundaries[rank + 1]]
            value.copy_(source)
        for step in range(2):
            ids = torch.arange(
                3 + step, 11 + step + (0 if cp > 1 else rank), device=rank
            )
            batch = PackedBatch(ids, ids, torch.tensor([len(ids)], device=rank))
            logits = []
            for bundle in (reference, actual):
                bundle.optimizer.zero_grad()
                result = bundle.forward_step(bundle.chunks[0], batch)
                logits.append(result["logits"].detach())
                # Use the production normalization and gradient finalization.
                bundle.optimizer.zero_grad()
                run_microbatch_loop(
                    bundle.chunks[0],
                    iter([batch]),
                    1,
                    bundle.forward_step,
                    prepare_microbatches=bundle.extras["prepare_microbatches"],
                )
                if bundle.finalize_grads is not None:
                    bundle.finalize_grads()
            print(
                f"SHARD_STEP rank={rank} step={step} logits_max_abs={float((logits[0] - logits[1]).abs().max())}",
                flush=True,
            )
            torch.testing.assert_close(*logits, atol=0, rtol=0)
            expected = dict(full.named_parameters())
            for name, p in local.named_parameters():
                q = expected[name]
                assert (p.grad is None) == (q.grad is None), name
                if p.grad is not None:
                    grad = q.grad
                    if ".engram.embed.master" in name:
                        lookup = local.layers[
                            int(name.split(".")[1])
                        ].engram.embed.lookup
                        grad = grad[
                            lookup.boundaries[rank] : lookup.boundaries[rank + 1]
                        ]
                    torch.testing.assert_close(
                        p.grad, grad, atol=0, rtol=0, msg=name
                    )
            ref_step, actual_step = reference.optimizer.step(), actual.optimizer.step()
            assert ref_step[0] and actual_step[0]
            assert actual_step[1] == ref_step[1]
            print(
                f"SHARD_NORM rank={rank} step={step} reference={ref_step[1]} sharded={actual_step[1]}",
                flush=True,
            )
            for name, p in local.named_parameters():
                q = expected[name]
                if ".engram.embed.master" in name:
                    lookup = local.layers[int(name.split(".")[1])].engram.embed.lookup
                    q = q[lookup.boundaries[rank] : lookup.boundaries[rank + 1]]
                torch.testing.assert_close(p, q, atol=0, rtol=0, msg=name)
        print(
            f"SHARDED_ENGRAM_OK rank={rank} world={world} ep={ep} cp={cp} trainable={trainable}",
            flush=True,
        )
        from megatron.lite.model.deepseek_v41.lite.checkpoint import export_model
        from megatron.lite.primitive.ckpt import (
            load_training_checkpoint,
            save_training_checkpoint,
        )

        # The active export assembles full rows and all expert owners. It is
        # deliberately distinct from a release export with archival DSpark.
        ref_export = dict(export_model(full))
        for name, value in export_model(local):
            target = ref_export.pop(name)
            if value.element_size() == 1:
                value, target = value.view(torch.uint8), target.view(torch.uint8)
            torch.testing.assert_close(value, target, atol=0, rtol=0, msg=name)
        assert not ref_export
        snapshot = {name: value.clone() for name, value in local.state_dict().items()}
        save_training_checkpoint(local, actual.optimizer, 2, directory, use_dcp=False)
        with torch.no_grad():
            for parameter in local.parameters():
                parameter.zero_()
        assert (
            load_training_checkpoint(local, actual.optimizer, directory, use_dcp=False)
            == 2
        )
        for name, value in local.state_dict().items():
            torch.testing.assert_close(
                value.view(torch.uint8),
                snapshot[name].view(torch.uint8),
                atol=0,
                rtol=0,
            )
        print(f"SHARDED_CHECKPOINT_EXPORT_OK rank={rank}", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(4)
@pytest.mark.parametrize("world,ep,cp", [(2, 1, 1), (4, 2, 1), (2, 1, 2)])
@pytest.mark.parametrize("trainable", [False, True])
def test_sharded_model_matches_replicated_training(
    model_config, tmp_path, trainable, world, ep, cp
):
    assert torch.cuda.device_count() >= world
    torch.multiprocessing.spawn(
        _train_shards,
        args=(model_config, trainable, world, ep, cp, str(tmp_path)),
        nprocs=world,
        join=True,
    )
