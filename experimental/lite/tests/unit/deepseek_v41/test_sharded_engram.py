# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
import torch.distributed as dist
from megatron.lite.primitive.parallel.state import ParallelState


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
            model_config, parallel_state=ps, trainable_engram=trainable, quantized=False
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


def test_protocol_defaults_to_owner_sharding():
    from megatron.lite.model.deepseek_v41.lite.protocol import ImplConfig

    assert ImplConfig().shard_engram


@pytest.mark.parametrize("trainable", [False, True])
def test_official_engram_storage_is_local(moe, model_config, monkeypatch, trainable):
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    config = model_config.to_hf_dict()
    totals = [384006168, 384016682]
    config['text_config'].update(engram_num_embeddings=totals, engram_head_dim=256)
    group, owners, rank = object(), 8, 7
    monkeypatch.setattr(dist, 'get_world_size', lambda g: owners)
    monkeypatch.setattr(dist, 'get_rank', lambda g: rank)
    ps = ParallelState(dp_cp_group=group, dp_cp_size=owners, dp_cp_rank=rank)
    with torch.device('meta'):
        model = DeepseekV41Model(
            DeepseekV41Config(config),
            parallel_state=ps,
            trainable_engram=trainable,
            quantized=False,
        )
    for layer, total in zip(model.engram_layer_ids, totals):
        table = model.layers[layer].engram.embed
        rows = total - total * rank // owners
        assert table.weight.shape == (rows, 256)
        assert table.scale.shape == (rows, 8)
        assert (table.master is not None) == trainable
        if trainable:
            assert table.master.shape == (rows, 256)


def _gather_reference_rows(value, counts, group):
    width = max(counts)
    padded = value.new_zeros((width, *value.shape[1:]))
    padded[: value.shape[0]].copy_(value)
    wire = padded.view(torch.uint8) if value.element_size() == 1 else padded
    output = [torch.empty_like(wire) for _ in counts]
    dist.all_gather(output, wire, group=group)
    return torch.cat(
        [part.view(value.dtype)[:count] for part, count in zip(output, counts)]
    )


class _ReferenceRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, master, ids, boundaries, group):
        counts = [None] * dist.get_world_size(group)
        dist.all_gather_object(counts, ids.numel(), group=group)
        all_ids = _gather_reference_rows(ids.flatten(), counts, group)
        rank = dist.get_rank(group)
        begin, end = boundaries[rank : rank + 2]
        owned = (all_ids >= begin) & (all_ids < end)
        ctx.save_for_backward(master, all_ids[owned] - begin, owned)
        ctx.counts, ctx.group = counts, group
        rows = [b - a for a, b in zip(boundaries, boundaries[1:])]
        return _gather_reference_rows(master, rows, group)[ids]

    @staticmethod
    def backward(ctx, gradient):
        master, ids, owned = ctx.saved_tensors
        incoming = _gather_reference_rows(
            gradient.reshape(-1, master.shape[1]), ctx.counts, ctx.group
        )[owned]
        # Match the physical index-backward shape and rank/query order while
        # obtaining requests by all-gather, independently of all-to-all routing.
        with torch.enable_grad():
            local = master.detach().requires_grad_()
            result = torch.autograd.grad(local[ids], local, incoming)[0]
        return result, None, None, None


def _reference_lookup(table, ids):
    lookup = table.lookup
    rows = [b - a for a, b in zip(lookup.boundaries, lookup.boundaries[1:])]
    values = _gather_reference_rows(table.weight, rows, lookup.group)
    scales = _gather_reference_rows(table.scale, rows, lookup.group)
    return (
        values.view(torch.uint8)[ids].view(values.dtype),
        scales.view(torch.uint8)[ids].view(scales.dtype),
        _ReferenceRows.apply(table.master, ids, lookup.boundaries, lookup.group),
    )


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
        reference = protocol.build_model(
            config, impl_cfg=replace(impl, shard_engram=trainable)
        )
        actual = protocol.build_model(config, impl_cfg=replace(impl, shard_engram=True))
        full, local = reference.chunks[0], actual.chunks[0]
        if trainable:
            from types import MethodType

            def forbidden(*args, **kwargs):
                raise AssertionError('Independent reference must not call RowLookup')

            # Keep physical row/index-backward/Sinkhorn and DDP bucket shapes
            # equal. The oracle gathers the logical table and all requests;
            # production exchanges only owner requests through all-to-all.
            for index in full.engram_layer_ids:
                table = full.layers[index].engram.embed
                table.lookup_fp8 = MethodType(_reference_lookup, table)
                table.lookup.fetch = forbidden
        # Row-dependent nonzero values ensure lookup ownership and backward
        # affect the result; zero-filled tables would hide routing defects.
        for layer_id in full.engram_layer_ids:
            table = full.layers[layer_id].engram.embed
            value = (
                (
                    torch.arange(table.weight.numel(), device=rank)
                    + (
                        table.lookup.boundaries[rank] * table.weight.shape[1]
                        if trainable
                        else 0
                    )
                ).reshape(table.weight.shape)
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
            if ".engram.embed." in name and not trainable:
                layer_id = int(name.split(".")[1])
                lookup = local.layers[layer_id].engram.embed.lookup
                source = source[lookup.boundaries[rank] : lookup.boundaries[rank + 1]]
            value.copy_(source)
        for step in range(2):
            ids = torch.arange(
                3 + step, 11 + step + (0 if cp > 1 else rank), device=rank
            )
            batch = PackedBatch(ids, ids, torch.tensor([len(ids)], device=rank))
            logits, losses = [], []
            for bundle in (reference, actual):
                bundle.optimizer.zero_grad()
                result = bundle.forward_step(bundle.chunks[0], batch)
                logits.append(result["logits"].detach())
                losses.append(result["loss"].detach())
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
            torch.testing.assert_close(*losses, atol=0, rtol=0)
            expected = dict(full.named_parameters())
            for layer_id in local.engram_layer_ids:
                table = local.layers[layer_id].engram.embed
                counts = [
                    b - a
                    for a, b in zip(
                        table.lookup.boundaries, table.lookup.boundaries[1:]
                    )
                ]
                assert table.weight.shape[0] == counts[rank]
                if trainable:
                    assert table.master.grad.shape == table.master.shape
                    assert (
                        table.master.main_grad.data_ptr()
                        == table.master.grad.data_ptr()
                    )
                    assert (
                        table.master.grad.untyped_storage().nbytes()
                        == table.master.numel() * 4
                    )
                print(
                    f"OWNER_STORAGE rank={rank} layer={layer_id} rows={counts[rank]} "
                    f"fp8_scale_bytes={table.weight.numel() + table.scale.numel()} "
                    f"master_bytes={0 if table.master is None else table.master.numel() * 4} "
                    f"main_grad_bytes={0 if table.master is None else table.master.grad.numel() * 4}",
                    flush=True,
                )
            for name, p in local.named_parameters():
                q = expected[name]
                assert (p.grad is None) == (q.grad is None), name
                if p.grad is not None:
                    grad = q.grad
                    torch.testing.assert_close(p.grad, grad, atol=0, rtol=0, msg=name)
            ref_step, actual_step = reference.optimizer.step(), actual.optimizer.step()
            assert ref_step[0] and actual_step[0]
            assert actual_step[1] == ref_step[1]
            print(
                f"SHARD_NORM rank={rank} step={step} reference={ref_step[1]} sharded={actual_step[1]}",
                flush=True,
            )
            for name, p in local.named_parameters():
                q = expected[name]
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
        import json

        from megatron.lite.model.deepseek_v41.lite.checkpoint import load_model
        from megatron.lite.primitive.ckpt.hf_weights import stream_export_to_shards

        destination = Path(directory) / 'active_export'
        stream_export_to_shards(
            ((name, value.cpu()) for name, value in export_model(local)),
            str(destination),
        )
        if rank == 0:
            (destination / 'config.json').write_text(
                json.dumps(local.config.to_hf_dict())
            )
        dist.barrier()
        with torch.no_grad():
            for parameter in local.parameters():
                parameter.zero_()
        load_model(local, destination, allow_missing_mtp=True)
        for name, value in local.state_dict().items():
            torch.testing.assert_close(
                value.reshape(-1).view(torch.uint8),
                snapshot[name].reshape(-1).view(torch.uint8),
                atol=0,
                rtol=0,
                msg=name,
            )
        print(f"SHARDED_CHECKPOINT_EXPORT_RELOAD_OK rank={rank}", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(4)
@pytest.mark.parametrize("world,ep,cp", [(2, 1, 1), (4, 2, 1), (2, 1, 2)])
@pytest.mark.parametrize("trainable", [False, True])
def test_sharded_model_matches_layout_matched_reference(
    model_config, tmp_path, trainable, world, ep, cp
):
    assert torch.cuda.device_count() >= world
    torch.multiprocessing.spawn(
        _train_shards,
        args=(model_config, trainable, world, ep, cp, str(tmp_path)),
        nprocs=world,
        join=True,
    )
