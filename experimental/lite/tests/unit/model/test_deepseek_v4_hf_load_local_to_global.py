# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4 HF load must lift LOCAL pipeline layer indices to GLOBAL.

Under PP the model's ``self.layers`` ModuleDict is keyed by LOCAL pipeline
position, so a non-first stage's native ``state_dict`` keys carry local indices
(``layers.0`` ...). The HF release is keyed by GLOBAL layer index, so -- exactly
like the exporter -- ``load_hf_weights`` must map local->global via
``to_global_layer_name(name, layer_map)`` before resolving HF names. Without it a
non-first stage reads the wrong global layer's weights.

This is a CPU unit test: a minimal stand-in stage (no GPU/TE) whose layers are
keyed locally but carry ``layer_indices`` = the global ids it owns, plus a tiny
on-disk safetensors keyed by GLOBAL names. ``load_hf_weights`` must copy each
local layer the GLOBAL layer's tensor; pre-fix it resolves local names, finds
nothing, and leaves the params untouched.
"""
from __future__ import annotations

import json
import multiprocessing
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.mlite


class _LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.input_layernorm = _LayerNorm(dim)


class _Stage(nn.Module):
    """A non-first PP stage: layers keyed by LOCAL position, ``layer_indices``
    gives the GLOBAL ids it owns (e.g. [4, 5] for the 3rd stage of pp)."""

    def __init__(self, global_ids: list[int], dim: int):
        super().__init__()
        self.layer_indices = list(global_ids)
        self.layers = nn.ModuleDict({str(i): _Block(dim) for i in range(len(global_ids))})


def _run_ds4_replicated_gloo_load(
    rank: int, world_size: int, init_file: str, checkpoint_dir: str
) -> None:
    import torch.distributed as dist

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite import checkpoint as ckpt

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    original_get_tensor = ckpt.SafeTensorReader.get_tensor
    try:
        if rank != 0:
            ckpt.SafeTensorReader.get_tensor = lambda self, name: (_ for _ in ()).throw(
                AssertionError(f"replica follower unexpectedly read {name}")
            )
        model = _Stage([0], dim=4)
        config = DeepseekV4Config(num_hidden_layers=1, n_routed_experts=8)
        ps = SimpleNamespace(
            tp_size=1,
            etp_size=1,
            ep_size=1,
            ep_rank=0,
            expert_dp_size=world_size,
            dp_cp_group=dist.group.WORLD,
            ep_dp_group=dist.group.WORLD,
        )

        ckpt.load_hf_weights(model, checkpoint_dir, config, ps)

        torch.testing.assert_close(
            model.layers["0"].input_layernorm.weight.detach(),
            torch.full((4,), 11.0),
        )
        dist.barrier()
    finally:
        ckpt.SafeTensorReader.get_tensor = original_get_tensor
        dist.destroy_process_group()


def test_ds4_load_hf_resolves_local_pp_layer_to_global(tmp_path):
    from safetensors.torch import save_file

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite import checkpoint as ckpt

    dim = 4
    global_ids = [4, 5]  # this stage owns global layers 4 and 5, keyed local 0 and 1
    model = _Stage(global_ids, dim)
    cfg = DeepseekV4Config(num_hidden_layers=8, n_routed_experts=8)
    ps = SimpleNamespace(tp_size=1, etp_size=1, ep_size=1, ep_rank=0)

    # Real-release layout is keyed by GLOBAL layer index; input_layernorm maps to
    # the bare V4-Flash ``attn_norm.weight``.
    save_file(
        {f"layers.{g}.attn_norm.weight": torch.full((dim,), float(g)) for g in global_ids},
        str(tmp_path / "model.safetensors"),
    )

    ckpt.load_hf_weights(model, str(tmp_path), cfg, ps)

    # local layer 0 -> global 4 -> filled with 4.0; local 1 -> global 5 -> 5.0.
    # Pre-fix, load resolved layers.0/layers.1 (local), found nothing, left zeros.
    torch.testing.assert_close(
        model.layers["0"].input_layernorm.weight.detach(), torch.full((dim,), 4.0)
    )
    torch.testing.assert_close(
        model.layers["1"].input_layernorm.weight.detach(), torch.full((dim,), 5.0)
    )


def test_ds4_dense_replica_load_reads_once_across_real_gloo_group(tmp_path) -> None:
    from safetensors.torch import save_file

    shard = "model-00001-of-00001.safetensors"
    name = "layers.0.attn_norm.weight"
    save_file({name: torch.full((4,), 11.0)}, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard}})
    )

    context = multiprocessing.get_context("spawn")
    init_file = str(tmp_path / "gloo-init")
    processes = [
        context.Process(
            target=_run_ds4_replicated_gloo_load,
            args=(rank, 2, init_file, str(tmp_path)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert [process.exitcode for process in processes] == [0, 0]


def test_ds4_scaled_copy_moves_quantized_inputs_before_dequant(monkeypatch) -> None:
    from megatron.lite.model.deepseek_v4.lite import checkpoint as ckpt

    target = nn.Parameter(torch.empty(4, dtype=torch.bfloat16, device="meta"))
    quantized = torch.ones(2, dtype=torch.int8)
    scale = torch.ones(1, dtype=torch.uint8)
    calls = []

    def fake_dequant(tensor, block_scale, shape):
        calls.append((tensor.device.type, block_scale.device.type, tuple(shape)))
        return torch.empty(shape, dtype=torch.float32, device=tensor.device)

    monkeypatch.setattr(ckpt, "_dequantize_scaled_tensor", fake_dequant)

    ckpt._copy_param(target, quantized, scale=scale)

    assert calls == [("meta", "meta", (4,))]


def test_ds4_uint8_scale_conversion_stays_on_scale_device() -> None:
    from megatron.lite.model.deepseek_v4.lite import checkpoint as ckpt

    converted = ckpt._scale_to_float(torch.empty(2, dtype=torch.uint8, device="meta"))

    assert converted.device.type == "meta"


def test_ds4_fused_hf_weights_copy_directly_into_target_slices(monkeypatch) -> None:
    from megatron.lite.model.deepseek_v4.lite import checkpoint as ckpt

    class Reader:
        index = {"first.weight": "a", "second.weight": "b"}

        @staticmethod
        def get_tensor(name: str) -> torch.Tensor:
            values = {
                "first.weight": torch.full((2, 3), 1.0),
                "second.weight": torch.full((2, 3), 2.0),
            }
            return values[name]

    monkeypatch.setattr(
        ckpt.torch,
        "cat",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "fused checkpoint weights must not materialize a concatenation"
            )
        ),
    )
    target = torch.zeros(4, 3)

    ckpt._copy_hf_tensors(Reader(), target, ["first.weight", "second.weight"])

    torch.testing.assert_close(target[:2], torch.ones(2, 3))
    torch.testing.assert_close(target[2:], torch.full((2, 3), 2.0))


def test_ds4_replica_reader_uses_group_local_root_and_broadcasts(monkeypatch) -> None:
    from megatron.lite.model.deepseek_v4.lite import checkpoint as ckpt

    class Reader:
        index = {"weight": "shard"}

        def __init__(self) -> None:
            self.reads = []

        def get_tensor(self, name: str) -> torch.Tensor:
            self.reads.append(name)
            return torch.full((2, 2), 7.0)

    broadcasts = []
    monkeypatch.setattr(ckpt.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ckpt.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(ckpt.dist, "get_rank", lambda group: 2)
    monkeypatch.setattr(
        ckpt.dist, "get_process_group_ranks", lambda group: [64, 72, 80, 88]
    )
    monkeypatch.setattr(
        ckpt.dist,
        "broadcast",
        lambda tensor, src, group: broadcasts.append((tensor.clone(), src, group)),
    )
    reader = Reader()
    target = torch.zeros(2, 2)

    assert ckpt._copy_replicated_hf_tensors(
        reader,
        target,
        ["weight"],
        group="expert",
        source_group_rank=2,
    )

    assert reader.reads == ["weight"]
    torch.testing.assert_close(target, torch.full((2, 2), 7.0))
    assert len(broadcasts) == 1
    assert broadcasts[0][1:] == (80, "expert")


def test_ds4_replica_follower_never_reads_checkpoint(monkeypatch) -> None:
    from megatron.lite.model.deepseek_v4.lite import checkpoint as ckpt

    class Reader:
        index = {"weight": "shard"}

        @staticmethod
        def get_tensor(name: str) -> torch.Tensor:
            raise AssertionError(f"replica follower must not read {name}")

    monkeypatch.setattr(ckpt.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ckpt.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(ckpt.dist, "get_rank", lambda group: 2)
    monkeypatch.setattr(
        ckpt.dist, "get_process_group_ranks", lambda group: [96, 104, 112, 120]
    )

    def fake_broadcast(tensor, src, group) -> None:
        assert (src, group) == (96, "expert")
        tensor.fill_(9.0)

    monkeypatch.setattr(ckpt.dist, "broadcast", fake_broadcast)
    target = torch.zeros(2, 2)

    assert not ckpt._copy_replicated_hf_tensors(
        Reader(), target, ["weight"], group="expert"
    )

    torch.testing.assert_close(target, torch.full((2, 2), 9.0))


def test_ds4_replica_group_matches_fsdp_parameter_ownership() -> None:
    from megatron.lite.model.deepseek_v4.lite import checkpoint as ckpt

    ps = SimpleNamespace(
        dp_cp_group="dense",
        ep_dp_group="expert",
        ep_rank=6,
        expert_dp_size=4,
    )

    assert (
        ckpt._replica_group_for_state_key("layers.0.input_layernorm.weight", ps)
        == "dense"
    )
    assert (
        ckpt._replica_group_for_state_key("layers.0.mlp.shared_experts.down.weight", ps)
        == "dense"
    )
    assert (
        ckpt._replica_group_for_state_key("layers.0.mlp.experts.fc1.weight3", ps)
        == "expert"
    )
    assert ckpt._replica_source_group_rank("layers.0.input_layernorm.weight", ps) == 0
    assert ckpt._replica_source_group_rank("layers.0.mlp.experts.fc1.weight3", ps) == 2
