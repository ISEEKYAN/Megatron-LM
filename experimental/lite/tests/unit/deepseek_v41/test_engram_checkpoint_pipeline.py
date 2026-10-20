# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 Engram, checkpoint and pipeline state transport contracts."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from megatron.lite.model.deepseek_v41.lite.checkpoint import load_hf_weights
from megatron.lite.model.deepseek_v41.lite.protocol import forward_step
from megatron.lite.runtime.contracts import PackedBatch


@pytest.mark.parametrize("rank", [0, 1])
def test_forward_step_rows_are_original_unpadded_rows(rank):
    class Rows(torch.nn.Module):
        ps = SimpleNamespace(cp_size=2, cp_rank=rank, cp_group=object())

        def forward(self, input_ids, loss_mask, **_):  # physical rows, as the model reports
            n = input_ids.numel()
            rows = torch.arange(n) + rank * n
            return {"log_probs": torch.zeros(n), "rows": rows, "valid_mask": loss_mask.bool()}

    ids = torch.arange(12)
    output = forward_step(Rows(), PackedBatch(ids, ids, torch.tensor([5, 1, 6])))
    expected = [0, 1, 2, 3, 4, -1, -1, -1, 5, -1] if rank == 0 else [-1, -1, *range(6, 12), -1, -1]
    assert output["rows"].tolist() == expected
    assert not output["valid_mask"][output["rows"] < 0].any()
    assert output["valid_mask"].sum() == (4 if rank == 0 else 5)


class _TokenLogProbs(torch.nn.Module):
    """log_prob of a token = weight[token id]; the loss is the local mean, as both DS4.1 models."""

    def __init__(self, weight, ps):
        super().__init__()
        self.weight, self.ps = weight, ps

    def forward(self, input_ids, loss_mask, **_):
        log_probs = self.weight[input_ids.reshape(-1)]
        mask = loss_mask.reshape(-1).to(log_probs.dtype)
        return {"log_probs": log_probs, "loss": -(log_probs * mask).sum() / mask.sum().clamp_min(1)}


def _cp_loss_worker(rank, path, results, denominator):
    import torch.distributed as dist

    from megatron.lite.runtime.contracts.loss import LossContext, use_loss_context

    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=2)
    try:
        weight = torch.linspace(-3, -1, 12, dtype=torch.float64).requires_grad_()
        ps = SimpleNamespace(cp_size=2, cp_rank=rank, cp_group=dist.group.WORLD)
        ids = torch.arange(12)
        with use_loss_context(LossContext(normalization_denominator=denominator)):
            loss = forward_step(_TokenLogProbs(weight, ps), PackedBatch(ids, ids, torch.tensor([5, 1, 6])))[
                "loss"
            ]
        loss.backward()
        results[rank] = (loss.detach(), weight.grad.clone())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("denominator", [None, 7.0])
def test_forward_step_cp_loss_and_grads_match_cp1(tmp_path, denominator):
    """DDP averages over CP ranks: the CP2 mean of loss and gradients must equal CP=1."""
    import torch.multiprocessing as mp

    from megatron.lite.runtime.contracts.loss import LossContext, use_loss_context

    weight = torch.linspace(-3, -1, 12, dtype=torch.float64).requires_grad_()
    ids = torch.arange(12)
    ps = SimpleNamespace(cp_size=1, cp_rank=0, cp_group=None)
    with use_loss_context(LossContext(normalization_denominator=denominator)):
        expected = forward_step(_TokenLogProbs(weight, ps), PackedBatch(ids, ids, torch.tensor([5, 1, 6])))[
            "loss"
        ]
    expected.backward()

    results = mp.Manager().dict()
    mp.spawn(_cp_loss_worker, args=(str(tmp_path / "store"), results, denominator), nprocs=2)
    loss = (results[0][0] + results[1][0]) / 2
    grad = (results[0][1] + results[1][1]) / 2
    torch.testing.assert_close(loss, expected.detach(), rtol=1e-12, atol=0)
    torch.testing.assert_close(grad, weight.grad, rtol=1e-12, atol=0)
    assert weight.grad.ne(0).sum() == 9  # 12 tokens, each document's last token is unlabelled


def test_release_load_fails_on_unread_stage_keys_or_unloaded_state(tmp_path):
    names = [
        "layers.0.attn_norm.weight",
        "layers.1.attn_norm.weight",
        "layers.0.ffn.gate.bias_vl",
        "mtp.x",
    ]
    tensors = {n: torch.ones(4, dtype=torch.bfloat16) for n in names}
    tensors["embed.weight"] = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
    tensors["embed.weight_scale_inv"] = torch.full((1, 1), 2.0)
    save_file(tensors, tmp_path / "a.safetensors")
    names += ["embed.weight", "embed.weight_scale_inv"]
    index = {"weight_map": {n: "a.safetensors" for n in names}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    def stage():  # first stage holding layer 0; layer 1 and MTP belong elsewhere
        model, layer = torch.nn.Module(), torch.nn.Module()
        layer.attn_norm = torch.nn.Module()
        layer.attn_norm.weight = torch.nn.Parameter(torch.zeros(4))
        model.embed, model.embed.weight = torch.nn.Module(), torch.nn.Parameter(torch.zeros(2, 4))
        model.layers, model.local_layer_range = torch.nn.ModuleDict({"0": layer}), (0, 1)
        model.layout = SimpleNamespace(has_embed=True, has_head=False)
        return model

    cfg, ps = SimpleNamespace(n_routed_experts=1), SimpleNamespace(ep_size=1, ep_rank=0)
    model = stage()
    load_hf_weights(model, str(tmp_path), cfg, ps)
    assert torch.equal(model.layers["0"].attn_norm.weight, torch.ones(4))
    assert torch.equal(model.embed.weight, torch.full((2, 4), 2.0))  # dequantized
    from megatron.lite.model.deepseek_v41.lite.checkpoint import _sharded_state

    unequal = torch.nn.Module()
    unequal.lookup = SimpleNamespace(
        group=SimpleNamespace(size=lambda: 2, rank=lambda: 1), boundaries=(0, 2, 5)
    )
    with pytest.raises(ValueError, match="equal Core row shards"):
        _sharded_state(unequal, "", (), None, {})
    unread = stage()
    del unread.embed  # a stage-local release key nobody reads
    with pytest.raises(ValueError, match=r"unread \['embed.weight'\]"):
        load_hf_weights(unread, str(tmp_path), cfg, ps)
    unloaded = stage()
    unloaded.layers["0"].register_buffer("state", torch.zeros(1))  # persistent, unmapped
    with pytest.raises(ValueError, match=r"unloaded \['layers.0.state'\]"):
        load_hf_weights(unloaded, str(tmp_path), cfg, ps)

    key = "layers.0.ffn.gate.bias_vl_extra"  # near-match must not be ignored
    tensors[key] = torch.ones(1)
    save_file(tensors, tmp_path / "a.safetensors")
    index["weight_map"][key] = "a.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="unread.*bias_vl_extra"):
        load_hf_weights(stage(), str(tmp_path), cfg, ps)


def test_vision_router_bias_loads_bitwise_and_counts_as_read(tmp_path):
    """A model that routes image sentinels reads ffn.gate.bias_vl as exact FP32."""
    bias_vl = torch.tensor([1 + 2**-20, -3.0e-7, 0.125, 7.0], dtype=torch.float32)
    tensors = {
        "layers.0.ffn.gate.weight": torch.ones(4, 2, dtype=torch.bfloat16),
        "layers.0.ffn.gate.bias": torch.full((4,), 0.5),
        "layers.0.ffn.gate.bias_vl": bias_vl,
    }
    save_file(tensors, tmp_path / "a.safetensors")
    index = {"weight_map": {n: "a.safetensors" for n in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    model, layer = torch.nn.Module(), torch.nn.Module()
    layer.ffn = torch.nn.Module()
    layer.ffn.router = torch.nn.Module()
    layer.ffn.router.weight = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.bfloat16))
    layer.ffn.router.register_buffer("expert_bias", torch.zeros(4))
    layer.ffn.router.register_buffer("bias_vl", torch.zeros(4))
    model.layers, model.local_layer_range = torch.nn.ModuleDict({"0": layer}), (0, 1)
    model.layout = SimpleNamespace(has_embed=False, has_head=False)
    cfg, ps = SimpleNamespace(n_routed_experts=1), SimpleNamespace(ep_size=1, ep_rank=0)
    load_hf_weights(model, str(tmp_path), cfg, ps)
    router = model.layers["0"].ffn.router
    assert torch.equal(router.bias_vl.view(torch.int32), bias_vl.view(torch.int32))
    assert torch.equal(router.expert_bias, torch.full((4,), 0.5))


@pytest.mark.gpus(1)
def test_native_fp4_fp8_gpu_decode_matches_cpu_bits(tmp_path):
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

    codes = torch.arange(256, dtype=torch.uint8).repeat(2)
    codes[(codes & 127) == 127] = 0  # every finite E4M3 byte, exclude the two NaNs
    tensors = {
        "expert.weight": torch.arange(256, dtype=torch.uint8).view(torch.int8).reshape(8, 32),
        "expert.scale": torch.arange(120, 136, dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .reshape(8, 2),
        "attention.weight": codes.view(torch.float8_e4m3fn).reshape(8, 64),
        "attention.scale": torch.tensor([[120, 125]], dtype=torch.uint8).view(torch.float8_e8m0fnu),
    }
    save_file(tensors, tmp_path / "model.safetensors")
    with SafeTensorReader(tmp_path) as reader:
        for key in ("expert.weight", "attention.weight"):
            args = dict(target_shape=torch.Size((8, 64)), target_dtype=torch.bfloat16)
            cpu = reader.get_tensor(key, device="cpu", **args)
            gpu = reader.get_tensor(key, device="cuda", **args)
            for dtype in (torch.float32, torch.bfloat16):
                assert torch.equal(
                    cpu.to(dtype).view(torch.uint8), gpu.to(dtype).cpu().view(torch.uint8)
                )


def test_native_row_reads_keep_owned_bytes_across_shards(tmp_path):
    from megatron.lite.primitive.ckpt.row_reader import RowSliceReader as SafeTensorReader

    tensors = {
        "weight": torch.arange(128, dtype=torch.uint8).reshape(8, 16).view(torch.float8_e4m3fn),
        "scale": torch.arange(16, dtype=torch.uint8).reshape(8, 2).view(torch.float8_e8m0fnu),
    }
    for key, value in tensors.items():
        save_file({key: value}, tmp_path / f"{key}.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: f"{k}.safetensors" for k in tensors}})
    )
    with SafeTensorReader(tmp_path) as reader:
        rows = reader.get_row_tensors([(k, 3, 6) for k in tensors])
    for key, value in rows.items():
        assert value.dtype == tensors[key].dtype and value.shape[0] == 3
        assert torch.equal(value.view(torch.uint8), tensors[key][3:6].view(torch.uint8))
