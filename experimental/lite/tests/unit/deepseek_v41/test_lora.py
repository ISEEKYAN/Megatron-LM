# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 LoRA gates on the random small model: zero adapter, gradients, checkpoints, SFT, parallel."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _ds41_fixture import tiny_config
from test_training import _build, _initialize, _run, _seed_adapters

from megatron.lite.model.deepseek_v41.lite import lora
from megatron.lite.primitive.modules.lora import LoraConfig, normalize_lora_config
from megatron.lite.primitive.modules.moe_lora import expand_shared_outer

LORA = dict(rank=8, alpha=16, layers=(8, 12))  # the fixture's ratio-one layers


def test_lora_layer_range_defaults_to_the_last_twenty():
    assert lora.lora_layers(LoraConfig(rank=32), 40) == (20, 40)
    assert normalize_lora_config(dict(rank=8, layers=[8, 12])).layers == (8, 12)
    with pytest.raises(ValueError, match="outside"):
        lora.lora_layers(LoraConfig(rank=8, layers=(30, 41)), 40)


def test_serving_expansion_matches_the_per_expert_layout():
    e, h, i, r, scale = 3, 8, 4, 2, 2.0
    a, b, c, d = (torch.randn(s) for s in ((r, h), (e, 2 * i, r), (e, r, i), (h, r)))
    out = expand_shared_outer(a, b, c, d, scale)
    for expert in range(e):  # per expert: A rows expert-major, B column rank * E + expert
        assert torch.equal(out["gate_up.lora_A"].view(e, r, h)[expert], a)
        assert torch.equal(out["gate_up.lora_B"].view(2 * i, r, e)[..., expert], scale * b[expert])
        assert torch.equal(out["down.lora_A"].view(e, r, i)[expert], c[expert])
        assert torch.equal(out["down.lora_B"].view(h, r, e)[..., expert], scale * d)
    state = {
        "layers.9.ffn.experts.lora_gate_up_A.weight": a,
        "layers.9.ffn.experts.lora_down_B.weight": d,
    }
    for expert in range(e):
        state[f"layers.9.ffn.experts.{expert}.lora_gate_up_B.weight"] = b[expert]
        state[f"layers.9.ffn.experts.{expert}.lora_down_A.weight"] = c[expert]
    served = lora.serving_lora_state(state, e, scale)
    assert all(torch.equal(served[f"layers.9.ffn.experts.{k}.weight"], v) for k, v in out.items())


@pytest.mark.gpus(1)
def test_zero_adapter_forward_matches_base(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    base = _run(tmp_path / "base", (1, 1, 1, None))
    for name, config in {"rank0": dict(rank=0), "b0": LORA}.items():
        assert torch.equal(
            base, _run(tmp_path / name, (1, 1, 1, None), impl=dict(lora=config), seed=False)
        )


def _spawn(root, world, case):
    root.mkdir()
    (root / "config.json").write_text(json.dumps(asdict(tiny_config())))
    mp.start_processes(_worker, (str(root), world, case), nprocs=world, start_method="spawn")
    return [json.loads((root / f"rank-{r}.json").read_text()) for r in range(world)]


def _worker(rank, root, world, case):
    from megatron.lite.runtime.contracts import LossContext, PackedBatch

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", f"file://{root}/store", rank=rank, world_size=world)
    record, chunks = {}, None
    try:
        if case == "no_adapter_stage":
            _build(root, pp=2, sft=True, impl=dict(lora=dict(LORA, layers=(9, 12))))
        rt, handle, model, ps = _build(root, ep=world, sft=case == "grad", impl=dict(lora=LORA))
        _initialize(model, ps, root)
        _seed_adapters(model)
        adapters = {n: p for n, p in model.named_parameters() if "lora" in n}
        chunks = [model]
        if case == "grad":
            handle._optimizer.reload_model_params()
            frozen = {
                n: t.clone()
                for n, t in model.state_dict().items()
                if torch.is_tensor(t) and "lora" not in n
            }
            before = {n: p.detach().clone() for n, p in adapters.items()}
            idle = set()  # per-expert factors of routed experts that receive no token
            for key, block in model.layers.items():
                block.ffn.experts.linear_fc1.register_forward_hook(
                    lambda m, args, out, k=key: idle.update(
                        f"layers.{k}.ffn.experts.lora.{f}.weight{i}"
                        for i, n in enumerate(args[1])
                        if int(n) == 0
                        for f in ("fc1_lora_b", "fc2_lora_a")
                    )
                )
            ids = torch.arange(81, device="cuda") % 127 + 1
            batch = PackedBatch(ids, ids, torch.tensor([5, 1, 75], device="cuda"))
            loss = lambda out, batch, ctx=None: (-(out["log_probs"] * out["valid_mask"]).mean(), {})
            with rt.train_mode(handle):
                rt.zero_grad(handle)
                rt.forward_backward(
                    handle, iter([(batch, LossContext(return_log_probs=True))]), loss_fn=loss
                )
            grads = {n: getattr(p, "main_grad", p.grad) for n, p in model.named_parameters()}
            record["trainable"] = sorted(n for n, p in model.named_parameters() if p.requires_grad)
            record["bad_grads"] = [  # idle: exactly zero; otherwise finite and non-zero
                n
                for n in adapters
                if not torch.isfinite(grads[n]).all() or bool(grads[n].abs().sum()) == (n in idle)
            ]
            record["idle"] = sorted(idle)
            optimized = handle._optimizer.optimizers
            record["optimizer_numel"] = sum(
                p.numel() for o in optimized for g in o.param_groups for p in g["params"]
            )
            assert rt.optimizer_step(handle)[0]
            after = model.state_dict()
            record["base_changed"] = [n for n, t in frozen.items() if not torch.equal(t, after[n])]
            record["unmoved"] = [
                n for n, p in adapters.items() if torch.equal(before[n], p) != (n in idle)
            ]
        else:  # checkpoint: LoRA-only DCP and PEFT safetensors, then the resync payload
            saved = {n: p.detach().clone() for n, p in adapters.items()}
            lora.save_lora_dcp(chunks, Path(root, "dcp"), ps)
            lora.save_lora_hf(chunks, Path(root, "hf"), normalize_lora_config(LORA))
            for path, load in (("dcp", lora.load_lora_dcp), ("hf", lora.load_lora_hf)):
                for p in adapters.values():
                    p.data.zero_()
                load(chunks, Path(root, path), *((ps,) if path == "dcp" else ()))
                record[path] = all(torch.equal(saved[n], p) for n, p in adapters.items())
            record["resync"] = sorted(lora.lora_state(model))
        record["adapters"] = sorted(adapters)
        record["adapter_numel"] = sum(p.numel() for p in adapters.values())
    except ValueError as error:
        record["error"] = str(error)
    Path(root, f"rank-{rank}.json").write_text(json.dumps(record))
    dist.destroy_process_group()


@pytest.mark.gpus(1)
def test_gradients_only_on_adapters_and_base_frozen(tmp_path):
    (record,) = _spawn(tmp_path / "grad", 1, "grad")
    assert record["trainable"] == record["adapters"] and len(record["adapters"]) == 4 * (
        16 + 2 + 2 * 8
    )
    assert not record["bad_grads"] and not record["unmoved"], (
        record["bad_grads"],
        record["unmoved"],
    )
    assert not record["base_changed"], record["base_changed"]
    assert record["optimizer_numel"] == record["adapter_numel"]


@pytest.mark.gpus(2)
def test_lora_checkpoints_round_trip_bitwise(tmp_path):
    records = _spawn(tmp_path / "ckpt", 2, "ckpt")
    from safetensors.torch import load_file

    saved = load_file(str(tmp_path / "ckpt" / "hf" / "adapter_model.safetensors"))
    assert all(r["dcp"] and r["hf"] for r in records)
    assert sorted(saved) == sorted({k for r in records for k in r["resync"]})  # 8 experts over EP2
    metadata = json.loads((tmp_path / "ckpt" / "hf" / "adapter_config.json").read_text())
    assert metadata["r"] == 8 and metadata["layers_to_transform"] == [8, 9, 10, 11]


@pytest.mark.gpus(2)
def test_pipeline_stage_without_adapters_fails_loud(tmp_path):
    records = _spawn(tmp_path / "pp", 2, "no_adapter_stage")
    assert all("holds no LoRA layer" in r.get("error", "") for r in records)


@pytest.mark.gpus(1)
def test_lora_sft_loss_decreases(tmp_path):
    impl = dict(lora=LORA, optimizer_config=dict(lr=0.01, min_lr=0.01, weight_decay=0.0))
    _run(tmp_path / "sft", (1, 1, 1, None), sft=True, impl=impl)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(s, marks=pytest.mark.gpus(max(s[0], s[1] * s[2])))
        for s in [(2, 1, 1, None), (1, 2, 1, None), (1, 1, 2, None)]
    ],
)
def test_lora_parallel_forward_matches_single(tmp_path, monkeypatch, shape):
    """The same comparison and tolerance as the base model's parallel contract (#255)."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    impl = dict(lora=LORA)
    reference = _run(
        tmp_path / "reference", (1, shape[1] if shape[2] > 1 else 1, 1, None), impl=impl
    )
    actual = _run(tmp_path / "parallel", shape, impl=impl)
    if shape[1] == 1 or shape[2] > 1:
        error = (reference - actual).abs()
        assert error.max() <= 0.125 and torch.quantile(error, 0.99) <= 0.03
        torch.testing.assert_close(reference.mean(), actual.mean(), atol=0.02, rtol=0.005)
    else:
        (tmp_path / "cp-report.json").write_text(
            json.dumps({"report_only_max_abs": (reference - actual).abs().max().item()})
        )
