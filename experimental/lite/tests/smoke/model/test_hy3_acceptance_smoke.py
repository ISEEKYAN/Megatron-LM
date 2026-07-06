# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]


def _require_cuda_and_te() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Hy3 acceptance smoke tests.")
    pytest.importorskip("transformer_engine.pytorch")


@pytest.fixture(scope="module", autouse=True)
def _cuda_dist():
    _require_cuda_and_te()
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created = False
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")
        created = True
    yield
    if created and dist.is_initialized():
        dist.destroy_process_group()


def _config():
    from megatron.lite.model.hy3.config import Hy3Config

    return Hy3Config(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        intermediate_size=48,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        first_k_dense_replace=1,
        max_position_embeddings=32,
        num_nextn_predict_layers=0,
    )


def _build(*, tp: int = 1, ep: int = 1, cp: int = 1, use_thd: bool = False):
    from megatron.lite.model.hy3.lite import protocol
    from megatron.lite.runtime.contracts.config import ParallelConfig

    impl = protocol.ImplConfig(
        parallel=ParallelConfig(tp=tp, ep=ep, etp=1, pp=1, cp=cp),
        optimizer=None,
        use_deepep=False,
        use_thd=use_thd,
        deterministic=True,
    )
    return protocol.build_model(_config(), impl_cfg=impl), protocol


def _named_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().float().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def test_hy3_real_hf_file_save_load_export_reload(tmp_path: Path):
    if dist.get_world_size() != 1:
        pytest.skip("HF file IO acceptance runs on one rank.")
    from safetensors import safe_open

    from megatron.lite.model.hy3.lite.checkpoint import load_hf_weights, save_hf_weights

    torch.manual_seed(101)
    first, _ = _build()
    torch.manual_seed(202)
    second, _ = _build()
    first_model, second_model = first.chunks[0], second.chunks[0]
    first_before = _named_state(first_model)
    second_before = _named_state(second_model)
    assert any(not torch.equal(first_before[name], second_before[name]) for name in first_before)

    first_dir = tmp_path / "first_hf"
    first_dir.mkdir()
    save_hf_weights(first_model, str(first_dir), _config(), first.parallel_state)
    load_hf_weights(second_model, str(first_dir), _config(), second.parallel_state)

    second_dir = tmp_path / "reexported_hf"
    second_dir.mkdir()
    save_hf_weights(second_model, str(second_dir), _config(), second.parallel_state)

    def read_all(root: Path) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for file in sorted(root.glob("*.safetensors")):
            with safe_open(file, framework="pt") as handle:
                tensors.update({name: handle.get_tensor(name) for name in handle.keys()})
        return tensors

    original = read_all(first_dir)
    reexported = read_all(second_dir)
    assert original
    assert original.keys() == reexported.keys()
    mismatches = [
        name
        for name in original
        if not torch.equal(original[name], reexported[name])
    ]
    assert not mismatches, "HF load/export/reload changed tensors: " + ", ".join(mismatches)


def test_hy3_short_train_reduces_fixed_batch_loss():
    if dist.get_world_size() != 1:
        pytest.skip("short-train acceptance runs on one rank.")
    torch.manual_seed(303)
    bundle, _ = _build()
    model = bundle.chunks[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=0.0)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
    labels = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], device="cuda")
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=labels)["loss"]
        assert torch.isfinite(loss)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 0.5, losses


@pytest.mark.parametrize("topology", [os.environ.get("HY3_TOPOLOGY", "ep2")])
def test_hy3_distributed_forward_backward_uses_requested_paths(topology: str):
    expected_world = {"ep2": 2, "cp2_thd": 2, "cp2_ep2": 4}[topology]
    assert dist.get_world_size() == expected_world
    ep = 2 if topology in {"ep2", "cp2_ep2"} else 1
    cp = 2 if topology in {"cp2_thd", "cp2_ep2"} else 1
    use_thd = cp > 1
    torch.manual_seed(404)
    bundle, _ = _build(ep=ep, cp=cp, use_thd=use_thd)
    model = bundle.chunks[0]
    ps = bundle.parallel_state
    assert (ps.ep_size, ps.cp_size) == (ep, cp)

    sparse = model.layers[1].moe
    assert sparse is not None
    assert sparse.dispatcher.ep_size == ep
    assert not sparse.dispatcher.use_deepep
    path_calls = {"alltoall": 0}
    if ep > 1:
        original = sparse.dispatcher._dispatch_alltoall

        def counted(*args, **kwargs):
            path_calls["alltoall"] += 1
            return original(*args, **kwargs)

        sparse.dispatcher._dispatch_alltoall = counted

    if use_thd:
        from megatron.lite.primitive.parallel import pack_nested_thd

        sequences = [
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device="cuda"),
            torch.tensor([9, 10, 11, 12], device="cuda"),
        ]
        labels = [
            torch.tensor([2, 3, 4, 5, 6, 7, 8, 9], device="cuda"),
            torch.tensor([10, 11, 12, 13], device="cuda"),
        ]
        packed = pack_nested_thd(
            torch.nested.nested_tensor(sequences, layout=torch.jagged),
            cp_size=cp,
            cp_rank=ps.cp_rank,
            cp_group=ps.cp_group,
            labels=torch.nested.nested_tensor(labels, layout=torch.jagged),
        )
        assert packed.packed_seq_params.qkv_format == "thd"
        assert packed.packed_seq_params.local_cp_size == cp
        output = model(
            input_ids=packed.input_ids,
            labels=packed.labels,
            position_ids=packed.position_ids,
            packed_seq_params=packed.packed_seq_params,
        )
    else:
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")
        labels = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], device="cuda")
        output = model(input_ids=input_ids, labels=labels)

    loss = output["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad.float()).all()
        for parameter in model.parameters()
    )
    if ep > 1:
        assert path_calls["alltoall"] == 1
    marker = torch.tensor([1], device="cuda")
    dist.all_reduce(marker)
    assert marker.item() == expected_world
