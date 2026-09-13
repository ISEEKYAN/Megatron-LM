# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from megatron.lite.primitive.recompute import wrap_checkpoint
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _step(
    model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor
):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.detach()


def _clone_model_and_optimizer(model: nn.Module):
    clone = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(clone.parameters(), lr=1.0e-3, weight_decay=0.0)
    return clone, optimizer


def _assert_model_close(lhs: nn.Module, rhs: nn.Module):
    for (lhs_name, lhs_param), (rhs_name, rhs_param) in zip(
        lhs.named_parameters(), rhs.named_parameters(), strict=True
    ):
        assert lhs_name == rhs_name
        torch.testing.assert_close(lhs_param, rhs_param, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "case",
    ["keyword_only", "pos_kw_alias", "duplicate_pos", "duplicate_kwargs", "metadata"],
)
def test_recompute_kwargs_parity(case):
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(1, 5, dtype=torch.float64))

        def forward(self, x=None, duplicate=None, *, hidden_states, residual, metadata):
            assert metadata["label"] == "packed" and metadata["optional"] is None
            assert metadata["scale"] == 2, "non_tensor_metadata_preserved"
            value = hidden_states * metadata["scale"] + residual * 3
            if "nested" in metadata:
                value = value + metadata["nested"][0]
            if x is not None:
                value = value + x * 5
            if duplicate is not None:
                value = value + duplicate * 7
            return value * self.weight

    direct, checkpointed = Layer(), Layer()
    wrap_checkpoint(checkpointed, preserve_rng_state=False)
    results = []
    for model in (direct, checkpointed):
        source = torch.arange(1, 5, dtype=torch.float64, requires_grad=True)
        other = torch.arange(5, 9, dtype=torch.float64, requires_grad=True)
        hidden, residual = (
            source * 2,
            other * 4,
        )  # Non-leaves expose captured-graph bugs.
        args = (hidden,) if case == "pos_kw_alias" else ()
        if case == "duplicate_pos":
            args = (hidden, hidden)
        if case == "duplicate_kwargs":
            residual = hidden
        metadata = {"label": "packed", "optional": None, "scale": 2}
        if case == "metadata":
            metadata["positions"] = torch.arange(4)
            metadata["nested"] = [hidden.view(4)]
        output = model(
            *args, hidden_states=hidden, residual=residual, metadata=metadata
        )
        assert output.requires_grad, "keyword_tensor_has_backward_edge"
        output.sum().backward()
        assert source.grad is not None, "keyword_source_gradient_present"
        results.append((output.detach(), source.grad, other.grad, model.weight.grad))
    for name, expected, actual in zip(
        ("output", "alias_source_gradient", "residual_gradient", "parameter_gradient"),
        *results,
        strict=True,
    ):
        if expected is None:
            assert actual is None, name
        else:
            assert actual is not None, name
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)


@pytest.mark.gpus(1)
@pytest.mark.parametrize(
    "preserve,stochastic", [(True, True), (False, False), (False, True)]
)
def test_recompute_kwargs_rng_parity(preserve, stochastic):
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.arange(1, 5, device="cuda", dtype=torch.float64)
            )
            self.outputs = []

        def forward(self, *, hidden_states):
            value = hidden_states * self.weight
            if stochastic:
                value = (
                    value * torch.rand_like(value) * torch.rand((), device="cpu").item()
                )
            self.outputs.append(value.detach().clone())
            return value

    results = []
    for checkpointed in (False, True):
        torch.manual_seed(2031)
        torch.cuda.manual_seed_all(2031)
        model = Layer()
        if checkpointed:
            wrap_checkpoint(model, preserve_rng_state=preserve)
        source = torch.arange(
            1, 5, device="cuda", dtype=torch.float64, requires_grad=True
        )
        output = model(hidden_states=source * 2)
        assert output.requires_grad, "rng_keyword_tensor_has_backward_edge"
        # Without preservation, stochastic replay uses the NEXT RNG draw. Compare its
        # gradients to an ordinary forward at that state, not to the first dropout mask.
        if not checkpointed and not preserve and stochastic:
            output = model(hidden_states=source * 2)
        output.sum().backward()
        results.append(
            (
                model.outputs,
                source.grad,
                model.weight.grad,
                torch.get_rng_state(),
                torch.cuda.get_rng_state(),
            )
        )
    direct, checkpointed = results
    for name, actual, expected in (
        ("rng_forward", checkpointed[0][0], direct[0][0]),
        ("rng_replay", checkpointed[0][1], direct[0][-1]),
        ("rng_source_gradient", checkpointed[1], direct[1]),
        ("rng_parameter_gradient", checkpointed[2], direct[2]),
        ("cpu_rng_state", checkpointed[3], direct[3]),
        ("cuda_rng_state", checkpointed[4], direct[4]),
    ):
        assert actual is not None, name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)


def test_runtime_checkpoint_load_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(2029)
    base = TinyMLP()
    ckpt_model, ckpt_optimizer = _clone_model_and_optimizer(base)
    direct_model, direct_optimizer = _clone_model_and_optimizer(base)
    loaded_model, loaded_optimizer = _clone_model_and_optimizer(base)
    for model in (ckpt_model, loaded_model):
        wrap_checkpoint(model.layers[2], preserve_rng_state=False)
    x0, y0 = torch.randn(3, 4), torch.randn(3, 2)
    x1, y1 = torch.randn(3, 4), torch.randn(3, 2)

    _step(ckpt_model, ckpt_optimizer, x0, y0)
    _step(direct_model, direct_optimizer, x0, y0)

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    ckpt_handle = ModelHandle(
        model=ckpt_model,
        optimizer=ckpt_optimizer,
        _extras={"model_chunks": [ckpt_model]},
    )
    runtime.save_checkpoint(ckpt_handle, str(tmp_path), step=1, use_dcp=False)

    loaded_handle = ModelHandle(
        model=loaded_model,
        optimizer=loaded_optimizer,
        _extras={"model_chunks": [loaded_model]},
    )
    assert runtime.load_checkpoint(loaded_handle, str(tmp_path), use_dcp=False) == 1

    _step(direct_model, direct_optimizer, x1, y1)
    _step(loaded_model, loaded_optimizer, x1, y1)

    _assert_model_close(direct_model, loaded_model)


class DistOptLike:
    """Small optimizer wrapper with the same checkpoint contract as dist_opt."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.load_calls = 0
        self.parameter_save_calls = 0
        self.parameter_load_calls = 0

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()
        return True, 0.0, 0

    def state_dict(self):
        state = self.optimizer.state_dict()
        state["dist_opt_like_marker"] = {"load_calls": self.load_calls}
        return state

    def load_state_dict(self, state):
        marker = state.pop("dist_opt_like_marker")
        self.load_calls = int(marker["load_calls"]) + 1
        self.optimizer.load_state_dict(state)

    def save_parameter_state(self, filename: str):
        self.parameter_save_calls += 1
        torch.save({"parameter_save_calls": self.parameter_save_calls}, filename)

    def load_parameter_state(
        self, filename: str, *, update_legacy_format: bool = False
    ):
        state = torch.load(filename)
        self.parameter_load_calls = int(state["parameter_save_calls"])


def test_runtime_checkpoint_uses_optimizer_state_dict_contract(tmp_path):
    torch.manual_seed(2030)
    model = TinyMLP()
    optimizer = DistOptLike(torch.optim.AdamW(model.parameters(), lr=1.0e-3))
    x, y = torch.randn(3, 4), torch.randn(3, 2)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(model(x), y).backward()
    optimizer.step()

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.save_checkpoint(
        ModelHandle(
            model=model, optimizer=optimizer, _extras={"model_chunks": [model]}
        ),
        str(tmp_path),
        step=7,
        use_dcp=False,
    )

    loaded_model = TinyMLP()
    loaded_optimizer = DistOptLike(
        torch.optim.AdamW(loaded_model.parameters(), lr=1.0e-3)
    )
    loaded_handle = ModelHandle(
        model=loaded_model,
        optimizer=loaded_optimizer,
        _extras={"model_chunks": [loaded_model]},
    )

    assert runtime.load_checkpoint(loaded_handle, str(tmp_path), use_dcp=False) == 7
    assert loaded_optimizer.load_calls == 1
    assert loaded_optimizer.parameter_load_calls == 1
    assert (tmp_path / "training_state.optimizer_parameter_state.pt").exists()
    _assert_model_close(model, loaded_model)
