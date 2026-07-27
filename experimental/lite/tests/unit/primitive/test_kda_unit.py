# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

pytestmark = pytest.mark.mlite


def _kda_ops():
    from megatron.lite.primitive.ops.kda import kda, torch_recurrent_kda

    return kda, torch_recurrent_kda


def _inputs(*, requires_grad: bool = False):
    torch.manual_seed(7)
    shape = (2, 3, 2, 4)
    q = torch.randn(shape, requires_grad=requires_grad)
    k = torch.randn(shape, requires_grad=requires_grad)
    v = torch.randn(shape, requires_grad=requires_grad)
    gate = torch.randn(shape, requires_grad=requires_grad)
    beta = torch.randn(shape[:-1], requires_grad=requires_grad)
    a_log = torch.randn(shape[2], requires_grad=requires_grad)
    dt_bias = torch.randn(shape[2:], requires_grad=requires_grad)
    return q, k, v, gate, beta, a_log, dt_bias


def test_kda_cpu_auto_matches_recurrent_reference():
    kda, torch_recurrent_kda = _kda_ops()
    q, k, v, gate, beta, a_log, dt_bias = _inputs()
    kwargs = {
        "a_log": a_log,
        "dt_bias": dt_bias,
        "lower_bound": -5.0,
        "scale": q.shape[-1] ** -0.5,
    }

    actual, actual_state = kda(q, k, v, gate, beta, backend="auto", **kwargs)
    expected, expected_state = torch_recurrent_kda(q, k, v, gate, beta, **kwargs)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)


def test_kda_recurrent_reference_is_differentiable():
    _kda, torch_recurrent_kda = _kda_ops()
    inputs = _inputs(requires_grad=True)

    output, final_state = torch_recurrent_kda(
        *inputs[:5], a_log=inputs[5], dt_bias=inputs[6], lower_bound=-5.0, scale=0.5
    )
    (output.square().mean() + final_state.square().mean()).backward()

    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("backend", ["auto", "torch", "fla"])
def test_kda_rejects_invalid_parameter_shapes_before_backend_dispatch(backend):
    kda, _torch_recurrent_kda = _kda_ops()
    q, k, v, gate, beta, a_log, dt_bias = _inputs()

    with pytest.raises(ValueError, match="dt_bias"):
        kda(
            q,
            k,
            v,
            gate,
            beta,
            a_log=a_log,
            dt_bias=dt_bias[:, :-1],
            lower_bound=-5.0,
            backend=backend,
        )


def test_kda_rejects_unsafe_gate_bound():
    kda, _torch_recurrent_kda = _kda_ops()
    q, k, v, gate, beta, a_log, dt_bias = _inputs()

    with pytest.raises(ValueError, match="lower_bound"):
        kda(q, k, v, gate, beta, a_log=a_log, dt_bias=dt_bias, lower_bound=0.0)


def test_kda_fla_backend_uses_trainable_chunk_contract(monkeypatch):
    kda, _torch_recurrent_kda = _kda_ops()
    q, k, v, gate, beta, a_log, dt_bias = _inputs()
    calls = []
    module = ModuleType("fla.ops.kda")

    def chunk_kda(**kwargs):
        calls.append(kwargs)
        return kwargs["v"] + 1, None

    module.chunk_kda = chunk_kda
    monkeypatch.setitem(sys.modules, "fla", ModuleType("fla"))
    monkeypatch.setitem(sys.modules, "fla.ops", ModuleType("fla.ops"))
    monkeypatch.setitem(sys.modules, "fla.ops.kda", module)

    output, final_state = kda(
        q,
        k,
        v,
        gate,
        beta,
        a_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        output_final_state=False,
        backend="fla",
    )

    torch.testing.assert_close(output, v + 1)
    assert final_state is None
    assert calls[0]["use_qk_l2norm_in_kernel"] is True
    assert calls[0]["use_gate_in_kernel"] is True
    assert calls[0]["use_beta_sigmoid_in_kernel"] is True
    assert calls[0]["safe_gate"] is True
    assert calls[0]["state_v_first"] is True
