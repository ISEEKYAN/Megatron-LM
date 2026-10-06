# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed protocol uses the materialized deployment head and keeps its gradients."""
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.runtime.contracts import PackedBatch
from megatron.lite.runtime.contracts.loss import LossContext, use_loss_context


def _batch():
    return PackedBatch(
        input_ids=torch.tensor([0, 1, 2, 3, 4, 5, 6]),
        labels=torch.tensor([0, 1, 2, 3, 4, 5, 6]),
        seq_lens=torch.tensor([3, 4]),
        loss_mask=torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
    )


def test_owned_head_packed_loss_and_gradient(monkeypatch, v41_core_te):
    from megatron.lite.model.deepseek_v41.lite import protocol

    monkeypatch.setattr(
        protocol, 'linear_cross_entropy', lambda *a: pytest.fail('head projected twice')
    )
    torch.manual_seed(31)
    logits = torch.randn(7, 11, requires_grad=True)
    independent = logits.detach().clone().requires_grad_()
    with use_loss_context(
        LossContext(
            temperature=0.8,
            calculate_entropy=True,
            loss_scale=0.75,
            normalization_denominator=17.0,
        )
    ):
        result = protocol.text_output(None, None, _batch(), logits=logits)
    # Independently shift labels/masks within each document, never across boundaries.
    target = torch.tensor([1, 2, 0, 4, 5, 6, 0])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    lp = torch.log_softmax(independent / 0.8, -1)
    expected = lp.gather(-1, target[:, None]).squeeze(-1)
    loss = -(expected * mask).sum() / 17.0 * 0.75
    assert torch.equal(result['log_probs'], expected)
    assert torch.equal(result['entropy'], -(lp.exp() * lp).sum(-1))
    assert torch.equal(result['loss'], loss)
    actual_grad = torch.autograd.grad(result['loss'], logits)[0]
    expected_grad = torch.autograd.grad(loss, independent)[0]
    assert torch.equal(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0


def test_owned_head_forward_step_dispatch(monkeypatch, v41_core_te):
    from megatron.lite.model.deepseek_v41.lite import protocol

    calls = []
    logits = torch.randn(1, 7, 11, requires_grad=True)

    class Model:
        deployment_math = True
        vision_schedule = None
        residual_dtype = torch.bfloat16
        training = True
        ps = SimpleNamespace(pp_size=1, cp_size=1, tp_group=None)

        def __call__(self, ids, *, return_head_hidden, **kwargs):
            calls.append(return_head_hidden)
            assert ids.shape == (1, 7)
            assert not return_head_hidden
            return {'logits': logits}

    monkeypatch.setattr(protocol, '_validate_replay', lambda *a: None)
    result = protocol._forward_step(Model(), _batch())
    assert calls == [False]
    result['loss'].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_owned_head_unlabelled_entropy(v41_core_te):
    from megatron.lite.model.deepseek_v41.lite import protocol

    logits = torch.randn(7, 11, requires_grad=True)
    batch = _batch()
    batch.labels = None
    with use_loss_context(LossContext(calculate_entropy=True)):
        result = protocol.text_output(None, None, batch, logits=logits)
    assert torch.equal(result['logits'], logits)
    assert result['entropy'].shape == (7,)
    assert torch.isfinite(torch.autograd.grad(result['entropy'].sum(), logits)[0]).all()


@pytest.mark.gpus(1)
@pytest.mark.parametrize('vocab', [64, 129280])
def test_deployment_log_softmax_native_and_vjp(v41_core_te, vocab):
    import os

    if os.environ.get('MEGATRON_LITE_REQUIRE_CUDA_TESTS') == '1':
        assert torch.cuda.is_available()
    elif not torch.cuda.is_available():
        pytest.skip('real CUDA log probabilities')
    from megatron.lite.primitive.modules import deployment_math
    from vllm.model_executor.determinism.batch_invariant import log_softmax

    torch.manual_seed(57)
    x = torch.randn(7, vocab, device='cuda', requires_grad=True)
    actual = deployment_math.log_softmax(x)
    native = log_softmax(x.detach(), dim=-1)
    assert torch.equal(actual, native)
    assert torch.equal(
        actual,
        torch.cat([deployment_math.log_softmax(part) for part in x.detach().split(2)]),
    )
    incoming = torch.randn_like(actual)
    grad = torch.autograd.grad(actual, x, incoming)[0]
    probability = torch.softmax(x.detach(), -1)
    independent = incoming - probability * incoming.sum(-1, keepdim=True)
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(grad, independent, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize('forward_only', [False, True])
def test_runtime_respects_explicit_optimizer_selection(v41_core_te, forward_only):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.backends.mlite.runtime import _build_impl_cfg
    from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig

    selected = (
        dict(optimizer=None)
        if forward_only
        else dict(
            optimizer='muon',
            optimizer_config=dict(lr=1e-6, ns_steps=2, coefficient_type='quintic'),
        )
    )
    runtime = SimpleNamespace(
        impl_cfg=dict(
            selected,
            device='cpu',
            dtype='bfloat16',
            quantized=True,
            w4a8_experts=True,
            deployment_math=True,
        ),
        parallel=ParallelConfig(),
        attention_backend_override=None,
        router_aux_loss_coef=0.0,
        hf_path='',
        optimizer=OptimizerConfig(),
    )
    config = _build_impl_cfg(protocol, runtime)
    protocol._validate_parallel(config, runtime.parallel)
    if forward_only:
        assert config.optimizer is None and config.optimizer_config is None
    else:
        assert config.optimizer == 'muon' and config.optimizer_config.lr == 1e-6
