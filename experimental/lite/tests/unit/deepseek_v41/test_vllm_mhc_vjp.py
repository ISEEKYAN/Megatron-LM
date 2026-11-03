# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Owned mHC post/collapse derivatives; native CUDA forward is a separate gate."""
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.mhc import _Collapse, _Joint, _Post, collapse, post_mix


@pytest.fixture(autouse=True)
def core_kernels_on_cpu(monkeypatch):
    """Run the actual Core kernel bodies; only autotuning is replaced on CPU."""
    import triton.experimental.gluon  # noqa: F401; initialize before patching tl helpers
    import triton.language as tl
    from triton.runtime.interpreter import InterpretedFunction
    from triton.runtime.jit import JITFunction

    from megatron.core.fusions import fused_mhc_kernels as core

    # Also interpret the standard-language helpers if pytest was started
    # without TRITON_INTERPRET=1. monkeypatch restores them after each test.
    for module in (tl, tl.standard):
        for name, value in list(vars(module).items()):
            if isinstance(value, JITFunction):
                monkeypatch.setattr(module, name, InterpretedFunction(value.fn))

    class FixedKernel:
        def __init__(self, kernel, **config):
            self.kernel = InterpretedFunction(kernel.fn.fn)
            self.config = config

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                return self.kernel[grid](*args, **self.config, **kwargs)

            return launch

    for name in ("_triton_hpb_bwd_g_x_orig_kernel", "_triton_hpb_bwd_g_hp_hr_kernel"):
        monkeypatch.setattr(core, name, FixedKernel(getattr(core, name), BLOCK_C=128, BLOCK_S=2))
    name = "_triton_sinkhorn_bwd_kernel"
    monkeypatch.setattr(core, name, FixedKernel(getattr(core, name)))


def _eager_joint(values, pre, gamma, fn, scale, base, output, residual, post, comb, options):
    """Independent copy of the pre-optimization graph, kept outside production."""
    eps, hc_eps, sinkhorn_eps, iterations, broadcast, pending = options
    copies, width = values.shape[-2:]
    stream = values
    if pending:
        mixed = torch.einsum("...ij,...id->...jd", comb.float(), residual.float())
        stream = (mixed + post.float().unsqueeze(-1) * output.float().unsqueeze(-2)).to(
            values.dtype
        )
    if broadcast:
        token = stream[..., 0, :].float()
        weight = fn.reshape(-1, copies, width).sum(1)
    else:
        token, weight = stream.flatten(-2).float(), fn
    projected = torch.nn.functional.linear(token, weight.float()) * torch.rsqrt(
        token.square().mean(-1, keepdim=True) + eps
    )
    a, b, c = projected.split((copies, copies, copies * copies), -1)
    ba, bb, bc = base.float().split((copies, copies, copies * copies))
    next_pre = torch.sigmoid(a * scale[0] + ba) + hc_eps
    next_post = 2 * torch.sigmoid(b * scale[1] + bb)
    logits = (c * scale[2] + bc).reshape(*token.shape[:-1], copies, copies)
    next_comb = logits.softmax(-1) + sinkhorn_eps
    next_comb = next_comb / (next_comb.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(iterations - 1):
        next_comb = next_comb / (next_comb.sum(-1, keepdim=True) + sinkhorn_eps)
        next_comb = next_comb / (next_comb.sum(-2, keepdim=True) + sinkhorn_eps)
    collapsed = (
        stream[..., 0, :]
        if broadcast
        else (stream.float() * pre.unsqueeze(-1)).sum(-2).to(stream.dtype)
    )
    rounded_gamma = gamma + (gamma.bfloat16().to(gamma.dtype) - gamma).detach()
    normed = torch.nn.functional.rms_norm(collapsed, (width,), rounded_gamma, eps)
    return stream, next_pre, next_post, next_comb, normed


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("broadcast,pending", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("grad_slots", [(0, 1, 2, 3, 4), (1, 2, 3), (4,), (0,)])
def test_joint_vjp_matches_eager_without_autograd_replay(
    monkeypatch, dtype, broadcast, pending, grad_slots
):
    torch.manual_seed(1704)
    rows, batch, copies, width = 3, 2, 4, 16
    shapes = [
        (rows, batch, copies, width),
        (rows, batch, copies),
        (width,),
        (copies * (copies + 2), copies * width),
        (3,),
        (copies * (copies + 2),),
        (rows, batch, width),
        (rows, batch, copies, width),
        (rows, batch, copies),
        (rows, batch, copies, copies),
    ]
    operands = [
        (torch.randn(s, dtype=dtype if i in (0, 6, 7) else torch.float32) * 0.3).requires_grad_()
        for i, s in enumerate(shapes)
    ]
    options = (1e-6, 1e-6, 1e-6, 20, broadcast, pending)
    target = _eager_joint(*operands, options)
    grads = [torch.randn_like(t) if i in grad_slots else None for i, t in enumerate(target)]
    expected = torch.autograd.grad(
        [target[i] for i in grad_slots], operands, [grads[i] for i in grad_slots], allow_unused=True
    )
    saved = (*[x.detach() for x in operands], target[0].detach())
    ctx = SimpleNamespace(
        saved_tensors=saved, options=options, needs_input_grad=(True,) * 10 + (False,)
    )
    monkeypatch.setattr(torch.autograd, "grad", lambda *a, **kw: pytest.fail("whole-graph replay"))
    actual = _Joint.backward(ctx, *grads)
    assert actual[-1] is None
    for value, reference in zip(actual[:-1], expected):
        if reference is None:
            # An inactive branch may return an explicit zero instead of None.
            assert value is None or not torch.count_nonzero(value)
            continue
        assert value is not None and value.dtype == reference.dtype
        assert torch.isfinite(value).all()
        if dtype == torch.float32:
            torch.testing.assert_close(value, reference, rtol=1e-4, atol=2e-5)
        elif reference.count_nonzero():
            a, b = value.double().flatten(), reference.double().flatten()
            assert (a - b).norm() / b.norm() <= 0.03
            assert torch.nn.functional.cosine_similarity(a, b, dim=0) >= 0.999
        else:
            assert not torch.count_nonzero(value)
    # Frozen norm/mixing parameters must not receive gradients, but the
    # trainable incoming activation and pending operands still do.
    ctx.needs_input_grad = (True, False, False, False, False, False, True, True, True, True, False)
    frozen = _Joint.backward(ctx, *grads)
    assert all(x is None for x in frozen[1:6])
    for i in (0, 6, 7, 8, 9):
        if actual[i] is None:
            assert frozen[i] is None
        else:
            assert torch.equal(frozen[i], actual[i])


@pytest.mark.parametrize("broadcast,pending", [(False, False), (True, False), (False, True)])
def test_joint_forward_keeps_native_calls_and_saves_original_projection(
    monkeypatch, broadcast, pending
):
    import megatron.lite.model.deepseek_v41.vllm.mhc as mhc

    copies, width, rows = 4, 16, 6
    values = torch.randn(2, 3, copies, width)
    pre = torch.randn(2, 3, copies)
    gamma = torch.randn(width)
    fn = torch.randn(24, copies * width)
    scale, base = torch.randn(3), torch.randn(24)
    output, residual = torch.randn(2, 3, width), torch.randn_like(values)
    post, comb = torch.randn_like(pre), torch.randn(2, 3, copies, copies)
    native = [
        torch.randn(rows, copies, width),
        torch.randn(rows, copies),
        torch.randn(rows, copies, copies),
        torch.randn(rows, width),
        torch.randn(rows, copies),
    ]
    calls = []

    def delayed(stream_values, weight, *args, **kwargs):
        calls.append("delayed")
        assert torch.equal(weight, fn.reshape(24, copies, width).sum(1) if broadcast else fn)
        assert torch.equal(stream_values, values.reshape(rows, copies, width))
        assert torch.equal(kwargs["norm_weight"], gamma.bfloat16())
        assert ("x" in kwargs) == broadcast
        assert ("pre_mix" in kwargs) != broadcast
        return native[1:]

    def shifted(y, r, p, c, weight, *args, **kwargs):
        calls.append("shifted")
        assert torch.equal(weight, fn)
        for got, expected in zip((y, r, p, c), (output, residual, post, comb)):
            assert torch.equal(got.flatten(), expected.flatten())
        return *native, None

    name = "vllm.models.deepseek_v41.nvidia.ops.mhc"
    monkeypatch.setitem(
        sys.modules,
        name,
        SimpleNamespace(mhc_pre_delayed_overlap=delayed, mhc_shifted_post_pre=shifted),
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(mhc, "_post_backward", lambda *a: pytest.fail("adjoint in native forward"))
    saved = []
    ctx = SimpleNamespace(
        save_for_backward=lambda *t: saved.extend(t), set_materialize_grads=lambda flag: None
    )
    result = _Joint.forward(
        ctx,
        values,
        pre,
        gamma,
        fn,
        scale,
        base,
        output,
        residual,
        post,
        comb,
        (1e-6, 1e-6, 1e-6, 20, broadcast, pending),
    )
    expected = (native[0] if pending else values, native[4], native[1], native[2], native[3])
    assert calls == ["shifted" if pending else "delayed"]
    assert saved[3] is fn
    for got, target in zip(result, expected):
        assert torch.equal(got.flatten(), target.flatten())
    assert torch.equal(saved[-1], result[0])


@pytest.mark.parametrize("rows", [0, 3])
def test_post_owned_vjp_matches_composed_residual_reference(rows):
    torch.manual_seed(71)
    shapes = [(rows, 1, 16), (rows, 1, 4, 16), (rows, 1, 4), (rows, 1, 4, 4)]
    operands = [torch.randn(s, requires_grad=True) for s in shapes]
    output, residual, post, comb = operands
    mixed = torch.einsum("...ij,...id->...jd", comb, residual)
    target = mixed + post.unsqueeze(-1) * output.unsqueeze(-2)
    grad = torch.randn_like(target)
    expected = torch.autograd.grad(target, operands, grad)
    actual = _Post.backward(SimpleNamespace(saved_tensors=operands), grad)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [16, 129])
def test_core_post_vjp_strided_views_and_asymmetric_comb(dtype, width):
    torch.manual_seed(704)
    # A sequence and batch prefix, strided native output/residual/coefficients,
    # and a ragged width exercise the adapter as well as both Core kernels.
    shapes = [(2, 3, width), (2, 3, 4, width), (2, 3, 4), (2, 3, 4, 4)]
    operands = [
        torch.randn(*s, 2, dtype=dtype if i < 2 else torch.float32)[..., 0].requires_grad_()
        for i, s in enumerate(shapes)
    ]
    output, residual, post, comb = operands
    mixed = torch.einsum("...ij,...id->...jd", comb.float(), residual.float())
    target = (mixed + post.float().unsqueeze(-1) * output.float().unsqueeze(-2)).to(dtype)
    grad = torch.randn(*target.shape, 2, dtype=dtype)[..., 0]
    expected = torch.autograd.grad(target, operands, grad)
    actual = _Post.backward(SimpleNamespace(saved_tensors=operands), grad)
    for value, reference in zip(actual, expected):
        assert value.dtype == reference.dtype
        if value.dtype == torch.float32:
            torch.testing.assert_close(value, reference, rtol=1e-5, atol=3e-5)
        else:
            # The interpreter's BF16 conversion can differ by one ULP.
            # Use the same per-group VJP bar as the full-model GPU gate.
            a, b = value.double().flatten(), reference.double().flatten()
            assert (a - b).norm() / b.norm() <= 0.03
            assert torch.nn.functional.cosine_similarity(a, b, dim=0) >= 0.999
    # A transpose bug cannot hide behind a symmetric or identity residual mix.
    wrong = torch.einsum("...ji,...jd->...id", comb.float(), grad.float()).to(dtype)
    assert not torch.allclose(actual[1], wrong)


def test_collapse_owned_vjp_keeps_all_stream_dependencies():
    values = torch.randn(3, 1, 4, 16, requires_grad=True)
    pre = torch.randn(3, 1, 4, requires_grad=True)
    grad = torch.randn(3, 1, 16)
    expected = torch.autograd.grad((values * pre.unsqueeze(-1)).sum(-2), (values, pre), grad)
    actual = _Collapse.backward(SimpleNamespace(saved_tensors=(values, pre)), grad)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


def test_materialized_mhc_boundaries_reject_cpu_before_vllm_import():
    x = torch.zeros(1, 1, 4, 16)
    pre = torch.zeros(1, 1, 4)
    with pytest.raises(RuntimeError, match="CUDA"):
        collapse(x, pre)
    with pytest.raises(RuntimeError, match="CUDA"):
        post_mix(x[..., 0, :], x, pre, torch.zeros(1, 1, 4, 4))
