# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Execute real CUDA providers and check their declared FP32 surrogate VJPs.

The independent composed reference preserves each live CUDA boundary value,
then differentiates separate Torch equations. It never substitutes a provider
in the production block. REQUIRE_CUDA=1 makes missing CUDA a failure.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_deployment_contract import baseline_csa  # noqa: F401

pytestmark = pytest.mark.gpus(1)
from torch import nn
from torch.nn import functional as F


@pytest.fixture
def cuda(v41_core_te):
    if os.environ.get('MEGATRON_LITE_REQUIRE_CUDA_TESTS') == '1':
        assert torch.cuda.is_available(), 'Production VJP evidence requires real CUDA'
    elif not torch.cuda.is_available():
        pytest.skip('CUDA production providers')
    torch.manual_seed(823)
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device('cuda')


def _leaf(shape, device, dtype=torch.float32):
    return torch.randn(shape, device=device, dtype=dtype).requires_grad_()


def _compare(actual, expected, leaves):
    if isinstance(actual, torch.Tensor):
        actual, expected = (actual,), (expected,)
    incoming = [torch.randn_like(x) for x in actual]
    a = torch.autograd.grad(
        actual, leaves, incoming, allow_unused=True, retain_graph=True
    )
    b = torch.autograd.grad(
        expected, leaves, incoming, allow_unused=True, retain_graph=True
    )
    rows = []
    for index, (leaf, got, wanted) in enumerate(zip(leaves, a, b, strict=True)):
        assert (got is None) == (wanted is None), index
        if got is None:
            continue
        assert torch.isfinite(got).all()
        assert got.dtype == leaf.dtype
        delta = float((got.float() - wanted.float()).abs().max())
        rows.append(
            dict(
                leaf=index,
                dtype=str(got.dtype),
                max_abs=delta,
                mismatch=int((got != wanted).sum()),
            )
        )
        torch.testing.assert_close(got, wanted, atol=2e-5, rtol=2e-5)
    record = dict(test=os.environ.get('PYTEST_CURRENT_TEST'), gradients=rows)
    print('PRODUCTION_VJP ' + json.dumps(record), flush=True)
    dest = os.environ.get('W4_R1_METRICS_DIR')
    if dest:
        path = Path(dest)
        path.mkdir(parents=True, exist_ok=True)
        with (path / 'wrapper-vjp.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')


def _decoded(weight):
    return weight + (weight.bfloat16().float() - weight).detach()


def _coefficients(h, w, scale, base, mixes, broadcast):
    copies = h.shape[-2]
    x = h[..., 0, :].float() if broadcast else h.flatten(2).float()
    weight = w.reshape(-1, copies, h.shape[-1]).sum(1) if broadcast else w
    v = F.linear(x, weight) * torch.rsqrt(
        x.square().mean(-1, keepdim=True) + mixes.norm_eps
    )
    a, b, c = v.split([copies, copies, copies * copies], -1)
    ba, bb, bc = base.split([copies, copies, copies * copies])
    pre = torch.sigmoid(a * scale[0] + ba) + mixes.hc_eps
    post = 2 * torch.sigmoid(b * scale[1] + bb)
    matrix = (c * scale[2] + bc).reshape(*h.shape[:2], copies, copies)
    matrix = matrix.softmax(-1) + mixes.hc_eps
    matrix = matrix / (matrix.sum(-2, keepdim=True) + mixes.hc_eps)
    for _ in range(mixes.iterations - 1):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + mixes.hc_eps)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + mixes.hc_eps)
    return pre, post, matrix


def _post(y, h, p, c):
    return (
        torch.einsum('bsij,bsid->bsjd', c.float(), h.float())
        + p.unsqueeze(-1) * y.float().unsqueeze(-2)
    ).to(h.dtype)


def _joint(h, pre, gamma, mixes, pending=None):
    stream = h if pending is None else _post(*pending)
    broadcast = pending is None and mixes.broadcast_projection
    a, b, c = _coefficients(stream, mixes.fn, mixes.scale, mixes.base, mixes, broadcast)
    folded = (
        stream[..., 0, :]
        if broadcast
        else (stream.float() * pre.unsqueeze(-1)).sum(-2).to(stream.dtype)
    )
    norm = F.rms_norm(folded, (h.shape[-1],), _decoded(gamma), mixes.norm_eps)
    return stream, a, b, c, norm


@pytest.mark.parametrize('limit', [0.0, 10.0])
def test_shared_swiglu_production_backward_includes_intermediate_rounding(cuda, limit):
    from megatron.lite.primitive.modules.deployment_math import shared_swiglu

    y = (torch.randn(3, 1024, device=cuda) * 7).bfloat16().requires_grad_()
    actual = shared_swiglu(y, limit)
    gate, up = y.float().chunk(2, -1)
    if limit:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    activation = F.silu(gate)
    rounded = activation + (activation.bfloat16().float() - activation).detach()
    expected = (rounded * up).bfloat16()
    assert torch.equal(actual, expected)
    # The omitted-rounding mutation must be numerically distinguishable.
    wrong = (activation * up).bfloat16()
    assert torch.count_nonzero(actual != wrong) > 0
    _compare(actual, expected, (y,))


@pytest.mark.parametrize('persistent', [False, True])
def test_bf16_fp32_linear_production_backward_preserves_master(cuda, persistent):
    from megatron.lite.primitive.modules import deployment_math as dm

    x = _leaf((7, 32), cuda, torch.bfloat16)
    w = _leaf((16, 32), cuda)
    actual = dm.bf16_fp32_linear(x, w, persistent=persistent)
    expected = F.linear(x.float(), _decoded(w))
    _compare(actual, expected, (x, w))


@pytest.mark.parametrize(
    'broadcast,pending', [(True, False), (False, False), (False, True)]
)
@pytest.mark.parametrize('width', [512, 1024, 5120])
def test_joint_production_backward_all_live_operands(cuda, broadcast, pending, width):
    from megatron.lite.primitive.modules.attention.mhc import HCMixes
    from megatron.lite.primitive.modules.deployment_math import mhc_joint

    h = _leaf((1, 3, 4, width), cuda, torch.bfloat16)
    pre = _leaf((1, 3, 4), cuda)
    gamma = _leaf((width,), cuda)
    mixes = HCMixes(width, 4, iterations=20).cuda()
    mixes.broadcast_projection = broadcast
    carry = None
    leaves = [h, pre, gamma, mixes.fn, mixes.scale, mixes.base]
    if pending:
        carry = (
            _leaf((1, 3, width), cuda, torch.bfloat16),
            _leaf((1, 3, 4, width), cuda, torch.bfloat16),
            _leaf((1, 3, 4), cuda),
            _leaf((1, 3, 4, 4), cuda),
        )
        leaves.extend(carry)
    actual = mhc_joint(h, pre, gamma, mixes, carry)
    expected = _joint(h, pre, gamma, mixes, carry)
    _compare(actual, expected, leaves)
    if broadcast:
        # Broadcast cannot depend on pre or copies 1..3. This kills the old
        # concatenated-reference mutation even on unreplicated direct inputs.
        grad = torch.autograd.grad(
            actual[-1].float().sum(), pre, allow_unused=True, retain_graph=True
        )[0]
        assert grad is None


def _graft(reference, visible):
    value = reference + (visible.detach() - reference.detach())
    assert torch.equal(value.detach(), visible.detach())
    return value


class _Gain(nn.Module):
    def __init__(self, factor, attention=False):
        super().__init__()
        self.factor, self.attention = factor, attention

    def forward(self, x, state=None, **kwargs):
        out = x * self.factor
        return (out, state) if self.attention else out


def test_two_production_blocks_joint_call_counts_and_composed_layer0_vjp(
    cuda, monkeypatch
):
    import vllm.model_executor.kernels.mhc.tilelang as native
    from megatron.lite.model.deepseek_v41.lite.block import DeepseekV41Block
    from megatron.lite.primitive.modules import deployment_math as dm
    from megatron.lite.primitive.modules.attention.mhc import expand_hc

    counts = dict(pre=0, fused=0, post=0)
    for name, key in [
        ('mhc_pre_delayed_tilelang', 'pre'),
        ('mhc_fused_post_pre_delayed_tilelang', 'fused'),
        ('mhc_post_tilelang', 'post'),
    ]:
        original = getattr(native, name)

        def counted(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(native, name, counted)
    blocks = [
        DeepseekV41Block(512, 4, _Gain(2, True), _Gain(3), iterations=20).cuda()
        for _ in range(2)
    ]
    for i, block in enumerate(blocks):
        block.attn_mixes.deployment_math = block.ffn_mixes.deployment_math = True
        block.attn_mixes.broadcast_projection = i == 0
        with torch.no_grad():
            block.attn_mixes.base.normal_(0, 0.02)
            block.ffn_mixes.base.normal_(0, 0.02)
    joints, posts = [], []
    joint_op, post_op = dm.mhc_joint, dm.mhc_post

    def trace_joint(*args, **kwargs):
        result = joint_op(*args, **kwargs)
        joints.append(result)
        return result

    def trace_post(*args, **kwargs):
        result = post_op(*args, **kwargs)
        posts.append(result)
        return result

    monkeypatch.setattr(dm, 'mhc_joint', trace_joint)
    monkeypatch.setattr(dm, 'mhc_post', trace_post)
    tokens = _leaf((1, 3, 512), cuda, torch.bfloat16)
    hidden, pre = expand_hc(tokens, 4)
    sentinel = object()
    h1, p1, state, carry = blocks[0](hidden, pre, sentinel)
    h2, p2, state2, _ = blocks[1](h1, p1, state, previous_post=carry)
    assert state2 is sentinel
    assert counts == dict(pre=1, fused=3, post=2)
    assert len(joints) == 4 and len(posts) == 2
    # Replay separate FP32 equations with actual live boundary values. This
    # checks the composed surrogate VJP rather than comparing two CUDA paths.
    ref_h, ref_pre = expand_hc(tokens, 4)
    ref_pending = None
    expected = []
    for i, block in enumerate(blocks):
        values = _joint(
            ref_h, ref_pre, block.attn_norm.weight, block.attn_mixes, ref_pending
        )
        h, a, b, c, norm = tuple(_graft(x, y) for x, y in zip(values, joints[2 * i]))
        y = norm * 2
        values = _joint(h, a, block.ffn_norm.weight, block.ffn_mixes, (y, h, b, c))
        h, a, b, c, norm = tuple(
            _graft(x, y) for x, y in zip(values, joints[2 * i + 1])
        )
        ref_pending = (norm * 3, h, b, c)
        ref_h, ref_pre = _graft(_post(*ref_pending), posts[i]), a
        expected.extend((ref_h, ref_pre))
    leaves = [tokens, *[p for block in blocks for p in block.parameters()]]
    _compare((h1, p1, h2, p2), expected, leaves)


@pytest.mark.parametrize('ratio', [1, 2])
def test_compressor_production_backward_raw_and_gamma(cuda, monkeypatch, ratio):
    from megatron.lite.primitive.modules import deployment_math as dm

    native_raw = []
    linear = dm.bf16_fp32_linear

    def trace(*args, **kwargs):
        result = linear(*args, **kwargs)
        native_raw.append(result)
        return result

    monkeypatch.setattr(dm, 'bf16_fp32_linear', trace)
    x = _leaf((2, 5, 32), cuda, torch.bfloat16)
    w = _leaf((512, 32), cuda)
    gate = _leaf((512, 32), cuda) if ratio == 2 else None
    gamma = _leaf((512,), cuda)
    from megatron.lite.primitive.kernels import deployment_compressor

    compress_norm = deployment_compressor.compress_norm

    raw_shape = (2, 5, 512 * ratio)
    with pytest.raises(ValueError, match='CUDA FP32'):
        compress_norm(
            torch.empty(raw_shape, device=cuda, dtype=torch.bfloat16),
            gamma,
            ratio,
            1e-6,
        )
    with pytest.raises(ValueError, match='gamma'):
        compress_norm(torch.empty(raw_shape, device=cuda), gamma[:-1], ratio, 1e-6)
    with pytest.raises(ValueError, match='CR1/2'):
        compress_norm(torch.empty(2, 5, 1536, device=cuda), gamma, 3, 1e-6)
    actual = dm.compressor(x, w, gate, gamma, ratio, 1e-6)
    matrices = w if gate is None else torch.cat((w, gate))
    raw = _graft(F.linear(x.float(), _decoded(matrices)), native_raw[0])
    cutoff = x.shape[1] // ratio * ratio
    pooled = raw[:, :cutoff, :512]
    if ratio == 2:
        values = pooled.unflatten(1, (-1, 2))
        weights = raw[:, :cutoff, 512:].unflatten(1, (-1, 2)).softmax(2)
        pooled = (values * weights).sum(2)
    expected = F.rms_norm(pooled, (512,), _decoded(gamma), 1e-6).bfloat16()
    leaves = [x, w, gamma] if gate is None else [x, w, gate, gamma]
    _compare(actual, expected, leaves)


def test_default_ds4_forward_and_gradients_match_04c736eed_on_gpu(cuda, baseline_csa):
    # The pinned 04c CSA THD adapter requires the newer compressed_rows ABI.
    # Keep DS41/W4's validated core unchanged; isolate this legacy compatibility
    # arm on the recorded nv/dev reference used for the adapter's original tests.
    compat_core = os.environ.get('DS41_CSA_COMPAT_CORE')
    if compat_core and os.environ.get('DS41_CSA_COMPAT_CHILD') != '1':
        env = dict(
            os.environ,
            DS41_CSA_COMPAT_CHILD='1',
            PYTHONPATH=compat_core + os.pathsep + os.environ['PYTHONPATH'],
        )
        node = (
            str(Path(__file__).resolve())
            + '::test_default_ds4_forward_and_gradients_match_04c736eed_on_gpu'
        )
        result = subprocess.run(
            [
                sys.executable,
                '-m',
                'pytest',
                '-c',
                '/dev/null',
                '-s',
                '-v',
                '-p',
                'no:cacheprovider',
                node,
            ],
            env=env,
            check=True,
        )
        assert result.returncode == 0
        return
    import test_redo_v4_preservation as preserved

    for ratio in (0, 2, 4):
        preserved.test_v4_default_forward_and_all_parameter_gradients_are_bitwise(
            baseline_csa, ratio
        )


@pytest.mark.parametrize('width', [1024, 5120])
def test_mega_joint_matches_native_shifted_outputs_and_batch_partition(cuda, width):
    import vllm.models.deepseek_v41.nvidia.ops.mega_mhc as native_mhc
    from megatron.lite.primitive.modules.attention.mhc import HCMixes
    from megatron.lite.primitive.modules.deployment_math import mhc_joint
    from vllm.utils.deep_gemm import _import_deep_gemm

    _import_deep_gemm().set_batch_invariant(True)
    rows, copies = 33, 4
    h = _leaf((1, rows, copies, width), cuda, torch.bfloat16)
    pre = _leaf((1, rows, copies), cuda)
    gamma = _leaf((width,), cuda)
    mixes = HCMixes(width, copies, iterations=20).cuda()
    output = _leaf((1, rows, width), cuda, torch.bfloat16)
    post = _leaf((1, rows, copies), cuda)
    comb = _leaf((1, rows, copies, copies), cuda)
    actual = mhc_joint(h, pre, gamma, mixes, (output, h, post, comb))
    native = native_mhc.mhc_shifted_post_pre_deep_gemm(
        output[0].contiguous(),
        h[0].contiguous(),
        pre[0].contiguous(),
        post[0].unsqueeze(-1).contiguous(),
        comb[0].contiguous(),
        mixes.fn,
        mixes.scale,
        mixes.base,
        mixes.norm_eps,
        mixes.hc_eps,
        2.0,
        mixes.hc_eps,
        mixes.iterations,
        gamma.bfloat16(),
        mixes.norm_eps,
    )
    # Native ABI order is residual, post, comb, normalized, shifted-pre.
    for got, expected in zip(
        actual, (native[0], native[4], native[1], native[2], native[3])
    ):
        assert torch.equal(got.reshape_as(expected), expected)
    parts = []
    for start, end in ((0, 1), (1, 16), (16, rows)):
        sl = slice(start, end)
        parts.append(
            mhc_joint(
                h[:, sl],
                pre[:, sl],
                gamma,
                mixes,
                (output[:, sl], h[:, sl], post[:, sl], comb[:, sl]),
            )
        )
    for index, full in enumerate(actual):
        assert torch.equal(full, torch.cat([part[index] for part in parts], dim=1))
