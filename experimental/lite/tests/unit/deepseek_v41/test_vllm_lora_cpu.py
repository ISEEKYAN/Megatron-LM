# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU call/adjoint controls. Fake kernels here establish no native numerical parity."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm import lora


@pytest.fixture
def calls(monkeypatch):
    events = []
    ops = ModuleType('vllm.lora.ops.triton_ops')
    utils = ModuleType('vllm.lora.ops.triton_ops.utils')
    utils._LORA_A_PTR_DICT, utils._LORA_B_PTR_DICT = {}, {}

    def shrink(x, weights, output, *args):
        utils._LORA_A_PTR_DICT.setdefault(tuple(w.data_ptr() for w in weights), weights[0])
        assert x.dtype == weights[0].dtype == torch.bfloat16
        if len(args) == 1:
            assert args == (1.0,)
            output[0].copy_(x.float() @ weights[0][0, 0].float().T)
        else:
            groups, scale = args
            assert scale == 1.0 and weights[0].shape[0] == groups
            for group, rows in enumerate(x.chunk(groups)):
                output[0].chunk(groups)[group].copy_(rows.float() @ weights[0][group, 0].float().T)
        events.append(('shrink', x.clone(), weights[0].clone()))

    def expand(hidden, weights, output, *args, add_inputs):
        assert all(not w.requires_grad and w.grad_fn is None for w in weights)
        utils._LORA_B_PTR_DICT.setdefault(tuple(w.data_ptr() for w in weights), weights[0])
        assert add_inputs and not args
        start = 0
        for index, b in enumerate(weights):
            delta = hidden[index].bfloat16().float() @ b[0, 0].float().T
            end = start + b.shape[-2]
            output[:, start:end].add_(delta.bfloat16())
            start = end
        events.append(('expand', hidden.clone(), [b.clone() for b in weights]))

    ops.lora_shrink, ops.lora_expand = shrink, expand
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setattr(lora, '_metadata', lambda x: ())
    monkeypatch.setattr(lora, 'metadata', lambda x, groups=1: (groups,))
    return events


@pytest.mark.parametrize('rows', [0, 3])
def test_native_calls_scaling_and_closed_adjoints(calls, rows):
    torch.manual_seed(4)
    x = torch.randn(rows, 8, dtype=torch.bfloat16, requires_grad=True)
    a = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(6, 4, dtype=torch.bfloat16, requires_grad=True)
    base = torch.randn(rows, 6, dtype=torch.bfloat16, requires_grad=True)
    saved = base.detach().clone()
    module = SimpleNamespace(lora_a=a, lora_b=b, scale=0.5)
    actual = lora.add(x, base, module)
    effective_b = (b.detach() * 0.5).bfloat16()
    hidden = x.detach().float() @ a.detach().float().T
    delta = (hidden.bfloat16().float() @ effective_b.float().T).bfloat16()
    assert torch.equal(actual, saved + delta)
    assert torch.equal(base, saved)  # expand never mutates a producer's saved/view output
    actual.float().sum().backward()
    dh = torch.ones(rows, 6) @ effective_b.float()
    assert torch.equal(x.grad, (dh @ a.detach().float()).bfloat16())
    assert torch.equal(a.grad, (dh.T @ x.detach().float()).bfloat16())
    db = (torch.ones(rows, 6).T @ hidden.bfloat16().float()).bfloat16()
    assert torch.equal(b.grad, (db * 0.5).bfloat16())
    assert torch.equal(base.grad, torch.ones_like(base))
    assert len(calls) == (2 if rows else 0)
    if rows:
        assert torch.equal(calls[1][2][0][0, 0], effective_b)


def test_grouped_wo_a_uses_one_native_expand_and_one_batched_shrink(calls):
    torch.manual_seed(7)
    x = torch.randn(3, 2, 8, dtype=torch.bfloat16, requires_grad=True)
    base = torch.randn(3, 2, 6, dtype=torch.bfloat16, requires_grad=True)
    module = SimpleNamespace(
        lora_a=torch.randn(2, 4, 8, dtype=torch.bfloat16, requires_grad=True),
        lora_b=torch.randn(2, 6, 4, dtype=torch.bfloat16, requires_grad=True),
        scale=1.0,
    )
    result = lora.grouped_add(x, base, module)
    assert [e[0] for e in calls] == ['shrink', 'expand']
    assert len(calls[-1][2]) == 2
    result.float().square().sum().backward()
    for p in (x, base, module.lora_a, module.lora_b):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()


def test_bypass_does_not_import_or_launch_kernels(monkeypatch):
    def forbidden(*args):
        raise AssertionError('disabled adapter launched a kernel')

    monkeypatch.setattr(lora, '_metadata', forbidden)
    base = torch.tensor([[1.0, -0.0]], dtype=torch.bfloat16)
    assert lora.add(None, base, None) is base
    assert lora.grouped_add(None, base, None) is base
    assert lora.adapter(SimpleNamespace(), 'absent') is None
    assert lora.adapter(SimpleNamespace(a=SimpleNamespace(disabled=True)), 'a') is None


def test_serving_targets_share_config_and_exclude_engram():
    from megatron.lite.primitive.modules.lora import LoraConfig

    names = lora.serving_targets(LoraConfig(rank=32), 40)
    assert len(names) == 20 * 9  # packed wq_a/wkv share one serving module
    assert all('.engram.' not in name for name in names)
    assert {int(name.split('.')[3]) for name in names} == set(range(20, 40))
    assert lora.serving_targets(dict(rank=0), 4) == ()
    assert lora.serving_targets(dict(rank=8, layers=(1, 2), target_modules=('wq_a',)), 4) == (
        'language_model.model.layers.1.attn.fused_wqa_wkv',
    )
    for values in (dict(target_modules=('engram.wkv',)), dict(dropout=0.1), dict(layers=(4, 5))):
        with pytest.raises(ValueError):
            lora.serving_targets(dict(rank=8, **values), 4)


@pytest.mark.parametrize('width', [2304, 4096])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('native_input', [False, True])
def test_empty_expert_activation_keeps_fp32_scale_layout(monkeypatch, width, dtype, native_input):
    from megatron.lite.model.deepseek_v41.vllm import experts

    ops = ModuleType('vllm.model_executor.kernels.batch_invariant')

    def forbidden(*args, **kwargs):
        raise AssertionError('An empty activation must not launch the native kernel')

    ops.swiglu_apply_weight_to_fp8 = forbidden
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    # Metadata-only CUDA admission: all zero-row math stays on the CPU.
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    first = torch.empty(0, 2 * width, dtype=dtype, requires_grad=True)
    scores = torch.empty(0, dtype=torch.float32, requires_grad=True)
    hidden, (codes, scale) = experts.activation(first, scores, None, native_input=native_input)
    assert hidden.shape == codes.shape == (0, width)
    assert hidden.dtype == (torch.bfloat16 if native_input else torch.float32)
    assert codes.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32 and scale.shape == (0, width // 32)
    # The real grouped GEMM pads an empty owner before packing UE8M0 scales.
    packed = experts.packed_scales(scale.new_ones((128, width // 32)))
    assert packed.dtype == torch.int32 and packed.shape == (128, width // 128)
    assert experts.dequantize_activation(codes, scale).shape == (0, width)
    hidden.float().sum().backward()
    assert first.grad.shape == first.shape and scores.grad.shape == scores.shape
    assert torch.isfinite(first.grad).all() and torch.isfinite(scores.grad).all()


@pytest.mark.parametrize('fp8', [False, True])
def test_frozen_base_linear_keeps_input_vjp_without_allocating_base_dw(fp8):
    from megatron.lite.model.deepseek_v41.vllm.linear import _BF16Linear, _MXFP8Linear

    x = torch.randn(3, 8, dtype=torch.bfloat16)
    weight = torch.randn(6, 8, dtype=torch.bfloat16)
    grad = torch.randn(3, 6, dtype=torch.bfloat16)
    ctx = SimpleNamespace(
        saved_tensors=(x, weight),
        input_shape=x.shape,
        weight_dtype=weight.dtype,
        needs_input_grad=(True, False, False),
    )
    result = (_MXFP8Linear if fp8 else _BF16Linear).backward(ctx, grad)
    assert result[1] is None
    assert torch.equal(result[0], (grad.float() @ weight.float()).bfloat16())


@pytest.mark.parametrize('limit', [None, 1.0])
def test_down_reads_native_bf16_not_surrogate_and_owns_activation_adjoint(limit):
    torch.manual_seed(6)
    first = torch.randn(4, 16, dtype=torch.bfloat16, requires_grad=True)
    score = torch.randn(4, dtype=torch.float32, requires_grad=True)
    native = torch.full((4, 8), 3.0, dtype=torch.bfloat16)  # deliberately not surrogate values
    output = lora.activation_input(native, first, score, limit)
    assert torch.equal(output.view(torch.uint8), native.view(torch.uint8))
    output.float().sum().backward()
    reference = first.detach().float().requires_grad_()
    weights = score.detach().requires_grad_()
    gate, up = reference.chunk(2, -1)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    (torch.nn.functional.silu(gate) * up * weights[:, None]).sum().backward()
    assert torch.equal(first.grad, reference.grad.bfloat16())
    torch.testing.assert_close(score.grad, weights.grad, rtol=1e-6, atol=1e-6)


def test_fp8_wo_a_delta_uses_native_prequant_rotation_contract(monkeypatch):
    from megatron.lite.model.deepseek_v41.vllm.attention import _InverseRoPE

    module = ModuleType('vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant')
    positions, cache = torch.arange(3), torch.zeros(3, 64)
    native = torch.randn(3, 8, 4096, dtype=torch.bfloat16)
    seen = []

    def rotate(x, actual_positions, actual_cache, **kwargs):
        assert actual_positions is positions and actual_cache is cache
        seen.append(kwargs)
        return native, torch.empty(0)

    module.fused_inv_rope_fp8_quant = rotate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ctx = SimpleNamespace(save_for_backward=lambda *args: None)
    result = _InverseRoPE.forward(
        ctx, SimpleNamespace(is_cuda=True, shape=(3, 64, 512)), positions, cache, 8
    )
    assert result is native  # no second cast, rotation or FP8 dequantization
    assert seen == [
        dict(
            n_groups=8,
            heads_per_group=8,
            nope_dim=448,
            rope_dim=64,
            quant_group_size=32,
            quantize=False,
        )
    ]


@pytest.fixture
def pointer_cache(monkeypatch):
    utils = ModuleType('vllm.lora.ops.triton_ops.utils')
    utils._LORA_B_PTR_DICT = {(12345,): object()}
    utils._LORA_A_PTR_DICT = {}
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return utils._LORA_B_PTR_DICT


@pytest.mark.parametrize('groups', [None, 2])
def test_scaled_b_reuses_storage_but_not_a_freed_autograd_graph(groups, pointer_cache):
    shape = (4, 2) if groups is None else (groups, 4, 2)
    owner = SimpleNamespace(
        lora_a=torch.nn.Parameter(torch.randn(2, 2)),
        lora_b=torch.nn.Parameter(torch.randn(shape)),
        scale=0.3,
    )
    outputs = []
    for _ in range(3):
        value = lora.scaled_b(owner)
        outputs.append(value)
        expected = owner.lora_b.detach() * owner.scale
        assert torch.equal(value, expected)
        value.sum().backward()  # independent microbatch backward, no retain_graph
    assert len({v.data_ptr() for v in outputs}) == 1
    assert not owner._native_scaled_b.requires_grad
    assert owner._native_scaled_b.grad_fn is None
    torch.testing.assert_close(owner.lora_b.grad, torch.full(shape, 0.9))
    value = outputs[0]
    key = (value.data_ptr(),) if groups is None else tuple(v.data_ptr() for v in value)
    pointer_cache[key] = value
    lora.invalidate_scaled_b(owner)
    assert owner._native_scaled_b_stamp is None
    assert owner._native_scaled_b.data_ptr() == value.data_ptr()
    assert set(pointer_cache) == {(12345,), key}
    lora.invalidate_scaled_b(owner, release=True)
    assert owner._native_scaled_b is None
    assert set(pointer_cache) == {(12345,)}


def test_native_optimizer_step_invalidates_even_when_model_version_does_not(pointer_cache):
    model = torch.nn.Module()
    model.lora_a = torch.nn.Parameter(torch.randn(2, 2))
    model.lora_b = torch.nn.Parameter(torch.randn(4, 2))
    model.scale = 0.5
    master = torch.nn.Parameter(model.lora_b.detach().clone())
    inner = torch.optim.Adam([master])
    lora.prepare_scaled_b(model)
    lora.release_scaled_b_after_step([model], SimpleNamespace(optimizers=[inner]))
    before = lora.scaled_b(model)
    old_value = before.detach().clone()
    version = model.lora_b._version
    inner.zero_grad()
    master.grad = torch.ones_like(master)
    inner.step()
    assert model._native_scaled_b_stamp is None
    model.lora_b.data.copy_(master.data)  # Core's buffer-copy model update
    assert model.lora_b._version == version
    after = lora.scaled_b(model)
    assert after.data_ptr() == before.data_ptr()
    assert torch.equal(after, model.lora_b.detach() * model.scale)
    assert not torch.equal(after, old_value)
    with torch.no_grad():
        model.lora_b.add_(1)
    after = lora.scaled_b(model)
    assert after.data_ptr() == before.data_ptr()
    assert torch.equal(after, model.lora_b.detach() * model.scale)


def test_persistent_buffer_rejects_changed_geometry_at_same_parameter_address(pointer_cache):
    b = torch.nn.Parameter(torch.randn(4, 8))
    owner = SimpleNamespace(lora_a=torch.nn.Parameter(torch.randn(2, 8)), lora_b=b, scale=1.0)
    lora.scaled_b(owner)
    old_address = b.data_ptr()
    owner.lora_b = torch.nn.Parameter(b.detach().reshape(8, 4))
    assert owner.lora_b.data_ptr() == old_address
    with pytest.raises(ValueError, match='geometry changed'):
        lora.scaled_b(owner)


def test_model_migration_releases_pointer_metadata_before_reallocating(pointer_cache):
    from megatron.lite.model.deepseek_v41.vllm.model import DeepseekV41Model

    model = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.lora_a = torch.nn.Parameter(torch.randn(2, 8))
    model.lora_b = torch.nn.Parameter(torch.randn(4, 2))
    model.scale = 0.5
    value = lora.scaled_b(model)
    pointer_cache[(value.data_ptr(),)] = value.detach()
    model.to('meta')
    assert model.lora_b.is_meta and model._native_scaled_b.is_meta
    assert model._native_scaled_b_stamp is None
    assert set(pointer_cache) == {(12345,)}


@pytest.mark.parametrize('groups', [None, 2])
def test_fifty_native_call_steps_keep_a_and_b_pointer_keys_constant(calls, groups):
    model = torch.nn.Module()
    prefix = () if groups is None else (groups,)
    model.lora_a = torch.nn.Parameter(torch.randn(*prefix, 4, 8, dtype=torch.bfloat16))
    model.lora_b = torch.nn.Parameter(torch.randn(*prefix, 6, 4, dtype=torch.bfloat16))
    model.scale = 0.3
    lora.prepare_scaled_b(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lora.release_scaled_b_after_step([model], SimpleNamespace(optimizers=[optimizer]))
    native = sys.modules['vllm.lora.ops.triton_ops.utils']
    address = model._native_scaled_b.data_ptr()
    for _ in range(50):
        optimizer.zero_grad()
        x = torch.randn(3, 8 * (groups or 1), dtype=torch.bfloat16)
        base = torch.zeros(3, 6 * (groups or 1), dtype=torch.bfloat16)
        fn = lora.add if groups is None else lora.grouped_add
        for microbatch in range(2):
            fn(x, base, model).float().sum().backward()
        assert model._native_scaled_b.data_ptr() == address
        assert torch.equal(model._native_scaled_b, model.lora_b.detach() * model.scale)
        assert (len(native._LORA_A_PTR_DICT), len(native._LORA_B_PTR_DICT)) == (1, 1)
        optimizer.step()
        assert model._native_scaled_b_stamp is None


def test_initial_allocation_drops_stale_multi_slice_metadata_at_reused_address(
    pointer_cache, monkeypatch
):
    storage = torch.empty(2, 32)
    old_views = storage.reshape(2, 4, 8)
    key = tuple(v.data_ptr() for v in old_views)
    pointer_cache[key] = ('old geometry', (4, 8))
    owner = SimpleNamespace(
        lora_a=torch.nn.Parameter(torch.randn(2, 8, 3)),
        lora_b=torch.nn.Parameter(torch.randn(2, 8, 4)),
        scale=0.5,
    )
    monkeypatch.setattr(torch, 'empty_like', lambda *a, **kw: storage.reshape(2, 8, 4))
    lora.prepare_scaled_b(owner)
    assert tuple(v.data_ptr() for v in owner._native_scaled_b) == key
    assert key not in pointer_cache  # do not reuse the other module's cached geometry
