# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native storage lifecycle/bit controls; GPU kernel parity remains a separate gate."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from megatron.lite.model.deepseek_v41.lite import frozen
from megatron.lite.model.deepseek_v41.vllm.batching import packed_weights
from megatron.lite.primitive.ckpt.row_reader import RowSliceReader
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
from megatron.lite.primitive.quantization.frozen import native_payload
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4


def encoded(kind, shape=(64, 64)):
    torch.manual_seed(19)
    x = torch.randn(shape).bfloat16()
    codes, scale = (
        quantize_mxfp4(x)
        if kind == 'mxfp4'
        else quantize_block_fp8(x, (32, 32), scale_format='e8m0')
    )
    return frozen.FrozenWeight(codes, scale, shape, kind)


@pytest.mark.parametrize('kind', ['mxfp4', 'fp8_block'])
def test_core_parameter_metadata_and_to_retain_only_codes(kind):
    w = encoded(kind)
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(w, requires_grad=False)
    module.to('cpu', dtype=torch.bfloat16)
    p = module.weight
    assert isinstance(p, frozen.FrozenWeight) and not p.requires_grad
    assert p.shape == (64, 64) and p.dtype == torch.bfloat16
    assert p.codes.data_ptr() == w.codes.data_ptr()
    assert p.scale.data_ptr() == w.scale.data_ptr()
    assert all(
        t.dtype not in (torch.bfloat16, torch.float32)
        for t in vars(p).values()
        if isinstance(t, torch.Tensor)
    )
    assert p.codes.nbytes + p.scale.nbytes < p.numel() * p.element_size()
    assert torch.equal(p.decoded().view(torch.uint8), w.decoded().view(torch.uint8))
    moved = torch.nn.Module()
    moved.weight = torch.nn.Parameter(w, requires_grad=False)
    moved.to('meta')
    assert moved.weight.device.type == moved.weight.codes.device.type == 'meta'
    assert moved.weight.scale.device.type == 'meta'
    with pytest.raises(ValueError, match='immutable'):
        p.requires_grad_(True)
    with pytest.raises((RuntimeError, ValueError)):
        p.copy_(torch.zeros(p.shape, dtype=p.dtype))


@pytest.mark.parametrize('kind', ['mxfp4', 'fp8_block'])
def test_concat_reshape_preserves_raw_native_bytes_and_input_adjoint(kind):
    w = encoded(kind)
    merged = torch.cat([w, w])
    assert isinstance(merged, frozen.FrozenWeight)
    assert torch.equal(
        merged.codes.view(torch.uint8), torch.cat([w.codes, w.codes]).view(torch.uint8)
    )
    assert isinstance(merged.reshape(2, 64, 64), frozen.FrozenWeight)
    x = torch.randn(3, 64, requires_grad=True)
    expected = x.detach().clone().requires_grad_()
    torch.nn.functional.linear(x, merged.float()).sum().backward()
    torch.nn.functional.linear(expected, merged.decoded().float()).sum().backward()
    assert torch.equal(x.grad, expected.grad)


def test_grouped_native_weights_bypass_quantizer(monkeypatch):
    from megatron.lite.primitive.quantization import mxfp4

    w = encoded('mxfp4')
    monkeypatch.setattr(
        mxfp4, 'quantize_mxfp4', lambda *a, **k: pytest.fail('Native codes requantized')
    )
    codes, scale = packed_weights([w, w])
    assert torch.equal(codes.view(torch.uint8), torch.stack([w.codes, w.codes]).view(torch.uint8))
    assert torch.equal(scale.view(torch.uint8), torch.stack([w.scale, w.scale]).view(torch.uint8))
    with pytest.raises(ValueError, match='mix'):
        packed_weights([w, w.decoded()])


def test_release_loader_reads_native_without_decode_or_bf16_copy(tmp_path, monkeypatch):
    from megatron.lite.model.deepseek_v41.lite.checkpoint import _experts

    w = encoded('mxfp4')
    names = ['layers.0.ffn.experts.4.w1.weight', 'layers.0.ffn.experts.4.w3.weight']
    raw = {}
    for key in names:
        raw[key], raw[key[:-6] + 'scale'] = w.codes, w.scale
    save_file({k: v.clone() for k, v in raw.items()}, str(tmp_path / 'model.safetensors'))
    layer = torch.nn.Module()
    layer.ffn = torch.nn.Module()
    layer.ffn.experts = torch.nn.Module()
    layer.ffn.experts.linear_fc1 = torch.nn.Module()
    layer.ffn.experts.linear_fc1.weight0 = torch.nn.Parameter(
        torch.empty(128, 64, dtype=torch.bfloat16)
    )
    layer.ffn.experts.linear_fc1.weight_tensor = layer.ffn.experts.linear_fc1.weight0.detach()
    model = SimpleNamespace(
        release=SimpleNamespace(n_routed_experts=8), ps=SimpleNamespace(ep_rank=2, ep_size=4)
    )
    with RowSliceReader(tmp_path) as reader:
        frozen.prepare_reader(reader)
        frozen.install(layer, 'layers.0.', model, reader)
        p = layer.ffn.experts.linear_fc1.weight0
        assert layer.ffn.experts.linear_fc1.weight_tensor is None
        assert isinstance(p, frozen.FrozenWeight) and p.shape == (128, 64)
        monkeypatch.setattr(
            frozen.FrozenWeight, 'decoded', lambda *a: pytest.fail('Decoded native load')
        )
        loaded = _experts(
            reader, 'layers.0.ffn.experts.linear_fc1.weight0', p, model.release, model.ps
        )
        p.copy_(loaded.to(dtype=p.dtype, device=p.device))
        assert torch.equal(
            p.codes.view(torch.uint8), torch.cat([w.codes, w.codes]).view(torch.uint8)
        )
        assert set(reader._frozen_native) == set(names)
    assert not reader._frozen_native and not reader._frozen_raw


@pytest.mark.parametrize('fault', ['size', 'scale_dtype', 'kind'])
def test_invalid_native_payload_is_rejected(fault):
    w = encoded('mxfp4')
    with pytest.raises(ValueError):
        frozen.FrozenWeight(
            w.codes,
            w.scale.float() if fault == 'scale_dtype' else w.scale,
            (2, 64) if fault == 'size' else w.shape,
            'fp8_block' if fault == 'kind' else 'mxfp4',
        )


@pytest.mark.parametrize('kind', ['mxfp4', 'fp8_block'])
def test_temporary_decode_is_bitwise_the_existing_release_loader(kind):
    from megatron.lite.primitive.ckpt.hf_weights import _dequantize_block_scaled_tensor

    w = encoded(kind)
    reference = _dequantize_block_scaled_tensor(w.codes, w.scale, w.shape).bfloat16()
    assert torch.equal(w.decoded().view(torch.uint8), reference.view(torch.uint8))


def test_lite_leaf_decode_is_replayed_released_and_exception_safe():
    from torch.utils.checkpoint import checkpoint

    w = encoded('fp8_block')
    layer = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    frozen.install_bf16_hooks(layer)
    observed = []
    layer.register_forward_pre_hook(lambda module, args: observed.append(type(module.weight)))
    x = torch.randn(3, 64, dtype=torch.bfloat16, requires_grad=True)
    expected = x.detach().clone().requires_grad_()
    output = checkpoint(lambda v: layer(v), x, use_reentrant=True)
    assert isinstance(layer.weight, frozen.FrozenWeight)
    output.float().sum().backward()
    torch.nn.functional.linear(expected, w.decoded()).float().sum().backward()
    assert len(observed) == 2 and all(t is torch.nn.Parameter for t in observed)
    assert isinstance(layer.weight, frozen.FrozenWeight)
    assert torch.equal(x.grad, expected.grad)
    assert not any(
        isinstance(v, torch.Tensor) and not isinstance(v, frozen.FrozenWeight)
        for v in vars(layer).values()
    )
    with pytest.raises(RuntimeError):
        layer(torch.zeros(1, 63, dtype=torch.bfloat16))
    assert isinstance(layer.weight, frozen.FrozenWeight)


def test_grouped_leaf_decodes_all_experts_in_one_codec_call(monkeypatch):
    class Grouped(torch.nn.Module):
        def forward(self, x):
            assert all(not isinstance(p, frozen.FrozenWeight) for p in self.parameters())
            return torch.stack([torch.nn.functional.linear(x, p) for p in self.parameters()])

    layer = Grouped()
    weights = [encoded('mxfp4') for _ in range(3)]
    for index, w in enumerate(weights):
        layer.register_parameter(f'weight{index}', torch.nn.Parameter(w, requires_grad=False))
    original, calls = frozen.FrozenWeight.decoded, []

    def decode(w):
        calls.append(tuple(w.shape))
        return original(w)

    monkeypatch.setattr(frozen.FrozenWeight, 'decoded', decode)
    frozen.install_bf16_hooks(layer)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    y = layer(x)
    assert calls == [(3, 64, 64)]
    assert torch.equal(
        y, torch.stack([torch.nn.functional.linear(x, original(w)) for w in weights])
    )
    assert all(isinstance(p, frozen.FrozenWeight) for p in layer.parameters())


def test_native_install_leaves_generated_ape_parameter_untouched():
    layer = torch.nn.Module()
    layer.attention = torch.nn.Module()
    layer.attention.core_attention = torch.nn.Module()
    layer.attention.core_attention.native = torch.nn.Module()
    layer.attention.core_attention.native.compressor = torch.nn.Module()
    original = torch.nn.Parameter(torch.zeros(2, 64), requires_grad=False)
    layer.attention.core_attention.native.compressor.ape = original
    model = SimpleNamespace(release=SimpleNamespace(), ps=SimpleNamespace())
    frozen.install(layer, 'layers.2.', model, SimpleNamespace())
    assert layer.attention.core_attention.native.compressor.ape is original


@pytest.mark.parametrize('kind', ['mxfp4', 'fp8_block'])
def test_module_apply_native_payload_follows_parameter_storage(kind):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(encoded(kind), requires_grad=False)
    parameter = module.weight
    parameter.owner_marker = object()
    marker = parameter.owner_marker
    moved = []

    def move(value):
        result = frozen.FrozenWeight(
            value.codes.clone(), value.scale.clone(), value.shape, value.kind, torch.float32
        )
        moved.append(result)
        return result

    original_codes = parameter.codes.view(torch.uint8).clone()
    original_scale = parameter.scale.view(torch.uint8).clone()
    module._apply(move)  # actual same-backend set_data path, as CPU -> CUDA uses
    assert module.weight is parameter and parameter.owner_marker is marker
    assert parameter.dtype == torch.float32 and not parameter.requires_grad
    assert parameter.codes is moved[0].codes and parameter.scale is moved[0].scale
    assert torch.equal(parameter.codes.view(torch.uint8), original_codes)
    assert torch.equal(parameter.scale.view(torch.uint8), original_scale)
    assert parameter.data.codes is parameter.codes
    with pytest.raises(ValueError, match='matching native storage'):
        parameter.data = torch.zeros(parameter.shape)
    assert parameter.codes is moved[0].codes


@pytest.mark.parametrize('kind', ['mxfp4', 'fp8_block'])
@pytest.mark.parametrize('mode', ['set_data', 'overwrite', 'swap'])
def test_native_module_apply_conversion_branches(kind, mode):
    old_overwrite = torch.__future__.get_overwrite_module_params_on_conversion()
    old_swap = torch.__future__.get_swap_module_params_on_conversion()
    try:
        torch.__future__.set_overwrite_module_params_on_conversion(mode == 'overwrite')
        torch.__future__.set_swap_module_params_on_conversion(mode == 'swap')
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(encoded(kind), requires_grad=False)
        original = module.weight
        marker = original.owner_marker = object()
        moved = []

        def convert(value):
            result = frozen.FrozenWeight(
                value.codes.clone(), value.scale.clone(), value.shape, value.kind, torch.float32
            )
            moved.append(result)
            return result

        module._apply(convert)
        assert module.weight.codes is moved[0].codes and module.weight.scale is moved[0].scale
        assert module.weight.dtype == torch.float32
        assert (module.weight is original) == (mode != 'overwrite')
        if mode == 'set_data':
            assert module.weight.owner_marker is marker
        assert torch.equal(module.weight.codes.view(torch.uint8), moved[0].codes.view(torch.uint8))
        module.to('meta')
        assert module.weight.device == module.weight.codes.device == module.weight.scale.device
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(old_overwrite)
        torch.__future__.set_swap_module_params_on_conversion(old_swap)


@pytest.mark.skipif(
    not torch.cuda._is_compiled(), reason='Requires CUDA-built torch, no GPU allocation'
)
@pytest.mark.parametrize('kind', ['mxfp4', 'fp8_block'])
def test_module_to_cuda_api_with_fake_cuda_allocator(kind, monkeypatch):
    # Execute Module.to/_apply with CUDA device semantics on CPU-only nodes.
    # Actual copies and kernels are checked again by the allocated GPU probe.
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class NativeStorageMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            # Keep the real wrapper's dispatch; fake only its inner allocations.
            if args and isinstance(args[0], frozen.FrozenWeight):
                return frozen.FrozenWeight.__torch_dispatch__(func, types, args, kwargs)
            return func(*args, **(kwargs or {}))

    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: None)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    with FakeTensorMode(), NativeStorageMode():
        if kind == 'mxfp4':
            codes = torch.empty((32, 16), dtype=torch.int8)
            scale = torch.empty((32, 1), dtype=torch.float8_e8m0fnu)
        else:
            codes = torch.empty((32, 32), dtype=torch.float8_e4m3fn)
            scale = torch.empty((1, 1), dtype=torch.float8_e8m0fnu)
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(
            frozen.FrozenWeight(codes, scale, (32, 32), kind), requires_grad=False
        )
        module.to('cuda:0')
        assert module.weight.device == module.weight.codes.device == module.weight.scale.device
        assert module.weight.device == torch.device('cuda:0')
        module.cpu()
        assert module.weight.device == module.weight.codes.device == module.weight.scale.device
        assert module.weight.device == torch.device('cpu')
        module.cuda(0)
        assert module.weight.device == module.weight.codes.device == module.weight.scale.device
        assert module.weight.device == torch.device('cuda:0')
