# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from megatron.lite.primitive.quantization.ds41_fp8 import (
    dynamic_fp8_linear,
    fake_quant_swa,
    quantize_swa,
)
from megatron.lite.primitive.quantization.ds41_index import (
    fake_quant_index,
    quantize_index,
)
from megatron.lite.primitive.quantization.ds41_kv import (
    fake_quant_main_kv,
    quantize_main_kv,
)


@pytest.mark.parametrize(
    'codec,group,dtype,scale',
    [
        (quantize_main_kv, 16, torch.float8_e4m3fn, 56),
        (quantize_index, 32, torch.float8_e8m0fnu, 128),
    ],
)
def test_group16_e4m3_vs_group32_e8m0(codec, group, dtype, scale):
    result = codec(torch.full((2, 64), 6.25))
    assert result.scale.shape == (2, 64 // group)
    assert result.scale.dtype == dtype
    assert result.scale.view(torch.uint8).tolist() == [[scale] * (64 // group)] * 2
    assert torch.equal(result.decoded, torch.full((2, 64), 6.0))
    assert codec(torch.zeros(32)).scale.view(torch.uint8).tolist() == [1] * (
        32 // group
    )
    row = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
    values = torch.tensor([row + [-x for x in row]] * 2).reshape(1, 32)
    before = values.clone()
    encoded = codec(values)
    assert encoded.packed.view(torch.uint8).tolist() == [
        [0x20, 0x42, 0x64, 0x76, 0xA8, 0xCA, 0xEC, 0xFE] * 2
    ]
    assert torch.equal(before, values) and not encoded.scale.requires_grad


@pytest.mark.parametrize(
    'main_enabled,index_enabled',
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_independent_qat_switches_and_ste(main_enabled, index_enabled):
    x = torch.full((32,), 0.3)
    x[-1] = 6
    x.requires_grad_()
    main, index = fake_quant_main_kv(x, enabled=main_enabled), fake_quant_index(
        x, enabled=index_enabled
    )
    assert torch.equal(main, quantize_main_kv(x).decoded if main_enabled else x)
    assert torch.equal(index, quantize_index(x).decoded if index_enabled else x)
    for result in (main, index, fake_quant_swa(x)):
        grad = torch.arange(32).float()
        torch.testing.assert_close(
            torch.autograd.grad((result * grad).sum(), x)[0], grad, atol=0, rtol=0
        )


@pytest.mark.parametrize(
    'case,message',
    [
        ('phase', 'post-training'),
        ('width', 'divisible by'),
        ('nan', 'must be finite'),
        ('integer', 'must be F32'),
        ('enabled', 'enabled must be bool'),
        ('fp8_cpu', 'requires CUDA'),
    ],
)
def test_quantization_rejects(case, message):
    x = torch.ones(16)
    if case == 'width':
        x = torch.ones(17)
    elif case == 'nan':
        x.fill_(float('nan'))
    elif case == 'integer':
        x = x.long()
    with pytest.raises((ValueError, TypeError, RuntimeError), match=message):
        if case == 'phase':
            fake_quant_main_kv(x, phase='pretraining')
        elif case == 'enabled':
            fake_quant_index(torch.ones(32), enabled=1)
        elif case == 'fp8_cpu':
            dynamic_fp8_linear(torch.ones(2, 32), torch.ones(32, 32))
        else:
            quantize_main_kv(x)
