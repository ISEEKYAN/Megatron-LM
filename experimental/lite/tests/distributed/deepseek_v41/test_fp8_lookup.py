"""Native GPU arithmetic acceptance, run through tests7064.sbatch."""
import os

import pytest
import torch


@pytest.mark.parametrize('trainable', [False, True])
def test_published_lookup_native_gemm_and_backward(trainable, monkeypatch):
    from megatron.lite.model.deepseek_v41.lite.engram import ShardedEngramTable, EngramFP8Projection
    from megatron.lite.primitive.modules.engram_lookup import RowLookup
    import megatron.lite.primitive.quantization.ds41_fp8 as fp8

    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    values = ((torch.arange(7 * 256, device='cuda') % 3) - 1).float().reshape(7, 256).to(torch.float8_e4m3fn)
    scales = (2.0 ** (torch.arange(7 * 8, device='cuda') % 3 - 1)).reshape(7, 8).to(torch.float8_e8m0fnu)
    table = ShardedEngramTable(values, scales, RowLookup((0, 7)), trainable=trainable)
    ids = (torch.arange(16 * 24, device='cuda') % 7).reshape(16, 24)
    weight = ((torch.arange(32 * 6144, device='cuda') % 3) - 1).float().reshape(32, 6144)
    projection = EngramFP8Projection(weight, output_dtype=torch.float32)
    captured = []
    native = fp8._fp8_gemm
    def observe(a, a_scale, b, b_scale):
        captured.append((a.detach().clone(), a_scale.detach().clone()))
        return native(a, a_scale, b, b_scale)
    monkeypatch.setattr(fp8, '_fp8_gemm', observe)
    def forbidden(*args, **kwargs):
        raise AssertionError('Published activations were requantized')
    monkeypatch.setattr(fp8, 'quantize_linear_activation', forbidden)
    result = projection.forward_lookup(table, ids)
    assert len(captured) == 1
    assert torch.equal(captured[0][0].view(torch.uint8), values.view(torch.uint8)[ids].flatten(-2))
    assert torch.equal(captured[0][1].view(torch.uint8), scales.view(torch.uint8)[ids].flatten(-2))
    decoded = (values.float() * scales.float().repeat_interleave(32, -1))[ids].flatten(-2)
    # The chosen {-1,0,1} weight is exactly representable by weight quantization.
    expected = decoded @ weight.T
    torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-5)
    result.sum().backward()
    torch.testing.assert_close(projection.weight.grad, torch.ones_like(expected).T @ decoded, atol=1e-3, rtol=1e-5)
    if trainable:
        contribution = weight.sum(0).reshape(24, 256)
        expected_grad = torch.zeros(7, 256, device='cuda')
        for token in range(16):
            for head in range(24):
                expected_grad[ids[token, head]] += contribution[head]
        assert table.master.grad.dtype == torch.float32
        torch.testing.assert_close(table.master.grad, expected_grad, atol=1e-3, rtol=1e-5)
