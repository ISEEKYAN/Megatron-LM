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


def test_streamed_checkpoint_provider_stays_on_gpu(tmp_path):
    from safetensors.torch import save_file
    from megatron.lite.model.deepseek_v41.lite.checkpoint_store import CheckpointTensorStore
    from megatron.lite.model.deepseek_v41.lite.engram import ShardedEngramTable
    from megatron.lite.primitive.modules.engram_lookup import RowLookup

    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    name = 'layers.0.engram.embed.weight'
    values = (torch.arange(7 * 256) % 120).byte().reshape(7, 256).view(torch.float8_e4m3fn)
    scales = torch.full((7, 8), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    path = tmp_path / 'table.safetensors'
    save_file({name: values, name[:-6] + 'scale': scales}, path)
    store = CheckpointTensorStore.load([path], expected_keys=[name, name[:-6] + 'scale'])
    ids = torch.tensor([[6, 0, 6]], device='cuda')
    for trainable in (False, True):
        table = ShardedEngramTable.from_checkpoint(store, name, RowLookup((0, 7)), device='cuda', trainable=trainable, chunk_rows=2)
        raw, scale, master = table.lookup_fp8(ids)
        assert table.weight.device.type == table.scale.device.type == 'cuda'
        assert torch.equal(raw.view(torch.uint8).cpu(), values.view(torch.uint8)[ids.cpu()])
        assert torch.equal(scale.view(torch.uint8).cpu(), scales.view(torch.uint8)[ids.cpu()])
        if trainable:
            assert master.device.type == 'cuda' and master.dtype == torch.float32
        else:
            assert master is None and not list(table.parameters())


@pytest.mark.parametrize('tokens', [0, 1, 3])
def test_native_projection_handles_empty_and_uneven_token_batches(tokens):
    from megatron.lite.primitive.quantization.ds41_fp8 import published_fp8_linear
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    values = torch.ones(tokens, 64, device='cuda').to(torch.float8_e4m3fn)
    scales = torch.ones(tokens, 2, device='cuda').to(torch.float8_e8m0fnu)
    weight = torch.ones(32, 64, device='cuda', requires_grad=True)
    master = torch.ones(tokens, 64, device='cuda', requires_grad=True)
    result = published_fp8_linear(values, scales, weight, master=master, output_dtype=torch.float32)
    torch.testing.assert_close(result, torch.full((tokens, 32), 64.0, device='cuda'), atol=0, rtol=0)
    result.sum().backward()
    torch.testing.assert_close(weight.grad, torch.full_like(weight, float(tokens)), atol=0, rtol=0)
    torch.testing.assert_close(master.grad, torch.full_like(master, 32.0), atol=0, rtol=0)
