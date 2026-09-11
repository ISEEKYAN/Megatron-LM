import pytest
import torch

from megatron.lite.primitive.quantization.ds41_fp8 import published_fp8_linear
from megatron.lite.model.deepseek_v41.lite.engram import EngramTable, EngramFP8Projection, Engram


def test_raw_projection_rejects_cpu_instead_of_fallback():
    values = torch.ones(16, 32).to(torch.float8_e4m3fn)
    scales = torch.ones(16, 1).to(torch.float8_e8m0fnu)
    with pytest.raises(RuntimeError, match='CUDA'):
        published_fp8_linear(values, scales, torch.ones(32, 32))


def test_engram_passes_published_bytes_to_projection(monkeypatch):
    import megatron.lite.primitive.quantization.ds41_fp8 as fp8
    values = torch.arange(64).byte().reshape(2, 32).view(torch.float8_e4m3fn)
    scales = torch.tensor([[126], [129]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    table = EngramTable(values, scales, trainable=True)
    projection = EngramFP8Projection(torch.ones(32, 64))
    ids = torch.tensor([[[1, 0]]])
    seen = []
    def capture(raw, scale, weight, *, master=None, output_dtype=torch.bfloat16):
        seen.append((raw, scale, master))
        return torch.ones(1, 1, 32, dtype=output_dtype)
    monkeypatch.setattr(fp8, 'published_fp8_linear', capture)
    model = Engram(16, 1, table, projection)
    assert model(torch.ones(1, 1, 1, 16), ids).shape == (1, 1, 1, 16)
    assert len(seen) == 1
    raw, scale, master = seen[0]
    assert torch.equal(raw.view(torch.uint8), values.view(torch.uint8)[ids].flatten(-2))
    assert torch.equal(scale.view(torch.uint8), scales.view(torch.uint8)[ids].flatten(-2))
    assert master.dtype == torch.float32 and master.requires_grad
