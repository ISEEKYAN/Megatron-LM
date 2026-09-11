"""Independent code cards for checkpoint numerical binding (CPU scope)."""

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite.checkpoint import load_weight
from megatron.lite.model.deepseek_v41.lite.checkpoint_store import CheckpointTensorStore
from safetensors.torch import save_file


def store_tensors(tmp_path, tensors, filename="source.safetensors"):
    path = tmp_path / filename
    save_file(tensors, path)
    return CheckpointTensorStore.load([path], expected_keys=tensors)


def test_fp4_known_codes_loaded_gemm_and_bf16_reload(tmp_path):
    packed = torch.tensor(
        list(bytes.fromhex("10 32 54 76 98 ba dc fe") * 2), dtype=torch.uint8
    )
    packed = packed.repeat(2, 1).view(torch.int8)
    scales = torch.tensor([[127], [128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    name = "layers.0.ffn.experts.0.w1.weight"
    store = store_tensors(tmp_path, {name: packed, name[:-6] + "scale": scales})
    result = load_weight(store, name, output_dtype=torch.float32)
    row = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6] * 2
    expected = torch.tensor([row, [2 * v for v in row]])
    assert torch.equal(result, expected)
    x = torch.arange(32, dtype=torch.float32)
    assert torch.equal(result @ x, expected @ x)
    export = store_tensors(tmp_path, {name: result.bfloat16()}, "export.safetensors")
    assert torch.equal(load_weight(export, name), result.bfloat16())


def test_fp8_blocks_and_loaded_gemm(tmp_path):
    name = "layers.0.attn.wq_a.weight"
    weight = torch.ones(64, 64).to(torch.float8_e4m3fn)
    scale = torch.tensor([[126, 127], [128, 129]], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    store = store_tensors(tmp_path, {name: weight, name[:-6] + "scale": scale})
    decoded = load_weight(store, name, output_dtype=torch.float32)
    expected = torch.tensor([48.0] * 32 + [192.0] * 32)
    assert torch.equal(decoded @ torch.ones(64), expected)
    exported = store_tensors(tmp_path, {name: decoded.bfloat16()}, "export.safetensors")
    assert torch.equal(load_weight(exported, name), decoded.bfloat16())


def test_engram_row_scales_are_not_matrix_block_scales(tmp_path):
    name = "layers.1.engram.embed.weight"
    weight = torch.ones(3, 32).to(torch.float8_e4m3fn)
    scale = torch.tensor([[126], [127], [128]], dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    store = store_tensors(tmp_path, {name: weight, name[:-6] + "scale": scale})
    expected = torch.tensor([0.5, 1, 2])[:, None].expand(3, 32)
    assert torch.equal(load_weight(store, name).float(), expected)


@pytest.mark.parametrize("kind", ["missing", "wrong_shape", "wrong_dtype", "nonfinite"])
def test_quantized_scale_failures(tmp_path, kind):
    name = "layers.0.ffn.experts.0.w1.weight"
    tensors = {name: torch.zeros(2, 16, dtype=torch.int8)}
    if kind != "missing":
        scales = torch.tensor([[127], [127]], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        if kind == "wrong_shape":
            scales = scales[:1]
        elif kind == "wrong_dtype":
            scales = scales.float()
        elif kind == "nonfinite":
            scales.view(torch.uint8)[0, 0] = 255
        tensors[name[:-6] + "scale"] = scales
    store = store_tensors(tmp_path, tensors)
    with pytest.raises((ValueError, TypeError)):
        load_weight(store, name)


def test_plain_weight_rejects_stale_scale(tmp_path):
    name = "norm.weight"
    store = store_tensors(
        tmp_path,
        {name: torch.ones(4, dtype=torch.bfloat16), "norm.scale": torch.ones(1)},
    )
    with pytest.raises(ValueError, match="scale"):
        load_weight(store, name)
