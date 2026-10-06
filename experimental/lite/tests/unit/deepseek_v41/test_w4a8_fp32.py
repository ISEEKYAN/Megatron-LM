# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 W4A8: live FP32 masters, native gradients, export and W2 oracle.

w4a8_reference.py is copied unchanged from vLLM W2 commit 5db5732a7d98,
tests/kernels/moe/w4a8_reference.py (Apache-2.0). CPU GEMM is not a GPU oracle.
"""

import os
from pathlib import Path
from types import MethodType

import pytest
import torch
from safetensors.torch import save_file
from test_redo_parity import release_config
from w4a8_reference import quantize_a8, swiglu_clamp, topk_fma


def tiny_config():
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    cfg = release_config().to_hf_dict()
    cfg['text_config'].update(
        hidden_size=128,
        moe_intermediate_size=128,
        num_experts_per_tok=2,
        engram_layer_ids=[],
        engram_num_embeddings=[],
    )
    return DeepseekV41Config(cfg)


def build(*, w4a8=False, optimize=False, dtype=torch.bfloat16):
    from megatron.lite.model.deepseek_v41.lite import protocol

    kwargs = dict(device='cpu', dtype=dtype, quantized=False)
    # Omit the new field entirely for the external 01b252f37 baseline run.
    if w4a8:
        kwargs['w4a8_experts'] = True
    if optimize:
        kwargs.update(
            optimizer='muon',
            optimizer_config=protocol.OptimizerConfig(
                lr=0.02, ns_steps=2, coefficient_type='quintic', clip_grad=0
            ),
        )
    return protocol.build_model(tiny_config(), impl_cfg=protocol.ImplConfig(**kwargs))


def bytes_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def archive(model, path):
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

    path.mkdir()
    # Synthetic archival payload, never executed; exercise normal exporter coverage.
    save_file(
        {name: torch.arange(7, dtype=torch.uint8) for name in model.archival_bindings},
        str(path / 'model.safetensors'),
    )
    model.archival_store = SafeTensorReader(str(path))
    model.archival_keys = sorted(model.archival_bindings)


def decode_w(packed, scale):
    # Independent HF MXFP4 decoder: low nibble first, UE8M0 exponent minus 127.
    raw = packed.view(torch.uint8)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    nib = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
    return (
        table[nib & 7]
        * torch.where(nib & 8 != 0, -1.0, 1.0)
        * torch.exp2(scale.view(torch.uint8).float() - 127).repeat_interleave(32, -1)
    )


def a8(x):
    codes, scales = quantize_a8(x)
    return codes.float() * scales.repeat_interleave(128, -1)


def install_rollout_reference(model, exported):
    for index, layer in enumerate(model.layers):

        def reference(self, x, *, image_mask=None, load_sink=None, index=index):
            flat = x.reshape(-1, x.shape[-1])
            weights, routes, stats = self.gate(flat, image_mask)
            slots = torch.empty((*routes.shape, flat.shape[-1]), dtype=torch.bfloat16)
            # Independent expert dispatch; no training W4A8/dispatcher/combine calls.
            for expert in range(len(self.experts)):
                tokens, positions = torch.where(routes == expert)
                prefix = f'layers.{index}.ffn.experts.{expert}'
                matrices = [
                    decode_w(
                        exported[f'{prefix}.w{k}.weight'],
                        exported[f'{prefix}.w{k}.scale'],
                    )
                    for k in (1, 3, 2)
                ]
                fc1 = (a8(flat[tokens]) @ torch.cat(matrices[:2]).T).bfloat16()
                act = swiglu_clamp(fc1, self.experts._modules['0'].swiglu_limit)
                slots[tokens, positions] = (a8(act) @ matrices[2].T).bfloat16()
            result = topk_fma(slots, weights)
            if self.shared_experts is not None:
                result = result + self.shared_experts(flat)
            if load_sink is not None:
                load_sink.append(stats)
            return result.reshape_as(x)

        layer.ffn.forward = MethodType(reference, layer.ffn)


def test_fp32_wgrad_uses_native_operands_even_under_autocast():
    import megatron.lite.primitive.quantization.w4a8_experts as _imports_w4a8_experts

    dequantize_fp8_act = _imports_w4a8_experts.dequantize_fp8_act
    quantize_fp8_act = _imports_w4a8_experts.quantize_fp8_act
    w4a8_grouped_gemm = _imports_w4a8_experts.w4a8_grouped_gemm

    torch.manual_seed(97)
    x = torch.randn(19, 128).bfloat16().requires_grad_()
    w = torch.nn.Parameter(torch.randn(128, 128))
    grad = torch.randn(19, 128).bfloat16()
    y = w4a8_grouped_gemm(x, [w], [19])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y.backward(grad)
    xhat = dequantize_fp8_act(*quantize_fp8_act(x)).bfloat16().float()
    expected = grad.float().T @ xhat
    bytes_equal(w.grad, expected)
    assert not torch.equal(expected, expected.bfloat16().float())
    assert x.grad.dtype == torch.bfloat16


def test_model_export_reference_and_two_muon_steps(v41_core_te, tmp_path, monkeypatch):
    from inspect import unwrap

    from megatron.lite.model.deepseek_v41.lite import checkpoint
    from megatron.lite.primitive.modules.attention import mhc
    from megatron.lite.primitive.quantization import w4a8_experts as primitive

    # Core's CPU-only mHC bmm cannot mix FP32 coefficients and BF16 residuals.
    # Use its existing native arithmetic on FP32 operands in both arms; the
    # caller retains the BF16 boundary. No expert/quantization/routing is mocked.
    native = unwrap(mhc.native_h_post_bda)
    monkeypatch.setattr(
        mhc,
        'native_h_post_bda',
        lambda *args: native(
            *(x.float() if isinstance(x, torch.Tensor) else x for x in args)
        ),
    )
    torch.manual_seed(315)
    bundle = build(w4a8=True, optimize=True)
    model, optimizer = bundle.chunks[0], bundle.optimizer
    archive(model, tmp_path / 'archive')
    bindings = {
        b.release_key: b for b in model.parameter_bindings() if b.role == 'expert'
    }
    assert all(b.tensor.dtype == torch.float32 for b in bindings.values())
    # Exact midpoint-neighbor, zero, negative-zero and tiny deploy-codec cases.
    with torch.no_grad():
        for b in bindings.values():
            b.tensor[0].zero_()
            b.tensor[0, :4] = torch.tensor([0.7501, 6.0, -0.0, 1e-38])
            b.tensor[1].zero_()
    ids = torch.tensor([[2, 3, 9]])
    previous = None
    for generation in range(3):
        exported = {
            k: v.clone() for k, v in checkpoint.export_checkpoint(model, cpu=True)
        }
        master_before = {k: b.tensor.detach().clone() for k, b in bindings.items()}
        captured = []
        quantize = primitive.quantize_mxfp4

        def capture(w):
            result = quantize(w)
            captured.append(tuple(t.detach().clone() for t in result))
            assert w.dtype == torch.float32
            return result

        with monkeypatch.context() as patch:
            patch.setattr(primitive, 'quantize_mxfp4', capture)
            actual = model(ids)['logits']
        assert len(captured) == len(model.layers) * 4
        cursor = iter(captured)
        for layer in range(len(model.layers)):
            fc1 = [next(cursor) for _ in range(2)]
            fc2 = [next(cursor) for _ in range(2)]
            for expert in range(2):
                for k, pair in (
                    (1, tuple(t[:128] for t in fc1[expert])),
                    (3, tuple(t[128:] for t in fc1[expert])),
                    (2, fc2[expert]),
                ):
                    key = f'layers.{layer}.ffn.experts.{expert}.w{k}'
                    bytes_equal(pair[0], exported[key + '.weight'])
                    bytes_equal(pair[1], exported[key + '.scale'])
        for k, b in bindings.items():
            bytes_equal(b.tensor, master_before[k])

        reference = build(w4a8=True, optimize=True).chunks[0]
        reference.load_state_dict(model.state_dict())
        install_rollout_reference(reference, exported)
        with torch.no_grad():
            bytes_equal(actual, reference(ids)['logits'])

        # Real save/load keeps FP32 masters, so re-export is a byte mirror.
        if generation == 2:
            checkpoint.save_model(model, tmp_path / 'saved')
            restored = build(w4a8=True, optimize=True).chunks[0]
            checkpoint.load_model(restored, tmp_path / 'saved')
            again = dict(checkpoint.export_checkpoint(restored, cpu=True))
            for key in bindings:
                bytes_equal(restored.tensor_bindings[key].tensor, master_before[key])
                bytes_equal(again[key], exported[key])
                bytes_equal(again[key[:-6] + 'scale'], exported[key[:-6] + 'scale'])

        # Online receiver/export parity is covered by the separate resync tests.
        payload = torch.cat([exported[k].view(torch.uint8).flatten() for k in bindings])
        if previous is not None:
            assert not torch.equal(payload, previous)
        previous = payload
        if generation < 2:
            optimizer.zero_grad()
            actual.square().mean().backward()
            assert all(b.tensor.grad.dtype == torch.float32 for b in bindings.values())
            assert optimizer.step()[0]


@pytest.mark.parametrize('kwargs', [dict(dtype=torch.float32), dict(use_deepep=True)])
def test_reject_incompatible_config(v41_core_te, kwargs):
    from megatron.lite.model.deepseek_v41.lite import protocol

    with pytest.raises(ValueError, match='W4A8 requires'):
        protocol.build_model(
            tiny_config(),
            impl_cfg=protocol.ImplConfig(device='cpu', w4a8_experts=True, **kwargs),
        )


def test_reject_ep_fused_dispatch_and_misaligned_shapes(v41_core_te, monkeypatch):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    with pytest.raises(ValueError, match='EP'):
        protocol.build_model(
            tiny_config(),
            impl_cfg=protocol.ImplConfig(
                device='cpu', w4a8_experts=True, parallel=ParallelConfig(ep=2)
            ),
        )
    with monkeypatch.context() as patch:
        patch.setenv('MEGATRON_LITE_MOE_PERMUTE_FUSION', '1')
        with pytest.raises(ValueError, match='unfused dispatch'):
            build(w4a8=True)
    with pytest.raises(ValueError, match='divisible by 128'):
        protocol.build_model(
            release_config(),
            impl_cfg=protocol.ImplConfig(
                device='cpu', quantized=False, w4a8_experts=True
            ),
        )


def test_default_snapshot(v41_core_te):
    """Run this exact test in both trees; compare serialized tensor byte views."""
    from megatron.lite.model.deepseek_v41.lite import protocol

    assert Path(protocol.__file__).is_relative_to(Path(__file__).resolve().parents[3])
    torch.manual_seed(413)
    model = build(optimize=True, dtype=torch.float32).chunks[0]
    out = model(torch.tensor([[2, 3, 9]]))['logits']
    moe_bf16 = model.layers[0].ffn(torch.randn(3, 128).bfloat16())
    loss = out.square().mean() + moe_bf16.float().square().mean()
    loss.backward()
    tensors = {
        'logits': out.detach(),
        'loss': loss.detach(),
        'moe_bf16': moe_bf16.detach(),
    }
    tensors.update({'weight:' + n: p.detach() for n, p in model.named_parameters()})
    tensors.update(
        {'grad:' + n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    )
    if os.environ.get('DS41_DEFAULT_SNAPSHOT'):
        torch.save(
            {
                k: v.contiguous().reshape(-1).view(torch.uint8)
                for k, v in tensors.items()
            },
            os.environ['DS41_DEFAULT_SNAPSHOT'],
        )
    assert torch.isfinite(out).all()
