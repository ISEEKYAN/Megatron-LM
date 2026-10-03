# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4 CSA as a consumer of the opt-in FP32-master Linear providers.

CPU cases construct real CSA / V4 layers. When Megatron Core's experimental CSA
kernels or Transformer Engine are unavailable, only the import-time symbols CSA
needs are stubbed; RMSNorm becomes ``torch.nn.RMSNorm``. No forward runs on CPU.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from megatron.lite.primitive.modules import native_fp32_linear as nfl
from megatron.lite.primitive.quantization import mxfp8

_PROJECTIONS = ("wq_a", "wq_b", "wkv", "wo_b")
_CORE_SYMBOLS = {
    "megatron.core.tensor_parallel.mappings": ["gather_from_sequence_parallel_region"],
    "megatron.core.transformer.experimental_attention_variant": [
        "csa_cp_layout_kernels",
        "csa_cp_utils",
    ],
    "megatron.core.transformer.experimental_attention_variant.csa": [
        "_unfused_indexer_sparse_attn_from_topk",
        "unfused_compressed_sparse_attn",
    ],
    "megatron.core.transformer.experimental_attention_variant.csa_kernels": [
        "FusedCSAIndexerSparseAttnFromTopkFunc",
        "csa_sparse_attn",
    ],
    "megatron.core.transformer.experimental_attention_variant.dsa": [
        "DSAIndexerLossAutoScaler",
        "DSAIndexerLossLoggingHelper",
    ],
}
_STUBBED_IMPORTERS = (
    "megatron.lite.primitive.modules.attention",
    "megatron.lite.model.deepseek_v4",
)


@pytest.fixture
def v4(transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    before, stubbed = set(sys.modules), False
    try:
        from megatron.lite.model.deepseek_v4.lite import model, protocol
    except ImportError:
        stubbed = True
        for name, symbols in _CORE_SYMBOLS.items():
            module = types.ModuleType(name)
            for symbol in symbols:
                setattr(module, symbol, object())
            monkeypatch.setitem(sys.modules, name, module)
        for name in set(sys.modules) - before:
            if name.startswith(_STUBBED_IMPORTERS):
                del sys.modules[name]
        from megatron.lite.model.deepseek_v4.lite import model, protocol
    from megatron.lite.primitive import transformer_engine as te

    monkeypatch.setattr(te, "RMSNorm", lambda size, eps: nn.RMSNorm(size, eps=eps))
    yield SimpleNamespace(model=model, protocol=protocol)
    # Modules imported against the stubs must not leak into later tests.
    for name in set(sys.modules) - before if stubbed else ():
        if name.startswith(_STUBBED_IMPORTERS):
            del sys.modules[name]


def _config():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    return DeepseekV4Config(
        hidden_size=64,
        num_attention_heads=2,
        head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        compress_ratios=[4, 0],
        index_head_dim=32,
        index_n_heads=2,
        num_hidden_layers=2,
    )


def _ps():
    return SimpleNamespace(cp_size=1, cp_rank=0, cp_group=None)


def _csa(v4, seed=0, **kwargs):
    torch.manual_seed(seed)
    return v4.model.CompressedSparseAttention(
        _config(), layer_idx=0, ps=_ps(), **kwargs
    )


def test_default_csa_construction_is_unchanged(v4):
    omitted = _csa(v4)
    explicit = _csa(v4, linear_provider=None)
    default = _csa(v4, linear_provider=nfl.linear_provider("default"))
    for name in _PROJECTIONS:
        assert type(getattr(omitted, name)) is nn.Linear
    reference = omitted.state_dict()
    for other in (explicit, default):
        state = other.state_dict()
        assert list(state) == list(reference)
        assert all(torch.equal(state[key], reference[key]) for key in reference)


def test_block32_provider_keeps_names_and_fp32_masters(v4):
    default = _csa(v4).to(torch.bfloat16)
    provided = _csa(v4, linear_provider=nfl.linear_provider("block32_fp8"))
    nfl.restore_fp32_masters(provided.to(torch.bfloat16))
    default_state, state = default.state_dict(), provided.state_dict()
    assert list(state) == list(default_state)
    assert all(state[key].shape == default_state[key].shape for key in state)
    for key, tensor in state.items():
        projection = key.split(".")[0] in _PROJECTIONS
        assert tensor.dtype == (
            torch.float32 if projection else default_state[key].dtype
        )
    for name in _PROJECTIONS:
        assert getattr(provided, name).mode == "block32_fp8"
    # Only the four dense projections move; compressor / indexer / wo_a stay.
    assert type(provided.wo_a) is type(default.wo_a)
    assert type(provided.indexer.wq_b) is nn.Linear


def test_v4_layers_thread_provider_to_decoder_and_mtp_attention(v4, monkeypatch):
    class _Stub(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    for name in ("DeepseekV4MoE", "HyperConnection", "MultiHeadHyperConnectionHead"):
        monkeypatch.setattr(v4.model, name, _Stub)
    provider = nfl.linear_provider("native_fp32")
    layer = v4.model.DeepseekV4Layer(_config(), _ps(), 0, linear_provider=provider)
    mtp = v4.model.DeepseekV4MTPLayer(
        _config(),
        _ps(),
        2,
        embedding=_Stub(),
        use_deepep=False,
        detach_encoder=False,
        linear_provider=provider,
    )
    default = v4.model.DeepseekV4Layer(_config(), _ps(), 1)
    for name in _PROJECTIONS:
        assert getattr(layer.self_attn.self_attn, name).mode == "native_fp32"
        assert getattr(mtp.self_attn.self_attn, name).mode == "native_fp32"
        assert type(getattr(default.self_attn.self_attn, name)) is nn.Linear
    assert type(mtp.e_proj) is nn.Linear


def test_protocol_selects_provider_opt_in_and_rejects_unsupported(v4):
    ImplConfig = v4.protocol.ImplConfig
    select = v4.protocol._attention_linear_provider
    assert ImplConfig().attention_linear == "default"
    assert select(ImplConfig()) is None
    built = select(ImplConfig(attention_linear="block32_fp8"))(64, 32, bias=False)
    assert built.mode == "block32_fp8" and built.weight.dtype == torch.float32
    with pytest.raises(ValueError, match="unknown linear provider"):
        select(ImplConfig(attention_linear="mxfp4"))
    with pytest.raises(ValueError, match="QAT"):
        select(ImplConfig(attention_linear="native_fp32", qat={"enabled": True}))
    with pytest.raises(ValueError, match="fsdp2"):
        select(ImplConfig(attention_linear="block32_fp8", optimizer="fsdp2"))
    assert select(ImplConfig(qat={"enabled": True})) is None


@pytest.fixture
def single_rank_nccl(tmp_path):
    import torch.distributed as dist

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(
            "nccl", init_method=f"file://{tmp_path / 'nccl'}", rank=0, world_size=1
        )
    assert dist.get_world_size() == 1
    try:
        yield dist.group.WORLD
    finally:
        if created:
            dist.destroy_process_group()


def _cuda_config():
    config = _config()
    config.num_attention_heads, config.head_dim = 64, 512
    config.qk_rope_head_dim, config.index_head_dim, config.index_n_heads = 64, 128, 64
    config.sliding_window, config.index_topk = 4, 4
    return config


class _DecodedLinear(nn.Module):
    """Independent FP64 block sums and decoded FP32 autograd with identity STE.

    Does not call the provider's forward or backward. Autograd differentiates
    the FP32 matmul directly, returning FP32 dW and activation-dtype dX.
    """

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        activation = mxfp8.quantize_block32(flat)
        weight = mxfp8.quantize_block32(self.weight, mxfp8.WEIGHT_BLOCK)
        # Exact FP64 sums of E4M3 products independently check each real FP8
        # GEMM, followed by the specified FP32 scaling and block accumulation.
        value = torch.zeros(
            flat.shape[0], self.weight.shape[0], device=x.device, dtype=torch.float32
        )
        for block, start in enumerate(range(0, flat.shape[1], 32)):
            product = (
                activation.values[:, start : start + 32].double()
                @ weight.values[:, start : start + 32].double().T
            ).float()
            value += (
                product
                * activation.scale[:, block].float()[:, None]
                * weight.scale[:, block].float().repeat_interleave(32)[None, :]
            )
        decoded_x = activation.decoded.float()
        decoded_w = weight.decoded.float()
        x_ste = flat.float() + (decoded_x - flat.float()).detach()
        w_ste = self.weight + (decoded_w - self.weight).detach()
        surrogate = F.linear(x_ste, w_ste)
        # Use the independent block result in forward and ordinary PyTorch
        # matmul autograd for the native FP32 derivative contract.
        result = value.detach() + (surrogate - surrogate.detach())
        return result.reshape(*x.shape[:-1], self.weight.shape[0]).to(x.dtype)


def _assert_consumer_close(actual, expected):
    # Real FP8 tensor-core accumulation differs from exact FP64 block sums.
    # BF16 rounding and subsequent block quantization can amplify that error.
    # The consumer contract bounds BOTH relative L2 error (2%) and maximum
    # absolute error / reference maximum (5%), for output, dX and every dW.
    # Normalizing max error by the tensor scale avoids a vacuous large absolute
    # tolerance for tiny gradients, and unstable relative errors near zero.
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    expected = expected.detach().float()
    delta = actual.detach().float() - expected
    assert expected.norm() > 0 and expected.abs().max() > 0
    assert delta.norm() <= 0.02 * expected.norm()
    assert delta.abs().max() <= 0.05 * expected.abs().max()


@pytest.mark.gpus(1)
def test_csa_forward_backward_calls_fp8_gemm_with_fp32_wgrad(
    single_rank_nccl, monkeypatch
):
    # Real TE and fused DSA kernels: no import or RMSNorm stubs on CUDA.
    from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention

    torch.manual_seed(17)
    config = _cuda_config()
    ps = _ps()
    ps.cp_group = single_rank_nccl
    module = CompressedSparseAttention(
        config, layer_idx=0, ps=ps, linear_provider=nfl.linear_provider("block32_fp8")
    )
    module = nfl.restore_fp32_masters(module.to(device="cuda", dtype=torch.bfloat16))
    module.attention_backend = "flash"
    module.apply_dsa_kernel_fusion = True
    # Eval suppresses the indexer auxiliary training loss, but retains autograd.
    module.eval()
    # ProcessGroup is not picklable; both consumers use the same single rank.
    reference = copy.deepcopy(module, memo={id(ps): ps})
    for name in _PROJECTIONS:
        setattr(reference, name, _DecodedLinear(getattr(module, name).weight))
    calls = []
    scaled_mm = torch._scaled_mm

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return scaled_mm(*args, **kwargs)

    monkeypatch.setattr(torch, "_scaled_mm", spy)
    x = torch.randn(
        1,
        8,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    ref_x = x.detach().clone().requires_grad_()
    positions = torch.arange(8, device="cuda").unsqueeze(0)
    out = module(x, position_ids=positions)
    expected = reference(ref_x, position_ids=positions)
    assert calls
    _assert_consumer_close(out, expected)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    _assert_consumer_close(x.grad, ref_x.grad)
    for name in _PROJECTIONS:
        actual = getattr(module, name).weight.grad
        wanted = getattr(reference, name).weight.grad
        assert actual.dtype == wanted.dtype == torch.float32
        assert torch.count_nonzero(wanted) > 0
        _assert_consumer_close(actual, wanted)


# Pin the pre-provider main sources, not a second construction of HEAD. Only
# these three existing production files differ from main in this feature.
_MAIN_REVISION = "d8e010069fef5c6da3690d008b899d480864a8de"
_MAIN_BLOBS = {
    "megatron.lite.primitive.modules.attention.csa": "2f6be962a418f5e23a2012777bd469358b8cb328",
    "megatron.lite.model.deepseek_v4.lite.model": "dc22977213fd439db1dabb088e2957c1ce0a6d76",
    "megatron.lite.model.deepseek_v4.lite.protocol": "5d0441ad00e0839e1f07720c8b44461e1f0e2295",
}


def _main_source(name):
    relative = "experimental/lite/" + name.replace(".", "/") + ".py"
    # An exported main tree supports GPU runners without a Git checkout.
    exported = os.environ.get("MLITE_MAIN_REFERENCE")
    if exported:
        source = (Path(exported) / relative).read_bytes()
    else:
        source = subprocess.check_output(
            ["git", "show", f"{_MAIN_REVISION}:{relative}"],
            cwd=Path(__file__).resolve().parents[7],
        )
    blob = b"blob " + str(len(source)).encode() + b"\0" + source
    assert hashlib.sha1(blob).hexdigest() == _MAIN_BLOBS[name]
    return source


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8)


@pytest.mark.gpus(1)
def test_default_build_model_is_byte_identical_to_main(single_rank_nccl, monkeypatch):
    from megatron.lite.model.deepseek_v4.lite import protocol

    config = _cuda_config()
    config.vocab_size = 32
    config.moe_intermediate_size = 32
    config.n_routed_experts = 2
    config.num_experts_per_tok = 1
    config.num_nextn_predict_layers = 0
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda")

    def run(implementation):
        torch.manual_seed(29)
        bundle = implementation.build_model(
            copy.deepcopy(config),
            impl_cfg=implementation.ImplConfig(
                optimizer=None, mtp_enable=False, attention_backend_override="flash"
            ),
        )
        model = bundle.chunks[0].eval()
        with torch.no_grad():
            outputs = model(input_ids=ids, enable_mtp=False)
        assert set(outputs) == {"hidden_states", "logits"}
        assert all(torch.isfinite(value).all() for value in outputs.values())
        return (
            {
                key: (value.dtype, value.shape, _bytes(value))
                for key, value in model.state_dict().items()
            },
            {
                key: (value.dtype, value.shape, _bytes(value))
                for key, value in outputs.items()
                if isinstance(value, torch.Tensor)
            },
        )

    actual = run(protocol)
    # Load the pinned main CSA -> model -> protocol in dependency order. All
    # unrelated primitives and real GPU kernels are shared by the two runs.
    with monkeypatch.context() as patch:
        for name in _MAIN_BLOBS:
            spec = importlib.util.spec_from_loader(name, loader=None)
            module = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, name, module)
            exec(compile(_main_source(name), f"main:{name}", "exec"), module.__dict__)
        expected = run(module)
    for actual_group, expected_group in zip(actual, expected):
        assert actual_group.keys() == expected_group.keys()
        assert actual_group
        for key in expected_group:
            dtype, shape, data = actual_group[key]
            ref_dtype, ref_shape, ref_data = expected_group[key]
            assert (dtype, shape) == (ref_dtype, ref_shape), key
            assert torch.equal(data, ref_data), key
