# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reduced single-rank V4.1 semantics and gradient checks."""

from types import SimpleNamespace

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import attention as attn
from megatron.lite.model.deepseek_v41.lite import block as hc
from megatron.lite.model.deepseek_v41.lite import candidates, engram, packing
from megatron.lite.primitive.quantization import ds41_fp8


@pytest.fixture
def moe(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v41.lite import moe

    return moe


@pytest.mark.parametrize('layer', [2, 8, 14, 20])
def test_v41_owner_reuse_reindex_gradients(layer):
    torch.manual_seed(42)
    config = attn.CSA2Config(
        dim=32,
        heads=2,
        head_dim=32,
        rope_dim=4,
        q_rank=32,
        o_rank=4,
        groups=2,
        index_heads=2,
        index_dim=32,
        topk=2,
        window=3,
        candidate_blocks=1,
        block_size=2,
        linear_fp8=False,
        main_qat=False,
        index_qat=False,
        swa_fp8=False,
    )
    owner, reuse = [attn.CSA2Attention(config, n).float() for n in (layer, layer + 1)]
    x = torch.randn(1, 4, 32, requires_grad=True)
    _, state = owner(x, attn.AttentionState())
    y, shared = reuse(x, state)
    assert shared.main_kv is state.main_kv and shared.indices is state.indices
    assert all((not p.requires_grad for p in owner.indexer.parameters()))
    assert torch.count_nonzero(torch.autograd.grad(y.square().sum(), state.main_kv)[0])
    if layer == 20:
        reindex = attn.CSA2Attention(config, 24).float()
        _, refreshed = reindex(x, state)
        assert refreshed.index_owner == 24 and refreshed.main_kv is state.main_kv
    block = hc.DeepseekV41Block(32, 2, owner, torch.nn.Linear(32, 32)).float()

    def sequence(h, p):
        h, p, _ = block.forward_with_state(h, p, attn.AttentionState())
        return (h, p)

    h, p = hc.expand_hc(x, 2)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    actual, mixes = packing.packed_forward(sequence, h, p, cu)
    expected = [sequence(h[:, n : n + 2], p[:, n : n + 2]) for n in (0, 2)]
    torch.testing.assert_close(actual, torch.cat([v[0] for v in expected], 1))
    torch.testing.assert_close(mixes, torch.cat([v[1] for v in expected], 1))
    grad = torch.autograd.grad(actual[:, 2:].square().sum(), x)[0]
    assert not grad[:, :2].any() and grad[:, 2:].any()


@pytest.mark.parametrize('trainable', [False, True])
def test_v41_engram_storage_and_reset(trainable):
    encoded = ds41_fp8.quantize_swa(torch.linspace(-2, 2, 96).reshape(3, 32))
    table = engram.EngramTable(encoded.values, encoded.scale, trainable=trainable)
    ids = torch.tensor([[0, 0, 2]])
    original = table(ids).float()
    table.bfloat16()
    assert table.weight.dtype == torch.float8_e4m3fn
    assert table.scale.dtype == torch.float8_e8m0fnu
    if trainable:
        table(ids).float().sum().backward()
        assert table.master.dtype == torch.float32
        torch.testing.assert_close(
            table.master.grad, torch.tensor([2.0, 0.0, 1.0])[:, None].expand(3, 32)
        )
        with torch.no_grad():
            table.master.mul_(16)
        table.refresh_storage()
        torch.testing.assert_close(table(ids).float(), original * 16)
    else:
        assert not list(table.parameters()) and 'master' not in table.state_dict()
    hasher = engram.NgramHash(
        [0, 1, 2, 3], 0, torch.tensor([[3, 5, 7]]), torch.tensor([[[11], [13]]])
    )
    hashed = hasher(torch.tensor([[1, 2, 3]]), cu_seqlens=torch.tensor([0, 2, 3]))
    assert hashed.tolist() == [[[[3, 14]], [[3, 14]], [[9, 20]]]]


@pytest.mark.parametrize('image', [False, True])
def test_v41_modality_bias_selection_and_vjp(image, moe):
    config = SimpleNamespace(
        hidden_size=3,
        n_routed_experts=3,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        scoring_func='sqrtsoftplus',
    )
    router = moe.ModalityRouter(config, SimpleNamespace(tp_size=1), gate_temperature=0.7)
    with torch.no_grad():
        router.router.gate.weight.copy_(torch.eye(3))
        (router.bias_vl if image else router.bias)[2] = 4
    x = torch.tensor([[2.0, 1.0, -1.0]], requires_grad=True)
    weights, indices, stats = router(x, torch.tensor([image]))
    assert indices.tolist() == [[0, 2]]
    raw = torch.nn.functional.softplus(x / 0.7).sqrt()[:, [0, 2]]
    expected = raw / raw.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(weights, expected)
    cotangent = torch.tensor([[1.0, -2.0]])
    torch.testing.assert_close(
        torch.autograd.grad((weights * cotangent).sum(), x)[0],
        torch.autograd.grad((expected * cotangent).sum(), x)[0],
    )
    before = torch.stack([router.bias.clone(), router.bias_vl.clone()])
    router.update_bias(stats)
    delta = torch.zeros(2, 3)
    delta[int(image)] = torch.tensor([-0.001, 0.001, -0.001])
    torch.testing.assert_close(torch.stack([router.bias, router.bias_vl]), before + delta)
    assert not router.router.compute_aux_loss


@pytest.mark.parametrize('visible', [0, 3, 9])
def test_v41_candidate_pool_visibility(visible):
    scores = torch.arange(9, 0, -1).float().reshape(1, 1, 9)
    pool = candidates.candidate_blocks(scores, visible, topk_blocks=2, block_size=2)
    later = scores.clone()
    later[..., 6] = 100
    selected = candidates.select_positions(later, visible, 2, candidates=pool)
    assert not ((selected >= visible) & (selected != -1)).any()
    if visible == 9:
        assert pool[..., -1].all() and (not pool[..., 6].any())
        assert 6 not in selected
    if visible == 0:
        assert (selected == -1).all()


@pytest.mark.parametrize('copies', [2, 4])
def test_v41_mhc_source_destination_orientation(copies):
    torch.manual_seed(9)
    residual = torch.randn(1, 2, copies, 3, requires_grad=True)
    output = torch.randn(1, 2, 3, requires_grad=True)
    post = torch.randn(1, 2, copies, requires_grad=True)
    comb = torch.randn(1, 2, copies, copies, requires_grad=True)
    expected = torch.stack(
        [
            post[..., j, None] * output
            + sum((comb[..., i, j, None] * residual[..., i, :] for i in range(copies)))
            for j in range(copies)
        ],
        -2,
    )
    actual = hc.mix_residual(output, residual, post, comb)
    torch.testing.assert_close(actual, expected)
    args = (output, residual, post, comb)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), args),
        torch.autograd.grad(expected.square().sum(), args),
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("copies", [2, 3, 4])
def test_v41_mhc_two_sublayer_shift_uses_unequal_coefficients(copies):
    """Independently derive the two shifted HC inputs, not block internals."""

    class FixedMix(torch.nn.Module):
        def __init__(self, pre):
            super().__init__()
            self.pre = pre

        def forward(self, hidden):
            copies = hidden.shape[-2]
            post = torch.full_like(self.pre, 0.25)
            comb = torch.eye(copies).expand(*hidden.shape[:2], -1, -1)
            return self.pre, post, comb

    class RecordInput(torch.nn.Module):
        def __init__(self, increment):
            super().__init__()
            self.increment = increment
            self.inputs = []

        def forward(self, x):
            self.inputs.append(x.detach().clone())
            return x + self.increment

    hidden = torch.arange(1, 1 + copies * 2).reshape(1, 1, copies, 2).float()
    pre_mix = torch.arange(1, copies + 1).reshape(1, 1, copies).float()
    pre_mix = pre_mix / pre_mix.sum(-1, keepdim=True)
    expected_hidden, expected_pre = hidden.clone(), pre_mix.clone()
    for layer in range(2):
        attn_pre = pre_mix.roll(layer + 1, -1) * (layer + 2)
        ffn_pre = pre_mix.flip(-1) * (layer + 3)
        attention, ffn = RecordInput(17), RecordInput(-9)
        block = hc.DeepseekV41Block(2, copies, attention, ffn)
        block.attn_norm = torch.nn.Identity()
        block.ffn_norm = torch.nn.Identity()
        block.attn_mixes = FixedMix(attn_pre)
        block.ffn_mixes = FixedMix(ffn_pre)

        hidden, returned_pre = block(hidden, pre_mix)

        # Independent equations across both sublayers AND the next block boundary.
        expected_attn_input = (expected_hidden * expected_pre.unsqueeze(-1)).sum(-2)
        expected_hidden = expected_hidden + 0.25 * (expected_attn_input + 17).unsqueeze(-2)
        expected_ffn_input = (expected_hidden * attn_pre.unsqueeze(-1)).sum(-2)
        wrong_ffn_input = (expected_hidden * expected_pre.unsqueeze(-1)).sum(-2)
        expected_hidden = expected_hidden + 0.25 * (expected_ffn_input - 9).unsqueeze(-2)
        torch.testing.assert_close(attention.inputs[0], expected_attn_input)
        torch.testing.assert_close(ffn.inputs[0], expected_ffn_input)
        torch.testing.assert_close(hidden, expected_hidden)
        torch.testing.assert_close(returned_pre, ffn_pre)
        assert not torch.allclose(expected_ffn_input, wrong_ffn_input)
        pre_mix, expected_pre = returned_pre, ffn_pre.clone()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [1, 9])
def test_v41_ced_boundaries_match_pinned_official_oracle(dtype, length, monkeypatch):
    """Compare the real block's CED chain with B1's pinned pre-RoPE methods.

    Execute original methods as CPU cards, as in C's test_oracle.py. Stop the
    official indexer at its k_norm hook: RoPE/FP4 publication follows this AC's
    boundary and is deliberately outside this floating-point comparison.
    """
    import ast
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "tools/deepseek_v41"))
    from fixtures import REFERENCE_SHA256, dense_values, reduced_overrides
    from oracle import Recorder

    reference = root / "tests/fixtures/deepseek_v41/reference"
    for name in ("model.py", "config.json", "inference_config.json"):
        assert hashlib.sha256((reference / name).read_bytes()).hexdigest() == REFERENCE_SHA256[name]
    source = reference / "model.py"
    classes = {
        node.name: node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef)
    }

    def official_method(cls, method):
        node = next(
            node
            for node in classes[cls].body
            if isinstance(node, ast.FunctionDef) and node.name == method
        )
        namespace = {"torch": torch}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        return namespace[method]

    # Only method containers are constructed here; no official arithmetic is rewritten.
    namespace = {"torch": torch, "nn": torch.nn}
    exec(
        compile(ast.Module(body=[classes["RMSNorm"]], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    official_norm = namespace["RMSNorm"]
    official_compressor = type(
        "OfficialCompressor",
        (torch.nn.Module,),
        {"forward": official_method("Compressor", "forward")},
    )()
    official_indexer = type(
        "OfficialIndexer",
        (torch.nn.Module,),
        {"forward": official_method("Indexer", "forward")},
    )()
    args = reduced_overrides()
    rows = json.loads((root / "tests/fixtures/deepseek_v41/manifest.json").read_text())["tensors"]
    rows = {row["name"]: (ordinal, row) for ordinal, row in enumerate(rows)}

    def bind(module, name):
        ordinal, row = rows["layers.20." + name]
        assert list(module.weight.shape) == row["shape"]
        role = "norm" if "norm" in name else "weight"
        # Reconstruct exactly the C fixture payload before optional FP32 diagnostics.
        value = dense_values(row["shape"], ordinal, role).to(
            getattr(torch, row["dtype"].split(".")[1])
        )
        with torch.no_grad():
            module.weight.copy_(value)

    torch.manual_seed(1729)
    config = attn.CSA2Config(
        dim=args["dim"],
        heads=args["n_heads"],
        head_dim=args["head_dim"],
        rope_dim=args["rope_head_dim"],
        q_rank=args["q_lora_rank"],
        o_rank=args["o_lora_rank"],
        groups=args["o_groups"],
        index_heads=args["index_n_heads"],
        index_dim=args["index_head_dim"],
        eps=args["norm_eps"],
        linear_fp8=False,
        main_qat=False,
        index_qat=False,
        swa_fp8=False,
    )
    attention = attn.CSA2Attention(config, 20).to(dtype)
    block = hc.DeepseekV41Block(
        config.dim, args["hc_mult"], attention, torch.nn.Identity(), norm_eps=config.eps
    ).to(dtype)
    official_attn_norm = official_norm(config.dim, config.eps).to(dtype)
    official_compressor.compress_ratio = 1
    official_compressor.wkv = torch.nn.Linear(config.dim, config.head_dim, bias=False, dtype=dtype)
    official_compressor.norm = official_norm(config.head_dim, config.eps).to(dtype)
    official_indexer.owns_k = True
    official_indexer.compress_ratio = 1
    official_indexer.rope_head_dim = config.rope_dim
    official_indexer.freqs_cis = torch.ones(length, config.rope_dim // 2)
    official_indexer.wk = torch.nn.Linear(
        config.head_dim, config.index_dim, bias=False, dtype=dtype
    )
    official_indexer.k_norm = official_norm(config.index_dim, config.eps).to(dtype)
    for actual, expected, name in (
        (block.attn_norm, official_attn_norm, "attn_norm.weight"),
        (
            attention.compressor.wkv,
            official_compressor.wkv,
            "attn.compressor.wkv.weight",
        ),
        (
            attention.compressor.norm,
            official_compressor.norm,
            "attn.compressor.norm.weight",
        ),
        (attention.indexer.wk, official_indexer.wk, "attn.indexer.wk.weight"),
        (
            attention.indexer.k_norm,
            official_indexer.k_norm,
            "attn.indexer.k_norm.weight",
        ),
    ):
        bind(actual, name)
        bind(expected, name)
    hidden = dense_values((1, length, args["hc_mult"], config.dim), 10001).to(dtype)
    pre_mix = dense_values(hidden.shape[:-1], 10002, "multiplier").roll(1, -1)
    assert not torch.equal(hidden[:, :, 0], hidden[:, :, 1])
    recorder = Recorder("ced-component")
    recorder.layer, recorder.length = 20, length
    x20 = official_attn_norm(official_method("Block", "hc_pre")(None, hidden, pre_mix))
    recorder.add("ced.x20", x20)
    latent20 = official_compressor(x20, 0)
    recorder.add("compressor.latent_pre_rope", latent20)

    class BoundaryCaptured(Exception):
        pass

    def capture_index(module, inputs, output):
        recorder.add("index.k_pre_rope", output)
        raise BoundaryCaptured("official pre-RoPE boundary reached")

    handle = official_indexer.k_norm.register_forward_hook(capture_index)
    try:
        with pytest.raises(BoundaryCaptured, match="official pre-RoPE boundary reached"):
            official_indexer(x20, None, latent20, 0, 0)
    finally:
        handle.remove()
    actual = {}
    handles = [
        attention.register_forward_pre_hook(
            lambda module, inputs: actual.update({"ced.x20": inputs[0].detach().clone()})
        )
    ]
    for module, stage in (
        (attention.compressor, "compressor.latent_pre_rope"),
        (attention.indexer.k_norm, "index.k_pre_rope"),
    ):
        handles.append(
            module.register_forward_hook(
                lambda module, inputs, output, stage=stage: actual.update(
                    {stage: output.detach().clone()}
                )
            )
        )
    try:
        block.forward_with_state(hidden, pre_mix, attn.AttentionState())
    finally:
        for handle in handles:
            handle.remove()
    assert set(actual) == {record["stage"] for record in recorder.records}
    for record in recorder.records:
        torch.testing.assert_close(
            actual[record["stage"]],
            record["value"],
            rtol=0,
            atol=0,
            msg=record["stage"],
        )


def test_v41_nested_config_drives_attention(tmp_path):
    import copy
    import json
    from pathlib import Path
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    reference = Path(__file__).parents[2] / 'fixtures/deepseek_v41/reference/config.json'
    release = json.loads(reference.read_text())
    tiny = copy.deepcopy(release)
    tiny['text_config'].update(
        hidden_size=32,
        num_attention_heads=2,
        head_dim=32,
        qk_rope_head_dim=4,
        q_lora_rank=32,
        o_lora_rank=4,
        o_groups=2,
        index_n_heads=2,
        index_head_dim=32,
        sliding_window=1,
    )
    config = DeepseekV41Config._from_hf_dict(tiny)
    source = tmp_path / 'config.json'
    source.write_text(json.dumps(tiny))
    assert DeepseekV41Config.from_hf(source).to_hf_dict() == tiny
    assert config.to_hf_dict() == tiny
    tiny['text_config']['sliding_window'] = 4
    other = DeepseekV41Config._from_hf_dict(tiny)
    flags = dict(linear_fp8=False, main_qat=False, index_qat=False, swa_fp8=False)
    short = attn.CSA2Attention(config.attention_config(**flags), 0).float()
    long = attn.CSA2Attention(other.attention_config(**flags), 0).float()
    long.load_state_dict(short.state_dict())
    torch.manual_seed(991)
    x = torch.randn(1, 4, 32)
    y_short, _ = short(x, attn.AttentionState())
    y_long, _ = long(x, attn.AttentionState())
    torch.testing.assert_close(y_short[:, 0], y_long[:, 0])
    assert not torch.allclose(y_short[:, 1:], y_long[:, 1:])
    assert config.to_hf_dict()['text_config']['sliding_window'] == 1


@pytest.mark.parametrize(
    'field,value',
    [
        ('kv_source_layer_ids', [2, 8, 14]),
        ('index_source_layer_ids', [2, 8, 14, 20]),
        ('candidate_source_layer_id', 24),
        ('num_hidden_layers', 39),
        ('compress_ratios', [0] * 43),
    ],
)
def test_v41_config_rejects_unimplemented_topology(field, value):
    import json
    from pathlib import Path
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    reference = Path(__file__).parents[2] / 'fixtures/deepseek_v41/reference/config.json'
    release = json.loads(reference.read_text())
    release['text_config'][field] = value
    with pytest.raises(ValueError, match=field):
        DeepseekV41Config._from_hf_dict(release)


def _assembly_config():
    import json
    from pathlib import Path
    import importlib.util
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    root = Path(__file__).parents[2] / 'fixtures/deepseek_v41'
    release = json.loads((root / 'reference/config.json').read_text())
    modules = {}
    for name in ('config_mapping', 'fixtures'):
        path = Path(__file__).parents[3] / 'tools/deepseek_v41' / (name + '.py')
        spec = importlib.util.spec_from_file_location('c4_' + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    overrides = modules['fixtures'].reduced_overrides()
    for path, target in modules['config_mapping'].MAPPING.items():
        if target in overrides:
            parts = path.split('.')
            section = release
            for part in parts[:-1]:
                section = section[part]
            section[parts[-1]] = overrides[target]
    return DeepseekV41Config(release)


def _assembly_bundle(device='cpu', trainable_engram=False):
    from megatron.lite.model import registry

    cfg = _assembly_config()
    proto = registry.get_train_runtime_module(
        registry.resolve_runtime_model_name('deepseek_v41', 'lite')
    )

    return proto, proto.build_model(
        cfg,
        impl_cfg=proto.ImplConfig(
            device=device,
            dtype=torch.float32,
            quantized=False,
            token_map=list(range(256)),
            trainable_engram=trainable_engram,
        ),
    )


@pytest.mark.parametrize(
    'device', ['cpu', pytest.param('cuda', marks=pytest.mark.gpus(1))]
)
def test_v41_headwise_muon_distinct_heads_and_resume(device):
    from copy import deepcopy

    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz
    from megatron.lite.primitive.optimizers.headwise_muon import HeadwiseMuon

    from emerging_optimizers.utils import fp32_matmul_precision

    torch.manual_seed(712)
    weight = torch.nn.Parameter(torch.randn(6, 5, device=device))
    expected = weight.detach().clone()
    momentum = torch.zeros_like(weight)
    opt = HeadwiseMuon(
        [{'params': [weight], 'matrix_shape': (2, 3, 5)}],
        lr=0.03,
        ns_steps=5,
        coefficient_type='quintic',
    )
    for step in range(3):
        gradient = torch.randn_like(weight)
        gradient[3:] *= 17
        weight.grad = gradient
        momentum = 0.95 * momentum + 0.05 * gradient
        nesterov = 0.95 * momentum + 0.05 * gradient
        directions = []
        # Match the declared FP32 NS arithmetic, independent of ambient TF32 settings.
        with fp32_matmul_precision('highest'):
            for head in nesterov.split(3):
                update = newton_schulz(head, 5, coefficient_type='quintic')
                directions.append(update * (0.18 / update.square().mean().sqrt()))
            expected = expected * (1 - 0.03 * 0.1) - 0.03 * torch.cat(directions)
            vanilla = newton_schulz(nesterov, 5, coefficient_type='quintic')
            vanilla *= 0.18 / vanilla.square().mean().sqrt()
        assert not torch.allclose(vanilla, torch.cat(directions), atol=1e-3, rtol=1e-3)
        assert opt.step()
        torch.testing.assert_close(weight, expected, atol=0, rtol=0)
        torch.testing.assert_close(opt.state[weight]['momentum_buffer'], momentum)
        assert opt.state[weight]['momentum_buffer'].dtype == torch.float32
        if step == 1:
            saved = deepcopy(opt.state_dict())
            opt = HeadwiseMuon(
                [{'params': [weight], 'matrix_shape': (2, 3, 5)}],
                lr=0.03,
                ns_steps=5,
                coefficient_type='quintic',
            )
            opt.load_state_dict(saved)


def test_v41_headwise_muon_rejects_layout_and_atomic_nonfinite():
    from megatron.lite.primitive.optimizers.headwise_muon import HeadwiseMuon

    a = torch.nn.Parameter(torch.ones(6, 5))
    b = torch.nn.Parameter(torch.ones(6, 5))
    settings = dict(lr=0.01, ns_steps=5, coefficient_type='quintic')
    with pytest.raises(ValueError, match='logical'):
        HeadwiseMuon([{'params': [a], 'matrix_shape': (4, 3, 5)}], **settings)
    opt = HeadwiseMuon([{'params': [a, b], 'matrix_shape': (2, 3, 5)}], **settings)
    a.grad = torch.ones_like(a)
    b.grad = torch.full_like(b, float('nan'))
    assert not opt.step()
    assert not opt.state
    assert torch.equal(a, torch.ones_like(a))
    assert torch.equal(b, torch.ones_like(b))


@pytest.mark.parametrize('trainable', [False, True])
@pytest.mark.parametrize(
    'device', ['cpu', pytest.param('cuda', marks=pytest.mark.gpus(1))]
)
def test_v41_actual_optimizer_routes_and_native_gradients(moe, trainable, device):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig
    from megatron.lite.runtime.contracts import PackedBatch

    torch.manual_seed(41)
    bundle = protocol.build_model(
        _assembly_config(),
        impl_cfg=protocol.ImplConfig(
            device=device,
            quantized=device == 'cuda',
            dtype=torch.bfloat16,
            token_map=list(range(256)),
            trainable_engram=trainable,
            optimizer='muon',
            optimizer_config=OptimizerConfig(
                lr=1e-3, ns_steps=5, coefficient_type='quintic'
            ),
        ),
    )
    model, opt = bundle.chunks[0], bundle.optimizer
    groups = {g['owner_key']: g for g in opt.param_groups}
    a = model.layers[0].attn
    assert groups['layers.0.attn.wq_a.weight']['matrix_shape'] == tuple(
        a.wq_a.weight.shape
    )
    assert groups['layers.0.attn.wkv.weight']['matrix_shape'] == tuple(
        a.wkv.weight.shape
    )
    assert groups['layers.0.attn.wq_b.weight']['matrix_shape'] == (8, 64, 64)
    assert all('indexer' not in key for key in groups)
    assert {id(p) for g in opt.param_groups for p in g['params']} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    assert len([p for g in opt.param_groups for p in g['params']]) == len(groups)
    for block in model.layers:
        if block.engram is not None:
            table = block.engram.embed
            assert (table.master is not None) == trainable
            assert table.weight.dtype == torch.float8_e4m3fn
    assert {type(o).__name__ for o in opt.optimizers} == {
        'HeadwiseMuon',
        'Sinkhorn',
        'AdamW',
    }
    for key, group in groups.items():
        if '.engram.' in key:
            assert group['lr'] == 5e-3
        if key.endswith(('q_weight', 'k_weight')):
            assert group['algorithm'] == 'adamw' and group['weight_decay'] == 0.1
    ids = torch.tensor([3, 4, 3, 5], device=device)
    batch = PackedBatch(
        ids, ids, torch.tensor([4], device=device), torch.ones(4, device=device)
    )
    bundle.forward_step(model, batch)['loss'].backward()
    gradient = a.wq_b.weight.main_grad
    assert gradient.dtype == torch.float32 and gradient is a.wq_b.weight.grad
    assert not torch.equal(gradient, gradient.bfloat16().float())
    before = a.wq_b.weight.detach().clone()
    success, norm, _ = opt.step()
    assert success and norm > 0 and not torch.equal(before, a.wq_b.weight)
    for backend in opt.optimizers:
        for state in backend.state.values():
            for key, value in state.items():
                if 'momentum' in key:
                    assert value.dtype == torch.float32
    opt.zero_grad()
    assert a.wq_b.weight.grad is None and a.wq_b.weight.main_grad is None


def test_v41_optimizer_routing_ignores_misleading_names(moe, monkeypatch):
    from dataclasses import replace

    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import parameter_groups

    _, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    a = model.layers[0].attn
    # Both public naming surfaces lie: the shared matrices look like per-head Q.
    labels = {
        id(a.wq_a.weight): "decoy.attn.wq_b.weight",
        id(a.wkv.weight): "decoy.query_projection.weight",
        id(a.wq_b.weight): "decoy.shared_latent.weight",
    }
    named = model.named_parameters
    monkeypatch.setattr(
        model,
        "named_parameters",
        lambda *args, **kwargs: (
            (labels.get(id(p), name), p) for name, p in named(*args, **kwargs)
        ),
    )
    for key, binding in list(model.tensor_bindings.items()):
        if id(binding.tensor) in labels:
            model.tensor_bindings[key] = replace(
                binding, release_key=labels[id(binding.tensor)]
            )
    groups = parameter_groups(model, lr=1e-3)
    by_id = {id(p): g for g in groups for p in g["params"]}
    assert by_id[id(a.wq_a.weight)]["matrix_shape"] == tuple(a.wq_a.weight.shape)
    assert by_id[id(a.wkv.weight)]["matrix_shape"] == tuple(a.wkv.weight.shape)
    assert by_id[id(a.wq_b.weight)]["matrix_shape"] == (8, 64, 64)
    indexer_ids = {
        id(p)
        for block in model.layers
        if block.attn.indexer is not None
        for p in block.attn.indexer.parameters()
    }
    assert indexer_ids and indexer_ids.isdisjoint(by_id)


@pytest.mark.parametrize("trainable", [False, True])
def test_v41_optimizer_engram_persistent_master_and_reencoding(moe, trainable):
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import (
        OptimizerConfig,
        V41Optimizer,
    )
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8

    _, bundle = _assembly_bundle(trainable_engram=trainable)
    model = bundle.chunks[0]
    opt = V41Optimizer(
        model, OptimizerConfig(lr=0.1, ns_steps=5, coefficient_type="quintic")
    )
    tables = [b.engram.embed for b in model.layers if b.engram is not None]
    before = [(t.weight.clone(), t.scale.clone()) for t in tables]
    masters = [t.master for t in tables]
    for t in tables:
        if trainable:
            assert t.master.dtype == torch.float32
            # Force a scale boundary crossing so stale scale publication is observable.
            with torch.no_grad():
                t.master.mul_(128)
            t.master.grad = torch.randn_like(t.master)
        else:
            assert t.master is None and not list(t.parameters())
    assert opt.step()[0]
    for t, master, (weight, scale) in zip(tables, masters, before):
        assert t.master is master
        if trainable:
            expected_weight, expected_scale = quantize_block_fp8(
                master, (1, 32), scale_format="e8m0"
            )
            assert torch.equal(
                t.weight.view(torch.uint8), expected_weight.view(torch.uint8)
            )
            assert torch.equal(
                t.scale.view(torch.uint8), expected_scale.view(torch.uint8)
            )
            assert not torch.equal(t.weight.view(torch.uint8), weight.view(torch.uint8))
            assert not torch.equal(t.scale.view(torch.uint8), scale.view(torch.uint8))
        else:
            assert torch.equal(t.weight.view(torch.uint8), weight.view(torch.uint8))
            assert torch.equal(t.scale.view(torch.uint8), scale.view(torch.uint8))


def test_v41_native_linear_accumulates_unrounded_weight_gradients():
    from megatron.lite.primitive.modules.native_fp32_linear import native_fp32_linear

    torch.manual_seed(417)
    weight = torch.nn.Parameter(torch.randn(7, 5))
    reference = torch.zeros_like(weight)
    for _ in range(2):
        x = torch.randn(11, 5).bfloat16().requires_grad_()
        grad = torch.randn(11, 7).bfloat16()
        output = native_fp32_linear(x, weight)
        assert torch.equal(output, torch.nn.functional.linear(x, weight.bfloat16()))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output.backward(grad)
        reference += grad.float().T @ x.float()
    assert torch.equal(weight.grad, reference)
    assert not torch.equal(weight.grad, weight.grad.bfloat16().float())


def test_v41_optimizer_resume_and_atomic_skip(moe, monkeypatch):
    from copy import deepcopy

    from megatron.lite.model.deepseek_v41.lite import optimizer_groups, protocol
    from megatron.lite.runtime.contracts import PackedBatch

    config = optimizer_groups.OptimizerConfig(
        lr=1e-3, ns_steps=5, coefficient_type='quintic'
    )
    impl = protocol.ImplConfig(
        device='cpu',
        quantized=False,
        dtype=torch.bfloat16,
        token_map=list(range(256)),
        trainable_engram=True,
        optimizer='muon',
        optimizer_config=config,
    )
    bundle = protocol.build_model(_assembly_config(), impl_cfg=impl)
    model, opt = bundle.chunks[0], bundle.optimizer
    ids = torch.tensor([2, 3, 4])
    batch = PackedBatch(ids, ids, torch.tensor([3]), torch.ones(3))
    bundle.forward_step(model, batch)['loss'].backward()
    assert opt.step()[0]
    saved_weights, saved_opt = deepcopy(model.state_dict()), deepcopy(opt.state_dict())
    # A nonfinite gradient in AdamW must not let Muon or Sinkhorn publish first.
    p = model.norm.weight
    old = p.main_grad.clone()
    p.main_grad.fill_(float('inf'))
    assert not opt.step()[0]
    p.main_grad.copy_(old)
    for key, value in model.state_dict().items():
        assert torch.equal(
            value.view(torch.uint8), saved_weights[key].view(torch.uint8)
        )

    def reject_publication(*args, **kwargs):
        raise RuntimeError('publication probe')

    with monkeypatch.context() as patch:
        patch.setattr(optimizer_groups, 'quantize_block_fp8', reject_publication)
        with pytest.raises(RuntimeError, match='publication probe'):
            opt.step()
    for key, value in model.state_dict().items():
        assert torch.equal(
            value.view(torch.uint8), saved_weights[key].view(torch.uint8)
        )
    restored = protocol.build_model(_assembly_config(), impl_cfg=impl)
    restored.chunks[0].load_state_dict(saved_weights)
    restored.optimizer.load_state_dict(saved_opt)
    for candidate in (bundle, restored):
        candidate.optimizer.zero_grad()
        candidate.forward_step(candidate.chunks[0], batch)['loss'].backward()
        assert candidate.optimizer.step()[0]
    for p, q in zip(model.parameters(), restored.chunks[0].parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_v41_optimizer_rejects_unknown_alias_and_live_indexer(moe):
    from dataclasses import replace

    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import parameter_groups

    _, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    model.extra = torch.nn.Parameter(torch.ones(2, 2))
    with pytest.raises(ValueError):
        parameter_groups(model, lr=1e-3)
    del model.extra
    binding = model.tensor_bindings['layers.0.attn.wq_b.weight']
    model.tensor_bindings['layers.0.attn.wq_b.weight'] = replace(
        binding, head_count=None
    )
    with pytest.raises(ValueError, match='head'):
        parameter_groups(model, lr=1e-3)
    model.tensor_bindings['layers.0.attn.wq_b.weight'] = binding
    model.extra = model.layers[0].attn.wq_a.weight
    with pytest.raises(ValueError, match='alias'):
        parameter_groups(model, lr=1e-3)
    del model.extra
    model.layers[2].attn.indexer.requires_grad_(True)
    with pytest.raises(ValueError, match='indexer'):
        parameter_groups(model, lr=1e-3)


def test_v41_registry_assembly_forward_and_parameter_owners(moe):
    from megatron.lite.model.registry import resolve_model_type_from_hf
    from megatron.lite.runtime.contracts import PackedBatch

    proto, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    assert resolve_model_type_from_hf(_assembly_config().to_hf_dict()) == 'deepseek_v41'
    assert len(model.layers) == 40
    assert model.vision is not None and model.aligner is not None and model.mtp is not None
    owners = list(model.parameter_bindings())
    assert len({id(b.tensor) for b in owners}) == len(owners)
    assert {id(b.tensor) for b in owners} == {id(p) for p in model.parameters()}
    assert model.tensor_bindings['layers.0.attn.wq_b.weight'].head_count == 8
    assert model.tensor_bindings['layers.0.attn.wq_a.weight'].head_count is None
    assert model.tensor_bindings['layers.0.attn.wkv.weight'].head_count is None
    ids = torch.tensor([3, 4, 5, 6, 7, 8])
    batch = PackedBatch(ids, ids.roll(-1), torch.tensor([3, 3]), torch.ones(6))
    output = bundle.forward_step(model, batch)
    assert output['log_probs'].shape == (6,)
    assert torch.isfinite(output['log_probs']).all()
    targets = torch.tensor([5, 6, 0, 8, 3, 0])
    expected_loss = torch.nn.functional.cross_entropy(output['logits'], targets, reduction='none')
    torch.testing.assert_close(output['log_probs'], -expected_loss)
    torch.testing.assert_close(output['loss'], expected_loss[[0, 1, 3, 4]].mean())
    output['loss'].backward()
    assert model.embed.weight.grad is not None
    assert model.layers[20].attn.compressor.wkv.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.layers[20].attn.indexer.parameters())
    # Each packed sample gets fresh attention/Engram state and local positions.
    independent = torch.cat([model(ids[:3][None])['logits'], model(ids[3:][None])['logits']], 1)
    torch.testing.assert_close(model(ids[None], cu_seqlens=batch.cu_seqlens)['logits'], independent)
    for call in (
        lambda: model.vision(None),
        lambda: model.aligner(None),
        lambda: model.encode_image(None, 1, 1),
        lambda: model(ids[None], images=[]),
        lambda: model.forward_spec(ids[None]),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_v41_fixture_headers_and_complete_release_key_owners(moe):
    import itertools
    import json
    from pathlib import Path
    from megatron.lite.model.deepseek_v41.lite.checkpoint import bind_checkpoint
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    root = Path(__file__).parents[3]
    records = json.loads((root / 'tests/fixtures/deepseek_v41/manifest.json').read_text())[
        'tensors'
    ]
    _, bundle = _assembly_bundle(device='meta')
    bindings = bind_checkpoint(bundle.chunks[0], records, allow_missing_mtp=True)
    assert len(bindings) == 3204
    with pytest.raises(ValueError, match='coverage'):
        bind_checkpoint(bundle.chunks[0], records)
    # Quantization scales are owned by the same live module as their weights.
    assert bindings['layers.0.attn.wo_a.scale'].owner is bundle.chunks[0].layers[0].attn.wo_a
    bad = [dict(r) for r in records]
    next(r for r in bad if r['name'] == 'layers.0.attn.wq_a.weight')['shape'] = [1, 1]
    with pytest.raises(ValueError, match='shape'):
        bind_checkpoint(bundle.chunks[0], bad, allow_missing_mtp=True)
    with pytest.raises(ValueError, match='coverage'):
        bind_checkpoint(bundle.chunks[0], records[1:], allow_missing_mtp=True)
    with pytest.raises(ValueError, match='duplicate'):
        bind_checkpoint(bundle.chunks[0], records + records[:1], allow_missing_mtp=True)
    release = json.loads((root / 'tests/fixtures/deepseek_v41/reference/config.json').read_text())
    with torch.device('meta'):
        model = DeepseekV41Model(DeepseekV41Config(release))
    families = json.loads((root / 'docs/contracts/deepseek_v41/weights.json').read_text())[
        'families'
    ]
    keys = [f['pattern'].format(*ix) for f in families for ix in itertools.product(*f['indices'])]
    all_bindings = bind_checkpoint(model, keys)
    assert len(all_bindings) == 96085
    assert sum(k.startswith('mtp.') for k in all_bindings) == 2401
    assert all(b.owner is not None for b in all_bindings.values())
    assert {id(b.tensor) for b in model.parameter_bindings()} == {id(p) for p in model.parameters()}
    with pytest.raises(ValueError, match='coverage'):
        bind_checkpoint(model, keys + ['layers.3.attn.compressor.wkv.weight'])


def test_v41_assembly_checkpoint_updates_backbone_preserves_archive(tmp_path, moe):
    from megatron.lite.model.deepseek_v41.lite.checkpoint import (
        save_model,
        load_model,
    )
    from megatron.lite.model.deepseek_v41.lite.checkpoint_store import CheckpointTensorStore
    from safetensors.torch import save_file

    _, bundle = _assembly_bundle(trainable_engram=True)
    model = bundle.chunks[0]
    archive = {key: torch.tensor([17.0, -3.0]) for key in model.archival_bindings}
    source = tmp_path / 'archive.safetensors'
    save_file(archive, source)
    store = CheckpointTensorStore.load([source], expected_keys=archive)
    model.archival_store = store
    before = model.embed.weight.detach().clone()
    model.embed.weight.data.add_(0.25)
    model.layers[1].engram.embed.master.data.add_(0.25)
    model.layers[1].engram.embed.refresh_storage()
    output = tmp_path / 'export'
    save_model(model, output)
    _, restored = _assembly_bundle(trainable_engram=True)
    load_model(restored.chunks[0], output)
    torch.testing.assert_close(restored.chunks[0].embed.weight, before + 0.25)
    torch.testing.assert_close(
        restored.chunks[0].layers[1].engram.embed.master, model.layers[1].engram.embed.master
    )
    for binding in restored.chunks[0].checkpoint_bindings.values():
        assert binding.header is not None and binding.store is not None
    for key in archive:
        assert restored.chunks[0].archival_store.read(key) == store.read(key)
    ids = torch.tensor([[1, 3, 5]])
    torch.testing.assert_close(restored.chunks[0](ids)['logits'], model(ids)['logits'])


def test_v41_assembly_final_shifted_mix_and_engram_reachability(moe, monkeypatch):
    _, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    calls = []
    hooks = [
        model.layers[i].engram.register_forward_hook(lambda *args: calls.append(1)) for i in (1, 14)
    ]
    ids = torch.tensor([[1, 3, 5]])
    model(ids)
    assert len(calls) == 2
    for hook in hooks:
        hook.remove()
    h = torch.randn(1, 3, 4, 128)
    p = torch.tensor([0.1, 0.7, 0.2, 0.4]).expand(1, 3, 4)
    monkeypatch.setattr(model, '_sequence', lambda *args, **kwargs: (h, p))
    expected = torch.nn.functional.linear(model.norm((h * p[..., None]).sum(2)), model.head.weight)
    torch.testing.assert_close(model(ids)['logits'], expected)
    # Alias injection must be rejected even when the key count stays unchanged.
    from dataclasses import replace

    key = 'layers.0.attn.wq_a.weight'
    model.tensor_bindings[key] = replace(
        model.tensor_bindings[key], owner=model.layers[1].attn.wq_a
    )
    with pytest.raises(ValueError, match='exactly one'):
        model.validate_parameter_bindings()


@pytest.mark.gpus(1)
def test_v41_registry_quantized_cuda_forward_backward(moe):
    from megatron.lite.model import registry
    from megatron.lite.runtime.contracts import PackedBatch

    proto = registry.get_train_runtime_module('deepseek_v41')
    bundle = proto.build_model(
        _assembly_config(),
        impl_cfg=proto.ImplConfig(
            device='cuda',
            token_map=list(range(256)),
            quantized=True,
        ),
    )
    model = bundle.chunks[0]
    ids = torch.tensor([1, 3, 5], device='cuda')
    batch = PackedBatch(ids, ids, torch.tensor([3], device='cuda'))
    output = bundle.forward_step(model, batch)
    assert torch.isfinite(output['loss'])
    output['loss'].backward()
    gradient = model.layers[20].attn.compressor.wkv.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
