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
    router = moe.ModalityRouter(
        config, SimpleNamespace(tp_size=1), gate_temperature=0.7
    )
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
    torch.testing.assert_close(
        torch.stack([router.bias, router.bias_vl]), before + delta
    )
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
        expected_hidden = expected_hidden + 0.25 * (expected_attn_input + 17).unsqueeze(
            -2
        )
        expected_ffn_input = (expected_hidden * attn_pre.unsqueeze(-1)).sum(-2)
        wrong_ffn_input = (expected_hidden * expected_pre.unsqueeze(-1)).sum(-2)
        expected_hidden = expected_hidden + 0.25 * (expected_ffn_input - 9).unsqueeze(
            -2
        )
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
    import os

    from fixtures import REFERENCE_SHA256, dense_values, reduced_overrides
    from oracle import Recorder

    reference = Path(
        os.environ.get("DS41_REFERENCE_DIR", "/tmp/ds41-fixture-reference")
    )
    for name in ("model.py", "config.json", "inference_config.json"):
        assert (
            hashlib.sha256((reference / name).read_bytes()).hexdigest()
            == REFERENCE_SHA256[name]
        )
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
        compile(
            ast.Module(body=[classes["RMSNorm"]], type_ignores=[]), str(source), "exec"
        ),
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
    rows = json.loads((root / "tests/fixtures/deepseek_v41/manifest.json").read_text())[
        "tensors"
    ]
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
    official_compressor.wkv = torch.nn.Linear(
        config.dim, config.head_dim, bias=False, dtype=dtype
    )
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
        with pytest.raises(
            BoundaryCaptured, match="official pre-RoPE boundary reached"
        ):
            official_indexer(x20, None, latent20, 0, 0)
    finally:
        handle.remove()
    actual = {}
    handles = [
        attention.register_forward_pre_hook(
            lambda module, inputs: actual.update(
                {"ced.x20": inputs[0].detach().clone()}
            )
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

    reference = (
        Path(__file__).parents[2] / 'fixtures/deepseek_v41/reference/config.json'
    )
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

    reference = (
        Path(__file__).parents[2] / 'fixtures/deepseek_v41/reference/config.json'
    )
    release = json.loads(reference.read_text())
    release['text_config'][field] = value
    with pytest.raises(ValueError, match=field):
        DeepseekV41Config._from_hf_dict(release)


def _assembly_config():
    import importlib.util
    import json
    from pathlib import Path

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


def _assembly_bundle(device='cpu', trainable_engram=False, text_only=True):
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
            text_only=text_only,
        ),
    )


def test_v41_registry_assembly_forward_and_parameter_owners(moe):
    from megatron.lite.model.registry import resolve_model_type_from_hf
    from megatron.lite.runtime.contracts import PackedBatch

    proto, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    assert resolve_model_type_from_hf(_assembly_config().to_hf_dict()) == 'deepseek_v41'
    assert len(model.layers) == 40
    assert (
        model.vision is not None and model.aligner is not None and model.mtp is not None
    )
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
    expected_loss = torch.nn.functional.cross_entropy(
        output['logits'], targets, reduction='none'
    )
    torch.testing.assert_close(output['log_probs'], -expected_loss)
    torch.testing.assert_close(output['loss'], expected_loss[[0, 1, 3, 4]].mean())
    output['loss'].backward()
    assert model.embed.weight.grad is not None
    assert model.layers[20].attn.compressor.wkv.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.layers[20].attn.indexer.parameters())
    # Each packed sample gets fresh attention/Engram state and local positions.
    independent = torch.cat(
        [model(ids[:3][None])['logits'], model(ids[3:][None])['logits']], 1
    )
    torch.testing.assert_close(
        model(ids[None], cu_seqlens=batch.cu_seqlens)['logits'], independent
    )
    for call in (lambda: model.forward_spec(ids[None]),):
        with pytest.raises(NotImplementedError):
            call()


def test_v41_fixture_headers_and_complete_release_key_owners(moe):
    import itertools
    import json
    from pathlib import Path

    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.model.deepseek_v41.lite.checkpoint import bind_checkpoint
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    root = Path(__file__).parents[3]
    records = json.loads(
        (root / 'tests/fixtures/deepseek_v41/manifest.json').read_text()
    )['tensors']
    _, bundle = _assembly_bundle(device='meta')
    bindings = bind_checkpoint(bundle.chunks[0], records, allow_missing_mtp=True)
    assert len(bindings) == 3204
    with pytest.raises(ValueError, match='coverage'):
        bind_checkpoint(bundle.chunks[0], records)
    # Quantization scales are owned by the same live module as their weights.
    assert (
        bindings['layers.0.attn.wo_a.scale'].owner
        is bundle.chunks[0].layers[0].attn.wo_a
    )
    bad = [dict(r) for r in records]
    next(r for r in bad if r['name'] == 'layers.0.attn.wq_a.weight')['shape'] = [1, 1]
    with pytest.raises(ValueError, match='shape'):
        bind_checkpoint(bundle.chunks[0], bad, allow_missing_mtp=True)
    with pytest.raises(ValueError, match='coverage'):
        bind_checkpoint(bundle.chunks[0], records[1:], allow_missing_mtp=True)
    with pytest.raises(ValueError, match='duplicate'):
        bind_checkpoint(bundle.chunks[0], records + records[:1], allow_missing_mtp=True)
    release = json.loads(
        (root / 'tests/fixtures/deepseek_v41/reference/config.json').read_text()
    )
    with torch.device('meta'):
        model = DeepseekV41Model(DeepseekV41Config(release))
    families = json.loads(
        (root / 'docs/contracts/deepseek_v41/weights.json').read_text()
    )['families']
    keys = [
        f['pattern'].format(*ix)
        for f in families
        for ix in itertools.product(*f['indices'])
    ]
    all_bindings = bind_checkpoint(model, keys)
    assert len(all_bindings) == 96085
    assert sum(k.startswith('mtp.') for k in all_bindings) == 2401
    assert all(b.owner is not None for b in all_bindings.values())
    assert {id(b.tensor) for b in model.parameter_bindings()} == {
        id(p) for p in model.parameters()
    }
    with pytest.raises(ValueError, match='coverage'):
        bind_checkpoint(model, keys + ['layers.3.attn.compressor.wkv.weight'])


def test_v41_assembly_checkpoint_updates_backbone_preserves_archive(tmp_path, moe):
    from megatron.lite.model.deepseek_v41.lite.checkpoint import load_model, save_model
    from megatron.lite.model.deepseek_v41.lite.checkpoint_store import (
        CheckpointTensorStore,
    )
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
        restored.chunks[0].layers[1].engram.embed.master,
        model.layers[1].engram.embed.master,
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
        model.layers[i].engram.register_forward_hook(lambda *args: calls.append(1))
        for i in (1, 14)
    ]
    ids = torch.tensor([[1, 3, 5]])
    model(ids)
    assert len(calls) == 2
    for hook in hooks:
        hook.remove()
    h = torch.randn(1, 3, 4, 128)
    p = torch.tensor([0.1, 0.7, 0.2, 0.4]).expand(1, 3, 4)
    monkeypatch.setattr(model, '_sequence', lambda *args, **kwargs: (h, p))
    expected = torch.nn.functional.linear(
        model.norm((h * p[..., None]).sum(2)), model.head.weight
    )
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
            device='cuda', token_map=list(range(256)), quantized=True
        ),
    )
    model = bundle.chunks[0]
    ids = torch.tensor([1, 3, 5], device='cuda')
    batch = PackedBatch(ids, ids, torch.tensor([3], device='cuda'))
    output = bundle.forward_step(model, batch)
    assert torch.isfinite(output['loss'])
    output['loss'].backward()
    gradient = model.layers[20].attn.compressor.wkv.weight.grad
    assert (
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().sum() > 0
    )


def test_v41_real_pipeline_ranges_match_monolithic_gradients(moe):
    _, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    expected = model(ids)['logits']
    coefficients = torch.linspace(-0.5, 1.0, expected.numel()).reshape_as(expected)
    (expected * coefficients).sum().backward()
    gradients = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    payload, owners = None, (-1, -1)
    for start, end in ((0, 20), (20, 24), (24, 32), (32, 40)):
        payload, owners = model.forward_pipeline_range(
            ids, start=start, end=end, payload=payload, owners=owners
        )
        assert payload.ced_h is not None and payload.ced_p is not None
        if end > 20:
            assert owners == (20, ((end - 1 - 20) // 4) * 4 + 20)
            assert payload.candidates is not None
        if end == 24:
            owner_kv = payload.kv
        if end > 24:
            assert payload.kv is owner_kv
    actual = model.finish_pipeline(payload)
    torch.testing.assert_close(
        actual,
        expected,
        atol=0,
        rtol=0,
        msg=lambda detail: 'Real C4 pipeline logits differ from monolithic reference: '
        + detail,
    )
    (actual * coefficients).sum().backward()
    for name, parameter in model.named_parameters():
        reference = gradients[name]
        if reference is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(
                parameter.grad,
                reference,
                atol=0,
                rtol=0,
                msg=lambda detail, name=name: "Real C4 pipeline gradient differs for "
                + name
                + ": "
                + detail,
            )
    assert gradients['layers.20.attn.compressor.wkv.weight'].abs().sum() > 0


def test_v41_pipeline_protocol_preserves_text_loss(moe):
    from megatron.lite.runtime.contracts import PackedBatch

    proto, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    ids = torch.arange(3, 11)
    batch = PackedBatch(ids, ids.roll(-1), torch.tensor([8]), torch.ones(8))
    expected = bundle.forward_step(model, batch)
    payload, owners = None, (-1, -1)
    for start, end in ((0, 20), (20, 24), (24, 32), (32, 40)):
        output = proto.pipeline_forward_step(
            model, batch, start=start, end=end, payload=payload, owners=owners
        )
        payload, owners = output['pipeline_payload'], output['pipeline_owners']
        assert ('loss' in output) == (end == 40)
    for name in ('logits', 'log_probs', 'loss'):
        torch.testing.assert_close(output[name], expected[name], atol=0, rtol=0)
    packed = PackedBatch(ids, ids.roll(-1), torch.tensor([4, 4]), torch.ones(8))
    with pytest.raises(NotImplementedError, match='packed'):
        proto.pipeline_forward_step(model, packed, start=0, end=20)


def test_v41_pipeline_constructor_allocates_only_local_owners(moe, monkeypatch):
    from megatron.lite.model.deepseek_v41.lite import model as model_module

    _, reference_bundle = _assembly_bundle()
    reference = reference_bundle.chunks[0]
    expected = {b.release_key for b in reference.parameter_bindings()}
    original_attention = model_module.CSA2Attention
    constructed = []

    def counted(config, layer_id):
        constructed.append(layer_id)
        return original_attention(config, layer_id)

    monkeypatch.setattr(model_module, 'CSA2Attention', counted)
    assigned = set()
    for start, end in ((0, 20), (20, 24), (24, 32), (32, 40)):
        constructed.clear()
        stage = model_module.DeepseekV41Model(
            reference.config,
            token_map=list(range(256)),
            quantized=False,
            layer_range=(start, end),
        )
        assert constructed == list(range(start, end)), 'Non-owner layer was allocated'
        assert stage.local_layer_range == (start, end)
        assert [i for i, layer in enumerate(stage.layers) if layer is not None] == list(
            range(start, end)
        )
        assert (stage.embed is not None) == (start == 0)
        assert (stage.norm is not None) == (end == 40)
        assert (stage.head is not None) == (end == 40)
        bindings = list(stage.parameter_bindings())
        keys = {binding.release_key for binding in bindings}
        assert not assigned & keys, 'A parameter owner was assigned to two PP ranks'
        assigned.update(keys)
        assert {id(binding.tensor) for binding in bindings} == {
            id(p) for p in stage.parameters()
        }
        stage.validate_parameter_bindings()
        from megatron.lite.model.deepseek_v41.lite.checkpoint import export_model

        with pytest.raises(NotImplementedError, match='stage export'):
            list(export_model(stage))
        with pytest.raises(RuntimeError, match='stage'):
            stage(torch.tensor([[3, 4]]))
    assert assigned == expected, 'PP local owners do not cover the monolithic model'


def test_v41_pipeline_rejects_range_outside_local_stage(moe):
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    stage = DeepseekV41Model(
        _assembly_config(),
        token_map=list(range(256)),
        quantized=False,
        layer_range=(0, 20),
    )
    with pytest.raises(
        ValueError, match='^Requested range is outside this pipeline stage$'
    ):
        stage.forward_pipeline_range(torch.tensor([[3, 4]]), start=0, end=21)


def test_v41_pipeline_rejects_output_on_nonfinal_stage(moe):
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    stage = DeepseekV41Model(
        _assembly_config(),
        token_map=list(range(256)),
        quantized=False,
        layer_range=(0, 20),
    )
    payload, _ = stage.forward_pipeline_range(torch.tensor([[3, 4]]), start=0, end=20)
    with pytest.raises(
        RuntimeError, match='^Only the final pipeline stage owns the output head$'
    ):
        stage.finish_pipeline(payload)


def test_v41_packed_pipeline_matches_monolithic(moe):
    from megatron.lite.runtime.contracts import PackedBatch

    proto, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    ids = torch.arange(3, 19)
    batch = PackedBatch(ids, ids.roll(-1), torch.tensor([5, 8, 3]), torch.ones(16))
    expected = bundle.forward_step(model, batch)
    expected['loss'].backward()
    gradients = {
        n: None if p.grad is None else p.grad.clone()
        for n, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    state = None
    for start, end in ((0, 20), (20, 24), (24, 32), (32, 40)):
        output = proto.packed_pipeline_forward_step(
            model, batch, start=start, end=end, state=state
        )
        state = output['packed_pipeline_state']
        assert [payload.h.shape[1] for payload, owners in state] == [5, 8, 3]
    for key in ('logits', 'log_probs', 'loss'):
        torch.testing.assert_close(
            output[key],
            expected[key],
            atol=0,
            rtol=0,
            msg=lambda detail: 'Packed PP differs from monolithic reference: ' + detail,
        )
    output['loss'].backward()
    for name, parameter in model.named_parameters():
        if gradients[name] is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(
                parameter.grad,
                gradients[name],
                atol=1e-6,
                rtol=1e-5,
                msg=lambda detail, name=name: 'Packed PP gradient differs: '
                + name
                + ': '
                + detail,
            )


def _real_model_worker(rank, rendezvous, scheduled=False, recompute=False, mixed=False):
    from datetime import timedelta

    import torch.distributed as dist
    from megatron.lite.model.deepseek_v41.lite import pipeline, protocol
    from megatron.lite.primitive.parallel import tensor_payload as transport
    from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig

    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig

    optimizer_options = (
        dict(
            optimizer='muon',
            trainable_engram=True,
            optimizer_config=OptimizerConfig(
                lr=1e-4, ns_steps=5, coefficient_type='quintic', clip_grad=0.5
            ),
        )
        if mixed
        else {}
    )
    torch.manual_seed(1729)
    # Compute an independent monolithic reference, retain only this rank's
    # weights/gradients, then discard the full model before PP construction.
    bundle = protocol.build_model(
        _assembly_config(),
        impl_cfg=protocol.ImplConfig(
            **optimizer_options,
            device=str(device),
            dtype=torch.float32,
            quantized=False,
            token_map=list(range(256)),
        ),
    )
    model = bundle.chunks[0]
    reference_parameter_count = sum(p.numel() for p in model.parameters())
    boundaries = (0, 10, 20, 30, 40)
    start, end = boundaries[rank : rank + 2]
    owned_modules = list(model.layers[start:end])
    if rank == 0:
        owned_modules.extend((model.embed, model.vision, model.aligner))
    if rank == 3:
        owned_modules.extend((model.norm, model.head))
    owned_ids = {id(p) for module in owned_modules for p in module.parameters()}
    if rank == 0:
        owned_ids.update(
            id(getattr(model, key))
            for key in ('image_start', 'image_end', 'image_newline')
        )
    bindings = [b for b in model.parameter_bindings() if id(b.tensor) in owned_ids]
    assert {id(b.tensor) for b in bindings} == owned_ids
    params = [b.tensor for b in bindings if b.tensor.requires_grad]
    lengths = ((5, 8, 3), (4, 7)) if scheduled else ((8,), (8,))
    inputs = [
        torch.arange(3 + mb, 3 + mb + sum(ns), device=device)[None]
        for mb, ns in enumerate(lengths)
    ]
    batches = [
        PackedBatch(ids[0], None, torch.tensor(ns, device=device))
        for ids, ns in zip(inputs, lengths)
    ]
    reference_logits = []
    for mb, ids in enumerate(inputs):
        logits = bundle.forward_step(model, batches[mb])['logits']
        reference_logits.append(logits.detach())
        coefficients = torch.linspace(
            -0.5, 1.0, logits.numel(), device=device
        ).reshape_as(logits)
        ((logits * coefficients * (mb + 1)).sum() / (2 if scheduled else 1)).backward()
    reference_grads = {
        binding.release_key: (
            None
            if binding.tensor.grad is None
            else binding.tensor.grad.detach().clone()
        )
        for binding in bindings
        if binding.tensor.requires_grad
    }
    original_parameters = (
        {name: t.detach().clone() for name, t in model.state_dict().items()}
        if mixed
        else {}
    )
    reference_update, reference_norm = {}, None
    if mixed:
        success, reference_norm, _ = bundle.optimizer.step()
        assert success
        reference_update = {
            binding.release_key: binding.tensor.detach().clone() for binding in bindings
        }
        model.load_state_dict(original_parameters, strict=True)
        del original_parameters
    if model.engram_hash is not None:
        owned_modules.append(model.engram_hash)
    owned_values = {
        id(value)
        for module in owned_modules
        for value in (*module.parameters(), *module.buffers())
    }
    owned_values.update(owned_ids)
    local_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict(keep_vars=True).items()
        if id(tensor) in owned_values
    }
    import gc
    import weakref

    reference_model = weakref.ref(model)
    del model, bundle, owned_modules, bindings, params, logits
    gc.collect()
    assert (
        reference_model() is None
    ), 'Full reference model remained live during PP construction'
    dist.init_process_group(
        'nccl',
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    bundle = protocol.build_model(
        _assembly_config(),
        impl_cfg=protocol.ImplConfig(
            **optimizer_options,
            device=str(device),
            dtype=torch.float32,
            quantized=False,
            token_map=list(range(256)),
            parallel=ParallelConfig(pp=4),
            pipeline_recompute=recompute,
        ),
    )
    model = bundle.chunks[0]
    assert model.local_layer_range == (start, end)
    assert [i for i, layer in enumerate(model.layers) if layer is not None] == list(
        range(start, end)
    ), 'PP constructor allocated non-owner layers'
    assert (
        sum(p.numel() for p in model.parameters()) < reference_parameter_count
    ), 'PP parameter allocation did not shrink to the local stage'
    assert (
        local_state.keys() == model.state_dict().keys()
    ), 'G1_STAGE_STATE: fixture must retain every first-stage visual owner and every local text owner'
    model.load_state_dict(local_state, strict=True)
    del local_state
    bindings = list(model.parameter_bindings())
    owned_ids = {id(p) for p in model.parameters()}
    assert {id(binding.tensor) for binding in bindings} == owned_ids
    params = [binding.tensor for binding in bindings if binding.tensor.requires_grad]
    assert bundle.parallel_state.pp_rank == rank and bundle.parallel_state.pp_size == 4
    group = bundle.parallel_state.pp_group
    ledger = pipeline.PipelineLedger()
    incoming, outputs, errors = [], [], []
    stage_results = []

    def owners_at(boundary):
        if boundary == 0:
            return (-1, -1)
        if boundary == 10:
            return (8, 8)
        if boundary == 20:
            return (14, 14)
        return (20, 20 + ((boundary - 1 - 20) // 4) * 4)

    def tag_at(boundary, mb):
        return pipeline.PipelineTag(11, mb, boundary, 0, *owners_at(boundary))

    def compare(actual, expected, label):
        try:
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        except AssertionError as exc:
            errors.append(label + ': ' + str(exc))

    try:
        if scheduled:
            from types import SimpleNamespace

            from megatron.lite.primitive.parallel.pipeline import (
                forward_backward_pipelining,
            )

            adapter = bundle.extras['pipeline_payload_adapter']
            seen = []

            def objective(out, batch):
                mb = next(
                    i for i, candidate in enumerate(batches) if candidate is batch
                )
                logits = out['logits']
                compare(logits, reference_logits[mb], 'packed scheduled logits')
                seen.append(mb)
                coefficients = torch.linspace(
                    -0.5, 1.0, logits.numel(), device=device
                ).reshape_as(logits)
                return (logits * coefficients * (mb + 1)).sum(), {}

            forward_backward_pipelining(
                bundle.forward_step,
                [model],
                iter(batches),
                SimpleNamespace(num_microbatches=2),
                bundle.parallel_state,
                loss_fn=objective,
                payload_adapter=adapter,
            )
            assert seen == ([0, 1] if rank == 3 else [])
            assert adapter.step == 1 and not adapter._records
            with pytest.raises(RuntimeError, match='stale'):
                adapter.backward(0, None)
        else:
            for mb, ids in enumerate(inputs):
                payload = None
                if rank:
                    payload = pipeline.PairedPayload.from_tensors(
                        transport.recv_tensor_payload(
                            (*tag_at(start, mb).as_tuple(), 0),
                            peer=rank - 1,
                            group=group,
                            device=device,
                        )
                    )
                incoming.append(payload)
                stage_result = protocol.pipeline_forward_step(
                    model,
                    batches[mb],
                    start=start,
                    end=end,
                    payload=payload,
                    owners=owners_at(start),
                )
                output, owners = (
                    stage_result['pipeline_payload'],
                    stage_result['pipeline_owners'],
                )
                stage_results.append(stage_result)
                assert owners == owners_at(
                    end
                ), 'Real C4 range changed the canonical owner'
                outputs.append(output)
                if rank < 3:
                    ledger.publish(tag_at(end, mb), output, consumers=(rank + 1,))
                    transport.send_tensor_payload(
                        output.tensors(),
                        (*tag_at(end, mb).as_tuple(), 0),
                        peer=rank + 1,
                        group=group,
                        device=device,
                    )
                else:
                    compare(
                        stage_result['logits'], reference_logits[mb], 'real C4 logits'
                    )
            for mb in reversed(range(2)):
                if rank == 3:
                    logits = stage_results[mb]['logits']
                    coefficients = torch.linspace(
                        -0.5, 1.0, logits.numel(), device=device
                    ).reshape_as(logits)
                    (logits * coefficients * (mb + 1)).sum().backward()
                else:
                    returned = transport.recv_tensor_payload(
                        (*tag_at(end, mb).as_tuple(), 1),
                        peer=rank + 1,
                        group=group,
                        device=device,
                    )
                    ledger.return_gradients(
                        tag_at(end, mb),
                        rank + 1,
                        {
                            name: value
                            for name, value in zip(pipeline.PAYLOAD_FIELDS, returned)
                            if value is not None
                        },
                    )
                    ledger.backward(tag_at(end, mb))
                if rank:
                    fields = incoming[mb].differentiable()
                    returned = {
                        name: (
                            torch.zeros_like(value)
                            if value.grad is None
                            else value.grad
                        )
                        for name, value in fields.items()
                    }
                    transport.send_tensor_payload(
                        tuple(returned.get(name) for name in pipeline.PAYLOAD_FIELDS),
                        (*tag_at(start, mb).as_tuple(), 1),
                        peer=rank - 1,
                        group=group,
                        device=device,
                    )
        ledger.assert_quiescent()
        for binding in bindings:
            p = binding.tensor
            if not p.requires_grad:
                assert p.grad is None, 'Frozen indexer received a gradient'
                continue
            expected = reference_grads[binding.release_key]
            if expected is None or p.grad is None:
                if (expected is None) != (p.grad is None):
                    errors.append(
                        binding.release_key + ': missing or unexpected gradient'
                    )
            else:
                compare(p.grad, expected, binding.release_key + ' gradient')
        before = {id(p): p.detach().clone() for p in params}
        if mixed:
            success, norm, _ = bundle.optimizer.step()
            assert success
            compare(
                torch.tensor(norm), torch.tensor(reference_norm), 'global clip norm'
            )
        else:
            torch.optim.SGD(params, lr=1e-4).step()
        for binding in bindings:
            p = binding.tensor
            if p.requires_grad:
                g = reference_grads[binding.release_key]
                expected = (
                    reference_update[binding.release_key]
                    if mixed
                    else before[id(p)] if g is None else before[id(p)] - 1e-4 * g
                )
                compare(p, expected, binding.release_key + ' single owner update')
        if mixed:
            from copy import deepcopy

            opt = bundle.optimizer
            saved = deepcopy(opt.state_dict())
            weights = {n: p.detach().clone() for n, p in model.named_parameters()}
            opt.zero_grad()
            for p in params:
                p.grad = torch.zeros_like(p)
                p.main_grad = p.grad
            if rank == 0:
                params[0].main_grad.flatten()[0] = float('nan')
            assert not opt.step()[0], 'Nonfinite peer must skip every PP optimizer'
            _g1_equal(opt.state_dict(), saved, 'PP_ATOMIC_STATE')
            for n, p in model.named_parameters():
                torch.testing.assert_close(p, weights[n], atol=0, rtol=0)
            for p in params:
                p.grad.zero_()
            backend = opt.optimizers[0]
            prepare = backend.prepare_step
            if rank == 1:

                def rejected():
                    prepare()
                    return False

                backend.prepare_step = rejected
            assert not opt.step()[0], 'Failed staged peer must abort all PP commits'
            backend.prepare_step = prepare
            _g1_equal(opt.state_dict(), saved, 'PP_STAGED_ATOMIC_STATE')
            for n, p in model.named_parameters():
                torch.testing.assert_close(p, weights[n], atol=0, rtol=0)
        all_errors = [None] * 4
        dist.all_gather_object(all_errors, errors)
        failures = [message for rank_errors in all_errors for message in rank_errors]
        if failures:
            raise AssertionError(
                'Real C4 pipeline differs from monolithic reference: ' + failures[0]
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(4)
def test_real_c4_pipeline_four_rank_forward_backward_update(tmp_path):
    import os

    import torch.multiprocessing as mp

    assert os.getenv('SLURM_JOB_ID'), 'Real C4 pipeline validation requires Slurm'
    if torch.cuda.device_count() < 4:
        pytest.skip('Requires the declared four-GPU allocation')
    mp.spawn(
        _real_model_worker, args=(f'file://{tmp_path}/real-c4',), nprocs=4, join=True
    )


@pytest.mark.parametrize(
    'device', ['cpu', pytest.param('cuda', marks=pytest.mark.gpus(1))]
)
def test_v41_headwise_muon_distinct_heads_and_resume(device):
    from copy import deepcopy

    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz
    from emerging_optimizers.utils import fp32_matmul_precision
    from megatron.lite.primitive.optimizers.headwise_muon import HeadwiseMuon

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


def _check_vision_official(monkeypatch, dtype):
    import hashlib
    import importlib.util
    from pathlib import Path

    from megatron.lite.model.deepseek_v41.lite import vision

    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / 'tools/deepseek_v41'))
    import os

    from fixtures import REFERENCE_SHA256

    path = (
        Path(os.environ.get('DS41_REFERENCE_DIR', '/tmp/ds41-fixture-reference'))
        / 'vision.py'
    )
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256['vision.py']
    )
    spec = importlib.util.spec_from_file_location('official_vision', path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    args = SimpleNamespace(
        vision_patch_size=14,
        vision_dim=64,
        vision_n_heads=4,
        vision_inter_dim=128,
        vision_n_layers=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        dim=12,
    )
    torch.manual_seed(19)
    actual = torch.nn.Sequential(vision.ViT(args), vision.Aligner(args)).to(dtype)
    reference = torch.nn.Sequential(official.ViT(args), official.Aligner(args)).to(
        dtype
    )
    reference.load_state_dict(actual.state_dict())
    for height, width in [(3, 6), (4, 5), (1, 2)]:
        x = torch.randn(height * width, 3, 14, 14, dtype=dtype, requires_grad=True)
        y = actual[1](actual[0](x, height, width), height, width)
        expected = reference[1](reference[0](x, height, width), height, width)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        probe = torch.randn_like(y)
        ga = torch.autograd.grad((y * probe).sum(), (x, *actual.parameters()))
        gb = torch.autograd.grad(
            (expected * probe).sum(),
            (
                x,
                *[
                    dict(reference.named_parameters())[name]
                    for name, _ in actual.named_parameters()
                ],
            ),
        )
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_v41_multimage_copy_gradient():
    from megatron.lite.model.deepseek_v41.lite import image_data

    types = image_data.image_token_types(1, 2)
    images = [
        [
            image_data.ImageInput(1, torch.zeros(2, 3, 2, 2), 1, 2, types),
            image_data.ImageInput(7, torch.zeros(2, 3, 2, 2), 1, 2, types),
        ]
    ]
    text = torch.randn(1, 13, 4, requires_grad=True)
    features = [torch.randn(2, 4, requires_grad=True) for _ in range(2)]
    delimiters = [torch.randn(4, requires_grad=True) for _ in range(3)]
    merged = image_data.merge_image_embeddings(text, images, [features], *delimiters)
    expanded, _ = hc.expand_hc(merged, 3)
    probe = torch.arange(expanded.numel()).reshape_as(expanded).float()
    gradients = torch.autograd.grad(
        (expanded * probe).sum(), (text, *features, *delimiters)
    )
    expected = text.clone()
    for start, feature in zip((1, 7), features):
        expected[:, start : start + 5] = torch.stack(
            (delimiters[0], feature[0], feature[1], delimiters[2], delimiters[1])
        )
    for copy in range(3):
        torch.testing.assert_close(expanded[:, :, copy], expected)
    summed = probe.sum(2)
    mask = torch.ones(13, dtype=torch.bool)
    mask[1:6] = False
    mask[7:12] = False
    torch.testing.assert_close(gradients[0], summed * mask[None, :, None])
    for grad, start in zip(gradients[1:3], (2, 8)):
        torch.testing.assert_close(grad, summed[0, start : start + 2])
    for grad, offsets in zip(gradients[3:], ((1, 7), (5, 11), (4, 10))):
        torch.testing.assert_close(grad, summed[0, list(offsets)].sum(0))


@pytest.mark.parametrize(
    'size', [(17, 23), (125, 97), (2000, 7), (7, 2000), (1024, 768)]
)
@pytest.mark.parametrize('max_ratio', [None, 4])
def test_v41_image_processor_official(size, max_ratio, monkeypatch):
    import hashlib
    import importlib.util
    import io
    import os
    import sys
    from pathlib import Path

    import numpy as np
    from megatron.lite.model.deepseek_v41.lite import image_data as data
    from PIL import Image

    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / 'tools/deepseek_v41'))
    from fixtures import REFERENCE_SHA256

    path = (
        Path(os.environ.get('DS41_REFERENCE_DIR', '/tmp/ds41-fixture-reference'))
        / 'image_processor.py'
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256[path.name]
    spec = importlib.util.spec_from_file_location('official_image_processor', path)
    official = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, official)
    spec.loader.exec_module(official)
    config = data.ImageConfig(max_wh_ratio=max_ratio)
    args = SimpleNamespace(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=1024,
        vision_min_pixels=295936,
        vision_max_wh_ratio=max_ratio,
    )
    width, height = size
    image = Image.fromarray(
        np.random.default_rng(7).integers(0, 256, (height, width, 3), dtype=np.uint8)
    )
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    expected = official.load_image({'data': buffer.getvalue()}, args)
    actual = data.preprocess_image(image, config)
    assert actual[1:] == expected[1:]
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    ids, types, images = data.prepare_image_inputs(
        [1, 99, 2, 99, 3], [image, image], 99, config
    )
    layout = official.image_token_types(*actual[-2:])
    assert images[0].start == 1 and images[1].start == len(layout) + 2
    assert ids == [1] + [99] * len(layout) + [2] + [99] * len(layout) + [3]
    assert types == [-1] + layout.tolist() + [-1] + layout.tolist() + [-1]
    torch.testing.assert_close(images[1].patches, expected[0], rtol=0, atol=0)
    assert data.prepare_image_inputs([1, 2], [], 99) == ([1, 2], [-1, -1], None)
    with pytest.raises(ValueError, match='placeholder'):
        data.prepare_image_inputs([99], [], 99)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_v41_vision_nondivisible_official(monkeypatch, dtype):
    _check_vision_official(monkeypatch, dtype)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_v41_vision_cuda_official(monkeypatch, dtype):
    assert torch.cuda.is_available()
    with torch.device("cuda"):
        _check_vision_official(monkeypatch, dtype)
        test_v41_multimage_copy_gradient()


@pytest.mark.parametrize('mutation', ['spatial_order', 'padding', 'rope_axes', 'seam'])
def test_v41_vision_rejects_mutations(monkeypatch, mutation):
    from megatron.lite.model.deepseek_v41.lite import vision

    if mutation == 'seam':
        original = vision.ViT.forward

        def wrong(self, patches, h, w):
            return original(self, patches, h, w).roll(1, 0)

        monkeypatch.setattr(vision.ViT, 'forward', wrong)
    elif mutation == 'rope_axes':
        original = vision.get_vision_cos_sin

        def wrong(h, w, dim, theta, device=None):
            return original(w, h, dim, theta, device)

        monkeypatch.setattr(vision, 'get_vision_cos_sin', wrong)
    else:

        def wrong(self, x, h, w):
            r = self.downsample_ratio
            spatial = x.reshape(h, w, -1).permute(2, 0, 1)
            if mutation == 'padding':
                spatial = torch.nn.functional.pad(spatial, (-w % r, 0, -h % r, 0))
            else:
                spatial = torch.nn.functional.pad(spatial, (0, -w % r, 0, -h % r))
            cells = torch.nn.functional.unfold(spatial[None], r, stride=r)[0].T
            if mutation == 'spatial_order':
                cells = cells.reshape(len(cells), -1, r * r).transpose(1, 2).flatten(1)
            return self.w2(torch.nn.functional.gelu(self.w1(cells)))

        monkeypatch.setattr(vision.Aligner, 'forward', wrong)
    with pytest.raises(AssertionError):
        _check_vision_official(monkeypatch, torch.float32)


@pytest.mark.parametrize('fault', ['overlap', 'bounds', 'delimiter', 'features'])
def test_v41_image_span_rejects_invalid(fault):
    from megatron.lite.model.deepseek_v41.lite import image_data as data

    layout = data.image_token_types(1, 2)
    start = -1 if fault == 'overlap' else 4 if fault == 'bounds' else 0
    if fault == 'delimiter':
        layout[0] = data.IMAGE
    image = data.ImageInput(start, torch.zeros(2, 3, 2, 2), 1, 2, layout)
    features = torch.ones(3 if fault == 'features' else 2, 4)
    with pytest.raises(ValueError):
        data.merge_image_embeddings(
            torch.zeros(1, 5, 4),
            [[image]],
            [[features]],
            *[torch.ones(4) for _ in range(3)],
        )


def test_v41_vision_assembly_live_owners_and_reachability(moe, monkeypatch):
    from megatron.lite.model.deepseek_v41.lite import image_data as data

    _, bundle = _assembly_bundle(text_only=False)
    model = bundle.chunks[0]
    patches = torch.randn(4, 3, 14, 14, requires_grad=True)
    result = model.encode_image(patches, 2, 2)
    result.square().sum().backward()
    assert patches.grad.abs().sum() > 0
    for root in ('vision', 'aligner'):
        for name, parameter in getattr(model, root).named_parameters():
            binding = model.tensor_bindings[root + '.' + name]
            assert binding.tensor is parameter and parameter.grad is not None

    class Reached(Exception):
        pass

    def probe(*args, **kwargs):
        raise Reached()

    for root in ('vision', 'aligner'):
        with monkeypatch.context() as patch:
            patch.setattr(getattr(model, root), 'forward', probe)
            with pytest.raises(Reached):
                model.encode_image(patches, 2, 2)
    layout = data.image_token_types(1, 1)
    images = [[data.ImageInput(1, patches, 2, 2, layout)]]
    embeddings = torch.randn(1, 6, model.embed.embedding_dim, requires_grad=True)
    merged = model.merge_image_embeddings(images, embeddings)
    assert merged.shape == embeddings.shape
    torch.testing.assert_close(merged[0, 2], result.detach()[0])
    merged.sum().backward()
    assert model.image_start.grad is not None


def test_v41_multimage_packed_model_forward(moe, device='cpu'):
    from megatron.lite.model.deepseek_v41.lite import image_data as data

    _, bundle = _assembly_bundle(device=device, text_only=False)
    model = bundle.chunks[0]
    first, second = [torch.randn(4, 3, 14, 14, requires_grad=True) for _ in range(2)]
    layout = data.image_token_types(1, 1)
    images = [
        [
            data.ImageInput(1, first, 2, 2, layout),
            data.ImageInput(7, second, 2, 2, layout),
        ]
    ]
    ids = torch.tensor([[1, 99, 99, 99, 99, 2, 3, 99, 99, 99, 99, 4]])
    types = torch.tensor([[-1, 0, 1, 2, 3, -1] * 2])
    result = model(
        ids, cu_seqlens=torch.tensor([0, 6, 12]), images=images, token_types=types
    )['logits']
    independent = []
    for index, patches in enumerate((first, second)):
        local = [[data.ImageInput(1, patches, 2, 2, layout)]]
        independent.append(
            model(ids[:, index * 6 : (index + 1) * 6], images=local)['logits']
        )
    torch.testing.assert_close(result, torch.cat(independent, 1))
    gradients = torch.autograd.grad(
        result[:, 6:].square().sum(), (first, second, model.image_start)
    )
    assert not gradients[0].any() and gradients[1].abs().sum() > 0
    assert torch.isfinite(gradients[2]).all() and gradients[2].abs().sum() > 0
    with pytest.raises(ValueError, match='Token types'):
        model(ids, images=images, token_types=torch.full_like(types, -1))
    with pytest.raises(ValueError, match='crosses'):
        model(ids, images=images, cu_seqlens=torch.tensor([0, 3, 12]))


@pytest.mark.gpus(1)
def test_v41_multimage_cuda_model_forward(moe):
    assert torch.cuda.is_available()
    with torch.device('cuda'):
        test_v41_multimage_packed_model_forward(moe, device='cuda')


@pytest.mark.parametrize('mutation', ['patch_order', 'delimiter'])
def test_v41_image_processor_rejects_mutations(monkeypatch, mutation):
    from megatron.lite.model.deepseek_v41.lite import image_data as data

    if mutation == 'patch_order':
        original = data.preprocess_image

        def wrong(*args, **kwargs):
            patches, *grid = original(*args, **kwargs)
            return patches.flip(0), *grid

        monkeypatch.setattr(data, 'preprocess_image', wrong)
    else:
        original = data.image_token_types

        def wrong(*args):
            return original(*args).roll(1)

        monkeypatch.setattr(data, 'image_token_types', wrong)
    with pytest.raises(AssertionError):
        test_v41_image_processor_official((125, 97), None, monkeypatch)


def test_v41_packed_record_preserves_every_sequence(moe):
    from megatron.lite.primitive.modules.router_replay import (
        RouterReplay,
        RouterReplayAction,
        attach_router_replay,
        detach_router_replay,
    )

    _, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    try:
        assert attach_router_replay(model) == 40
        RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
        ids = torch.tensor([[3, 4, 5, 6, 7]])
        with torch.no_grad():
            model(ids, cu_seqlens=torch.tensor([0, 2, 5], dtype=torch.int32))
        actual = [x.clone() for x in RouterReplay.get_recorded_data()]
        assert all(
            x.shape == (5, 6) for x in actual
        ), 'G2_PACKED_RECORD_LOSS: [2,3] sequences must retain all five routing rows'
        expected = []
        for row in (ids[:, :2], ids[:, 2:]):
            with torch.no_grad():
                model(row)
            expected.append([x.clone() for x in RouterReplay.get_recorded_data()])
        assert all(
            torch.equal(x, torch.cat([a, b])) for x, a, b in zip(actual, *expected)
        ), 'G2_PACKED_BOUNDARY: packed routing differs from independent sequence routing'
    finally:
        RouterReplay.clear_global_state()
        detach_router_replay(model)
        RouterReplay.clear_global_router_replay_instances()


@pytest.mark.parametrize('cp_rank', [0, 1])
def test_v41_replay_contiguous_cp_pp_alignment(cp_rank):
    from megatron.lite.model import protocol_utils
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.backends.mlite.router_replay import (
        RouterReplayDriver,
        _protocol_fn,
    )
    from megatron.lite.runtime.contracts import PackedBatch

    ps = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        cp_size=2,
        cp_rank=cp_rank,
        cp_group=None,
        pp_size=2,
        pp_rank=1,
    )
    model = SimpleNamespace(ps=ps)
    ids = torch.arange(12)
    mask = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool)
    batch = PackedBatch(ids, ids, torch.tensor([5, 7]), r3_replay_mask=mask)
    rows = [
        torch.arange(5 * 4 * 2).reshape(5, 4, 2),
        torch.arange(7 * 4 * 2).reshape(7, 4, 2) + 100,
    ]
    routed = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    driver = RouterReplayDriver(SimpleNamespace(_extras={}, _model=model), 'replay')
    driver._ps, driver._num_routers, driver._pp_offset, driver._pp_total = ps, 2, 2, 4
    local_layers = driver._select_local_layers(routed)
    pack = _protocol_fn(
        protocol, 'pack_routed_experts', protocol_utils.pack_routed_experts
    )
    pack_mask = _protocol_fn(
        protocol, 'pack_r3_replay_mask', protocol_utils.pack_r3_replay_mask
    )
    actual, actual_mask = pack(model, batch, local_layers), pack_mask(model, batch)
    # E contiguous metadata pads [5,7] to [6,8]; CP cuts inside sample two.
    full = torch.zeros(14, 4, 2, dtype=torch.long)
    full[:5], full[6:13] = rows
    full_mask = torch.zeros(14, dtype=torch.bool)
    full_mask[:5], full_mask[6:13] = mask[:5], mask[5:]
    expected = full[cp_rank * 7 : (cp_rank + 1) * 7, 2:4]
    assert torch.equal(
        torch.stack(actual, dim=1), expected
    ), 'G2_CONTIGUOUS_ROUTES: CP token order or PP global layer selection differs'
    assert torch.equal(
        actual_mask, full_mask[cp_rank * 7 : (cp_rank + 1) * 7]
    ), 'G2_CONTIGUOUS_MASK: replay mask must share the contiguous CP token layout'


def test_v41_driver_record_replay_roundtrip_and_fail_loud(moe):
    from megatron.lite.runtime.backends.mlite.router_replay import RouterReplayDriver
    from megatron.lite.runtime.contracts import PackedBatch

    proto, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    ids = torch.tensor([3, 4, 5, 6, 7])
    batch = PackedBatch(
        ids,
        ids,
        torch.tensor([2, 3]),
        torch.ones(5),
        r3_replay_mask=torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool),
    )
    handle = SimpleNamespace(_model=model, _extras={'protocol': proto})
    record = RouterReplayDriver(handle, 'record')
    try:
        record.begin()
        expected = record.wrap(bundle.forward_step)(model, batch)
        assert (
            'routed_experts' in expected
        ), 'G2_RECORD_OUTPUT: runtime must return recorded routes'
        expected['loss'].backward()
        gradients = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
    finally:
        record.end()
    model.zero_grad(set_to_none=True)
    batch.routed_experts = expected['routed_experts']
    replay = RouterReplayDriver(handle, 'replay')
    try:
        replay.begin()
        actual = replay.wrap(bundle.forward_step)(model, batch)
        torch.testing.assert_close(
            actual['logits'],
            expected['logits'],
            atol=3e-5,
            rtol=3e-5,
            msg='G2_REPLAY_LOGITS: identical-weight record/replay baseline differs',
        )
        actual['loss'].backward()
        for name, p in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(
                    p.grad,
                    gradients[name],
                    atol=3e-5,
                    rtol=3e-5,
                    msg=f'G2_REPLAY_GRADIENT: identical-route backward differs: {name}',
                )
    finally:
        replay.end()
    with pytest.raises(RuntimeError, match='active replay driver'):
        bundle.forward_step(model, batch)


def test_v41_replay_stage_roots_follow_e_global_slots():
    from megatron.lite.model.deepseek_v41.lite import protocol

    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    stage = SimpleNamespace(layers=[None, None, a, b], local_layer_range=(2, 4))
    assert protocol.router_replay_roots(SimpleNamespace(module=stage)) == [a, b]
    stage.layers[0] = a
    with pytest.raises(ValueError, match='stage-owned global layer slots'):
        protocol.router_replay_roots(stage)


def _v41_replay_parallel_worker(rank, rendezvous, pp, tp):
    from datetime import timedelta

    import torch.distributed as dist
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter
    from megatron.lite.primitive.modules.router_replay import (
        PackedRouterReplay,
        RouterReplay,
        RouterReplayAction,
        attach_router_replay,
        detach_router_replay,
    )
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.backends.mlite.router_replay import RouterReplayDriver
    from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig

    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group(
        'nccl',
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    model = torch.nn.Module()
    try:
        ps = init_parallel(ParallelConfig(tp=tp, cp=2, pp=pp, ep=2))
        start, end = ((0, 1) if ps.pp_rank == 0 else (1, 4)) if pp == 2 else (0, 4)
        cfg = SimpleNamespace(
            hidden_size=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
            routed_scaling_factor=1.0,
        )
        model.layers = torch.nn.ModuleList(
            [
                (
                    SigmoidTopKRouter(cfg, ps, compute_aux_loss=False).to(device)
                    if start <= i < end
                    else None
                )
                for i in range(4)
            ]
        )
        model.ps, model.local_layer_range = ps, (start, end)
        ids = torch.arange(12, device=device)
        mask = torch.tensor(
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], device=device, dtype=torch.bool
        )
        batch = PackedBatch(
            ids, ids, torch.tensor([5, 7], device=device), r3_replay_mask=mask
        )
        # Independent E padding: TP1 [6,8], TP2 [8,8]; retain global sample positions.
        second_start, total = (6, 14) if tp == 1 else (8, 16)
        local_size = total // (2 * tp)
        offset = ps.cp_rank * (total // 2) + ps.tp_rank * local_size
        tokens = torch.arange(offset, offset + local_size, device=device)
        base = torch.stack([((tokens + i) % 7).float() / 3 for i in range(4)], dim=-1)
        local_routers = protocol.router_replay_roots(model)
        # No-feature baseline, before any hook is attached.
        baseline = [
            router.route_logits(base.roll(start + i, dims=-1))
            for i, router in enumerate(local_routers)
        ]
        attach_router_replay(model)
        RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
        scope = PackedRouterReplay(local_size)
        recorded_scores = [[] for _ in local_routers]
        # Two segments exercise per-checkpoint concatenation independently of the scheduler.
        for lo, hi in ((0, 1), (1, local_size)):
            with scope.sequence(lo, hi):
                for i, router in enumerate(local_routers):
                    scores, _ = router.route_logits(
                        base[lo:hi].roll(start + i, dims=-1)
                    )
                    recorded_scores[i].append(scores)
        scope.finish()
        recorded = RouterReplay.get_recorded_data()
        for i, (scores, indices) in enumerate(baseline):
            assert torch.equal(
                recorded[i], indices
            ), 'G2_GPU_BASELINE: recording changed native routes'
            torch.testing.assert_close(
                torch.cat(recorded_scores[i]),
                scores,
                atol=0,
                rtol=0,
                msg='G2_GPU_BASELINE: recording changed native scores',
            )
        # All stage-local forwards have finished; this is the E post-drain seam.
        dist.barrier()
        routes = protocol.unpack_recorded_routed_experts(
            model, batch, recorded, pipeline_drained=True
        )
        assert all(
            row.shape[1:] == (4, 2) for row in routes.unbind()
        ), 'G2_GPU_PP_GATHER: variable-width stage routes lost global layer columns'
        batch.routed_experts = routes
        detach_router_replay(model)
        RouterReplay.clear_global_router_replay_instances()
        driver = RouterReplayDriver(
            SimpleNamespace(_model=model, _extras={'protocol': protocol}), 'replay'
        )
        driver.begin()
        try:
            local_routes = driver._select_local_layers(routes)
            targets = protocol.pack_routed_experts(model, batch, local_routes)
            local_mask = protocol.pack_r3_replay_mask(model, batch)
            expected_mask = torch.zeros(total, dtype=torch.bool, device=device)
            expected_mask[:5], expected_mask[second_start : second_start + 7] = (
                mask[:5],
                mask[5:],
            )
            assert torch.equal(
                local_mask, expected_mask[offset : offset + local_size]
            ), 'G2_GPU_CP_MASK: mask differs from independently sliced contiguous tokens'
            for actual, expected in zip(targets, recorded):
                assert torch.equal(
                    actual[local_mask], expected[local_mask]
                ), 'G2_GPU_CP_PP_ROUTES: record/gather/pack changed active token or layer positions'
            logits = (-base).detach().requires_grad_()

            def forward(_model, _batch):
                scope = PackedRouterReplay(local_size)
                outputs = [[] for _ in local_routers]
                for lo, hi in ((0, 1), (1, local_size)):
                    with scope.sequence(lo, hi):
                        for i, router in enumerate(local_routers):
                            scores, indices = router.route_logits(
                                logits[lo:hi].roll(start + i, dims=-1)
                            )
                            outputs[i].append(scores * (indices + 1))
                scope.finish()
                return {
                    'loss': sum(torch.cat(parts).square().sum() for parts in outputs)
                }

            actual = driver.wrap(forward)(model, batch)['loss']
            actual.backward()
            reference_logits = logits.detach().clone().requires_grad_()
            reference = 0
            for i, target in enumerate(targets):
                dense = reference_logits.roll(start + i, dims=-1).sigmoid()
                native = torch.sort(dense.topk(2, dim=-1).indices, dim=-1).values
                selected = torch.where(local_mask[:, None], target, native)
                scores = dense.gather(1, selected)
                scores = scores / scores.sum(-1, keepdim=True)
                reference = reference + (scores * (selected + 1)).square().sum()
            reference.backward()
            torch.testing.assert_close(
                actual,
                reference,
                atol=2e-5,
                rtol=2e-5,
                msg='G2_GPU_REPLAY_VALUE: replay differs from independent live scores',
            )
            torch.testing.assert_close(
                logits.grad,
                reference_logits.grad,
                atol=2e-5,
                rtol=2e-5,
                msg='G2_GPU_REPLAY_GRADIENT: route/mask boundary gradient differs',
            )
        finally:
            driver.end()
    finally:
        RouterReplay.clear_global_state()
        RouterReplay.clear_global_router_replay_instances()
        dist.destroy_process_group()


@pytest.mark.gpus(4)
@pytest.mark.parametrize('pp,tp', [(2, 1), (1, 2)])
def test_v41_replay_parallel_primitives(tmp_path, pp, tp):
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 4:
        pytest.skip(
            'Routing replay parallel coverage requires at least 4 visible GPUs.'
        )

    mp.spawn(
        _v41_replay_parallel_worker,
        args=(f'file://{tmp_path / "replay-rendezvous"}', pp, tp),
        nprocs=4,
        join=True,
    )


def test_v41_pp_record_fails_before_stage_collectives():
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.backends.mlite.router_replay import RouterReplayDriver

    model = SimpleNamespace(layers=[torch.nn.Linear(2, 2)], ps=ParallelState(pp_size=2))
    handle = SimpleNamespace(_model=model, _extras={'protocol': protocol})
    with pytest.raises(NotImplementedError, match='post-drain route collection'):
        RouterReplayDriver(handle, 'record').begin()


def test_v41_replay_forward_keeps_named_boundary_failure():
    from megatron.lite.runtime.backends.mlite.router_replay import RouterReplayDriver

    batch = SimpleNamespace(routed_experts=torch.zeros(1, 1, 1, 1, dtype=torch.long))
    proto = SimpleNamespace(
        pack_routed_experts=lambda *args: [], pack_r3_replay_mask=lambda *args: None
    )
    driver = RouterReplayDriver(
        SimpleNamespace(_model=None, _extras={'protocol': proto}), 'replay'
    )
    driver._num_routers = 1

    def unavailable(model, batch):
        raise NotImplementedError('E_PIPELINE_BOUNDARY_NOT_READY')

    with pytest.raises(NotImplementedError, match='E_PIPELINE_BOUNDARY_NOT_READY'):
        driver.wrap(unavailable)(None, batch)


def test_v41_replay_keeps_legacy_padding_default():
    from megatron.lite.model import protocol_utils
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.contracts import PackedBatch

    model = SimpleNamespace(ps=ParallelState(cp_size=2))
    batch = PackedBatch(torch.arange(12), torch.arange(12), torch.tensor([5, 7]))
    routes = torch.nested.as_nested_tensor(
        [torch.arange(5).reshape(5, 1, 1), torch.arange(7).reshape(7, 1, 1) + 10],
        layout=torch.jagged,
    )
    legacy = protocol_utils.pack_routed_experts(model, batch, routes, contiguous=True)[
        0
    ]
    current = protocol.pack_routed_experts(model, batch, routes)[0]
    assert legacy[:, 0].tolist() == [
        0,
        1,
        2,
        3,
        4,
        0,
        0,
        0,
    ], 'G2_LEGACY_PADDING: existing DS4/GLM contiguous slicing must keep its old alignment'
    assert current[:, 0].tolist() == [
        0,
        1,
        2,
        3,
        4,
        0,
        10,
    ], 'G2_E_PADDING: V4.1 must use E TP*CP alignment and keep the cross-sample CP boundary'


def test_v41_g1_text_only_mask_excludes_live_visual_owners(moe):
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import parameter_groups

    _, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    visual = [
        b.tensor
        for b in model.parameter_bindings()
        if b.role in ('vision', 'aligner', 'image_delimiter')
    ]
    assert visual and not any(
        p.requires_grad for p in visual
    ), 'G1_TEXT_MASK: text-only visual owners must be frozen'
    routed = {id(p) for g in parameter_groups(model, lr=1e-3) for p in g['params']}
    assert routed == {
        id(p) for p in model.parameters() if p.requires_grad
    }, 'G1_ROUTED_SET: every trainable owner must reach the optimizer exactly once'


def _g1_release_layout():
    import hashlib
    import itertools
    import json
    from pathlib import Path

    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.checkpoint import bind_checkpoint

    root = Path(__file__).parents[3]
    raw = (root / 'tests/fixtures/deepseek_v41/reference/config.json').read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879'
    ), 'G1_CONFIG: release dimensions must remain pinned'
    cfg = protocol.build_model_config(json.loads(raw))
    bundle = protocol.build_model(cfg, impl_cfg=protocol.ImplConfig(device='meta'))
    model = bundle.chunks[0]
    families = json.loads(
        (root / 'docs/contracts/deepseek_v41/weights.json').read_text()
    )['families']
    keys = [
        f['pattern'].format(*ix)
        for f in families
        for ix in itertools.product(*f['indices'])
    ]
    bindings = bind_checkpoint(model, keys)
    assert len(bindings) == 96085, 'G1_FULL_KEYS: every release key must be bound'
    assert model.local_layer_range == (
        0,
        40,
    ), 'G1_LOCAL_LAYOUT: single-rank construction must own all 40 global layers'
    assert len(model.layers) == 40 and all(
        len(b.ffn.experts) == 384 for b in model.layers
    ), 'G1_EXPERT_LAYOUT: release experts must not be reduced'
    assert all(
        p.is_meta for p in model.parameters()
    ), 'G1_META: structural initialization must not allocate release payloads'
    assert all(
        b.is_meta for b in model.buffers()
    ), 'G1_META: resident buffers must share the structural device'
    assert tuple(model.embed.weight.shape) == (
        129280,
        5120,
    ), 'G1_EMBED_SHAPE: release vocabulary and hidden width'
    for i, layer in enumerate(model.layers):
        a = layer.attn
        assert (a.compressor is not None) == (
            i in (2, 8, 14, 20)
        ), 'G1_OWNER: compressor allocated at a non-owner layer'
        assert (a.indexer is not None) == (
            i in (2, 8, 14, 20, 24, 28, 32, 36)
        ), 'G1_OWNER: indexer allocated at a non-owner layer'
        layer_owners = {id(m) for m in layer.modules()}
        for binding in model.parameter_bindings():
            if binding.release_key.startswith(f'layers.{i}.'):
                assert (
                    id(binding.owner) in layer_owners
                ), 'G1_OWNER: release key must bind its actual global layer owner'
        assert tuple(a.wq_b.weight.shape) == (
            64 * 512,
            1280,
        ), 'G1_HEAD_LAYOUT: headwise query axes differ from release'
    parameter_ids = [id(b.tensor) for b in model.parameter_bindings()]
    assert (
        len(parameter_ids) == len(set(parameter_ids)) == len(list(model.parameters()))
    ), 'G1_OWNER: physical parameters must have unique owners'
    table_bytes = 0
    for index, rows in zip(
        (1, 14), cfg.to_hf_dict()['text_config']['engram_num_embeddings']
    ):
        table = model.layers[index].engram.embed
        assert table.master is None and not list(
            table.parameters()
        ), 'G1_FROZEN_STORAGE: frozen Engram must not allocate a high-precision master'
        assert (
            table.weight.dtype == torch.float8_e4m3fn
            and table.scale.dtype == torch.float8_e8m0fnu
        ), 'G1_FROZEN_STORAGE: preserve FP8 table and E8M0 scales'
        assert tuple(table.weight.shape) == (rows, 256) and tuple(
            table.scale.shape
        ) == (rows, 8), 'G1_TABLE_LAYOUT: row-wise scale layout must match release'
        table_bytes += table.weight.numel() + table.scale.numel()
    # These are physical dtype/shape estimates, not a claim of materialized GPU memory.
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    assert (
        table_bytes > 190_000_000_000
    ), 'G1_MEMORY: full Engram storage must not be replaced by a proxy'
    assert (
        parameter_bytes > 900_000_000_000
    ), 'G1_MEMORY: report actual floating numerical-provider storage'
    print(
        f'G1_STRUCTURE keys={len(bindings)} parameter_bytes={parameter_bytes} buffer_bytes={buffer_bytes} frozen_engram_bytes={table_bytes}'
    )
    return model


def test_v41_g1_release_structure(moe):
    _g1_release_layout()


def _g1_equal(actual, expected, label):
    if isinstance(expected, torch.Tensor):
        assert (
            isinstance(actual, torch.Tensor)
            and actual.dtype == expected.dtype
            and actual.shape == expected.shape
        ), label
        assert torch.equal(
            actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        ), label
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), label
        for key in expected:
            _g1_equal(actual[key], expected[key], f'{label}/{key}')
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), label
        for i, (a, e) in enumerate(zip(actual, expected)):
            _g1_equal(a, e, f'{label}/{i}')
    else:
        assert actual == expected, label


@pytest.mark.gpus(1)
@pytest.mark.parametrize('trainable', [False, True])
@pytest.mark.parametrize('quantized', [False, True])
def test_v41_g1_training_checkpoint_bitwise(
    tmp_path, moe, trainable, quantized, monkeypatch
):
    import random
    from copy import deepcopy

    import numpy as np
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig
    from megatron.lite.primitive.ckpt import dcp as checkpoint
    from megatron.lite.runtime.contracts import PackedBatch

    config = _assembly_config()
    impl = protocol.ImplConfig(
        device='cuda',
        dtype=torch.bfloat16,
        quantized=quantized,
        token_map=list(range(256)),
        trainable_engram=trainable,
        optimizer='muon',
        optimizer_config=OptimizerConfig(
            lr=1e-3, ns_steps=5, coefficient_type='quintic'
        ),
    )

    def build():
        bundle = protocol.build_model(config, impl_cfg=impl)
        schedulers = [
            torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=0.8)
            for o in bundle.optimizer.optimizers
        ]
        return bundle, schedulers

    def seed():
        random.seed(915)
        np.random.seed(915)
        torch.manual_seed(915)
        torch.cuda.manual_seed_all(915)

    def step(bundle, schedulers):
        model, opt = bundle.chunks[0], bundle.optimizer
        # Consume all four RNG streams; a reset or missing restore changes the batch.
        offset = (
            random.randrange(30)
            + int(np.random.randint(30))
            + int(torch.randint(30, (1,)))
        ) % 200
        ids = (torch.randint(1, 32, (4,), device='cuda') + offset).long()
        batch = PackedBatch(ids, ids, torch.tensor([4], device='cuda'))
        opt.zero_grad()
        output = bundle.forward_step(model, batch)
        assert output['loss'].requires_grad and torch.isfinite(
            output['loss']
        ), 'G1_LOSS: actual protocol must produce finite differentiable loss'
        output['loss'].backward()
        for key in (
            'embed.weight',
            'head.weight',
            'layers.0.attn.wq_b.weight',
            'layers.20.attn.compressor.wkv.weight',
        ):
            p = model.tensor_bindings[key].tensor
            assert (
                p.grad is not None
                and p.grad is p.main_grad
                and torch.isfinite(p.grad).all()
                and p.grad.abs().sum() > 0
            ), f'G1_GRADIENT: missing native gradient at {key}'
        for block in model.layers:
            if block.attn.indexer is not None:
                assert all(
                    p.grad is None and not p.requires_grad
                    for p in block.attn.indexer.parameters()
                ), 'G1_INDEXER: frozen indexer must stay outside backward'
        for index in model.engram_layer_ids:
            table = model.layers[index].engram.embed
            assert (
                table.weight.is_cuda and table.scale.is_cuda
            ), 'G1_ENGRAM_RESIDENCY: no offload'
            if trainable:
                assert (
                    table.master.is_cuda and table.master.dtype == torch.float32
                ), 'G1_ENGRAM_MASTER: persistent FP32 owner'
                assert (
                    table.master.grad is not None
                    and torch.isfinite(table.master.grad).all()
                    and table.master.grad.abs().sum() > 0
                ), 'G1_ENGRAM_GRADIENT: trainable table must receive real gradients'
            else:
                assert (
                    table.master is None
                ), 'G1_FROZEN_STORAGE: frozen table must not allocate a master'
        before = model.embed.weight.detach().clone()
        accepted, norm, _ = opt.step()
        assert (
            accepted and norm > 0 and not torch.equal(before, model.embed.weight)
        ), 'G1_UPDATE: training exit must publish an optimizer update'
        for scheduler in schedulers:
            scheduler.step()
        return (
            ids.cpu(),
            output['loss'].detach().cpu(),
            [g['lr'] for g in opt.param_groups],
        )

    def state(bundle, schedulers):
        model = bundle.chunks[0]
        return deepcopy(
            dict(
                parameters={n: p.detach().cpu() for n, p in model.named_parameters()},
                buffers={n: b.detach().cpu() for n, b in model.named_buffers()},
                optimizer=bundle.optimizer.state_dict(),
                schedulers=[s.state_dict() for s in schedulers],
            )
        )

    seed()
    continuous, schedulers = build()
    history = [step(continuous, schedulers)]
    path = str(tmp_path / 'restart')
    checkpoint.save_training_checkpoint(
        continuous.chunks, continuous.optimizer, 1, path, use_dcp=False
    )
    # Scheduler ownership stays with the training caller, alongside the common checkpoint.
    torch.save([s.state_dict() for s in schedulers], tmp_path / 'schedulers.pt')
    saved = state(continuous, schedulers)
    history += [step(continuous, schedulers) for _ in range(2)]
    expected = state(continuous, schedulers)
    # Reconstructing model/optimizer deliberately consumes RNG before load restores it.
    restored, restarted_schedulers = build()
    iteration = checkpoint.load_training_checkpoint(
        restored.chunks, restored.optimizer, path, use_dcp=False
    )
    assert iteration == 1, 'G1_STEP: resume must restore the saved step'
    for scheduler, payload in zip(
        restarted_schedulers, torch.load(tmp_path / 'schedulers.pt', weights_only=False)
    ):
        scheduler.load_state_dict(payload)
    _g1_equal(state(restored, restarted_schedulers), saved, 'G1_RESTORE')
    resumed_history = [
        step(restored, restarted_schedulers) for _ in range(iteration, 3)
    ]
    _g1_equal(resumed_history, history[iteration:], 'G1_RNG_SCHEDULE_LOSS')
    _g1_equal(state(restored, restarted_schedulers), expected, 'G1_RESUME_BITWISE')
    # Corrupt the actual restore boundary, not just the comparison's input.
    original_load = checkpoint.load_training_checkpoint

    def corrupt_load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        with torch.no_grad():
            restored.chunks[0].embed.weight.flatten()[0].add_(1)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, 'load_training_checkpoint', corrupt_load)
        checkpoint.load_training_checkpoint(
            restored.chunks, restored.optimizer, path, use_dcp=False
        )
        with pytest.raises(AssertionError, match='G1_RESTORE/parameters/embed.weight'):
            _g1_equal(
                state(restored, restarted_schedulers)['parameters'],
                saved['parameters'],
                'G1_RESTORE/parameters',
            )
    original_load(restored.chunks, restored.optimizer, path, use_dcp=False)
    hook = restored.chunks[0].head.weight.register_hook(torch.zeros_like)
    try:
        with pytest.raises(
            AssertionError, match='G1_GRADIENT: missing native gradient at head.weight'
        ):
            step(restored, restarted_schedulers)
    finally:
        hook.remove()

    # Entry probes verify the production save/load/forward calls are truly reachable.
    class Reached(RuntimeError):
        pass

    def probe(*args, **kwargs):
        raise Reached('G1_ENTRY_REACHED')

    for module, attribute, call in (
        (
            checkpoint,
            'save_training_checkpoint',
            lambda: checkpoint.save_training_checkpoint(
                restored.chunks, restored.optimizer, 3, path, use_dcp=False
            ),
        ),
        (
            checkpoint,
            'load_training_checkpoint',
            lambda: checkpoint.load_training_checkpoint(
                restored.chunks, restored.optimizer, path, use_dcp=False
            ),
        ),
        (restored.chunks[0], 'forward', lambda: step(restored, restarted_schedulers)),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(module, attribute, probe)
            with pytest.raises(Reached, match='G1_ENTRY_REACHED'):
                call()
    print(
        f'G1_TRAINING quantized={quantized} trainable_engram={trainable} continuous=3 resumed=1+2 bitwise=true'
    )


@pytest.mark.parametrize(
    'mutation,reason',
    [
        ('layout', 'G1_LOCAL_LAYOUT'),
        ('owner', 'G1_OWNER'),
        ('storage', 'G1_FROZEN_STORAGE'),
    ],
)
def test_v41_g1_release_rejects_semantic_mutations(moe, monkeypatch, mutation, reason):
    from dataclasses import replace

    from megatron.lite.model.deepseek_v41.lite import protocol

    original = protocol.build_model

    def changed(*args, **kwargs):
        bundle = original(*args, **kwargs)
        model = bundle.chunks[0]
        if mutation == 'layout':
            model.local_layer_range = (0, 20)
        elif mutation == 'owner':
            key = 'layers.0.attn.wq_a.weight'
            model.tensor_bindings[key] = replace(
                model.tensor_bindings[key], owner=model.layers[1].attn.wq_a
            )
            other = 'layers.1.attn.wq_a.weight'
            model.tensor_bindings[other] = replace(
                model.tensor_bindings[other], owner=model.layers[0].attn.wq_a
            )
        else:
            table = model.layers[1].engram.embed
            table.scale = table.scale.float()
        return bundle

    monkeypatch.setattr(protocol, 'build_model', changed)
    with pytest.raises(AssertionError, match=reason):
        _g1_release_layout()


def test_v41_g1_release_constructor_reachable(moe, monkeypatch):
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    class Reached(RuntimeError):
        pass

    def probe(*args, **kwargs):
        raise Reached('G1_CONSTRUCTOR_REACHED')

    monkeypatch.setattr(DeepseekV41Model, '__init__', probe)
    with pytest.raises(Reached, match='G1_CONSTRUCTOR_REACHED'):
        _g1_release_layout()


def test_v41_g1_checkpoint_rng_does_not_import_unused_core(monkeypatch):
    import builtins
    import sys

    from megatron.lite.primitive.ckpt import dcp

    original = builtins.__import__

    def reject_core(name, *args, **kwargs):
        assert not name.startswith(
            'megatron.core'
        ), 'G1_RNG_DEPENDENCY: standalone checkpoint must not initialize an unused Core RNG tracker'
        return original(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, 'megatron.core.tensor_parallel', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(builtins, '__import__', reject_core)
    assert dcp._get_cuda_rng_tracker_states() == {}


def test_v41_g1_checkpoint_preserves_loaded_core_tracker(monkeypatch):
    import sys

    from megatron.lite.primitive.ckpt import dcp

    values = {'model-parallel-rng': torch.tensor([1, 2, 3], dtype=torch.uint8)}
    tracker = SimpleNamespace(get_states=lambda: values)
    module = SimpleNamespace(get_cuda_rng_tracker=lambda: tracker)
    monkeypatch.setitem(sys.modules, 'megatron.core.tensor_parallel', module)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    saved = dcp._get_cuda_rng_tracker_states()
    _g1_equal(saved, values, 'G1_CORE_RNG')
    values['model-parallel-rng'].zero_()
    assert (
        saved['model-parallel-rng'].sum() == 6
    ), 'G1_CORE_RNG: snapshot must not alias live tracker state'


@pytest.mark.parametrize('axis', ['tp', 'ep', 'cp'])
def test_v41_g1_distributed_construction_boundary(moe, axis):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    with pytest.raises(NotImplementedError, match='TP/EP/CP/VPP remain pending'):
        protocol.build_model(
            _assembly_config(),
            impl_cfg=protocol.ImplConfig(
                device='meta', parallel=ParallelConfig(**{axis: 2})
            ),
        )


def test_v41_g1_stage_export_boundary(moe):
    from megatron.lite.model.deepseek_v41.lite.checkpoint import export_model
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    with torch.device('meta'):
        stage = DeepseekV41Model(_assembly_config(), layer_range=(0, 20))
    with pytest.raises(NotImplementedError, match='distributed checkpoint assembly'):
        next(export_model(stage))


@pytest.mark.gpus(4)
@pytest.mark.parametrize('recompute', [False, True])
def test_real_packed_pipeline_scheduler_forward_backward_update(tmp_path, recompute):
    import os

    import torch.multiprocessing as mp

    assert os.getenv('SLURM_JOB_ID'), 'Packed pipeline validation requires Slurm'
    if torch.cuda.device_count() < 4:
        pytest.skip('Requires the declared four-GPU allocation')
    mp.spawn(
        _real_model_worker,
        args=(f'file://{tmp_path}/packed-scheduler', True, recompute),
        nprocs=4,
        join=True,
    )


@pytest.mark.parametrize('recompute', [False, True])
def test_packed_pipeline_scheduler_local_gradients_and_lifetime(moe, recompute):
    from types import SimpleNamespace

    from megatron.lite.model.deepseek_v41.lite.pipeline import PackedPipelineAdapter
    from megatron.lite.primitive.parallel.pipeline import forward_backward_pipelining
    from megatron.lite.runtime.contracts import PackedBatch

    _, bundle = _assembly_bundle()
    model = bundle.chunks[0]
    batches = [
        PackedBatch(torch.arange(3, 3 + sum(ns)), None, torch.tensor(ns))
        for ns in ((5, 8, 3), (4, 7))
    ]
    expected = [bundle.forward_step(model, batch)['logits'] for batch in batches]
    sum(logits.float().square().sum() / 2 for logits in expected).backward()
    gradients = {
        name: None if p.grad is None else p.grad.clone()
        for name, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    adapter = PackedPipelineAdapter(model, bundle.parallel_state, recompute=recompute)
    seen = []

    def objective(out, batch):
        index = next(i for i, candidate in enumerate(batches) if candidate is batch)
        torch.testing.assert_close(out['logits'], expected[index], atol=0, rtol=0)
        seen.append(index)
        return out['logits'].float().square().sum(), {}

    forward_backward_pipelining(
        bundle.forward_step,
        [model],
        iter(batches),
        SimpleNamespace(num_microbatches=2),
        bundle.parallel_state,
        loss_fn=objective,
        payload_adapter=adapter,
    )
    assert seen == [0, 1] and adapter.step == 1
    for name, p in model.named_parameters():
        if gradients[name] is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, gradients[name], atol=1e-5, rtol=1e-5)
    with pytest.raises(RuntimeError, match='stale'):
        adapter.backward(0, None)
    with pytest.raises(RuntimeError, match='active'):
        adapter.finish()


def test_v41_pipeline_optimizer_groups_cover_local_owners(moe):
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import parameter_groups

    _, bundle = _assembly_bundle()
    expected = {g['owner_key'] for g in parameter_groups(bundle.chunks[0], lr=1e-3)}
    actual = set()
    for start, end in ((0, 10), (10, 20), (20, 30), (30, 40)):
        model = DeepseekV41Model(
            _assembly_config(),
            token_map=list(range(256)),
            quantized=False,
            layer_range=(start, end),
        )
        for binding in model.parameter_bindings():
            if binding.role in ('vision', 'aligner', 'image_delimiter'):
                binding.tensor.requires_grad_(False)
        groups = parameter_groups(model, lr=1e-3)
        owned = {g['owner_key'] for g in groups}
        assert not owned & actual
        actual.update(owned)
        assert {id(p) for g in groups for p in g['params']} == {
            id(p) for p in model.parameters() if p.requires_grad
        }
    assert actual == expected


@pytest.mark.gpus(4)
def test_real_packed_pipeline_mixed_optimizer_clip_and_atomic_skip(tmp_path):
    import os

    import torch.multiprocessing as mp

    assert os.getenv('SLURM_JOB_ID'), 'Mixed PP validation requires Slurm'
    if torch.cuda.device_count() < 4:
        pytest.skip('Requires the declared four-GPU allocation')
    mp.spawn(
        _real_model_worker,
        args=(f'file://{tmp_path}/mixed-pp', True, True, True),
        nprocs=4,
        join=True,
    )
