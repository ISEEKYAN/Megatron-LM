# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Closed text prefixes execute a real head and preserve checkpoint mirrors.

Small CPU fixtures test ownership, not CUDA deployment arithmetic or release
weight fidelity. Engram CUDA arithmetic is covered separately.
"""
from dataclasses import replace

import pytest
import torch
from megatron.lite.model.deepseek_v41 import topology
from safetensors.torch import save_file
from test_redo_parity import release_config


@pytest.mark.parametrize('count', [2, 4])
def test_closed_prefix_head_backward_and_checkpoint_mirror(
    v41_core_te, tmp_path, count
):
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.model.deepseek_v41.lite import checkpoint, protocol
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

    source = release_config().to_hf_dict()
    text = source['text_config']
    text.update(
        num_hidden_layers=count,
        compress_ratios=text['compress_ratios'][:count] + [0] * 3,
        kv_source_layer_ids=[2] if count > 2 else [],
        index_source_layer_ids=[2] if count > 2 else [],
        candidate_source_layer_id=-1,
        engram_layer_ids=[],
        engram_num_embeddings=[],
    )
    cfg = DeepseekV41Config(source)
    impl = protocol.ImplConfig(device='cpu', dtype=torch.float32, quantized=False)
    torch.manual_seed(119)
    model = protocol.build_model(cfg, impl_cfg=impl).chunks[0]
    assert model.pipeline_cut == count
    ids = torch.tensor([[2, 3, 9], [7, 11, 13]])
    result = model(ids)
    assert 'hidden_states' not in result
    assert result['logits'].shape == (2, 3, 64)
    assert torch.isfinite(result['logits']).all()
    result['logits'].square().mean().backward()
    for tensor in (model.head.weight, model.embed.weight):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all() and torch.count_nonzero(tensor.grad)

    archive = tmp_path / 'archive'
    archive.mkdir()
    save_file(
        {key: torch.arange(7, dtype=torch.uint8) for key in model.archival_bindings},
        str(archive / 'model.safetensors'),
    )
    model.archival_store = SafeTensorReader(str(archive))
    model.archival_keys = sorted(model.archival_bindings)
    checkpoint.save_model(model, tmp_path / 'checkpoint')
    restored = protocol.build_model(cfg, impl_cfg=impl).chunks[0]
    checkpoint.load_model(restored, tmp_path / 'checkpoint')
    assert torch.equal(model(ids)['logits'], restored(ids)['logits'])
    original = dict(checkpoint.export_checkpoint(model, cpu=True))
    loaded = dict(checkpoint.export_checkpoint(restored, cpu=True))
    assert original.keys() == loaded.keys()
    for key in original:
        a, b = original[key], loaded[key]
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        ), key


@pytest.mark.parametrize("count", [2, 4])
def test_closed_release_prefix_keeps_published_owner_boundaries(count):
    release = topology.TopologySpec()
    sources = (2,) if count > 2 else ()
    prefix = replace(
        release,
        num_hidden_layers=count,
        compress_ratios=release.compress_ratios[:count] + (0,) * 3,
        kv_source_layer_ids=sources,
        index_source_layer_ids=sources,
        candidate_source_layer_id=-1,
        engram_layer_ids=(1,),
        engram_num_embeddings=(release.engram_num_embeddings[0],),
    )
    policies = topology.build_topology(prefix)
    assert [p.compress_ratio for p in policies] == list(release.compress_ratios[:count])
    assert [p.kv_owner for p in policies] == [None, None] + [2] * (count - 2)
    assert [p.index_owner for p in policies] == [None, None] + [2] * (count - 2)
    assert all(p.candidate_mode == "none" and not p.is_ced_boundary for p in policies)
    assert policies[1].engram_rows == 384006168
    assert policies[1].engram_slot == 0


def test_disabled_candidate_cannot_silently_remove_ratio_one_publisher():
    with pytest.raises(ValueError, match="disabled candidate.*ratio 1"):
        topology.build_topology(
            replace(topology.TopologySpec(), candidate_source_layer_id=-1)
        )


def run_documented_recipe(path, cfg, tmp_path, monkeypatch):
    """Execute the shipped recipe; substitute only CPU/EP1 hardware settings."""
    import ast
    import json

    source = path.read_text().split('```python\n')[1].split('```')[0]
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == 'ImplConfig':
                for keyword in node.keywords:
                    if keyword.arg == 'device':
                        keyword.value = ast.Constant('cpu')
                    elif keyword.arg == 'quantized':
                        keyword.value = ast.Constant(False)
            elif node.func.id == 'ParallelConfig':
                for keyword in node.keywords:
                    if keyword.arg == 'ep':
                        keyword.value = ast.Constant(1)
    (tmp_path / 'config.json').write_text(json.dumps(cfg.to_hf_dict()))
    monkeypatch.chdir(tmp_path)
    scope = {}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), scope)
    assert scope['bundle'].optimizer is not None
    assert scope['impl'].optimizer_config.lr == 1e-6
    return scope['bundle']


def test_documented_closed_prefix_recipe_constructs(v41_core_te, tmp_path, monkeypatch):
    from pathlib import Path

    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    source = release_config().to_hf_dict()
    source['text_config'].update(
        num_hidden_layers=2,
        compress_ratios=[0] * 5,
        candidate_source_layer_id=-1,
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        engram_layer_ids=[],
        engram_num_embeddings=[],
    )
    path = Path(__file__).resolve().parents[3] / 'examples/verl/DS41_CLOSED_PREFIX.md'
    bundle = run_documented_recipe(
        path, DeepseekV41Config(source), tmp_path, monkeypatch
    )
    assert bundle.chunks[0].pipeline_cut == 2
