# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""New primitive files have resolvable, executable-test-backed skill routes."""
import ast
from pathlib import Path


def test_bound_training_routes_and_skill_contract():
    root = Path(__file__).resolve().parents[3]
    skills = root / 'skills'
    leaf = skills / 'primitive/bound-training.md'
    text = leaf.read_text()
    schema = ast.parse(text.split('```python')[1].split('```')[0]).body[0].value
    assert ast.literal_eval(schema.args[0]) == 'primitive.bound_training'
    keywords = {k.arg: ast.literal_eval(k.value) for k in schema.keywords}
    for name in keywords['imports'] + keywords['calls']:
        assert (skills / (name.replace('.', '/').replace('_', '-') + '.md')).is_file()
    body = ast.parse(text.split('```python')[2].split('```')[0])
    routes = ast.literal_eval(body.body[0].value)
    required = {
        'config_fields.py',
        'ckpt/binding_records.py',
        'modules/engram_lookup.py',
        'modules/owner_row_transport.py',
        'modules/row_memory_build.py',
        'modules/paired_stream.py',
        'modules/image_data.py',
        'modules/vision.py',
        'modules/vision_training.py',
        'modules/native_fp32_linear.py',
        'quantization/mxfp8.py',
        'quantization/nvfp4.py',
        'optimizers/headwise_muon.py',
        'optimizers/owned_groups.py',
        'optimizers/sinkhorn.py',
        'optimizers/staged_update.py',
        'parallel/owned_ddp.py',
    }
    assert routes.keys() == required
    for source, test in routes.items():
        assert (root / 'megatron/lite/primitive' / source).is_file()
        tree = ast.parse((root / 'tests/unit' / test).read_text())
        assert any(
            isinstance(n, ast.FunctionDef) and n.name.startswith('test_')
            for n in tree.body
        )
    function = body.body[1]
    assert function.name == 'bound_training'
    assert [a.arg for a in function.args.args] == keywords['inputs']
    exits = {
        n.value.func.id
        for n in ast.walk(function)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
    }
    assert exits == set(keywords['exits'])
    loops = [n for n in ast.walk(function) if isinstance(n, (ast.For, ast.While))]
    assert len(loops) == 1 and isinstance(loops[0].iter, ast.Subscript)
    assert 'primitive.bound_training' in (skills / 'README.md').read_text()
    compose = (skills / 'primitive/select-for-compose.md').read_text()
    assert 'bound = primitive.bound_training(' in compose
