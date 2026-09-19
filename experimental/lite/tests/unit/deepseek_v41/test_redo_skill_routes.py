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
        'config_fields.py': 'test_redo_parity.py::test_real_pp2_boundary_restarts_attention_state',
        'ckpt/binding_records.py': 'test_redo_bindings.py::test_parameter_bindings_require_exact_owner_inventory',
        'modules/engram_lookup.py': 'test_redo_engram_residency.py::test_forward_does_not_mutate_or_release_storage',
        'modules/owner_row_transport.py': 'test_redo_engram_residency.py::test_owner_transport_preserves_compact_rows_and_backward',
        'modules/row_memory_build.py': 'test_redo_parity.py::test_real_pp2_boundary_restarts_attention_state',
        'modules/paired_stream.py': 'test_redo_parity.py::test_real_pp2_boundary_restarts_attention_state',
        'modules/image_data.py': 'test_redo_parity.py::test_image_tokens_backpropagate_into_vision_and_aligner',
        'modules/vision.py': 'test_redo_parity.py::test_image_tokens_backpropagate_into_vision_and_aligner',
        'modules/vision_training.py': 'test_redo_parity.py::test_image_tokens_backpropagate_into_vision_and_aligner',
        'modules/native_fp32_linear.py': 'test_redo_codecs.py::test_cross_layer_indexer_fp8_projection',
        'quantization/mxfp8.py': 'test_redo_codecs.py::test_cross_layer_indexer_fp8_projection',
        'quantization/mxfp4.py': 'test_redo_codecs.py::test_fp4_codec_rounding_and_surface',
        'quantization/nvfp4.py': 'test_redo_codecs.py::test_fp4_codec_rounding_and_surface',
        'optimizers/headwise_muon.py': 'test_redo_parity.py::test_optimizer_two_steps_and_nonfinite_transaction',
        'optimizers/owned_groups.py': 'test_redo_parity.py::test_optimizer_two_steps_and_nonfinite_transaction',
        'optimizers/sinkhorn.py': 'test_redo_parity.py::test_optimizer_two_steps_and_nonfinite_transaction',
        'optimizers/staged_update.py': 'test_redo_parity.py::test_remote_nonfinite_skips_replicated_optimizer',
        'parallel/owned_ddp.py': 'test_redo_parity.py::test_packed_loss_head_gradient_with_ddp_unused_detection',
    }
    assert routes == {
        source: 'deepseek_v41/' + target for source, target in required.items()
    }
    for source, target in routes.items():
        assert (root / 'megatron/lite/primitive' / source).is_file()
        path, function_name = target.split('::')
        tree = ast.parse((root / 'tests/unit' / path).read_text())
        assert function_name in {
            n.name for n in tree.body if isinstance(n, ast.FunctionDef)
        }
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
