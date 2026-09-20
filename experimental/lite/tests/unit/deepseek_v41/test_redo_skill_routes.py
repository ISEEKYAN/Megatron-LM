# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""New primitive files have resolvable, executable-test-backed skill routes."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


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
        'ckpt/row_stream.py': 'test_bound_row_export.py::test_bound_save_streams_rows_and_exact_masters_under_budget',
        'ckpt/hf_weights.py': 'test_redo_export_contract.py::test_hf_export_obeys_external_quantized_storage',
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
        'optimizers/headwise_muon.py': 'test_redo_ep_finalize.py::test_ep_finalize_matches_single_global_batch',
        'optimizers/owned_groups.py': 'test_redo_parity.py::test_optimizer_two_steps_and_nonfinite_transaction',
        'optimizers/sinkhorn.py': 'test_redo_parity.py::test_optimizer_two_steps_and_nonfinite_transaction',
        'optimizers/staged_update.py': 'test_redo_parity.py::test_remote_nonfinite_skips_replicated_optimizer',
        'parallel/owned_ddp.py': 'test_redo_parity.py::test_packed_loss_head_gradient_with_ddp_unused_detection',
        'modules/attention/cp.py': 'test_redo_cp_loss.py::test_cp_loss_matches_global_token_weighted_ce',
        'parallel/cp.py': 'test_redo_cp_loss.py::test_cp_loss_matches_global_token_weighted_ce',
        'train_step.py': 'test_redo_cp_loss.py::test_cp_loss_matches_global_token_weighted_ce',
        'modules/experts.py': 'test_redo_moe_dual_bias.py::test_dispatch_option_reaches_model_and_preserves_local_moe',
    }
    assert routes == {
        source: 'deepseek_v41/' + target for source, target in required.items()
    }
    # Closed inventory: adding a checkpoint primitive requires a route or an
    # explicit review of this legacy non-bound checkpoint surface.
    legacy = {
        '__init__.py',
        'dcp.py',
        'distckpt.py',
        'weight_sync_fingerprint.py',
        'weight_sync_probe.py',
    }
    checkpoint_files = {
        p.name for p in (root / 'megatron/lite/primitive/ckpt').glob('*.py')
    }
    assert checkpoint_files == legacy | {
        Path(source).name for source in routes if source.startswith('ckpt/')
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


SKILLS = Path(__file__).resolve().parents[3] / 'skills'


def skill_body(name):
    return (
        (SKILLS / (name.replace('.', '/').replace('_', '-') + '.md'))
        .read_text()
        .split('```python')[2]
        .split('```')[0]
    )


def execute_skill(name, **dependencies):
    # Execute the specification with fake host I/O; done is not GPU evidence.
    exits = {
        status: (
            lambda *args, status=status, **kw: NS(
                done=status == 'done', status=status, args=args, **kw
            )
        )
        for status in ('done', 'blocked', 'out_of_scope')
    }
    scope = {**exits, **dependencies}
    exec(compile(skill_body(name), name, 'exec'), scope)
    return scope[name.split('.')[-1]]


@pytest.mark.parametrize(
    'case, expected',
    [
        ('complete', 'done'),
        ('duplicate', 'done'),
        ('repository_path', 'done'),
        ('checkpoint', 'done'),
        ('empty', 'out_of_scope'),
        ('unmapped', 'out_of_scope'),
        ('mixed', 'blocked'),
        ('reference', 'blocked'),
        ('budget', 'blocked'),
        ('validation', 'blocked'),
        ('smoke', 'blocked'),
    ],
)
def test_bound_training_executes_complete_coverage_contract(case, expected):
    files = (
        ['ckpt/row_stream.py', 'ckpt/hf_weights.py']
        if case == 'checkpoint'
        else ['train_step.py', 'modules/experts.py']
    )
    if case == 'duplicate':
        files.append(files[0])
    if case == 'repository_path':
        files = ['experimental/lite/megatron/lite/primitive/' + f for f in files]
    if case == 'empty':
        files = []
    if case in ('unmapped', 'mixed'):
        files = (files if case == 'mixed' else []) + ['uncovered.py']
    validate = Mock(return_value=NS(done=case != 'validation'))
    smoke = Mock(return_value=NS(done=case != 'smoke'))
    run = execute_skill(
        'primitive.bound_training',
        primitive=NS(validate=validate),
        load_source_and_tests=Mock(),
        require_real_gpu_smoke=smoke,
    )
    result = run(
        'task',
        files,
        None if case == 'reference' else object(),
        NS(max_candidates=1 if case == 'budget' else 20),
    )
    assert result.status == expected
    if case in ('unmapped', 'mixed'):
        assert result.evidence == ['uncovered.py']
        validate.assert_not_called()
    if expected == 'done':
        assert [c.kwargs['primitive'] for c in validate.call_args_list] == sorted(
            {
                f.removeprefix('experimental/lite/megatron/lite/primitive/')
                for f in files
            }
        )
        smoke.assert_called_once_with(
            'task',
            cases=['multimodal', 'quantized', 'ep2', 'cp2'],
            skip_is_failure=True,
        )
    elif case != 'smoke':
        smoke.assert_not_called()


CHECKS = (
    'static_contract',
    'single_gpu_or_node_proxy',
    'controlled_variable_precision',
    'composition_with_adjacent_primitives',
    'usage_example_runs',
)


@pytest.mark.parametrize('failure', [None, 'missing', 'proxy', 'precision', *CHECKS])
def test_validate_runs_checks_and_rejects_missing_or_failed_evidence(failure):
    checks = {name: Mock(return_value=NS(done=name != failure)) for name in CHECKS}
    implementation = NS(reference=object(), variables=[], checks=checks)
    if failure == 'missing':
        del implementation.checks
    basic = NS(
        construct_proxy_task=Mock(return_value=NS(done=failure != 'proxy')),
        align_precision=Mock(return_value=NS(done=failure != 'precision', next=[])),
    )
    result = execute_skill('primitive.validate', basic=basic)(
        'task', 'primitive', implementation, NS(proxy=1, precision=1)
    )
    assert result.status == ('done' if failure is None else 'blocked')
    count = (
        len(CHECKS)
        if failure is None
        else CHECKS.index(failure) + 1 if failure in CHECKS else 0
    )
    assert [check.call_count for check in checks.values()] == [
        int(i < count) for i in range(len(CHECKS))
    ]
    if failure is None:
        assert result.validation == [
            (name, checks[name].return_value) for name in CHECKS
        ]


@pytest.mark.parametrize('name', ['primitive.bound_training', 'primitive.validate'])
def test_changed_skill_lint_is_a_unit_gate(name):
    text = (SKILLS / (name.replace('.', '/').replace('_', '-') + '.md')).read_text()
    assert (
        text.count('MLITE_SKILL_SCHEMA_BEGIN')
        == text.count('MLITE_SKILL_SCHEMA_END')
        == 1
    )
    schema = ast.parse(text.split('```python')[1].split('```')[0]).body[0].value
    assert ast.literal_eval(schema.args[0]) == name
    spec = {k.arg: ast.literal_eval(k.value) for k in schema.keywords}
    body = ast.parse(skill_body(name))
    (function,) = [node for node in body.body if isinstance(node, ast.FunctionDef)]
    assert function.name == name.split('.')[-1]
    assert [a.arg for a in function.args.args] == spec['inputs']
    exits = {
        node.value.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
    }
    assert exits == set(spec['exits'])
    for dependency in spec['imports'] + spec['calls']:
        assert (
            SKILLS / (dependency.replace('.', '/').replace('_', '-') + '.md')
        ).is_file()
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }
    assert {call for call in calls if call.startswith(('basic.', 'primitive.'))} == set(
        spec['calls']
    )
    assert not any(isinstance(node, ast.While) for node in ast.walk(function))
    assignments = {
        node.targets[0].id: node.value
        for node in function.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    for loop in (node for node in ast.walk(function) if isinstance(node, ast.For)):
        assert (
            isinstance(loop.iter, ast.Subscript)
            and isinstance(loop.iter.slice, ast.Slice)
            and ast.unparse(loop.iter.slice.upper) == 'budget.max_candidates'
        ) or (
            isinstance(loop.iter, ast.Name)
            and isinstance(assignments[loop.iter.id], ast.Tuple)
        )
