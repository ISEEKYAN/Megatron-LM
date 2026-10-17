# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""An optional diagnostic must not consume the PP carrier twice."""

import ast
import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def test_pp2_diagnostic_never_reenters_forward(monkeypatch):
    path = ROOT / 'examples/verl/scripts/ds41_w4a8_proxy/indexer_diagnostic.py'
    spec = importlib.util.spec_from_file_location('indexer_diagnostic_pp_test', path)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    calls = []
    result = {'log_probs': torch.zeros(3)}

    def original(model, batch, **kwargs):
        calls.append(True)
        assert len(calls) == 1, 'receiving-stage carrier already consumed'
        return result

    leaves = {
        'megatron.lite.model.deepseek_v41.lite.protocol': types.SimpleNamespace(
            _forward_step_impl=original
        ),
        'megatron.lite.primitive.modules.deployment_math': types.SimpleNamespace(
            indexer_scores=lambda *args: None
        ),
    }
    for name, module in leaves.items():
        parts = name.split('.')
        for length in range(1, len(parts)):
            parent = '.'.join(parts[:length])
            package = types.ModuleType(parent)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, parent, package)
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(
            sys.modules[name.rsplit('.', 1)[0]], parts[-1], module, raising=False
        )
    monkeypatch.setattr(torch.random, 'fork_rng', lambda **kwargs: nullcontext())
    monkeypatch.setenv('DS41_INDEXER_DIAGNOSTIC_LIMIT', '1')
    model = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_size=2),
        training=False,
        layers=[None] * 4,
        parameters=lambda: iter(()),
    )
    batch = types.SimpleNamespace(
        input_ids=types.SimpleNamespace(
            is_cuda=True, device=types.SimpleNamespace(index=None)
        )
    )
    events = []
    with torch.no_grad():
        diagnostic.install(lambda *args: events.append(args))
        actual = leaves[
            'megatron.lite.model.deepseek_v41.lite.protocol'
        ]._forward_step_impl(model, batch)
    assert actual is result
    assert calls == [True] and not events


@pytest.mark.parametrize(
    'layers,pp,expected', [(4, 1, '1'), (4, 2, '0'), (6, 1, '0'), (9, 1, '0')]
)
def test_smoke_enables_indexer_diagnostic_only_for_4l_pp1(layers, pp, expected):
    source = (ROOT / 'tests/smoke/workflows/test_ds41_grpo_recipe.py').read_text()
    tree = ast.parse(source)
    expressions = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == 'DS41_INDEXER_DIAGNOSTIC'
    ]
    assert len(expressions) == 1
    expression = ast.Expression(expressions[0])
    helpers = _smoke_probe_helpers(tree)
    actual = eval(
        compile(expression, '<smoke-diagnostic-env>', 'eval'),
        dict(helpers, layers=layers, base={'DS41_PP': str(pp)}),
    )
    assert actual == expected


def _smoke_probe_helpers(tree):
    helpers = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ('_indexer_probe_enabled', '_check_indexer_probes')
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(helpers, '<smoke-probe-helpers>', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize(
    'pp,has_probe,passes',
    [(1, True, True), (2, False, True), (1, False, False), (2, True, False)],
)
def test_probe_receipts_require_pp1_probe_and_pp2_absence(pp, has_probe, passes):
    source = (ROOT / 'tests/smoke/workflows/test_ds41_grpo_recipe.py').read_text()
    check = _smoke_probe_helpers(ast.parse(source))['_check_indexer_probes']
    probes = (
        [
            dict(
                kind='indexer_fp32_definition_probe',
                native_replay_equal=True,
                parameter_versions_unchanged=True,
                score_calls=[dict(truncated_rows=1)],
            )
        ]
        if has_probe
        else []
    )
    if passes:
        assert check(4, pp, probes) == probes
    else:
        with pytest.raises(AssertionError):
            check(4, pp, probes)
