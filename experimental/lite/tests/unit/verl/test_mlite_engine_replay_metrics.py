# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU boundary coverage for both production engine metric return paths."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('collect_outputs', [False, True])
def test_engine_preserves_runtime_replay_metrics(collect_outputs):
    source = (
        Path(__file__).parents[3] / 'examples/verl/verl_mlite/engine/mlite_engine.py'
    )
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == 'MegatronLiteEngine'
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == '_forward_backward_batch_with_runtime'
    )
    # Execute the real method; only optional VERL batching/postprocessing is isolated.
    scope = dict(
        torch=torch,
        TensorDict=object,
        Any=object,
        _VerlMetric=None,
        get_device_id=lambda: 'cpu',
        tu=SimpleNamespace(
            get_non_tensor_data=lambda **kw: 4, assign_non_tensor=lambda *a, **kw: None
        ),
        postprocess_batch_func=lambda **kw: {
            'metrics': {'loss_metric': [9.0]},
            'model_output': {'kept': True},
        },
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope
    )
    evidence = {
        'router_replay/calls': 2,
        'router_replay/rows': 8,
        'router_replay/changed': 2,
        'router_replay/changed_frac': 0.25,
        'router_replay/routers': 1,
    }
    calls = []

    def forward_backward(handle, batches, **kwargs):
        calls.append((list(batches), kwargs))
        return SimpleNamespace(
            metrics={**evidence, 'loss_metric': [9.0]},
            model_output=SimpleNamespace(loss=torch.tensor(7.0)),
        )

    packed = SimpleNamespace(routed_experts=torch.ones(1))
    engine = SimpleNamespace(
        handle=SimpleNamespace(_extras={}),
        engine_config=SimpleNamespace(router_replay_mode='R3'),
        runtime=SimpleNamespace(forward_backward=forward_backward),
        get_data_parallel_size=lambda: 1,
        is_mp_src_rank_with_outputs=lambda: True,
        _make_runtime_batch=lambda batch: packed,
        _make_runtime_loss_context=lambda batch, **kw: None,
        _make_runtime_loss_fn=lambda *args: object(),
    )
    batch = SimpleNamespace(to=lambda device: packed)
    result = scope[method.name](
        engine,
        data=object(),
        micro_batches=[batch],
        indices=None,
        loss_function=(lambda: None) if collect_outputs else None,
        forward_only=False,
    )
    assert len(calls) == 1 and calls[0][1]['router_replay'] == {'action': 'replay'}
    assert result['metrics'] == {
        **{k: [v] for k, v in evidence.items()},
        'loss_metric': [9.0],
    }, 'ENGINE_R3_METRICS_RETURNED'
    if collect_outputs:
        assert result['model_output'] == {'kept': True}
    else:
        assert result['loss'] == [7.0]
