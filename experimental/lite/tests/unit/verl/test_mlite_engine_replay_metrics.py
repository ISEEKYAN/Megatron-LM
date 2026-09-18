# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU boundary coverage for both production engine metric return paths."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('collect_outputs', [False, True])
def test_engine_preserves_runtime_replay_metrics(
    collect_outputs, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter
    from megatron.lite.primitive.modules.router_replay import RouterReplay
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.backends.mlite.runtime import (
        MegatronLiteRuntime,
        ModelHandle,
    )
    from megatron.lite.runtime.contracts import PackedBatch

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
    # Execute the real engine method and runtime/router below. Optional VERL
    # TensorDict/loss collection is isolated so this CPU contract needs neither
    # the VERL training stack nor distributed actors; it is not VERL e2e coverage.
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

    model = SigmoidTopKRouter(
        SimpleNamespace(
            num_experts_per_tok=2,
            n_routed_experts=4,
            routed_scaling_factor=1.0,
            hidden_size=4,
        ),
        ParallelState(),
        compute_aux_loss=False,
    )
    with torch.no_grad():
        model.gate.weight.zero_()
        model.gate.weight[:, 0] = torch.tensor([4.0, 3.0, 2.0, 1.0])

    def forward(module, batch):
        weights = module(torch.ones(len(batch.input_ids), 4))[0]
        return {'loss': weights.sum() * 0 + 7.0}

    handle = ModelHandle(
        model=model,
        parallel_state=SimpleNamespace(pp_size=1),
        _extras={'forward_step': forward},
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    def forward_backward(handle, batches, **kwargs):
        batches = list(batches)
        calls.append((batches, kwargs))
        result = runtime.forward_backward(handle, iter(batches), **kwargs)
        assert result.metrics == evidence, 'REAL_RUNTIME_REPLAY_METRICS'
        # Unrelated loss metric verifies the engine preserves collector output.
        result.metrics['loss_metric'] = [9.0]
        return result

    batches = [
        PackedBatch(
            torch.arange(n),
            None,
            torch.tensor([n]),
            routed_experts=torch.tensor([[routes] * n]),
            r3_replay_mask=torch.ones(n, dtype=torch.bool),
        )
        for n, routes in [(3, [[0, 1]]), (1, [[2, 3]])]
    ]
    engine = SimpleNamespace(
        handle=handle,
        engine_config=SimpleNamespace(router_replay_mode='R3'),
        runtime=SimpleNamespace(forward_backward=forward_backward),
        get_data_parallel_size=lambda: 1,
        is_mp_src_rank_with_outputs=lambda: True,
        _make_runtime_batch=lambda batch: batch,
        _make_runtime_loss_context=lambda batch, **kw: None,
        _make_runtime_loss_fn=lambda *args: None,
    )
    instances = RouterReplay.global_router_replay_instances[:]
    try:
        result = scope[method.name](
            engine,
            data=object(),
            micro_batches=[SimpleNamespace(to=lambda device, b=b: b) for b in batches],
            indices=None,
            loss_function=(lambda: None) if collect_outputs else None,
            forward_only=False,
        )
    finally:
        RouterReplay.clear_global_state()
        RouterReplay.global_router_replay_instances[:] = instances
    assert len(calls) == 1 and calls[0][1]['router_replay'] == {'action': 'replay'}
    assert result['metrics'] == {
        **{k: [v] for k, v in evidence.items()},
        'loss_metric': [9.0],
    }, 'ENGINE_R3_METRICS_RETURNED'
    if collect_outputs:
        assert result['model_output'] == {'kept': True}
    else:
        assert result['loss'] == [7.0]


@pytest.mark.parametrize('mask_state', ['missing', 'none', 'present'])
def test_engine_r3_packing_requires_response_mask(mask_state):
    from megatron.lite.primitive.modules import router_replay

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
        if isinstance(n, ast.FunctionDef) and n.name == '_r3_replay_mask_for_packing'
    )
    method.decorator_list = []
    scope = dict(torch=torch, TensorDict=dict, router_replay=router_replay)
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope
    )
    pack = scope[method.name]
    input_ids = torch.nested.as_nested_tensor(
        [torch.arange(4), torch.arange(3)], layout=torch.jagged
    )
    batch = {'input_ids': input_ids}
    if mask_state != 'missing':
        batch['response_mask'] = (
            torch.tensor([[1, 1], [0, 0]]) if mask_state == 'present' else None
        )
    if mask_state == 'present':
        actual = pack(batch, input_ids)
        assert [row.tolist() for row in actual.unbind()] == [
            [True, True, True, False],
            [False, False, False],
        ]
    else:
        with pytest.raises(
            ValueError, match='R3 replay requires micro_batch.response_mask'
        ):
            pack(batch, input_ids)
