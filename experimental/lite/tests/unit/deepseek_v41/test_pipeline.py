# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.model import protocol_utils
from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.runtime.contracts import PackedBatch


@pytest.mark.parametrize(
    'case', ['range_outside_local_stage', 'output_on_nonfinal_stage']
)
def test_v41_pipeline_rejects(moe, model_config, case):
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    stage = DeepseekV41Model(
        model_config, token_map=list(range(256)), quantized=False, layer_range=(0, 20)
    )
    ids = torch.tensor([[3, 4]])
    if case == 'range_outside_local_stage':
        with pytest.raises(
            ValueError, match='^Requested range is outside this pipeline stage$'
        ):
            stage.forward_pipeline_range(ids, start=0, end=21)
    else:
        payload, _ = stage.forward_pipeline_range(ids, start=0, end=20)
        with pytest.raises(
            RuntimeError, match='^Only the final pipeline stage owns the output head$'
        ):
            stage.finish_pipeline(payload)


def test_v41_packed_pipeline_matches_monolithic(moe, model_config):
    torch.manual_seed(17)
    bundle = protocol.build_model(
        model_config,
        impl_cfg=protocol.ImplConfig(
            device='cpu',
            dtype=torch.float32,
            quantized=False,
            token_map=list(range(256)),
        ),
    )
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
        output = protocol.packed_pipeline_forward_step(
            model, batch, start=start, end=end, state=state
        )
        state = output['packed_pipeline_state']
        assert [payload.h.shape[1] for payload, _ in state] == [5, 8, 3]
    for key in ('logits', 'log_probs', 'loss'):
        torch.testing.assert_close(
            output[key], expected[key], atol=0, rtol=0, msg='packed:' + key
        )
    output['loss'].backward()
    for name, parameter in model.named_parameters():
        assert (gradients[name] is None) == (parameter.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, gradients[name], atol=1e-6, rtol=1e-5, msg=name
            )


@pytest.mark.parametrize('rank', [0, 1])
def test_contiguous_override_and_legacy_default(rank):
    model = SimpleNamespace(ps=ParallelState(cp_size=2, cp_rank=rank))
    ids = torch.arange(12)
    batch = PackedBatch(ids, ids, torch.tensor([5, 7]))
    routes = torch.nested.as_nested_tensor(
        [torch.arange(5).reshape(5, 1, 1), torch.arange(7).reshape(7, 1, 1) + 10],
        layout=torch.jagged,
    )
    legacy = protocol_utils.pack_routed_experts(model, batch, routes, contiguous=True)[
        0
    ]
    current = protocol.pack_routed_experts(model, batch, routes)[0]
    zigzag = protocol_utils.pack_routed_experts(model, batch, routes)[0]
    assert legacy[:, 0].tolist() == (
        [0, 1, 2, 3, 4, 0, 0, 0] if rank == 0 else [10, 11, 12, 13, 14, 15, 16, 0]
    )
    assert current[:, 0].tolist() == (
        [0, 1, 2, 3, 4, 0, 10] if rank == 0 else [11, 12, 13, 14, 15, 16, 0]
    )
    assert zigzag[:, 0].tolist() == (
        [0, 1, 0, 0, 10, 11, 16, 0] if rank == 0 else [2, 3, 4, 0, 12, 13, 14, 15]
    )
    with pytest.raises(
        ValueError, match='Contiguous padding requires contiguous CP slicing'
    ):
        protocol_utils.pack_routed_experts(
            model, batch, routes, contiguous_padding=True
        )


@pytest.mark.parametrize('pp', [2, 4])
def test_pipeline_build_rejects_unsupported_parallelism(moe, model_config, pp):
    with pytest.raises(
        NotImplementedError, match='^V4.1_UNSUPPORTED_PARALLELISM: pp;'
    ) as error:
        protocol.build_model(
            model_config,
            impl_cfg=protocol.ImplConfig(
                parallel=protocol.ParallelConfig(pp=pp), device='cpu'
            ),
        )
    assert 'supports PP only' not in str(error.value)


@pytest.mark.parametrize('pp', [1, 2, 4])
@pytest.mark.parametrize('vision', ['local', 'external', 'external_frozen'])
def test_pipeline_build_reports_current_support(moe, model_config, pp, vision):
    external = vision != 'local'
    enabled = vision != 'external_frozen'
    impl_cfg = protocol.ImplConfig(
        parallel=protocol.ParallelConfig(pp=pp),
        device='cpu',
        dtype=torch.float32,
        quantized=False,
        token_map=list(range(256)),
        # An external schedule overrides the text-only default even with a
        # frozen vision mask: configuration must reject it before allocation.
        text_only=external,
        external_vision_device='cpu' if external else None,
        vision_trainability=(
            protocol.VisionTrainability(enabled, enabled, enabled, enabled)
            if external
            else None
        ),
    )
    if pp == 1:
        bundle = protocol.build_model(model_config, impl_cfg=impl_cfg)
        assert (bundle.extras['vision_schedule'] is not None) == external
        return
    try:
        protocol.build_model(model_config, impl_cfg=impl_cfg)
    except NotImplementedError as error:
        # PP itself is not enabled yet. Model-side PP integration must update
        # this expectation when it removes pp from the unsupported axes.
        assert str(error) == (
            'V4.1_UNSUPPORTED_PARALLELISM: pp; '
            'supported: DP, EP with CP=1, or contiguous CP-only; '
            'TP/PP/VPP/ETP and custom pipeline layouts are unsupported'
        ), 'PP_UNSUPPORTED_MUST_PRECEDE_TEXT_ONLY_CONTRACT'
    else:
        pytest.fail('PP_MULTIMODAL_BUILD_MUST_REJECT')


@pytest.mark.parametrize(
    'settings, rejected',
    [
        ({}, None),
        ({'ep': 2}, None),
        ({'cp': 2}, None),
        ({'tp': 2}, 'tp'),
        ({'pp': 2}, 'pp'),
        ({'vpp': 2}, 'vpp'),
        ({'etp': 2}, 'etp'),
        ({'pp_layout': 'Et|L'}, 'pp_layout'),
        ({'tp': 2, 'pp': 2, 'etp': 2}, 'tp, pp, etp'),
        ({'cp': 2, 'ep': 2}, 'CP_AND_EP_NOT_SIMULTANEOUSLY_SUPPORTED'),
    ],
)
def test_parallel_guard_message_consistency(
    moe, model_config, monkeypatch, settings, rejected
):
    # Stop after the guard: accepted cases must reach parallel-state construction.
    # Expected support and rejected axes above are independent of production metadata.
    def reached_parallel_state():
        raise RuntimeError('GUARD_ACCEPTED')

    monkeypatch.setattr(protocol, 'ParallelState', reached_parallel_state)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    try:
        protocol.build_model(
            model_config,
            impl_cfg=protocol.ImplConfig(
                device='meta', parallel=protocol.ParallelConfig(**settings)
            ),
        )
    except (NotImplementedError, RuntimeError) as error:
        actual = str(error)
    else:
        actual = 'NO_GUARD_RESULT'
    if rejected is None:
        expected = 'GUARD_ACCEPTED'
    elif rejected == 'CP_AND_EP_NOT_SIMULTANEOUSLY_SUPPORTED':
        expected = (
            'CP_AND_EP_NOT_SIMULTANEOUSLY_SUPPORTED: V4.1 requires EP=1 with CP>1; '
            'use CP-only or EP with CP=1'
        )
    else:
        expected = (
            f'V4.1_UNSUPPORTED_PARALLELISM: {rejected}; '
            'supported: DP, EP with CP=1, or contiguous CP-only; '
            'TP/PP/VPP/ETP and custom pipeline layouts are unsupported'
        )
    assert actual == expected, 'PARALLEL_GUARD_MESSAGE_CONSISTENCY'
