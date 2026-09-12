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
