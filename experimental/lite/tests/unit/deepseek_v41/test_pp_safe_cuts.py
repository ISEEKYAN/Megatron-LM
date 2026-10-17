# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PP>2 and custom pipeline layouts at owner-safe cuts match the whole model."""
import pytest
import torch
from test_pp_stage_training import prefix_config
from test_redo_parity import release_config


def test_every_cut_keeps_owners_with_their_readers():
    from megatron.lite.model.deepseek_v41.topology import pipeline_cuts

    topology = release_config().topology
    assert pipeline_cuts(topology, (2, 8, 14, 20), 5) == (2, 8, 14, 20)
    assert pipeline_cuts(topology, None) == (20,)
    assert pipeline_cuts(topology, 8) == (8,)
    with pytest.raises(NotImplementedError, match='crosses an attention owner'):
        pipeline_cuts(topology, (8, 10), 3)
    for cuts, stages in (((8, 8), 3), ((20, 8), 3), ((8, 20), 4)):
        with pytest.raises(ValueError, match='V4.1_PP_SPLIT'):
            pipeline_cuts(topology, cuts, stages)
    with pytest.raises(ValueError, match='more than two stages'):
        pipeline_cuts(topology, None, 3)


@pytest.mark.parametrize('split, layout', [((1, 2), None), (None, 'Et|t|t*2L')])
def test_stage_table_from_cuts_or_pp_layout(split, layout):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    impl = protocol.ImplConfig(pipeline_split_layer=split)
    parallel = ParallelConfig(pp=3, pp_layout=layout)
    assert protocol._pipeline_stages(prefix_config(), impl, parallel) == (
        (0, 1),
        (1, 2),
        (2, 4),
    )


@pytest.mark.parametrize(
    'kwargs, error, match',
    [
        # Layer 3 reads owner layer 2 across the cut.
        (dict(pp=2, pp_layout='Et*3|tL'), NotImplementedError, 'crosses'),
        # Two virtual stages per rank: VPP stays unsupported.
        (dict(pp=2, pp_layout='Et|t|t|tL'), NotImplementedError, 'VPP'),
    ],
)
def test_pp_layout_rejects_crossing_cuts_and_vpp(kwargs, error, match):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    with pytest.raises(error, match=match):
        protocol._pipeline_stages(
            prefix_config(), protocol.ImplConfig(), ParallelConfig(**kwargs)
        )
    with pytest.raises(ValueError, match='pp_layout or pipeline_split_layer'):
        protocol._pipeline_stages(
            prefix_config(),
            protocol.ImplConfig(pipeline_split_layer=(1, 2)),
            ParallelConfig(pp=3, pp_layout='Et|t|t*2L'),
        )


def test_three_stages_with_a_middle_relay_match_the_whole_model(v41_core_te):
    # The middle stage both receives and sends the paired carrier.
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model
    from megatron.lite.primitive.parallel.state import ParallelState

    cfg = prefix_config()
    torch.manual_seed(383)
    full = DeepseekV41Model(cfg, quantized=False)
    spans = ((0, 1), (1, 2), (2, 4))
    stages = [
        DeepseekV41Model(
            cfg,
            quantized=False,
            layer_range=span,
            parallel_state=ParallelState(
                pp_size=3, pp_rank=i, pp_is_first=i == 0, pp_is_last=i == 2
            ),
            pipeline_split_layer=(1, 2),
        )
        for i, span in enumerate(spans)
    ]
    for model in [full, *stages]:
        model.float()
    for stage in stages:
        stage.pipeline_residual_dtype = torch.float32
        own = stage.state_dict()
        stage.load_state_dict({k: v for k, v in full.state_dict().items() if k in own})
    ids = torch.tensor([[2, 3, 9, 4], [7, 1, 5, 6]])
    expected = full(ids)
    carrier = None
    for stage in stages:
        if carrier is not None:
            stage.set_input_tensor(carrier)
        out = stage(ids)
        carrier = out.get('hidden_states')
    assert 'hidden_states' not in out
    assert torch.equal(out['logits'], expected['logits'])
    expected['logits'].square().mean().backward()
    out['logits'].square().mean().backward()
    reference = dict(full.named_parameters())
    checked = 0
    for stage in stages:
        for name, value in stage.named_parameters():
            wanted = reference[name].grad
            assert (value.grad is None) == (wanted is None), name
            if wanted is not None:
                assert torch.equal(value.grad, wanted), name
                checked += 1
    assert checked
    with pytest.raises(RuntimeError, match='V4.1_PP_INPUT_STAGE'):
        stages[0].set_input_tensor(carrier)
