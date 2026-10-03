# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native direct-copy parameters must survive meta reload and padded sleep."""
from types import SimpleNamespace as NS

import pytest
import torch
from test_receiver_staging import adapter


@pytest.mark.parametrize('tp', [4, 8])
@pytest.mark.parametrize('rank', range(8))
def test_direct_sink_reload_matches_cold_and_preserves_address(
    v41_core_te, monkeypatch, tp, rank
):
    if rank >= tp:
        pytest.skip('outside TP group')
    consumer = adapter(v41_core_te)
    model = torch.nn.Module()
    attention = torch.nn.Module()
    model.attention = attention
    kernel = torch.nn.Parameter(torch.full((64,), -float('inf')), requires_grad=False)
    attention._ds41_attn_sink_initializer = kernel.detach().clone()
    ptr = kernel.data_ptr()
    for generation in (1, 2):
        kernel.zero_()  # CuMem sleep(level=2) discards checkpoint storage.
        attention.attn_sink = torch.nn.Parameter(
            torch.empty(64, device='meta'), requires_grad=False
        )
        info = NS(kernel_tensors=({'attn_sink': kernel}, {}), loaded_weights=[])
        info.reset = lambda: setattr(info, 'kernel_tensors', None)
        layerwise = NS(
            get_layerwise_info=lambda layer: info,
            _place_kernel_tensors=lambda layer, state: setattr(
                layer, 'attn_sink', state.kernel_tensors[0]['attn_sink']
            ),
            LOADING_LAYERS=set(),
        )
        if hasattr(consumer, 'restore_attn_sink_parameters'):
            consumer.restore_attn_sink_parameters(model, layerwise)
        weight = torch.arange(64, dtype=torch.float32) + generation
        count = 64 // tp
        # Native DS4.1 load_weights writes this slice directly; it never calls
        # attn_sink.weight_loader, so copying into a meta parameter loses it.
        attention.attn_sink[:count].copy_(weight[rank * count : (rank + 1) * count])
        cold = torch.full((64,), -float('inf'))
        cold[:count] = weight[rank * count : (rank + 1) * count]
        assert attention.attn_sink.device.type != 'meta'
        torch.testing.assert_close(attention.attn_sink, cold, rtol=0, atol=0)
        assert attention.attn_sink is kernel
        assert attention.attn_sink.data_ptr() == ptr


@pytest.mark.parametrize('tp', [4, 8])
def test_mxfp4_materialization_initializes_padding_once(v41_core_te, monkeypatch, tp):
    from test_receiver_staging import Mxfp4MoEMethod

    consumer = adapter(v41_core_te)
    layer = torch.nn.Module()
    layer.quant_method = Mxfp4MoEMethod()
    layer.moe_config = NS(
        hidden_dim_unpadded=64,
        intermediate_size=2304,
        moe_parallel_config=NS(tp_size=tp),
        tp_shard_with_padding=False,
    )
    real = 2304 // tp
    padded = ((real + 127) // 128) * 128
    layer.quant_method.create_weights(layer, 2, 64, padded, torch.bfloat16)

    def load(param, value, expert):
        param[expert, :, : real // 2].copy_(value)

    layer.w2_weight.weight_loader = load
    consumer.set_mxfp4_load_numel(layer)
    loader = layer.w2_weight.weight_loader
    metadata = layer.w2_weight.to('meta')
    metadata.__dict__ = layer.w2_weight.__dict__.copy()
    loader(metadata, torch.ones(64, real // 2, dtype=torch.uint8), 0)
    raw = torch.nn.Parameter(torch.empty_like(layer.w2_weight), requires_grad=False)
    raw.__dict__ = metadata.__dict__.copy()
    model = torch.nn.Module()
    model.weight = raw
    from megatron.lite.model.deepseek_v41.lite.resync import transport_weights
    from reload_fixture import install_reload

    def initialize(model):
        # Same Parameter, new poisoned storage, during native initialize.
        raw.data = torch.full_like(raw, 239)

    install_reload(monkeypatch, model, initialize)

    def load_weights(pairs):
        for name, tensor in pairs:
            loader(raw, tensor, int(name))
        return {name for name, _ in pairs}

    model.load_weights = load_weights
    for generation in (1, 2):
        receiver = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
        for expert in (0, 1):
            receiver.receive(
                [
                    (
                        str(expert),
                        torch.full(
                            (64, real // 2), generation + expert, dtype=torch.uint8
                        ),
                    )
                ]
            )
        cold = torch.zeros_like(raw)
        for expert in (0, 1):
            cold[expert, :, : real // 2] = generation + expert
        assert torch.equal(raw, cold)
        receiver.receive(list(transport_weights([], deployment=True)))
        receiver.finish()


def test_padding_loader_remains_a_rebindable_method(v41_core_te):
    from types import MethodType

    consumer = adapter(v41_core_te)

    class Owner:
        def load(self, param, value):
            param.data[0].copy_(value)

    owner = Owner()
    load = consumer._mxfp4_initialized_loader(owner.load)
    assert isinstance(load, MethodType)
    assert load.__self__ is owner
    # Native metadata sanitizes and then rebinds method owners.
    rebound = MethodType(load.__func__, Owner())
    param = torch.nn.Parameter(
        torch.full((2, 2), 239, dtype=torch.uint8), requires_grad=False
    )
    rebound(param, torch.ones(2, dtype=torch.uint8))
    assert torch.equal(param, torch.tensor([[1, 1], [0, 0]], dtype=torch.uint8))


@pytest.mark.parametrize('tp,rank', [(4, 3), (8, 7)])
@pytest.mark.parametrize('return_names', [True, False])
def test_native_sink_return_names_drive_receiver_finish(
    v41_core_te, monkeypatch, tp, rank, return_names
):
    from types import MethodType

    from megatron.lite.model.deepseek_v41.lite.resync import transport_weights
    from native_reload import functions
    from reload_fixture import install_reload

    native = functions(
        'vllm/models/deepseek_v41/nvidia/model.py',
        ['load_weights'],
        'DeepseekV4Model',
        get_tensor_model_parallel_world_size=lambda: tp,
        get_tensor_model_parallel_rank=lambda: rank,
        is_pp_missing_parameter=lambda name, model: False,
    )
    consumer = adapter(v41_core_te)
    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    kernel = torch.nn.Parameter(torch.full((64,), -float('inf')), requires_grad=False)
    model.attn._ds41_attn_sink_initializer = kernel.detach().clone()
    model.attn.attn_sink = torch.nn.Parameter(
        torch.empty(64, device='meta'), requires_grad=False
    )
    model.config = NS(num_attention_heads=64)
    model.quant_config = None
    model.get_expert_mapping = lambda: []
    infos, _ = install_reload(monkeypatch, model)
    infos[model.attn].kernel_tensors = ({'attn_sink': kernel}, {})
    load = MethodType(native.load_weights, model)
    returned = []

    def load_weights(pairs):
        names = load(pairs)
        returned.append(names)
        return names if return_names else None

    model.load_weights = load_weights
    receiver = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    receiver.receive(
        list(
            transport_weights(
                [('attn.attn_sink', torch.arange(64).float())], deployment=True
            )
        )
    )
    assert returned == [{'attn.attn_sink'}]
    assert receiver.received_sinks == ({'attn.attn_sink'} if return_names else set())
    expected = torch.full((64,), -float('inf'))
    expected[: 64 // tp] = torch.arange(64)[
        rank * (64 // tp) : (rank + 1) * (64 // tp)
    ].float()
    assert torch.equal(kernel, expected)
    if return_names:
        receiver.finish()
        assert receiver.finished
    else:
        with pytest.raises(ValueError, match='missing attention sinks'):
            receiver.finish()
        receiver.abort()
