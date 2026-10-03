# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU lifecycle tests; native MXFP4 is additionally exercised on GB200."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


def adapter(v41_core_te):
    path = (
        Path(__file__).resolve().parents[3]
        / 'examples/verl/verl_mlite/rollout/deepseek_v41.py'
    )
    spec = importlib.util.spec_from_file_location('ds41_receiver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Mxfp4MoEMethod:
    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **kwargs
    ):
        self.intermediate_size = intermediate_size_per_partition
        e, h, i = num_experts, hidden_size, intermediate_size_per_partition
        for name, shape in {
            'w13_weight': (e, 2 * i, h // 2),
            'w2_weight': (e, h, i // 2),
            'w13_weight_scale': (e, 2 * i, h // 32),
            'w2_weight_scale': (e, h, i // 32),
        }.items():
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(shape, dtype=torch.uint8), requires_grad=False
                ),
            )
            layer.get_parameter(name).weight_loader = lambda *args: None


@pytest.mark.parametrize('tp', [4, 8])
@pytest.mark.parametrize('rank', range(8))
@pytest.mark.parametrize('intermediate', [2304, 3072])
def test_unpadded_completion_waits_for_all_scales(v41_core_te, tp, rank, intermediate):
    if rank >= tp:
        pytest.skip('outside TP group')
    module = adapter(v41_core_te)
    layer = torch.nn.Module()
    from native_reload import mxfp4_method

    layer.quant_method = mxfp4_method()
    layer.moe_config = NS(
        hidden_dim_unpadded=5120,
        intermediate_size=intermediate,
        moe_parallel_config=NS(tp_size=tp),
        tp_shard_with_padding=False,
    )
    with torch.device('meta'):
        layer.quant_method.create_weights(
            layer,
            384,
            5120,
            ((intermediate // tp + 127) // 128) * 128,
            torch.bfloat16,
            weight_loader=lambda *a: None,
        )
    original_size = layer.quant_method.intermediate_size
    module.set_mxfp4_load_numel(layer)
    assert layer.quant_method.intermediate_size == original_size
    checkpoint = torch.nn.Module()
    with torch.device('meta'):
        mxfp4_method().create_weights(
            checkpoint,
            384,
            5120,
            intermediate // tp,
            torch.bfloat16,
            weight_loader=lambda *a: None,
        )
    expected = sum(p.numel() for p in checkpoint.parameters())
    counts = [getattr(p, 'weight_loader_numel', p.numel()) for p in layer.parameters()]
    assert sum(counts) == expected
    # Last scale is essential: earlier W/W/S arrivals must not complete.
    assert sum(counts[:-1]) < expected
    assert sum(counts) <= sum(p.numel() for p in layer.parameters())


def test_staging_counts_storage_and_releases_at_layer_boundary(v41_core_te):
    module = adapter(v41_core_te)
    first, second = torch.empty(64, dtype=torch.uint8), torch.empty(
        32, dtype=torch.uint8
    )
    infos = [
        NS(
            loaded_weights=[
                ('w', NS(arguments={'loaded_weight': first[:8]})),
                ('s', NS(arguments={'loaded_weight': first[8:]})),
            ]
        ),
        NS(loaded_weights=[]),
    ]
    meter = module.StagingBudget(infos, 96)
    assert meter.refresh() == 64  # full backing storage, deduplicated
    meter.check(second.nbytes)
    infos[0].loaded_weights.append(('s', NS(arguments={'loaded_weight': second})))
    assert meter.refresh() == 96
    assert meter.peak_bytes == 96
    infos[0].loaded_weights.clear()  # native layerwise process/reset
    assert meter.refresh() == 0
    infos[1].loaded_weights.append(('w', NS(arguments={'loaded_weight': second})))
    assert meter.refresh() == 32
    assert meter.peak_bytes == 96
    with pytest.raises(RuntimeError, match='staging.*budget'):
        meter.check(65)
    assert meter.current_bytes == 32


@pytest.mark.parametrize('budget', [0, -1, True, 1.5])
def test_invalid_budget(v41_core_te, budget):
    with pytest.raises(ValueError, match='budget'):
        adapter(v41_core_te).StagingBudget([], budget)


def test_receiver_over_budget_aborts_and_restores_hooks(v41_core_te, monkeypatch):
    import sys
    import types

    module = adapter(v41_core_te)
    model = torch.nn.Module()
    model._ds41_reload_metadata = True
    model.process_weights_after_loading = lambda: None
    hook = model.process_weights_after_loading
    info = NS(loaded_weights=[], kernel_tensors={}, can_load=lambda: True)
    info.reset = lambda: info.loaded_weights.clear()
    reloading = types.ModuleType('vllm.model_executor.model_loader.reload')
    layerwise = types.ModuleType(reloading.__name__ + '.layerwise')
    layerwise.get_layerwise_info = lambda layer: info
    layerwise.LOADING_LAYERS = {model}
    restored = []
    layerwise._place_kernel_tensors = lambda layer, state: restored.append(layer)
    reloading.layerwise = layerwise
    old_info = info

    def initialize(model):
        nonlocal info
        info = NS(loaded_weights=[], kernel_tensors={}, can_load=lambda: True)
        info.reset = lambda: info.loaded_weights.clear()

    reloading.initialize_layerwise_reload = initialize
    parent = None
    for name in ('vllm', 'vllm.model_executor', 'vllm.model_executor.model_loader'):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
        if parent is not None:
            setattr(parent, name.rsplit('.', 1)[-1], package)
        parent = package
    parent.reload = reloading
    monkeypatch.setitem(sys.modules, reloading.__name__, reloading)
    monkeypatch.setitem(sys.modules, layerwise.__name__, layerwise)
    model.load_weights = lambda pairs: info.loaded_weights.extend(
        (name, NS(arguments={'loaded_weight': tensor})) for name, tensor in pairs
    )
    receiver = module.ResyncReceiver(
        model, NS(cpu_offload_gb=0), staging_budget_bytes=12
    )
    assert receiver.staging.infos == [info]
    assert info is not old_info
    receiver.receive([('weight', torch.zeros(8, dtype=torch.uint8))])
    assert receiver.staging.current_bytes == 8
    with pytest.raises(RuntimeError, match='staging.*budget'):
        receiver.receive([('scale', torch.zeros(8, dtype=torch.uint8))])
    assert receiver.staging.current_bytes == 0
    assert receiver.staging.peak_bytes == 8
    assert not info.loaded_weights
    assert restored == [model]
    assert model.process_weights_after_loading is hook
    assert not layerwise.LOADING_LAYERS
    with pytest.raises(RuntimeError, match='finalized'):
        receiver.receive([('weight', torch.zeros(1, dtype=torch.uint8))])


def test_native_288_to_384_completion_through_receiver(v41_core_te, monkeypatch):
    from megatron.lite.model.deepseek_v41.lite.resync import transport_weights
    from native_reload import mxfp4_method, online_loader
    from reload_fixture import install_reload

    consumer = adapter(v41_core_te)
    model = torch.nn.Module()
    model.quant_method = mxfp4_method()
    model.moe_config = NS(
        hidden_dim_unpadded=64,
        intermediate_size=2304,
        moe_parallel_config=NS(tp_size=8),
        tp_shard_with_padding=False,
    )
    model.quant_method.create_weights(
        model, 2, 64, 384, torch.bfloat16, weight_loader=lambda *a: None
    )
    checkpoint = torch.nn.Module()
    mxfp4_method().create_weights(
        checkpoint, 2, 64, 288, torch.bfloat16, weight_loader=lambda *a: None
    )
    expected = sum(p.numel() for p in checkpoint.parameters())
    consumer.set_mxfp4_load_numel(model)
    assert sum(p.weight_loader_numel for p in model.parameters()) == expected
    assert expected < sum(p.numel() for p in model.parameters())
    infos, _ = install_reload(monkeypatch, model)
    info = infos[model]
    completed = []

    info.can_load = lambda: True
    info.load_numel = 0

    def process(layer, state):
        completed.append(state.load_numel)
        state.reset()

    make_loader = online_loader(info, process, monkeypatch)
    for name, param in model.named_parameters():

        def copy_checkpoint(param, loaded_weight):
            # Copy the unpadded checkpoint slice; native CopyCounter observes it.
            slices = tuple(slice(0, dim) for dim in loaded_weight.shape)
            param.data[slices].copy_(loaded_weight)

        param.weight_loader = copy_checkpoint
        param.weight_loader = make_loader(model, name)

    def load_weights(pairs):
        for name, tensor in pairs:
            param = model.get_parameter(name)
            param.weight_loader(param, tensor)
        return set(name for name, _ in pairs)

    model.load_weights = load_weights
    receiver = consumer.ResyncReceiver(model, NS(cpu_offload_gb=0))
    weights = list(checkpoint.named_parameters())
    receiver.receive(weights[:-1])
    assert not completed
    assert receiver.staging.current_bytes > 0
    receiver.receive(weights[-1:])
    assert completed == [expected]
    assert receiver.staging.current_bytes == 0
    receiver.receive(list(transport_weights([], deployment=True)))
    receiver.finish()
